"""Shared helpers to extract *user-authored* text from Claude Code sessions.

A Claude Code `.jsonl` session interleaves real user prompts with synthetic
messages that Claude Code injects under `type: "user"`: slash-command
boilerplate, command output, task notifications, system reminders and the
output of `!` bash commands. For search ranking and previews we want only
what the user actually wrote — including the arguments they pass to custom
slash commands and the `!` bash commands they type — and nothing else.

This module is the single source of truth for that filtering, shared by the
indexer (`__main__.py`) and the fzf preview subprocess.
"""

import json
import re
import unicodedata

# Strip terminal colour codes that leak into command output (e.g. stdout).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# Control characters (incl. \r and \t) and any escape-sequence leftovers.
_CTRL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]+")


def one_line(text: str) -> str:
    """Collapse ``text`` to a single, terminal-safe line.

    Pasted Windows text carries ``\\r\\n``: replacing only ``\\n`` left the
    ``\\r``, which sends the cursor back to column 0 and makes the rest of the
    line overwrite what was already drawn (garbled rows in fzf).
    """
    text = _CTRL_RE.sub(" ", _ANSI_RE.sub("", text or ""))
    return " ".join(text.split())


def display_width(text: str) -> int:
    """Terminal cell width: wide chars (emoji, CJK) take 2, combining marks 0."""
    return sum(_char_width(ch) for ch in text)


def _char_width(ch: str) -> int:
    if unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def fit(text: str, width: int, keep_end: bool = False) -> str:
    """Truncate (with "…") and pad ``text`` to exactly ``width`` terminal cells.

    ``keep_end`` keeps the tail instead of the head (useful for paths).
    """
    if display_width(text) > width:
        chars = list(reversed(text)) if keep_end else list(text)
        out, used = [], 1  # 1 cell reserved for "…"
        for ch in chars:
            w = _char_width(ch)
            if used + w > width:
                break
            out.append(ch)
            used += w
        text = "…" + "".join(reversed(out)) if keep_end else "".join(out) + "…"
    return text + " " * (width - display_width(text))

# Messages that are purely Claude Code machinery — no user-authored value.
_NOISE_PREFIXES = (
    "<local-command-caveat>",   # "Caveat: the messages below were generated…"
    "<local-command-stdout>",   # output of a slash command (e.g. /model, /effort)
    "<local-command-stderr>",
    "<bash-stdout>",            # output of a `!` bash command
    "<bash-stderr>",
    "<task-notification>",      # background-task / agent notifications
    "<system-reminder>",        # harness-injected reminders
)

_CMD_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.S)
_CMD_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.S)
_BASH_IN_RE = re.compile(r"<bash-input>(.*?)</bash-input>", re.S)


def clean_user_text(raw: str) -> str:
    """Return the user-authored part of a single user-message string.

    Keeps real content (prose, custom slash-command arguments, `!` bash input)
    and drops Claude Code's synthetic wrappers. Returns "" when the message
    carries no user-authored value.
    """
    if not raw:
        return ""
    text = _ANSI_RE.sub("", raw).strip()
    if not text or text.startswith(_NOISE_PREFIXES):
        return ""

    # Slash-command invocation. Keep the command name + the arguments the user
    # typed; drop built-in/config commands with no arguments (/model, /effort…)
    # since they carry no search value.
    if "<command-name>" in text:
        args = _CMD_ARGS_RE.search(text)
        args_s = args.group(1).strip() if args else ""
        if not args_s:
            return ""
        name = _CMD_NAME_RE.search(text)
        name_s = name.group(1).strip().lstrip("/") if name else ""
        return f"{name_s} {args_s}".strip()

    # `! cmd` bash input the user typed — keep the command, drop its output.
    if "<bash-input>" in text:
        return " ".join(m.strip() for m in _BASH_IN_RE.findall(text)).strip()

    return text


def content_to_text(content) -> str:
    """Extract cleaned user text from a message `content` (str or block list).

    For block lists only `text` blocks are considered (tool results, images,
    etc. are tool/system output, not user prose) and each is cleaned
    individually so an injected reminder block can't drag real text with it.
    """
    if isinstance(content, str):
        return clean_user_text(content)
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                cleaned = clean_user_text(block.get("text", ""))
                if cleaned:
                    parts.append(cleaned)
        return " ".join(parts).strip()
    return ""


def iter_user_texts(filepath):
    """Yield cleaned, non-empty user texts from a session file, in order."""
    with open(filepath, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "user":
                continue
            text = content_to_text(obj.get("message", {}).get("content", ""))
            if text:
                yield text


def preview_text(filepath, n: int = 10) -> str:
    """Build the multi-line preview shown in fzf / the numbered-list fallback."""
    lines = []
    for i, text in enumerate(iter_user_texts(filepath), 1):
        lines.append(f"[{i}] " + one_line(text)[:400])
        if i >= n:
            break
    return "\n".join(lines) if lines else "(no messages)"


# ── --all mode: Claude's side of the conversation ─────────────────────────────

# Tool-input fields skipped when indexing tool calls: they carry bulk payloads
# (whole file contents, edit strings) that would bloat the cache without
# helping a search. Paths, commands, patterns and descriptions are kept, so a
# session can be found by a file it wrote or a command it ran.
_TOOL_SKIP_KEYS = {"content", "old_string", "new_string", "edits", "new_source"}


def _tool_input_text(value) -> str:
    """Flatten the searchable string values of a tool_use `input`."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(
            _tool_input_text(v) for k, v in value.items() if k not in _TOOL_SKIP_KEYS
        ).strip()
    if isinstance(value, list):
        return " ".join(_tool_input_text(v) for v in value).strip()
    return ""


def assistant_content_to_text(content) -> str:
    """Extract Claude's reply text and tool-call inputs from an assistant message.

    Thinking blocks and tool results are left out: the former is internal, the
    latter is tool output (often huge) rather than part of the conversation.
    """
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            parts.append(_tool_input_text(block.get("input", {})))
    return " ".join(p.strip() for p in parts if p and p.strip())
