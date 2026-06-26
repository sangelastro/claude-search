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

# Strip terminal colour codes that leak into command output (e.g. stdout).
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

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
        lines.append(f"[{i}] " + text[:400].replace("\n", " "))
        if i >= n:
            break
    return "\n".join(lines) if lines else "(no messages)"
