#!/usr/bin/env python3
"""
claude-search: Search across Claude Code sessions and resume them.

Usage:
  claude-search <query>       Search sessions by what you wrote
  claude-search --all <query> Also search Claude's replies and tool calls
                              (file paths written/edited, commands run)
  claude-search -a <query>    (same)
  claude-search --list        List all sessions alphabetically by name
  claude-search -l            (same)
  claude-search "location history cluster"
  claude-search -a Report_Vendite_Q3_v2

Sessions are read from $CLAUDE_CONFIG_DIR/projects when the variable is set,
otherwise from ~/.claude/projects.

Requires: python3.11+ (stdlib only)
Optional:
  rank-bm25  Better ranking than TF-IDF:  pip install rank-bm25
  fzf        Interactive UI with preview:
               Linux/Mac:  sudo apt install fzf  |  brew install fzf
               Windows:    winget install fzf
"""

import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from claude_search._extract import assistant_content_to_text, content_to_text


MAX_RESULTS = 30
PREVIEW_MESSAGES = 10
CACHE_PATH = Path.home() / ".cache" / "claude-search" / "index.json"
# Bumped to 6: entries also store the session's created/updated timestamps.
CACHE_VERSION = 6


def get_claude_dir() -> Path:
    # Respect the same variable Claude Code uses to pick the account, so
    # `CLAUDE_CONFIG_DIR=... claude-search` searches that account's sessions.
    config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / "projects"
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home()))
        candidate = base / "Claude" / "projects"
        if candidate.exists():
            return candidate
    return Path.home() / ".claude" / "projects"


# ── cache ──────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        if data.get("_version") != CACHE_VERSION:
            return {}
        return data
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        cache["_version"] = CACHE_VERSION
        CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    except Exception:
        pass


# ── text extraction ────────────────────────────────────────────────────────────

def extract_session(filepath: Path):
    """Return (full_text, assistant_text, cwd, first_user_msg, name, created, updated).

    ``created`` / ``updated`` are the first and last ISO timestamps found in
    the session file ("" if it has none).

    Only user-authored text is kept (see ``_extract.content_to_text``):
    Claude Code's synthetic messages — command boilerplate/output, task
    notifications, system reminders and `!` bash output — are filtered out.

    ``assistant_text`` holds Claude's replies and tool-call inputs, searched
    only with --all.

    name priority: customTitle (from /rename) > slug (auto-generated) > ""
    """
    texts = []
    assistant_texts = []
    cwd = None
    first_user_msg = None
    custom_title = None
    slug = None
    created = ""
    updated = ""

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = obj.get("timestamp")
            if ts:
                created = created or ts
                updated = ts

            if obj.get("type") == "custom-title" and obj.get("customTitle"):
                custom_title = obj["customTitle"]

            if not slug and obj.get("slug"):
                slug = obj["slug"]

            if obj.get("type") == "assistant":
                text = assistant_content_to_text(obj.get("message", {}).get("content", ""))
                if text:
                    assistant_texts.append(text)
                continue

            if obj.get("type") != "user":
                continue

            text = content_to_text(obj.get("message", {}).get("content", ""))
            if text:
                texts.append(text)
                if first_user_msg is None:
                    first_user_msg = text
            if not cwd and obj.get("cwd"):
                cwd = obj["cwd"]

    name = custom_title or slug or ""
    return (
        " ".join(texts), " ".join(assistant_texts), cwd, first_user_msg or "", name,
        created, updated,
    )


def format_dates(created: str, updated: str) -> str:
    """Render "created -> updated" in local time, e.g. "30/09/26 13:05 -> 30/09/26 15:33"."""
    def fmt(ts: str) -> str:
        try:
            return datetime.fromisoformat(ts).astimezone().strftime("%d/%m/%y %H:%M")
        except (ValueError, TypeError):
            return "?"
    if not created:
        return "?"
    return f"{fmt(created)} -> {fmt(updated or created)}"


# ── scoring ────────────────────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


def tokenize(text: str) -> list[str]:
    # Identifiers like `Report_Vendite_Q3_v2_2026` are kept whole and
    # also split on `_`, so a partial name (`Report_Vendite_Q3_v2`)
    # still matches through its parts.
    tokens = []
    for tok in re.findall(r"[a-zA-Z0-9àèéìòùÀÈÉÌÒÙ_]+", text.lower()):
        tokens.append(tok)
        if "_" in tok:
            tokens.extend(p for p in tok.split("_") if p)
    return tokens


def _score_bm25(query: str, corpus: list[str]) -> list[float]:
    tokenized = [tokenize(text) for text in corpus]
    bm25 = BM25Okapi(tokenized)
    return list(bm25.get_scores(tokenize(query)))


def _score_tfidf(query: str, corpus: list[str]) -> list[float]:
    N = len(corpus)
    tf_list = []
    df: dict[str, int] = defaultdict(int)

    for text in corpus:
        tokens = tokenize(text)
        tf: dict[str, float] = defaultdict(float)
        total = len(tokens) or 1
        for t in tokens:
            tf[t] += 1.0 / total
        tf_list.append(tf)
        for term in tf:
            df[term] += 1

    idf = {term: math.log((N + 1) / (count + 1)) + 1 for term, count in df.items()}

    def cosine(a: dict, b: dict) -> float:
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    q_tokens = tokenize(query)
    total = len(q_tokens) or 1
    q_tf: dict[str, float] = defaultdict(float)
    for t in q_tokens:
        q_tf[t] += 1.0 / total
    q_vec = {t: q_tf[t] * idf.get(t, 1.0) for t in q_tf}

    tfidf_vectors = [{t: tf[t] * idf.get(t, 1.0) for t in tf} for tf in tf_list]
    return [cosine(q_vec, v) for v in tfidf_vectors]


def score_sessions(query: str, corpus: list[str]) -> tuple[list[float], str]:
    """Return (scores, method_name). Uses BM25 if available, TF-IDF otherwise."""
    if _HAS_BM25:
        return _score_bm25(query, corpus), "BM25"
    return _score_tfidf(query, corpus), "TF-IDF"


# ── banner (animated) ──────────────────────────────────────────────────────────

# Compact Claude logo (~20 lines), same = / + style as the original.
_LOGO_LINES = [
    "          ======                              ",
    "        ==========                            ",
    "       ============          ====             ",
    "       ============        =======            ",
    "       ============       ========            ",
    "        ===========      ========             ",
    "          =========     ========+             ",
    "           =======    +========               ",
    "            ==================                ",
    "             ================                 ",
    "            ==================                ",
    "           =======    +========               ",
    "          =========     ========+             ",
    "        ===========      ========             ",
    "       ============       ========            ",
    "       ============        =======            ",
    "       ============          ====             ",
    "        ==========                            ",
    "          ======                              ",
]

# Gradient palette: dark amber → bright orange → pale gold
_PALETTE = [
    "\033[38;5;94m",   # dark amber-brown
    "\033[38;5;130m",  # dark orange
    "\033[38;5;166m",  # medium orange
    "\033[38;5;208m",  # Claude orange
    "\033[38;5;214m",  # amber-gold
    "\033[38;5;220m",  # gold
    "\033[38;5;222m",  # pale gold
]


def _print_banner() -> None:
    """Animated Claude logo: diagonal colour wave, same style as Claude chat."""
    if not sys.stderr.isatty():
        return

    R    = "\033[0m"
    W    = "\033[1;97m"
    D    = "\033[2;37m"
    HIDE = "\033[?25l"
    SHOW = "\033[?25h"

    P    = _PALETTE
    NP   = len(P)
    NLINES = len(_LOGO_LINES) + 2   # logo + 2 text lines
    UP   = f"\033[{NLINES}A"

    def _render(t: float) -> str:
        rows = []
        for y, line in enumerate(_LOGO_LINES):
            row = []
            for x, ch in enumerate(line):
                if ch in ("=", "+"):
                    # diagonal wave: top-right → bottom-left
                    wave = math.sin((x * 0.15 - y * 0.25 + t) * math.pi)
                    idx = min(NP - 1, max(0, int((wave + 1) / 2 * (NP - 1))))
                    row.append(P[idx] + ch + R)
                else:
                    row.append(ch)
            rows.append("".join(row))
        rows.append(f"  {W}CLAUDE  SEARCH{R}")
        rows.append(f"  {D}AI Session Explorer{R}")
        return "\n".join(rows) + "\n"

    try:
        sys.stderr.write(HIDE)
        sys.stderr.write(_render(0.0))
        sys.stderr.flush()

        for i in range(1, 20):          # 20 frames × 70 ms ≈ 1.4 s
            time.sleep(0.07)
            sys.stderr.write(UP)
            sys.stderr.write(_render(i * 0.18))
            sys.stderr.flush()

        # Erase logo, keep only the text header for context
        sys.stderr.write(UP)
        sys.stderr.write("\033[0J")     # erase to end of screen
        sys.stderr.flush()

    except Exception:
        pass
    finally:
        sys.stderr.write(SHOW)
        sys.stderr.flush()


# ── fzf helpers ────────────────────────────────────────────────────────────────

def _make_preview_cmd(id_to_path: dict, tmpdir: str, field: int = 2) -> str:
    """Build fzf preview shell command. `field` is the 1-based fzf field with the session_id."""
    # The preview runs in a separate Python subprocess spawned by fzf, so it
    # reuses the same filtering as the indexer by importing claude_search._extract
    # (the package dir is injected into sys.path to work for any install mode).
    pkg_parent = str(Path(__file__).resolve().parent.parent)
    preview_script = os.path.join(tmpdir, "preview.py")
    with open(preview_script, "w", encoding="utf-8") as f:
        f.write("import sys\n")
        f.write(f"sys.path.insert(0, {json.dumps(pkg_parent)})\n")
        f.write("from claude_search._extract import preview_text\n")
        f.write(f"id_to_path = {json.dumps(id_to_path)}\n")
        f.write(
            "sid = sys.argv[1].strip() if len(sys.argv) > 1 else ''\n"
            "path = id_to_path.get(sid)\n"
            f"print(preview_text(path, {PREVIEW_MESSAGES}) if path else '(not found)')\n"
        )

    fld_placeholder = f"{{{field}}}"   # fzf substitution, e.g. {3}
    is_windows = platform.system() == "Windows"
    if is_windows:
        preview_bat = os.path.join(tmpdir, "preview.bat")
        with open(preview_bat, "w", encoding="utf-8") as f:
            # fzf passes the extracted field as the FIRST argument to the bat;
            # use %1 regardless of which field number was selected.
            f.write(f'@echo off\n"{sys.executable}" "{preview_script}" %1\n')
        return f"{preview_bat} {fld_placeholder}"
    else:
        os.chmod(preview_script, 0o755)
        return f"{sys.executable} {preview_script} {fld_placeholder}"


def _run_fzf(fzf_lines: list[str], header: str, preview_cmd: str, with_nth: str) -> str | None:
    """Run fzf and return the selected line, or None if cancelled."""
    tmpdir = tempfile.mkdtemp(prefix="claude-search-")
    input_file = os.path.join(tmpdir, "input.txt")
    output_file = os.path.join(tmpdir, "output.txt")
    try:
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("\n".join(fzf_lines))
        subprocess.run(
            f'fzf'
            f' --delimiter="|"'
            f' --with-nth={with_nth}'
            f' --preview="{preview_cmd}"'
            f' --preview-window=down:40%:wrap'
            f' --height=90%'
            f' --layout=reverse'
            f' --border'
            f' --header="{header}"'
            f' --prompt="Select session > "'
            f' < "{input_file}"'
            f' > "{output_file}"',
            shell=True,
        )
        return open(output_file, encoding="utf-8").read().strip() or None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── selection UI (search mode) ─────────────────────────────────────────────────

# Search fzf line:  pct% | name | session_id | dates | cwd | first_msg
#   displayed (--with-nth=1,2,4,5,6): score%, name, dates, cwd, first_msg
#   field 3 = session_id (hidden, used by preview via {3})
#   parse: parts[2]=session_id, parts[4]=cwd

def _fzf_select(ranked, query: str, id_to_path: dict) -> tuple[str, str] | None:
    """Search results with fzf. Returns (session_id, cwd) or None."""
    max_score = ranked[0][0] if ranked else 1.0
    tmpdir = tempfile.mkdtemp(prefix="claude-search-")
    try:
        preview_cmd = _make_preview_cmd(id_to_path, tmpdir, field=3)
        fzf_lines = []
        for score, session_id, _, cwd, first_msg, name, _, dates in ranked:
            pct = int(score / max_score * 100) if max_score > 0 else 0
            label = first_msg[:80].replace("\n", " ")
            short_cwd = cwd.replace(str(Path.home()), "~")
            name_col = name if name else "(no name)"
            fzf_lines.append(f"{pct:3d}% | {name_col} | {session_id} | {dates} | {short_cwd} | {label}")

        header = f"Search: {query} — {len(ranked)} results"
        selected = _run_fzf(fzf_lines, header, preview_cmd, with_nth="1,2,4,5,6")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not selected:
        return None
    parts = [p.strip() for p in selected.split("|")]
    # parts[2]=session_id, parts[4]=cwd
    return parts[2], parts[4].replace("~", str(Path.home()))


def _list_select(ranked) -> tuple[str, str] | None:
    """Numbered list fallback for search (no fzf). Returns (session_id, cwd) or None."""
    max_score = ranked[0][0] if ranked else 1.0
    print(file=sys.stderr)
    for i, (score, session_id, _, cwd, first_msg, name, _, dates) in enumerate(ranked, 1):
        pct = int(score / max_score * 100) if max_score > 0 else 0
        short_cwd = cwd.replace(str(Path.home()), "~")
        label = first_msg[:90].replace("\n", " ")
        name_str = f" [{name}]" if name else ""
        print(f"  {i:2}. {pct:3d}%{name_str} {label}", file=sys.stderr)
        print(f"       {dates}  {short_cwd}  ({session_id[:8]}...)\n", file=sys.stderr)

    choice = input("Select number (Enter to cancel): ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        _, chosen_id, _, chosen_cwd, _, _, _, _ = ranked[idx]
        return chosen_id, chosen_cwd
    except (ValueError, IndexError):
        print("Invalid selection.", file=sys.stderr)
        return None


# ── selection UI (alphabetical list mode) ─────────────────────────────────────

# List fzf line:  name | session_id | dates | cwd | first_msg
#   displayed (--with-nth=1,3,4,5): name, dates, cwd, first_msg
#   field 2 = session_id (hidden, used by preview via {2})
#   parse: parts[1]=session_id, parts[3]=cwd

def _fzf_list_all(sessions, id_to_path: dict) -> tuple[str, str] | None:
    """Alphabetical list of all sessions with fzf. Returns (session_id, cwd) or None."""
    tmpdir = tempfile.mkdtemp(prefix="claude-search-")
    try:
        preview_cmd = _make_preview_cmd(id_to_path, tmpdir, field=2)
        # sessions tuple: (session_id, text, cwd, first_msg, name, jsonl_path, dates)
        sorted_sessions = sorted(
            sessions,
            key=lambda s: (s[4].lower() if s[4] else "\xff", s[3][:60].lower()),
        )
        fzf_lines = []
        for session_id, _, cwd, first_msg, name, _, dates in sorted_sessions:
            label = first_msg[:80].replace("\n", " ")
            short_cwd = cwd.replace(str(Path.home()), "~")
            name_col = name if name else f"({first_msg[:40].replace(chr(10), ' ')})"
            fzf_lines.append(f"{name_col} | {session_id} | {dates} | {short_cwd} | {label}")

        header = f"All sessions — {len(sessions)} total (alphabetical)"
        selected = _run_fzf(fzf_lines, header, preview_cmd, with_nth="1,3,4,5")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not selected:
        return None
    parts = [p.strip() for p in selected.split("|")]
    # parts[1]=session_id, parts[3]=cwd
    return parts[1], parts[3].replace("~", str(Path.home()))


def _list_all_select(sessions) -> tuple[str, str] | None:
    """Numbered alphabetical list fallback (no fzf). Returns (session_id, cwd) or None."""
    # sessions tuple: (session_id, text, cwd, first_msg, name, jsonl_path, dates)
    sorted_sessions = sorted(
        sessions,
        key=lambda s: (s[4].lower() if s[4] else "\xff", s[3][:60].lower()),
    )
    print(file=sys.stderr)
    for i, (session_id, _, cwd, first_msg, name, _, dates) in enumerate(sorted_sessions, 1):
        short_cwd = cwd.replace(str(Path.home()), "~")
        display_name = name if name else f"({first_msg[:40].replace(chr(10), ' ')})"
        print(f"  {i:3}. {display_name}", file=sys.stderr)
        print(f"        {dates}  {short_cwd}  ({session_id[:8]}...)\n", file=sys.stderr)

    choice = input("Select number (Enter to cancel): ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        chosen = sorted_sessions[idx]
        return chosen[0], chosen[2]
    except (ValueError, IndexError):
        print("Invalid selection.", file=sys.stderr)
        return None


# ── resume ─────────────────────────────────────────────────────────────────────

def resume(session_id: str, cwd: str) -> None:
    print(f"\nResuming {session_id}")
    print(f"Directory: {cwd}\n")
    os.chdir(cwd)
    if platform.system() == "Windows":
        subprocess.run(["claude", "--resume", session_id], check=False)
    else:
        os.execvp("claude", ["claude", "--resume", session_id])


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]

    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    _print_banner()

    all_mode = any(a in ("--all", "-a") for a in args)
    args = [a for a in args if a not in ("--all", "-a")]
    if not args:
        print(__doc__)
        sys.exit(0)

    list_mode = args[0] in ("--list", "-l")
    query = "" if list_mode else " ".join(args)

    claude_dir = get_claude_dir()

    if not claude_dir.exists():
        print(f"Claude sessions directory not found: {claude_dir}", file=sys.stderr)
        sys.exit(1)

    cache = _load_cache()
    updated = False
    sessions = []

    for project_dir in sorted(claude_dir.iterdir()):
        if not project_dir.is_dir():
            continue
        for jsonl_file in sorted(project_dir.glob("*.jsonl")):
            key = str(jsonl_file)
            mtime = jsonl_file.stat().st_mtime
            entry = cache.get(key)
            if entry and entry.get("mtime") == mtime:
                text = entry["text"]
                assistant_text = entry.get("assistant_text", "")
                cwd = entry["cwd"]
                first_msg = entry["first_msg"]
                name = entry.get("name", "")
                created = entry.get("created", "")
                last_update = entry.get("updated", "")
            else:
                (text, assistant_text, cwd, first_msg, name,
                 created, last_update) = extract_session(jsonl_file)
                cache[key] = {
                    "mtime": mtime,
                    "text": text,
                    "assistant_text": assistant_text,
                    "cwd": cwd or str(project_dir),
                    "first_msg": first_msg,
                    "name": name,
                    "created": created,
                    "updated": last_update,
                }
                updated = True
            if all_mode:
                text = f"{text} {assistant_text}"
            if text.strip():
                sessions.append((
                    jsonl_file.stem,
                    text,
                    cwd or str(project_dir),
                    first_msg,
                    name,
                    jsonl_file,
                    format_dates(created, last_update),
                ))

    if updated:
        _save_cache(cache)

    if not sessions:
        print("No sessions found.", file=sys.stderr)
        sys.exit(1)

    has_fzf = shutil.which("fzf") is not None and sys.stdin.isatty()

    if list_mode:
        print(f"Listing {len(sessions)} sessions alphabetically ...\n", file=sys.stderr)
        id_to_path = {s[0]: str(s[5]) for s in sessions}
        selection = (
            _fzf_list_all(sessions, id_to_path)
            if has_fzf
            else _list_all_select(sessions)
        )
        if selection:
            resume(*selection)
        return

    print(f"Indexing {len(sessions)} sessions ...", file=sys.stderr)

    scores, method = score_sessions(query, [s[1] for s in sessions])

    ranked = [
        (score, *sess)
        for score, sess in sorted(zip(scores, sessions), key=lambda x: x[0], reverse=True)
        if score > 0
    ][:MAX_RESULTS]

    if not ranked:
        hint = "" if all_mode else "  (only your messages were searched; try --all)"
        print("No results found." + hint, file=sys.stderr)
        sys.exit(1)

    # ranked tuple: (score, session_id, text, cwd, first_msg, name, jsonl_path, dates)
    id_to_path = {r[1]: str(r[6]) for r in ranked}

    scope = " incl. Claude's replies" if all_mode else ""
    print(f"Found {len(ranked)} results [{method}{scope}] for: '{query}'\n", file=sys.stderr)

    selection = (
        _fzf_select(ranked, query, id_to_path)
        if has_fzf
        else _list_select(ranked)
    )

    if selection:
        resume(*selection)


if __name__ == "__main__":
    main()
