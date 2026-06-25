#!/usr/bin/env python3
"""
claude-search: Search across Claude Code sessions and resume them.

Usage:
  claude-search <query>       Search sessions by content
  claude-search --list        List all sessions alphabetically by name
  claude-search -l            (same)
  claude-search "location history cluster"

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
from collections import defaultdict
from pathlib import Path


MAX_RESULTS = 30
PREVIEW_MESSAGES = 10
CACHE_PATH = Path.home() / ".cache" / "claude-search" / "index.json"
CACHE_VERSION = 2


def get_claude_dir() -> Path:
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

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return ""


def extract_session(filepath: Path):
    """Return (full_text, cwd, first_user_msg, slug) from a JSONL session file."""
    texts = []
    cwd = None
    first_user_msg = None
    slug = None

    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not slug and obj.get("slug"):
                slug = obj["slug"]

            if obj.get("type") != "user":
                continue

            text = _content_to_text(obj.get("message", {}).get("content", ""))
            if text:
                texts.append(text)
                if first_user_msg is None:
                    first_user_msg = text
            if not cwd and obj.get("cwd"):
                cwd = obj["cwd"]

    return " ".join(texts), cwd, first_user_msg or "", slug or ""


def build_preview(filepath: Path, n: int = PREVIEW_MESSAGES) -> str:
    lines = []
    count = 0
    with open(filepath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") != "user":
                continue
            text = _content_to_text(obj.get("message", {}).get("content", ""))
            if text:
                count += 1
                lines.append(f"[{count}] {text[:400].replace(chr(10), ' ')}")
                if count >= n:
                    break
    return "\n".join(lines) if lines else "(no messages)"


# ── scoring ────────────────────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9àèéìòùÀÈÉÌÒÙ_]+", text.lower())


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


# ── fzf preview script builder ─────────────────────────────────────────────────

def _make_preview_cmd(id_to_path: dict, tmpdir: str) -> str:
    preview_script = os.path.join(tmpdir, "preview.py")
    with open(preview_script, "w", encoding="utf-8") as f:
        f.write("import sys, json\n")
        f.write(f"id_to_path = {json.dumps(id_to_path)}\n")
        f.write(
            "sid = sys.argv[1].strip() if len(sys.argv) > 1 else ''\n"
            "path = id_to_path.get(sid)\n"
            "if not path: print('(not found)'); sys.exit(0)\n"
            "count = 0\n"
            "with open(path, encoding='utf-8', errors='ignore') as fh:\n"
            "  for row in fh:\n"
            "    row = row.strip()\n"
            "    if not row: continue\n"
            "    try: obj = json.loads(row)\n"
            "    except: continue\n"
            "    if obj.get('type') != 'user': continue\n"
            "    c = obj.get('message', {}).get('content', '')\n"
            "    t = c if isinstance(c, str) else ' '.join(b.get('text','') for b in c if isinstance(b,dict) and b.get('type')=='text')\n"
            "    t = t.strip()\n"
            "    if t:\n"
            "      count += 1\n"
            "      print(f'[{count}] ' + t[:400].replace('\\n',' '))\n"
            "      if count >= 10: break\n"
        )

    is_windows = platform.system() == "Windows"
    if is_windows:
        preview_bat = os.path.join(tmpdir, "preview.bat")
        with open(preview_bat, "w", encoding="utf-8") as f:
            f.write(f'@echo off\n"{sys.executable}" "{preview_script}" %1\n')
        return f"{preview_bat} {{2}}"
    else:
        os.chmod(preview_script, 0o755)
        return f"{sys.executable} {preview_script} {{2}}"


def _run_fzf(fzf_lines: list[str], header: str, preview_cmd: str) -> str | None:
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
            f' --preview="{preview_cmd}"'
            f' --preview-window=right:50%:wrap'
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


# ── selection UI ───────────────────────────────────────────────────────────────

# Session tuple inside ranked: (score, session_id, text, cwd, first_msg, slug, jsonl_path)
# Fields in fzf search line:    score | session_id | slug | cwd | label
#   → field {2} = session_id (used by preview), parts[1]=session_id, parts[3]=cwd

def _fzf_select(ranked, query: str, id_to_path: dict) -> tuple[str, str] | None:
    """Search results with fzf. Returns (session_id, cwd) or None."""
    tmpdir = tempfile.mkdtemp(prefix="claude-search-")
    try:
        preview_cmd = _make_preview_cmd(id_to_path, tmpdir)
        fzf_lines = []
        for score, session_id, _, cwd, first_msg, slug, _ in ranked:
            label = first_msg[:80].replace("\n", " ")
            short_cwd = cwd.replace(str(Path.home()), "~")
            slug_col = slug if slug else "-"
            fzf_lines.append(f"{score:.3f} | {session_id} | {slug_col} | {short_cwd} | {label}")

        header = f"Search: {query} — {len(ranked)} results"
        selected = _run_fzf(fzf_lines, header, preview_cmd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not selected:
        return None
    parts = [p.strip() for p in selected.split("|")]
    # parts[1]=session_id, parts[3]=cwd
    return parts[1], parts[3].replace("~", str(Path.home()))


def _list_select(ranked) -> tuple[str, str] | None:
    """Numbered list fallback for search. Returns (session_id, cwd) or None."""
    print(file=sys.stderr)
    for i, (score, session_id, _, cwd, first_msg, slug, _) in enumerate(ranked, 1):
        short_cwd = cwd.replace(str(Path.home()), "~")
        label = first_msg[:90].replace("\n", " ")
        slug_str = f" [{slug}]" if slug else ""
        print(f"  {i:2}. [{score:.3f}]{slug_str} {label}", file=sys.stderr)
        print(f"       {short_cwd}  ({session_id[:8]}...)\n", file=sys.stderr)

    choice = input("Select number (Enter to cancel): ").strip()
    if not choice:
        return None
    try:
        idx = int(choice) - 1
        _, chosen_id, _, chosen_cwd, _, _, _ = ranked[idx]
        return chosen_id, chosen_cwd
    except (ValueError, IndexError):
        print("Invalid selection.", file=sys.stderr)
        return None


# ── alphabetical list UI ───────────────────────────────────────────────────────

# fzf list line: slug_or_name | session_id | cwd | label
#   → field {2} = session_id (preview), parts[1]=session_id, parts[2]=cwd

def _fzf_list_all(sessions, id_to_path: dict) -> tuple[str, str] | None:
    """Alphabetical list of all sessions with fzf. Returns (session_id, cwd) or None."""
    tmpdir = tempfile.mkdtemp(prefix="claude-search-")
    try:
        preview_cmd = _make_preview_cmd(id_to_path, tmpdir)
        # sessions tuple: (session_id, text, cwd, first_msg, slug, jsonl_path)
        sorted_sessions = sorted(
            sessions,
            key=lambda s: (s[4].lower() if s[4] else "\xff", s[3][:60].lower()),
        )
        fzf_lines = []
        for session_id, _, cwd, first_msg, slug, _ in sorted_sessions:
            label = first_msg[:80].replace("\n", " ")
            short_cwd = cwd.replace(str(Path.home()), "~")
            name_col = slug if slug else f"({first_msg[:40].replace(chr(10), ' ')})"
            fzf_lines.append(f"{name_col} | {session_id} | {short_cwd} | {label}")

        header = f"All sessions — {len(sessions)} total (alphabetical)"
        selected = _run_fzf(fzf_lines, header, preview_cmd)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if not selected:
        return None
    parts = [p.strip() for p in selected.split("|")]
    # parts[1]=session_id, parts[2]=cwd
    return parts[1], parts[2].replace("~", str(Path.home()))


def _list_all_select(sessions) -> tuple[str, str] | None:
    """Numbered alphabetical list fallback. Returns (session_id, cwd) or None."""
    sorted_sessions = sorted(
        sessions,
        key=lambda s: (s[4].lower() if s[4] else "\xff", s[3][:60].lower()),
    )
    print(file=sys.stderr)
    for i, (session_id, _, cwd, first_msg, slug, _) in enumerate(sorted_sessions, 1):
        short_cwd = cwd.replace(str(Path.home()), "~")
        label = first_msg[:70].replace("\n", " ")
        name = slug if slug else f"({label[:40]})"
        print(f"  {i:3}. {name}", file=sys.stderr)
        print(f"        {short_cwd}  ({session_id[:8]}...)\n", file=sys.stderr)

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

    list_mode = args[0] in ("--list", "-l")
    if not list_mode:
        query = " ".join(args)
    else:
        query = ""

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
                cwd = entry["cwd"]
                first_msg = entry["first_msg"]
                slug = entry.get("slug", "")
            else:
                text, cwd, first_msg, slug = extract_session(jsonl_file)
                cache[key] = {
                    "mtime": mtime,
                    "text": text,
                    "cwd": cwd or str(project_dir),
                    "first_msg": first_msg,
                    "slug": slug,
                }
                updated = True
            if text.strip():
                sessions.append((
                    jsonl_file.stem,
                    text,
                    cwd or str(project_dir),
                    first_msg,
                    slug,
                    jsonl_file,
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
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # ranked tuple: (score, session_id, text, cwd, first_msg, slug, jsonl_path)
    id_to_path = {r[1]: str(r[6]) for r in ranked}

    print(f"Found {len(ranked)} results [{method}] for: '{query}'\n", file=sys.stderr)

    selection = (
        _fzf_select(ranked, query, id_to_path)
        if has_fzf
        else _list_select(ranked)
    )

    if selection:
        resume(*selection)


if __name__ == "__main__":
    main()
