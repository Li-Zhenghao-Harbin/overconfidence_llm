"""MHPP (Mostly Hard Python Problems) → `tasks.jsonl` for local pipeline runs.

Official release: https://github.com/SparksofAGI/MHPP — public `MHPP.jsonl` has prompts only;
full hidden tests are evaluated on the authors' server. For local experiments we therefore use a
**smoke invocation** (compile + one callable probe with type-hint–guided default args) as a proxy
signal — not comparable to the official MHPP leaderboard.
"""

from __future__ import annotations

import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MHPP_URL = "https://raw.githubusercontent.com/SparksofAGI/MHPP/main/data/MHPP.jsonl"


def _last_def_line(prompt: str) -> str:
    lines = [ln.rstrip() for ln in prompt.splitlines() if ln.strip().startswith("def ")]
    return lines[-1] if lines else "def solution():\n"


def mhpp_program_stub(question: str, prompt: str) -> str:
    """Executable prefix for merge+exec: `question` is usually the def+docstring; `prompt` adds NL preamble."""
    q = (question or "").strip()
    if q.startswith("def "):
        return q
    p = (prompt or "").strip()
    i = p.find("def ")
    if i != -1:
        return p[i:].strip()
    return p


def _complexity_from_difficulty(d: int) -> str:
    """Map MHPP `difficulty_types` (paper: 7 challenge types) to three buckets for RQ4."""
    try:
        x = int(d)
    except (TypeError, ValueError):
        return "medium"
    if x <= 2:
        return "basic"
    if x <= 5:
        return "medium"
    return "complex"


def download_mhpp_rows(url: str) -> list[dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "OverconfidenceLens-dataset"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        text = resp.read().decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rows_to_task_dicts(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    if limit and limit > 0:
        rows = rows[:limit]
    out: list[dict[str, Any]] = []
    for r in rows:
        fn = str(r.get("function_name") or "").strip()
        prompt = str(r.get("prompt") or "")
        if not fn or not prompt:
            continue
        raw_id = r.get("id", fn)
        safe_id = f"MHPP_{raw_id}"
        diff = r.get("difficulty_types", 5)
        stub = mhpp_program_stub(str(r.get("question") or ""), prompt)
        out.append(
            {
                "task_id": safe_id,
                "complexity": _complexity_from_difficulty(diff),
                "domain": "mhpp",
                "title": f"MHPP/{fn}",
                "description": prompt,
                "function_signature": _last_def_line(stub),
                "examples": [],
                "mhpp": True,
                "entry_point": fn,
                "mhpp_parameters": r.get("parameters") or [],
                "mhpp_row_id": raw_id,
                "mhpp_program_stub": stub,
            }
        )
    return out


def write_tasks_jsonl(rows: list[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    out_path.write_text(text, encoding="utf-8")
    return len(rows)


def tasks_file_looks_mhpp(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                return False
            return bool(d.get("mhpp"))
    return False


def ensure_mhpp_tasks(config: dict, log: logging.Logger | None = None) -> None:
    lg = log or logger
    tasks_cfg = config.get("tasks") or {}
    out_path = Path(tasks_cfg.get("task_file", "data/raw/tasks.jsonl"))
    mhpp_cfg = tasks_cfg.get("mhpp") or {}
    url = str(mhpp_cfg.get("url") or DEFAULT_MHPP_URL)
    limit = int(mhpp_cfg.get("limit", 0) or 0)
    always = bool(mhpp_cfg.get("always_refresh", False))

    if out_path.is_file() and not always and tasks_file_looks_mhpp(out_path):
        lg.info("MHPP tasks file present; skipping download (%s)", out_path)
        return

    lg.info("Downloading MHPP → %s (limit=%s)", out_path, limit or "all")
    raw_rows = download_mhpp_rows(url)
    task_rows = rows_to_task_dicts(raw_rows, limit=limit)
    n = write_tasks_jsonl(task_rows, out_path)
    lg.info("Wrote %d MHPP tasks to %s", n, out_path.resolve())
