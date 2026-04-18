"""HumanEval → `tasks.jsonl` conversion (shared by `main.py` and `scripts/import_humaneval.py`)."""

from __future__ import annotations

import gzip
import io
import json
import logging
import re
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HUMANEVAL_URL = (
    "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"
)


def _last_def_line(prompt: str) -> str:
    lines = [ln.rstrip() for ln in prompt.splitlines() if ln.strip().startswith("def ")]
    return lines[-1] if lines else "def solution():\n"


def _complexity_bucket(lengths: list[int], x: int) -> str:
    if not lengths:
        return "medium"
    s = sorted(lengths)
    n = len(s)
    t1 = s[max(0, n // 3 - 1)]
    t2 = s[max(2 * n // 3 - 1, 0)]
    if x <= t1:
        return "basic"
    if x <= t2:
        return "medium"
    return "complex"


def download_humaneval_rows(url: str) -> list[dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "OverconfidenceLens-dataset"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        raw = resp.read()
    buf = gzip.GzipFile(fileobj=io.BytesIO(raw), mode="rb")
    text = buf.read().decode("utf-8")
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def rows_to_task_dicts(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    if limit and limit > 0:
        rows = rows[:limit]
    lengths = [len(r.get("prompt", "")) for r in rows]
    out: list[dict[str, Any]] = []
    for r in rows:
        raw_id = r.get("task_id", "HumanEval/unknown")
        safe_id = str(raw_id).replace("/", "_").replace(" ", "_")
        prompt = r.get("prompt", "")
        test_src = r.get("test", "")
        entry = r.get("entry_point", "")
        if not test_src or not entry:
            continue
        L = len(prompt)
        out.append(
            {
                "task_id": safe_id,
                "complexity": _complexity_bucket(lengths, L),
                "domain": "humaneval",
                "title": raw_id,
                "description": prompt,
                "function_signature": _last_def_line(prompt),
                "examples": [],
                "humaneval": True,
                "humaneval_test": test_src,
                "entry_point": entry,
            }
        )
    return out


def write_tasks_jsonl(rows: list[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    out_path.write_text(text, encoding="utf-8")
    return len(rows)


def tasks_file_looks_humaneval(path: Path) -> bool:
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
            return bool(d.get("humaneval") or d.get("humaneval_test"))
    return False


def ensure_dataset_tasks(config: dict, log: logging.Logger | None = None) -> None:
    """Prepare `tasks.task_file` according to `tasks.dataset` before phases load tasks."""
    lg = log or logger
    tasks_cfg = config.get("tasks") or {}
    dataset = str(tasks_cfg.get("dataset", "builtin")).strip().lower()
    out_path = Path(tasks_cfg.get("task_file", "data/raw/tasks.jsonl"))

    if dataset in ("", "builtin", "default"):
        if out_path.is_file() and tasks_file_looks_humaneval(out_path):
            lg.warning(
                "tasks.dataset is builtin but %s looks like HumanEval. "
                "Delete the file or switch tasks.dataset to humaneval.",
                out_path,
            )
        from src.module1_data.mhpp_dataset import tasks_file_looks_mhpp

        if out_path.is_file() and tasks_file_looks_mhpp(out_path):
            lg.warning(
                "tasks.dataset is builtin but %s looks like MHPP. "
                "Delete the file or switch tasks.dataset to mhpp.",
                out_path,
            )
        from src.module1_data.apps_dataset import tasks_file_looks_apps

        if out_path.is_file() and tasks_file_looks_apps(out_path):
            lg.warning(
                "tasks.dataset is builtin but %s looks like APPS. "
                "Delete the file or switch tasks.dataset to apps.",
                out_path,
            )
        return

    if dataset == "mhpp":
        from src.module1_data.mhpp_dataset import ensure_mhpp_tasks

        ensure_mhpp_tasks(config, lg)
        return

    if dataset == "apps":
        from src.module1_data.apps_dataset import ensure_apps_tasks

        ensure_apps_tasks(config, lg)
        return

    if dataset != "humaneval":
        raise ValueError(
            f"Unknown tasks.dataset: {dataset!r} (use 'builtin', 'humaneval', 'mhpp', or 'apps')"
        )

    he_cfg = tasks_cfg.get("humaneval") or {}
    url = str(he_cfg.get("url") or DEFAULT_HUMANEVAL_URL)
    limit = int(he_cfg.get("limit", 0) or 0)
    always = bool(he_cfg.get("always_refresh", False))

    if out_path.is_file() and not always and tasks_file_looks_humaneval(out_path):
        lg.info("HumanEval tasks file present; skipping download (%s)", out_path)
        return

    lg.info("Downloading HumanEval → %s (limit=%s)", out_path, limit or "all")
    raw_rows = download_humaneval_rows(url)
    task_rows = rows_to_task_dicts(raw_rows, limit=limit)
    n = write_tasks_jsonl(task_rows, out_path)
    lg.info("Wrote %d HumanEval tasks to %s", n, out_path.resolve())
