"""APPS benchmark → `tasks.jsonl` (Hugging Face `codeparrot/apps`).

Uses public **stdin/stdout** test pairs from `input_output` when `fn_name` is absent
(call-based tasks are skipped in this integration — extend later if needed).

See: https://huggingface.co/datasets/codeparrot/apps
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_HF_ID = "codeparrot/apps"


def normalize_apps_stdout(s: str) -> str:
    """Normalize APPS stdout comparison (line endings + trailing spaces)."""
    return "\n".join(
        line.rstrip() for line in (s or "").replace("\r\n", "\n").splitlines()
    ).rstrip("\n")


def _last_def_line_from_starter(starter: str) -> str:
    lines = [ln.rstrip() for ln in (starter or "").splitlines() if ln.strip().startswith("def ")]
    return lines[-1] if lines else "def solution():\n    pass\n"


def _complexity_from_apps(difficulty: str) -> str:
    d = (difficulty or "").strip().lower()
    if d == "introductory":
        return "basic"
    if d == "interview":
        return "medium"
    if d == "competition":
        return "complex"
    return "medium"


def _parse_json_field(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return None
        return json.loads(s)
    return val


def hf_rows_to_task_dicts(
    rows: list[dict[str, Any]],
    *,
    max_tests_per_task: int = 5,
    skip_fn_name: bool = True,
) -> list[dict[str, Any]]:
    """Convert HF APPS rows into task records."""
    out: list[dict[str, Any]] = []

    for r in rows:
        pid = r.get("problem_id")
        safe_id = f"APPS_{pid}"
        question = str(r.get("question") or "")
        starter = str(r.get("starter_code") or "")
        diff = str(r.get("difficulty") or "interview")

        io = _parse_json_field(r.get("input_output")) or {}
        inputs = io.get("inputs") or []
        outputs = io.get("outputs") or []
        fn_name = io.get("fn_name")

        if skip_fn_name and fn_name:
            continue
        if not inputs or not outputs or len(inputs) != len(outputs):
            continue

        n = min(len(inputs), max(1, int(max_tests_per_task)))
        tests: list[dict[str, str]] = []
        for i in range(n):
            tests.append(
                {
                    "stdin": str(inputs[i]),
                    "expected_stdout": normalize_apps_stdout(str(outputs[i])),
                }
            )

        desc_parts = [question.strip()]
        if starter.strip():
            desc_parts.append("Starter code (you may complete or replace):\n" + starter.strip())
        desc_parts.append(
            "Return a **complete Python program** in a single ```python``` block that reads from "
            "standard input and writes to standard output exactly as required by the problem."
        )
        description = "\n\n".join(desc_parts)

        cx = _complexity_from_apps(diff)

        out.append(
            {
                "task_id": safe_id,
                "complexity": cx,
                "domain": "apps",
                "title": f"APPS/{pid}",
                "description": description,
                "function_signature": _last_def_line_from_starter(starter),
                "examples": [],
                "apps": True,
                "apps_problem_id": pid,
                "apps_difficulty": diff,
                "apps_tests": tests,
            }
        )
    return out


def write_tasks_jsonl(rows: list[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else "")
    out_path.write_text(text, encoding="utf-8")
    return len(rows)


def tasks_file_looks_apps(path: Path) -> bool:
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
            return bool(d.get("apps"))
    return False


def ensure_apps_tasks(config: dict, log: logging.Logger | None = None) -> None:
    """Download / filter APPS via Hugging Face `datasets` and write `tasks.task_file`."""
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "APPS 需要安装 Hugging Face datasets：\n  pip install datasets\n"
        ) from e

    lg = log or logger
    tasks_cfg = config.get("tasks") or {}
    out_path = Path(tasks_cfg.get("task_file", "data/raw/tasks.jsonl"))
    apps_cfg = tasks_cfg.get("apps") or {}
    hf_id = str(apps_cfg.get("hf_dataset", DEFAULT_HF_ID))
    split = str(apps_cfg.get("split", "train"))
    # datasets>=3 no longer runs Hub Python dataset scripts; use parquet export revision.
    hf_revision = apps_cfg.get("hf_revision", "refs/convert/parquet")
    limit = int(apps_cfg.get("limit", 0) or 0)
    max_tests = int(apps_cfg.get("max_tests_per_task", 5) or 5)
    difficulties = apps_cfg.get("difficulties")  # e.g. ["introductory"] or null for all
    always = bool(apps_cfg.get("always_refresh", False))
    skip_fn = bool(apps_cfg.get("skip_call_based", True))

    if out_path.is_file() and not always and tasks_file_looks_apps(out_path):
        lg.info("APPS tasks file present; skipping download (%s)", out_path)
        return

    lg.info(
        "Loading APPS from Hugging Face %s revision=%s split=%s (limit=%s)",
        hf_id,
        hf_revision or "default",
        split,
        limit or "all",
    )

    load_kw: dict[str, Any] = {"split": split}
    if hf_revision:
        load_kw["revision"] = str(hf_revision)

    kwargs: dict[str, Any] = {}
    if difficulties:
        kwargs["difficulties"] = difficulties

    try:
        ds = load_dataset(hf_id, **load_kw, **kwargs)
    except TypeError:
        ds = load_dataset(hf_id, **load_kw)

    if difficulties:
        allowed = set(str(x) for x in difficulties)
        ds = ds.filter(lambda ex: ex.get("difficulty") in allowed)

    n_ds = len(ds)
    end = n_ds if not limit or limit <= 0 else min(int(limit), n_ds)
    rows = [dict(ds[i]) for i in range(end)]

    task_rows = hf_rows_to_task_dicts(
        rows,
        max_tests_per_task=max_tests,
        skip_fn_name=skip_fn,
    )
    n = write_tasks_jsonl(task_rows, out_path)
    lg.info("Wrote %d APPS tasks (stdio tests) to %s", n, out_path.resolve())
    if n == 0:
        lg.warning(
            "No APPS tasks exported. Try difficulties=['introductory'], increase limit, or set "
            "apps.skip_call_based: false (call-based support is limited)."
        )
