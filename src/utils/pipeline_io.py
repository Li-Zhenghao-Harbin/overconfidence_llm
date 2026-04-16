"""Persist Phase 2/3 outputs so `main.py --phase 4` can run in a new process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_JSONL = Path("data/intermediate/phase2_baseline.jsonl")
STRATEGY_JSON = Path("data/intermediate/phase3_strategy_results.json")


def _serialize_test_result(t: Any) -> dict[str, Any]:
    if isinstance(t, dict):
        return t
    return {
        "test_id": getattr(t, "test_id", ""),
        "passed": bool(getattr(t, "passed", False)),
        "actual_output": getattr(t, "actual_output", None),
        "error": getattr(t, "error", "") or "",
    }


def _baseline_row_to_jsonable(row: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in row.items() if k != "test_results"}
    trs = row.get("test_results") or []
    out["test_results"] = [_serialize_test_result(t) for t in trs]
    return out


def save_baseline_results(rows: list[dict[str, Any]]) -> None:
    BASELINE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(_baseline_row_to_jsonable(row), ensure_ascii=False, default=str)
                + "\n"
            )


def load_baseline_results() -> list[dict[str, Any]] | None:
    if not BASELINE_JSONL.is_file():
        return None
    rows: list[dict[str, Any]] = []
    with open(BASELINE_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows if rows else None


def save_strategy_results(data: dict[str, list[Any]]) -> None:
    from dataclasses import asdict, is_dataclass

    STRATEGY_JSON.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, Any] = {}
    for k, v in data.items():
        serializable[k] = [asdict(item) if is_dataclass(item) else item for item in v]
    with open(STRATEGY_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)


def load_strategy_results() -> dict[str, list[Any]] | None:
    if not STRATEGY_JSON.is_file():
        return None
    with open(STRATEGY_JSON, encoding="utf-8") as f:
        return json.load(f)
