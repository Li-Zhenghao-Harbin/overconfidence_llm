"""Persist Phase 2/3 outputs so `main.py --phase 4` can run in a new process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BASELINE_JSONL = Path("data/processed/baseline_results.jsonl")
STRATEGY_JSONL = Path("data/processed/strategy_results.jsonl")
STRATEGY_EXECUTION_JSONL = Path("data/processed/strategy_execution_records.jsonl")

# Back-compat (older runs / debugging)
BASELINE_INTERMEDIATE_JSONL = Path("data/intermediate/phase2_baseline.jsonl")
STRATEGY_INTERMEDIATE_JSON = Path("data/intermediate/phase3_strategy_results.json")


def _serialize_test_result(t: Any) -> dict[str, Any]:
    if isinstance(t, dict):
        return t
    # dataclass or simple object
    d = dict(getattr(t, "__dict__", {}) or {})
    if d:
        d.setdefault("test_id", getattr(t, "test_id", ""))
        d.setdefault("passed", bool(getattr(t, "passed", False)))
        return d
    return {"test_id": getattr(t, "test_id", ""), "passed": bool(getattr(t, "passed", False))}


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
    # Also write the intermediate file for older tooling / quick inspection
    BASELINE_INTERMEDIATE_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_INTERMEDIATE_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(_baseline_row_to_jsonable(row), ensure_ascii=False, default=str)
                + "\n"
            )


def load_baseline_results() -> list[dict[str, Any]] | None:
    if not BASELINE_JSONL.is_file():
        if not BASELINE_INTERMEDIATE_JSONL.is_file():
            return None
        path = BASELINE_INTERMEDIATE_JSONL
    else:
        path = BASELINE_JSONL
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows if rows else None


def save_strategy_results(data: dict[str, list[Any]]) -> None:
    from dataclasses import asdict, is_dataclass

    STRATEGY_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # Save StrategyResult objects (one row per task × condition).
    rows: list[dict[str, Any]] = []
    for cond, items in (data or {}).items():
        for item in items or []:
            rows.append(asdict(item) if is_dataclass(item) else dict(item))
            rows[-1].setdefault("condition", cond)
    with open(STRATEGY_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    # Also export a flat execution-record view (one row per round) for Phase 4.
    STRATEGY_EXECUTION_JSONL.parent.mkdir(parents=True, exist_ok=True)
    flat: list[dict[str, Any]] = []
    for r in rows:
        cond = r.get("condition", "unknown")
        model = r.get("model")
        task_id = r.get("task_id")
        rounds = r.get("rounds") or []
        anns = r.get("annotations") or []
        ann_by_sid = {a.get("sample_id"): a for a in anns if isinstance(a, dict)}
        for i, rec in enumerate(rounds, start=1):
            if not isinstance(rec, dict):
                continue
            sid = rec.get("sample_id")
            a = ann_by_sid.get(sid) or {}
            flat.append(
                {
                    **rec,
                    "condition": cond,
                    "model": model,
                    "task_id": task_id,
                    "round_number": int(a.get("round_number") or i),
                    "assertiveness_level": a.get("assertiveness_level"),
                    "annotation_note": a.get("annotation_note", ""),
                    "annotator_id": a.get("annotator_id", "auto_regex"),
                }
            )
    with open(STRATEGY_EXECUTION_JSONL, "w", encoding="utf-8") as f:
        for row in flat:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    # Also keep the intermediate JSON blob for inspection
    STRATEGY_INTERMEDIATE_JSON.parent.mkdir(parents=True, exist_ok=True)
    serializable: dict[str, Any] = {}
    for k, v in data.items():
        serializable[k] = [asdict(item) if is_dataclass(item) else item for item in v]
    with open(STRATEGY_INTERMEDIATE_JSON, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2, default=str)


def load_strategy_results() -> dict[str, list[Any]] | None:
    # Prefer processed JSONL if available; fall back to intermediate JSON.
    if STRATEGY_JSONL.is_file():
        by_cond: dict[str, list[Any]] = {}
        with open(STRATEGY_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                cond = row.get("condition", "unknown")
                by_cond.setdefault(cond, []).append(row)
        return by_cond
    if STRATEGY_INTERMEDIATE_JSON.is_file():
        with open(STRATEGY_INTERMEDIATE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return None
