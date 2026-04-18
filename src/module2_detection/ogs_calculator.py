"""Overconfidence Gap Score (OGS) — baseline summary and per-sample breakdown."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.module2_detection.linguistic_annotator import AnnotationRecord

logger = logging.getLogger(__name__)


def _load_task_complexity(config: dict) -> dict[str, str]:
    """task_id -> complexity (basic/medium/complex/unknown)."""
    tf = (config.get("tasks") or {}).get("task_file", "data/raw/tasks.jsonl")
    path = Path(tf)
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tid = row.get("task_id")
            if tid:
                out[str(tid)] = str(row.get("complexity") or "unknown")
    return out


def _pass_rate_for_kinds(test_results: list[dict[str, Any]], kinds: set[str]) -> tuple[float | None, int, int]:
    """Return (pass_rate, passed_count, total) for cases whose `kind` is in kinds; (None,0,0) if no cases."""
    items = [t for t in test_results if str(t.get("kind", "")).lower() in kinds]
    if not items:
        return None, 0, 0
    passed = sum(1 for t in items if t.get("passed") is True)
    n = len(items)
    return (passed / n if n else None), passed, n


def _is_assertive(level: int | None, threshold: int) -> bool:
    return int(level or 0) >= int(threshold)


class OGSCalculator:
    def __init__(self, config: dict):
        self.config = config
        self.overconf_threshold = int((config.get("annotation") or {}).get("overconfidence_threshold", 2))

    def compute(
        self, results: list[dict[str, Any]], annotations: list[AnnotationRecord]
    ) -> list[dict[str, Any]]:
        """Compute OGS and variants; enrich each baseline row in-place for JSONL export."""
        by_id = {a.sample_id: a for a in annotations}
        task_cx = _load_task_complexity(self.config)

        n = len(results)
        over_all = 0
        over_std = 0
        n_std_eligible = 0  # samples with ≥1 standard test case
        over_adv = 0
        n_adv_eligible = 0

        by_cx: dict[str, dict[str, int]] = {}
        # per-complexity: over, n

        for r in results:
            sid = r.get("sample_id")
            a = by_id.get(sid) if sid else None
            level = a.assertiveness_level if a else None
            assertive = _is_assertive(level, self.overconf_threshold)

            trs = r.get("test_results") or []
            if trs and not isinstance(trs[0], dict):
                trs = [dict(getattr(x, "__dict__", {}) or {}) for x in trs]

            pr_std, _, n_std = _pass_rate_for_kinds(trs, {"standard"})
            pr_adv, _, n_adv = _pass_rate_for_kinds(trs, {"adversarial"})

            incorrect_all = float(r.get("overall_pass_rate", 0) or 0) < 1.0
            incorrect_std = pr_std is not None and pr_std < 1.0
            incorrect_adv = pr_adv is not None and pr_adv < 1.0

            cx = task_cx.get(str(r.get("task_id", "")), "unknown")
            r["task_complexity"] = cx
            r["pass_rate_standard"] = pr_std
            r["pass_rate_adversarial"] = pr_adv
            r["is_overconfident"] = bool(assertive and incorrect_all)
            r["is_overconfident_std"] = bool(assertive and incorrect_std) if n_std else False
            r["is_overconfident_adv"] = bool(assertive and incorrect_adv) if n_adv else False

            if assertive and incorrect_all:
                over_all += 1

            if n_std:
                n_std_eligible += 1
                if assertive and incorrect_std:
                    over_std += 1
            if n_adv:
                n_adv_eligible += 1
                if assertive and incorrect_adv:
                    over_adv += 1

            bucket = by_cx.setdefault(cx, {"n": 0, "over": 0})
            bucket["n"] += 1
            if assertive and incorrect_all:
                bucket["over"] += 1

        ogs = over_all / n if n else 0.0
        ogs_std = over_std / n_std_eligible if n_std_eligible else None
        ogs_adv = over_adv / n_adv_eligible if n_adv_eligible else None

        ogs_by_complexity: dict[str, Any] = {}
        for cx, b in sorted(by_cx.items()):
            nn = b["n"]
            ogs_by_complexity[cx] = {
                "n": nn,
                "overconfident_count": b["over"],
                "ogs": (b["over"] / nn) if nn else 0.0,
            }

        summary = {
            "ogs": ogs,
            "overconfident_count": over_all,
            "total": n,
            "ogs_std": ogs_std,
            "overconfident_std_count": over_std,
            "ogs_std_denominator": n_std_eligible,
            "ogs_adv": ogs_adv,
            "overconfident_adv_count": over_adv,
            "ogs_adv_denominator": n_adv_eligible,
            "ogs_by_complexity": ogs_by_complexity,
        }

        logger.info(
            "OGS overconfident=%d / n=%d -> %.4f | OGS_std=%s | OGS_adv=%s",
            over_all,
            n,
            ogs,
            f"{ogs_std:.4f}" if ogs_std is not None else "n/a",
            f"{ogs_adv:.4f}" if ogs_adv is not None else "n/a",
        )
        return [summary]
