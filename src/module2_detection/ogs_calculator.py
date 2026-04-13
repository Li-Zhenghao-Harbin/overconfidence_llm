"""Overconfidence Gap Score (OGS) — baseline summary."""

from __future__ import annotations

import logging
from typing import Any

from src.module2_detection.linguistic_annotator import AnnotationRecord

logger = logging.getLogger(__name__)


class OGSCalculator:
    def __init__(self, config: dict):
        self.config = config

    def compute(
        self, results: list[dict[str, Any]], annotations: list[AnnotationRecord]
    ) -> list[dict[str, Any]]:
        by_id = {a.sample_id: a for a in annotations}
        over = 0
        for r in results:
            a = by_id.get(r["sample_id"])
            if not a:
                continue
            if a.assertiveness_level >= 2 and float(r.get("overall_pass_rate", 0)) < 1.0:
                over += 1
        n = len(results)
        ogs = over / n if n else 0.0
        logger.info("OGS overconfident=%d / n=%d -> %.4f", over, n, ogs)
        return [{"ogs": ogs, "overconfident_count": over, "total": n}]
