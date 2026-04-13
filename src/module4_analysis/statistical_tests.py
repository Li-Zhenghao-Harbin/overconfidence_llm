"""Module 4 — Statistical analysis (minimal stub)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TestReport:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    effect_size_label: str | None = None
    notes: str = ""
    raw_table: list[list] | None = None


@dataclass
class AnalysisReport:
    rq1: list[TestReport] = field(default_factory=list)
    rq2: list[TestReport] = field(default_factory=list)
    rq3: list[TestReport] = field(default_factory=list)
    rq4: list[TestReport] = field(default_factory=list)
    summary_table: dict[str, Any] = field(default_factory=dict)


class StatisticalAnalyzer:
    def __init__(self, config: dict):
        self.alpha = config["analysis"]["significance_level"]

    def run_full_analysis(
        self,
        baseline_results: list[dict],
        strategy_results: dict[str, list],
    ) -> AnalysisReport:
        logger.info(
            "StatisticalAnalyzer stub: skipping RQ tests (strategy results empty or not wired)."
        )
        return AnalysisReport(
            summary_table={"note": "stub", "baseline_n": len(baseline_results)}
        )
