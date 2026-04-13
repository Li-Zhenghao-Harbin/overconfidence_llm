"""Module 3 — Mitigation strategies (stub: returns empty results until prompts are wired)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.module1_data.test_suite import TestSuite
from src.module2_detection.execution_runner import ExecutionRecord
from src.module2_detection.linguistic_annotator import AnnotationRecord

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    condition: str
    model: str
    task_id: str
    rounds: list[ExecutionRecord] = field(default_factory=list)
    annotations: list[AnnotationRecord] = field(default_factory=list)

    @property
    def final_pass_rate(self) -> float:
        return self.rounds[-1].overall_pass_rate if self.rounds else 0.0

    @property
    def reached_correct(self) -> bool:
        return any(r.overall_pass_rate == 1.0 for r in self.rounds)

    @property
    def repair_efficiency(self) -> int:
        for i, r in enumerate(self.rounds):
            if r.overall_pass_rate == 1.0:
                return i + 1
        return -1


class StrategyRunner:
    def __init__(self, config: dict):
        self.config = config
        self.max_rounds = max(
            self.config["strategies"]["C2"].get("max_rounds", 3),
            self.config["strategies"]["C3"].get("max_rounds", 3),
        )

    def run_all_strategies(
        self, tasks: list, test_suites: dict[str, TestSuite] | None = None
    ) -> dict[str, list[StrategyResult]]:
        logger.warning(
            "Mitigation strategies C1–C3 are not implemented yet; returning empty results."
        )
        return {"C1": [], "C2": [], "C3": []}
