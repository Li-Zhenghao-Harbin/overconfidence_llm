"""Module 1 — Test suites built from task examples."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.module1_data.task_manager import Task

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    test_id: str
    task_id: str
    input: Any
    expected_output: Any
    kind: str = "standard"  # standard | adversarial


@dataclass
class TestSuite:
    task_id: str
    cases: list[TestCase]


class TestSuiteBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.suite_file = Path("data/raw/test_suites.jsonl")

    def build_all(self, tasks: list[Task]) -> dict[str, TestSuite]:
        self.suite_file.parent.mkdir(parents=True, exist_ok=True)
        suites: dict[str, TestSuite] = {}
        lines: list[str] = []
        for t in tasks:
            cases: list[TestCase] = []
            for i, ex in enumerate(t.examples):
                tc = TestCase(
                    test_id=f"{t.task_id}_ex{i}",
                    task_id=t.task_id,
                    input=ex["input"],
                    expected_output=ex["expected_output"],
                    kind="standard",
                )
                cases.append(tc)
                lines.append(
                    json.dumps(
                        {
                            "test_id": tc.test_id,
                            "task_id": tc.task_id,
                            "input": tc.input,
                            "expected_output": tc.expected_output,
                            "kind": tc.kind,
                        },
                        ensure_ascii=False,
                    )
                )
            suites[t.task_id] = TestSuite(t.task_id, cases)
        with open(self.suite_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        logger.info("Wrote %d test suite rows to %s", len(lines), self.suite_file)
        return suites
