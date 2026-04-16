"""Module 1 — Test suites built from task examples."""

from __future__ import annotations

import json
import logging
import re
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
    kind: str = "standard"  # standard | adversarial | humaneval


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
            meta = t.metadata or {}
            if meta.get("humaneval") or meta.get("humaneval_test"):
                cases = self._build_humaneval(t)
            else:
                cases = self._build_standard(t) + self._build_adversarial(t)
            for tc in cases:
                lines.append(json.dumps(tc.__dict__, ensure_ascii=False))
            suites[t.task_id] = TestSuite(t.task_id, cases)
        with open(self.suite_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        logger.info("Wrote %d test suite rows to %s", len(lines), self.suite_file)
        return suites

    def _build_humaneval(self, task: Task) -> list[TestCase]:
        """One bundled HumanEval test (official `check(candidate)` script)."""
        meta = task.metadata or {}
        test_src = meta.get("humaneval_test") or meta.get("test")
        if not test_src:
            logger.warning("Task %s marked humaneval but missing humaneval_test", task.task_id)
            return []
        entry = meta.get("entry_point")
        if not entry:
            m = re.search(r"def\s+(\w+)\s*\(", task.function_signature or "")
            entry = m.group(1) if m else "solution"
        tc = TestCase(
            test_id=f"{task.task_id}_humaneval0",
            task_id=task.task_id,
            input={
                "runner": "humaneval",
                "prompt": task.description,
                "test": test_src,
                "entry_point": entry,
            },
            expected_output=True,
            kind="humaneval",
        )
        return [tc]

    def _build_standard(self, task: Task) -> list[TestCase]:
        """5–8 standard cases per task."""
        tid = task.task_id
        out: list[TestCase] = []

        def add(i: int, inp: Any, exp: Any) -> None:
            out.append(
                TestCase(
                    test_id=f"{tid}_std{i}",
                    task_id=tid,
                    input=inp,
                    expected_output=exp,
                    kind="standard",
                )
            )

        if tid == "basic_001":  # sort_list
            add(0, [3, 1, 2], [1, 2, 3])
            add(1, [], [])
            add(2, [1], [1])
            add(3, [2, 2, 1], [1, 2, 2])
            add(4, [-1, 3, 0], [-1, 0, 3])
        elif tid == "basic_002":  # reverse_string
            add(0, "abc", "cba")
            add(1, "", "")
            add(2, "a", "a")
            add(3, "ab cd", "dc ba")
            add(4, "你好", "好你")
        elif tid == "basic_003":  # is_palindrome
            add(0, "Racecar", True)
            add(1, "", True)
            add(2, "A man, a plan, a canal: Panama", True)
            add(3, "hello", False)
            add(4, "No 'x' in Nixon", True)
        elif tid == "medium_001":  # format_usd
            add(0, 12.5, "$12.50")
            add(1, 0.0, "$0.00")
            add(2, 12.345, "$12.35")
            add(3, -1.2, "$-1.20")
            add(4, 1000000.1, "$1000000.10")
        elif tid == "medium_002":  # is_valid_email
            add(0, "a@b.com", True)
            add(1, "a@b", False)
            add(2, "a@@b.com", False)
            add(3, "@b.com", True)  # spec per prompt: only checks @ count and dot after @
            add(4, "a@.com", True)
        elif tid == "medium_003":  # parse_csv_line
            add(0, "a, b, c", ["a", "b", "c"])
            add(1, " a ,b,c ", ["a", "b", "c"])
            add(2, "", [""])
            add(3, "a,,c", ["a", "", "c"])
            add(4, "a", ["a"])
        elif tid == "complex_001":  # merge_sorted
            add(0, {"a": [1, 3], "b": [2, 4]}, [1, 2, 3, 4])
            add(1, {"a": [], "b": []}, [])
            add(2, {"a": [1, 2], "b": []}, [1, 2])
            add(3, {"a": [], "b": [0]}, [0])
            add(4, {"a": [1, 1], "b": [1]}, [1, 1, 1])
        elif tid == "complex_002":  # has_overlap
            add(0, [[0, 2], [2, 4]], False)
            add(1, [[0, 2], [1, 3]], True)
            add(2, [], False)
            add(3, [[5, 7]], False)
            add(4, [[1, 10], [2, 3], [11, 12]], True)
        elif tid == "complex_003":  # flatten_once
            add(0, [[1, 2], 3, [4]], [1, 2, 3, 4])
            add(1, [], [])
            add(2, [1, 2, 3], [1, 2, 3])
            add(3, [[1], [2], [3]], [1, 2, 3])
            add(4, [[1, [2, 3]], 4], [1, [2, 3], 4])  # one-level only
        else:
            # Fall back to task-provided examples if present
            for i, ex in enumerate(task.examples[:6]):
                add(i, ex.get("input"), ex.get("expected_output"))

        return out

    def _build_adversarial(self, task: Task) -> list[TestCase]:
        """3–5 adversarial cases per task (boundary / exception / logical traps)."""
        tid = task.task_id
        out: list[TestCase] = []

        def add(i: int, inp: Any, exp: Any) -> None:
            out.append(
                TestCase(
                    test_id=f"{tid}_adv{i}",
                    task_id=tid,
                    input=inp,
                    expected_output=exp,
                    kind="adversarial",
                )
            )

        if tid == "basic_001":
            add(0, [0, -1, -1, 2], [-1, -1, 0, 2])
            add(1, [10, 9, 8, 7, 6, 5], [5, 6, 7, 8, 9, 10])
            add(2, [1, 0, 1, 0, 1], [0, 0, 1, 1, 1])
        elif tid == "basic_002":
            add(0, "🙂🙃", "🙃🙂")
            add(1, "a\u0301", "\u0301a")  # combining mark
            add(2, "line1\nline2", "2enil\n1enil")
        elif tid == "basic_003":
            add(0, "!!!", True)
            add(1, "0P", False)
            add(2, "Was it a car or a cat I saw?", True)
        elif tid == "medium_001":
            add(0, 1e-9, "$0.00")
            add(1, 9999999999.999, "$10000000000.00")
            add(2, -0.004, "$-0.00")
        elif tid == "medium_002":
            add(0, "a@b.c", True)
            add(1, "a@b..com", True)  # by spec, still True
            add(2, "no-at-symbol.com", False)
        elif tid == "medium_003":
            add(0, " a , \"b\" , c ", ["a", "\"b\"", "c"])
            add(1, ",", ["", ""])
            add(2, "a, b, c, ", ["a", "b", "c", ""])
        elif tid == "complex_001":
            add(0, {"a": [-3, -1], "b": [-2, 0]}, [-3, -2, -1, 0])
            add(1, {"a": [1, 4, 4], "b": [2, 4]}, [1, 2, 4, 4, 4])
            add(2, {"a": [], "b": [1, 2, 3]}, [1, 2, 3])
        elif tid == "complex_002":
            add(0, [[1, 2], [2, 2]], False)  # touching / zero-length
            add(1, [[1, 3], [3, 3], [2, 4]], True)
            add(2, [[3, 5], [1, 2], [2, 3]], False)
        elif tid == "complex_003":
            add(0, [[[]], []], [[], []])  # one level only
            add(1, [[1, 2], [3, 4], 5], [1, 2, 3, 4, 5])
            add(2, [[[1]], [[2]]], [[1], [2]])
        else:
            add(0, task.examples[0]["input"] if task.examples else None, task.examples[0]["expected_output"] if task.examples else None)

        return out
