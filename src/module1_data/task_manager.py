"""Module 1 — Task Manager (loads JSONL or built-in default tasks)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    complexity: str
    domain: str
    title: str
    description: str
    function_signature: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskManager:
    COMPLEXITY_LEVELS = ("basic", "medium", "complex")

    def __init__(self, config: dict):
        self.config = config
        self.task_file = Path(config["tasks"]["task_file"])
        self.tasks_per_level = int(config["tasks"]["tasks_per_level"])
        self._tasks: list[Task] = []

    def load_tasks(self) -> list[Task]:
        self.task_file.parent.mkdir(parents=True, exist_ok=True)
        if self.task_file.exists():
            self._tasks = []
            with open(self.task_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    self._tasks.append(self._dict_to_task(d))
            logger.info("Loaded %d tasks from %s", len(self._tasks), self.task_file)
        else:
            self._tasks = self._get_default_tasks()
            self.save_tasks()
            logger.info("Wrote default tasks to %s", self.task_file)
        return sorted(
            self._tasks,
            key=lambda t: self.COMPLEXITY_LEVELS.index(t.complexity)
            if t.complexity in self.COMPLEXITY_LEVELS
            else 99,
        )

    def _dict_to_task(self, d: dict) -> Task:
        known = {
            "task_id",
            "complexity",
            "domain",
            "title",
            "description",
            "function_signature",
            "examples",
        }
        meta = {k: v for k, v in d.items() if k not in known}
        return Task(
            task_id=d["task_id"],
            complexity=d["complexity"],
            domain=d.get("domain", "general"),
            title=d["title"],
            description=d["description"],
            function_signature=d["function_signature"],
            examples=d.get("examples", []),
            metadata=meta,
        )

    def save_tasks(self, path: Path | None = None) -> None:
        target = path or self.task_file
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for t in self._tasks:
                row = {
                    "task_id": t.task_id,
                    "complexity": t.complexity,
                    "domain": t.domain,
                    "title": t.title,
                    "description": t.description,
                    "function_signature": t.function_signature,
                    "examples": t.examples,
                    **t.metadata,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _get_default_tasks(self) -> list[Task]:
        """Nine tasks (3 per complexity) — minimal specs for the pipeline."""
        return [
            Task(
                "basic_001",
                "basic",
                "array",
                "Sort a list",
                "Implement a function that sorts a list of integers ascending.",
                "def sort_list(lst: list) -> list:",
                [{"input": [3, 1, 2], "expected_output": [1, 2, 3]}],
            ),
            Task(
                "basic_002",
                "basic",
                "string",
                "Reverse a string",
                "Return the reverse of the input string.",
                "def reverse_string(s: str) -> str:",
                [{"input": "abc", "expected_output": "cba"}],
            ),
            Task(
                "basic_003",
                "basic",
                "string",
                "Palindrome check",
                "Return True if the string reads the same forwards and backwards (ignore case and non-alphanumeric).",
                "def is_palindrome(s: str) -> bool:",
                [{"input": "Racecar", "expected_output": True}],
            ),
            Task(
                "medium_001",
                "medium",
                "math",
                "Currency formatter",
                "Format amount as USD string with two decimals, e.g. 12.5 -> '$12.50'.",
                "def format_usd(amount: float) -> str:",
                [{"input": 12.5, "expected_output": "$12.50"}],
            ),
            Task(
                "medium_002",
                "medium",
                "validation",
                "Email simple check",
                "Return True if s contains exactly one '@' and a '.' after '@'.",
                "def is_valid_email(s: str) -> bool:",
                [{"input": "a@b.com", "expected_output": True}],
            ),
            Task(
                "medium_003",
                "medium",
                "data",
                "Parse CSV line",
                "Given a comma-separated line, return list of stripped fields.",
                "def parse_csv_line(line: str) -> list:",
                [{"input": "a, b, c", "expected_output": ["a", "b", "c"]}],
            ),
            Task(
                "complex_001",
                "complex",
                "array",
                "Merge sorted lists",
                "Merge two sorted lists of ints into one sorted list (allow duplicates).",
                "def merge_sorted(a: list, b: list) -> list:",
                [
                    {
                        "input": {"a": [1, 3], "b": [2, 4]},
                        "expected_output": [1, 2, 3, 4],
                    }
                ],
            ),
            Task(
                "complex_002",
                "complex",
                "scheduling",
                "Meeting conflict",
                "Given list of (start,end) int tuples, return True if any pair overlaps (touching at endpoint is not overlap).",
                "def has_overlap(intervals: list) -> bool:",
                [{"input": [[0, 2], [2, 4]], "expected_output": False}],
            ),
            Task(
                "complex_003",
                "complex",
                "data",
                "One-level flatten",
                "Flatten a list one level: [[1,2], 3, [4]] -> [1, 2, 3, 4].",
                "def flatten_once(nested: list) -> list:",
                [{"input": [[1, 2], 3, [4]], "expected_output": [1, 2, 3, 4]}],
            ),
        ]
