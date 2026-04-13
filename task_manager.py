"""
Module 1 — Task Manager
Handles loading, validating, and organizing programming tasks across 3 complexity levels.

Tasks are stored in data/raw/tasks.jsonl in the format:
{
    "task_id": "basic_001",
    "complexity": "basic",       # basic | medium | complex
    "domain": "array",           # array | string | math | api | state
    "title": "Sort a list",
    "description": "...",
    "function_signature": "def sort_list(lst: list) -> list:",
    "examples": [{"input": [3,1,2], "expected_output": [1,2,3]}]
}
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    complexity: str          # basic | medium | complex
    domain: str
    title: str
    description: str
    function_signature: str
    examples: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TaskManager:
    """
    Loads and manages programming tasks for the experiment.

    Responsibilities:
    - Load tasks from JSONL file or define built-in tasks
    - Validate task schema completeness
    - Stratify tasks by complexity level (basic / medium / complex)
    - Provide task lookup by ID or complexity
    """

    COMPLEXITY_LEVELS = ("basic", "medium", "complex")

    def __init__(self, config: dict):
        self.config = config
        self.task_file = Path(config["tasks"]["task_file"])
        self.tasks_per_level = config["tasks"]["tasks_per_level"]
        self._tasks: list[Task] = []

    def load_tasks(self) -> list[Task]:
        """
        Load tasks from JSONL file. Falls back to built-in default tasks
        if the file does not exist (useful for first-run / testing).

        Returns:
            List of Task objects, sorted by complexity level.
        """
        # TODO: Load from self.task_file if it exists
        # TODO: Validate each task with _validate_task()
        # TODO: Fall back to _get_default_tasks() if file missing
        # TODO: Log task count per complexity level
        raise NotImplementedError

    def get_tasks_by_complexity(self, complexity: str) -> list[Task]:
        """Return tasks filtered by complexity level."""
        # TODO: Filter self._tasks by complexity
        raise NotImplementedError

    def get_task(self, task_id: str) -> Task | None:
        """Look up a single task by ID."""
        # TODO: Linear search or dict lookup
        raise NotImplementedError

    def _validate_task(self, task_data: dict) -> bool:
        """
        Ensure a task dict contains all required fields and valid complexity level.
        Returns True if valid, raises ValueError otherwise.
        """
        # TODO: Check required keys: task_id, complexity, title, description,
        #       function_signature, examples
        # TODO: Validate complexity in COMPLEXITY_LEVELS
        raise NotImplementedError

    def _get_default_tasks(self) -> list[Task]:
        """
        Return 9 hard-coded default tasks (3 per complexity level) to bootstrap
        experiments without requiring a pre-existing data file.

        Basic (3):
          - Sort a list of integers
          - Reverse a string
          - Check if a string is a palindrome

        Medium (3):
          - Currency converter with real-time API call
          - Input validator for user registration form
          - CSV file parser with error handling

        Complex (3):
          - Todo-list manager with file-based state persistence
          - Meeting scheduler with conflict detection
          - Event-driven pub/sub message bus
        """
        # TODO: Return list of Task objects with full descriptions and examples
        raise NotImplementedError

    def save_tasks(self, path: Path | None = None) -> None:
        """Serialize current task list to JSONL for reproducibility."""
        # TODO: Write each task as a JSON line to path or self.task_file
        raise NotImplementedError
