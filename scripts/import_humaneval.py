#!/usr/bin/env python3
"""CLI wrapper: import HumanEval → tasks JSONL (same logic as `tasks.dataset: humaneval`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module1_data.humaneval_dataset import (  # noqa: E402
    DEFAULT_HUMANEVAL_URL,
    download_humaneval_rows,
    rows_to_task_dicts,
    write_tasks_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Import HumanEval → tasks JSONL")
    parser.add_argument("--url", default=DEFAULT_HUMANEVAL_URL, help="HumanEval.jsonl.gz URL")
    parser.add_argument("--out", default="data/raw/tasks.jsonl", help="Output path")
    parser.add_argument("--limit", type=int, default=0, help="If >0, import only first N problems")
    args = parser.parse_args()

    rows = download_humaneval_rows(args.url)
    task_rows = rows_to_task_dicts(rows, limit=args.limit or 0)
    n = write_tasks_jsonl(task_rows, Path(args.out))
    print(f"Wrote {n} tasks to {Path(args.out).resolve()}")
    print("Next: python main.py --phase 1")


if __name__ == "__main__":
    main()
