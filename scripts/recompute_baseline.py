"""Recompute baseline test results from saved code (no LLM calls).

Use when test suites change (e.g., adding adversarial cases) and you want to
re-evaluate the already-saved baseline code on the new suites.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module1_data.task_manager import TaskManager
from src.module1_data.test_suite import TestSuiteBuilder
from src.module2_detection.execution_runner import (
    _classify_error,
    _func_name,
    _invoke_in_process,
    _outputs_equal,
)
from src.utils.config import load_config
from src.utils.pipeline_io import save_baseline_results


def main() -> None:
    cfg = load_config("configs/experiment.yaml")
    timeout = int(cfg.get("execution", {}).get("case_timeout_sec", 15))

    tasks = TaskManager(cfg).load_tasks()
    suites = TestSuiteBuilder(cfg).build_all(tasks)

    in_path = "data/processed/baseline_results.jsonl"
    rows = [
        json.loads(line)
        for line in open(in_path, encoding="utf-8")  # noqa: PTH123
        if line.strip()
    ]
    by_task = {r["task_id"]: r for r in rows}

    new_rows: list[dict] = []
    for task in tasks:
        r = by_task[task.task_id]
        code = r["code"]
        fn = _func_name(task.function_signature)
        results = []
        passed_n = 0
        for case in suites[task.task_id].cases:
            ok, out, err, runtime_ms = _invoke_in_process(code, fn, case.input, timeout)
            passed = ok and _outputs_equal(out, case.expected_output)
            if passed:
                passed_n += 1
            results.append(
                {
                    "test_id": case.test_id,
                    "passed": passed,
                    "kind": case.kind,
                    "expected_output": case.expected_output,
                    "actual_output": out if ok else None,
                    "error": "" if passed else (err or f"got {out!r}"),
                    "runtime_ms": runtime_ms,
                    "error_type": ""
                    if passed
                    else (_classify_error(err) if err else "logical_bug"),
                }
            )
        r["test_results"] = results
        r["overall_pass_rate"] = passed_n / len(results) if results else 0.0
        new_rows.append(r)

    save_baseline_results(new_rows)
    print(f"Recomputed baseline for {len(new_rows)} tasks -> {in_path}")


if __name__ == "__main__":
    main()

