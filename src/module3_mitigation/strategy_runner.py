"""Module 3 — Mitigation strategies (C1/C2/C3).

Implements:
- C1 self-verification prompting (single round)
- C2 execution-feedback refinement (multi-round)
- C3 in-execution debugging with lightweight runtime trace (multi-round)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from src.module1_data.task_manager import Task
from src.module1_data.test_suite import TestSuite, TestSuiteBuilder
from src.module2_detection.execution_runner import (
    ExecutionRecord,
    TestResult,
    _LLMClient,
    _classify_error,
    _func_name,
    _invoke_in_process,
    _invoke_in_process_trace,
    _outputs_equal,
)
from src.module2_detection.severity_dl import SeverityPredictor, annotate_test_results_list
from src.module2_detection.linguistic_annotator import (
    AnnotationRecord,
    LinguisticAnnotator,
)

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
        self._llm = _LLMClient(config)
        self._annotator = LinguisticAnnotator(config)
        self._severity = SeverityPredictor(config)
        self.timeout = int(config.get("execution", {}).get("case_timeout_sec", 15))
        self.max_rounds = int(
            max(
                self.config.get("strategies", {}).get("C2", {}).get("max_rounds", 3),
                self.config.get("strategies", {}).get("C3", {}).get("max_rounds", 3),
            )
        )

    def run_all_strategies(
        self, tasks: list[Task], test_suites: dict[str, TestSuite] | None = None
    ) -> dict[str, list[StrategyResult]]:
        builder = TestSuiteBuilder(self.config)
        suites: dict[str, TestSuite] = test_suites or builder.build_all(tasks)

        models = self.config.get("models", {}).get("baseline") or [
            {"name": "primary", "model": self.config["llm"]["default_model"]}
        ]

        out: dict[str, list[StrategyResult]] = {"C1": [], "C2": [], "C3": []}
        for mc in models:
            model_id = mc.get("model") or self.config["llm"]["default_model"]
            label = mc.get("name", model_id)
            for task in tasks:
                logger.info("Strategy run start model=%s task=%s", label, task.task_id)
                out["C1"].append(self._run_c1(task, suites[task.task_id], label, model_id))
                out["C2"].append(self._run_c2(task, suites[task.task_id], label, model_id))
                out["C3"].append(self._run_c3(task, suites[task.task_id], label, model_id))
                logger.info("Strategy run done  model=%s task=%s", label, task.task_id)
        return out

    def _prompt_base(self, task: Task) -> tuple[str, str]:
        system = (
            "You are an expert Python programmer. "
            "Respond with exactly one markdown ```python code block containing the full implementation, "
            "then a short natural-language explanation."
        )
        user = (
            f"Task: {task.title}\n\n{task.description}\n\n"
            f"Required signature (must match exactly):\n{task.function_signature}\n\n"
            "Implement only the function (or class) above. No main guard, no tests."
        )
        return system, user

    def _exec_once(
        self,
        task: Task,
        suite: TestSuite,
        label: str,
        condition: str,
        code: str,
        explanation: str,
        round_number: int,
    ) -> tuple[ExecutionRecord, AnnotationRecord]:
        fn = _func_name(task.function_signature)
        results: list[TestResult] = []
        passed_n = 0
        for case in suite.cases:
            ok, out, err, runtime_ms = _invoke_in_process(code, fn, case.input, self.timeout)
            passed = ok and _outputs_equal(out, case.expected_output)
            if passed:
                passed_n += 1
            results.append(
                TestResult(
                    test_id=case.test_id,
                    passed=passed,
                    kind=case.kind,
                    expected_output=case.expected_output,
                    actual_output=out if ok else None,
                    error="" if passed else (err or f"got {out!r}"),
                    runtime_ms=runtime_ms,
                    error_type=(
                        "" if passed else (_classify_error(err) if err else "logical_bug")
                    ),
                )
            )
        annotate_test_results_list(self.config, results, self._severity)
        rate = passed_n / len(results) if results else 0.0
        sid = f"{label}_{condition}_{task.task_id}_r{round_number}"
        rec = ExecutionRecord(
            sample_id=sid,
            task_id=task.task_id,
            model=label,
            condition=condition,
            code=code,
            explanation=explanation,
            overall_pass_rate=rate,
            test_results=results,
        )
        level, note = self._annotator.auto_annotate(explanation or "")
        ann = AnnotationRecord(
            sample_id=sid,
            task_id=task.task_id,
            model=label,
            condition=condition,
            explanation=explanation,
            assertiveness_level=level,
            annotator_id="auto_regex",
            annotation_note=note,
            round_number=round_number,
        )
        return rec, ann

    def _failed_case_summaries(self, suite: TestSuite, rec: ExecutionRecord, limit: int = 6) -> list[dict[str, Any]]:
        by_id = {c.test_id: c for c in suite.cases}
        failed = []
        for tr in rec.test_results:
            if tr.passed:
                continue
            c = by_id.get(tr.test_id)
            failed.append(
                {
                    "test_id": tr.test_id,
                    "kind": getattr(tr, "kind", ""),
                    "input": c.input if c else None,
                    "expected": getattr(tr, "expected_output", None),
                    "actual": getattr(tr, "actual_output", None),
                    "error": tr.error,
                    "error_type": getattr(tr, "error_type", ""),
                    "severity": getattr(tr, "severity", ""),
                }
            )
            if len(failed) >= limit:
                break
        return failed

    def _run_c1(self, task: Task, suite: TestSuite, label: str, model_id: str) -> StrategyResult:
        system, user = self._prompt_base(task)
        user = (
            user
            + "\n\nBefore finalizing, review your code for edge cases, flag uncertainties, and suggest test scenarios."
        )
        text = self._llm.complete(system, user, model_id)
        code = text.split("```python", 1)[-1].split("```", 1)[0].strip() if "```python" in text else text
        explanation = text if "```" not in text else text.split("```")[-1].strip()
        rec, ann = self._exec_once(task, suite, label, "C1", code, explanation, 1)
        return StrategyResult(condition="C1", model=label, task_id=task.task_id, rounds=[rec], annotations=[ann])

    def _run_c2(self, task: Task, suite: TestSuite, label: str, model_id: str) -> StrategyResult:
        system, base_user = self._prompt_base(task)
        rounds: list[ExecutionRecord] = []
        anns: list[AnnotationRecord] = []

        user = base_user
        code = ""
        explanation = ""
        for r in range(1, self.max_rounds + 1):
            logger.info("C2 round %d model=%s task=%s", r, label, task.task_id)
            text = self._llm.complete(system, user, model_id)
            code = text.split("```python", 1)[-1].split("```", 1)[0].strip() if "```python" in text else text
            explanation = text if "```" not in text else text.split("```")[-1].strip()
            rec, ann = self._exec_once(task, suite, label, "C2", code, explanation, r)
            rounds.append(rec)
            anns.append(ann)
            if rec.overall_pass_rate == 1.0:
                break
            failed = self._failed_case_summaries(suite, rec)
            user = (
                base_user
                + "\n\nThe following test cases failed. Fix the code and explain your changes.\n"
                + json.dumps(failed, ensure_ascii=False, indent=2)
            )
        return StrategyResult(condition="C2", model=label, task_id=task.task_id, rounds=rounds, annotations=anns)

    def _run_c3(self, task: Task, suite: TestSuite, label: str, model_id: str) -> StrategyResult:
        system, base_user = self._prompt_base(task)
        rounds: list[ExecutionRecord] = []
        anns: list[AnnotationRecord] = []

        user = base_user
        for r in range(1, self.max_rounds + 1):
            logger.info("C3 round %d model=%s task=%s", r, label, task.task_id)
            text = self._llm.complete(system, user, model_id)
            code = text.split("```python", 1)[-1].split("```", 1)[0].strip() if "```python" in text else text
            explanation = text if "```" not in text else text.split("```")[-1].strip()
            rec, ann = self._exec_once(task, suite, label, "C3", code, explanation, r)
            rounds.append(rec)
            anns.append(ann)
            if rec.overall_pass_rate == 1.0:
                break

            # Build a trace for the first failing case to provide stronger evidence.
            failed = self._failed_case_summaries(suite, rec, limit=1)
            trace_blob = ""
            if failed:
                # locate the testcase input again
                tc = next((c for c in suite.cases if c.test_id == failed[0]["test_id"]), None)
                if tc is not None:
                    fn = _func_name(task.function_signature)
                    ok_t, out_t, err_t, runtime_ms_t, trace = _invoke_in_process_trace(
                        code, fn, tc.input, self.timeout
                    )
                    trace_blob = json.dumps(
                        {
                            "test_id": tc.test_id,
                            "input": tc.input,
                            "expected": tc.expected_output,
                            "ok": ok_t,
                            "actual": out_t if ok_t else None,
                            "error": err_t,
                            "runtime_ms": runtime_ms_t,
                            "trace": json.loads(trace) if trace else [],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )

            user = (
                base_user
                + "\n\nThe following test cases failed. You also get a lightweight runtime trace for one failing test.\n"
                + "Failed cases:\n"
                + json.dumps(self._failed_case_summaries(suite, rec), ensure_ascii=False, indent=2)
                + ("\n\nRuntime trace evidence:\n" + trace_blob if trace_blob else "")
                + "\n\nPlease diagnose and fix. If you are uncertain, say so explicitly."
            )
        return StrategyResult(condition="C3", model=label, task_id=task.task_id, rounds=rounds, annotations=anns)
