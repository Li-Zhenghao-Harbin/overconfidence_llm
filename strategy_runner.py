"""
Module 3 — Strategy Runner
Runs mitigation strategies C1, C2, C3 and tracks per-round OGS + correctness.

Strategies:
  C1 — Self-Verification Prompting
       Single round: model reviews its own output and flags uncertainties.
       Hypothesis: Reduces linguistic overconfidence, limited correctness gain.

  C2 — Execution-Feedback Refinement
       Multi-round (up to max_rounds): failed test cases + error tracebacks fed back.
       Hypothesis: Improves correctness but may not calibrate linguistic confidence.

  C3 — In-Execution Debugging
       Multi-round: provides intermediate runtime states (variable values at key
       checkpoints) via instrumented execution, enabling targeted revision.
       Hypothesis: Best correctness + potential linguistic calibration.

Each strategy reuses ExecutionRunner and LinguisticAnnotator from Module 2.
All results are annotated with (condition, round_number) for longitudinal analysis.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from ..module2_detection.execution_runner import ExecutionRecord, ExecutionRunner
from ..module2_detection.linguistic_annotator import AnnotationRecord, LinguisticAnnotator
from ..module2_detection.ogs_calculator import OGSCalculator
from ..module1_data.test_suite import TestSuite

logger = logging.getLogger(__name__)


@dataclass
class StrategyResult:
    condition: str                              # C1 | C2 | C3
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
        """Number of rounds until first fully correct solution. -1 if never reached."""
        for i, r in enumerate(self.rounds):
            if r.overall_pass_rate == 1.0:
                return i + 1
        return -1


class StrategyRunner:
    """
    Coordinates execution of C1, C2, C3 strategies across all tasks × models.

    Responsibilities:
    - Run each strategy for every (task × model) pair
    - Track OGS and correctness at each round (for C2/C3)
    - Detect when a solution is fully correct and stop early
    - Persist per-round results to data/processed/strategy_results.jsonl
    """

    def __init__(self, config: dict):
        self.config = config
        self.runner = ExecutionRunner(config)
        self.annotator = LinguisticAnnotator(config)
        self.ogs_calc = OGSCalculator(config)
        self.max_rounds = max(
            config["strategies"]["C2"].get("max_rounds", 3),
            config["strategies"]["C3"].get("max_rounds", 3),
        )

    def run_all_strategies(
        self, tasks: list, test_suites: dict[str, TestSuite] | None = None
    ) -> dict[str, list[StrategyResult]]:
        """
        Execute C1, C2, C3 for all tasks × models.

        Returns:
            Dict mapping condition → list[StrategyResult]
        """
        # TODO: For each condition in [C1, C2, C3], call run_strategy()
        # TODO: Aggregate results per condition
        raise NotImplementedError

    def run_strategy(
        self,
        condition: str,
        tasks: list,
        test_suites: dict[str, TestSuite] | None = None,
    ) -> list[StrategyResult]:
        """
        Run a single strategy for all tasks × models.

        Dispatches to:
          _run_c1() for self-verification
          _run_c2() for execution-feedback refinement
          _run_c3() for in-execution debugging
        """
        # TODO: Dispatch based on condition string
        raise NotImplementedError

    def _run_c1(self, task, model_config: dict) -> StrategyResult:
        """
        C1 — Self-Verification Prompting.

        Prompt addition (from prompts/c1_self_verification.txt):
          "Before finalizing, review your code for potential edge cases,
           flag any uncertainties, and suggest test scenarios."

        Single round only. Annotate the explanation and compute OGS.
        """
        # TODO: Build prompt with C1 instruction
        # TODO: Call runner._query_and_execute() with extra_prompt from template
        # TODO: Annotate response
        # TODO: Return StrategyResult with one round
        raise NotImplementedError

    def _run_c2(
        self, task, model_config: dict, test_suite: TestSuite | None = None
    ) -> StrategyResult:
        """
        C2 — Execution-Feedback Refinement.

        Loop (up to max_rounds):
          1. Query model (or use prior response on round > 0)
          2. Execute code against test suite
          3. If all pass → stop early
          4. Build feedback: failed test name, input, expected vs actual, error traceback
          5. Include feedback in next-round prompt

        Feedback format (from prompts/c2_execution_feedback.txt):
          "The following test cases failed: [list]
           Please fix the code. Explain your changes and flag remaining uncertainties."
        """
        # TODO: Iterative loop with prior_feedback accumulation
        # TODO: Track StrategyResult.rounds for each iteration
        # TODO: Annotate explanation at each round
        raise NotImplementedError

    def _run_c3(
        self, task, model_config: dict, test_suite: TestSuite | None = None
    ) -> StrategyResult:
        """
        C3 — In-Execution Debugging.

        Like C2, but feedback includes intermediate runtime states:
          - Variable values at key checkpoints (via instrumented execution / settrace)
          - Execution trace: which branches were taken

        Instrumentation approach:
          1. Inject sys.settrace() hooks into the test harness
          2. Capture variable snapshots at function entry, each loop iteration, return
          3. Serialize snapshots to JSON and include in next-round prompt

        Feedback format (from prompts/c3_in_execution.txt):
          "Runtime trace for failed test case [id]:
           - Entry state: {args}
           - Line 5: x = 3, y = None
           - Line 8: loop_var = 'foo'
           - Return: None (expected: [1,2,3])
           Please diagnose and fix."
        """
        # TODO: Instrument execution with sys.settrace()
        # TODO: Serialize runtime states
        # TODO: Build richer feedback prompt
        # TODO: Iterative loop identical to C2 but with extended feedback
        raise NotImplementedError

    def _build_execution_feedback(
        self, record: ExecutionRecord, include_trace: bool = False
    ) -> str:
        """
        Format failed test results into a feedback string for the next prompt round.

        Args:
            include_trace: If True, include intermediate variable states (C3 only).
        """
        # TODO: Filter record.test_results to failed cases
        # TODO: Format: test_id, input, expected, actual, error
        # TODO: Optionally append runtime trace
        raise NotImplementedError

    def _instrument_execution(self, code: str, test_case) -> dict:
        """
        Run code with sys.settrace() instrumentation to capture intermediate states.

        Returns:
            Dict with keys: variable_snapshots (list), branch_trace (list), final_output, error
        """
        # TODO: sys.settrace with custom Tracer class
        # TODO: Capture locals() at each line event
        # TODO: Return serialized trace dict
        raise NotImplementedError
