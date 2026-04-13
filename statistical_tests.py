"""
Module 4 — Statistical Analyzer
Runs all planned hypothesis tests and produces summary tables.

Research Questions addressed:
  RQ1: Does overconfidence exist systematically?
       → Chi-square / Fisher's exact test: assertiveness × correctness independence
  RQ2: Which prompting strategy reduces overconfidence most?
       → Chi-square test of homogeneity across C0-C3; pairwise with Bonferroni
  RQ3: Does execution feedback improve correctness AND calibration?
       → McNemar test: before vs after correctness; ΔOGS = OGS_round1 − OGS_final
  RQ4: Does task complexity moderate the strategy effect?
       → Cochran-Mantel-Haenszel test stratified by complexity

All tests use α = 0.05. Fisher's exact test used when any expected cell count < 5.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class TestReport:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    effect_size_label: str | None = None   # "Cramer's V", "Cohen's d", etc.
    notes: str = ""
    raw_table: list[list] | None = None    # Contingency table used


@dataclass
class AnalysisReport:
    rq1: list[TestReport] = field(default_factory=list)
    rq2: list[TestReport] = field(default_factory=list)
    rq3: list[TestReport] = field(default_factory=list)
    rq4: list[TestReport] = field(default_factory=list)
    summary_table: dict[str, Any] = field(default_factory=dict)


class StatisticalAnalyzer:
    """
    Runs all statistical analyses and returns structured reports.
    """

    def __init__(self, config: dict):
        self.alpha = config["analysis"]["significance_level"]

    def run_full_analysis(
        self,
        baseline_results: list[dict],
        strategy_results: dict[str, list],
    ) -> AnalysisReport:
        """
        Run all four RQ analyses and assemble AnalysisReport.

        Args:
            baseline_results: Labeled sample list from OGSCalculator.label_samples()
                              for condition C0.
            strategy_results: Dict mapping condition → labeled sample list (C1, C2, C3).
        """
        # TODO: Call rq1_overconfidence_exists()
        # TODO: Call rq2_strategy_comparison()
        # TODO: Call rq3_feedback_calibration()
        # TODO: Call rq4_complexity_moderation()
        # TODO: Build summary_table
        raise NotImplementedError

    def rq1_overconfidence_exists(self, labeled_samples: list[dict]) -> list[TestReport]:
        """
        RQ1: Is the association between assertiveness and incorrectness significant?

        Contingency table: assertiveness_level (1|2|3) × is_correct (T|F)
        Test: chi_square_test() or fisher_exact_test() if any expected cell < 5
        Effect size: Cramer's V

        Additional: compute proportion of overconfident samples per model.
        """
        # TODO: Build 3×2 contingency table
        # TODO: Check expected cell counts → select chi-square or Fisher
        # TODO: Compute Cramer's V = sqrt(χ² / (n × (k−1)))
        raise NotImplementedError

    def rq2_strategy_comparison(
        self, all_labeled: dict[str, list[dict]]
    ) -> list[TestReport]:
        """
        RQ2: Which strategy most reduces overconfidence and improves correctness?

        Tests:
          - Chi-square of homogeneity: OGS across C0-C3 (4×2 table)
          - Pairwise chi-square: (C0 vs C1), (C0 vs C2), (C0 vs C3)
          - Bonferroni correction: adjusted α = 0.05 / 3 = 0.0167
          - Same tests for functional correctness (pass rate ≥ threshold)
        """
        # TODO: Aggregate OGS per condition
        # TODO: Overall homogeneity test
        # TODO: Pairwise tests with Bonferroni correction
        raise NotImplementedError

    def rq3_feedback_calibration(
        self, strategy_results: dict[str, list]
    ) -> list[TestReport]:
        """
        RQ3: Does execution feedback improve both correctness and linguistic calibration?

        For C2 and C3:
          - McNemar test: correctness at round 1 vs final round (paired)
          - ΔOGS per task: OGS_round1 − OGS_final (positive = improvement)
          - Track assertiveness level changes across rounds
        """
        # TODO: Extract per-task round-level records for C2 and C3
        # TODO: Build paired correctness tables for McNemar
        # TODO: Compute ΔOGS per condition
        raise NotImplementedError

    def rq4_complexity_moderation(
        self, all_labeled: dict[str, list[dict]]
    ) -> list[TestReport]:
        """
        RQ4: Does task complexity moderate strategy effectiveness?

        Test: Cochran-Mantel-Haenszel (CMH) controlling for complexity strata.
          - Strata: basic / medium / complex
          - Exposure: condition (C0 vs C1/C2/C3)
          - Outcome: is_overconfident or is_correct
        """
        # TODO: Group samples by complexity
        # TODO: Build per-stratum 2×2 tables
        # TODO: CMH test: sum(A_k - E_k)² / sum(Var_k)
        raise NotImplementedError

    # ─── Core Test Methods ─────────────────────────────────────────────────────

    def chi_square_test(self, table: list[list[int]]) -> tuple[float, float]:
        """scipy.stats.chi2_contingency wrapper. Returns (statistic, p_value)."""
        # TODO: np.array(table); stats.chi2_contingency(arr)
        raise NotImplementedError

    def fisher_exact_test(self, table: list[list[int]]) -> tuple[float, float]:
        """scipy.stats.fisher_exact wrapper for 2×2 tables. Returns (statistic, p_value)."""
        # TODO: stats.fisher_exact(table)
        raise NotImplementedError

    def mcnemar_test(self, b: int, c: int) -> tuple[float, float]:
        """
        McNemar test for paired before/after proportions.
        b = correct after, incorrect before
        c = incorrect after, correct before
        """
        # TODO: statsmodels.stats.contingency_tables.mcnemar or manual
        raise NotImplementedError

    def bonferroni_correct(self, p_values: list[float], n_tests: int) -> list[float]:
        """Apply Bonferroni correction. Returns adjusted p-values."""
        # TODO: [min(p * n_tests, 1.0) for p in p_values]
        raise NotImplementedError

    def cramers_v(self, chi2: float, n: int, k: int) -> float:
        """Cramer's V effect size for chi-square tests."""
        # TODO: sqrt(chi2 / (n * (k - 1)))
        raise NotImplementedError

    def export_tables(self, report: AnalysisReport, output_dir: str = "results/tables") -> None:
        """Export all test reports as CSV/LaTeX tables."""
        # TODO: Write summary_table to CSV
        # TODO: Optionally produce LaTeX-formatted tables for the report
        raise NotImplementedError
