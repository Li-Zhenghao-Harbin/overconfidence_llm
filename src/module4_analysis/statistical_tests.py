"""Module 4 — Statistical analysis & exports.

This module is intentionally pragmatic: data sizes are small (course project),
so we implement robust-but-simple analyses that match `WORKFLOW.md` RQ1–RQ4 and
always export intermediate tables for inspection.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import StratifiedTable

from src.utils import pipeline_io as pio

logger = logging.getLogger(__name__)


@dataclass
class TestReport:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None = None
    effect_size_label: str | None = None
    notes: str = ""
    raw_table: list[list] | None = None


@dataclass
class AnalysisReport:
    rq1: list[TestReport] = field(default_factory=list)
    rq2: list[TestReport] = field(default_factory=list)
    rq3: list[TestReport] = field(default_factory=list)
    rq4: list[TestReport] = field(default_factory=list)
    summary_table: dict[str, Any] = field(default_factory=dict)


class StatisticalAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.alpha = config["analysis"]["significance_level"]
        self.overconf_threshold = int((config.get("annotation") or {}).get("overconfidence_threshold", 2))

    def run_full_analysis(
        self,
        baseline_results: list[dict],
        strategy_results: dict[str, list],
    ) -> AnalysisReport:
        # Load raw dataframes
        df_tasks = _load_tasks(self.config)
        df_c0 = _baseline_to_df(baseline_results, df_tasks, self.overconf_threshold)
        df_strat_final, df_strat_rounds = _strategy_to_df(df_tasks, self.overconf_threshold)

        # Combine final per-condition samples (C0 + C1/C2/C3 final)
        df_final = pd.concat([df_c0, df_strat_final], ignore_index=True)

        out = AnalysisReport()
        out.summary_table = _summary_metrics(df_final, df_strat_rounds)

        out.rq1 = _rq1_overconfidence_exists(df_c0, alpha=self.alpha)
        out.rq2 = _rq2_strategy_comparison(df_final, alpha=self.alpha)
        out.rq3 = _rq3_feedback_calibration(df_strat_rounds, alpha=self.alpha)
        out.rq4 = _rq4_complexity_moderation(df_final, alpha=self.alpha)

        _export_tables(out, df_final, df_strat_rounds, self.config)
        td = (self.config.get("outputs") or {}).get("tables_dir", "results/tables")
        logger.info("Statistical analysis exported to %s/", td)
        return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_tasks(cfg: dict) -> pd.DataFrame:
    tf = (cfg.get("tasks") or {}).get("task_file", "data/raw/tasks.jsonl")
    rows = _load_jsonl(Path(tf))
    if not rows:
        return pd.DataFrame(columns=["task_id", "complexity"])
    df = pd.DataFrame(rows)
    if "complexity" not in df.columns:
        df["complexity"] = "unknown"
    return df[["task_id", "complexity"]]


def _load_baseline_annotations() -> pd.DataFrame:
    rows = _load_jsonl(pio.auto_annotations_path())
    if not rows:
        return pd.DataFrame(columns=["sample_id", "assertiveness_level"])
    df = pd.DataFrame(rows)
    return df[["sample_id", "assertiveness_level"]]


def _baseline_to_df(
    baseline_results: list[dict], df_tasks: pd.DataFrame, overconf_threshold: int
) -> pd.DataFrame:
    df = pd.DataFrame(baseline_results or [])
    if df.empty:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "task_id",
                "model",
                "condition",
                "overall_pass_rate",
                "assertiveness_level",
                "complexity",
            ]
        )
    ann = _load_baseline_annotations()
    df = df.merge(ann, on="sample_id", how="left")
    df = df.merge(df_tasks, on="task_id", how="left")
    df["condition"] = df.get("condition", "C0").fillna("C0")
    df["is_correct"] = (df["overall_pass_rate"].astype(float) == 1.0).astype(int)
    df["is_assertive"] = (
        df["assertiveness_level"].fillna(0).astype(int) >= int(overconf_threshold)
    ).astype(
        int
    )
    df["is_overconfident"] = ((df["is_assertive"] == 1) & (df["is_correct"] == 0)).astype(
        int
    )
    return df


def _strategy_to_df(
    df_tasks: pd.DataFrame, overconf_threshold: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = _load_jsonl(pio.strategy_execution_records_path())
    df = pd.DataFrame(rows)
    if df.empty:
        empty_final = pd.DataFrame(
            columns=[
                "sample_id",
                "task_id",
                "model",
                "condition",
                "overall_pass_rate",
                "assertiveness_level",
                "complexity",
                "round_number",
            ]
        )
        return empty_final, empty_final

    df = df.merge(df_tasks, on="task_id", how="left")
    df["overall_pass_rate"] = df["overall_pass_rate"].astype(float)
    df["round_number"] = df["round_number"].astype(int)
    df["is_correct"] = (df["overall_pass_rate"] == 1.0).astype(int)
    df["is_assertive"] = (
        df["assertiveness_level"].fillna(0).astype(int) >= int(overconf_threshold)
    ).astype(
        int
    )
    df["is_overconfident"] = ((df["is_assertive"] == 1) & (df["is_correct"] == 0)).astype(
        int
    )

    # Final-round view per (condition, model, task)
    df_final = (
        df.sort_values(["condition", "model", "task_id", "round_number"])
        .groupby(["condition", "model", "task_id"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    return df_final, df


def _summary_metrics(df_final: pd.DataFrame, df_rounds: pd.DataFrame) -> dict[str, Any]:
    if df_final.empty:
        return {"note": "no data"}
    by_cond = (
        df_final.groupby("condition")
        .agg(
            n=("sample_id", "count"),
            ogs=("is_overconfident", "mean"),
            pass_rate=("is_correct", "mean"),
            assertive_rate=("is_assertive", "mean"),
        )
        .reset_index()
    )
    # Keep a JSON-serializable structure
    return {"by_condition": by_cond.to_dict(orient="records"), "round_rows": int(len(df_rounds))}


def _cramers_v(table: np.ndarray) -> float:
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return float("nan")
    try:
        chi2, _, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        return float("nan")
    n = table.sum()
    if n == 0:
        return float("nan")
    r, k = table.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else float("nan")


def _chi2_or_fisher(table: np.ndarray) -> tuple[str, float, float, str]:
    """Return (test_name, statistic, p_value, notes)."""
    if table.shape == (2, 2):
        # Degenerate rows/cols can make chi2_contingency crash; use Fisher.
        if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
            odds, p = fisher_exact(table)
            return ("fisher_exact", float(odds), float(p), "degenerate_row_or_col")
        # expected < 5 -> Fisher
        try:
            chi2, p_chi, _, exp = chi2_contingency(table, correction=False)
        except ValueError:
            odds, p = fisher_exact(table)
            return ("fisher_exact", float(odds), float(p), "chi2_expected_zero")
        if (exp < 5).any():
            odds, p = fisher_exact(table)
            return ("fisher_exact", float(odds), float(p), "expected<5")
        return ("chi_square", float(chi2), float(p_chi), "")
    try:
        chi2, p, _, _ = chi2_contingency(table, correction=False)
        return ("chi_square", float(chi2), float(p), "")
    except ValueError as e:
        # e.g. expected frequency has zeros — common when a column is all-zero
        return ("chi_square_skipped", float("nan"), float("nan"), str(e))


def _rq1_overconfidence_exists(df_c0: pd.DataFrame, alpha: float) -> list[TestReport]:
    # assertiveness (>=2) × correctness for baseline
    if df_c0.empty:
        return []
    ct = pd.crosstab(df_c0["is_assertive"], df_c0["is_correct"])
    table = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
    test_name, stat, p, notes = _chi2_or_fisher(table)
    v = _cramers_v(table)
    return [
        TestReport(
            test_name=f"RQ1: assertive×correctness ({test_name})",
            statistic=stat,
            p_value=p,
            significant=p < alpha,
            effect_size=v,
            effect_size_label="Cramer's V",
            notes=notes,
            raw_table=table.tolist(),
        )
    ]


def _rq2_strategy_comparison(df_final: pd.DataFrame, alpha: float) -> list[TestReport]:
    if df_final.empty:
        return []
    # condition × overconfident(yes/no)
    ct = pd.crosstab(df_final["condition"], df_final["is_overconfident"])
    table = ct.reindex(columns=[0, 1], fill_value=0).to_numpy()
    test_name, stat, p, notes = _chi2_or_fisher(table)
    v = _cramers_v(table)
    rep = [
        TestReport(
            test_name=f"RQ2: overconfidence by condition ({test_name})",
            statistic=stat,
            p_value=p,
            significant=p < alpha,
            effect_size=v,
            effect_size_label="Cramer's V",
            notes=notes,
            raw_table=table.tolist(),
        )
    ]

    # Pairwise vs C0 with Bonferroni (3 comparisons)
    pair_alpha = alpha / 3
    for cond in ["C1", "C2", "C3"]:
        sub = df_final[df_final["condition"].isin(["C0", cond])]
        if sub.empty:
            continue
        ct2 = pd.crosstab(sub["condition"], sub["is_overconfident"])
        t2 = ct2.reindex(index=["C0", cond], columns=[0, 1], fill_value=0).to_numpy()
        tn, st, pv, nt = _chi2_or_fisher(t2)
        rep.append(
            TestReport(
                test_name=f"RQ2 pairwise: C0 vs {cond} ({tn})",
                statistic=st,
                p_value=pv,
                significant=pv < pair_alpha,
                effect_size=_cramers_v(t2),
                effect_size_label="Cramer's V",
                notes=f"bonferroni_alpha={pair_alpha:.4f} {nt}".strip(),
                raw_table=t2.tolist(),
            )
        )
    return rep


def _mcnemar_from_pairs(before: pd.Series, after: pd.Series) -> tuple[float, float, list[list[int]]]:
    """Exact McNemar via binomial sign test on discordant pairs."""
    # discordant counts
    b = int(((before == 1) & (after == 0)).sum())
    c = int(((before == 0) & (after == 1)).sum())
    n = b + c
    if n == 0:
        return 0.0, 1.0, [[0, b], [c, 0]]
    # two-sided exact binomial: 2*min(P(X<=min(b,c)), P(X>=max(b,c))) where X~Bin(n,0.5)
    from scipy.stats import binom

    k = min(b, c)
    p = float(2 * binom.cdf(k, n, 0.5))
    p = min(p, 1.0)
    # chi-square approx stat for reference
    stat = float(((abs(b - c) - 1) ** 2) / n) if n > 0 else 0.0
    return stat, p, [[0, b], [c, 0]]


def _rq3_feedback_calibration(df_rounds: pd.DataFrame, alpha: float) -> list[TestReport]:
    if df_rounds.empty:
        return []
    reps: list[TestReport] = []
    for cond in ["C2", "C3"]:
        sub = df_rounds[df_rounds["condition"] == cond].copy()
        if sub.empty:
            continue
        # paired by (model, task)
        g = sub.sort_values(["model", "task_id", "round_number"]).groupby(
            ["model", "task_id"], as_index=False
        )
        first = g.head(1).set_index(["model", "task_id"])
        last = g.tail(1).set_index(["model", "task_id"])
        idx = first.index.intersection(last.index)
        before = first.loc[idx, "is_correct"]
        after = last.loc[idx, "is_correct"]
        stat, p, raw = _mcnemar_from_pairs(before, after)
        reps.append(
            TestReport(
                test_name=f"RQ3: correctness improves in {cond} (McNemar exact)",
                statistic=stat,
                p_value=p,
                significant=p < alpha,
                effect_size=None,
                effect_size_label=None,
                notes="paired by (model,task); discordant table [[-,b],[c,-]]",
                raw_table=raw,
            )
        )

        # OGS per round (for plotting/export)
        ogs_by_round = (
            sub.groupby("round_number")["is_overconfident"].mean().reset_index()
        )
        reps.append(
            TestReport(
                test_name=f"RQ3: OGS by round {cond}",
                statistic=float("nan"),
                p_value=float("nan"),
                significant=False,
                notes="descriptive (exported separately)",
                raw_table=ogs_by_round.values.tolist(),
            )
        )
    return reps


def _rq4_complexity_moderation(df_final: pd.DataFrame, alpha: float) -> list[TestReport]:
    if df_final.empty:
        return []
    reps: list[TestReport] = []
    # CMH for each strategy vs baseline, stratified by complexity
    for cond in ["C1", "C2", "C3"]:
        sub = df_final[df_final["condition"].isin(["C0", cond])].copy()
        if sub.empty:
            continue
        strata = []
        for comp, df_s in sub.groupby("complexity"):
            # 2x2 table: rows condition (C0, cond) × cols overconfident (0,1)
            ct = pd.crosstab(df_s["condition"], df_s["is_overconfident"]).reindex(
                index=["C0", cond], columns=[0, 1], fill_value=0
            )
            t = ct.to_numpy()
            if t.sum() == 0:
                continue
            strata.append(t)
        if not strata:
            continue
        try:
            st = StratifiedTable(strata)
            chi2 = float(st.test_null_odds().statistic)
            p = float(st.test_null_odds().pvalue)
            reps.append(
                TestReport(
                    test_name=f"RQ4: CMH overconfidence C0 vs {cond}",
                    statistic=chi2,
                    p_value=p,
                    significant=p < alpha,
                    effect_size=None,
                    effect_size_label=None,
                    notes="stratified by complexity",
                    raw_table=[s.tolist() for s in strata],
                )
            )
        except Exception as e:  # noqa: BLE001
            reps.append(
                TestReport(
                    test_name=f"RQ4: CMH overconfidence C0 vs {cond} (failed)",
                    statistic=float("nan"),
                    p_value=float("nan"),
                    significant=False,
                    notes=str(e),
                )
            )
    return reps


def _export_tables(
    report: AnalysisReport,
    df_final: pd.DataFrame,
    df_rounds: pd.DataFrame,
    cfg: dict,
) -> None:
    tables_dir = Path((cfg.get("outputs") or {}).get("tables_dir", "results/tables"))
    tables_dir.mkdir(parents=True, exist_ok=True)

    def export_testreports(name: str, items: list[TestReport]) -> None:
        if not items:
            return
        df = pd.DataFrame(
            [
                {
                    "test_name": t.test_name,
                    "statistic": t.statistic,
                    "p_value": t.p_value,
                    "significant": t.significant,
                    "effect_size": t.effect_size,
                    "effect_size_label": t.effect_size_label,
                    "notes": t.notes,
                }
                for t in items
            ]
        )
        df.to_csv(tables_dir / name, index=False, encoding="utf-8")

    export_testreports("rq1_overconfidence_test.csv", report.rq1)
    export_testreports("rq2_strategy_comparison.csv", report.rq2)
    export_testreports("rq3_calibration_improvement.csv", report.rq3)
    export_testreports("rq4_complexity_moderation.csv", report.rq4)

    if not df_final.empty:
        summary = (
            df_final.groupby(["condition", "complexity"])
            .agg(
                n=("sample_id", "count"),
                ogs=("is_overconfident", "mean"),
                pass_rate=("is_correct", "mean"),
                assertive_rate=("is_assertive", "mean"),
            )
            .reset_index()
        )
        summary.to_csv(tables_dir / "summary_metrics.csv", index=False, encoding="utf-8")

    if not df_rounds.empty:
        # Round-level descriptive stats for RQ3 plots
        round_summary = (
            df_rounds.groupby(["condition", "round_number"])
            .agg(
                n=("sample_id", "count"),
                ogs=("is_overconfident", "mean"),
                pass_rate=("is_correct", "mean"),
                assertive_rate=("is_assertive", "mean"),
            )
            .reset_index()
        )
        round_summary.to_csv(
            tables_dir / "round_summary.csv", index=False, encoding="utf-8"
        )
