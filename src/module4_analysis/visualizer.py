"""Module 4 — Figures.

Generates a small set of core figures described in WORKFLOW.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from src.module4_analysis.statistical_tests import AnalysisReport

logger = logging.getLogger(__name__)


class ResultVisualizer:
    def __init__(self, config: dict):
        self.config = config

    def generate_all_figures(self, stats: AnalysisReport) -> None:
        fig_dir = Path("results/figures")
        fig_dir.mkdir(parents=True, exist_ok=True)

        tables_dir = Path("results/tables")
        summary_path = tables_dir / "summary_metrics.csv"
        round_path = tables_dir / "round_summary.csv"

        if summary_path.is_file():
            df = pd.read_csv(summary_path)
            self._ogs_by_condition(df, fig_dir / "ogs_by_condition.png")
            self._correctness_by_complexity(df, fig_dir / "correctness_by_complexity.png")

        if round_path.is_file():
            df_r = pd.read_csv(round_path)
            self._calibration_improvement(df_r, fig_dir / "calibration_improvement.png")

        # Additional figures derived from raw processed data (no re-running phases).
        self._calibration_curves(fig_dir / "calibration_curves.png")
        self._repair_efficiency(fig_dir / "repair_efficiency_bar.png")

        logger.info("Figures generated under results/figures/")

    def _ogs_by_condition(self, df: pd.DataFrame, out: Path) -> None:
        # Aggregate across complexity
        agg = df.groupby("condition").agg(ogs=("ogs", "mean")).reset_index()
        plt.figure(figsize=(6, 4))
        plt.bar(agg["condition"], agg["ogs"])
        plt.ylim(0, max(0.05, float(agg["ogs"].max()) + 0.05))
        plt.title("OGS by Condition (final)")
        plt.ylabel("OGS")
        plt.xlabel("Condition")
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    def _correctness_by_complexity(self, df: pd.DataFrame, out: Path) -> None:
        pivot = (
            df.pivot_table(
                index="complexity", columns="condition", values="pass_rate", aggfunc="mean"
            )
            .fillna(0.0)
            .reindex(index=["basic", "medium", "complex"], fill_value=0.0)
        )
        ax = pivot.plot(kind="bar", figsize=(8, 4))
        ax.set_title("Pass Rate by Complexity × Condition (final)")
        ax.set_ylabel("Pass rate")
        ax.set_xlabel("Complexity")
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    def _calibration_improvement(self, df_r: pd.DataFrame, out: Path) -> None:
        # Line plot of OGS by round for C2/C3
        plt.figure(figsize=(7, 4))
        for cond in ["C2", "C3"]:
            sub = df_r[df_r["condition"] == cond].sort_values("round_number")
            if sub.empty:
                continue
            plt.plot(sub["round_number"], sub["ogs"], marker="o", label=cond)
        plt.title("OGS by Round (C2/C3)")
        plt.xlabel("Round")
        plt.ylabel("OGS")
        plt.ylim(0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    def _calibration_curves(self, out: Path) -> None:
        """Assertiveness level vs mean pass rate (final per condition)."""
        import json

        def read_jsonl(p: Path) -> list[dict]:
            if not p.is_file():
                return []
            rows = []
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows

        # Baseline (needs annotations)
        base = read_jsonl(Path("data/processed/baseline_results.jsonl"))
        ann = {r["sample_id"]: r.get("assertiveness_level") for r in read_jsonl(Path("data/annotations/auto_annotations.jsonl")) if "sample_id" in r}
        base_rows = []
        for r in base:
            base_rows.append(
                {
                    "condition": "C0",
                    "assertiveness_level": int(ann.get(r.get("sample_id"), 0) or 0),
                    "overall_pass_rate": float(r.get("overall_pass_rate", 0.0)),
                }
            )

        strat = read_jsonl(Path("data/processed/strategy_execution_records.jsonl"))
        df_s = pd.DataFrame(strat)
        if not df_s.empty:
            # final round per task×cond
            df_s["round_number"] = df_s["round_number"].astype(int)
            df_s = (
                df_s.sort_values(["condition", "model", "task_id", "round_number"])
                .groupby(["condition", "model", "task_id"], as_index=False)
                .tail(1)
            )
            strat_rows = df_s[
                ["condition", "assertiveness_level", "overall_pass_rate"]
            ].to_dict(orient="records")
        else:
            strat_rows = []

        df = pd.DataFrame(base_rows + strat_rows)
        if df.empty:
            return
        df["assertiveness_level"] = df["assertiveness_level"].fillna(0).astype(int)
        df["overall_pass_rate"] = df["overall_pass_rate"].astype(float)

        plt.figure(figsize=(7, 4))
        for cond in ["C0", "C1", "C2", "C3"]:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            agg = (
                sub.groupby("assertiveness_level")["overall_pass_rate"]
                .mean()
                .reset_index()
                .sort_values("assertiveness_level")
            )
            plt.plot(
                agg["assertiveness_level"],
                agg["overall_pass_rate"],
                marker="o",
                label=cond,
            )
        plt.title("Calibration Curves (assertiveness vs pass rate)")
        plt.xlabel("Assertiveness level")
        plt.ylabel("Mean pass rate")
        plt.ylim(0, 1.05)
        plt.xticks([1, 2, 3])
        plt.legend()
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()

    def _repair_efficiency(self, out: Path) -> None:
        """Mean rounds to reach correctness for C2/C3 (−1 excluded)."""
        import json

        p = Path("data/processed/strategy_execution_records.jsonl")
        if not p.is_file():
            return
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        df = pd.DataFrame(rows)
        if df.empty:
            return
        df["round_number"] = df["round_number"].astype(int)
        df["overall_pass_rate"] = df["overall_pass_rate"].astype(float)

        eff_rows = []
        for cond in ["C2", "C3"]:
            sub = df[df["condition"] == cond].sort_values(["model", "task_id", "round_number"])
            if sub.empty:
                continue
            for (m, t), g in sub.groupby(["model", "task_id"]):
                reached = g[g["overall_pass_rate"] == 1.0]
                if reached.empty:
                    continue
                eff_rows.append({"condition": cond, "repair_efficiency": int(reached["round_number"].iloc[0])})

        eff = pd.DataFrame(eff_rows)
        plt.figure(figsize=(5, 4))
        if eff.empty:
            plt.title("Repair efficiency (no successes)")
            plt.tight_layout()
            plt.savefig(out, dpi=200)
            plt.close()
            return
        agg = eff.groupby("condition")["repair_efficiency"].mean().reindex(["C2", "C3"])
        plt.bar(agg.index, agg.values)
        plt.title("Repair Efficiency (mean rounds to correct)")
        plt.xlabel("Condition")
        plt.ylabel("Mean rounds")
        plt.ylim(0, max(1.0, float(agg.max()) + 0.5))
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
