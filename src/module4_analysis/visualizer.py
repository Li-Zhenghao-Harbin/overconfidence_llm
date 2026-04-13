"""Module 4 — Figures (stub)."""

from __future__ import annotations

import logging
from pathlib import Path

from src.module4_analysis.statistical_tests import AnalysisReport

logger = logging.getLogger(__name__)


class ResultVisualizer:
    def __init__(self, config: dict):
        self.config = config

    def generate_all_figures(self, stats: AnalysisReport) -> None:
        Path("results/figures").mkdir(parents=True, exist_ok=True)
        logger.info("ResultVisualizer stub: no PNG/PDF generated.")
