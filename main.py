"""
OverconfidenceLens: Detecting and Mitigating Overconfidence in LLM-Based Coding Agents
Main entry point for running the full experimental pipeline.

Usage:
    python main.py --phase all
    python main.py --phase 1           # Data collection only
    python main.py --phase 2           # Detection only (requires phase 1)
    python main.py --phase 3           # Mitigation comparison (requires phase 1)
    python main.py --phase 4           # Statistical analysis (requires phase 2&3)
    python main.py --config configs/experiment.yaml
"""

import argparse
import logging
from pathlib import Path

from src.module1_data.task_manager import TaskManager
from src.module1_data.test_suite import TestSuiteBuilder
from src.module2_detection.linguistic_annotator import LinguisticAnnotator
from src.module2_detection.execution_runner import ExecutionRunner
from src.module2_detection.ogs_calculator import OGSCalculator
from src.module3_mitigation.strategy_runner import StrategyRunner
from src.module4_analysis.statistical_tests import StatisticalAnalyzer
from src.module4_analysis.visualizer import ResultVisualizer
from src.module1_data.humaneval_dataset import ensure_dataset_tasks
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.pipeline_io import (
    load_baseline_results,
    load_strategy_results,
    save_baseline_results,
    save_strategy_results,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OverconfidenceLens Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                        choices=["all", "1", "2", "3", "4"],
                        help="Which pipeline phase to run")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml",
                        help="Path to experiment config file")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_phase1(config, logger):
    """Phase 1: Task Design & Data Collection"""
    logger.info("=== Phase 1: Task Design & Data Collection ===")
    task_manager = TaskManager(config)
    tasks = task_manager.load_tasks()

    test_builder = TestSuiteBuilder(config)
    test_suites = test_builder.build_all(tasks)

    logger.info(f"Loaded {len(tasks)} tasks, built {len(test_suites)} test suites")
    return tasks, test_suites


def run_phase2(config, tasks, logger):
    """Phase 2: Dual-Channel Overconfidence Detection (Baseline C0)"""
    logger.info("=== Phase 2: Dual-Channel Overconfidence Detection ===")

    runner = ExecutionRunner(config)
    results = runner.run_baseline(tasks)

    annotator = LinguisticAnnotator(config)
    annotations = annotator.annotate_batch(results)

    ogs_calc = OGSCalculator(config)
    ogs_scores = ogs_calc.compute(results, annotations)

    logger.info(f"OGS computed for {len(ogs_scores)} samples")
    return results, annotations, ogs_scores


def run_phase3(config, tasks, logger):
    """Phase 3: Mitigation Strategy Comparison (C1, C2, C3)"""
    logger.info("=== Phase 3: Mitigation Strategy Comparison ===")
    strategy_runner = StrategyRunner(config)
    strategy_results = strategy_runner.run_all_strategies(tasks)
    logger.info("All mitigation strategies executed")
    return strategy_results


def run_phase4(config, baseline_results, strategy_results, logger):
    """Phase 4: Statistical Analysis & Recommendations"""
    logger.info("=== Phase 4: Statistical Analysis & Recommendations ===")
    analyzer = StatisticalAnalyzer(config)
    stats = analyzer.run_full_analysis(baseline_results, strategy_results)

    visualizer = ResultVisualizer(config)
    visualizer.generate_all_figures(stats)

    logger.info("Analysis complete. See results/ directory.")
    return stats


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logger(args.log_level)

    logger.info("OverconfidenceLens Pipeline Starting")
    Path("results").mkdir(exist_ok=True)

    if args.phase in ("all", "1", "2", "3"):
        ensure_dataset_tasks(config, logger)

    tasks, test_suites = None, None
    baseline_results, annotations, ogs_scores = None, None, None
    strategy_results = None

    if args.phase in ("all", "1"):
        tasks, test_suites = run_phase1(config, logger)

    # Phase 2/3 run in a separate process do not keep in-memory `tasks`; reload from
    # data/raw/tasks.jsonl (written in Phase 1).
    if tasks is None and args.phase in ("2", "3"):
        logger.info("=== Loading tasks from Phase 1 data ===")
        task_manager = TaskManager(config)
        tasks = task_manager.load_tasks()
        logger.info("Loaded %d tasks for phase %s", len(tasks), args.phase)

    if args.phase in ("all", "2"):
        assert tasks is not None, "Run Phase 1 first or ensure data/raw/tasks.jsonl exists"
        baseline_results, annotations, ogs_scores = run_phase2(config, tasks, logger)
        save_baseline_results(baseline_results)

    if args.phase in ("all", "3"):
        assert tasks is not None, "Run Phase 1 first"
        strategy_results = run_phase3(config, tasks, logger)
        save_strategy_results(strategy_results)

    if args.phase in ("all", "4"):
        if baseline_results is None:
            baseline_results = load_baseline_results()
        if strategy_results is None:
            strategy_results = load_strategy_results()
        assert baseline_results is not None, (
            "Run Phase 2 first (or rerun it) so data/intermediate/phase2_baseline.jsonl exists"
        )
        assert strategy_results is not None, (
            "Run Phase 3 first (or rerun it) so data/intermediate/phase3_strategy_results.json exists"
        )
        stats = run_phase4(config, baseline_results, strategy_results, logger)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
