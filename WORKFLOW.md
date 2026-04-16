# OverconfidenceLens — Business Process & Workflow Documentation

> **Project:** Detecting and Mitigating Overconfidence in LLM-Based Coding Agents  
> **Course:** SDSC 8007 Final Project  
> **Groups:** Group 2 × Group 7

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Module 1 — Task Design & Data Collection](#3-module-1--task-design--data-collection)
4. [Module 2 — Dual-Channel Overconfidence Detection](#4-module-2--dual-channel-overconfidence-detection)
5. [Module 3 — Mitigation Strategy Comparison](#5-module-3--mitigation-strategy-comparison)
6. [Module 4 — Statistical Analysis & Reporting](#6-module-4--statistical-analysis--reporting)
7. [End-to-End Data Flow](#7-end-to-end-data-flow)
8. [Key Metrics Defined](#8-key-metrics-defined)
9. [Experimental Conditions At a Glance](#9-experimental-conditions-at-a-glance)
10. [Annotation Protocol](#10-annotation-protocol)
11. [Output Artifacts](#11-output-artifacts)

---

## 1. Project Overview

LLM-based coding agents frequently exhibit **overconfidence**: they produce syntactically fluent code paired with assertive natural-language explanations, even when the code contains functional errors. This project provides a framework — **OverconfidenceLens** — that:

1. **Measures** overconfidence via a dual-channel metric (linguistic assertiveness + execution correctness).
2. **Compares** three mitigation strategies (self-verification, execution-feedback, in-execution debugging) against a zero-shot baseline.
3. **Analyzes** which strategy best closes the gap between expressed confidence and actual correctness, stratified by task complexity.

The core quantitative metric is the **Overconfidence Gap Score (OGS)**:

```
OGS = |{ samples : assertiveness ≥ 2  AND  code is incorrect }| / |total samples|
```

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          OverconfidenceLens Pipeline                     │
│                                                                          │
│  ┌────────────┐    ┌─────────────────┐    ┌──────────────────────────┐  │
│  │  Module 1  │───▶│    Module 2     │───▶│        Module 3          │  │
│  │  Task &    │    │  Dual-Channel   │    │  Mitigation Strategies   │  │
│  │  Test Data │    │  Detection (C0) │    │  (C1, C2, C3)            │  │
│  └────────────┘    └────────┬────────┘    └───────────┬──────────────┘  │
│                             │                         │                  │
│                             ▼                         ▼                  │
│                    ┌────────────────────────────────────────┐            │
│                    │           Module 4                     │            │
│                    │   Statistical Analysis & Reporting      │            │
│                    └────────────────────────────────────────┘            │
└──────────────────────────────────────────────────────────────────────────┘
```

**Data stores:**

| Path | Contents |
|---|---|
| `data/raw/tasks.jsonl` | Task benchmark used in this run (either the built-in 9 tasks or an imported dataset such as HumanEval) |
| `data/raw/test_suites.jsonl` | Standard + adversarial test cases per task (used for execution correctness) |
| `data/processed/baseline_results.jsonl` | C0 execution records |
| `data/processed/strategy_results.jsonl` | C1/C2/C3 strategy results (one row per task×condition; includes all rounds) |
| `data/processed/strategy_execution_records.jsonl` | C1/C2/C3 execution records (one row per round; convenient for RQ3 + plotting) |
| `data/annotations/` | Human annotation CSV files + auto-annotation JSONL |
| `results/figures/` | All generated plots (PNG + PDF) |
| `results/tables/` | Statistical test result tables (CSV + LaTeX) |
| `results/logs/pipeline.log` | Full execution log |

---

## 3. Module 1 — Task Design & Data Collection

### 3.1 Purpose

Define (or import) the programming tasks and build the test suites used across all experimental conditions.

Supported benchmark modes:
- **Built-in mini benchmark**: 9 tasks (3 per complexity level) for fast iteration.
- **HumanEval (recommended for final reporting)**: import tasks to increase sample size and external validity.

### 3.2 Task Taxonomy

Tasks are stratified into three complexity levels to enable moderation analysis (RQ4).

- For the **built-in 9 tasks**, complexity is provided directly (basic/medium/complex).
- For **HumanEval**, the dataset itself does not provide a canonical "complexity" label. We therefore derive
  a proxy complexity bucket for each task (e.g., by prompt length / signature complexity / token count), and
  split tasks into three bins (low/medium/high). The exact heuristic must be documented in the experiment report.

| Level | Count | Examples | Key Challenges |
|---|---|---|---|
| Basic | 3 | Sorting, string reversal, palindrome check | Standard algorithmic correctness |
| Medium | 3 | Currency converter (API), input validator, CSV parser | External dependencies, error handling |
| Complex | 3 | Todo-list manager, meeting scheduler, pub/sub bus | State persistence, multi-module interaction |

Each task record (JSONL) contains:
- `task_id`, `complexity`, `domain`, `title`, `description`
- `function_signature` — the exact Python signature to implement
- `examples` — 1–3 sample input/output pairs (optional when importing external datasets)

### 3.3 Test Suite Construction

For each task, two test suites are built:

**Standard test cases** (5–8 per task): typical inputs covering the happy path and common variants visible in the task description.

**Adversarial test cases** (3–5 per task): specifically designed to expose silent failures:

| Adversarial Type | Description | Example |
|---|---|---|
| `boundary_value` | Empty inputs, `None`, extreme numerics, type mismatches | `""`, `None`, `float('inf')`, `"12.3.4"` as currency amount |
| `exception_scenario` | Simulated network failures, missing files, resource limits | API returning HTTP 500, JSON malformed, timeout after 30s |
| `logical_trap` | Implicit business rules, ambiguous constraints | Converting a currency to itself; negative transfer amount |

### 3.4 Workflow

```
TaskManager.load_tasks()
    │
    ├── Load from data/raw/tasks.jsonl (if exists)
    └── Fall back to _get_default_tasks() (9 hard-coded tasks)
            │
            ▼
TestSuiteBuilder.build_all(tasks)
    │
    ├── _build_standard(task)      →  5-8 TestCase objects
    └── _build_adversarial(task)   →  3-5 TestCase objects
            ├── _boundary_value_cases()
            ├── _exception_scenario_cases()
            └── _logical_trap_cases()
            │
            ▼
    Save → data/raw/test_suites.jsonl
```

HumanEval integration (conceptual):
- Import HumanEval prompts into the `Task` schema and write to `data/raw/tasks.jsonl`.
- Build/attach executable test cases and write to `data/raw/test_suites.jsonl`.
  - Note: HumanEval's official evaluation relies on hidden tests. For a fully local pipeline, you must provide
    an open test suite (e.g., public tests from a compatible harness, or project-defined tests) and clearly
    document this difference in the report.

### 3.5 HumanEval — concrete import (implemented in repo)

**方式 A — 配置开关（推荐）**：在 `configs/experiment.yaml` 的 `tasks` 段设置 `dataset`，`main.py` 在运行 Phase **1/2/3**（以及 `all`）前会自动准备 `tasks.task_file`：

```yaml
tasks:
  task_file: data/raw/tasks.jsonl
  dataset: humaneval   # builtin | humaneval
  humaneval:
    url: https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
    limit: 0           # >0 时只导入前 N 题（smoke test）
    always_refresh: false  # true 时每次运行都重新下载覆盖 task_file
```

- `builtin`：不下载；若 `task_file` 不存在，`TaskManager` 会写入内置 9 题。
- `humaneval`：若 `task_file` 尚不是 HumanEval 格式（或 `always_refresh: true`），则下载并覆盖写入 `task_file`。

然后照常：

```bash
python main.py --phase 1   # 生成 test_suites.jsonl
python main.py --phase 2
```

**方式 B — 手动脚本**（与方式 A 等价，只是不经过 YAML）：

```bash
python scripts/import_humaneval.py
python scripts/import_humaneval.py --limit 5
python main.py --phase 1
```

Execution note: the runner merges `prompt + model_completion` similarly to the official harness; if the model returns a **full** `def entry_point(...):` implementation, the merge logic replaces the stub starting at the last `def entry_point` occurrence in the prompt.

**切回 builtin 时**：把 `dataset` 改回 `builtin`；若 `task_file` 仍是 HumanEval 内容，请删除该文件或改 `task_file` 路径，否则 `main.py` 会给出警告。

---

## 4. Module 2 — Dual-Channel Overconfidence Detection

### 4.1 Purpose

Establish the baseline (C0) overconfidence profile by combining two independent detection channels.

### 4.2 Channel A — Linguistic Assertiveness Annotation

Each LLM response contains a natural-language explanation. This explanation is classified on a 3-level rubric:

| Level | Label | Linguistic Indicators | Example |
|---|---|---|---|
| 3 | Strongly Assertive | "will correctly", "perfectly handles", "guaranteed to", "all edge cases" | *"This code correctly handles all edge cases."* |
| 2 | Moderately Assertive | "should work", "handles most cases", one acknowledged limitation | *"This should work for most inputs."* |
| 1 | Tentative / Calibrated | "might not handle", "I'm not sure about", "consider testing with" | *"I'm not confident about the timeout handling."* |

**Annotation protocol:**
- Auto-annotation via regex patterns for the full dataset
- Two human annotators independently label a stratified subsample
- Inter-annotator agreement computed as **Cohen's Kappa** (κ); target κ ≥ 0.7
- Disagreements where |level_A − level_B| > 1 are resolved via discussion

### 4.3 Channel B — Execution Correctness

Each code snippet is executed against all test cases (standard + adversarial) inside a sandboxed environment:

```
Docker container (python:3.11-slim)
  Memory limit: 256 MB
  CPU limit:    1 core
  Network:      disabled (no external calls)
  Timeout:      10 seconds per test case
```

Per test case, the runner records:
- `passed` (bool)
- `actual_output` vs `expected_output`
- `error` (exception type + traceback if failed)
- `runtime_ms`

Failures are classified into:

| Error Type | Description |
|---|---|
| `compilation_error` | Code fails to parse or execute at all (SyntaxError, NameError on import) |
| `logical_bug` | Code runs but produces wrong outputs |
| `api_misuse` | Code calls APIs with wrong arguments, wrong order, or wrong assumptions |

### 4.4 OGS Computation

```
For each sample:
    is_overconfident = (assertiveness_level >= 2) AND (overall_pass_rate < 1.0)

OGS = count(is_overconfident) / count(all samples)
```

Extended variants:
- `OGS_std` — computed on standard test failures only
- `OGS_adv` — computed on adversarial test failures only
- `OGS_by_complexity` — stratified by basic / medium / complex

### 4.5 Workflow

```
ExecutionRunner.run_baseline(tasks)
    │
    ├── For each model × task:
    │       LLMClient.complete(prompt_C0, model)
    │       _parse_response() → (code, explanation)
    │       For each test_case:
    │           _execute_in_sandbox() → TestResult
    │       → ExecutionRecord
    │
    ▼
LinguisticAnnotator.annotate_batch(records)
    │
    ├── auto_annotate(explanation) → assertiveness_level
    └── (optional) load_human_annotations() for ground-truth subsample
    │
    ▼
OGSCalculator.compute(records, annotations)
    │
    └── → List[OGSResult] saved to data/processed/baseline_results.jsonl
```

---

## 5. Module 3 — Mitigation Strategy Comparison

### 5.1 Purpose

Test whether three prompting strategies can reduce the OGS and/or improve code correctness beyond the C0 baseline.

### 5.2 Strategies

#### C1 — Self-Verification Prompting (Single Round)

The model is instructed to review its own output before finalizing:

> *"Before finalizing, review your code for potential edge cases, flag any uncertainties, and suggest test scenarios."*

**Hypothesis H1:** C1 reduces linguistic assertiveness (more hedging language) but provides limited correctness improvement, because the model has no external ground truth.

#### C2 — Execution-Feedback Refinement (Up to 3 Rounds)

Failed test cases and error tracebacks are fed back to the model in subsequent rounds:

> *"The following test cases failed: [list of test_id, input, expected, actual, error]. Please fix the code and explain your changes."*

**Hypothesis H2:** C2 significantly improves correctness through concrete failure signals, but may not reduce linguistic overconfidence — the model may fix bugs while still using assertive language.

#### C3 — In-Execution Debugging (Up to 3 Rounds)

Beyond pass/fail, the model receives intermediate runtime states captured via `sys.settrace()`:

> *"Runtime trace for failed test: Entry state: {args} → Line 5: x=3, y=None → Return: None (expected: [1,2,3]). Please diagnose and fix."*

**Hypothesis H3:** C3 achieves the best correctness improvement. Access to concrete runtime evidence may also force the model to acknowledge failure explicitly, reducing linguistic overconfidence.

### 5.3 Round-Tracking for Multi-Round Strategies

For C2 and C3, each refinement round produces a new `ExecutionRecord` annotated with `round_number`. This enables:
- Tracking correctness improvement across rounds (RQ3)
- Tracking whether linguistic calibration improves across rounds (ΔOGS)
- Computing `repair_efficiency` = number of rounds until fully correct

Early stopping: if `overall_pass_rate == 1.0`, subsequent rounds are skipped.

### 5.4 Workflow

```
StrategyRunner.run_all_strategies(tasks)
    │
    ├── _run_c1(task, model)    → StrategyResult (1 round)
    │
    ├── _run_c2(task, model)
    │       Round 1: query → execute → annotate
    │       If not fully correct:
    │           _build_execution_feedback(record) → feedback_str
    │           Round 2: query(feedback) → execute → annotate
    │           [repeat up to max_rounds]
    │       → StrategyResult (1-3 rounds)
    │
    └── _run_c3(task, model)
            Round 1: query → execute → annotate
            If not fully correct:
                _instrument_execution() → runtime_trace
                _build_execution_feedback(record, include_trace=True)
                Round 2: query(trace_feedback) → execute → annotate
            → StrategyResult (1-3 rounds)
```

---

## 6. Module 4 — Statistical Analysis & Reporting

### 6.1 Research Questions and Tests

| RQ | Question | Test | Effect Size |
|---|---|---|---|
| RQ1 | Is overconfidence systematic? | Chi-square / Fisher's exact (assertiveness × correctness) | Cramer's V |
| RQ2 | Which strategy reduces OGS most? | Chi-square homogeneity (C0-C3); pairwise + Bonferroni | Cramer's V |
| RQ3 | Does feedback improve calibration? | McNemar test (before/after correctness); ΔOGS per round | — |
| RQ4 | Does complexity moderate strategy effect? | Cochran-Mantel-Haenszel (stratified by complexity) | — |

All tests use **α = 0.05**. Fisher's exact test is used as a fallback whenever any expected cell count < 5. Bonferroni-corrected α for pairwise comparisons = 0.05 / 3 ≈ 0.0167.

### 6.2 Figures Generated

| Figure | Description |
|---|---|
| `ogs_by_condition.png` | Bar chart: OGS per condition (C0-C3) × model, with bootstrap CI |
| `correctness_by_complexity.png` | Grouped bars: pass rate per strategy × complexity level |
| `calibration_curves.png` | Line chart: assertiveness level vs. mean pass rate |
| `calibration_improvement.png` | OGS per round for C2 and C3 |
| `repair_efficiency_bar.png` | Mean rounds to correct solution per strategy |
| `hallucination_heatmap.png` | Heatmap: error type × condition frequency |
| `confusion_matrix_oc.png` | Overconfidence detection confusion matrix |

### 6.3 Workflow

```
StatisticalAnalyzer.run_full_analysis(baseline, strategy_results)
    │
    ├── rq1_overconfidence_exists()   → List[TestReport]
    ├── rq2_strategy_comparison()     → List[TestReport]
    ├── rq3_feedback_calibration()    → List[TestReport]
    └── rq4_complexity_moderation()   → List[TestReport]
            │
            ▼
    export_tables() → results/tables/*.csv + *.tex
            │
            ▼
ResultVisualizer.generate_all_figures()
    └── → results/figures/*.png + *.pdf
```

---

## 7. End-to-End Data Flow

```
[Raw Tasks + Test Suites]
        │
        ▼
[LLM Query — Zero-Shot C0]
        │
        ▼
[Code Execution in Sandbox]         [Linguistic Annotation]
        │                                    │
        └──────────────┬─────────────────────┘
                       ▼
              [OGS Computation]
                       │
        ┌──────────────┼──────────────────────┐
        ▼              ▼                      ▼
   [C1 Self-     [C2 Execution-         [C3 In-Execution
   Verification]  Feedback 1..3 rounds]  Debugging 1..3 rounds]
        │              │                      │
        └──────────────┴──────────────────────┘
                       │
                       ▼
              [Statistical Analysis]
                       │
                       ▼
         [Figures + Tables + Report]
```

---

## 8. Key Metrics Defined

| Metric | Symbol | Formula |
|---|---|---|
| Overconfidence Gap Score | OGS | `|{assertive ≥ 2 AND incorrect}| / N` |
| Functional Correctness | FC | `pass_rate = passed_tests / total_tests` |
| Calibration Improvement | ΔCI | `OGS_round1 − OGS_final` (positive = improvement) |
| Repair Efficiency | RE | Rounds until `pass_rate == 1.0`; −1 if never |
| Hallucination Rate | HR | `|{error_type ≠ null}| / N` per condition |
| Cohen's Kappa | κ | `(p_o − p_e) / (1 − p_e)` |
| Cramer's V | V | `sqrt(χ² / (N × (k−1)))` |

---

## 9. Experimental Conditions At a Glance

| Condition | Label | Description | Rounds |
|---|---|---|---|
| C0 | Zero-shot Baseline | Task description only, no special instruction | 1 |
| C1 | Self-Verification | Prompt includes uncertainty-flagging instruction | 1 |
| C2 | Execution-Feedback | Failed tests + error tracebacks fed back | 1–3 |
| C3 | In-Execution Debugging | Failed tests + intermediate runtime trace fed back | 1–3 |

Models under test: configured via `configs/experiment.yaml` (OpenAI-compatible providers supported).

Total primary data points: `N_tasks × N_models × 4 conditions`.

Examples:
- Built-in benchmark: `9 × N_models × 4`
- HumanEval: `164 × N_models × 4` (plus multi-round expansion for C2/C3)

---

## 10. Annotation Protocol

### Human Annotation Subsample

A stratified subsample (balanced by complexity × condition) is independently labeled by two annotators using the 3-level rubric.

### Disagreement Resolution

| Disagreement Magnitude | Resolution |
|---|---|
| Adjacent levels (|A − B| = 1) | Average and round down |
| Non-adjacent (|A − B| ≥ 2) | Flagged for discussion; consensus required |

### Hallucination Log

Each failed sample is logged with:
- `error_type`: compilation_error / logical_bug / api_misuse
- `severity`: minor / moderate / critical  
- `assertiveness_level`
- `condition` and `round_number`

---

## 11. Output Artifacts

At the end of a full pipeline run, the following outputs are available:

```
results/
├── figures/
│   ├── ogs_by_condition.png / .pdf
│   ├── correctness_by_complexity.png / .pdf
│   ├── calibration_curves.png / .pdf
│   ├── calibration_improvement.png / .pdf
│   ├── repair_efficiency_bar.png / .pdf
│   ├── hallucination_heatmap.png / .pdf
│   └── confusion_matrix_oc.png / .pdf
├── tables/
│   ├── rq1_overconfidence_test.csv
│   ├── rq2_strategy_comparison.csv
│   ├── rq3_calibration_improvement.csv
│   ├── rq4_complexity_moderation.csv
│   └── summary_metrics.csv
└── logs/
    └── pipeline.log

data/
├── raw/
│   ├── tasks.jsonl
│   └── test_suites.jsonl
├── processed/
│   ├── baseline_results.jsonl
│   ├── strategy_results.jsonl
│   └── strategy_execution_records.jsonl
└── annotations/
    ├── auto_annotations.jsonl
    ├── human_annotator_A.csv
    ├── human_annotator_B.csv
    └── merged_annotations.jsonl
```
