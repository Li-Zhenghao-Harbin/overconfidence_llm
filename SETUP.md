# OverconfidenceLens — Environment Setup & Startup Guide

> Tested on: Ubuntu 22.04 / macOS 14 (Apple Silicon & x86) / Windows 10+  
> Python: 3.11  
> Package manager: Conda (Miniconda or Anaconda) recommended

**Execution:** generated code runs in a **local subprocess** sandbox (`execution.sandbox: subprocess` in `configs/experiment.yaml`). **Docker is not required** for this repository’s default pipeline.

**Outputs:** Phase 4 writes **PNG figures** and **CSV tables** under `results/` (see `WORKFLOW.md`). There is **no** built-in PDF/LaTeX export or auto-generated Markdown report.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create the Conda Environment](#2-create-the-conda-environment)
3. [Install Python Dependencies](#3-install-python-dependencies)
4. [Configure API Keys](#4-configure-api-keys)
5. [Execution Sandbox (Subprocess)](#5-execution-sandbox-subprocess)
6. [Verify the Installation](#6-verify-the-installation)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Optional — Train the Error-Severity Model (PyTorch)](#8-optional--train-the-error-severity-model-pytorch)
9. [Common Errors & Fixes](#9-common-errors--fixes)
10. [Environment Management](#10-environment-management)

---

## 1. Prerequisites

Before starting, ensure the following are installed on your system:

| Tool | Minimum Version | Install Guide |
|---|---|---|
| Conda | 23.x | https://docs.conda.io/en/latest/miniconda.html |
| Git | 2.x | https://git-scm.com |

Check your Conda version:

```bash
conda --version
# conda 23.x.x or higher
```

---

## 2. Create the Conda Environment

### Option A — From the environment YAML (recommended)

The repository includes a pre-configured environment file:

```bash
# Clone the repository (if not already done)
git clone <your-repo-url>
cd overconfidence_llm

# Create the environment from the YAML spec
conda env create -f environment.yml

# Activate the environment
conda activate overconflens
```

### Option B — Manual creation

If you prefer to create the environment step by step:

```bash
# Create a new environment named 'overconflens' with Python 3.11
conda create -n overconflens python=3.11 -y

# Activate the environment
conda activate overconflens
```

> **Important:** Always make sure `(overconflens)` appears in your shell prompt before running any project commands. Every terminal session requires a fresh `conda activate overconflens`.

---

## 3. Install Python Dependencies

With the environment activated, install all required packages:

```bash
# Upgrade pip first (avoids compatibility issues)
pip install --upgrade pip

# Install all project dependencies
pip install -r requirements.txt
```

### Verify key packages

```bash
python -c "import openai, scipy, matplotlib, torch, yaml; print('All packages OK')"
```

Expected output:

```
All packages OK
```

`torch` is used by the optional **error severity** ByteCNN (`severity_dl` in `configs/experiment.yaml`). If you disable `severity_dl.enabled`, PyTorch is still listed in `requirements.txt` for a reproducible one-line install.

---

## 4. Configure API Keys

The pipeline calls an **OpenAI-compatible** HTTP API (OpenAI, GitHub Models, 讯飞星火, etc.) using the `openai` Python SDK. Configure the key and base URL to match your provider.

### Step 1 — Copy the environment template

```bash
cp .env.example .env
```

### Step 2 — Fill in your credentials

Open `.env` in a text editor and set (examples):

```dotenv
# API key for your provider (name may vary — see configs/experiment.yaml llm.base_url)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: override API base URL
# OPENAI_BASE_URL=https://api.openai.com/v1
```

> **Security note:** Never commit `.env` to version control. It is listed in `.gitignore` by default.

### Step 3 — Verify the key loads

```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('Key loaded:', bool(os.getenv('OPENAI_API_KEY')))"
```

Expected output:

```
Key loaded: True
```

---

## 5. Execution Sandbox (Subprocess)

By default, `configs/experiment.yaml` uses:

```yaml
execution:
  sandbox: subprocess
  case_timeout_sec: 15
```

Model-generated Python is executed in a **short-lived local Python subprocess** per test case (see `src/module2_detection/execution_runner.py`). This matches the current codebase: **no Docker daemon is required**.

> **Warning:** Subprocess execution does **not** provide strong isolation (e.g. no network namespace). Only run the pipeline on **trusted benchmarks** in a controlled experiment environment.

---

## 6. Verify the Installation

Quick checks (no separate script required):

```bash
# Python version
python --version

# Config loads
python -c "from src.utils.config import load_config; c=load_config('configs/experiment.yaml'); print('dataset:', c['tasks'].get('dataset'))"

# Core imports
python -c "import openai, scipy, matplotlib, torch; print('imports OK')"
```

If these succeed, you are ready to run `main.py` (Section 7).

---

## 7. Running the Pipeline

End-to-end behaviour is documented in **`WORKFLOW.md`** (modules, metrics, artifacts). Typical order:

| Phase | Command | Role |
|---:|---|---|
| 1 | `python main.py --phase 1` | Tasks + test suites → `data/raw/` |
| 2 | `python main.py --phase 2` | C0 baseline: LLM code, execution, **regex assertiveness**, **OGS fields**, optional **DL `severity`** on failed tests → `data/processed/baseline_results.jsonl` |
| 3 | `python main.py --phase 3` | C1/C2/C3 strategies → `strategy_results.jsonl` + `strategy_execution_records.jsonl` |
| 4 | `python main.py --phase 4` | Stats + **PNG** + **CSV** → `results/figures/`, `results/tables/` |

### Run the full pipeline (all 4 phases)

```bash
python main.py --phase all
```

### Run individual phases

```bash
# Phase 1 only — generate tasks and test suites
python main.py --phase 1

# Phase 2 only — baseline detection (requires phase 1 to have run first)
python main.py --phase 2

# Phase 3 only — mitigation strategies (requires phase 1)
python main.py --phase 3

# Phase 4 only — statistical analysis and figures (requires phases 2 & 3)
python main.py --phase 4
```

### Use a custom config

```bash
python main.py --phase all --config configs/experiment.yaml
```

### Adjust log verbosity

```bash
python main.py --phase all --log-level DEBUG
```

### Run tests (if `tests/` exists)

```bash
pytest tests/ -v
```

### Expected runtime

| Phase | Estimated Time |
|---|---|
| Phase 1 (task + test generation) | < 1 minute |
| Phase 2 (C0 baseline; scales with `#tasks × #models`) | minutes–hours |
| Phase 3 (C1/C2/C3, multi-round) | often longer than Phase 2 |
| Phase 4 (analysis + figures) | ~1–5 minutes |

Runtime depends on API latency, task count (`tasks.dataset`, limits in YAML), and strategy `max_rounds`.

---

## 8. Optional — Train the Error-Severity Model (PyTorch)

When `severity_dl.enabled` is `true` in `configs/experiment.yaml`, Phase 2/3 attach a **`severity`** label (`minor` / `moderate` / `critical`) to each **failed** test case. If the checkpoint file is missing, the code **falls back to rule-based severity** and logs a warning.

1. Run at least Phase 2 (and optionally Phase 3) once so `data/processed/` contains JSONL with failures.
2. Train:

```bash
python scripts/train_severity_dl.py ^
  --inputs data/processed/baseline_results.jsonl data/processed/strategy_execution_records.jsonl ^
  --out models/severity_cnn.pt
```

(On macOS/Linux, use `\` line continuation instead of `^`.)

3. Re-run Phase 2/3 so new rows include **model-predicted** `severity` (same field name; values come from the ByteCNN when the checkpoint loads).

Training uses **pseudo-labels** from `rule_severity` inside `src/module2_detection/severity_dl.py` — no manual severity annotation is required for the first model.

---

## 9. Common Errors & Fixes

### `ModuleNotFoundError: No module named 'openai'`

The environment is not activated, or dependencies were not installed:

```bash
conda activate overconflens
pip install -r requirements.txt
```

---

### `openai.AuthenticationError` or provider 401

The `.env` file is missing, the key is wrong, or `llm.base_url` does not match your provider. Fix `.env` and `configs/experiment.yaml`, then re-run Phase 2.

---

### `AssertionError: Run Phase 1 first`

Phases must be run in order. Either run `--phase all`, or run phases sequentially:

```bash
python main.py --phase 1
python main.py --phase 2
```

---

### `scipy.stats` import warning on Apple Silicon

Install scipy with conda instead of pip to get the native ARM build:

```bash
conda install scipy -c conda-forge -y
```

---

### Rate limits / throttling from the API

The LLM client retries with backoff for some providers. If errors persist, reduce concurrency implicitly by fewer tasks, increase delays, or use a higher quota tier.

---

### Severity always looks rule-based

Ensure `models/severity_cnn.pt` exists (Section 8), `severity_dl.enabled: true`, and `torch` imports successfully. Check logs for “Loaded severity DL model” vs “checkpoint missing”.

---

## 10. Environment Management

### List all conda environments

```bash
conda env list
```

### Deactivate the environment

```bash
conda deactivate
```

### Update dependencies after a `requirements.txt` change

```bash
conda activate overconflens
pip install -r requirements.txt --upgrade
```

### Export your environment for reproducibility

```bash
conda activate overconflens
conda env export > environment.yml
```

### Remove the environment (clean slate)

```bash
conda deactivate
conda env remove -n overconflens
```

---

## Appendix — `environment.yml` Reference

The `environment.yml` file used in Option A of Section 2:

```yaml
name: overconflens
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pip
  - pip:
    - -r requirements.txt
```

This file is located at the project root and pins the Python version to 3.11 to ensure reproducibility across team members.
