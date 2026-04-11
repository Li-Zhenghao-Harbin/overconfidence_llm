# OverconfidenceLens — Environment Setup & Startup Guide

> Tested on: Ubuntu 22.04 / macOS 14 (Apple Silicon & x86)  
> Python: 3.11  
> Package manager: Conda (Miniconda or Anaconda)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Create the Conda Environment](#2-create-the-conda-environment)
3. [Install Python Dependencies](#3-install-python-dependencies)
4. [Configure API Keys](#4-configure-api-keys)
5. [Set Up Docker Sandbox (Recommended)](#5-set-up-docker-sandbox-recommended)
6. [Verify the Installation](#6-verify-the-installation)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [Common Errors & Fixes](#8-common-errors--fixes)
9. [Environment Management](#9-environment-management)

---

## 1. Prerequisites

Before starting, ensure the following are installed on your system:

| Tool | Minimum Version | Install Guide |
|---|---|---|
| Conda | 23.x | https://docs.conda.io/en/latest/miniconda.html |
| Git | 2.x | https://git-scm.com |
| Docker | 24.x *(optional but recommended)* | https://docs.docker.com/get-docker/ |

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
cd overconfidence_lens

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
python -c "import openai, scipy, matplotlib, docker; print('All packages OK')"
```

Expected output:

```
All packages OK
```

---

## 4. Configure API Keys

The pipeline needs access to GPT-4o (OpenAI) and optionally GitHub Copilot.

### Step 1 — Copy the environment template

```bash
cp .env.example .env
```

### Step 2 — Fill in your credentials

Open `.env` in a text editor and set:

```dotenv
# Required: OpenAI API key for GPT-4o
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: GitHub token with Copilot scope (if testing Copilot)
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
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

## 5. Set Up Docker Sandbox (Recommended)

The execution sandbox runs generated code inside an isolated Docker container to prevent any harmful code from affecting your machine.

### Pull the base image

```bash
docker pull python:3.11-slim
```

### Verify Docker is accessible

```bash
docker run --rm python:3.11-slim python -c "print('Docker sandbox OK')"
```

Expected output:

```
Docker sandbox OK
```

### Fallback: subprocess sandbox

If Docker is not available (e.g., on an HPC cluster or CI environment), switch to the subprocess backend by editing `configs/experiment.yaml`:

```yaml
execution:
  sandbox: subprocess   # change from "docker" to "subprocess"
```

> **Warning:** The subprocess backend does not provide network isolation. Do not use it with untrusted code outside of a controlled experiment.

---

## 6. Verify the Installation

Run the built-in check script to confirm everything is configured correctly:

```bash
python scripts/check_env.py
```

Expected output:

```
[OK] Python 3.11.x
[OK] All required packages installed
[OK] OPENAI_API_KEY found in environment
[OK] Docker daemon reachable
[OK] configs/experiment.yaml is valid
[OK] data/ directories exist
Environment check passed. Ready to run.
```

---

## 7. Running the Pipeline

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

### Run tests

```bash
pytest tests/ -v
```

### Expected runtime

| Phase | Estimated Time |
|---|---|
| Phase 1 (task + test generation) | < 1 minute |
| Phase 2 (C0 baseline, 2 models × 9 tasks) | 5–15 minutes |
| Phase 3 (C1/C2/C3, multi-round) | 20–60 minutes |
| Phase 4 (analysis + figures) | 1–3 minutes |

Runtime varies by API latency and whether Docker is used.

---

## 8. Common Errors & Fixes

### `ModuleNotFoundError: No module named 'openai'`

The environment is not activated, or dependencies were not installed:

```bash
conda activate overconflens
pip install -r requirements.txt
```

---

### `openai.AuthenticationError: No API key provided`

The `.env` file is missing or the key is not loaded:

```bash
# Check the file exists and contains the key
cat .env | grep OPENAI_API_KEY

# Re-run after confirming
python main.py --phase 2
```

---

### `docker.errors.DockerException: Error while fetching server API version`

Docker daemon is not running. Start it:

```bash
# macOS
open -a Docker

# Linux (systemd)
sudo systemctl start docker
```

Or switch to subprocess sandbox in `configs/experiment.yaml` (see Section 5).

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

### `RateLimitError` from OpenAI

The API key has hit its rate limit. The `LLMClient` retries with exponential backoff (up to 3 attempts). If errors persist, add a delay between calls or use a different API key tier.

---

## 9. Environment Management

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
