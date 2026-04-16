"""Load experiment YAML and merge environment (OpenAI-compatible providers e.g. 讯飞星火)."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_DEFAULTS: dict[str, Any] = {
    "execution": {"sandbox": "subprocess"},
    "tasks": {
        "task_file": "data/raw/tasks.jsonl",
        "tasks_per_level": 3,
        "dataset": "builtin",
        "humaneval": {
            "url": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
            "limit": 0,
            "always_refresh": False,
        },
    },
    "annotation": {"kappa_threshold": 0.7, "rubric_levels": 3},
    "strategies": {"C1": {}, "C2": {"max_rounds": 3}, "C3": {"max_rounds": 3}},
    "analysis": {"significance_level": 0.05},
    "llm": {
        "base_url": None,
        "default_model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 4096,
        "request_timeout_sec": 60,
    },
    "models": {
        "baseline": [{"name": "primary", "model": None}],
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | Path) -> dict[str, Any]:
    load_dotenv()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")

    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = copy.deepcopy(_DEFAULTS)
    _deep_merge(cfg, raw)

    llm = cfg.setdefault("llm", {})
    fly_key = (os.getenv("IFLYTEK_SPARK_API_KEY") or "").strip()
    fly_secret = (os.getenv("IFLYTEK_SPARK_API_SECRET") or "").strip()
    if fly_key and fly_secret:
        # 讯飞部分 OpenAI 兼容地址要求 Bearer 为「APIKey:APISecret」；与单独 HTTP 的 APIPassword 不同
        llm["api_key"] = f"{fly_key}:{fly_secret}"
    elif os.getenv("OPENAI_API_KEY"):
        llm["api_key"] = os.getenv("OPENAI_API_KEY", "").strip()
    if os.getenv("OPENAI_BASE_URL"):
        llm["base_url"] = os.getenv("OPENAI_BASE_URL").strip().rstrip("/")

    for m in cfg.get("models", {}).get("baseline", []) or []:
        if m.get("model") is None:
            m["model"] = llm.get("default_model")

    return cfg
