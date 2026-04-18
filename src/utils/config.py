"""Load experiment YAML and merge environment (OpenAI-compatible providers e.g. 讯飞星火)."""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_DEFAULTS: dict[str, Any] = {
    "execution": {"sandbox": "subprocess"},
    "outputs": {
        # null / omit: follow `tasks.raw_run_id` when it versions raw data; else flat layout.
        # "auto": always use `{dataset}_{baseline_slug}` subdirectory.
        # "flat": force flat `data/processed` + `results` even if raw is versioned.
        # other string: custom slug subdirectory.
        "run_id": None,
        "processed_dir": "data/processed",
        "results_dir": "results",
        "tables_dir": "results/tables",
        "figures_dir": "results/figures",
        "annotations_dir": "data/annotations",
        "intermediate_dir": "data/intermediate",
    },
    "tasks": {
        "task_file": "data/raw/tasks.jsonl",
        "suite_file": None,
        "tasks_per_level": 3,
        "dataset": "builtin",
        "raw_run_id": None,
        "humaneval": {
            "url": "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
            "limit": 0,
            "always_refresh": False,
        },
        "mhpp": {
            "url": "https://raw.githubusercontent.com/SparksofAGI/MHPP/main/data/MHPP.jsonl",
            "limit": 0,
            "always_refresh": False,
        },
        "apps": {
            "hf_dataset": "codeparrot/apps",
            "hf_revision": "refs/convert/parquet",
            "split": "train",
            "limit": 0,
            "difficulties": None,
            "max_tests_per_task": 5,
            "always_refresh": False,
            "skip_call_based": True,
        },
    },
    "annotation": {"kappa_threshold": 0.7, "rubric_levels": 3},
    "severity_dl": {
        "enabled": False,
        "checkpoint": "models/severity_cnn.pt",
        "device": "cpu",
        "max_len": 384,
        "emb_dim": 48,
        "conv_dim": 96,
    },
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


def _slug_label(s: str, max_len: int = 48) -> str:
    s = (s or "x").lower().strip()
    out: list[str] = []
    for c in s:
        if c.isalnum():
            out.append(c)
        else:
            out.append("_")
    slug = re.sub(r"_+", "_", "".join(out)).strip("_")
    return (slug or "x")[:max_len]


def resolve_auto_run_slug(cfg: dict) -> str:
    """Same slug as `tasks.raw_run_id: auto` (dataset + first baseline model label)."""
    t = cfg.setdefault("tasks", {})
    ds = str(t.get("dataset", "builtin")).strip().lower() or "builtin"
    models = (cfg.get("models") or {}).get("baseline") or []
    label = ""
    if models:
        label = str(models[0].get("name") or models[0].get("model") or "")
    if not label:
        label = str((cfg.get("llm") or {}).get("default_model") or "model")
    return f"{ds}_{_slug_label(label)}"


def _derive_suite_file(task_file: Path) -> Path:
    """Pair `tasks*.jsonl` with `test_suites*.jsonl` in the same directory."""
    parent = task_file.parent
    name = task_file.name
    if name.startswith("tasks_") and name.endswith(".jsonl") and name != "tasks.jsonl":
        mid = name[len("tasks_") : -len(".jsonl")]
        return parent / f"test_suites_{mid}.jsonl"
    return parent / "test_suites.jsonl"


def apply_task_raw_paths(cfg: dict) -> None:
    """Resolve `tasks.task_file` / `tasks.suite_file` from optional `raw_run_id`.

    - `raw_run_id` unset / null / "": keep YAML `task_file` basename; set `suite_file` if missing.
    - `raw_run_id: auto`: `tasks_{dataset}_{baseline_model_slug}.jsonl` (+ paired test_suites) under the
      same directory as the configured `task_file` (directory only; default basename is ignored).
    - other string: `tasks_{slug(raw_run_id)}.jsonl` (+ paired test_suites).
    """
    t = cfg.setdefault("tasks", {})
    tf_in = Path(t.get("task_file", "data/raw/tasks.jsonl"))
    parent = tf_in.parent
    rid = t.get("raw_run_id")

    if rid is None or (isinstance(rid, str) and not str(rid).strip()):
        cfg["_run_slug"] = None
        t["task_file"] = str(tf_in)
        if not t.get("suite_file"):
            t["suite_file"] = str(_derive_suite_file(tf_in))
        return

    rid_s = str(rid).strip()
    if rid_s.lower() == "auto":
        rid_use = resolve_auto_run_slug(cfg)
    else:
        rid_use = _slug_label(rid_s)

    cfg["_run_slug"] = rid_use
    t["task_file"] = str(parent / f"tasks_{rid_use}.jsonl")
    t["suite_file"] = str(parent / f"test_suites_{rid_use}.jsonl")


def apply_output_paths(cfg: dict) -> None:
    """Set `outputs.*_dir` from `outputs.run_id` and optional coupling to `tasks.raw_run_id`.

    - `outputs.run_id` null/empty: use `cfg['_run_slug']` when raw paths were versioned; else flat.
    - `outputs.run_id: auto`: always `{dataset}_{baseline_slug}` (even if raw is flat).
    - `outputs.run_id: flat`: always flat dirs regardless of raw.
    - other string: slug subdirectory under `data/processed`, `results`, `data/annotations`,
      `data/intermediate`.
    """
    o = cfg.setdefault("outputs", {})
    out_rid = o.get("run_id")
    slug: str | None

    if isinstance(out_rid, str) and out_rid.strip().lower() == "flat":
        slug = None
    elif out_rid is None or (isinstance(out_rid, str) and not str(out_rid).strip()):
        slug = cfg.get("_run_slug")
    elif str(out_rid).strip().lower() == "auto":
        slug = resolve_auto_run_slug(cfg)
    else:
        slug = _slug_label(str(out_rid).strip())

    if not slug:
        o["processed_dir"] = str(Path("data/processed"))
        o["results_dir"] = str(Path("results"))
        o["annotations_dir"] = str(Path("data/annotations"))
        o["intermediate_dir"] = str(Path("data/intermediate"))
    else:
        o["processed_dir"] = str(Path("data/processed") / slug)
        o["results_dir"] = str(Path("results") / slug)
        o["annotations_dir"] = str(Path("data/annotations") / slug)
        o["intermediate_dir"] = str(Path("data/intermediate") / slug)

    rd = Path(o["results_dir"])
    o["tables_dir"] = str(rd / "tables")
    o["figures_dir"] = str(rd / "figures")


def load_config(path: str | Path) -> dict[str, Any]:
    # Prefer the repo-local `.env` for reproducibility across runs.
    # Many users have global OPENAI_API_KEY set (e.g., for another provider);
    # without override=True, python-dotenv will not replace it.
    load_dotenv(override=True)
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

    apply_task_raw_paths(cfg)
    apply_output_paths(cfg)

    from src.utils.pipeline_io import configure_from_config

    configure_from_config(cfg)

    return cfg
