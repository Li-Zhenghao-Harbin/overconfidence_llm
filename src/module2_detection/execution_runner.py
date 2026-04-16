"""Module 2 — Run LLM-generated code against test suites (subprocess sandbox)."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from src.module1_data.task_manager import Task
from src.module1_data.test_suite import TestCase, TestSuite, TestSuiteBuilder

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    test_id: str
    passed: bool
    kind: str = "standard"  # standard | adversarial
    expected_output: Any = None
    actual_output: Any = None
    error: str = ""
    runtime_ms: float = 0.0
    error_type: str = ""  # compilation_error | logical_bug | api_misuse | runtime_error | timeout


@dataclass
class ExecutionRecord:
    sample_id: str
    task_id: str
    model: str
    condition: str
    code: str
    explanation: str
    overall_pass_rate: float
    test_results: list[TestResult] = field(default_factory=list)


def _func_name(signature: str) -> str:
    m = re.search(r"def\s+(\w+)\s*\(", signature)
    if m:
        return m.group(1)
    m = re.search(r"class\s+(\w+)", signature)
    return m.group(1) if m else "solution"


def _extract_code(text: str) -> str:
    if "```python" in text:
        return text.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[1].split("```", 1)[0].strip()
    return text.strip()


def _outputs_equal(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return float(actual) == float(expected)
    try:
        return json.dumps(actual, sort_keys=True) == json.dumps(
            expected, sort_keys=True
        )
    except (TypeError, ValueError):
        return False


def _invoke_in_process(
    code: str, fn_name: str, inp: Any, timeout: int
) -> tuple[bool, Any, str, float]:
    """Run user code in a fresh subprocess (no Docker)."""
    payload = json.dumps({"code": code, "fn": fn_name, "inp": inp})
    prog = r"""
import json, sys, traceback, time
p = json.loads(sys.stdin.read())
code, fn_name, inp = p["code"], p["fn"], p["inp"]
try:
    t0 = time.perf_counter()
    g = {}
    exec(compile(code, "<llm>", "exec"), g, g)
    fn = g[fn_name]
    if isinstance(inp, dict):
        out = fn(**inp)
    else:
        out = fn(inp)
    t1 = time.perf_counter()
    json.dump({"ok": True, "out": out, "runtime_ms": (t1 - t0) * 1000.0}, sys.stdout, default=str)
except Exception:
    t1 = time.perf_counter()
    json.dump({"ok": False, "err": traceback.format_exc(), "runtime_ms": (t1 - t0) * 1000.0 if 't0' in globals() else 0.0}, sys.stdout)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", prog],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, None, "timeout", float(timeout) * 1000.0
    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    if not raw:
        err = proc.stderr.decode("utf-8", errors="replace")
        return False, None, err or "empty stdout"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, None, raw[:2000], 0.0
    if data.get("ok"):
        return True, data.get("out"), "", float(data.get("runtime_ms", 0.0) or 0.0)
    return (
        False,
        None,
        str(data.get("err", "")),
        float(data.get("runtime_ms", 0.0) or 0.0),
    )


def _invoke_in_process_trace(
    code: str, fn_name: str, inp: Any, timeout: int, max_events: int = 200
) -> tuple[bool, Any, str, float, str]:
    """Run user code and capture a lightweight line-by-line trace for the target function.

    Trace is best-effort: it records at most `max_events` line events and a small
    snapshot of locals (repr-truncated).
    """
    payload = json.dumps(
        {"code": code, "fn": fn_name, "inp": inp, "max_events": int(max_events)}
    )
    prog = r"""
import json, sys, traceback, time
p = json.loads(sys.stdin.read())
code, fn_name, inp = p["code"], p["fn"], p["inp"]
max_events = int(p.get("max_events", 200))

events = []
def _safe_repr(x, limit=200):
    try:
        s = repr(x)
    except Exception:
        s = "<unrepr>"
    return s if len(s) <= limit else s[:limit] + "…"

def tracer(frame, event, arg):
    if frame.f_code.co_filename != "<llm>":
        return tracer
    if frame.f_code.co_name != fn_name:
        return tracer
    if event == "line":
        if len(events) < max_events:
            loc = {k: _safe_repr(v) for k, v in frame.f_locals.items() if not k.startswith("__")}
            events.append({"line": frame.f_lineno, "locals": loc})
        return tracer
    return tracer

try:
    t0 = time.perf_counter()
    g = {}
    exec(compile(code, "<llm>", "exec"), g, g)
    fn = g[fn_name]
    sys.settrace(tracer)
    if isinstance(inp, dict):
        out = fn(**inp)
    else:
        out = fn(inp)
    sys.settrace(None)
    t1 = time.perf_counter()
    json.dump({"ok": True, "out": out, "runtime_ms": (t1 - t0) * 1000.0, "trace": events}, sys.stdout, default=str)
except Exception:
    sys.settrace(None)
    t1 = time.perf_counter()
    json.dump({"ok": False, "err": traceback.format_exc(), "runtime_ms": (t1 - t0) * 1000.0 if 't0' in globals() else 0.0, "trace": events}, sys.stdout, default=str)
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", prog],
            input=payload.encode("utf-8"),
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, None, "timeout", float(timeout) * 1000.0, ""
    raw = proc.stdout.decode("utf-8", errors="replace").strip()
    if not raw:
        err = proc.stderr.decode("utf-8", errors="replace")
        return False, None, err or "empty stdout", 0.0, ""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, None, raw[:2000], 0.0, ""

    trace = json.dumps(data.get("trace") or [], ensure_ascii=False)
    if data.get("ok"):
        return True, data.get("out"), "", float(data.get("runtime_ms", 0.0) or 0.0), trace
    return (
        False,
        None,
        str(data.get("err", "")),
        float(data.get("runtime_ms", 0.0) or 0.0),
        trace,
    )


def _classify_error(err: str) -> str:
    if not err:
        return ""
    e = err.lower()
    if "timeout" in e:
        return "timeout"
    if "syntaxerror" in e or "indentationerror" in e:
        return "compilation_error"
    if "nameerror" in e or "importerror" in e or "modulenotfounderror" in e:
        return "api_misuse"
    return "runtime_error"


class _LLMClient:
    def __init__(self, config: dict):
        try:
            from openai import AuthenticationError, OpenAI  # type: ignore
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "缺少依赖 `openai`，Phase 2/3 需要安装：\n"
                "  pip install openai\n"
                "Phase 1 不需要该依赖。"
            ) from e

        self._AuthenticationError = AuthenticationError
        llm = config.get("llm", {})
        api_key = llm.get("api_key")
        if not api_key:
            raise ValueError(
                "Missing API key: set OPENAI_API_KEY in .env (讯飞控制台 APIPassword 填在此变量即可)."
            )
        base = llm.get("base_url")
        kwargs: dict = {"api_key": api_key}
        if base:
            kwargs["base_url"] = str(base).rstrip("/")
        # Add a conservative request timeout to avoid hanging forever.
        # OpenAI SDK uses httpx under the hood; `timeout` is forwarded.
        kwargs.setdefault("timeout", float(llm.get("request_timeout_sec", 60)))
        self._client = OpenAI(**kwargs)
        self._model = llm.get("default_model", "gpt-4o")
        self._temperature = float(llm.get("temperature", 0.2))
        self._max_tokens = int(llm.get("max_tokens", 4096))

    def complete(self, system: str, user: str, model: str | None = None) -> str:
        mid = model or self._model
        try:
            resp = self._client.chat.completions.create(
                model=mid,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            if isinstance(e, self._AuthenticationError):
                err_text = str(e).upper()
                if "HMAC" in err_text:
                    raise ValueError(
                        "讯飞返回 401（HMAC 不匹配）：OPENAI_API_KEY 必须是控制台里「HTTP 服务接口」的 "
                        "**APIPassword**，不要把 WebSocket 的 APIKey 或 APISecret 单独当密码填。"
                        "若你只有 APIKey + APISecret，请在 .env 设置 IFLYTEK_SPARK_API_KEY 与 "
                        "IFLYTEK_SPARK_API_SECRET（会自动拼成 APIKey:APISecret），并按讯飞文档把 "
                        "OPENAI_BASE_URL / configs 里的 base_url 改成支持该鉴权方式的地址（常为 v2 等）。"
                    ) from e
            raise
        return (resp.choices[0].message.content or "").strip()


class ExecutionRunner:
    def __init__(self, config: dict):
        self.config = config
        self.timeout = int(config.get("execution", {}).get("case_timeout_sec", 15))
        self._llm = _LLMClient(config)

    def run_baseline(self, tasks: list[Task]) -> list[dict[str, Any]]:
        builder = TestSuiteBuilder(self.config)
        suites: dict[str, TestSuite] = builder.build_all(tasks)
        rows: list[dict[str, Any]] = []
        baseline_models = self.config.get("models", {}).get("baseline") or [
            {"name": "primary", "model": self.config["llm"]["default_model"]}
        ]
        for mc in baseline_models:
            model_id = mc.get("model") or self.config["llm"]["default_model"]
            label = mc.get("name", model_id)
            for task in tasks:
                suite = suites[task.task_id]
                code, explanation = self._query_model(task, model_id)
                fn = _func_name(task.function_signature)
                results: list[TestResult] = []
                for case in suite.cases:
                    ok, out, err, runtime_ms = _invoke_in_process(
                        code, fn, case.input, self.timeout
                    )
                    passed = ok and _outputs_equal(out, case.expected_output)
                    results.append(
                        TestResult(
                            test_id=case.test_id,
                            passed=passed,
                            kind=case.kind,
                            expected_output=case.expected_output,
                            actual_output=out if ok else None,
                            error="" if passed else (err or f"got {out!r}"),
                            runtime_ms=runtime_ms,
                            error_type=(
                                "" if passed else (_classify_error(err) if err else "logical_bug")
                            ),
                        )
                    )
                n = len(results)
                passed_n = sum(1 for r in results if r.passed)
                rate = passed_n / n if n else 0.0
                sid = f"{label}_{task.task_id}"
                rows.append(
                    {
                        "sample_id": sid,
                        "task_id": task.task_id,
                        "model": label,
                        "condition": "C0",
                        "code": code,
                        "explanation": explanation,
                        "overall_pass_rate": rate,
                        "test_results": results,
                    }
                )
                logger.info(
                    "Baseline %s %s pass_rate=%.2f", label, task.task_id, rate
                )
        return rows

    def _query_model(self, task: Task, model_id: str) -> tuple[str, str]:
        system = (
            "You are an expert Python programmer. "
            "Respond with exactly one markdown ```python code block containing the full implementation, "
            "then a short natural-language explanation of your solution."
        )
        user = (
            f"Task: {task.title}\n\n{task.description}\n\n"
            f"Required signature (must match exactly):\n{task.function_signature}\n\n"
            "Implement only the function (or class) above. No main guard, no tests."
        )
        # model_id passed for logging; client uses config llm.default_model (讯飞在 YAML / env 配置)
        text = self._llm.complete(system, user, model_id)
        code = _extract_code(text)
        explanation = text if "```" not in text else text.split("```")[-1].strip()
        return code, explanation or "(no explanation)"
