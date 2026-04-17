"""Subprocess entry for MHPP smoke execution (stdin JSON → stdout JSON).

Invoked as ``python -m src.module2_detection.mhpp_exec_worker`` with payload::

    {"program": str, "entry_point": str}

``program`` is merged prompt + model completion (see ``_merge_humaneval_program``).
"""

from __future__ import annotations

import inspect
import json
import sys
import traceback
import types
import typing
from typing import Any, get_args, get_origin


def _coerce_default(ann: Any) -> Any:
    if ann is None or ann is inspect.Parameter.empty:
        return None
    if ann is str:
        return ""
    if ann is int:
        return 0
    if ann is float:
        return 0.0
    if ann is bool:
        return False
    origin = get_origin(ann)
    if origin is list:
        return []
    if origin is dict:
        return {}
    if origin is tuple:
        return ()
    union_type = getattr(types, "UnionType", None)
    if origin is typing.Union or (union_type is not None and origin is union_type):
        for sub in get_args(ann):
            if sub is type(None):
                continue
            v = _coerce_default(sub)
            if sub is str:
                return ""
            if sub is int:
                return 0
            if v is not None:
                return v
        return None
    return None


def _build_kwargs(fn: Any) -> dict[str, Any]:
    hints: dict[str, Any] = {}
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        pass
    sig = inspect.signature(fn)
    kw: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(name, param.annotation)
        kw[name] = _coerce_default(ann)
    return kw


def _try_call(fn: Any, kw: dict[str, Any]) -> None:
    fn(**kw)


def main() -> None:
    p = json.loads(sys.stdin.read())
    program = str(p["program"])
    entry = str(p["entry_point"])
    trace = bool(p.get("trace"))
    max_events = int(p.get("max_events", 200))
    events: list[dict[str, Any]] = []

    def _safe_repr(x: Any, limit: int = 200) -> str:
        try:
            s = repr(x)
        except Exception:
            s = "<unrepr>"
        return s if len(s) <= limit else s[:limit] + "…"

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if frame.f_code.co_filename != "<mhpp_prog>":
            return tracer
        if frame.f_code.co_name != entry:
            return tracer
        if event == "line":
            if len(events) < max_events:
                loc = {
                    k: _safe_repr(v)
                    for k, v in frame.f_locals.items()
                    if not k.startswith("__")
                }
                events.append({"line": frame.f_lineno, "locals": loc})
            return tracer
        return tracer

    t0 = __import__("time").perf_counter()
    try:
        g: dict[str, Any] = {}
        exec(compile(program, "<mhpp_prog>", "exec"), g, g)
        fn = g[entry]
        if not callable(fn):
            raise TypeError(f"{entry!r} is not callable")
        kw = _build_kwargs(fn)
        last_err: BaseException | None = None
        for attempt in (
            kw,
            {k: None for k in kw},
            {k: "" for k in kw},
            {k: 0 for k in kw},
        ):
            if not attempt:
                continue
            try:
                if trace:
                    tr = sys.gettrace()
                    sys.settrace(tracer)
                    try:
                        _try_call(fn, attempt)
                    finally:
                        sys.settrace(tr)
                else:
                    _try_call(fn, attempt)
                last_err = None
                break
            except TypeError as e:
                last_err = e
                continue
            except BaseException:
                raise
        if last_err is not None:
            raise last_err
        t1 = __import__("time").perf_counter()
        out = {"ok": True, "out": True, "runtime_ms": (t1 - t0) * 1000.0}
        if trace:
            out["trace"] = events
        json.dump(out, sys.stdout, default=str)
    except Exception:
        t1 = __import__("time").perf_counter()
        json.dump(
            {
                "ok": False,
                "err": traceback.format_exc(),
                "runtime_ms": (t1 - t0) * 1000.0,
                "trace": events if trace else [],
            },
            sys.stdout,
            default=str,
        )


if __name__ == "__main__":
    main()
