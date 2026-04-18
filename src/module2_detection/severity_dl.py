"""Deep-learning severity labels for failed test cases (minor / moderate / critical).

Trains a small **byte-level 1-D CNN** on `(error_type + error text)` and predicts one of
`minor | moderate | critical`. Training uses **teacher pseudo-labels** from `rule_severity`
so you can bootstrap from existing `baseline_results.jsonl` / strategy exports without manual
severity annotation.

If `severity_dl.enabled` is true but torch or the checkpoint is missing, the pipeline falls
back to `rule_severity` and logs a warning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

SEVERITY_LABELS = ("minor", "moderate", "critical")


def rule_severity(error_type: str, err: str) -> str:
    """Heuristic severity when no DL model is available."""
    et = (error_type or "").lower()
    e = (err or "").lower()
    if et == "compilation_error" or "syntaxerror" in e or "indentationerror" in e:
        return "critical"
    if et == "timeout":
        return "moderate"
    if et == "api_misuse" or "nameerror" in e or "importerror" in e or "modulenotfounderror" in e:
        return "moderate"
    if "assertionerror" in e or "assertion error" in e:
        return "minor"
    if len(err or "") > 900:
        return "critical"
    return "moderate"


def pseudo_severity_class(error_type: str, err: str) -> int:
    """Integer 0..2 for training (teacher labels)."""
    return SEVERITY_LABELS.index(rule_severity(error_type, err))


def _iter_failed_cases(rows: Iterable[dict[str, Any]]) -> Iterable[tuple[str, str]]:
    for row in rows:
        for tr in row.get("test_results") or []:
            if isinstance(tr, dict):
                passed = tr.get("passed")
                err = str(tr.get("error") or "")
                et = str(tr.get("error_type") or "")
            else:
                passed = getattr(tr, "passed", True)
                err = str(getattr(tr, "error", "") or "")
                et = str(getattr(tr, "error_type", "") or "")
            if passed:
                continue
            yield et, err


def collect_training_strings(jsonl_paths: list[Path]) -> list[tuple[str, str, int]]:
    """Return list of (error_type, error_text, class_index) from JSONL exports."""
    out: list[tuple[str, str, int]] = []
    for path in jsonl_paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows: list[dict[str, Any]] = []
                if "test_results" in row:
                    rows.append(row)
                elif "rounds" in row:
                    for rnd in row.get("rounds") or []:
                        if isinstance(rnd, dict) and rnd.get("test_results"):
                            rows.append(rnd)
                for et, err in _iter_failed_cases(rows):
                    out.append((et, err, pseudo_severity_class(et, err)))
    return out


def build_model_text(error_type: str, err: str) -> str:
    return f"{error_type or 'unknown'}\n{err or ''}"


def _make_model(max_len: int, emb_dim: int, conv_dim: int) -> Any:
    import torch.nn as nn

    class ByteCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(259, emb_dim, padding_idx=0)
            self.conv = nn.Conv1d(emb_dim, conv_dim, kernel_size=5, padding=2)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(conv_dim, 3)

        def forward(self, x: Any) -> Any:
            e = self.emb(x).transpose(1, 2)
            h = self.act(self.conv(e))
            h = self.pool(h).squeeze(-1)
            return self.fc(h)

    return ByteCNN()


def encode_bytes(text: str, max_len: int) -> Any:
    import torch

    raw = text.encode("utf-8", errors="replace")[:max_len]
    ids = [min(256, b) + 1 for b in raw]
    if len(ids) < max_len:
        ids.extend([0] * (max_len - len(ids)))
    return torch.tensor(ids[:max_len], dtype=torch.long)


class SeverityPredictor:
    """Loads a trained ByteCNN checkpoint when available."""

    def __init__(self, config: dict):
        sd = (config.get("severity_dl") or {})
        self.enabled = bool(sd.get("enabled", False))
        self.checkpoint = Path(sd.get("checkpoint", "models/severity_cnn.pt"))
        self.device_s = str(sd.get("device", "cpu"))
        self._model: Any = None
        self._torch: Any = None
        self._max_len = int(sd.get("max_len", 384))
        self._emb_dim = int(sd.get("emb_dim", 48))
        self._conv_dim = int(sd.get("conv_dim", 96))

        if not self.enabled:
            return

        try:
            import torch

            self._torch = torch
        except ImportError:
            logger.warning("severity_dl.enabled but torch is not installed; using rule_severity only")
            return

        if not self.checkpoint.is_file():
            logger.warning(
                "severity_dl.enabled but checkpoint missing at %s — run `python scripts/train_severity_dl.py` "
                "or set severity_dl.enabled: false. Using rule_severity for now.",
                self.checkpoint.resolve(),
            )
            return

        try:
            torch = self._torch
            try:
                payload = torch.load(self.checkpoint, map_location=self.device_s, weights_only=False)
            except TypeError:
                payload = torch.load(self.checkpoint, map_location=self.device_s)
            if isinstance(payload, dict) and "state_dict" in payload:
                self._max_len = int(payload.get("max_len", self._max_len))
                self._emb_dim = int(payload.get("emb_dim", self._emb_dim))
                self._conv_dim = int(payload.get("conv_dim", self._conv_dim))
                model = _make_model(self._max_len, self._emb_dim, self._conv_dim)
                model.load_state_dict(payload["state_dict"])
            else:
                model = _make_model(self._max_len, self._emb_dim, self._conv_dim)
                model.load_state_dict(payload)
            model.eval()
            model.to(self.device_s)
            self._model = model
            logger.info("Loaded severity DL model from %s", self.checkpoint)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load severity checkpoint: %s — using rules", e)
            self._model = None

    def predict_label(self, error_type: str, err: str) -> str:
        if self._model is None or self._torch is None:
            return rule_severity(error_type, err)
        import torch

        text = build_model_text(error_type, err)
        x = encode_bytes(text, self._max_len).unsqueeze(0).to(self.device_s)
        with torch.no_grad():
            logits = self._model(x)
            idx = int(logits.argmax(dim=-1).item())
        return SEVERITY_LABELS[idx]

    def annotate_test_result(self, tr: Any) -> None:
        if getattr(tr, "passed", False):
            tr.severity = ""
            return
        tr.severity = self.predict_label(getattr(tr, "error_type", ""), getattr(tr, "error", ""))


def annotate_test_results_list(config: dict, results: list[Any], predictor: SeverityPredictor | None = None) -> None:
    pred = predictor or SeverityPredictor(config)
    for tr in results:
        pred.annotate_test_result(tr)
