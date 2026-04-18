"""Deep-learning assertiveness classifier (levels 1/2/3).

This module provides:
- training data collection from annotation JSONL files
- a lightweight token-level CNN text classifier (PyTorch)
- runtime predictor with safe fallback support
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

LABELS = (1, 2, 3)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def _normalize_text(text: str) -> str:
    s = (text or "").strip().lower()
    if not s:
        return "(no explanation)"
    return re.sub(r"\s+", " ", s)


def _tokenize(text: str) -> list[str]:
    return _normalize_text(text).split(" ")


def collect_training_rows(jsonl_paths: list[Path]) -> list[tuple[str, int]]:
    """Collect (explanation, label) from JSONL annotation files."""
    out: list[tuple[str, int]] = []
    for path in jsonl_paths:
        if not path.is_file():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                level = int(row.get("assertiveness_level", 0) or 0)
                if level not in LABELS:
                    continue
                text = str(row.get("explanation", "") or "")
                out.append((_normalize_text(text), level))
    return out


def build_vocab(rows: list[tuple[str, int]], max_vocab: int = 8000, min_freq: int = 1) -> dict[str, int]:
    """Build token vocabulary from training rows."""
    counter: Counter[str] = Counter()
    for text, _ in rows:
        counter.update(_tokenize(text))
    vocab_items = [w for w, c in counter.most_common(max(0, int(max_vocab))) if c >= int(min_freq)]
    vocab: dict[str, int] = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1,
    }
    for w in vocab_items:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> Any:
    import torch

    toks = _tokenize(text)[:max_len]
    ids = [vocab.get(t, vocab.get(UNK_TOKEN, 1)) for t in toks]
    if len(ids) < max_len:
        ids.extend([vocab.get(PAD_TOKEN, 0)] * (max_len - len(ids)))
    return torch.tensor(ids[:max_len], dtype=torch.long)


def _make_model(vocab_size: int, emb_dim: int, conv_dim: int) -> Any:
    import torch.nn as nn

    class TokenCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
            self.conv = nn.Conv1d(emb_dim, conv_dim, kernel_size=5, padding=2)
            self.act = nn.ReLU()
            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(conv_dim, 3)

        def forward(self, x: Any) -> Any:
            e = self.emb(x).transpose(1, 2)
            h = self.act(self.conv(e))
            h = self.pool(h).squeeze(-1)
            return self.fc(h)

    return TokenCNN()


class AssertivenessPredictor:
    """Loads trained assertiveness text classifier when available."""

    def __init__(self, config: dict):
        ad = (config.get("assertiveness_dl") or {})
        self.enabled = bool(ad.get("enabled", False))
        self.checkpoint = Path(ad.get("checkpoint", "models/assertiveness_cnn.pt"))
        self.device_s = str(ad.get("device", "cpu"))
        self._max_len = int(ad.get("max_len", 128))
        self._emb_dim = int(ad.get("emb_dim", 64))
        self._conv_dim = int(ad.get("conv_dim", 128))
        self._vocab: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self._model: Any = None
        self._torch: Any = None

        if not self.enabled:
            return
        try:
            import torch

            self._torch = torch
        except ImportError:
            logger.warning("assertiveness_dl.enabled but torch is not installed; using regex fallback")
            return
        if not self.checkpoint.is_file():
            logger.warning(
                "assertiveness_dl.enabled but checkpoint missing at %s — run `python scripts/train_assertiveness_dl.py` "
                "or set assertiveness_dl.enabled: false. Using regex fallback.",
                self.checkpoint.resolve(),
            )
            return
        try:
            torch = self._torch
            try:
                payload = torch.load(self.checkpoint, map_location=self.device_s, weights_only=False)
            except TypeError:
                payload = torch.load(self.checkpoint, map_location=self.device_s)
            self._vocab = dict(payload.get("vocab") or {PAD_TOKEN: 0, UNK_TOKEN: 1})
            self._max_len = int(payload.get("max_len", self._max_len))
            self._emb_dim = int(payload.get("emb_dim", self._emb_dim))
            self._conv_dim = int(payload.get("conv_dim", self._conv_dim))
            model = _make_model(len(self._vocab), self._emb_dim, self._conv_dim)
            model.load_state_dict(payload["state_dict"])
            model.eval()
            model.to(self.device_s)
            self._model = model
            logger.info("Loaded assertiveness DL model from %s", self.checkpoint)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load assertiveness checkpoint: %s — using regex fallback", e)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None and self._torch is not None

    def predict_level(self, explanation: str) -> int:
        if not self.available:
            raise RuntimeError("AssertivenessPredictor model not available")
        import torch

        x = encode_text(explanation, self._vocab, self._max_len).unsqueeze(0).to(self.device_s)
        with torch.no_grad():
            logits = self._model(x)
            idx = int(logits.argmax(dim=-1).item())
        return LABELS[idx]

