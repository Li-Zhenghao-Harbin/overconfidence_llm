#!/usr/bin/env python3
"""Train the assertiveness CNN model from annotation JSONL files.

Example:
  python scripts/train_assertiveness_dl.py \
    --inputs data/annotations/mhpp_spark/auto_annotations.jsonl \
    --out models/assertiveness_cnn.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module2_detection.assertiveness_dl import (  # noqa: E402
    _make_model,
    build_vocab,
    collect_training_rows,
    encode_text,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train assertiveness text CNN (PyTorch)")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[str(ROOT / "data/annotations/auto_annotations.jsonl")],
        help="One or more JSONL annotation files with explanation + assertiveness_level",
    )
    ap.add_argument("--out", default=str(ROOT / "models/assertiveness_cnn.pt"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--conv-dim", type=int, default=128)
    ap.add_argument("--max-vocab", type=int, default=8000)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ModuleNotFoundError as e:
        print(
            "Missing dependency: torch.\nInstall with:\n  pip install torch",
            file=sys.stderr,
        )
        raise SystemExit(1) from e

    paths = [Path(p) for p in args.inputs]
    rows = collect_training_rows(paths)
    if not rows:
        print("No valid annotation rows found in inputs.", file=sys.stderr)
        sys.exit(1)

    random.shuffle(rows)
    if len(rows) < 90:
        rows = rows * max(2, 90 // max(1, len(rows)))
    random.shuffle(rows)

    vocab = build_vocab(rows, max_vocab=args.max_vocab, min_freq=args.min_freq)
    if len(vocab) < 8:
        print("Vocabulary too small to train a useful model.", file=sys.stderr)
        sys.exit(1)

    n_train = int(len(rows) * 0.9)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:] or rows[: max(1, len(rows) // 10)]

    device = torch.device(args.device)
    model = _make_model(len(vocab), args.emb_dim, args.conv_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def _to_class(level: int) -> int:
        return int(level) - 1

    def batch_iter(subset: list[tuple[str, int]], shuffle: bool):
        if shuffle:
            random.shuffle(subset)
        for i in range(0, len(subset), args.batch):
            chunk = subset[i : i + args.batch]
            xs = torch.stack([encode_text(text, vocab, args.max_len) for text, _ in chunk]).to(device)
            ys = torch.tensor([_to_class(y) for _, y in chunk], dtype=torch.long, device=device)
            yield xs, ys

    def eval_acc(subset: list[tuple[str, int]]) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xs, ys in batch_iter(subset, shuffle=False):
                logits = model(xs)
                pred = logits.argmax(dim=-1)
                correct += int((pred == ys).sum().item())
                total += ys.numel()
        return correct / total if total else 0.0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xs, ys in batch_iter(train_rows, shuffle=True):
            opt.zero_grad(set_to_none=True)
            logits = model(xs)
            loss = loss_fn(logits, ys)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        acc = eval_acc(val_rows)
        mean_loss = sum(losses) / max(1, len(losses))
        print(f"epoch {epoch + 1}/{args.epochs} loss={mean_loss:.4f} val_acc={acc:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "vocab": vocab,
        "max_len": args.max_len,
        "emb_dim": args.emb_dim,
        "conv_dim": args.conv_dim,
    }
    torch.save(payload, out_path)
    print("Wrote", out_path.resolve())


if __name__ == "__main__":
    main()

