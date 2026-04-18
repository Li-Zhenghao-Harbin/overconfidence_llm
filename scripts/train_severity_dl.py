#!/usr/bin/env python3
"""Train the byte-CNN severity model from existing JSONL execution logs.

Teacher labels come from `rule_severity` (see `src/module2_detection/severity_dl.py`).

Example:
  python scripts/train_severity_dl.py \\
    --inputs data/processed/baseline_results.jsonl data/processed/strategy_execution_records.jsonl \\
    --out models/severity_cnn.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.module2_detection.severity_dl import (  # noqa: E402
    _make_model,
    build_model_text,
    collect_training_strings,
    encode_bytes,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train error-severity ByteCNN (PyTorch)")
    ap.add_argument(
        "--inputs",
        nargs="+",
        default=[
            str(ROOT / "data/processed/baseline_results.jsonl"),
            str(ROOT / "data/processed/strategy_execution_records.jsonl"),
        ],
        help="One or more JSONL files with test_results on each line",
    )
    ap.add_argument(
        "--out",
        default=str(ROOT / "models/severity_cnn.pt"),
        help="Output checkpoint path",
    )
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--emb-dim", type=int, default=48)
    ap.add_argument("--conv-dim", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    paths = [Path(p) for p in args.inputs]
    rows = collect_training_strings(paths)
    if not rows:
        print("No failed test cases found in inputs — run Phase 2/3 first or pass more JSONL paths.", file=sys.stderr)
        sys.exit(1)

    # Ensure a minimum batch count for tiny datasets
    if len(rows) < 64:
        rows = rows * max(2, 64 // len(rows))

    random.shuffle(rows)
    n_train = int(len(rows) * 0.9)
    train_rows = rows[:n_train]
    val_rows = rows[n_train:] or rows[: max(1, len(rows) // 10)]

    device = torch.device(args.device)
    model = _make_model(args.max_len, args.emb_dim, args.conv_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    def batch_iter(subset: list[tuple[str, str, int]], shuffle: bool):
        if shuffle:
            random.shuffle(subset)
        for i in range(0, len(subset), args.batch):
            chunk = subset[i : i + args.batch]
            xs = torch.stack(
                [encode_bytes(build_model_text(et, er), args.max_len) for et, er, _ in chunk]
            ).to(device)
            ys = torch.tensor([c for _, _, c in chunk], dtype=torch.long, device=device)
            yield xs, ys

    def eval_acc(subset: list[tuple[str, str, int]]) -> float:
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
        print(f"epoch {epoch + 1}/{args.epochs}  loss={sum(losses) / max(1, len(losses)):.4f}  val_acc={acc:.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "max_len": args.max_len,
        "emb_dim": args.emb_dim,
        "conv_dim": args.conv_dim,
    }
    torch.save(payload, out_path)
    print("Wrote", out_path.resolve())


if __name__ == "__main__":
    main()
