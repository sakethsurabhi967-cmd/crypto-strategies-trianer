"""Train the EMA 9/21 trend model per (symbol, timeframe) pair.

Designed to run on a Colab GPU (auto-detects CUDA) but works on CPU too.

Usage:
    python -m src.train                          # everything in config.yaml
    python -m src.train --symbols BTC/USDT --timeframes 1h 4h
    python -m src.train --market futures
    python -m src.train --csv data/BTCUSDT_1h_spot.csv --symbols BTC/USDT --timeframes 1h
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config
from .data import fetch_ohlcv, load_csv
from .features import FEATURE_COLUMNS, make_dataset
from .model import EmaTrendNet, save_bundle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling windows: sample i = features[i-seq_len+1 .. i], label[i]."""
    n = len(X) - seq_len + 1
    if n <= 0:
        raise ValueError(f"not enough rows ({len(X)}) for seq_len={seq_len}")
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]
    return X[idx], y[seq_len - 1 :].copy()


def chronological_split(n: int, val_frac: float, test_frac: float) -> tuple[slice, slice, slice]:
    test_n = int(n * test_frac)
    val_n = int(n * val_frac)
    train_n = n - val_n - test_n
    return slice(0, train_n), slice(train_n, train_n + val_n), slice(train_n + val_n, n)


def train_one(
    symbol: str,
    timeframe: str,
    cfg,
    device: str,
    csv_path: str | None = None,
) -> dict:
    futures = cfg.market.type == "futures"
    if csv_path:
        raw = load_csv(csv_path)
    else:
        exid = cfg.exchange.futures_id if futures else cfg.exchange.id
        raw = fetch_ohlcv(
            symbol, timeframe, cfg.market.history_candles, exid, futures
        )
    print(f"[train] {symbol} {timeframe}: {len(raw)} candles "
          f"({raw.index[0]} .. {raw.index[-1]})")

    _, X, y = make_dataset(
        raw,
        cfg.strategy.ema_fast,
        cfg.strategy.ema_slow,
        cfg.strategy.label_horizon,
        cfg.strategy.label_deadband_atr,
    )

    seq_len = cfg.model.seq_len
    Xs, ys = make_sequences(X, y, seq_len)
    tr, va, te = chronological_split(len(Xs), cfg.train.val_fraction, cfg.train.test_fraction)

    # normalise with TRAIN stats only (no lookahead)
    flat = Xs[tr].reshape(-1, Xs.shape[-1])
    mean, std = flat.mean(0), flat.std(0) + 1e-8
    Xs = (Xs - mean) / std

    def loader(sl: slice, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.from_numpy(Xs[sl]), torch.from_numpy(ys[sl]))
        return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=shuffle,
                          drop_last=False)

    train_dl, val_dl, test_dl = loader(tr, True), loader(va, False), loader(te, False)

    # class weights counter the label imbalance (flat usually dominates)
    counts = np.bincount(ys[tr], minlength=3).astype(np.float64)
    weights = torch.tensor((counts.sum() / (3 * np.maximum(counts, 1))), dtype=torch.float32)

    model = EmaTrendNet(
        len(FEATURE_COLUMNS), cfg.model.hidden_size, cfg.model.num_layers, cfg.model.dropout
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    best_val, best_state, patience = float("inf"), None, 0
    for epoch in range(cfg.train.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        sched.step()
        tr_loss /= max(len(train_dl.dataset), 1)

        val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)
        flag = ""
        if val_loss < best_val:
            best_val, patience = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            patience += 1
        print(f"  epoch {epoch + 1:3d}  train {tr_loss:.4f}  "
              f"val {val_loss:.4f}  val_acc {val_acc:.3f}{flag}")
        if patience >= cfg.train.early_stop_patience:
            print("  early stop")
            break

    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_dl, loss_fn, device)
    print(f"[train] {symbol} {timeframe}  TEST loss {test_loss:.4f}  acc {test_acc:.3f}")

    meta = {
        "symbol": symbol,
        "timeframe": timeframe,
        "market_type": cfg.market.type,
        "features": FEATURE_COLUMNS,
        "seq_len": seq_len,
        "hidden_size": cfg.model.hidden_size,
        "num_layers": cfg.model.num_layers,
        "dropout": cfg.model.dropout,
        "ema_fast": cfg.strategy.ema_fast,
        "ema_slow": cfg.strategy.ema_slow,
        "label_horizon": cfg.strategy.label_horizon,
        "label_deadband_atr": cfg.strategy.label_deadband_atr,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "n_train": int(tr.stop),
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    fname = f"{symbol.replace('/', '')}_{timeframe}_{cfg.market.type}.pt"
    path = os.path.join(MODELS_DIR, fname)
    save_bundle(path, model, meta)
    print(f"[train] saved {path}")
    return {"model": fname, **{k: meta[k] for k in ("symbol", "timeframe", "test_acc")}}


@torch.no_grad()
def evaluate(model, dl, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss_sum += loss_fn(logits, yb).item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
        total += len(xb)
    if total == 0:
        return float("inf"), 0.0
    return loss_sum / total, correct / total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    p.add_argument("--market", choices=["spot", "futures"], default=None)
    p.add_argument("--csv", default=None, help="train from a local OHLCV csv instead of fetching")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.market:
        cfg["market"]["type"] = args.market
    symbols = args.symbols or cfg.market.symbols
    timeframes = args.timeframes or cfg.market.timeframes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    results = []
    for sym in symbols:
        for tf in timeframes:
            try:
                results.append(train_one(sym, tf, cfg, device, args.csv))
            except Exception as e:  # noqa: BLE001 - keep training the rest
                print(f"[train] FAILED {sym} {tf}: {e}")

    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[train] done. Summary -> {summary_path}")
    for r in results:
        print(f"  {r['symbol']:10s} {r['timeframe']:4s}  test_acc={r['test_acc']:.3f}  {r['model']}")


if __name__ == "__main__":
    main()
