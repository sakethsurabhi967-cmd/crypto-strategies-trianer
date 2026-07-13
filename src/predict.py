"""Predictor: load trained bundles and emit live trading signals.

This is the code you hand the trained models to. It fetches the latest
candles, rebuilds the exact features used in training, and combines the
EMA 9/21 state with the model's probabilities:

    action = LONG   if ema9 > ema21 and P(up)   >= min_confidence
    action = SHORT  if ema9 < ema21 and P(down) >= min_confidence   (futures)
    action = FLAT   otherwise

Usage:
    python -m src.predict                          # one pass over all bundles
    python -m src.predict --symbols BTC/USDT --timeframes 1h
    python -m src.predict --loop 60                # re-check every 60s
    python -m src.predict --json signals.json      # also write signals to file

Signals are printed as JSON — wire them into your own execution layer.
This code does NOT place orders.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import torch

from .config import load_config
from .data import fetch_ohlcv
from .features import FEATURE_COLUMNS, build_features
from .model import load_bundle

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASS_NAMES = ["down", "flat", "up"]


def find_bundles(models_dir: str, symbols=None, timeframes=None) -> list[str]:
    paths = sorted(glob.glob(os.path.join(models_dir, "*.pt")))
    if symbols:
        keys = [s.replace("/", "") for s in symbols]
        paths = [p for p in paths if any(os.path.basename(p).startswith(k + "_") for k in keys)]
    if timeframes:
        paths = [p for p in paths if os.path.basename(p).split("_")[1] in timeframes]
    return paths


@torch.no_grad()
def signal_for_bundle(bundle_path: str, cfg, device: str) -> dict:
    model, meta = load_bundle(bundle_path, device)
    symbol, timeframe = meta["symbol"], meta["timeframe"]
    futures = meta.get("market_type", "spot") == "futures"

    exid = cfg.exchange.futures_id if futures else cfg.exchange.id
    seq_len = meta["seq_len"]
    raw = fetch_ohlcv(symbol, timeframe, limit=seq_len + 120,
                      exchange_id=exid, futures=futures, cache=False)
    # drop the still-forming candle so features match training conditions
    raw = raw.iloc[:-1]

    feats = build_features(raw, meta["ema_fast"], meta["ema_slow"])
    X = feats[FEATURE_COLUMNS].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
    mean = np.asarray(meta["norm_mean"], dtype=np.float32)
    std = np.asarray(meta["norm_std"], dtype=np.float32)
    window = (X[-seq_len:] - mean) / std

    logits = model(torch.from_numpy(window[None]).to(device))
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    last = feats.iloc[-1]
    ema_state = "bullish" if last["ema_fast"] > last["ema_slow"] else "bearish"
    crossed = "golden_cross" if last["cross_up"] else ("death_cross" if last["cross_dn"] else None)

    min_conf = cfg.predict.min_confidence
    action = "FLAT"
    if ema_state == "bullish" and probs[2] >= min_conf:
        action = "LONG"
    elif ema_state == "bearish" and probs[0] >= min_conf and futures:
        action = "SHORT"
    elif ema_state == "bearish" and not futures:
        action = "EXIT/FLAT"

    return {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "symbol": symbol,
        "timeframe": timeframe,
        "market": "futures" if futures else "spot",
        "close": float(last["close"]),
        "ema9": round(float(last["ema_fast"]), 6),
        "ema21": round(float(last["ema_slow"]), 6),
        "ema_state": ema_state,
        "new_cross_this_bar": crossed,
        "prob_down": round(float(probs[0]), 4),
        "prob_flat": round(float(probs[1]), 4),
        "prob_up": round(float(probs[2]), 4),
        "model_view": CLASS_NAMES[int(probs.argmax())],
        "action": action,
        "atr_pct": round(float(last["atr_pct"]) * 100, 3),
        "suggested_stop": round(float(last["close"] - 2 * last["atr"]), 6)
        if action == "LONG"
        else round(float(last["close"] + 2 * last["atr"]), 6)
        if action == "SHORT"
        else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--models-dir", default=MODELS_DIR)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    p.add_argument("--loop", type=int, default=0, metavar="SECONDS",
                   help="poll continuously instead of a single pass")
    p.add_argument("--json", default=None, help="also write latest signals to this file")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bundles = find_bundles(args.models_dir, args.symbols, args.timeframes)
    if not bundles:
        raise SystemExit(
            f"no trained model bundles (*.pt) found in {args.models_dir} — "
            "train first (see notebooks/train_colab.ipynb) and copy the models/ folder here."
        )
    print(f"[predict] {len(bundles)} bundle(s), device={device}")

    while True:
        signals = []
        for b in bundles:
            try:
                sig = signal_for_bundle(b, cfg, device)
                signals.append(sig)
                print(json.dumps(sig))
            except Exception as e:  # noqa: BLE001 - keep the loop alive
                print(f"[predict] error on {os.path.basename(b)}: {e}")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(signals, f, indent=2)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
