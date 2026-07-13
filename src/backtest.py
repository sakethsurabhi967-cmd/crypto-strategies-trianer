"""Backtest the EMA 9/21 crossover strategy and rank timeframes.

Two modes per (symbol, timeframe):
  * raw   – classic crossover: long above golden cross, short (futures)
            or flat (spot) after death cross.
  * model – same entries, but only taken when the trained model agrees
            with at least `min_confidence`.

Usage:
    python -m src.backtest                       # rank all cfg timeframes, raw
    python -m src.backtest --use-model           # requires trained bundles
    python -m src.backtest --symbols BTC/USDT --market futures --use-model
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from .config import load_config
from .data import fetch_ohlcv, timeframe_ms
from .features import FEATURE_COLUMNS, build_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def ema_positions(feats: pd.DataFrame, allow_short: bool) -> pd.Series:
    """Target position per bar from the EMA relationship: +1, 0 or -1."""
    above = feats["ema_fast"] > feats["ema_slow"]
    pos = pd.Series(np.where(above, 1.0, -1.0 if allow_short else 0.0), index=feats.index)
    # trade executes on the bar AFTER the signal candle closes
    return pos.shift(1).fillna(0.0)


def model_gate(feats: pd.DataFrame, bundle_path: str, min_conf: float, device: str) -> pd.Series:
    """Per-bar model probabilities turned into a permission mask.

    Long bars stay long only if P(up) >= min_conf on the most recent
    crossover; shorts likewise with P(down).
    """
    import torch

    from .model import load_bundle

    model, meta = load_bundle(bundle_path, device)
    seq_len = meta["seq_len"]
    mean = np.asarray(meta["norm_mean"], dtype=np.float32)
    std = np.asarray(meta["norm_std"], dtype=np.float32)

    X = feats[FEATURE_COLUMNS].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).values
    X = (X - mean) / std

    n = len(X) - seq_len + 1
    idx = np.arange(seq_len)[None, :] + np.arange(n)[:, None]
    windows = torch.from_numpy(X[idx])

    probs = np.full((len(feats), 3), np.nan, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, 4096):
            batch = windows[i : i + 4096].to(device)
            p = torch.softmax(model(batch), dim=1).cpu().numpy()
            probs[seq_len - 1 + i : seq_len - 1 + i + len(p)] = p

    p_dn = pd.Series(probs[:, 0], index=feats.index)
    p_up = pd.Series(probs[:, 2], index=feats.index)

    regime = (feats["cross_up"] + feats["cross_dn"]).cumsum()
    # confidence measured on the crossover bar, held for the whole regime
    conf_up = p_up.where(feats["cross_up"] > 0).groupby(regime).ffill()
    conf_dn = p_dn.where(feats["cross_dn"] > 0).groupby(regime).ffill()

    gate = pd.Series(0.0, index=feats.index)
    gate[conf_up >= min_conf] = 1.0
    gate[conf_dn >= min_conf] = -1.0
    return gate.shift(1).fillna(0.0)


def run_backtest(
    feats: pd.DataFrame,
    pos: pd.Series,
    fee_pct: float,
    slippage_pct: float,
    timeframe: str,
) -> dict:
    ret = feats["close"].pct_change().fillna(0.0)
    cost = (fee_pct + slippage_pct) / 100.0
    turnover = pos.diff().abs().fillna(pos.abs())
    strat_ret = pos * ret - turnover * cost

    equity = (1 + strat_ret).cumprod()
    total_ret = equity.iloc[-1] - 1.0
    bar_per_year = (365 * 24 * 3_600_000) / timeframe_ms(timeframe)
    vol = strat_ret.std()
    sharpe = float(strat_ret.mean() / vol * np.sqrt(bar_per_year)) if vol > 0 else 0.0
    dd = (equity / equity.cummax() - 1.0).min()

    trades = int((turnover > 0).sum())
    in_market = float((pos != 0).mean())

    # per-trade win rate: group consecutive identical nonzero positions
    trade_id = (pos != pos.shift()).cumsum()
    trade_rets = strat_ret.groupby(trade_id).sum()
    active = trade_rets[pos.groupby(trade_id).first() != 0]
    win_rate = float((active > 0).mean()) if len(active) else 0.0

    return {
        "total_return_pct": round(float(total_ret) * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(float(dd) * 100, 2),
        "trades": trades,
        "win_rate_pct": round(win_rate * 100, 1),
        "time_in_market_pct": round(in_market * 100, 1),
        "bars": len(feats),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--symbols", nargs="+", default=None)
    p.add_argument("--timeframes", nargs="+", default=None)
    p.add_argument("--market", choices=["spot", "futures"], default=None)
    p.add_argument("--use-model", action="store_true",
                   help="filter crossover entries with the trained model")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.market:
        cfg["market"]["type"] = args.market
    futures = cfg.market.type == "futures"
    symbols = args.symbols or cfg.market.symbols
    timeframes = args.timeframes or cfg.market.timeframes

    device = "cpu"
    if args.use_model:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for sym in symbols:
        for tf in timeframes:
            exid = cfg.exchange.futures_id if futures else cfg.exchange.id
            raw = fetch_ohlcv(sym, tf, cfg.market.history_candles, exid, futures)
            feats = build_features(raw, cfg.strategy.ema_fast, cfg.strategy.ema_slow).iloc[60:]

            pos = ema_positions(feats, allow_short=futures)
            mode = "raw"
            if args.use_model:
                bundle = os.path.join(
                    MODELS_DIR, f"{sym.replace('/', '')}_{tf}_{cfg.market.type}.pt"
                )
                if not os.path.exists(bundle):
                    print(f"[backtest] no model bundle for {sym} {tf}, skipping model mode")
                else:
                    gate = model_gate(feats, bundle, cfg.backtest.min_confidence, device)
                    # keep EMA direction, but only while the model agrees
                    pos = pos.where(np.sign(gate) == np.sign(pos), 0.0)
                    mode = "model"

            stats = run_backtest(feats, pos, cfg.backtest.fee_pct,
                                 cfg.backtest.slippage_pct, tf)
            rows.append({"symbol": sym, "timeframe": tf, "mode": mode, **stats})
            print(f"[backtest] {sym} {tf} ({mode}): {stats}")

    df = pd.DataFrame(rows).sort_values(["symbol", "sharpe"], ascending=[True, False])
    print("\n=== Timeframe ranking (best first, by Sharpe) ===")
    print(df.to_string(index=False))

    best = df.groupby("symbol").first().reset_index()
    print("\n=== Best timescale per symbol ===")
    print(best[["symbol", "timeframe", "sharpe", "total_return_pct", "win_rate_pct"]]
          .to_string(index=False))

    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, "backtest_ranking.json")
    with open(out, "w") as f:
        json.dump({"ranking": rows,
                   "best_per_symbol": best.to_dict(orient="records")}, f, indent=2, default=str)
    print(f"\n[backtest] saved ranking -> {out}")


if __name__ == "__main__":
    main()
