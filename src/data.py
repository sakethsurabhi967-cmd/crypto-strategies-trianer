"""Fetch OHLCV history for spot or futures markets via ccxt.

Data is cached as CSV under data/ so repeated training runs (and the Colab
notebook) don't re-download everything.
"""
from __future__ import annotations

import os
import time

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Exchanges tried in order if the primary one is unreachable
# (Colab IPs are sometimes blocked by binance.com).
SPOT_FALLBACKS = ["binance", "bybit", "okx", "kucoin"]
FUTURES_FALLBACKS = ["binanceusdm", "bybit", "okx"]


def _make_exchange(exchange_id: str, futures: bool):
    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if futures and exchange_id in ("bybit", "okx"):
        ex.options["defaultType"] = "swap"
    return ex


def timeframe_ms(tf: str) -> int:
    units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
    return int(tf[:-1]) * units[tf[-1]]


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 20000,
    exchange_id: str = "binance",
    futures: bool = False,
    cache: bool = True,
) -> pd.DataFrame:
    """Download `limit` most-recent candles, paginating backwards.

    Returns a DataFrame indexed by UTC timestamp with columns
    open/high/low/close/volume.
    """
    market = "fut" if futures else "spot"
    cache_file = os.path.join(
        DATA_DIR, f"{symbol.replace('/', '')}_{timeframe}_{market}.csv"
    )
    if cache and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        if len(df) >= limit * 0.9:  # cache is good enough
            return df.iloc[-limit:]

    fallbacks = [exchange_id] + [
        e for e in (FUTURES_FALLBACKS if futures else SPOT_FALLBACKS) if e != exchange_id
    ]
    last_err: Exception | None = None
    for exid in fallbacks:
        try:
            df = _paginate(_make_exchange(exid, futures), symbol, timeframe, limit)
            if cache:
                os.makedirs(DATA_DIR, exist_ok=True)
                df.to_csv(cache_file)
            return df
        except Exception as e:  # noqa: BLE001 - try next venue
            print(f"[data] {exid} failed for {symbol} {timeframe}: {e}")
            last_err = e
    raise RuntimeError(f"all exchanges failed for {symbol} {timeframe}") from last_err


def _paginate(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    tf_ms = timeframe_ms(timeframe)
    per_call = min(getattr(ex, "ohlcvLimit", 1000) or 1000, 1000)
    since = ex.milliseconds() - limit * tf_ms
    rows: list[list] = []
    while len(rows) < limit:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=per_call)
        if not batch:
            break
        rows.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < per_call:
            break
        time.sleep((ex.rateLimit or 100) / 1000)
    if not rows:
        raise RuntimeError("no candles returned")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df.drop(columns="ts").astype(float)


def load_csv(path: str) -> pd.DataFrame:
    """Load a pre-downloaded OHLCV CSV (timestamp index + ohlcv columns)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].astype(float)
