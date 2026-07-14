"""Fetch OHLCV history for spot or futures markets via ccxt.

Data is cached as CSV under data/ so repeated training runs (and the Colab
notebook) don't re-download everything.
"""
from __future__ import annotations

import os
import time

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# Exchanges tried in order if the primary one is unreachable. Colab/US IPs
# are geo-blocked by binance.com, bybit and okx, so US-accessible venues
# (kucoin, gate, kraken, coinbase) are included, and fetch_ohlcv falls back
# to Binance's public data mirror (data-api.binance.vision) as a last
# resort — that mirror is not geo-blocked.
SPOT_FALLBACKS = ["binance", "kucoin", "gate", "kraken", "coinbase", "bybit", "okx"]
FUTURES_FALLBACKS = ["binanceusdm", "kucoinfutures", "gate", "bybit", "okx"]

SWAP_EXCHANGES = {"bybit", "okx", "gate", "kucoinfutures"}


def _make_exchange(exchange_id: str, futures: bool):
    import ccxt

    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    if futures and exchange_id in SWAP_EXCHANGES:
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
            ex = _make_exchange(exid, futures)
            # swap venues label perpetuals BTC/USDT:USDT in ccxt
            symbols = [f"{symbol}:USDT", symbol] if futures and exid in SWAP_EXCHANGES else [symbol]
            df = None
            for s in symbols:
                try:
                    df = _paginate(ex, s, timeframe, limit)
                    break
                except Exception as e:  # noqa: BLE001 - try next symbol form
                    last_err = e
            if df is None:
                raise last_err or RuntimeError("no symbol variant worked")
            if cache:
                os.makedirs(DATA_DIR, exist_ok=True)
                df.to_csv(cache_file)
            return df
        except Exception as e:  # noqa: BLE001 - try next venue
            print(f"[data] {exid} failed for {symbol} {timeframe}: {e}")
            last_err = e

    # Last resort: Binance's public spot-data mirror (never geo-blocked).
    try:
        if futures:
            print(f"[data] WARNING: no futures venue reachable for {symbol} — "
                  "using Binance SPOT candles from the public mirror instead "
                  "(perp prices track spot closely, but funding/basis is lost).")
        df = _fetch_binance_vision(symbol, timeframe, limit)
        if cache:
            os.makedirs(DATA_DIR, exist_ok=True)
            df.to_csv(cache_file)
        return df
    except Exception as e:  # noqa: BLE001
        print(f"[data] binance data mirror failed for {symbol} {timeframe}: {e}")
        last_err = e
    raise RuntimeError(f"all data sources failed for {symbol} {timeframe}") from last_err


def _fetch_binance_vision(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch spot klines from data-api.binance.vision (public mirror).

    Unlike api.binance.com this host serves market data worldwide, which
    makes it the reliable path on Google Colab / US IPs.
    """
    import requests

    url = "https://data-api.binance.vision/api/v3/klines"
    market = symbol.replace("/", "")
    tf_ms = timeframe_ms(timeframe)
    end = int(time.time() * 1000)
    since = end - limit * tf_ms
    rows: list[list[float]] = []
    while since < end and len(rows) < limit:
        r = requests.get(
            url,
            params={"symbol": market, "interval": timeframe,
                    "startTime": since, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend([[b[0]] + [float(x) for x in b[1:6]] for b in batch])
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    if not rows:
        raise RuntimeError("no candles returned by data mirror")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df.drop(columns="ts").astype(float)


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
