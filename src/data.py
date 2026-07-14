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

MIRROR = "binance-data-mirror"


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
    # The Binance data mirror has full history, 1000 candles/call and no
    # geo-block: for spot it's the best source the moment binance.com fails,
    # for futures it's a last resort (it only serves spot prices).
    sources = fallbacks.copy()
    sources.insert(len(sources) if futures else 1, MIRROR)

    last_err: Exception | None = None
    for exid in sources:
        try:
            if exid == MIRROR:
                if futures:
                    print(f"[data] WARNING: no futures venue reachable for {symbol} — "
                          "using Binance SPOT candles from the public mirror instead "
                          "(perp prices track spot closely, but funding/basis is lost).")
                df = _fetch_binance_vision(symbol, timeframe, limit)
            else:
                ex = _make_exchange(exid, futures)
                # swap venues label perpetuals BTC/USDT:USDT in ccxt
                variants = ([f"{symbol}:USDT", symbol]
                            if futures and exid in SWAP_EXCHANGES else [symbol])
                df = None
                for s in variants:
                    try:
                        df = _paginate(ex, s, timeframe, limit)
                        break
                    except Exception as e:  # noqa: BLE001 - try next symbol form
                        last_err = e
                if df is None:
                    raise last_err or RuntimeError("no symbol variant worked")
            if len(df) < limit * 0.8:
                print(f"[data] note: {exid} returned only {len(df)}/{limit} candles "
                      f"for {symbol} {timeframe} (short listing history is normal)")
            if cache:
                os.makedirs(DATA_DIR, exist_ok=True)
                df.to_csv(cache_file)
            return df
        except Exception as e:  # noqa: BLE001 - try next venue
            print(f"[data] {exid} failed for {symbol} {timeframe}: {e}")
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
    return df.drop(columns="ts").astype(float).iloc[-limit:]


def _paginate(ex, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Walk forward from (now - limit candles) to now.

    Exchanges cap candles-per-call at wildly different sizes (binance 1000,
    okx 100-300, kucoin 1500 ...), so a short page must NOT end the loop —
    only reaching the present may. Empty pages (before the pair listed, or
    beyond the venue's history window) are skipped forward in big steps.
    """
    tf_ms = timeframe_ms(timeframe)
    now = ex.milliseconds()
    since = now - limit * tf_ms
    rows: list[list] = []
    while since < now - tf_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if batch:
            rows.extend(batch)
            nxt = batch[-1][0] + tf_ms
            if nxt <= since:  # venue ignored `since`; nothing more to gain
                break
            since = nxt
        else:
            since += 500 * tf_ms  # probe forward for the listing date
        time.sleep(max((ex.rateLimit or 100) / 1000, 0.05))
    if not rows:
        raise RuntimeError("no candles returned")
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "timestamp"
    return df.drop(columns="ts").astype(float).iloc[-limit:]


def load_csv(path: str) -> pd.DataFrame:
    """Load a pre-downloaded OHLCV CSV (timestamp index + ohlcv columns)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].astype(float)
