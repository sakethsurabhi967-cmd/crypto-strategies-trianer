"""EMA 9/21 feature engineering and label construction.

Everything the model sees is derived from OHLCV and is scale-free
(ratios, z-scores, ATR-relative distances) so one model generalises
across price regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1",            # 1-bar log return
    "ret_5",            # 5-bar log return
    "ema_fast_dist",    # (close - ema9) / atr
    "ema_slow_dist",    # (close - ema21) / atr
    "ema_spread",       # (ema9 - ema21) / atr  -> trend strength/direction
    "ema_spread_slope", # change of the spread   -> momentum of the trend
    "cross_up",         # 1 on golden-cross bar (ema9 crosses above ema21)
    "cross_dn",         # 1 on death-cross bar
    "bars_since_cross", # tanh-squashed age of the current EMA regime
    "rsi",              # RSI(14), scaled to [-1, 1]
    "atr_pct",          # ATR / close (volatility regime)
    "vol_z",            # volume z-score over 50 bars
    "hl_range",         # (high - low) / close
    "close_pos",        # where close sits inside the bar's range
]


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["close"].shift()
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame, ema_fast: int = 9, ema_slow: int = 21) -> pd.DataFrame:
    """Return df extended with FEATURE_COLUMNS plus raw ema_fast/ema_slow/atr."""
    out = df.copy()
    close = out["close"]

    out["ema_fast"] = ema(close, ema_fast)
    out["ema_slow"] = ema(close, ema_slow)
    out["atr"] = atr(out)
    a = out["atr"].replace(0, np.nan)

    out["ret_1"] = np.log(close / close.shift(1))
    out["ret_5"] = np.log(close / close.shift(5))
    out["ema_fast_dist"] = (close - out["ema_fast"]) / a
    out["ema_slow_dist"] = (close - out["ema_slow"]) / a
    out["ema_spread"] = (out["ema_fast"] - out["ema_slow"]) / a
    out["ema_spread_slope"] = out["ema_spread"].diff()

    above = out["ema_fast"] > out["ema_slow"]
    prev_above = above.shift(1, fill_value=False)
    out["cross_up"] = (above & ~prev_above).astype(float)
    out["cross_dn"] = (~above & above.shift(1, fill_value=True)).astype(float)

    # bars since the last cross, squashed so it stays bounded
    cross_any = (out["cross_up"] + out["cross_dn"]) > 0
    grp = cross_any.cumsum()
    out["bars_since_cross"] = np.tanh(out.groupby(grp).cumcount() / 20.0)

    out["rsi"] = (rsi(close) - 50.0) / 50.0
    out["atr_pct"] = out["atr"] / close
    vol = out["volume"]
    out["vol_z"] = (vol - vol.rolling(50).mean()) / vol.rolling(50).std().replace(0, np.nan)
    out["hl_range"] = (out["high"] - out["low"]) / close
    rng = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_pos"] = ((close - out["low"]) / rng).fillna(0.5) * 2 - 1

    return out


def build_labels(df: pd.DataFrame, horizon: int = 10, deadband_atr: float = 0.25) -> pd.Series:
    """3-class label from the forward return `horizon` bars ahead.

    0 = down, 1 = flat, 2 = up. The deadband scales with volatility so
    "flat" means "moved less than deadband_atr ATRs".
    """
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    band = deadband_atr * (df["atr"] / df["close"])
    label = pd.Series(1, index=df.index, dtype="int64")
    label[fwd > band] = 2
    label[fwd < -band] = 0
    label[fwd.isna()] = -1  # unusable tail rows
    return label


def make_dataset(
    df: pd.DataFrame,
    ema_fast: int = 9,
    ema_slow: int = 21,
    horizon: int = 10,
    deadband_atr: float = 0.25,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Features + labels, with warmup/tail rows dropped.

    Returns (full feature df, X float32 [n, features], y int64 [n]).
    """
    feats = build_features(df, ema_fast, ema_slow)
    labels = build_labels(feats, horizon, deadband_atr)
    feats = feats.iloc[60:]           # drop indicator warmup
    labels = labels.iloc[60:]
    mask = labels >= 0
    feats, labels = feats[mask], labels[mask]
    X = feats[FEATURE_COLUMNS].astype("float32")
    X = X.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return feats, X.values, labels.values
