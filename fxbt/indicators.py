"""テクニカル指標。すべて因果的（各時点の値はその時点までのデータだけで決まる）。

RSI・ATR・ADX は Wilder の平滑化（alpha = 1/n）。ウォームアップ期間は NaN を返す。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    out = s.ewm(span=n, adjust=False, min_periods=n).mean()
    return out


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = _wilder(d.clip(lower=0), n)
    loss = _wilder((-d).clip(lower=0), n)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0).where(gain.notna())
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
                   axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _wilder(true_range(df), n)


def adx(df: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr_s = _wilder(true_range(df), n)
    plus_di = 100 * _wilder(plus_dm, n) / tr_s
    minus_di = 100 * _wilder(minus_dm, n) / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return pd.DataFrame({"adx": _wilder(dx, n), "plus_di": plus_di, "minus_di": minus_di})


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def rolling_high(high: pd.Series, n: int) -> pd.Series:
    """直近 n 本（当該足を含む）の最高値。"""
    return high.rolling(n, min_periods=n).max()


def rolling_low(low: pd.Series, n: int) -> pd.Series:
    return low.rolling(n, min_periods=n).min()


def rolling_argmax_offset(x: pd.Series, n: int) -> pd.Series:
    """直近 n 本の最大値が何本前にあったか（0 = 当該足）。ウォームアップ中は NaN。"""
    v = x.to_numpy(dtype=float)
    out = np.full(len(v), np.nan)
    if len(v) >= n:
        w = np.lib.stride_tricks.sliding_window_view(v, n)
        # 同値なら新しい方を採用するため反転して argmax
        out[n - 1:] = np.argmax(w[:, ::-1], axis=1)
    return pd.Series(out, index=x.index)


def rolling_argmin_offset(x: pd.Series, n: int) -> pd.Series:
    return rolling_argmax_offset(-x, n)
