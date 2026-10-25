"""テスト用の小さなデータ作成ヘルパー。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from fxbt.data.resample import aggregate
from fxbt.mtf import MarketData


def bars(prices, start="2021-03-01 00:00", spread=0, wick=0.0):
    """終値列から H1 足を作る（始値 = 前の終値、高安 = 始値/終値 ± wick）。"""
    close = np.asarray(prices, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    t = pd.date_range(start, periods=len(close), freq="1h")
    return pd.DataFrame({
        "time": t, "open": open_, "high": np.maximum(open_, close) + wick,
        "low": np.minimum(open_, close) - wick, "close": close,
        "tick_volume": 100, "spread": spread, "real_volume": 0,
    })


def ohlc(rows, start="2021-03-01 00:00"):
    """[(o, h, l, c), ...] から H1 足を作る。"""
    a = np.asarray(rows, dtype=float)
    t = pd.date_range(start, periods=len(a), freq="1h")
    return pd.DataFrame({"time": t, "open": a[:, 0], "high": a[:, 1], "low": a[:, 2], "close": a[:, 3],
                         "tick_volume": 100, "spread": 0, "real_volume": 0})


def market(h1: pd.DataFrame) -> MarketData:
    return MarketData(h1, aggregate(h1, "H4").drop(columns="n_h1"), aggregate(h1, "D1").drop(columns="n_h1"))


def signals_at(h1: pd.DataFrame, entries: dict[int, int], sl=0.5, tp=0.5, hold=72.0, ids=None):
    sig = np.zeros(len(h1), dtype=int)
    for i, d in entries.items():
        sig[i] = d
    sid = [f"s{i}" if sig[i] else "" for i in range(len(h1))]
    if ids:
        for i, v in ids.items():
            sid[i] = v
    return pd.DataFrame({"time": h1["time"], "signal": sig, "sl_dist": np.where(sig != 0, sl, np.nan),
                         "tp_dist": np.where(sig != 0, tp, np.nan), "max_hold_hours": hold, "signal_id": sid})
