"""動作確認・テスト用の合成 USD/JPY データ。

**実際の相場ではない。** ここから得たバックテスト結果を戦略の評価に使ってはならない。
トレンド局面とレンジ局面を切り替えるランダムウォークで、取引時間は月曜 0 時〜金曜 23 時（サーバー時刻）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data.resample import aggregate


def make_h1(start: str = "2015-01-01", end: str = "2025-12-31", seed: int = 0,
            start_price: float = 120.0, hourly_vol: float = 0.0012) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range(start, end, freq="1h", inclusive="left")
    times = times[times.dayofweek < 5]  # 週末は休場
    n = len(times)

    # 約 2 週間ごとにトレンド/レンジを切り替える
    regime_len = 240
    drift = np.zeros(n)
    for i in range(0, n, regime_len):
        if rng.random() < 0.5:
            drift[i:i + regime_len] = rng.choice([-1, 1]) * hourly_vol * 0.15
    log_ret = drift + rng.standard_t(df=5, size=n) * hourly_vol * np.sqrt(3 / 5)
    # レンジ局面では平均回帰成分を加える
    log_price = np.empty(n)
    lp = np.log(start_price)
    anchor = lp
    for i in range(n):
        if drift[i] == 0:
            log_ret[i] -= 0.02 * (lp - anchor)
        else:
            anchor = lp
        lp += log_ret[i]
        log_price[i] = lp
    close = np.exp(log_price)
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, hourly_vol * 0.6, size=(2, n))) * close
    high = np.maximum(open_, close) + wick[0]
    low = np.minimum(open_, close) - wick[1]
    df = pd.DataFrame({
        "time": times,
        "open": open_.round(3),
        "high": high.round(3),
        "low": low.round(3),
        "close": close.round(3),
        "tick_volume": rng.integers(500, 5000, size=n),
        "spread": rng.integers(2, 12, size=n),
        "real_volume": 0,
    })
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)
    return df


def make_all(start: str = "2015-01-01", end: str = "2025-12-31", seed: int = 0) -> dict[str, pd.DataFrame]:
    h1 = make_h1(start, end, seed)
    out = {"H1": h1}
    for tf in ("H4", "D1"):
        out[tf] = aggregate(h1, tf).drop(columns="n_h1")
    return out
