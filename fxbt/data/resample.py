"""H1 足から上位足を集計する（時間足間の整合性検査と合成データ生成に使用）。"""
from __future__ import annotations

import pandas as pd

RULE = {"H4": "4h", "D1": "1D"}


def aggregate(h1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """H1 を tf に集計。`time` は集計後の足の開始時刻（サーバー時刻 0 時起点）。"""
    g = h1.set_index("time").resample(RULE[tf], origin="start_day", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "tick_volume": g["tick_volume"].sum(),
        "spread": g["spread"].min(),
        "real_volume": g["real_volume"].sum(),
        "n_h1": g["close"].count(),
    })
    out = out[out["n_h1"] > 0].reset_index()
    return out
