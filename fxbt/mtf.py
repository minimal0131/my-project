"""上位足の特徴量を H1 に結合する（ルックアヘッド防止の要）。

MT5 の足の時刻は開始時刻。上位足の値は **足が確定した時刻（開始 + 足の長さ）** 以降でなければ使えない。
H1 足 t の判定は t の確定時刻（t + 1 時間）に行うので、
「確定時刻 <= H1 の確定時刻」を満たす最新の上位足だけを結合する。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data.mt5_loader import TF_DURATION


@dataclass
class MarketData:
    h1: pd.DataFrame
    h4: pd.DataFrame
    d1: pd.DataFrame

    def __post_init__(self) -> None:
        for name in ("h1", "h4", "d1"):
            df = getattr(self, name)
            if not df["time"].is_monotonic_increasing:
                raise ValueError(f"{name} の時刻が昇順ではありません")
        self.h1 = self.h1.reset_index(drop=True)

    def truncate(self, last_h1_time: pd.Timestamp) -> "MarketData":
        """last_h1_time の確定時点で入手可能だったデータだけを返す（ルックアヘッド検査用）。"""
        cutoff = pd.Timestamp(last_h1_time) + TF_DURATION["H1"]
        return MarketData(
            self.h1[self.h1["time"] <= last_h1_time],
            self.h4[self.h4["time"] + TF_DURATION["H4"] <= cutoff],
            self.d1[self.d1["time"] + TF_DURATION["D1"] <= cutoff],
        )


def attach(h1: pd.DataFrame, features: pd.DataFrame, tf: str, prefix: str) -> pd.DataFrame:
    """features（`time` 列 = 上位足の開始時刻、その他 = 特徴量）を H1 に結合して返す。

    戻り値は h1 と同じ行数・同じ順序で、特徴量列にのみ prefix を付ける。
    """
    left = pd.DataFrame({"_decision": h1["time"] + TF_DURATION["H1"]})
    right = features.copy()
    right["_available"] = right.pop("time") + TF_DURATION[tf]
    right = right.rename(columns={c: f"{prefix}{c}" for c in right.columns if c != "_available"})
    merged = pd.merge_asof(left, right.sort_values("_available"), left_on="_decision",
                           right_on="_available", direction="backward")
    merged = merged.drop(columns=["_decision", "_available"])
    merged.index = h1.index
    return merged
