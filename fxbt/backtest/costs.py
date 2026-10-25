"""取引コスト（スプレッド・スリッページ・スワップ）。単位はすべて価格（JPY）または pips。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    pip: float = 0.01
    point: float = 0.001
    spread_pips: float = 0.3
    use_bar_spread: bool = False
    slippage_pips: float = 0.2
    swap_long_pips_per_day: float = 0.0
    swap_short_pips_per_day: float = 0.0
    triple_swap_weekday: int = 2

    @classmethod
    def from_config(cls, cfg: dict) -> "CostModel":
        c = cfg["costs"]
        return cls(
            pip=cfg["pip_size"], point=cfg.get("point_size", cfg["pip_size"] / 10),
            spread_pips=c["spread_pips"], use_bar_spread=c.get("use_bar_spread", False),
            slippage_pips=c["slippage_pips"],
            swap_long_pips_per_day=c.get("swap_long_pips_per_day", 0.0),
            swap_short_pips_per_day=c.get("swap_short_pips_per_day", 0.0),
            triple_swap_weekday=c.get("triple_swap_weekday", 2),
        )

    @property
    def slippage(self) -> float:
        return self.slippage_pips * self.pip

    def spread_prices(self, bar_spread_points: np.ndarray) -> np.ndarray:
        """各足のスプレッド（価格単位）。MT5 のレートは BID なので ASK = BID + spread。"""
        fixed = np.full(len(bar_spread_points), self.spread_pips * self.pip)
        if self.use_bar_spread:
            return np.maximum(fixed, np.asarray(bar_spread_points, dtype=float) * self.point)
        return fixed

    def swap_pips(self, direction: int, entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
        """保有中に通過したサーバー時刻 0 時（ロールオーバー）ごとにスワップを計上。

        ロールオーバーは平日（月〜金）の終わりに 1 回、triple_swap_weekday の日は 3 回分。
        週末をまたぐ分は水曜の 3 倍で計上される（1 週間で 7 日分）。
        """
        per_day = self.swap_long_pips_per_day if direction > 0 else self.swap_short_pips_per_day
        if per_day == 0:
            return 0.0
        return per_day * rollover_days(entry_time, exit_time, self.triple_swap_weekday)


def rollover_days(entry_time: pd.Timestamp, exit_time: pd.Timestamp, triple_weekday: int = 2) -> int:
    first = pd.Timestamp(entry_time).normalize() + pd.Timedelta(days=1)
    last = pd.Timestamp(exit_time).normalize()
    if last < first:
        return 0
    days = 0
    for midnight in pd.date_range(first, last, freq="D"):
        wd = (midnight - pd.Timedelta(days=1)).dayofweek  # 終わった日の曜日
        if wd < 5:
            days += 3 if wd == triple_weekday else 1
    return days
