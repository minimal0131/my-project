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
    # 期間別スワップ ((開始日, 買い pips/日, 売り pips/日), ...) 開始日の昇順。空なら上の固定値を使う
    swap_schedule: tuple = ()
    # コスト感応度分析用（既定値では基準と同じ）。スプレッド = max(固定, 足の spread × spread_mult) + spread_add_pips
    spread_add_pips: float = 0.0
    spread_mult: float = 1.0
    swap_mult: float = 1.0

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
            swap_schedule=tuple(sorted((pd.Timestamp(r["from"]), float(r["long"]), float(r["short"]))
                                       for r in (c.get("swap_schedule") or []))),
        )

    def swap_rate(self, direction: int, day: pd.Timestamp) -> float:
        """その日のロールオーバーに適用する 1 日分のスワップ（pips）。"""
        long_, short_ = self.swap_long_pips_per_day, self.swap_short_pips_per_day
        for start, lg, sh in self.swap_schedule:
            if day >= start:
                long_, short_ = lg, sh
            else:
                break
        return long_ if direction > 0 else short_

    @property
    def slippage(self) -> float:
        return self.slippage_pips * self.pip

    def spread_prices(self, bar_spread_points: np.ndarray) -> np.ndarray:
        """各足のスプレッド（価格単位）。MT5 のレートは BID なので ASK = BID + spread。

        足の spread 列はその足の中のティックの最小スプレッド（XM の 2022 年・2026 年のティックで確認）なので、
        足の中の実際のスプレッドより小さいことが多い。
        """
        fixed = np.full(len(bar_spread_points), self.spread_pips * self.pip)
        if self.use_bar_spread:
            fixed = np.maximum(fixed, np.asarray(bar_spread_points, dtype=float) * self.point * self.spread_mult)
        return fixed + self.spread_add_pips * self.pip

    def swap_pips(self, direction: int, entry_time: pd.Timestamp, exit_time: pd.Timestamp) -> float:
        """保有中に通過したサーバー時刻 0 時（ロールオーバー）ごとにスワップを計上。

        ロールオーバーは平日（月〜金）の終わりに 1 回、triple_swap_weekday の日は 3 回分。
        週末をまたぐ分は水曜の 3 倍で計上される（1 週間で 7 日分）。
        """
        total = 0.0
        for day, n in rollover_events(entry_time, exit_time, self.triple_swap_weekday):
            total += n * self.swap_rate(direction, day)
        return total * self.swap_mult


def rollover_events(entry_time: pd.Timestamp, exit_time: pd.Timestamp, triple_weekday: int = 2
                    ) -> list[tuple[pd.Timestamp, int]]:
    """保有中のロールオーバー [(終わった日, 日数), ...]。"""
    first = pd.Timestamp(entry_time).normalize() + pd.Timedelta(days=1)
    last = pd.Timestamp(exit_time).normalize()
    out = []
    if last < first:
        return out
    for midnight in pd.date_range(first, last, freq="D"):
        day = midnight - pd.Timedelta(days=1)
        if day.dayofweek < 5:
            out.append((day, 3 if day.dayofweek == triple_weekday else 1))
    return out


def rollover_days(entry_time: pd.Timestamp, exit_time: pd.Timestamp, triple_weekday: int = 2) -> int:
    return sum(n for _, n in rollover_events(entry_time, exit_time, triple_weekday))
