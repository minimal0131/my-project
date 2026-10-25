"""戦略の共通部分。

各戦略は「トリガー（イベント）」と「フィルター（状態）」を分けて定義する。
フィルターは設定で個別に on/off でき、アブレーション（1 条件ずつ外して比較）で有効性を検証する。

シグナルは H1 足 t の確定時点の情報だけで決まり、エンジンは t+1 の始値で約定させる。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..mtf import MarketData

SIGNAL_COLUMNS = ["time", "signal", "sl_dist", "tp_dist", "max_hold_hours", "signal_id"]


def cross_above(s: pd.Series, level: float | pd.Series) -> pd.Series:
    level_prev = level.shift(1) if isinstance(level, pd.Series) else level
    return (s.shift(1) < level_prev) & (s >= level)


def cross_below(s: pd.Series, level: float | pd.Series) -> pd.Series:
    level_prev = level.shift(1) if isinstance(level, pd.Series) else level
    return (s.shift(1) > level_prev) & (s <= level)


@dataclass
class Conditions:
    long_trigger: pd.Series
    short_trigger: pd.Series
    long_filters: dict[str, pd.Series] = field(default_factory=dict)
    short_filters: dict[str, pd.Series] = field(default_factory=dict)
    sl_dist: pd.Series | None = None
    tp_dist_long: pd.Series | None = None
    tp_dist_short: pd.Series | None = None
    signal_id_long: pd.Series | None = None
    signal_id_short: pd.Series | None = None


class Strategy(ABC):
    name: str = ""
    filter_names: tuple[str, ...] = ()

    def __init__(self, params: dict):
        self.p = dict(params)
        self.filters = {k: True for k in self.filter_names}
        self.filters.update(params.get("filters") or {})
        unknown = set(self.filters) - set(self.filter_names)
        if unknown:
            raise ValueError(f"{self.name}: 未知のフィルター {sorted(unknown)}")

    @abstractmethod
    def conditions(self, md: MarketData) -> Conditions:
        ...

    def generate(self, md: MarketData) -> pd.DataFrame:
        c = self.conditions(md)
        h1 = md.h1
        long_ok = c.long_trigger.fillna(False).astype(bool)
        short_ok = c.short_trigger.fillna(False).astype(bool)
        for name, active in self.filters.items():
            if active:
                long_ok &= c.long_filters[name].fillna(False).astype(bool)
                short_ok &= c.short_filters[name].fillna(False).astype(bool)

        signal = np.where(long_ok & ~short_ok, 1, np.where(short_ok & ~long_ok, -1, 0))
        tp = np.where(signal == 1, c.tp_dist_long, np.where(signal == -1, c.tp_dist_short, np.nan))
        sl = c.sl_dist.to_numpy(dtype=float)
        # 損切り・利確幅が求まらない足（指標のウォームアップ等）ではシグナルを出さない
        valid = np.isfinite(sl) & (sl > 0) & np.isfinite(tp) & (tp > 0)
        signal = np.where(valid, signal, 0)

        ts = h1["time"].dt.strftime("%Y%m%d%H")
        default_id = self.name + ":" + ts
        id_long = c.signal_id_long if c.signal_id_long is not None else default_id + ":L"
        id_short = c.signal_id_short if c.signal_id_short is not None else default_id + ":S"
        sid = np.where(signal == 1, id_long, np.where(signal == -1, id_short, ""))

        return pd.DataFrame({
            "time": h1["time"],
            "signal": signal.astype(int),
            "sl_dist": np.where(signal != 0, sl, np.nan),
            "tp_dist": np.where(signal != 0, tp, np.nan),
            "max_hold_hours": float(self.p.get("max_hold_hours", 72)),
            "signal_id": sid,
        }, index=h1.index)
