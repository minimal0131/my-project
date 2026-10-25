"""MetaTrader5 モジュールの代替（テスト用）。合成データを MT5 と同じ形式で返す。"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from fxbt.synthetic import make_all

_DTYPE = [("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"), ("close", "<f8"),
          ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8")]


def to_mt5_array(df: pd.DataFrame) -> np.ndarray:
    arr = np.zeros(len(df), dtype=_DTYPE)
    arr["time"] = (df["time"].astype("datetime64[s]").astype("int64")).to_numpy()
    for c in ("open", "high", "low", "close", "tick_volume", "spread", "real_volume"):
        arr[c] = df[c].to_numpy()
    return arr


class FakeMT5:
    TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1 = 16385, 16388, 16408
    _TF = {16385: "H1", 16388: "H4", 16408: "D1"}

    def __init__(self, data: dict[str, pd.DataFrame] | None = None, *, fail_init=False,
                 fail_copy_times=0, server_now: pd.Timestamp | None = None, symbols=("USDJPY",)):
        self.data = data if data is not None else make_all("2020-01-01", "2021-01-01")
        self.fail_init = fail_init
        self.fail_copy_times = fail_copy_times
        self.server_now = server_now
        self.symbols = set(symbols)
        self.copy_calls = 0
        self.initialized = False

    def initialize(self, **kwargs):
        self.init_kwargs = kwargs
        self.initialized = not self.fail_init
        return self.initialized

    def shutdown(self):
        self.initialized = False

    def last_error(self):
        return (-1, "fake error")

    def symbol_select(self, symbol, enable):
        return symbol in self.symbols

    def symbol_info_tick(self, symbol):
        if self.server_now is None:
            return None
        return SimpleNamespace(time=int(self.server_now.timestamp()))

    def account_info(self):
        return SimpleNamespace(server="Fake-Server", company="Fake Broker")

    def copy_rates_range(self, symbol, tf, date_from, date_to):
        self.copy_calls += 1
        if self.fail_copy_times > 0:
            self.fail_copy_times -= 1
            return None
        df = self.data[self._TF[tf]]
        lo = pd.Timestamp(date_from).tz_localize(None)
        hi = pd.Timestamp(date_to).tz_localize(None)
        sub = df[(df["time"] >= lo) & (df["time"] <= hi)]
        return to_mt5_array(sub)
