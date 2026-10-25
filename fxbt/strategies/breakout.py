"""戦略 2: レンジブレイクアウト。

レンジ = 確定済み H4 の直近 range_bars_h4 本の最高値・最安値。
買い（売りは対称）:
  フィルター
    range_compression  (レンジ高値 - 安値) / H4 ATR <= max_range_atr（保ち合い）
    low_adx_before     H4 ADX <= adx_max_before（ブレイク前はトレンドが弱い）
    d1_trend           D1 終値 > D1 SMA(d1_sma)（初期設定では off）
  トリガー
    H1 終値がレンジ高値 + buffer_atr × ATR を下から上抜け
  重複防止
    signal_id はレンジ高値を作った H4 足の時刻。同じ高値のブレイクには 1 回しか入らない
    （だましで戻った後の再ブレイクでは再エントリーしない）。
  損切り / 利確: H4 ATR × sl_atr / tp_atr
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import indicators as ind
from ..mtf import MarketData, attach
from .base import Conditions, Strategy, cross_above, cross_below


class BreakoutStrategy(Strategy):
    name = "breakout"
    filter_names = ("range_compression", "low_adx_before", "d1_trend")

    def conditions(self, md: MarketData) -> Conditions:
        p = self.p
        n = int(p["range_bars_h4"])
        h4 = md.h4.reset_index(drop=True)
        hi_off = ind.rolling_argmax_offset(h4["high"], n)
        lo_off = ind.rolling_argmin_offset(h4["low"], n)
        idx = np.arange(len(h4))
        times = h4["time"].to_numpy()

        def _src_time(off: pd.Series) -> pd.Series:
            src = idx - off.fillna(0).astype(int).to_numpy()
            return pd.Series(np.where(off.notna(), pd.to_datetime(times[src]).strftime("%Y%m%d%H"), ""))

        h4f = h4[["time"]].assign(
            hi=ind.rolling_high(h4["high"], n),
            lo=ind.rolling_low(h4["low"], n),
            hi_src=_src_time(hi_off),
            lo_src=_src_time(lo_off),
            atr=ind.atr(h4, p["atr_period"]),
            adx=ind.adx(h4, p["adx_period"])["adx"],
        )
        d1 = md.d1
        d1f = d1[["time"]].assign(close=d1["close"], sma=ind.sma(d1["close"], p["d1_sma"]))

        h1 = md.h1
        f = attach(h1, h4f, "H4", "h4_")
        d = attach(h1, d1f, "D1", "d1_")
        atr = f["h4_atr"]
        up_level = f["h4_hi"] + p["buffer_atr"] * atr
        dn_level = f["h4_lo"] - p["buffer_atr"] * atr
        width_ok = (f["h4_hi"] - f["h4_lo"]) / atr <= p["max_range_atr"]
        adx_ok = f["h4_adx"] <= p["adx_max_before"]
        return Conditions(
            long_trigger=cross_above(h1["close"], up_level),
            short_trigger=cross_below(h1["close"], dn_level),
            long_filters={"range_compression": width_ok, "low_adx_before": adx_ok,
                          "d1_trend": d["d1_close"] > d["d1_sma"]},
            short_filters={"range_compression": width_ok, "low_adx_before": adx_ok,
                           "d1_trend": d["d1_close"] < d["d1_sma"]},
            sl_dist=atr * p["sl_atr"],
            tp_dist_long=atr * p["tp_atr"],
            tp_dist_short=atr * p["tp_atr"],
            signal_id_long=self.name + ":L:" + f["h4_hi_src"].astype(str) + ":" + f["h4_hi"].round(3).astype(str),
            signal_id_short=self.name + ":S:" + f["h4_lo_src"].astype(str) + ":" + f["h4_lo"].round(3).astype(str),
        )
