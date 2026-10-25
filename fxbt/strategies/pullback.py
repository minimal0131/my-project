"""戦略 1: トレンド中の押し目買い・戻り売り。

買い（売りは対称）:
  フィルター
    d1_trend          D1 終値 > D1 SMA(d1_sma)
    h4_ema_alignment  H4 EMA(fast) > H4 EMA(slow)
    h4_adx            H4 ADX >= adx_min（トレンドの強さ）
    h4_macd           H4 MACD ヒストグラム > 0（初期設定では off）
  トリガー
    H1 RSI が rsi_long_trigger を下から上抜け（押し目からの反発）。売りは 100 - 値 を上から下抜け。
  損切り / 利確: H4 ATR × sl_atr / tp_atr
"""
from __future__ import annotations

from .. import indicators as ind
from ..mtf import MarketData, attach
from .base import Conditions, Strategy, cross_above, cross_below


class PullbackStrategy(Strategy):
    name = "pullback"
    filter_names = ("d1_trend", "h4_ema_alignment", "h4_adx", "h4_macd")

    def conditions(self, md: MarketData) -> Conditions:
        p = self.p
        d1 = md.d1
        d1f = d1[["time"]].assign(close=d1["close"], sma=ind.sma(d1["close"], p["d1_sma"]))
        h4 = md.h4
        h4f = h4[["time"]].assign(
            ema_fast=ind.ema(h4["close"], p["h4_ema_fast"]),
            ema_slow=ind.ema(h4["close"], p["h4_ema_slow"]),
            adx=ind.adx(h4, p["adx_period"])["adx"],
            macd_hist=ind.macd(h4["close"])["hist"],
            atr=ind.atr(h4, p["atr_period"]),
        )
        h1 = md.h1
        d = attach(h1, d1f, "D1", "d1_")
        f = attach(h1, h4f, "H4", "h4_")
        r = ind.rsi(h1["close"], p["rsi_period"])
        lo = float(p["rsi_long_trigger"])

        atr = f["h4_atr"]
        return Conditions(
            long_trigger=cross_above(r, lo),
            short_trigger=cross_below(r, 100 - lo),
            long_filters={
                "d1_trend": d["d1_close"] > d["d1_sma"],
                "h4_ema_alignment": f["h4_ema_fast"] > f["h4_ema_slow"],
                "h4_adx": f["h4_adx"] >= p["adx_min"],
                "h4_macd": f["h4_macd_hist"] > 0,
            },
            short_filters={
                "d1_trend": d["d1_close"] < d["d1_sma"],
                "h4_ema_alignment": f["h4_ema_fast"] < f["h4_ema_slow"],
                "h4_adx": f["h4_adx"] >= p["adx_min"],
                "h4_macd": f["h4_macd_hist"] < 0,
            },
            sl_dist=atr * p["sl_atr"],
            tp_dist_long=atr * p["tp_atr"],
            tp_dist_short=atr * p["tp_atr"],
        )
