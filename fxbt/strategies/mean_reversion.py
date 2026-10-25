"""戦略 3: レンジ相場での平均回帰。

買い（売りは対称）:
  フィルター
    h4_low_adx  H4 ADX < adx_max（レンジ相場）
    d1_low_adx  D1 ADX < d1_adx_max（上位足でもトレンドが弱い）
    deviation   H4 SMA - H1 終値 >= min_dev_atr × H4 ATR（平均から十分に乖離）
  トリガー
    H1 RSI が rsi_oversold を下から上抜け。売りは 100 - 値 を上から下抜け。
  損切り: H4 ATR × sl_atr
  利確:   tp_to_mean=true ならシグナル時点の H4 SMA までの距離（平均への回帰）、
          false なら H4 ATR × tp_atr。価格がすでに平均の反対側にある場合はシグナルを出さない。
"""
from __future__ import annotations

import numpy as np

from .. import indicators as ind
from ..mtf import MarketData, attach
from .base import Conditions, Strategy, cross_above, cross_below


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"
    filter_names = ("h4_low_adx", "d1_low_adx", "deviation")

    def conditions(self, md: MarketData) -> Conditions:
        p = self.p
        h4 = md.h4
        h4f = h4[["time"]].assign(
            sma=ind.sma(h4["close"], p["h4_sma"]),
            adx=ind.adx(h4, p["adx_period"])["adx"],
            atr=ind.atr(h4, p["atr_period"]),
        )
        d1 = md.d1
        d1f = d1[["time"]].assign(adx=ind.adx(d1, p["adx_period"])["adx"])
        h1 = md.h1
        f = attach(h1, h4f, "H4", "h4_")
        d = attach(h1, d1f, "D1", "d1_")
        r = ind.rsi(h1["close"], p["rsi_period"])
        lo = float(p["rsi_oversold"])
        atr = f["h4_atr"]
        dist_long = f["h4_sma"] - h1["close"]   # 平均までの上昇余地
        dist_short = h1["close"] - f["h4_sma"]

        if p.get("tp_to_mean", True):
            tp_long = dist_long.where(dist_long > 0, np.nan)
            tp_short = dist_short.where(dist_short > 0, np.nan)
        else:
            tp_long = tp_short = atr * p["tp_atr"]

        low_adx = f["h4_adx"] < p["adx_max"]
        d1_low = d["d1_adx"] < p["d1_adx_max"]
        return Conditions(
            long_trigger=cross_above(r, lo),
            short_trigger=cross_below(r, 100 - lo),
            long_filters={"h4_low_adx": low_adx, "d1_low_adx": d1_low,
                          "deviation": dist_long >= p["min_dev_atr"] * atr},
            short_filters={"h4_low_adx": low_adx, "d1_low_adx": d1_low,
                           "deviation": dist_short >= p["min_dev_atr"] * atr},
            sl_dist=atr * p["sl_atr"],
            tp_dist_long=tp_long,
            tp_dist_short=tp_short,
        )
