"""H1 足ベースのバックテストエンジン（1 戦略につき同時に 1 ポジション）。

約定ルール（MT5 のレートは BID。ASK = BID + spread）:
  * シグナルは H1 足 t の確定時点で判定し、t+1 の始値で成行約定
      買い: ASK + スリッページ   売り: BID - スリッページ
  * 損切り・利確は約定価格から sl_dist / tp_dist 離れた価格に置く
      買いは BID で、売りは ASK で到達を判定
  * 各足では次の順に判定する
      1. 始値の時点で最大保有時間を超えていれば始値で成行決済
         （最大保有時間は暦時間で数える。期限が週末・休場にかかれば次の取引可能な足の始値で決済するので、
           記録上の保有時間は上限を超えうる。取引可能な時間で数える仕様ではない）
      2. 始値が損切りを越えて窓開け → 始値で決済（不利な約定）
         始値が利確を越えて窓開け → 利確価格で決済（有利な分は計上しない）
      3. 足の高安で損切り/利確に到達。両方到達した足は損切りを優先（保守的）
      4. 評価期間の最終足で未決済なら終値で強制決済（期間外の価格を使わない）
  * 成行決済（損切り・時間・期間終了）にはスリッページを加算。利確は指値のため加算しない
  * 重複エントリー防止: 保有中のシグナルは無視、決済後 cooldown_bars 本は新規禁止、
    同じ signal_id は一度しか使わない（保有中に出たものも使用済み扱い）
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .costs import CostModel

TRADE_COLUMNS = ["strategy", "signal_id", "direction", "signal_time", "entry_time", "entry_price",
                 "sl_price", "tp_price", "exit_time", "exit_price", "exit_reason", "hold_hours",
                 "sl_pips", "tp_pips", "price_pips", "swap_pips", "pnl_pips", "r_multiple",
                 "spread_cost_pips", "slippage_cost_pips"]


@dataclass
class Trade:
    strategy: str
    signal_id: str
    direction: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    hold_hours: float
    sl_pips: float
    tp_pips: float
    price_pips: float
    swap_pips: float
    pnl_pips: float
    r_multiple: float
    spread_cost_pips: float
    slippage_cost_pips: float


def run_backtest(h1: pd.DataFrame, signals: pd.DataFrame, costs: CostModel, *,
                 start: str | pd.Timestamp | None = None, end: str | pd.Timestamp | None = None,
                 cooldown_bars: int = 0, strategy_name: str = "") -> pd.DataFrame:
    """signals は h1 と同じ行順で `signal`, `sl_dist`, `tp_dist`, `max_hold_hours`, `signal_id` を持つ。

    評価期間 [start, end] 内でシグナルが出て、かつ約定足も期間内のものだけを取引する。
    """
    if len(h1) != len(signals) or not (h1["time"].to_numpy() == signals["time"].to_numpy()).all():
        raise ValueError("signals が h1 と整列していません")

    t = h1["time"].to_numpy()
    o = h1["open"].to_numpy(float)
    hi = h1["high"].to_numpy(float)
    lo = h1["low"].to_numpy(float)
    cl = h1["close"].to_numpy(float)
    spr = costs.spread_prices(h1["spread"].to_numpy() if "spread" in h1 else np.zeros(len(h1)))
    slip = costs.slippage
    pip = costs.pip

    sig = signals["signal"].to_numpy(int)
    sl_d = signals["sl_dist"].to_numpy(float)
    tp_d = signals["tp_dist"].to_numpy(float)
    hold = signals["max_hold_hours"].to_numpy(float)
    sid = signals["signal_id"].to_numpy(object)

    in_window = np.ones(len(t), dtype=bool)
    if start is not None:
        in_window &= t >= np.datetime64(pd.Timestamp(start))
    if end is not None:
        in_window &= t <= np.datetime64(pd.Timestamp(end))
    window_idx = np.flatnonzero(in_window)
    if len(window_idx) == 0:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    last_bar = int(window_idx[-1])

    candidates = np.flatnonzero(in_window & (sig != 0))
    used_ids: set[str] = set()
    trades: list[Trade] = []
    next_allowed = 0  # この足以降のシグナルなら受け付ける

    for i in candidates:
        i = int(i)
        sid_i = str(sid[i])
        if sid_i in used_ids:
            continue
        used_ids.add(sid_i)
        if i < next_allowed or i + 1 > last_bar:
            continue
        d = int(sig[i])
        e = i + 1
        if d > 0:
            entry = o[e] + spr[e] + slip
            sl, tp = entry - sl_d[i], entry + tp_d[i]
        else:
            entry = o[e] - slip
            sl, tp = entry + sl_d[i], entry - tp_d[i]
        deadline = t[e] + np.timedelta64(int(hold[i] * 3600), "s")

        j, exit_px, reason, at_open = _simulate_exit(d, e, last_bar, entry, sl, tp, deadline,
                                                     t, o, hi, lo, cl, spr, slip)
        entry_time = pd.Timestamp(t[e])
        # 始値での決済はその足の開始時刻、足の途中・終値での決済は足の終了時刻で記録する
        exit_time = pd.Timestamp(t[j]) + (pd.Timedelta(0) if at_open else pd.Timedelta(hours=1))
        price_pips = d * (exit_px - entry) / pip
        # スワップは実際に決済した時点までのロールオーバー（サーバー時刻 0 時）で数える。
        # 足の途中・終値での決済は決済足の終了前（23 時の足なら 0 時より前）なので、決済足の開始時刻を使う
        swap = costs.swap_pips(d, entry_time, pd.Timestamp(t[j]))
        pnl = price_pips + swap
        sl_pips = sl_d[i] / pip
        n_market = 2 if reason != "take_profit" else 1
        trades.append(Trade(
            strategy=strategy_name, signal_id=sid_i, direction=d,
            signal_time=pd.Timestamp(t[i]), entry_time=entry_time, entry_price=round(entry, 5),
            sl_price=round(sl, 5), tp_price=round(tp, 5), exit_time=exit_time, exit_price=round(exit_px, 5),
            exit_reason=reason, hold_hours=(exit_time - entry_time) / pd.Timedelta(hours=1),
            sl_pips=sl_pips, tp_pips=tp_d[i] / pip, price_pips=price_pips, swap_pips=swap, pnl_pips=pnl,
            r_multiple=pnl / sl_pips,
            # スプレッドは買いなら約定時（ASK で買う）、売りなら決済時（ASK で買い戻す）に払う
            spread_cost_pips=(spr[e] if d > 0 else spr[j]) / pip, slippage_cost_pips=n_market * costs.slippage_pips,
        ))
        # 決済足 j の確定時点のシグナルは決済後の情報なので利用可（cooldown_bars=0 の場合）
        next_allowed = j + max(cooldown_bars, 0)

    if not trades:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    return pd.DataFrame([asdict(tr) for tr in trades], columns=TRADE_COLUMNS)


def _simulate_exit(d, e, last_bar, entry, sl, tp, deadline, t, o, hi, lo, cl, spr, slip):
    """約定足 e から決済までを 1 本ずつ進める。戻り値: (決済足, 決済価格, 理由, 始値決済か)。"""
    j = e
    while True:
        ask_o = o[j] + spr[j]
        if j > e:  # 約定足の始値は約定価格そのものなので、始値判定は翌足から
            if t[j] >= deadline:
                return j, ((o[j] - slip) if d > 0 else (ask_o + slip)), "time", True
            if d > 0:
                if o[j] <= sl:
                    return j, o[j] - slip, "stop_gap", True
                if o[j] >= tp:
                    return j, tp, "take_profit", True
            else:
                if ask_o >= sl:
                    return j, ask_o + slip, "stop_gap", True
                if ask_o <= tp:
                    return j, tp, "take_profit", True
        if d > 0:
            if lo[j] <= sl:            # 損切り優先（同一足で両方到達しても損切り）
                return j, sl - slip, "stop", False
            if hi[j] >= tp:
                return j, tp, "take_profit", False
        else:
            if hi[j] + spr[j] >= sl:
                return j, sl + slip, "stop", False
            if lo[j] + spr[j] <= tp:
                return j, tp, "take_profit", False
        if j >= last_bar:
            return j, ((cl[j] - slip) if d > 0 else (cl[j] + spr[j] + slip)), "period_end", False
        j += 1
