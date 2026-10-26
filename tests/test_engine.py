import numpy as np
import pandas as pd
import pytest

from fxbt.backtest.costs import CostModel, rollover_days
from fxbt.backtest.engine import run_backtest
from tests.helpers import bars, ohlc, signals_at

NO_COST = CostModel(pip=0.01, spread_pips=0.0, slippage_pips=0.0)


def test_entry_at_next_bar_open_and_take_profit():
    h1 = ohlc([(100, 100.1, 99.9, 100), (100.2, 100.3, 100.1, 100.2),
               (100.2, 100.8, 100.1, 100.7), (100.7, 100.8, 100.6, 100.7)])
    tr = run_backtest(h1, signals_at(h1, {0: 1}, sl=0.5, tp=0.5), NO_COST)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t.entry_time == h1.time[1] and t.entry_price == pytest.approx(100.2)  # 次足の始値
    assert t.exit_reason == "take_profit" and t.exit_price == pytest.approx(100.7)
    assert t.pnl_pips == pytest.approx(50) and t.r_multiple == pytest.approx(1.0)


def test_same_bar_stop_and_target_takes_stop():
    h1 = ohlc([(100, 100, 100, 100), (100, 100.1, 99.9, 100), (100, 101, 99, 100)])
    tr = run_backtest(h1, signals_at(h1, {0: 1}, sl=0.5, tp=0.5), NO_COST)
    assert tr.iloc[0].exit_reason == "stop" and tr.iloc[0].pnl_pips == pytest.approx(-50)


def test_gap_through_stop_fills_at_open():
    h1 = ohlc([(100, 100, 100, 100), (100, 100.1, 99.9, 100), (99.0, 99.2, 98.9, 99.1)])
    tr = run_backtest(h1, signals_at(h1, {0: 1}, sl=0.5, tp=0.5), NO_COST)
    assert tr.iloc[0].exit_reason == "stop_gap" and tr.iloc[0].exit_price == pytest.approx(99.0)
    assert tr.iloc[0].pnl_pips == pytest.approx(-100)


def test_short_uses_ask_for_stops_and_costs():
    cm = CostModel(pip=0.01, spread_pips=1.0, slippage_pips=0.5)
    h1 = ohlc([(100, 100, 100, 100), (100, 100.05, 99.95, 100), (100, 100.05, 99.40, 99.5)])
    tr = run_backtest(h1, signals_at(h1, {0: -1}, sl=0.5, tp=0.5), cm)
    t = tr.iloc[0]
    # 売り約定 = BID - slip = 99.995、利確 = 99.495。ASK の安値 = 99.40 + 0.01 = 99.41 で到達
    assert t.entry_price == pytest.approx(99.995)
    assert t.exit_reason == "take_profit" and t.pnl_pips == pytest.approx(50)
    # 損切り判定は ASK: BID 高値 100.05 + spread 0.01 は 100.495 に届かない -> 損切りされない
    assert t.spread_cost_pips == pytest.approx(1.0)


def test_long_costs_reduce_pnl():
    cm = CostModel(pip=0.01, spread_pips=0.3, slippage_pips=0.2)
    h1 = bars([100.0] * 10)
    tr = run_backtest(h1, signals_at(h1, {0: 1}, sl=0.5, tp=0.5, hold=3), cm)
    t = tr.iloc[0]
    assert t.exit_reason == "time"
    # 価格は不変。コスト = spread 0.3 + 入口 0.2 + 出口 0.2 = 0.7 pips
    assert t.pnl_pips == pytest.approx(-0.7)


def test_max_hold_exits_at_open_after_deadline():
    h1 = bars([100.0] * 20)
    tr = run_backtest(h1, signals_at(h1, {2: 1}, hold=5), NO_COST)
    t = tr.iloc[0]
    assert t.entry_time == h1.time[3] and t.exit_time == h1.time[8] and t.hold_hours == 5


def test_period_end_forces_exit_and_no_future_prices_used():
    h1 = bars([100.0] * 10 + [110.0] * 10)
    tr = run_backtest(h1, signals_at(h1, {2: 1}, sl=5, tp=5), NO_COST, end=h1.time[9])
    t = tr.iloc[0]
    assert t.exit_reason == "period_end" and t.exit_price == pytest.approx(100.0)
    assert t.exit_time <= h1.time[9] + pd.Timedelta(hours=1)


def test_signal_outside_window_or_on_last_bar_ignored():
    h1 = bars([100.0] * 10)
    tr = run_backtest(h1, signals_at(h1, {1: 1, 9: 1}), NO_COST, start=h1.time[2])
    assert len(tr) == 0


def test_no_overlap_and_cooldown():
    h1 = bars([100.0] * 40)
    sig = signals_at(h1, {0: 1, 2: 1, 3: -1, 7: 1, 9: 1, 20: 1}, hold=5)
    tr = run_backtest(h1, sig, NO_COST, cooldown_bars=0)
    # 0 で建て 6 で時間決済。2,3 は保有中で無視。7 で再エントリー -> 13 で決済。9 無視。20 で建てる
    assert list(tr.signal_time.dt.hour) == [0, 7, 20]
    tr = run_backtest(h1, sig, NO_COST, cooldown_bars=5)
    assert list(tr.signal_time.dt.hour) == [0, 20]  # 7, 9 は 6+5=11 本目より前


def test_same_signal_id_is_used_once():
    h1 = bars([100.0] * 40)
    sig = signals_at(h1, {0: 1, 20: 1}, hold=5, ids={0: "level-A", 20: "level-A"})
    assert len(run_backtest(h1, sig, NO_COST)) == 1
    # 保有中に出たシグナルの ID も使用済み
    sig = signals_at(h1, {0: 1, 2: 1, 20: 1}, hold=5, ids={2: "level-B", 20: "level-B"})
    assert len(run_backtest(h1, sig, NO_COST)) == 1


def test_misaligned_signals_rejected():
    h1 = bars([100.0] * 10)
    sig = signals_at(h1, {}).iloc[1:]
    with pytest.raises(ValueError):
        run_backtest(h1, sig, NO_COST)


def test_rollover_days_and_swap():
    # 火曜 10 時 -> 木曜 10 時: 火曜終わり 1 + 水曜終わり 3 = 4
    assert rollover_days(pd.Timestamp("2021-03-02 10:00"), pd.Timestamp("2021-03-04 10:00")) == 4
    # 金曜 10 時 -> 月曜 10 時: 金曜終わり 1 のみ（週末分は水曜に計上済み）
    assert rollover_days(pd.Timestamp("2021-03-05 10:00"), pd.Timestamp("2021-03-08 10:00")) == 1
    assert rollover_days(pd.Timestamp("2021-03-02 01:00"), pd.Timestamp("2021-03-02 23:00")) == 0
    # 1 週間で 7 日分
    assert rollover_days(pd.Timestamp("2021-03-01 12:00"), pd.Timestamp("2021-03-08 12:00")) == 7
    cm = CostModel(swap_long_pips_per_day=1.5, swap_short_pips_per_day=-2.0)
    assert cm.swap_pips(1, pd.Timestamp("2021-03-02 10:00"), pd.Timestamp("2021-03-04 10:00")) == 6.0
    assert cm.swap_pips(-1, pd.Timestamp("2021-03-02 10:00"), pd.Timestamp("2021-03-04 10:00")) == -8.0


def test_swap_included_in_trade_pnl():
    cm = CostModel(pip=0.01, spread_pips=0, slippage_pips=0, swap_long_pips_per_day=1.0)
    h1 = bars([100.0] * 60, start="2021-03-02 00:00")  # 火曜
    tr = run_backtest(h1, signals_at(h1, {0: 1}, sl=5, tp=5, hold=48), cm)
    t = tr.iloc[0]
    assert t.swap_pips == 4.0 and t.pnl_pips == pytest.approx(4.0)


def test_bar_spread_option():
    cm = CostModel(pip=0.01, point=0.001, spread_pips=0.3, use_bar_spread=True)
    assert np.allclose(cm.spread_prices(np.array([1, 10])), [0.003, 0.010])


def test_swap_schedule_by_period():
    cm = CostModel(swap_long_pips_per_day=9.0, swap_short_pips_per_day=-9.0,
                   swap_schedule=((pd.Timestamp("2020-01-01"), 0.5, -1.0), (pd.Timestamp("2021-03-04"), 2.0, -3.0)))
    assert cm.swap_rate(1, pd.Timestamp("2019-06-01")) == 9.0   # 表の開始前は固定値
    assert cm.swap_rate(1, pd.Timestamp("2020-06-01")) == 0.5
    assert cm.swap_rate(-1, pd.Timestamp("2021-03-04")) == -3.0
    # 火曜 10 時 -> 金曜 10 時: 火(1, 0.5) + 水(3, 0.5) + 木 3/4(1, 2.0) = 4.0
    assert cm.swap_pips(1, pd.Timestamp("2021-03-02 10:00"), pd.Timestamp("2021-03-05 10:00")) == pytest.approx(4.0)


def test_swap_schedule_from_config():
    from fxbt.config import load_config
    cm = CostModel.from_config(load_config("config/default.yaml"))
    assert len(cm.swap_schedule) >= 5
    starts = [s for s, _, _ in cm.swap_schedule]
    assert starts == sorted(starts)
    assert cm.swap_rate(1, pd.Timestamp("2023-06-01")) > cm.swap_rate(1, pd.Timestamp("2016-06-01"))
    assert all(sh < 0 for _, _, sh in cm.swap_schedule)
