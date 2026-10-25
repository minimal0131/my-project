"""各戦略の判定テスト。

* シナリオ: 意図した値動きで意図した足にシグナルが出るか
* ルックアヘッド: 未来のデータを消しても / 書き換えても過去のシグナルが変わらないか
* 対称性: 価格を上下反転すると売買が反転するか
* フィルター: 条件を外すとシグナルが増える（減らない）か。判定がトリガー AND フィルターに一致するか
* 重複防止: 同じレンジのブレイクを 2 回取引しないか
"""
import numpy as np
import pandas as pd
import pytest

from fxbt.backtest.costs import CostModel
from fxbt.backtest.engine import run_backtest
from fxbt.config import load_config, strategy_params
from fxbt.mtf import MarketData
from fxbt.strategies import REGISTRY, build
from fxbt.synthetic import make_all
from tests.helpers import bars, market

CFG = load_config("config/default.yaml")
NAMES = list(REGISTRY)


def _params(name, **over):
    return strategy_params(CFG, name, over or None)


@pytest.fixture(scope="module")
def md():
    fr = make_all("2018-01-01", "2019-07-01", seed=11)
    return MarketData(fr["H1"], fr["H4"], fr["D1"])


def _mirror(df: pd.DataFrame, c: float = 240.0) -> pd.DataFrame:
    out = df.copy()
    out["open"], out["close"] = c - df["open"], c - df["close"]
    out["high"], out["low"] = c - df["low"], c - df["high"]
    return out


# ------------------------------------------------------------------ 共通性質
@pytest.mark.parametrize("name", NAMES)
def test_strategy_produces_signals_with_positive_stops(md, name):
    s = build(name, _params(name)).generate(md)
    sig = s[s.signal != 0]
    assert len(sig) > 10, "合成データで 1 件もシグナルが出ないのは判定の不具合の可能性"
    assert (sig.sl_dist > 0).all() and (sig.tp_dist > 0).all()
    assert set(s.signal.unique()) <= {-1, 0, 1}
    assert (sig.signal_id != "").all()


def _condition_frame(strat, md):
    """戦略の全判定材料（トリガー・各フィルター・SL/TP 幅・シグナル）を 1 つの表にする。"""
    c = strat.conditions(md)
    s = strat.generate(md)
    cols = {"long_trigger": c.long_trigger, "short_trigger": c.short_trigger,
            "sl_dist": c.sl_dist, "tp_long": c.tp_dist_long, "tp_short": c.tp_dist_short,
            "signal": s.signal, "signal_id": s.signal_id}
    for f in strat.filter_names:
        cols[f"L_{f}"] = c.long_filters[f]
        cols[f"S_{f}"] = c.short_filters[f]
    out = pd.DataFrame(cols)
    out.insert(0, "time", md.h1["time"].to_numpy())
    return out.reset_index(drop=True)


def _cut_points(md, strat, n=10):
    """切断点: 一様に選んだ足 + 実際にシグナルが出た足（境界での差を確実に検査するため）。"""
    h1 = md.h1
    sig_idx = np.flatnonzero(strat.generate(md).signal.to_numpy() != 0)
    uniform = np.linspace(len(h1) * 0.3, len(h1) - 2, n).astype(int)
    picks = np.concatenate([uniform, sig_idx[len(sig_idx) // 3::max(len(sig_idx) // 6, 1)][:6]])
    hours = np.array([0, 1, 2, 3])  # H4 の形成途中の各位置を含める
    return sorted({int(min(p + h, len(h1) - 2)) for p in picks for h in hours})


def _assert_tail_equal(a: pd.DataFrame, b: pd.DataFrame, cut_time, tail=8):
    a = a[a.time <= cut_time].tail(tail).reset_index(drop=True)
    b = b[b.time <= cut_time].tail(tail).reset_index(drop=True)
    assert len(a) == len(b)
    for col in a.columns:
        if a[col].dtype.kind in "fc":
            np.testing.assert_allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True,
                                       err_msg=f"{col} @ {cut_time}")
        else:
            assert a[col].fillna(False).tolist() == b[col].fillna(False).tolist(), f"{col} @ {cut_time}"


@pytest.mark.parametrize("name", NAMES)
def test_no_lookahead_truncation(md, name):
    """各時点の判定が「その時点で入手可能だったデータだけ」で計算した判定と一致する。"""
    strat = build(name, _params(name))
    full = _condition_frame(strat, md)
    for k in _cut_points(md, strat):
        cut_time = md.h1["time"].iloc[k]
        part = _condition_frame(strat, md.truncate(cut_time))
        _assert_tail_equal(full, part, cut_time)


@pytest.mark.parametrize("name", NAMES)
def test_no_lookahead_future_perturbation(md, name):
    """切断時点で未確定の足（H1/H4/D1）の価格を書き換えても、切断時点以前の判定は不変。"""
    strat = build(name, _params(name))
    full = _condition_frame(strat, md)

    def perturb(df, dur, cutoff):
        df = df.copy()
        fut = df["time"] + dur > cutoff
        for col in ("open", "high", "low", "close"):
            df.loc[fut, col] = df.loc[fut, col] * 1.03
        df.loc[fut, "high"] = df.loc[fut, "high"] * 1.01
        return df

    for k in _cut_points(md, strat, n=5):
        cut_time = md.h1["time"].iloc[k]
        cutoff = cut_time + pd.Timedelta(hours=1)
        md2 = MarketData(perturb(md.h1, pd.Timedelta(hours=1), cutoff),
                         perturb(md.h4, pd.Timedelta(hours=4), cutoff),
                         perturb(md.d1, pd.Timedelta(days=1), cutoff))
        _assert_tail_equal(full, _condition_frame(strat, md2), cut_time, tail=200)


@pytest.mark.parametrize("name", NAMES)
def test_mirror_symmetry(md, name):
    strat = build(name, _params(name))
    a = strat.generate(md)
    m = MarketData(_mirror(md.h1), _mirror(md.h4), _mirror(md.d1))
    b = strat.generate(m)
    mismatch = (a.signal != -b.signal).mean()
    assert mismatch < 0.001, f"上下反転で売買が反転しない足が {mismatch:.3%}"


@pytest.mark.parametrize("name", NAMES)
def test_removing_a_filter_never_removes_signals(md, name):
    strat = build(name, _params(name))
    base = strat.generate(md).signal
    for f in strat.filter_names:
        params = _params(name)
        if not params["filters"].get(f, True):
            continue
        params["filters"][f] = False
        loose = build(name, params).generate(md).signal
        lost = (base != 0) & (loose != base)
        assert not lost.any(), f"{name}: フィルター {f} を外したらシグナルが消えた"


@pytest.mark.parametrize("name", NAMES)
def test_signal_equals_trigger_and_enabled_filters(md, name):
    strat = build(name, _params(name))
    c = strat.conditions(md)
    long_ok = c.long_trigger.fillna(False).astype(bool)
    short_ok = c.short_trigger.fillna(False).astype(bool)
    for f, on in strat.filters.items():
        if on:
            long_ok &= c.long_filters[f].fillna(False).astype(bool)
            short_ok &= c.short_filters[f].fillna(False).astype(bool)
    s = strat.generate(md)
    valid = s.signal != 0
    assert ((s.signal == 1) <= long_ok).all() and ((s.signal == -1) <= short_ok).all()
    # 有効なトリガーで落とされるのは SL/TP が求まらない足（ウォームアップ等）だけ
    dropped = (long_ok ^ short_ok) & ~valid
    tp = c.tp_dist_long.where(long_ok, c.tp_dist_short)
    bad_sl = ~(c.sl_dist > 0)  # NaN も含む
    bad_tp = ~(tp > 0)
    assert (bad_sl | bad_tp)[dropped].all()


@pytest.mark.parametrize("name", ["pullback", "mean_reversion"])
def test_rsi_triggers_are_edges(md, name):
    s = build(name, _params(name)).generate(md).signal
    consecutive = (s != 0) & (s == s.shift(1))
    assert not consecutive.any()


def test_unknown_filter_rejected():
    with pytest.raises(ValueError):
        build("pullback", dict(_params("pullback"), filters={"nope": True}))


# ------------------------------------------------------------------ シナリオ
def _uptrend_with_dip(n_days=80, dip_at=1700, dip_len=12):
    i = np.arange(24 * n_days)
    p = 100 + 0.01 * i + 0.02 * np.sin(i / 3)
    dip = np.zeros(len(i))
    dip[dip_at:dip_at + dip_len] = -0.04 - 0.01  # トレンド分を打ち消して下落
    p = p + np.cumsum(dip)
    return bars(p, start="2021-01-04 00:00", wick=0.01)


def test_pullback_buys_after_dip_in_uptrend():
    h1 = _uptrend_with_dip()
    md = market(h1)
    strat = build("pullback", _params("pullback"))
    s = strat.generate(md)
    longs = np.flatnonzero(s.signal == 1)
    assert (s.signal == -1).sum() == 0, "上昇トレンド中に戻り売りが出ている"
    after_dip = longs[(longs > 1700) & (longs < 1760)]
    assert len(after_dip) >= 1, "押し目の後に買いシグナルが出ていない"
    k = after_dip[0]
    c = strat.conditions(md)
    assert c.long_trigger.iloc[k]  # RSI の上抜け
    assert all(c.long_filters[f].iloc[k] for f, on in strat.filters.items() if on)


def test_pullback_short_in_mirrored_downtrend_and_trend_filter_blocks_longs():
    h1 = _mirror(_uptrend_with_dip())
    s = build("pullback", _params("pullback")).generate(market(h1))
    shorts = np.flatnonzero(s.signal == -1)
    assert ((shorts > 1700) & (shorts < 1760)).any()
    assert (s.signal == 1).sum() == 0, "下降トレンド中に押し目買いが出ている"


def _range_then_breakout(n_days=40, tail=None):
    i = np.arange(24 * n_days)
    p = 100 + 0.2 * np.sin(2 * np.pi * i / 24) + 0.05 * np.sin(2 * np.pi * i / 97)
    if tail is not None:
        p = np.concatenate([p, tail])
    return bars(p, start="2021-01-04 00:00", wick=0.005)


def test_breakout_signal_at_first_close_above_range():
    tail = np.array([100.30, 100.55, 100.80, 100.90])
    h1 = _range_then_breakout(tail=tail)
    md = market(h1)
    strat = build("breakout", _params("breakout"))
    s = strat.generate(md)
    c = strat.conditions(md)
    start = 24 * 40
    longs = np.flatnonzero(s.signal == 1)
    assert len(longs) >= 1 and longs[0] >= start, "レンジ内でブレイクのシグナルが出ている"
    k = longs[0]
    lvl = c.long_trigger  # 上抜けトリガー
    assert lvl.iloc[k] and not lvl.iloc[start:k].any()
    assert (s.signal.iloc[:start] == 0).all()


def test_breakout_same_range_traded_only_once():
    # 同じ H4 足の中で ブレイク -> レンジ内へ戻る -> 再ブレイク
    tail = np.array([100.50, 100.00, 100.50, 100.50])
    h1 = _range_then_breakout(tail=tail)
    md = market(h1)
    s = build("breakout", _params("breakout")).generate(md)
    start = 24 * 40
    idx = np.flatnonzero(s.signal.iloc[start:] == 1) + start
    assert list(idx) == [start, start + 2]
    assert s.signal_id.iloc[start] == s.signal_id.iloc[start + 2], "同じレンジのブレイクで ID が異なる"
    cm = CostModel(pip=0.01, spread_pips=0, slippage_pips=0)
    sig = s.copy()
    sig.loc[:, "sl_dist"] = sig["sl_dist"].where(sig.signal == 0, 0.2)  # 戻りで損切りされる幅
    tr = run_backtest(md.h1, sig, cm, start=md.h1.time.iloc[start - 1])
    assert len(tr) == 1 and tr.iloc[0].exit_reason in ("stop", "stop_gap")


def _range_with_drop(n_days=60, drop_at=1300):
    i = np.arange(24 * n_days)
    p = 100 + 0.15 * np.sin(2 * np.pi * i / 24) + 0.1 * np.sin(2 * np.pi * i / (24 * 5))
    move = np.zeros(len(i))
    move[drop_at:drop_at + 6] = -0.25
    move[drop_at + 6:drop_at + 16] = 0.08
    move[drop_at + 16:drop_at + 26] = -0.08  # 元の水準へ戻す
    return bars(p + np.cumsum(move), start="2021-01-04 00:00", wick=0.01)


def test_mean_reversion_buys_oversold_bounce_in_range():
    h1 = _range_with_drop()
    md = market(h1)
    params = _params("mean_reversion")
    strat = build("mean_reversion", params)
    c = strat.conditions(md)
    s = strat.generate(md)
    longs = np.flatnonzero(s.signal == 1)
    hit = longs[(longs >= 1300) & (longs <= 1320)]
    assert len(hit) >= 1, "急落後の反発で買いシグナルが出ていない"
    k = hit[0]
    assert c.long_trigger.iloc[k]
    # 利確幅 = シグナル時点の H4 SMA までの距離
    assert s.tp_dist.iloc[k] == pytest.approx(c.tp_dist_long.iloc[k])
    assert c.tp_dist_long.iloc[k] > params["min_dev_atr"] * c.sl_dist.iloc[k] / params["sl_atr"] - 1e-12


def test_mean_reversion_blocked_in_strong_trend():
    h1 = _uptrend_with_dip()
    s = build("mean_reversion", _params("mean_reversion")).generate(market(h1))
    longs = np.flatnonzero(s.signal == 1)
    # 強い上昇トレンド中は H4 ADX が高いので押し目でも平均回帰の買いは出ない
    assert not ((longs > 1700) & (longs < 1760)).any()
