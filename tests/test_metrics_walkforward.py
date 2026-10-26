import copy

import numpy as np
import pandas as pd
import pytest

from fxbt.config import load_config
from fxbt.data.validation import DataValidationError
from fxbt.backtest.costs import CostModel
from fxbt.metrics import (betainc, bootstrap_mean_ci, compute_metrics, cost_breakdown, max_drawdown, t_test_mean,
                          wilson_ci, yearly_breakdown)
from fxbt.mtf import MarketData
from fxbt.pipeline import run
from fxbt.sensitivity import fixed_param_trades, run_sensitivity, scenario_costs
from fxbt.synthetic import make_all
from fxbt.walkforward import StrategyRunner, end_of_day, make_windows, param_grid, walk_forward


def _trades(pnls, sl=10.0, start="2020-01-01"):
    t = pd.date_range(start, periods=len(pnls), freq="D")
    return pd.DataFrame({"entry_time": t, "exit_time": t + pd.Timedelta(hours=5), "direction": 1,
                         "pnl_pips": pnls, "r_multiple": np.asarray(pnls) / sl, "hold_hours": 5.0,
                         "spread_cost_pips": 0.3, "slippage_cost_pips": 0.4, "swap_pips": 0.0})


def test_t_test_against_known_values():
    # t = 2.0, 自由度 10 の両側 p 値は 0.0734
    assert betainc(5, 0.5, 10 / (10 + 4)) == pytest.approx(0.07339, abs=1e-4)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t, p = t_test_mean(x)
    assert t == pytest.approx(3 / (np.std(x, ddof=1) / np.sqrt(5)))
    assert p == pytest.approx(0.01324, abs=1e-4)


def test_wilson_and_bootstrap():
    lo, hi = wilson_ci(90, 100)
    assert lo == pytest.approx(0.8256, abs=1e-3) and hi == pytest.approx(0.9448, abs=1e-3)
    lo, hi = bootstrap_mean_ci(np.r_[np.ones(50), -np.ones(50)], 2000)
    assert lo < 0 < hi


def test_max_drawdown():
    assert max_drawdown(np.array([10, -5, -10, 20, -30, 5])) == 30
    assert max_drawdown(np.array([-5, -5])) == 10
    assert max_drawdown(np.array([])) == 0


def test_compute_metrics_basic():
    m = compute_metrics(_trades([20, -10, 20, -10]), bootstrap_samples=100)
    assert m["n_trades"] == 4 and m["win_rate"] == 0.5
    assert m["avg_win_pips"] == 20 and m["avg_loss_pips"] == -10
    assert m["expectancy_pips"] == 5 and m["expectancy_r"] == 0.5
    assert m["profit_factor"] == 2.0 and m["total_pips"] == 20
    assert m["breakeven_win_rate"] == pytest.approx(1 / 3)
    assert m["max_drawdown_pips"] == 10
    assert m["total_cost_pips"] == pytest.approx(2.8)


def test_high_win_rate_can_have_negative_expectancy():
    # 勝率 90% でも 1 回の損失が大きければ期待値は負
    m = compute_metrics(_trades([5] * 9 + [-60]), bootstrap_samples=0)
    assert m["win_rate"] == 0.9 and m["expectancy_pips"] < 0
    assert m["breakeven_win_rate"] > 0.9


def test_empty_metrics():
    m = compute_metrics(_trades([]).iloc[0:0])
    assert m["n_trades"] == 0 and m["total_pips"] == 0


def test_yearly_breakdown_keeps_empty_years():
    tr = pd.concat([_trades([1, 2], start="2018-01-01"), _trades([3], start="2020-05-01")])
    y = yearly_breakdown(tr)
    assert list(y["year"]) == [2018, 2019, 2020] and list(y["n_trades"]) == [2, 0, 1]


def test_walkforward_windows_are_chronological_and_disjoint():
    ws = make_windows("2016-01-01", "2022-12-31", 24, 6, 6)
    assert len(ws) == 10
    for w in ws:
        assert w.train_start < w.train_end < w.test_start <= w.test_end
        assert w.test_end <= end_of_day("2022-12-31")
    tests = [(w.test_start, w.test_end) for w in ws]
    for (s1, e1), (s2, e2) in zip(tests, tests[1:]):
        assert e1 < s2  # 検証窓は重ならない
    assert ws[0].test_start == pd.Timestamp("2018-01-01")
    assert ws[-1].test_end == end_of_day("2022-12-31")


def test_param_grid():
    g = param_grid({"b": [1, 2], "a": [3]})
    assert g == [{"a": 3, "b": 1}, {"a": 3, "b": 2}]
    assert param_grid(None) == [{}]


# ---------------------------------------------------------------- パイプライン（小さな設定）
@pytest.fixture(scope="module")
def small_setup():
    cfg = copy.deepcopy(load_config("config/default.yaml"))
    cfg["data"].update(start="2016-01-01", end="2020-12-31")
    cfg["periods"].update(development_start="2016-07-01", development_end="2019-06-30",
                          holdout_start="2019-07-01", holdout_end="2020-12-31")
    cfg["walkforward"].update(train_months=12, test_months=6, step_months=6, min_train_trades=5)
    cfg["stats"]["bootstrap_samples"] = 200
    for s in cfg["strategies"].values():
        s["grid"] = {k: v[:2] for k, v in list(s["grid"].items())[:1]}
    frames = make_all("2016-01-01", "2020-12-31", seed=7)
    metas = {tf: {"synthetic": True} for tf in frames}
    return cfg, frames, metas


def test_pipeline_outputs_and_holdout_log(tmp_path, small_setup):
    cfg, frames, metas = small_setup
    out = run(cfg, frames, metas, evaluate_holdout=True, out_root=tmp_path)
    for name in ("pullback", "breakout", "mean_reversion"):
        for f in (f"trades_{name}_development.csv", f"trades_{name}_walkforward_oos.csv",
                  f"trades_{name}_holdout.csv", f"walkforward_windows_{name}.csv", f"ablation_{name}.csv"):
            assert (out / f).exists(), f
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["strategy"]) == {"pullback", "breakout", "mean_reversion"}
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "合成データ" in report and "将来の利益を保証しません" in report

    # ウォークフォワードの取引はすべて検証窓内で約定・決済し、ホールドアウトに入り込まない
    for name in ("pullback", "breakout", "mean_reversion"):
        wf = pd.read_csv(out / f"trades_{name}_walkforward_oos.csv", parse_dates=["entry_time", "exit_time"])
        if len(wf):
            assert wf["exit_time"].max() <= pd.Timestamp("2019-07-01")
        ho = pd.read_csv(out / f"trades_{name}_holdout.csv", parse_dates=["signal_time", "exit_time"])
        if len(ho):
            assert ho["signal_time"].min() >= pd.Timestamp("2019-07-01")
            assert ho["exit_time"].max() <= pd.Timestamp("2021-01-01")

    # 2 回目のホールドアウト評価は記録され、レポートで警告される
    out2 = run(cfg, frames, metas, evaluate_holdout=True, out_root=tmp_path, strategies=["pullback"])
    log = pd.read_csv(tmp_path / "holdout_log.csv")
    assert (log["strategy"] == "pullback").sum() == 2
    assert "過去に 1 回評価されています" in (out2 / "report.md").read_text(encoding="utf-8")


def test_pipeline_stops_on_bad_data(tmp_path, small_setup):
    cfg, frames, metas = small_setup
    bad = dict(frames)
    h1 = frames["H1"].copy()
    h1.loc[1000, "close"] = np.nan
    bad["H1"] = h1
    with pytest.raises(DataValidationError):
        run(cfg, bad, metas, out_root=tmp_path)
    assert not any(tmp_path.glob("run_*")), "異常データで出力が作られてはならない"


def test_cost_breakdown_identity():
    # pnl = price + swap。gross = price + spread + slippage
    tr = _trades([10, -5]).assign(price_pips=[9.0, -5.5], swap_pips=[1.0, 0.5])
    b = cost_breakdown(tr)
    assert b["n_trades"] == 2 and b["gross_pips"] == pytest.approx(3.5 + 1.4)
    assert b["spread_pips"] == pytest.approx(0.6) and b["slippage_pips"] == pytest.approx(0.8)
    assert b["net_pips"] == pytest.approx(b["gross_pips"] - b["spread_pips"] - b["slippage_pips"] + b["swap_pips"])
    assert b["expectancy_pips"] == pytest.approx(2.5)
    assert cost_breakdown(tr.iloc[0:0])["n_trades"] == 0


def test_scenario_costs_change_only_named_fields():
    base = CostModel(pip=0.01, point=0.001, spread_pips=1.2, use_bar_spread=True, slippage_pips=0.3,
                     swap_long_pips_per_day=1.0)
    assert np.allclose(base.spread_prices(np.array([5, 20])), [0.012, 0.020])  # 既定値では従来どおり
    sc = scenario_costs(base, {"name": "x", "basis": "y", "spread_add_pips": 0.5, "spread_mult": 1.5})
    assert np.allclose(sc.spread_prices(np.array([5, 20])), [0.012 + 0.005, 0.030 + 0.005])
    assert sc.slippage_pips == 0.3 and sc.swap_mult == 1.0
    half = scenario_costs(base, {"swap_mult": 0.5})
    t0, t1 = pd.Timestamp("2021-03-02 10:00"), pd.Timestamp("2021-03-04 10:00")
    assert half.swap_pips(1, t0, t1) == pytest.approx(base.swap_pips(1, t0, t1) / 2)
    with pytest.raises(ValueError):
        scenario_costs(base, {"spread_pip": 1.0})  # 綴り間違いは黙って無視しない


def test_cost_sensitivity_base_matches_walkforward_and_stays_in_development(small_setup):
    cfg, frames, _ = small_setup
    md = MarketData(frames["H1"], frames["H4"], frames["D1"])
    runner = StrategyRunner("pullback", cfg, md, CostModel.from_config(cfg))
    wf_table, wf_trades = walk_forward(runner, cfg)
    # 基準コスト・パラメーター固定の再計算はウォークフォワードの取引と完全に一致する
    fixed = fixed_param_trades(runner, wf_table)
    assert len(fixed) == len(wf_trades) > 0
    assert np.allclose(fixed["pnl_pips"], wf_trades["pnl_pips"])
    sens = run_sensitivity(runner, wf_table, cfg)
    base = sens[(sens["scenario"] == "基準") & (sens["method"] == "パラメーター固定")].iloc[0]
    assert base["net_pips"] == pytest.approx(wf_trades["pnl_pips"].sum()) and base["net_vs_base_pips"] == 0
    assert set(sens["method"]) == {"パラメーター固定", "選び直し（参考）"}
    for _, r in sens.iterrows():
        assert r["net_pips"] == pytest.approx(r["gross_pips"] - r["spread_pips"] - r["slippage_pips"] + r["swap_pips"])
    wide = sens[(sens["scenario"] == "スプレッド +1.0") & (sens["method"] == "パラメーター固定")].iloc[0]
    assert wide["spread_pips"] / wide["n_trades"] > base["spread_pips"] / base["n_trades"] + 0.99
    # シナリオの取引も開発期間（検証窓）の中だけ
    wide_trades = fixed_param_trades(StrategyRunner("pullback", cfg, md, scenario_costs(runner.costs, {"spread_add_pips": 1.0}),
                                                    runner._cache), wf_table)
    assert wide_trades["exit_time"].max() <= pd.Timestamp("2019-07-01")


def test_report_shows_broker_and_swap_estimates(tmp_path, small_setup):
    cfg, frames, metas = small_setup
    metas = {tf: {**m, "company": "Example Broker Ltd", "server": "Example-Server", "first_bar": "2016-01-01 00:00:00",
                  "last_bar": "2020-12-31 23:00:00"} for tf, m in metas.items()}
    out = run(cfg, frames, metas, out_root=tmp_path, strategies=["breakout"])
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "業者: Example Broker Ltd（サーバー Example-Server）" in report
    assert "スワップ 2015-01-01 〜 2025-01-31: **推定値**" in report
    assert "コスト感応度のシナリオ（仮定・推定。実績ではありません）" in report
    assert "暦時間" in report and "最小スプレッド" in report
    assert (out / "cost_sensitivity_breakout.csv").exists()
    # 業者の記録が無いときは不明と出す
    out2 = run(cfg, frames, {tf: {"synthetic": True} for tf in frames}, out_root=tmp_path, strategies=["breakout"])
    assert "業者: 不明（取得時の記録なし）" in (out2 / "report.md").read_text(encoding="utf-8")
