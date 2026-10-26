"""コスト感応度分析（開発期間のウォークフォワード検証窓だけを使う。ホールドアウトは使わない）。

戦略・パラメーターは変えず、取引コストだけを変えて再シミュレーションする。スプレッドが変わると
約定価格と損切り・利確の位置も動くため、基準の損益からコストを引き算するのではなく、毎回シミュレーションし直す。

  * パラメーター固定: 基準コストのウォークフォワードで各窓に選ばれたパラメーターをそのまま使う（主な比較）
  * 選び直し（任意）: シナリオのコストで学習期間からパラメーターを選び直したウォークフォワード（参考）

シナリオの値は仮定・推定であり、業者の実績ではない。根拠は設定ファイルの basis に書く。
"""
from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd

from .backtest.costs import CostModel
from .metrics import cost_breakdown
from .walkforward import StrategyRunner, walk_forward

SCENARIO_FIELDS = ("spread_add_pips", "spread_mult", "slippage_pips", "swap_mult")


def scenario_costs(base: CostModel, scenario: dict) -> CostModel:
    unknown = set(scenario) - set(SCENARIO_FIELDS) - {"name", "basis"}
    if unknown:
        raise ValueError(f"コスト感応度のシナリオに不明な項目があります: {sorted(unknown)}")
    return replace(base, **{k: float(scenario[k]) for k in SCENARIO_FIELDS if k in scenario})


def fixed_param_trades(runner: StrategyRunner, wf_table: pd.DataFrame) -> pd.DataFrame:
    """基準のウォークフォワードで選ばれたパラメーターで、各検証窓を runner のコストでバックテストする。"""
    out = []
    for _, w in wf_table.iterrows():
        try:
            chosen = json.loads(w["chosen_params"])
        except (TypeError, ValueError):
            continue  # 条件を満たす組がなかった窓は基準でも取引なし
        out.append(runner.backtest(chosen, pd.Timestamp(w["test_start"]),
                                   pd.Timestamp(w["test_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_sensitivity(base_runner: StrategyRunner, wf_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """戦略 1 つ分のシナリオ別内訳（1 行 = 1 シナリオ × 方式）。"""
    sc_cfg = cfg.get("cost_sensitivity") or {}
    scenarios = [{"name": "基準", "basis": "現行の設定値"}] + list(sc_cfg.get("scenarios") or [])
    reselect = bool(sc_cfg.get("reselect_params", False))
    rows = []
    for sc in scenarios:
        costs = scenario_costs(base_runner.costs, sc)
        # シグナルはコストに依存しないので、基準のキャッシュを共有する
        runner = StrategyRunner(base_runner.name, base_runner.cfg, base_runner.md, costs, base_runner._cache)
        label = {"scenario": sc["name"], "basis": sc.get("basis", ""),
                 **{k: getattr(costs, k) for k in SCENARIO_FIELDS}}
        rows.append({**label, "method": "パラメーター固定", **cost_breakdown(fixed_param_trades(runner, wf_table))})
        if reselect:
            _, tr = walk_forward(runner, cfg)
            rows.append({**label, "method": "選び直し（参考）", **cost_breakdown(tr)})
    df = pd.DataFrame(rows)
    base = df[(df["scenario"] == "基準")].set_index("method")["net_pips"]
    df["net_vs_base_pips"] = df["net_pips"] - df["method"].map(base)
    return df
