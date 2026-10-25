"""期間分割・ウォークフォワード分析・条件別アブレーション・ホールドアウト評価。

手順（過剰適合を防ぐため、評価は常にパラメーター選択に使っていない期間で行う）:
  1. 開発期間を「学習 train_months → 検証 test_months」の窓に区切り、step_months ずつずらす
  2. 各窓で、学習期間だけを使って小さな格子からパラメーターを選ぶ（取引数が min_train_trades 未満の組は不採用）
  3. 選んだパラメーターをその直後の検証期間に適用し、全窓の検証期間の取引を連結して評価する
  4. ホールドアウト（未使用の検証期間）は開発期間全体で選んだパラメーターで 1 回だけ評価する
指標は因果的に全データで計算してよい（各時点の値は過去のみに依存する）。取引は窓内でシグナル・約定し、
窓の最終足で強制決済するので、窓の外の価格が成績に入ることはない。
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

import pandas as pd

from .backtest.costs import CostModel
from .backtest.engine import TRADE_COLUMNS, run_backtest
from .config import strategy_params
from .metrics import compute_metrics
from .mtf import MarketData
from .strategies import build


def end_of_day(d: str | pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(d).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


@dataclass
class Window:
    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_windows(dev_start: str, dev_end: str, train_months: int, test_months: int, step_months: int
                 ) -> list[Window]:
    s0, end = pd.Timestamp(dev_start), end_of_day(dev_end)
    out, k = [], 0
    while True:
        tr_s = s0 + pd.DateOffset(months=k * step_months)
        te_s = tr_s + pd.DateOffset(months=train_months)
        if te_s > end:
            break
        te_e = min(te_s + pd.DateOffset(months=test_months) - pd.Timedelta(seconds=1), end)
        out.append(Window(k, tr_s, te_s - pd.Timedelta(seconds=1), te_s, te_e))
        k += 1
    if not out:
        raise ValueError("開発期間が短すぎてウォークフォワードの窓を作れません")
    return out


def param_grid(grid: dict | None) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


@dataclass
class StrategyRunner:
    """1 戦略分のシグナルをパラメーターごとにキャッシュしてバックテストする。"""
    name: str
    cfg: dict
    md: MarketData
    costs: CostModel
    _cache: dict = field(default_factory=dict)

    def signals(self, overrides: dict) -> pd.DataFrame:
        key = json.dumps(overrides, sort_keys=True, default=str)
        if key not in self._cache:
            params = strategy_params(self.cfg, self.name, overrides)
            self._cache[key] = build(self.name, params).generate(self.md)
        return self._cache[key]

    def backtest(self, overrides: dict, start, end) -> pd.DataFrame:
        return run_backtest(self.md.h1, self.signals(overrides), self.costs, start=start, end=end,
                            cooldown_bars=self.cfg["backtest"].get("cooldown_bars", 0), strategy_name=self.name)


def select_params(runner: StrategyRunner, grid: list[dict], start, end, objective: str, min_trades: int
                  ) -> tuple[dict | None, pd.DataFrame]:
    """学習期間だけでパラメーターを選ぶ。戻り値: (選んだ組 or None, 全候補の成績)。"""
    rows = []
    for combo in grid:
        m = compute_metrics(runner.backtest(combo, start, end), bootstrap_samples=0)
        rows.append({"params": json.dumps(combo, sort_keys=True), **m})
    table = pd.DataFrame(rows)
    ok = table[table["n_trades"] >= min_trades]
    if ok.empty:
        return None, table
    best = ok.sort_values([objective, "n_trades"], ascending=[False, False]).iloc[0]
    return json.loads(best["params"]), table


def walk_forward(runner: StrategyRunner, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """戻り値: (窓ごとの結果, 全検証窓の取引を連結したもの)。"""
    wf, per = cfg["walkforward"], cfg["periods"]
    grid = param_grid(cfg["strategies"][runner.name].get("grid"))
    windows = make_windows(per["development_start"], per["development_end"],
                           wf["train_months"], wf["test_months"], wf["step_months"])
    rows, oos = [], []
    for w in windows:
        chosen, _ = select_params(runner, grid, w.train_start, w.train_end, wf.get("objective", "expectancy_r"),
                                  wf.get("min_train_trades", 30))
        row = {"window": w.index, "train_start": w.train_start.date(), "train_end": w.train_end.date(),
               "test_start": w.test_start.date(), "test_end": w.test_end.date(),
               "chosen_params": json.dumps(chosen, sort_keys=True) if chosen is not None else "(条件を満たす組なし)"}
        if chosen is not None:
            train_m = compute_metrics(runner.backtest(chosen, w.train_start, w.train_end), bootstrap_samples=0)
            test_tr = runner.backtest(chosen, w.test_start, w.test_end).assign(wf_window=w.index)
            test_m = compute_metrics(test_tr, bootstrap_samples=0)
            oos.append(test_tr)
            row.update(train_trades=train_m["n_trades"], train_expectancy_r=train_m["expectancy_r"],
                       test_trades=test_m["n_trades"], test_win_rate=test_m["win_rate"],
                       test_expectancy_pips=test_m["expectancy_pips"], test_expectancy_r=test_m["expectancy_r"],
                       test_total_pips=test_m["total_pips"])
        else:
            row.update(train_trades=0, test_trades=0)
        rows.append(row)
    trades = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame(columns=TRADE_COLUMNS + ["wf_window"])
    return pd.DataFrame(rows), trades


def ablation(runner: StrategyRunner, start, end) -> list[tuple[str, dict, pd.DataFrame]]:
    """条件を 1 つずつ外す/加える比較（開発期間のみ。ホールドアウトは使わない）。"""
    strat = build(runner.name, strategy_params(runner.cfg, runner.name))
    base_filters = dict(strat.filters)
    variants: list[tuple[str, dict]] = [("baseline（設定どおり）", {})]
    for f, on in base_filters.items():
        label = f"- {f} を外す" if on else f"+ {f} を加える"
        variants.append((label, {"filters": {f: not on}}))
    variants.append(("トリガーのみ（フィルターなし）", {"filters": {f: False for f in base_filters}}))
    return [(label, ov, runner.backtest(ov, start, end)) for label, ov in variants]
