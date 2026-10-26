"""成績指標と統計検定。

期待値が 0 より大きいかを、t 検定（両側）と期待値のブートストラップ信頼区間で評価する。
勝率は Wilson の 95% 信頼区間付きで出す。勝率が高くても平均損失が大きければ期待値は負になりうるため、
損益分岐勝率（= 平均損失 / (平均利益 + 平均損失)）も併記する。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

METRIC_COLUMNS = ["n_trades", "n_long", "n_short", "win_rate", "win_rate_ci_low", "win_rate_ci_high",
                  "breakeven_win_rate", "avg_win_pips", "avg_loss_pips", "expectancy_pips", "expectancy_r",
                  "profit_factor", "total_pips", "max_drawdown_pips", "max_drawdown_r", "avg_hold_hours",
                  "total_cost_pips", "total_swap_pips", "t_stat", "p_value", "exp_ci_low", "exp_ci_high"]


# ------------------------------------------------------------ 統計関数
def _betacf(a: float, b: float, x: float) -> float:
    """不完全ベータ関数の連分数（Numerical Recipes）。"""
    max_iter, eps, fpmin = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > fpmin else fpmin)
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > fpmin else fpmin)
        c = 1 + aa / c if abs(1 + aa / c) > fpmin else fpmin
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > fpmin else fpmin)
        c = 1 + aa / c if abs(1 + aa / c) > fpmin else fpmin
        de = d * c
        h *= de
        if abs(de - 1) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """正則化不完全ベータ関数 I_x(a, b)。"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def t_test_mean(x: np.ndarray) -> tuple[float, float]:
    """H0: 平均 = 0 の 1 標本 t 検定。戻り値 (t, 両側 p 値)。"""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    sd = x.std(ddof=1)
    if sd == 0:
        return (float("inf") if x.mean() > 0 else float("-inf") if x.mean() < 0 else 0.0), (0.0 if x.mean() else 1.0)
    t = x.mean() / (sd / math.sqrt(n))
    df = n - 1
    p = betainc(df / 2, 0.5, df / (df + t * t))
    return float(t), float(p)


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def bootstrap_mean_ci(x: np.ndarray, n_samples: int = 5000, alpha: float = 0.05, seed: int = 42
                      ) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or n_samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_samples, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def max_drawdown(pnl_in_exit_order: np.ndarray) -> float:
    """決済順の損益列から、累積損益の最大ドローダウン（正の値）を返す。"""
    if len(pnl_in_exit_order) == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(pnl_in_exit_order)])
    return float((np.maximum.accumulate(equity) - equity).max())


# ------------------------------------------------------------ 成績
def compute_metrics(trades: pd.DataFrame, bootstrap_samples: int = 5000, seed: int = 42) -> dict:
    n = len(trades)
    if n == 0:
        out = {k: float("nan") for k in METRIC_COLUMNS}
        out.update(n_trades=0, n_long=0, n_short=0, total_pips=0.0, max_drawdown_pips=0.0)
        return out
    tr = trades.sort_values("exit_time")
    pnl = tr["pnl_pips"].to_numpy(float)
    r = tr["r_multiple"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    lo, hi = wilson_ci(len(wins), n)
    t, p = t_test_mean(pnl)
    ci = bootstrap_mean_ci(pnl, bootstrap_samples, seed=seed)
    gross_loss = -losses.sum()
    return {
        "n_trades": n,
        "n_long": int((tr["direction"] > 0).sum()),
        "n_short": int((tr["direction"] < 0).sum()),
        "win_rate": len(wins) / n,
        "win_rate_ci_low": lo,
        "win_rate_ci_high": hi,
        "breakeven_win_rate": (-avg_loss / (avg_win - avg_loss)) if (avg_win - avg_loss) > 0 else float("nan"),
        "avg_win_pips": avg_win,
        "avg_loss_pips": avg_loss,
        "expectancy_pips": float(pnl.mean()),
        "expectancy_r": float(r.mean()),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else float("inf"),
        "total_pips": float(pnl.sum()),
        "max_drawdown_pips": max_drawdown(pnl),
        "max_drawdown_r": max_drawdown(r),
        "avg_hold_hours": float(tr["hold_hours"].mean()),
        "total_cost_pips": float((tr["spread_cost_pips"] + tr["slippage_cost_pips"]).sum()),
        "total_swap_pips": float(tr["swap_pips"].sum()),
        "t_stat": t,
        "p_value": p,
        "exp_ci_low": ci[0],
        "exp_ci_high": ci[1],
    }


COST_BREAKDOWN_COLUMNS = ["n_trades", "gross_pips", "spread_pips", "slippage_pips", "swap_pips", "net_pips",
                          "expectancy_pips", "gross_expectancy_pips", "net_minus_gross_pips"]


def cost_breakdown(trades: pd.DataFrame) -> dict:
    """損益の内訳（pips）。net = gross − spread − slippage + swap が取引ごとに成り立つ。

    gross（コスト前）は BID 基準の値幅 = price_pips + スプレッド + スリッページ。
    """
    if trades is None or trades.empty:
        return {k: (0 if k == "n_trades" else float("nan")) for k in COST_BREAKDOWN_COLUMNS}
    n = len(trades)
    spread = float(trades["spread_cost_pips"].sum())
    slip = float(trades["slippage_cost_pips"].sum())
    swap = float(trades["swap_pips"].sum())
    gross = float(trades["price_pips"].sum()) + spread + slip
    net = float(trades["pnl_pips"].sum())
    return {"n_trades": n, "gross_pips": gross, "spread_pips": spread, "slippage_pips": slip, "swap_pips": swap,
            "net_pips": net, "expectancy_pips": net / n, "gross_expectancy_pips": gross / n,
            "net_minus_gross_pips": net - gross}


def yearly_breakdown(trades: pd.DataFrame) -> pd.DataFrame:
    """全年の成績（良い年だけを選ばないよう、取引のない年も 0 件として残す）。"""
    if trades.empty:
        return pd.DataFrame(columns=["year", "n_trades", "win_rate", "expectancy_pips", "total_pips"])
    t = trades.assign(year=pd.to_datetime(trades["entry_time"]).dt.year)
    years = range(int(t["year"].min()), int(t["year"].max()) + 1)
    rows = []
    for y in years:
        g = t[t["year"] == y]
        rows.append({"year": y, "n_trades": len(g),
                     "win_rate": (g["pnl_pips"] > 0).mean() if len(g) else float("nan"),
                     "expectancy_pips": g["pnl_pips"].mean() if len(g) else float("nan"),
                     "total_pips": g["pnl_pips"].sum()})
    return pd.DataFrame(rows)
