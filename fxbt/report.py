"""CSV と Markdown レポートの出力。"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

LABELS = {
    "n_trades": "取引回数", "win_rate": "勝率", "breakeven_win_rate": "損益分岐勝率",
    "avg_win_pips": "平均利益(pips)", "avg_loss_pips": "平均損失(pips)", "expectancy_pips": "期待値(pips)",
    "expectancy_r": "期待値(R)", "profit_factor": "PF", "total_pips": "合計(pips)",
    "max_drawdown_pips": "最大DD(pips)", "avg_hold_hours": "平均保有(h)", "p_value": "p値",
    "year": "年", "window": "窓",
}
SUMMARY_KEYS = ["n_trades", "win_rate", "breakeven_win_rate", "avg_win_pips", "avg_loss_pips",
                "expectancy_pips", "expectancy_r", "profit_factor", "total_pips", "max_drawdown_pips",
                "avg_hold_hours", "p_value"]


def fmt(v, key: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    if isinstance(v, float) and math.isinf(v):
        return "∞"
    if "win_rate" in key or key in ("ci",):
        return f"{v:.1%}"
    if key.endswith(("n_trades", "n_long", "n_short", "train_trades", "test_trades")) or key in ("window", "year"):
        return f"{int(v)}"
    if key.startswith("p_") or key.endswith("p_value") or key.endswith("p_adj"):
        return f"{v:.4f}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def md_table(df: pd.DataFrame, labels: dict | None = None) -> str:
    if df.empty:
        return "(データなし)\n"
    labels = labels or {}
    cols = list(df.columns)
    head = "| " + " | ".join(labels.get(c, c) for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(row[c], c) for c in cols) + " |" for _, row in df.iterrows()]
    return "\n".join([head, sep, *body]) + "\n"


def metrics_row(label: str, m: dict) -> dict:
    return {"区分": label, **{k: m.get(k) for k in SUMMARY_KEYS}}


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")  # Excel で文字化けしないよう BOM 付き
