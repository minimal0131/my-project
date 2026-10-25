"""設定ファイルの読込と妥当性検査。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

VALID_TIMEFRAMES = ("H1", "H4", "D1")
STRATEGY_NAMES = ("pullback", "breakout", "mean_reversion")


class ConfigError(ValueError):
    pass


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"設定ファイルが見つかりません: {path}")
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def _require(cfg: dict, dotted: str) -> Any:
    node: Any = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise ConfigError(f"必須の設定項目がありません: {dotted}")
        node = node[key]
    return node


def validate_config(cfg: dict[str, Any]) -> None:
    for key in ("symbol", "pip_size", "data.start", "data.end", "data.server_timezone",
                "costs.spread_pips", "costs.slippage_pips", "backtest.max_hold_hours",
                "periods.development_start", "periods.development_end",
                "periods.holdout_start", "periods.holdout_end",
                "walkforward.train_months", "walkforward.test_months", "walkforward.step_months",
                "strategies"):
        _require(cfg, key)

    for tf in cfg["data"].get("timeframes", []):
        if tf not in VALID_TIMEFRAMES:
            raise ConfigError(f"未対応の時間足: {tf}")
    if cfg["pip_size"] <= 0:
        raise ConfigError("pip_size は正の値にしてください")
    for key in ("spread_pips", "slippage_pips"):
        if cfg["costs"][key] < 0:
            raise ConfigError(f"costs.{key} は 0 以上にしてください")

    p = cfg["periods"]
    dev_s, dev_e = pd.Timestamp(p["development_start"]), pd.Timestamp(p["development_end"])
    ho_s, ho_e = pd.Timestamp(p["holdout_start"]), pd.Timestamp(p["holdout_end"])
    if not dev_s < dev_e:
        raise ConfigError("開発期間の開始が終了より後です")
    if not ho_s < ho_e:
        raise ConfigError("検証期間の開始が終了より後です")
    if not dev_e < ho_s:
        raise ConfigError("検証期間（ホールドアウト）は開発期間より後に、重複なく設定してください")
    data_s, data_e = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    if dev_s < data_s or ho_e > data_e + pd.Timedelta(days=1):
        raise ConfigError("開発・検証期間がデータ取得期間の外にあります")

    wf = cfg["walkforward"]
    for key in ("train_months", "test_months", "step_months"):
        if int(wf[key]) <= 0:
            raise ConfigError(f"walkforward.{key} は正の整数にしてください")

    for name, scfg in cfg["strategies"].items():
        if name not in STRATEGY_NAMES:
            raise ConfigError(f"未知の戦略: {name}")
        if "params" not in scfg:
            raise ConfigError(f"strategies.{name}.params がありません")
        for gkey in (scfg.get("grid") or {}):
            if gkey not in scfg["params"]:
                raise ConfigError(f"strategies.{name}.grid.{gkey} は params に存在しません")


def strategy_params(cfg: dict, name: str, overrides: dict | None = None) -> dict:
    """戦略のパラメーターとフィルター設定を 1 つの辞書にまとめて返す（元の設定は変更しない）。"""
    scfg = cfg["strategies"][name]
    params = copy.deepcopy(scfg["params"])
    params["filters"] = copy.deepcopy(scfg.get("filters") or {})
    if overrides:
        for k, v in overrides.items():
            if k == "filters":
                params["filters"].update(v)
            else:
                params[k] = v
    return params
