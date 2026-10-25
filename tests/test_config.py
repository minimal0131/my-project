import copy

import pytest

from fxbt.config import ConfigError, load_config, strategy_params, validate_config

CFG_PATH = "config/default.yaml"


def test_default_config_loads():
    cfg = load_config(CFG_PATH)
    assert cfg["symbol"] == "USDJPY"
    assert set(cfg["data"]["timeframes"]) == {"H1", "H4", "D1"}


def test_holdout_must_follow_development():
    cfg = load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    bad["periods"]["holdout_start"] = "2022-06-01"
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_grid_keys_must_exist_in_params():
    cfg = load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    bad["strategies"]["pullback"]["grid"]["no_such_param"] = [1, 2]
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_negative_cost_rejected():
    cfg = load_config(CFG_PATH)
    bad = copy.deepcopy(cfg)
    bad["costs"]["spread_pips"] = -1
    with pytest.raises(ConfigError):
        validate_config(bad)


def test_strategy_params_overrides_do_not_mutate_config():
    cfg = load_config(CFG_PATH)
    p = strategy_params(cfg, "pullback", {"sl_atr": 9.9, "filters": {"h4_adx": False}})
    assert p["sl_atr"] == 9.9 and p["filters"]["h4_adx"] is False
    assert cfg["strategies"]["pullback"]["params"]["sl_atr"] != 9.9
    assert cfg["strategies"]["pullback"]["filters"]["h4_adx"] is True
