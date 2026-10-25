"""スクリプトの結合テスト（MT5 はモック）。"""
import copy

import yaml

import scripts.fetch_data as fetch_data
from fxbt.config import load_config
from fxbt.synthetic import make_all
from tests.fake_mt5 import FakeMT5


def _cfg(tmp_path):
    cfg = copy.deepcopy(load_config("config/default.yaml"))
    cfg["data"].update(start="2016-01-01", end="2020-12-31", dir=str(tmp_path / "raw"))
    cfg["periods"].update(development_start="2016-07-01", development_end="2019-06-30",
                          holdout_start="2019-07-01", holdout_end="2020-12-31")
    cfg["output"]["dir"] = str(tmp_path / "results")
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def test_fetch_script_fetches_saves_and_validates(tmp_path, monkeypatch):
    from fxbt.data.mt5_loader import MT5Loader
    fake = FakeMT5(data=make_all("2016-01-01", "2021-01-01", seed=2))
    monkeypatch.setattr(fetch_data, "MT5Loader", lambda **kw: MT5Loader(client=fake, retry_wait_sec=0, **kw))
    assert fetch_data.main(["--config", str(_cfg(tmp_path))]) == 0
    for tf in ("H1", "H4", "D1"):
        assert (tmp_path / "raw" / f"USDJPY_{tf}.csv").exists()
    assert "合格" in (tmp_path / "results" / "data_validation.md").read_text(encoding="utf-8")


def test_fetch_script_stops_on_fetch_failure(tmp_path, monkeypatch):
    from fxbt.data.mt5_loader import MT5Loader
    fake = FakeMT5(fail_init=True)
    monkeypatch.setattr(fetch_data, "MT5Loader", lambda **kw: MT5Loader(client=fake, retry_wait_sec=0, **kw))
    assert fetch_data.main(["--config", str(_cfg(tmp_path))]) == 1


def test_fetch_script_stops_on_bad_data(tmp_path, monkeypatch):
    from fxbt.data.mt5_loader import MT5Loader
    data = make_all("2016-01-01", "2021-01-01", seed=2)
    h1 = data["H1"]
    data["H1"] = h1[~((h1["time"] >= "2018-03-01") & (h1["time"] < "2018-06-01"))].reset_index(drop=True)
    fake = FakeMT5(data=data)
    monkeypatch.setattr(fetch_data, "MT5Loader", lambda **kw: MT5Loader(client=fake, retry_wait_sec=0, **kw))
    assert fetch_data.main(["--config", str(_cfg(tmp_path))]) == 1
    assert "不合格" in (tmp_path / "results" / "data_validation.md").read_text(encoding="utf-8")
