import pandas as pd
import pytest

from fxbt.data import storage
from fxbt.synthetic import make_all
from fxbt.data.mt5_loader import DataFetchError, MT5Loader, drop_incomplete_bars
from tests.fake_mt5 import FakeMT5


def _loader(fake):
    return MT5Loader(client=fake, retry_wait_sec=0)


def test_fetch_returns_sorted_unique_bars_across_year_chunks():
    fake = FakeMT5()
    with _loader(fake) as ld:
        res = ld.fetch("USDJPY", "H1", "2020-01-01", "2021-01-01")
    df = res.bars
    assert len(df) > 6000
    assert df["time"].is_monotonic_increasing
    assert not df["time"].duplicated().any()
    assert list(df.columns)[:5] == ["time", "open", "high", "low", "close"]
    assert res.meta["rows"] == len(df)


def test_fetch_splits_request_by_year():
    fake = FakeMT5()
    with _loader(fake) as ld:
        ld.fetch("USDJPY", "D1", "2020-03-01", "2020-12-31")
    assert fake.copy_calls == 1
    fake2 = FakeMT5(data=make_all("2018-01-01", "2021-01-01"))
    with _loader(fake2) as ld:
        res = ld.fetch("USDJPY", "D1", "2018-06-01", "2020-12-31")
    assert fake2.copy_calls == 3
    assert not res.bars["time"].duplicated().any()


def test_forming_bar_is_dropped():
    fake = FakeMT5(server_now=pd.Timestamp("2020-06-10 13:30"))
    with _loader(fake) as ld:
        res = ld.fetch("USDJPY", "H4", "2020-06-01", "2020-06-11")
    last = res.bars["time"].iloc[-1]
    # 12:00 開始の H4 は 16:00 確定なので含まれない。最後は 08:00 開始（12:00 確定）
    assert last == pd.Timestamp("2020-06-10 08:00")
    assert res.meta["dropped_incomplete_bars"] >= 1


def test_drop_incomplete_bars_boundary():
    df = pd.DataFrame({"time": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"])})
    out = drop_incomplete_bars(df, "H1", pd.Timestamp("2020-01-01 02:00"))
    assert len(out) == 2  # 01:00 足は 02:00 に確定
    out = drop_incomplete_bars(df, "H1", pd.Timestamp("2020-01-01 01:59"))
    assert len(out) == 1


def test_initialize_failure_raises():
    with pytest.raises(DataFetchError):
        _loader(FakeMT5(fail_init=True)).connect()


def test_unknown_symbol_raises():
    with _loader(FakeMT5()) as ld, pytest.raises(DataFetchError):
        ld.fetch("XXXYYY", "H1", "2020-01-01", "2020-02-01")


def test_copy_failure_is_retried_then_raises():
    fake = FakeMT5(fail_copy_times=2)
    with _loader(fake) as ld:
        res = ld.fetch("USDJPY", "D1", "2020-01-01", "2020-06-01")  # 3 回目で成功
    assert len(res.bars) > 0
    fake = FakeMT5(fail_copy_times=10)
    with _loader(fake) as ld, pytest.raises(DataFetchError):
        ld.fetch("USDJPY", "D1", "2020-01-01", "2020-06-01")


def test_empty_range_raises():
    with _loader(FakeMT5()) as ld, pytest.raises(DataFetchError):
        ld.fetch("USDJPY", "H1", "2010-01-01", "2010-02-01")


def test_fetch_without_connect_raises():
    with pytest.raises(DataFetchError):
        _loader(FakeMT5()).fetch("USDJPY", "H1", "2020-01-01", "2020-02-01")


def test_save_and_load_roundtrip_and_hash_check(tmp_path):
    with _loader(FakeMT5()) as ld:
        res = ld.fetch("USDJPY", "H4", "2020-01-01", "2020-03-01")
    path = storage.save(res, tmp_path)
    df, meta = storage.load(tmp_path, "USDJPY", "H4")
    pd.testing.assert_frame_equal(df[["time", "open", "close"]], res.bars[["time", "open", "close"]],
                                  check_dtype=False)
    assert meta["server"] == "Fake-Server"
    assert meta["symbol_info"]["swap_rollover3days"] == 3
    # 改ざんを検出する
    text = path.read_text().splitlines()
    text[5] = text[5].replace(text[5].split(",")[1], "999.999", 1)
    path.write_text("\n".join(text) + "\n")
    with pytest.raises(storage.DataStoreError):
        storage.load(tmp_path, "USDJPY", "H4")
