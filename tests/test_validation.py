import copy

import pandas as pd
import pytest

from fxbt.config import load_config
from fxbt.data.validation import (DataValidationError, ValidationReport, check_bars, check_cross_timeframe,
                                  check_timezone, parse_timezone, server_to_utc, validate_dataset)
from fxbt.synthetic import make_all


@pytest.fixture(scope="module")
def cfg():
    c = load_config("config/default.yaml")
    c = copy.deepcopy(c)
    c["data"]["start"], c["data"]["end"] = "2019-01-01", "2021-01-01"
    return c


@pytest.fixture(scope="module")
def frames():
    return make_all("2019-01-01", "2021-01-01", seed=1)


def _errors(report, check=None):
    return [i for i in report.errors if check is None or i.check == check]


def test_clean_synthetic_data_passes(cfg, frames):
    report = validate_dataset(frames, cfg)
    assert report.ok, report.format()
    assert report.stats["d1_bars_per_week"] == 5
    assert report.stats["week_open_utc_plausible_ratio"] >= 0.9


def test_nan_is_error(cfg, frames):
    h1 = frames["H1"].copy()
    h1.loc[100, "close"] = float("nan")
    r = ValidationReport()
    check_bars(h1, "H1", cfg["validation"], r)
    assert _errors(r, "nan")


def test_ohlc_inconsistency_is_error(cfg, frames):
    h1 = frames["H1"].copy()
    h1.loc[50, "high"] = h1.loc[50, "low"] - 0.1
    r = ValidationReport()
    check_bars(h1, "H1", cfg["validation"], r)
    assert _errors(r, "ohlc")


def test_duplicate_and_unsorted_times_are_errors(cfg, frames):
    h1 = pd.concat([frames["H1"], frames["H1"].iloc[[10]]], ignore_index=True)
    r = ValidationReport()
    check_bars(h1, "H1", cfg["validation"], r)
    assert _errors(r, "duplicate") and _errors(r, "order")


def test_misaligned_h4_is_error(cfg, frames):
    h4 = frames["H4"].copy()
    h4["time"] = h4["time"] + pd.Timedelta(hours=1)
    r = ValidationReport()
    check_bars(h4, "H4", cfg["validation"], r)
    assert _errors(r, "alignment")


def test_price_spike_is_error(cfg, frames):
    h1 = frames["H1"].copy()
    h1.loc[200, ["open", "high", "low", "close"]] = [h1.loc[200, "open"], 200.0, h1.loc[200, "low"], 200.0]
    r = ValidationReport()
    check_bars(h1, "H1", cfg["validation"], r)
    assert _errors(r, "spike")


def test_weekend_gaps_are_not_missing_data(cfg, frames):
    r = ValidationReport()
    check_bars(frames["H1"], "H1", cfg["validation"], r)
    assert r.stats["H1_gap_events"] == 0


def test_intraweek_gap_is_detected_and_many_gaps_stop(cfg, frames):
    h1 = frames["H1"]
    # 2019-03-06(水) 10:00〜15:00 を欠落させる -> 6 時間欠落 > 4 時間
    cut = (h1["time"] >= "2019-03-06 10:00") & (h1["time"] < "2019-03-06 16:00")
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", cfg["validation"], r)
    assert r.stats["H1_gap_events"] == 1 and r.ok

    vcfg = dict(cfg["validation"], max_gap_events=0)
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", vcfg, r)
    assert _errors(r, "gap")


def test_allowed_holiday_gap_is_ignored(cfg, frames):
    h1 = frames["H1"]
    cut = h1["time"].dt.strftime("%m-%d") == "12-25"
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", cfg["validation"], r)
    assert r.stats["H1_gap_events"] == 0


def test_missing_weekday_d1_is_gap(cfg, frames):
    d1 = frames["D1"]
    cut = d1["time"] == pd.Timestamp("2019-05-15")
    r = ValidationReport()
    check_bars(d1[~cut].reset_index(drop=True), "D1", cfg["validation"], r)
    assert r.stats["D1_gap_events"] == 1


def test_insufficient_history_is_error(cfg, frames):
    h1 = frames["H1"][frames["H1"]["time"] >= "2019-06-01"].reset_index(drop=True)
    r = ValidationReport()
    check_bars(h1, "H1", cfg["validation"], r, requested=("2019-01-01", "2021-01-01"))
    assert _errors(r, "coverage")


def test_incomplete_last_bar_is_error(cfg, frames):
    d1 = frames["D1"]
    last = d1["time"].iloc[-1]
    r = ValidationReport()
    check_bars(d1, "D1", cfg["validation"], r, server_now=str(last + pd.Timedelta(hours=5)))
    assert _errors(r, "incomplete_bar")


def test_wrong_timezone_is_error(cfg, frames):
    r = ValidationReport()
    check_timezone(frames["H1"], frames["D1"], "Asia/Tokyo", cfg["validation"], r)
    assert _errors(r, "timezone")
    r = ValidationReport()
    check_timezone(frames["H1"], frames["D1"], "Europe/Athens", cfg["validation"], r)
    assert r.ok


def test_sunday_daily_bars_detected(cfg, frames):
    d1 = frames["D1"]
    sundays = d1.copy()
    sundays["time"] = sundays["time"] - pd.Timedelta(days=1)
    sundays = sundays[sundays["time"].dt.dayofweek == 6]
    d1x = pd.concat([d1, sundays]).sort_values("time").reset_index(drop=True)
    r = ValidationReport()
    check_timezone(frames["H1"], d1x, "Europe/Athens", cfg["validation"], r)
    assert r.stats["d1_bars_per_week"] == 6 and _errors(r, "timezone")


def test_cross_timeframe_mismatch_is_error(cfg, frames):
    h4 = frames["H4"].copy()
    h4.loc[h4.index[::10], "close"] += 0.20  # 20 pips ずらす（10%）
    r = ValidationReport()
    check_cross_timeframe(frames["H1"], h4, "H4", cfg["pip_size"], cfg["validation"], r)
    assert _errors(r, "cross_tf")
    r = ValidationReport()
    check_cross_timeframe(frames["H1"], frames["H4"], "H4", cfg["pip_size"], cfg["validation"], r)
    assert r.ok and r.stats["H4_cross_tf_mismatch_ratio"] == 0


def test_raise_if_errors(cfg, frames):
    bad = dict(frames)
    bad["H1"] = frames["H1"].iloc[:10]
    report = validate_dataset(bad, cfg)
    with pytest.raises(DataValidationError):
        report.raise_if_errors()


def test_timezone_parsing():
    assert parse_timezone("+02:00").utcoffset(None).total_seconds() == 7200
    t = pd.Series(pd.to_datetime(["2020-01-06 00:00", "2020-07-06 00:00"]))
    u = server_to_utc(t, "Europe/Athens")
    assert list(u.dt.hour) == [22, 21]  # 冬 UTC+2, 夏 UTC+3
