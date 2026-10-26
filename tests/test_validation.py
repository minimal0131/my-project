import copy

import pandas as pd
import pytest

from fxbt.config import load_config
from fxbt.data.resample import merge_sunday_bars
from fxbt.data.validation import (DataValidationError, ValidationReport, check_bars, check_cross_timeframe,
                                  check_sunday_bars, check_timezone, good_friday, parse_timezone, server_to_utc,
                                  utc_to_server, validate_dataset)
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


def test_long_single_gap_is_error(cfg, frames):
    h1 = frames["H1"]
    cut = (h1["time"] >= "2019-03-01") & (h1["time"] < "2019-06-01")  # 3 か月欠落 = 欠損 1 件
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", cfg["validation"], r)
    assert r.stats["H1_gap_events"] == 1
    assert _errors(r, "gap")


def test_missing_full_weekday_next_to_weekend(cfg, frames):
    h1 = frames["H1"]
    cut = h1["time"].dt.normalize() == pd.Timestamp("2019-05-10")  # 祝日ではない金曜が丸ごと欠落
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", cfg["validation"], r)
    assert r.stats["H1_gap_events"] == 1 and r.stats["H1_longest_gap_hours"] == 24
    assert r.ok  # 24 時間 <= 30 時間なので警告のみ


def test_good_friday_gap_is_allowed(cfg, frames):
    h1 = frames["H1"]
    cut = h1["time"].dt.normalize().isin([pd.Timestamp("2019-04-19"), pd.Timestamp("2020-04-10")])
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", cfg["validation"], r)
    assert r.stats["H1_gap_events"] == 0
    vcfg = dict(cfg["validation"], allowed_gap_dates=["12-25", "01-01"])
    r = ValidationReport()
    check_bars(h1[~cut].reset_index(drop=True), "H1", vcfg, r)
    assert r.stats["H1_gap_events"] == 2


def test_good_friday_dates():
    assert [good_friday(y).strftime("%Y-%m-%d") for y in (2016, 2019, 2024, 2025)] == \
        ["2016-03-25", "2019-04-19", "2024-03-29", "2025-04-18"]


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
    r = ValidationReport()
    check_timezone(frames["H1"], frames["D1"], "NY+7", cfg["validation"], r)
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


def test_ny_close_timezone_follows_us_dst():
    # 2021-03-15(月): 米国は 3/14 に夏時間入り、欧州はまだ冬時間
    t = pd.Series(pd.to_datetime(["2021-01-11 00:00", "2021-03-15 00:00", "2021-07-05 00:00",
                                  "2021-11-01 00:00", "2021-11-08 00:00"]))
    u = server_to_utc(t, "NY+7")
    assert list(u.dt.hour) == [22, 21, 21, 21, 22]  # 冬 GMT+2 / 夏 GMT+3（米国の切替日に連動）
    assert server_to_utc(t, "Europe/Athens").dt.hour.iloc[1] == 22  # 欧州連動だと 3/15 はまだ GMT+2
    # サーバーの 0 時は常に NY の 17 時
    assert (u.dt.tz_convert("America/New_York").dt.hour == 17).all()


# ------------------------------------------------------------------ 日曜足（欧州の夏時間に連動するサーバー）
def _with_sunday_bar(frames, sunday_time, tf_hours={"H1": 23, "H4": 20, "D1": 0}):
    """各時間足に日曜の足を 1 本足す（夏時間のずれる週に XM が作る 1 時間分の足を模したもの）。"""
    out = {}
    day = pd.Timestamp(sunday_time).normalize()
    for tf, df in frames.items():
        row = df.iloc[[0]].copy()
        row["time"] = day + pd.Timedelta(hours=tf_hours[tf]) if tf != "H1" else pd.Timestamp(sunday_time)
        out[tf] = pd.concat([df, row]).sort_values("time").reset_index(drop=True)
    return out


def test_sunday_bars_in_dst_mismatch_week_allowed_for_eu_server(frames):
    # 2019-03-10: 米国は夏時間入り、欧州は 3/31 まで冬時間 → NY 日曜 17 時 = サーバー（欧州連動）日曜 23 時
    fx = _with_sunday_bar(frames, "2019-03-10 23:00")
    r = ValidationReport()
    check_sunday_bars(fx, "Europe/Athens", r)
    assert r.ok and r.stats["H1_sunday_bars"] == 1 and r.stats["D1_sunday_bars"] == 1
    r = ValidationReport()
    check_sunday_bars(fx, "NY+7", r)  # NY+7 のサーバーなら寄り付きは常に月曜 0 時で、日曜足は説明がつかない
    assert len(_errors(r, "sunday")) == 3


def test_sunday_bar_outside_mismatch_week_is_error(frames):
    r = ValidationReport()
    check_sunday_bars(_with_sunday_bar(frames, "2019-06-09 23:00"), "Europe/Athens", r)
    assert len(_errors(r, "sunday")) == 3


def test_sunday_bar_before_market_open_is_error(frames):
    fx = {"H1": _with_sunday_bar(frames, "2019-03-10 22:00")["H1"]}
    r = ValidationReport()
    check_sunday_bars(fx, "Europe/Athens", r)
    assert _errors(r, "sunday")


def test_no_sunday_bars_passes(frames):
    r = ValidationReport()
    check_sunday_bars(frames, "Europe/Athens", r)
    assert r.ok and r.stats["H1_sunday_bars"] == 0


def test_week_session_match_detects_wrong_dst_rule(cfg, frames):
    # 合成データは月曜 0 時〜金曜 23 時（NY+7 の業者と同じ）。欧州連動と解釈すると、ずれる週で 1 時間ずれる
    r = ValidationReport()
    check_timezone(frames["H1"], frames["D1"], "NY+7", cfg["validation"], r)
    assert r.stats["week_open_ny17_ratio"] == 1.0 and r.stats["week_close_ny17_ratio"] == 1.0
    assert not [i for i in r.issues if i.level == "WARNING"]
    r = ValidationReport()
    check_timezone(frames["H1"], frames["D1"], "Europe/Athens", cfg["validation"], r)
    assert r.stats["week_open_ny17_ratio"] < 0.95
    assert [i for i in r.issues if i.level == "WARNING" and "NY 日曜 17 時" in i.message]


def test_utc_to_server_inverts_server_to_utc():
    t = pd.Series(pd.to_datetime(["2021-01-11 00:00", "2021-03-15 00:00", "2021-07-05 13:00", "2021-11-01 23:00"]))
    for tz in ("NY+7", "Europe/Athens", "+02:00"):
        assert (utc_to_server(server_to_utc(t, tz), tz) == t).all()


def test_merge_sunday_bars_into_monday():
    d1 = pd.DataFrame({
        "time": pd.to_datetime(["2019-03-08", "2019-03-10", "2019-03-11", "2019-03-12"]),
        "open": [110.0, 111.0, 111.1, 112.0], "high": [110.5, 111.9, 111.5, 112.5],
        "low": [109.5, 110.9, 110.0, 111.5], "close": [110.2, 111.05, 111.4, 112.2],
        "tick_volume": [1000, 50, 900, 800], "spread": [15, 30, 16, 15], "real_volume": [0, 0, 0, 0],
    })
    out, n = merge_sunday_bars(d1, "D1")
    assert n == 1 and list(out.columns) == list(d1.columns)
    assert list(out["time"]) == list(pd.to_datetime(["2019-03-08", "2019-03-11", "2019-03-12"]))
    mon = out.iloc[1]
    assert (mon.open, mon.high, mon.low, mon.close) == (111.0, 111.9, 110.0, 111.4)
    assert mon.tick_volume == 950 and mon.spread == 16
    assert out.iloc[[0, 2]].reset_index(drop=True).equals(d1.iloc[[0, 3]].reset_index(drop=True))


def test_merge_sunday_bars_keeps_h1_and_relabels_lonely_sunday():
    h1 = pd.DataFrame({"time": pd.to_datetime(["2019-03-10 23:00", "2019-03-11 00:00"]), "open": [1.0, 1.0],
                       "high": [1.0, 1.0], "low": [1.0, 1.0], "close": [1.0, 1.0], "tick_volume": [1, 1],
                       "spread": [1, 1], "real_volume": [0, 0]})
    out, n = merge_sunday_bars(h1, "H1")
    assert n == 0 and out.equals(h1)
    # 直後の月曜足が無い場合は月曜 0 時の足として扱う（使えるようになる時刻が遅れるだけ）
    h4 = h1.iloc[[0]].assign(time=pd.Timestamp("2019-03-10 20:00"))
    out, n = merge_sunday_bars(h4, "H4")
    assert n == 1 and out["time"].tolist() == [pd.Timestamp("2019-03-11 00:00")]
