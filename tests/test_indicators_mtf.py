import numpy as np
import pandas as pd
import pytest

from fxbt import indicators as ind
from fxbt.mtf import attach
from fxbt.synthetic import make_all
from tests.helpers import bars


def _wilder_ref(x, n):
    out = np.full(len(x), np.nan)
    acc = None
    for i, v in enumerate(x):
        if np.isnan(v):
            continue
        acc = v if acc is None else acc + (v - acc) / n
        out[i] = acc
    return out


def test_rsi_extremes_and_range():
    up = pd.Series(np.arange(1, 40, dtype=float))
    assert ind.rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(40, 1, -1, dtype=float))
    assert ind.rsi(down, 14).iloc[-1] == pytest.approx(0.0)
    s = make_all("2020-01-01", "2020-03-01")["H1"]["close"]
    r = ind.rsi(s, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()
    assert ind.rsi(s, 14).iloc[:13].isna().all()


def test_rsi_matches_reference_wilder():
    s = make_all("2020-01-01", "2020-02-01", seed=3)["H1"]["close"].reset_index(drop=True)
    d = s.diff()
    g = _wilder_ref(d.clip(lower=0).to_numpy(), 14)
    lo = _wilder_ref((-d).clip(lower=0).to_numpy(), 14)
    ref = 100 - 100 / (1 + g / lo)
    got = ind.rsi(s, 14).to_numpy()
    assert np.allclose(got[30:], ref[30:])


def test_atr_matches_reference_and_adx_in_range():
    df = make_all("2020-01-01", "2020-02-01", seed=4)["H1"].reset_index(drop=True)
    pc = df["close"].shift(1)
    tr = np.maximum.reduce([df["high"] - df["low"], (df["high"] - pc).abs().fillna(0), (df["low"] - pc).abs().fillna(0)])
    ref = _wilder_ref(np.asarray(tr, dtype=float), 14)
    assert np.allclose(ind.atr(df, 14).to_numpy()[20:], ref[20:])
    a = ind.adx(df, 14)["adx"].dropna()
    assert len(a) > 0 and ((a >= 0) & (a <= 100)).all()


def test_macd_hist_sign_follows_trend():
    up = pd.Series(np.linspace(100, 110, 200) + np.sin(np.arange(200)) * 0.01)
    m = ind.macd(up)
    assert m["macd"].iloc[-1] > 0


def test_rolling_argmax_offset():
    x = pd.Series([1, 5, 2, 3, 4, 1.0])
    off = ind.rolling_argmax_offset(x, 3)
    assert off.isna().sum() == 2
    assert list(off.iloc[2:]) == [1, 2, 0, 1]
    off_min = ind.rolling_argmin_offset(x, 3)
    assert list(off_min.iloc[2:]) == [2, 1, 2, 0]


def test_indicators_are_causal():
    df = make_all("2020-01-01", "2020-03-01", seed=5)["H1"].reset_index(drop=True)
    k = 700
    full = ind.adx(df, 14)["adx"].iloc[:k]
    part = ind.adx(df.iloc[:k], 14)["adx"]
    pd.testing.assert_series_equal(full, part)
    pd.testing.assert_series_equal(ind.rsi(df["close"], 14).iloc[:k], ind.rsi(df["close"].iloc[:k], 14))


def test_attach_h4_uses_only_completed_bars():
    h1 = bars(np.arange(24, dtype=float) + 100, start="2021-03-01 00:00")
    h4 = pd.DataFrame({"time": pd.date_range("2021-03-01", periods=6, freq="4h"), "x": [0, 1, 2, 3, 4, 5]})
    f = attach(h1, h4, "H4", "h4_")
    by_hour = dict(zip(h1["time"].dt.hour, f["h4_x"]))
    # H1 00:00〜02:00 の確定時点（1〜3 時）では H4 はまだ 1 本も確定していない
    assert np.isnan(by_hour[0]) and np.isnan(by_hour[2])
    # H1 03:00 足の確定 = 04:00 = 00:00 開始の H4 の確定
    assert by_hour[3] == 0
    assert by_hour[6] == 0 and by_hour[7] == 1
    assert by_hour[10] == 1 and by_hour[11] == 2


def test_attach_d1_over_weekend():
    h1 = bars([100.0] * 3, start="2021-03-08 00:00")  # 月曜
    d1 = pd.DataFrame({"time": pd.to_datetime(["2021-03-04", "2021-03-05", "2021-03-08"]), "x": [4, 5, 8]})
    f = attach(h1, d1, "D1", "d1_")
    assert list(f["d1_x"]) == [5, 5, 5]  # 月曜の足は金曜の日足まで。月曜の日足は使わない
