"""価格データの検査。重大な異常（ERROR）が 1 件でもあれば検証を停止する。

検査項目:
  * 列・型・NaN・価格の正値
  * 時刻の単調増加・重複・足の境界（H1=正時, H4=4 時間刻み, D1=0 時）
  * OHLC の整合性（high >= max(open, close), low <= min(open, close)）
  * 異常な変動率・出来高 0 の足
  * 欠損（週末・指定休場日以外の空白）とデータ期間のカバー率
  * 形成中の足が混入していないか
  * タイムゾーン（週明けの寄り付きが UTC で日曜夜になっているか、寄り付き・引けが NY 17 時に一致するか、
    日足の本数/週）
  * 日曜の足（夏時間のずれる週の寄り付き以降の足だけを許容）
  * 時間足間の整合性（H4/D1 と H1 集計の比較）
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .mt5_loader import RATE_COLUMNS, TF_DURATION
from .resample import aggregate


class DataValidationError(RuntimeError):
    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__("データ検査で重大な異常を検出したため停止します:\n" + report.format(errors_only=True))


@dataclass
class Issue:
    level: str          # "ERROR" | "WARNING" | "INFO"
    timeframe: str
    check: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, level: str, tf: str, check: str, message: str) -> None:
        self.issues.append(Issue(level, tf, check, message))

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.level == "ERROR"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def format(self, errors_only: bool = False) -> str:
        rows = [i for i in self.issues if not errors_only or i.level == "ERROR"]
        return "\n".join(f"[{i.level}] {i.timeframe} {i.check}: {i.message}" for i in rows) or "(問題なし)"

    def raise_if_errors(self) -> None:
        if not self.ok:
            raise DataValidationError(self)


# ---------------------------------------------------------------- タイムゾーン
NY_CLOSE = "NY+7"  # サーバー時刻 = ニューヨーク時刻 + 7 時間（冬 GMT+2 / 夏 GMT+3、米国の夏時間に連動）


def parse_timezone(spec: str) -> _dt.tzinfo:
    """'+02:00' のような固定オフセット、または IANA 名を受け付ける（'NY+7' は server_to_utc で扱う）。"""
    s = spec.strip()
    if s[:1] in "+-" and ":" in s:
        sign = 1 if s[0] == "+" else -1
        hh, mm = s[1:].split(":")
        return _dt.timezone(sign * _dt.timedelta(hours=int(hh), minutes=int(mm)))
    return ZoneInfo(s)


def server_to_utc(times: pd.Series, tz_spec: str) -> pd.Series:
    """サーバー時刻（タイムゾーン無し）を UTC に変換する。

    'NY+7' は MT5 業者で主流の NY クローズ基準。サーバーの 0 時が常に NY の 17 時になるよう、
    米国の夏時間（3 月第 2 日曜〜11 月第 1 日曜）に合わせて GMT+2 / GMT+3 が切り替わる。
    """
    if tz_spec.strip().upper() == NY_CLOSE:
        ny = (times - pd.Timedelta(hours=7)).dt.tz_localize("America/New_York", ambiguous="NaT",
                                                             nonexistent="NaT")
        return ny.dt.tz_convert("UTC")
    tz = parse_timezone(tz_spec)
    return times.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")


def utc_to_server(times: pd.Series, tz_spec: str) -> pd.Series:
    """UTC（タイムゾーン付き）をサーバー時刻（タイムゾーン無し）に変換する。server_to_utc の逆。"""
    if tz_spec.strip().upper() == NY_CLOSE:
        return times.dt.tz_convert("America/New_York").dt.tz_localize(None) + pd.Timedelta(hours=7)
    return times.dt.tz_convert(parse_timezone(tz_spec)).dt.tz_localize(None)


def ny_sunday_open_server(sundays: pd.Series, tz_spec: str) -> pd.Series:
    """各日曜（サーバー時刻の日付）について、NY 日曜 17 時の寄り付きをサーバー時刻で返す。"""
    ny = (sundays.dt.normalize() + pd.Timedelta(hours=17)).dt.tz_localize("America/New_York")
    return utc_to_server(ny.dt.tz_convert("UTC"), tz_spec)


# ---------------------------------------------------------------- 休場日
def good_friday(year: int) -> pd.Timestamp:
    """西方教会の復活祭（匿名グレゴリオ暦算法）の 2 日前。"""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year, month, day) - pd.Timedelta(days=2)


def is_allowed_gap_day(days: pd.DatetimeIndex, allowed: list[str]) -> np.ndarray:
    """休場・短縮取引として欠落を許容する日。'MM-DD' と 'good_friday'（移動祝日）を受け付ける。"""
    mmdd = {a for a in allowed if a != "good_friday"}
    mask = days.strftime("%m-%d").isin(mmdd) if mmdd else np.zeros(len(days), dtype=bool)
    if "good_friday" in allowed and len(days):
        gf = {good_friday(y) for y in set(days.year)}
        mask = mask | days.normalize().isin(list(gf))
    return np.asarray(mask)


# ---------------------------------------------------------------- 単一時間足
def check_bars(df: pd.DataFrame, tf: str, vcfg: dict, report: ValidationReport,
               requested: tuple[str, str] | None = None, server_now: str | None = None) -> None:
    missing = [c for c in RATE_COLUMNS if c not in df.columns]
    if missing:
        report.add("ERROR", tf, "schema", f"列がありません: {missing}")
        return
    n = len(df)
    report.stats[f"{tf}_rows"] = n
    min_bars = (vcfg.get("min_bars") or {}).get(tf, 1)
    if n < min_bars:
        report.add("ERROR", tf, "row_count", f"本数 {n} が下限 {min_bars} 未満")
        if n == 0:
            return

    px = df[["open", "high", "low", "close"]]
    if df[RATE_COLUMNS].isna().any().any():
        bad = df[df[RATE_COLUMNS].isna().any(axis=1)]["time"].head(5).tolist()
        report.add("ERROR", tf, "nan", f"欠損値 (NaN) を含む足: {bad}")
    if (px <= 0).any().any():
        report.add("ERROR", tf, "price", "0 以下の価格があります")

    t = df["time"]
    if t.isna().any():
        report.add("ERROR", tf, "time", "時刻が欠損している足があります")
        return
    if t.duplicated().any():
        report.add("ERROR", tf, "duplicate", f"重複時刻: {t[t.duplicated()].head(5).tolist()}")
    if not t.is_monotonic_increasing:
        report.add("ERROR", tf, "order", "時刻が昇順ではありません")

    # 足の境界
    if tf == "H1":
        misaligned = (t.dt.minute != 0) | (t.dt.second != 0)
    elif tf == "H4":
        misaligned = (t.dt.hour % 4 != 0) | (t.dt.minute != 0)
    else:
        misaligned = (t.dt.hour != 0) | (t.dt.minute != 0)
    if misaligned.any():
        report.add("ERROR", tf, "alignment",
                   f"足の境界がずれた時刻 {int(misaligned.sum())} 本（例 {t[misaligned].iloc[0]}）。"
                   "サーバー時刻またはデータ形式を確認してください")

    # OHLC 整合性
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    bad_ohlc = (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9) | (h < l)
    if bad_ohlc.any():
        report.add("ERROR", tf, "ohlc", f"OHLC が矛盾する足 {int(bad_ohlc.sum())} 本（例 {t[bad_ohlc].iloc[0]}）")

    ret = c.pct_change().abs()
    max_ret = vcfg.get("max_abs_return", 0.05)
    spikes = ret > max_ret
    if spikes.any():
        report.add("ERROR", tf, "spike", f"終値の変動率が {max_ret:.1%} を超える足 {int(spikes.sum())} 本"
                   f"（例 {t[spikes].iloc[0]}）")
    zero_vol = float((df["tick_volume"] <= 0).mean())
    if zero_vol > vcfg.get("max_zero_volume_ratio", 0.01):
        report.add("ERROR", tf, "volume", f"tick_volume が 0 の足の比率 {zero_vol:.2%}")

    sat = t.dt.dayofweek == 5
    if sat.any():
        report.add("WARNING", tf, "weekend", f"土曜日の足が {int(sat.sum())} 本あります（サーバー時刻を確認）")

    _check_gaps(t, tf, vcfg, report)

    if requested is not None:
        req_s, req_e = pd.Timestamp(requested[0]), pd.Timestamp(requested[1])
        if t.iloc[0] - req_s > pd.Timedelta(days=10):
            report.add("ERROR", tf, "coverage",
                       f"データ開始 {t.iloc[0]} が要求開始 {req_s} より大幅に遅い"
                       "（MT5 の履歴が不足。ツール→オプション→チャートの最大バー数を増やす）")
        ref_end = min(req_e, pd.Timestamp(server_now)) if server_now else req_e
        if ref_end - t.iloc[-1] > pd.Timedelta(days=10):
            report.add("ERROR", tf, "coverage", f"データ終了 {t.iloc[-1]} が要求終了 {ref_end} より大幅に早い")

    if server_now:
        last_close = t.iloc[-1] + TF_DURATION[tf]
        if last_close > pd.Timestamp(server_now):
            report.add("ERROR", tf, "incomplete_bar", f"形成中の足が含まれています: {t.iloc[-1]}")


def _check_gaps(t: pd.Series, tf: str, vcfg: dict, report: ValidationReport) -> None:
    """空白ごとに「本来あるはずの平日の足」を数える。週末と allowed_gap_dates（休場・短縮取引日）は数えない。

    * 欠落時間が max_intraweek_gap_hours を超える空白（D1 は 1 本でも）を欠損 1 件として記録
    * 欠損件数が max_gap_events を超える、または 1 回の空白の欠落時間が max_single_gap_hours を超える場合は停止
    """
    step = TF_DURATION[tf]
    step_hours = step / pd.Timedelta(hours=1)
    if len(t) < 2:
        return
    prev, cur = t.iloc[:-1].to_numpy(), t.iloc[1:].to_numpy()
    gaps = pd.DataFrame({"prev": prev, "cur": cur})
    gaps = gaps[(gaps["cur"] - gaps["prev"]) > step]
    allowed = list(vcfg.get("allowed_gap_dates") or [])
    max_hours = float(vcfg.get("max_intraweek_gap_hours", 4))
    max_single = float(vcfg.get("max_single_gap_hours", 30))
    events, longest = [], 0.0
    for p, cnx in zip(gaps["prev"], gaps["cur"]):
        p, cnx = pd.Timestamp(p), pd.Timestamp(cnx)
        expected = pd.date_range(p + step, cnx, freq=step, inclusive="left")
        expected = expected[expected.dayofweek < 5]
        if allowed:
            expected = expected[~is_allowed_gap_day(expected, allowed)]
        hours = len(expected) * step_hours
        if hours == 0:
            continue
        longest = max(longest, hours)
        if tf == "D1" or hours > max_hours:
            events.append((p, cnx, hours))
    report.stats[f"{tf}_gap_events"] = len(events)
    report.stats[f"{tf}_longest_gap_hours"] = longest
    for p, cnx, hours in events[:10]:
        report.add("WARNING", tf, "gap", f"{p} の次が {cnx}（平日 {hours:.0f} 時間分の欠落）")
    if len(events) > vcfg.get("max_gap_events", 20):
        report.add("ERROR", tf, "gap", f"欠損 {len(events)} 件が上限 {vcfg.get('max_gap_events', 20)} 件を超えています")
    if longest > max_single:
        report.add("ERROR", tf, "gap", f"1 回の欠落が平日 {longest:.0f} 時間分あり、上限 {max_single:.0f} 時間を超えています")


# ---------------------------------------------------------------- タイムゾーン検査
def check_timezone(h1: pd.DataFrame, d1: pd.DataFrame | None, tz_spec: str, vcfg: dict,
                   report: ValidationReport) -> None:
    try:
        utc = server_to_utc(h1["time"], tz_spec)
    except Exception as e:  # noqa: BLE001
        report.add("ERROR", "ALL", "timezone", f"server_timezone '{tz_spec}' を解釈できません: {e}")
        return
    n_nat = int(utc.isna().sum())
    if n_nat:
        report.add("WARNING", "H1", "timezone", f"夏時間切替で存在しない/曖昧な時刻 {n_nat} 本")

    # 週の最初の足（48 時間以上の空白の直後）は UTC で日曜 20:00〜月曜 01:00 のはず
    t = h1["time"]
    week_open = t[(t.diff() > pd.Timedelta(hours=48))]
    if len(week_open) >= 4:
        u = utc.loc[week_open.index].dropna()
        hours_from_sun = (u.dt.dayofweek - 6) % 7 * 24 + u.dt.hour  # 日曜 0 時からの時間
        plausible = ((hours_from_sun >= 20) & (hours_from_sun <= 25)).mean()
        report.stats["week_open_utc_plausible_ratio"] = round(float(plausible), 3)
        if plausible < 0.5:
            report.add("ERROR", "H1", "timezone",
                       f"週明けの寄り付きが UTC 日曜夜になる週が {plausible:.0%} のみ。server_timezone を確認してください")
        elif plausible < 0.9:
            report.add("WARNING", "H1", "timezone", f"週明けの寄り付き時刻が想定と異なる週が {1 - plausible:.0%}")

        # 厳密な照合: 寄り付き = NY 日曜 17 時、引け（週最後の足の終了時刻）= NY 金曜 17 時
        # 米国と欧州の夏時間がずれる週で、server_timezone が業者と違えば 1 時間ずれて不一致になる
        is_open = (t.diff() > pd.Timedelta(hours=48)).to_numpy()
        is_close = np.roll(is_open, -1)
        is_close[-1] = False
        ny_open = utc[is_open].dropna().dt.tz_convert("America/New_York")
        close_end = server_to_utc(t[is_close] + TF_DURATION["H1"], tz_spec).dropna().dt.tz_convert("America/New_York")
        open_ratio = float(((ny_open.dt.dayofweek == 6) & (ny_open.dt.hour == 17) & (ny_open.dt.minute == 0)).mean())
        close_ratio = float(((close_end.dt.dayofweek == 4) & (close_end.dt.hour == 17)
                             & (close_end.dt.minute == 0)).mean())
        report.stats["week_open_ny17_ratio"] = round(open_ratio, 3)
        report.stats["week_close_ny17_ratio"] = round(close_ratio, 3)
        min_match = vcfg.get("min_week_session_match_ratio", 0.95)
        if min(open_ratio, close_ratio) < min_match:
            report.add("WARNING", "H1", "timezone",
                       f"週の寄り付きが NY 日曜 17 時に一致する週 {open_ratio:.1%}、引けが NY 金曜 17 時に一致する週 "
                       f"{close_ratio:.1%}（下限 {min_match:.0%}）。server_timezone が業者の夏時間の扱いと"
                       "合っているか確認してください")

    if d1 is not None and len(d1) > 20:
        per_week = d1.groupby(d1["time"].dt.to_period("W"))["time"].count()
        typical = int(per_week.iloc[1:-1].median()) if len(per_week) > 2 else int(per_week.median())
        report.stats["d1_bars_per_week"] = typical
        expected = vcfg.get("expected_daily_bars_per_week", 5)
        if typical != expected:
            report.add("ERROR", "D1", "timezone",
                       f"日足が週 {typical} 本（想定 {expected} 本）。日曜足のあるブローカーは日足の意味が変わるため、"
                       "設定 expected_daily_bars_per_week とサーバー時刻を確認してください")


def check_sunday_bars(frames: dict[str, pd.DataFrame], tz_spec: str, report: ValidationReport) -> None:
    """日曜の足は「NY 日曜 17 時の寄り付きがサーバー時刻の日曜に入る週」の、寄り付き以降を含む足だけを許容する。

    欧州の夏時間に連動するサーバーでは、米国と欧州の夏時間がずれる週だけ寄り付きが日曜 23 時になり、
    H1 23 時・H4 20 時・D1 0 時の日曜足ができる。それ以外の日曜足（寄り付き前の足、ずれの無い週の足、
    NY+7 のサーバーでの日曜足）は時刻の解釈の誤りかデータの異常なので停止する。
    """
    for tf, df in frames.items():
        t = df["time"]
        sun = t[t.dt.dayofweek == 6]
        if sun.empty:
            report.stats[f"{tf}_sunday_bars"] = 0
            continue
        try:
            open_srv = ny_sunday_open_server(sun, tz_spec)
        except Exception as e:  # noqa: BLE001
            report.add("ERROR", tf, "sunday", f"server_timezone '{tz_spec}' で寄り付き時刻を計算できません: {e}")
            continue
        ok = (sun + TF_DURATION[tf] > open_srv) & (open_srv < sun.dt.normalize() + pd.Timedelta(days=1))
        report.stats[f"{tf}_sunday_bars"] = len(sun)
        if (~ok).any():
            report.add("ERROR", tf, "sunday",
                       f"説明のつかない日曜の足が {int((~ok).sum())} 本（例 {sun[~ok].iloc[0]}）。"
                       f"server_timezone '{tz_spec}' と業者の時刻を確認してください")
        if ok.any():
            report.add("INFO", tf, "sunday",
                       f"夏時間のずれる週の日曜足 {int(ok.sum())} 本（NY 17 時の寄り付きがサーバーの日曜に入る週）。"
                       + ("検証では直後の月曜足に統合する" if tf != "H1" else "H1 はそのまま使う"))


# ---------------------------------------------------------------- 時間足間の整合性
def check_cross_timeframe(h1: pd.DataFrame, higher: pd.DataFrame, tf: str, pip: float, vcfg: dict,
                          report: ValidationReport) -> None:
    agg = aggregate(h1, tf)
    m = higher.merge(agg, on="time", how="inner", suffixes=("", "_agg"))
    full_hours = 4 if tf == "H4" else 24
    m = m[m["n_h1"] == full_hours]  # H1 が揃っている足のみ比較
    if len(m) == 0:
        report.add("ERROR", tf, "cross_tf", "H1 と時刻が一致する足がありません（時刻体系の不一致）")
        return
    tol = vcfg.get("cross_tf_tolerance_pips", 3.0) * pip
    diff = pd.concat([(m[k] - m[f"{k}_agg"]).abs() for k in ("open", "high", "low", "close")], axis=1).max(axis=1)
    ratio = float((diff > tol).mean())
    report.stats[f"{tf}_cross_tf_mismatch_ratio"] = round(ratio, 4)
    limit = vcfg.get("max_cross_tf_mismatch_ratio", 0.02)
    if ratio > limit:
        report.add("ERROR", tf, "cross_tf",
                   f"H1 集計と {tol / pip:.1f} pips 以上ずれる足が {ratio:.2%}（上限 {limit:.2%}）")
    elif ratio > 0:
        report.add("INFO", tf, "cross_tf", f"H1 集計とのずれ（許容差超）{ratio:.2%}")

    only_higher = set(higher["time"]) - set(agg["time"])
    if only_higher:
        report.add("WARNING", tf, "cross_tf", f"H1 に対応する足が無い {tf} 足が {len(only_higher)} 本")


# ---------------------------------------------------------------- まとめ
def validate_dataset(frames: dict[str, pd.DataFrame], cfg: dict, metas: dict[str, dict] | None = None
                     ) -> ValidationReport:
    vcfg = cfg.get("validation", {})
    report = ValidationReport()
    metas = metas or {}
    for tf, df in frames.items():
        meta = metas.get(tf, {})
        check_bars(df, tf, vcfg, report,
                   requested=(cfg["data"]["start"], cfg["data"]["end"]),
                   server_now=meta.get("server_time_at_fetch"))
    if not report.ok:
        return report  # 単一時間足が壊れていれば以降の比較は意味がない
    if "H1" in frames:
        check_timezone(frames["H1"], frames.get("D1"), cfg["data"]["server_timezone"], vcfg, report)
        check_sunday_bars(frames, cfg["data"]["server_timezone"], report)
        for tf in ("H4", "D1"):
            if tf in frames:
                check_cross_timeframe(frames["H1"], frames[tf], tf, cfg["pip_size"], vcfg, report)
    return report
