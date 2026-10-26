"""H1 足から上位足を集計する（時間足間の整合性検査と合成データ生成に使用）。"""
from __future__ import annotations

import pandas as pd

RULE = {"H4": "4h", "D1": "1D"}


def aggregate(h1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """H1 を tf に集計。`time` は集計後の足の開始時刻（サーバー時刻 0 時起点）。"""
    g = h1.set_index("time").resample(RULE[tf], origin="start_day", label="left", closed="left")
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "tick_volume": g["tick_volume"].sum(),
        "spread": g["spread"].min(),
        "real_volume": g["real_volume"].sum(),
        "n_h1": g["close"].count(),
    })
    out = out[out["n_h1"] > 0].reset_index()
    return out


_MERGE_AGG = {"open": "first", "high": "max", "low": "min", "close": "last",
              "tick_volume": "sum", "spread": "min", "real_volume": "sum"}


def merge_sunday_bars(df: pd.DataFrame, tf: str) -> tuple[pd.DataFrame, int]:
    """日曜の上位足（H4/D1）を直後の月曜 0 時の足に統合する。戻り値: (統合後, 統合した日曜足の本数)。

    欧州の夏時間に連動するサーバー（XM など）では、米国と欧州の夏時間の切替日がずれる週に
    NY 17 時の寄り付きがサーバー時刻の日曜 23 時になり、1 時間分だけの日曜 H4/D1 足ができる。
    これを 1 本の足として数えると D1 の SMA・ADX・ATR が「1 日分」を誤って数えるため、月曜の足にまとめる。
    統合後の月曜足の値は月曜足の確定時刻以降にしか使われないので、日曜分の情報は遅れて使われるだけで
    未来の情報は混入しない。H1 は実際の取引時間なのでそのまま返す。
    """
    if tf not in RULE:
        return df, 0
    sunday = df["time"].dt.dayofweek == 6
    n = int(sunday.sum())
    if n == 0:
        return df, 0
    key = df["time"].where(~sunday, df["time"].dt.normalize() + pd.Timedelta(days=1))
    agg = {c: _MERGE_AGG.get(c, "last") for c in df.columns if c != "time"}
    out = df.drop(columns="time").groupby(key.rename("time"), sort=True).agg(agg).reset_index()
    return out[list(df.columns)], n
