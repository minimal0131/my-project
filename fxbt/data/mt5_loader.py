"""MetaTrader 5 から過去のローソク足を取得する（読み取り専用。発注機能は持たない）。

MT5 の `time` は足の **開始時刻** を、ブローカーのサーバー時刻で表したもの
（エポック秒として返るが UTC ではない）。本モジュールではサーバー時刻の
タイムゾーン無し datetime として保持し、タイムゾーンの解釈は検査段階で行う。
"""
from __future__ import annotations

import logging
import os
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

TF_DURATION = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}
RATE_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]


class DataFetchError(RuntimeError):
    """データ取得に失敗した。検証を続行してはならない。"""


@dataclass
class FetchResult:
    symbol: str
    timeframe: str
    bars: pd.DataFrame
    meta: dict[str, Any] = field(default_factory=dict)


class MT5Loader:
    """MT5 端末からレートを取得する。

    `client` には MetaTrader5 モジュール互換のオブジェクトを渡せる（テスト用）。
    None の場合は MetaTrader5 パッケージを import する（Windows のみ）。
    """

    def __init__(self, client: Any | None = None, terminal_path: str | None = None,
                 timeout_ms: int = 60000, max_retries: int = 3, retry_wait_sec: float = 2.0):
        if client is None:
            try:
                import MetaTrader5 as client  # type: ignore[no-redef]
            except ImportError as e:  # pragma: no cover - Windows 以外
                raise DataFetchError(
                    "MetaTrader5 パッケージを import できません。Windows で `pip install MetaTrader5` を実行してください。"
                ) from e
        self.mt5 = client
        self.terminal_path = terminal_path
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.retry_wait_sec = retry_wait_sec
        self._connected = False

    # ---- 接続 ----
    def connect(self) -> None:
        kwargs: dict[str, Any] = {"timeout": self.timeout_ms}
        if self.terminal_path:
            kwargs["path"] = self.terminal_path
        login = os.environ.get("MT5_LOGIN")
        if login:
            kwargs["login"] = int(login)
            kwargs["password"] = os.environ.get("MT5_PASSWORD", "")
            kwargs["server"] = os.environ.get("MT5_SERVER", "")
        if not self.mt5.initialize(**kwargs):
            raise DataFetchError(f"MT5 の初期化に失敗しました: {self.mt5.last_error()}")
        self._connected = True

    def shutdown(self) -> None:
        if self._connected:
            self.mt5.shutdown()
            self._connected = False

    def __enter__(self) -> "MT5Loader":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    # ---- 取得 ----
    def _tf_const(self, tf: str) -> int:
        try:
            return getattr(self.mt5, f"TIMEFRAME_{tf}")
        except AttributeError as e:
            raise DataFetchError(f"未対応の時間足: {tf}") from e

    def _server_now(self, symbol: str) -> pd.Timestamp | None:
        tick = self.mt5.symbol_info_tick(symbol)
        if tick is None or not getattr(tick, "time", 0):
            return None
        return pd.Timestamp(int(tick.time), unit="s")

    def _copy_chunk(self, symbol: str, tf: str, start: datetime, end: datetime):
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            rates = self.mt5.copy_rates_range(symbol, self._tf_const(tf), start, end)
            if rates is not None:
                return rates
            last_err = self.mt5.last_error()
            log.warning("copy_rates_range 失敗 (%s %s %s〜%s, 試行 %d): %s",
                        symbol, tf, start, end, attempt, last_err)
            if attempt < self.max_retries:
                _time.sleep(self.retry_wait_sec)
        raise DataFetchError(f"{symbol} {tf} {start:%Y-%m-%d}〜{end:%Y-%m-%d} の取得に失敗: {last_err}")

    def fetch(self, symbol: str, tf: str, start: str | datetime, end: str | datetime) -> FetchResult:
        if not self._connected:
            raise DataFetchError("MT5 に接続していません（connect() を先に呼んでください）")
        if tf not in TF_DURATION:
            raise DataFetchError(f"未対応の時間足: {tf}")
        if not self.mt5.symbol_select(symbol, True):
            raise DataFetchError(f"シンボル {symbol} を選択できません: {self.mt5.last_error()}")

        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if start_ts >= end_ts:
            raise DataFetchError("取得開始日時が終了日時以降です")

        # 端末の「チャートの最大バー数」制限を避けるため 1 年ずつ取得する
        frames = []
        bounds = sorted(set(pd.date_range(start_ts, end_ts, freq="YS")) | {start_ts, end_ts})
        for chunk_start, chunk_end in zip(bounds[:-1], bounds[1:]):
            rates = self._copy_chunk(
                symbol, tf,
                chunk_start.to_pydatetime().replace(tzinfo=timezone.utc),
                chunk_end.to_pydatetime().replace(tzinfo=timezone.utc),
            )
            if len(rates):
                frames.append(pd.DataFrame(rates))
        if not frames:
            raise DataFetchError(f"{symbol} {tf}: 指定期間のデータが 0 件です")

        df = pd.concat(frames, ignore_index=True)
        missing_cols = [c for c in RATE_COLUMNS if c not in df.columns]
        if missing_cols:
            raise DataFetchError(f"取得データに列がありません: {missing_cols}")
        df = df[RATE_COLUMNS].copy()
        df["time"] = pd.to_datetime(df["time"], unit="s")
        n_dup = int(df.duplicated("time").sum())
        # 年の境界で同じ足が 2 回返ることがあるため、完全一致の重複のみ除去
        df = df.drop_duplicates().sort_values("time").reset_index(drop=True)

        server_now = self._server_now(symbol)
        n_before = len(df)
        df = drop_incomplete_bars(df, tf, server_now if server_now is not None else end_ts)

        meta = {
            "symbol": symbol,
            "timeframe": tf,
            "requested_start": str(start_ts),
            "requested_end": str(end_ts),
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "server_time_at_fetch": str(server_now) if server_now is not None else None,
            "rows": len(df),
            "first_bar": str(df["time"].iloc[0]) if len(df) else None,
            "last_bar": str(df["time"].iloc[-1]) if len(df) else None,
            "dropped_incomplete_bars": n_before - len(df),
            "chunk_boundary_duplicates": n_dup,
            "time_semantics": "bar open time, broker server time (naive)",
        }
        info = self.mt5.account_info() if hasattr(self.mt5, "account_info") else None
        if info is not None:
            meta["server"] = getattr(info, "server", None)
            meta["company"] = getattr(info, "company", None)
        # 取引コスト設定の参考値（スワップは symbol_info の単位 swap_mode に注意）
        sinfo = self.mt5.symbol_info(symbol) if hasattr(self.mt5, "symbol_info") else None
        if sinfo is not None:
            meta["symbol_info"] = {k: getattr(sinfo, k, None) for k in
                                   ("digits", "point", "spread", "spread_float", "swap_mode",
                                    "swap_long", "swap_short", "swap_rollover3days")}
        if df.empty:
            raise DataFetchError(f"{symbol} {tf}: 確定足が 0 件です")
        return FetchResult(symbol, tf, df, meta)


def drop_incomplete_bars(df: pd.DataFrame, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    """終了時刻（開始 + 足の長さ）が `now` より後の足 = 形成中の足を除去する。"""
    close_time = df["time"] + TF_DURATION[tf]
    return df.loc[close_time <= pd.Timestamp(now)].reset_index(drop=True)
