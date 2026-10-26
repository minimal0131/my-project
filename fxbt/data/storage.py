"""取得データの CSV 保存・読込。メタデータに SHA256 を記録し、改変・破損を検出する。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .mt5_loader import RATE_COLUMNS, FetchResult


class DataStoreError(RuntimeError):
    pass


def _paths(data_dir: str | Path, symbol: str, tf: str) -> tuple[Path, Path]:
    d = Path(data_dir)
    return d / f"{symbol}_{tf}.csv", d / f"{symbol}_{tf}.meta.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(result: FetchResult, data_dir: str | Path) -> Path:
    csv_path, meta_path = _paths(data_dir, result.symbol, result.timeframe)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = result.bars.copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    out.to_csv(csv_path, index=False, lineterminator="\n")
    meta = dict(result.meta, sha256=_sha256(csv_path), file=csv_path.name)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return csv_path


def load(data_dir: str | Path, symbol: str, tf: str, verify_hash: bool = True) -> tuple[pd.DataFrame, dict]:
    csv_path, meta_path = _paths(data_dir, symbol, tf)
    if not csv_path.exists():
        raise DataStoreError(f"データファイルがありません: {csv_path}（先に scripts/fetch_data.py を実行）")
    meta: dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if verify_hash and meta.get("sha256") and meta["sha256"] != _sha256(csv_path):
            raise DataStoreError(f"{csv_path} のハッシュがメタデータと一致しません（改変または破損）")
    elif verify_hash:
        raise DataStoreError(f"メタデータがありません: {meta_path}")
    df = pd.read_csv(csv_path)
    missing = [c for c in RATE_COLUMNS if c not in df.columns]
    if missing:
        raise DataStoreError(f"{csv_path} に列がありません: {missing}")
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    return df, meta
