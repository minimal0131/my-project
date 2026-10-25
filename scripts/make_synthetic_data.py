"""MT5 の無い環境でパイプラインの動作確認をするための合成データを作る。

**実際の相場ではない。** 生成物のメタデータには synthetic=true が記録され、
レポートにも合成データである旨が表示される。戦略の評価には使わないこと。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxbt.config import load_config  # noqa: E402
from fxbt.data import storage  # noqa: E402
from fxbt.data.mt5_loader import FetchResult  # noqa: E402
from fxbt.synthetic import make_all  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    frames = make_all(cfg["data"]["start"], cfg["data"]["end"], seed=args.seed)
    for tf, df in frames.items():
        meta = {"symbol": cfg["symbol"], "timeframe": tf, "synthetic": True, "seed": args.seed,
                "rows": len(df), "first_bar": str(df["time"].iloc[0]), "last_bar": str(df["time"].iloc[-1])}
        print(storage.save(FetchResult(cfg["symbol"], tf, df, meta), cfg["data"]["dir"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
