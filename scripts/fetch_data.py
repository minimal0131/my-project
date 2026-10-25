"""段階2: MT5 から USD/JPY の H1/H4/D1 を取得して data/raw に保存する（Windows + MT5 端末が必要）。

使い方:  python scripts/fetch_data.py --config config/default.yaml
取得後に自動でデータ検査（段階3）を行い、異常があれば終了コード 1 で停止する。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxbt.config import load_config  # noqa: E402
from fxbt.data import storage  # noqa: E402
from fxbt.data.mt5_loader import DataFetchError, MT5Loader  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--skip-validation", action="store_true", help="取得のみ行う（検査は別途実行）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    m = cfg.get("mt5", {})
    try:
        with MT5Loader(terminal_path=m.get("terminal_path"), timeout_ms=m.get("timeout_ms", 60000),
                       max_retries=m.get("max_retries", 3)) as loader:
            for tf in cfg["data"]["timeframes"]:
                # 終了日はその日の終わりまで含める
                end = pd.Timestamp(cfg["data"]["end"]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                res = loader.fetch(cfg["symbol"], tf, cfg["data"]["start"], end)
                path = storage.save(res, cfg["data"]["dir"])
                logging.info("%s %s: %d 本 (%s〜%s) -> %s", cfg["symbol"], tf, res.meta["rows"],
                             res.meta["first_bar"], res.meta["last_bar"], path)
    except DataFetchError as e:
        logging.error("データ取得に失敗したため停止します: %s", e)
        return 1

    if args.skip_validation:
        return 0
    from scripts.validate_data import main as validate_main
    return validate_main(["--config", args.config])


if __name__ == "__main__":
    sys.exit(main())
