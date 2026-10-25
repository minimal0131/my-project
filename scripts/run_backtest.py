"""段階6: データ検査 → 3 戦略の検証（開発期間・ウォークフォワード・条件別）→ CSV / レポート出力。

使い方:
  python scripts/run_backtest.py --config config/default.yaml
  python scripts/run_backtest.py --evaluate-holdout   # 開発が終わってから 1 回だけ

ホールドアウトの評価履歴は results/holdout_log.csv に追記され、2 回目以降はレポートに警告が出る。
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxbt.config import load_config  # noqa: E402
from fxbt.data import storage  # noqa: E402
from fxbt.data.validation import DataValidationError  # noqa: E402
from fxbt.pipeline import run  # noqa: E402
from scripts.validate_data import load_frames  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--evaluate-holdout", action="store_true", help="未使用の検証期間を評価する（開発完了後に 1 回）")
    ap.add_argument("--strategy", action="append", help="対象戦略（複数指定可。省略時は有効な全戦略）")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    try:
        frames, metas = load_frames(cfg)
        out = run(cfg, frames, metas, evaluate_holdout=args.evaluate_holdout, strategies=args.strategy)
    except (storage.DataStoreError, DataValidationError) as e:
        logging.error("検証を停止しました: %s", e)
        return 1
    print(f"レポート: {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
