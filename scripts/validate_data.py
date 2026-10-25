"""段階3: 保存済みデータを検査する。重大な異常があれば終了コード 1 で停止する。

使い方:  python scripts/validate_data.py --config config/default.yaml
結果は results/data_validation.md に保存される。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fxbt.config import load_config  # noqa: E402
from fxbt.data import storage  # noqa: E402
from fxbt.data.validation import validate_dataset  # noqa: E402


def load_frames(cfg: dict) -> tuple[dict, dict]:
    frames, metas = {}, {}
    for tf in cfg["data"]["timeframes"]:
        frames[tf], metas[tf] = storage.load(cfg["data"]["dir"], cfg["symbol"], tf)
    return frames, metas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/default.yaml")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    try:
        frames, metas = load_frames(cfg)
    except storage.DataStoreError as e:
        logging.error("%s", e)
        return 1
    if any(m.get("synthetic") for m in metas.values()):
        logging.warning("*** 合成データです。実際の相場ではありません ***")
    report = validate_dataset(frames, cfg, metas)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# データ検査結果", "", f"- 判定: **{'合格' if report.ok else '不合格（検証を停止）'}**",
             f"- サーバー時刻のタイムゾーン設定: `{cfg['data']['server_timezone']}`", "", "## 統計", "",
             "```json", json.dumps(report.stats, ensure_ascii=False, indent=2), "```", "",
             "## 検出事項", ""]
    lines += [f"- [{i.level}] {i.timeframe} `{i.check}`: {i.message}" for i in report.issues] or ["- なし"]
    lines += ["", "## 取得メタデータ", "", "```json", json.dumps(metas, ensure_ascii=False, indent=2), "```"]
    (out_dir / "data_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(report.format())
    if not report.ok:
        logging.error("データ検査に不合格のため停止します（%s）", out_dir / "data_validation.md")
        return 1
    logging.info("データ検査に合格しました（%s）", out_dir / "data_validation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
