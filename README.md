# USD/JPY スイング戦略 検証ツール（MetaTrader 5 / Python）

USD/JPY の 3 つのスイング戦略（保有 1〜3 日）を、取引コスト込みで過去データにより比較し、
**統計的な優位性があるかどうか** を検証するツールです。「勝率 90%」のような結果を前提にせず、
優位性が見つからない場合もそのまま報告します。**発注機能はありません**（MT5 は価格取得のみ）。

詳しい設計とバイアス対策は [docs/PLAN.md](docs/PLAN.md) を参照してください。

## 必要環境

- Windows + MetaTrader 5 端末（ログイン済み、USDJPY が気配値表示にあること）
- Python 3.10 以上

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

MT5 の「ツール → オプション → チャート → チャートの最大バー数」を「無制限」にしておくと、長期の H1 履歴を取得できます。
口座を指定する場合は設定ファイルではなく環境変数 `MT5_LOGIN` / `MT5_PASSWORD` / `MT5_SERVER` を使います。

## 使い方

```powershell
# 1. 設定を確認（期間・コスト・パラメーター・ブローカーのサーバー時刻）
notepad config\default.yaml

# 2. MT5 から H1/H4/D1 を取得 → 自動でデータ検査（異常があれば停止）
python scripts\fetch_data.py

# 3. 検証（開発期間・ウォークフォワード・条件別検証）。ホールドアウトは評価しない
python scripts\run_backtest.py

# 4. 開発を終えてから 1 回だけ、未使用の検証期間を評価
python scripts\run_backtest.py --evaluate-holdout
```

結果は `results/run_YYYYMMDD_HHMMSS/` に出力されます。

| ファイル | 内容 |
|---|---|
| `report.md` | 結論・全指標・ウォークフォワード全窓・年別・条件別検証 |
| `summary.csv` | 戦略ごとの要約 |
| `trades_<戦略>_{development,walkforward_oos,holdout}.csv` | 全取引（約定・決済価格、理由、コスト、スワップ、R） |
| `walkforward_windows_<戦略>.csv` | 窓ごとの選択パラメーターと学習/検証成績 |
| `ablation_<戦略>.csv` | 条件を 1 つずつ外した/加えたときの成績 |
| `config_used.yaml` | 実行時の設定 |
| `../data_validation.md` | データ検査の結果 |
| `../holdout_log.csv` | ホールドアウトの評価履歴（2 回目以降はレポートに警告） |

## 初期設定の前提（一般的な MT5 業者）

| 項目 | 初期値 | 内容 |
|---|---|---|
| サーバー時刻 | `NY+7` | 冬 GMT+2 / 夏 GMT+3（米国の夏時間に連動）。サーバー 0 時 = NY 17 時、日足は週 5 本 |
| スプレッド | 1.2 pips + 足の spread 列との大きい方 | スタンダード口座（手数料なし）想定 |
| スリッページ | 片道 0.3 pips | 成行のエントリー・損切り・時間決済 |
| スワップ | 2015〜2025 年の期間別概算値 | 日米金利差の変化に合わせて 10 区間。水曜 3 倍 |
| 休場日 | 12/24, 12/25, 12/31, 1/1, 聖金曜日 | 欠落を許容するだけで、取引があっても問題なし |

業者の夏時間切替が欧州に連動する場合は `Europe/Athens` にします（3 月・11 月の 2〜3 週間だけ差が出ます）。
ECN/RAW 口座の場合は `spread_pips` を「RAW スプレッド + 往復手数料の pips 換算」にしてください。
データ取得時に `data/raw/*.meta.json` の `symbol_info` に業者の現在のスワップ・スプレッドが記録されるので、
大きく違う場合は設定を合わせてください。
データ検査は「週明けの寄り付きが UTC 日曜夜か」「日足が週 5 本か」を確認し、合わなければ停止します。

## MT5 の無い環境での動作確認

```bash
python scripts/make_synthetic_data.py   # 合成データ（実際の相場ではない）
python scripts/validate_data.py
python scripts/run_backtest.py
python -m pytest
```

合成データのレポートには警告が表示されます。合成データの成績で戦略を評価しないでください。
出力例は [docs/sample_report_synthetic.md](docs/sample_report_synthetic.md) にあります。

## 注意

- バックテストの結果は将来の利益を保証しません。
- 成績の良い期間だけを選んで評価しないよう、ウォークフォワードの全窓と全年を出力します。
- パラメーター格子を広げる・条件を足し引きして何度も試すほど、偶然よく見える結果を選ぶ危険が増えます。
