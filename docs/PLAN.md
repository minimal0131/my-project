# USD/JPY スイング戦略 検証基盤 — 実装計画

## 目的

「勝率90%」を前提にせず、3つの独立した戦略を過去データで比較し、
取引コスト込みで **統計的な優位性があるかどうか** を検証する。
優位性が見つからないことも正当な結論として報告する。
実口座への発注機能は実装しない（MT5 は価格取得にのみ使用）。

## 段階

| 段階 | 内容 | 主なモジュール |
|---|---|---|
| 1 | 計画・ファイル構成・設定ファイル | `docs/PLAN.md`, `config/default.yaml`, `fxbt/config.py` |
| 2 | MT5 から H1/H4/D1 を取得（確定足のみ）・CSV 保存 | `fxbt/data/mt5_loader.py`, `fxbt/data/storage.py`, `scripts/fetch_data.py` |
| 3 | 欠損・タイムゾーン・OHLC 整合性・時間足間整合性の検査（異常時は停止） | `fxbt/data/validation.py`, `scripts/validate_data.py` |
| 4 | 指標・時間足合成・バックテストエンジン・3戦略 | `fxbt/indicators.py`, `fxbt/mtf.py`, `fxbt/backtest/`, `fxbt/strategies/` |
| 5 | 単体テスト（各戦略の判定、ルックアヘッド検査、エンジン） | `tests/` |
| 6 | 開発/検証期間分離・ウォークフォワード・条件別検証・CSV/レポート出力 | `fxbt/walkforward.py`, `fxbt/metrics.py`, `fxbt/report.py`, `scripts/run_backtest.py` |

## ファイル構成

```
config/default.yaml          全パラメーター・期間・コスト・検査しきい値
fxbt/
  config.py                  YAML 読込と妥当性検査
  data/mt5_loader.py         MT5 取得（MetaTrader5 は遅延 import、テストではモック注入）
  data/storage.py            CSV + メタデータ(JSON, SHA256) の保存/読込
  data/validation.py         データ検査。重大異常は DataValidationError で停止
  indicators.py              SMA/EMA/RSI/MACD/ATR/ADX/直近高安（すべて因果的）
  mtf.py                     H4/D1 特徴量を「確定時刻」で H1 に結合
  strategies/base.py         戦略の共通インターフェース・エッジ検出
  strategies/pullback.py     1. トレンド中の押し目買い・戻り売り
  strategies/breakout.py     2. レンジブレイクアウト
  strategies/mean_reversion.py 3. レンジ相場での平均回帰
  backtest/costs.py          スプレッド・スリッページ・スワップ
  backtest/engine.py         H1 足ベースのイベント駆動シミュレーター
  metrics.py                 勝率・期待値・最大DD・t検定・ブートストラップ CI
  walkforward.py             期間分割・ウォークフォワード・条件別アブレーション
  report.py                  CSV / Markdown レポート
scripts/fetch_data.py        段階2: 取得
scripts/validate_data.py     段階3: 検査
scripts/run_backtest.py      段階6: 検証一式
scripts/make_synthetic_data.py  MT5 の無い環境での動作確認用の合成データ（実績ではない）
tests/                       pytest
```

## バイアス対策の設計

* **確定足のみ**: MT5 の `time` は足の開始時刻。開始時刻 + 足の長さ > 取得時点 の足（形成中の足）は破棄。
* **シグナル→執行**: H1 足 *t* の終値までの情報でシグナル判定し、執行は *t+1* の始値。
* **上位足の結合**: H4/D1 の値は「足の終了時刻」をキーにし、H1 足の終了時刻以前に確定したものだけを
  `merge_asof(direction="backward")` で結合する。形成中の H4/D1 は参照しない。
* **ルックアヘッド検査テスト**: 全データで計算したシグナルと、途中で切り捨てたデータで計算したシグナルが
  切り捨て時点まで完全一致することを各戦略で検査する。
* **同一足内で SL と TP の両方に到達**: 保守的に SL 優先。窓開けで SL を越えた場合は始値で約定（不利側）。
* **重複エントリー防止**: シグナルは条件が「偽→真」に変わった足のみ（エッジ）。保有中は新規シグナル無視、
  決済後のクールダウン、同一 `signal_id` の再使用禁止。
* **期間分離**: 開発期間（ウォークフォワード用）と未使用の検証期間（ホールドアウト）を時系列で分離。
  期間境界では強制決済し、境界をまたいで未来の価格を使わない。
* **過剰適合対策**: パラメーター探索は小さな格子に限定し、学習窓のみで選択。評価は常にアウトオブサンプル。
  試行回数を記録し、多重検定（Bonferroni）補正後の p 値も出力。ホールドアウト評価回数をログに記録しレポートに明記。
* **都合のよい期間選択の禁止**: ウォークフォワードの全テスト窓の結果をそのまま連結し、窓ごとの成績も全件出力。

## 取引コスト

* スプレッド（pips、固定値。MT5 の足スプレッドとの大きい方も選択可）
* スリッページ（pips、エントリー・決済の双方）
* スワップ（ロング/ショート別 pips/日、水曜 3 倍、サーバー日付の変わり目で計上）

## 出力指標

取引回数・勝率（Wilson 95%CI）・平均利益・平均損失・期待値（pips / R）・プロフィットファクター・
最大ドローダウン・平均保有時間・t 統計量と p 値・期待値のブートストラップ 95%CI・損益分岐勝率。

## 注意

バックテスト結果は将来の利益を保証しない。結果の良し悪しに関わらず全期間・全戦略を同じ手順で報告する。
