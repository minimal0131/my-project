"""検証一式（データ検査 → 3 戦略の開発期間評価・ウォークフォワード・アブレーション → 任意でホールドアウト）。"""
from __future__ import annotations

import hashlib
import inspect
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from .backtest.costs import CostModel
from .data.validation import DataValidationError, validate_dataset
from .metrics import compute_metrics, yearly_breakdown
from .mtf import MarketData
from .report import LABELS, md_table, metrics_row, write_csv
from .strategies import REGISTRY
from .walkforward import StrategyRunner, ablation, end_of_day, param_grid, select_params, walk_forward

log = logging.getLogger(__name__)


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _holdout_history(log_path: Path, strategy: str, cfg_period: str) -> int:
    if not log_path.exists():
        return 0
    df = pd.read_csv(log_path)
    return int(((df["strategy"] == strategy) & (df["holdout_period"] == cfg_period)).sum())


def run(cfg: dict, frames: dict[str, pd.DataFrame], metas: dict[str, dict] | None = None,
        evaluate_holdout: bool = False, out_root: str | Path | None = None,
        strategies: list[str] | None = None) -> Path:
    metas = metas or {}
    synthetic = any(m.get("synthetic") for m in metas.values())

    # ---- データ検査（異常時は停止）
    vreport = validate_dataset(frames, cfg, metas)
    if not vreport.ok:
        raise DataValidationError(vreport)

    md = MarketData(frames["H1"], frames["H4"], frames["D1"])
    costs = CostModel.from_config(cfg)
    per, st = cfg["periods"], cfg.get("stats", {})
    dev_s, dev_e = pd.Timestamp(per["development_start"]), end_of_day(per["development_end"])
    ho_s, ho_e = pd.Timestamp(per["holdout_start"]), end_of_day(per["holdout_end"])
    n_boot, seed, alpha = st.get("bootstrap_samples", 5000), st.get("random_seed", 42), st.get("significance_level", 0.05)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = Path(out_root or cfg["output"]["dir"])
    out = out_root / f"run_{run_id}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")

    names = [n for n in (strategies or REGISTRY) if cfg["strategies"].get(n, {}).get("enabled", True)]
    n_hyp = len(names)  # 最終判定で検定する仮説数（Bonferroni 補正に使う）
    summary_rows, sections = [], []

    for name in names:
        log.info("戦略 %s: 開発期間の評価", name)
        runner = StrategyRunner(name, cfg, md, costs)
        rule_doc = (inspect.getmodule(REGISTRY[name]).__doc__ or "").strip()
        sec = [f"## 戦略: {name}", "", "```", rule_doc, "```", ""]

        dev_trades = runner.backtest({}, dev_s, dev_e)
        m_dev = compute_metrics(dev_trades, n_boot, seed)

        log.info("戦略 %s: ウォークフォワード", name)
        wf_table, wf_trades = walk_forward(runner, cfg)
        m_wf = compute_metrics(wf_trades, n_boot, seed)

        log.info("戦略 %s: 条件別アブレーション", name)
        abl = ablation(runner, dev_s, dev_e)
        abl_rows = []
        base_exp = compute_metrics(abl[0][2], 0)["expectancy_pips"]
        for label, ov, tr in abl:
            m = compute_metrics(tr, n_boot, seed)
            abl_rows.append({"条件": label, "n_trades": m["n_trades"], "win_rate": m["win_rate"],
                             "expectancy_pips": m["expectancy_pips"], "期待値の差(pips)": m["expectancy_pips"] - base_exp,
                             "profit_factor": m["profit_factor"], "max_drawdown_pips": m["max_drawdown_pips"],
                             "p_value": m["p_value"]})
        abl_df = pd.DataFrame(abl_rows)

        rows = [metrics_row("開発期間・設定値（参考: 同じ期間で条件を検討しているため楽観的）", m_dev),
                metrics_row("ウォークフォワード（全検証窓の連結・アウトオブサンプル）", m_wf)]
        ho_trades = pd.DataFrame()
        m_ho: dict | None = None
        chosen = None
        if evaluate_holdout:
            grid = param_grid(cfg["strategies"][name].get("grid"))
            chosen, _ = select_params(runner, grid, dev_s, dev_e, cfg["walkforward"].get("objective", "expectancy_r"),
                                      cfg["walkforward"].get("min_train_trades", 30))
            if chosen is None:
                chosen = {}
            ho_trades = runner.backtest(chosen, ho_s, ho_e)
            m_ho = compute_metrics(ho_trades, n_boot, seed)
            rows.append(metrics_row("ホールドアウト（未使用期間・1 回のみ評価）", m_ho))
            period_key = f"{per['holdout_start']}..{per['holdout_end']}"
            log_path = out_root / "holdout_log.csv"
            prior = _holdout_history(log_path, name, period_key)
            entry = pd.DataFrame([{"run_id": run_id, "strategy": name, "holdout_period": period_key,
                                   "config_hash": config_hash(cfg), "params": json.dumps(chosen, sort_keys=True),
                                   "n_trades": m_ho["n_trades"], "expectancy_pips": m_ho["expectancy_pips"],
                                   "synthetic": synthetic}])
            entry.to_csv(log_path, mode="a", header=not log_path.exists(), index=False, encoding="utf-8")
            m_ho["prior_holdout_evaluations"] = prior

        # ---- CSV
        write_csv(dev_trades, out / f"trades_{name}_development.csv")
        write_csv(wf_trades, out / f"trades_{name}_walkforward_oos.csv")
        write_csv(wf_table, out / f"walkforward_windows_{name}.csv")
        write_csv(abl_df, out / f"ablation_{name}.csv")
        if evaluate_holdout:
            write_csv(ho_trades, out / f"trades_{name}_holdout.csv")

        # ---- 判定
        p_wf_adj = min(1.0, m_wf["p_value"] * n_hyp) if m_wf["n_trades"] >= 2 else float("nan")
        verdict = "優位性は確認できない"
        if m_wf["n_trades"] >= 30 and m_wf["expectancy_pips"] > 0 and p_wf_adj < alpha:
            verdict = "ウォークフォワードでは有意（ホールドアウト未評価）"
            if m_ho is not None:
                p_ho_adj = min(1.0, m_ho["p_value"] * n_hyp) if m_ho["n_trades"] >= 2 else float("nan")
                verdict = ("ホールドアウトでも有意な正の期待値" if m_ho["expectancy_pips"] > 0 and p_ho_adj < alpha
                           else "ホールドアウトで再現せず")
        summary = {"strategy": name, "verdict": verdict,
                   **{f"wf_{k}": m_wf[k] for k in ("n_trades", "win_rate", "expectancy_pips", "expectancy_r",
                                                  "profit_factor", "max_drawdown_pips", "p_value")},
                   "wf_p_adj": p_wf_adj, "wf_exp_ci_low": m_wf["exp_ci_low"], "wf_exp_ci_high": m_wf["exp_ci_high"],
                   "dev_n_trades": m_dev["n_trades"], "dev_win_rate": m_dev["win_rate"],
                   "dev_expectancy_pips": m_dev["expectancy_pips"]}
        if m_ho is not None:
            summary.update({f"ho_{k}": m_ho[k] for k in ("n_trades", "win_rate", "expectancy_pips", "profit_factor",
                                                        "max_drawdown_pips", "p_value")},
                           ho_p_adj=min(1.0, m_ho["p_value"] * n_hyp) if m_ho["n_trades"] >= 2 else float("nan"),
                           ho_params=json.dumps(chosen, sort_keys=True),
                           ho_prior_evaluations=m_ho["prior_holdout_evaluations"])
        summary_rows.append(summary)

        # ---- レポート本文
        sec += ["### 成績", "", md_table(pd.DataFrame(rows), LABELS)]
        sec += [f"ウォークフォワード期待値の 95% ブートストラップ区間: "
                f"{m_wf['exp_ci_low']:.2f} 〜 {m_wf['exp_ci_high']:.2f} pips / 勝率の 95% 区間: "
                f"{m_wf['win_rate_ci_low']:.1%} 〜 {m_wf['win_rate_ci_high']:.1%}" if m_wf["n_trades"] >= 2 else "", ""]
        if m_ho is not None and m_ho["prior_holdout_evaluations"] > 0:
            sec += [f"> ⚠ このホールドアウト期間は過去に {m_ho['prior_holdout_evaluations']} 回評価されています。"
                    "結果を見て設定を変えた場合、ホールドアウトはもはや未使用データではありません。", ""]
        sec += ["### ウォークフォワード窓ごとの結果（全窓）", "", md_table(wf_table)]
        yearly = yearly_breakdown(pd.concat([wf_trades.assign(区分="WF"), ho_trades.assign(区分="ホールドアウト")]
                                            if m_ho is not None else [wf_trades]))
        sec += ["### 年別（ウォークフォワード検証窓" + (" + ホールドアウト" if m_ho is not None else "") + "、全年）", "",
                md_table(yearly, LABELS)]
        sec += ["### 条件別の有効性（開発期間・設定値から 1 条件ずつ変更）", "",
                "期待値の差が小さい / 負の条件は、その条件が成績に寄与していない可能性を示す。"
                "ただし開発期間内での比較なので、これを見て条件を選ぶと過剰適合になりうる。", "",
                md_table(abl_df, LABELS)]
        sections.append("\n".join(sec))

    summary_df = pd.DataFrame(summary_rows)
    write_csv(summary_df, out / "summary.csv")
    _write_report(out, cfg, vreport, summary_df, sections, synthetic, evaluate_holdout, n_hyp, alpha, costs)
    log.info("出力先: %s", out)
    return out


def _write_report(out: Path, cfg: dict, vreport, summary: pd.DataFrame, sections: list[str], synthetic: bool,
                  evaluate_holdout: bool, n_hyp: int, alpha: float, costs: CostModel) -> None:
    per, wf, c = cfg["periods"], cfg["walkforward"], cfg["costs"]
    lines = ["# USD/JPY スイング戦略 検証レポート", ""]
    if synthetic:
        lines += ["> ## ⚠ 合成データによる動作確認です", "> 実際の相場データではありません。この結果を戦略の評価に使わないでください。", ""]
    lines += [
        f"- 作成: {datetime.now():%Y-%m-%d %H:%M} / 設定ハッシュ `{config_hash(cfg)}`",
        f"- データ: {cfg['symbol']} H1/H4/D1, {cfg['data']['start']} 〜 {cfg['data']['end']}（サーバー時刻 {cfg['data']['server_timezone']}）",
        f"- 開発期間: {per['development_start']} 〜 {per['development_end']}",
        f"- ホールドアウト: {per['holdout_start']} 〜 {per['holdout_end']}"
        + ("" if evaluate_holdout else "（**今回は未評価**。`--evaluate-holdout` で 1 回だけ評価）"),
        f"- ウォークフォワード: 学習 {wf['train_months']} か月 → 検証 {wf['test_months']} か月、{wf['step_months']} か月ずつ移動、"
        f"学習期間の取引が {wf.get('min_train_trades', 30)} 回未満のパラメーターは不採用",
        f"- 取引コスト: スプレッド {c['spread_pips']} pips"
        + ("（足の spread 列との大きい方）" if c.get("use_bar_spread") else "")
        + f"、スリッページ 片道 {c['slippage_pips']} pips、スワップ 買い {c.get('swap_long_pips_per_day', 0)} / "
          f"売り {c.get('swap_short_pips_per_day', 0)} pips/日（水曜 3 倍）",
        f"- 執行: シグナル足の次の H1 始値で約定、同一足で損切り・利確が両方到達した場合は損切り、"
        f"決済後 {cfg['backtest'].get('cooldown_bars', 0)} 本は新規エントリーなし",
        f"- データ検査: {'合格' if vreport.ok else '不合格'}（警告 {sum(i.level == 'WARNING' for i in vreport.issues)} 件）",
        "",
        "## 結論", "",
        f"判定基準: ウォークフォワードの取引が 30 回以上、期待値 > 0、Bonferroni 補正（×{n_hyp}）後の p 値 < {alpha}。"
        "ホールドアウトを評価した場合は、そこでも同じ基準を満たすこと。", "",
    ]
    cols = ["strategy", "verdict", "wf_n_trades", "wf_win_rate", "wf_expectancy_pips", "wf_profit_factor",
            "wf_max_drawdown_pips", "wf_p_value", "wf_p_adj"]
    if evaluate_holdout and "ho_n_trades" in summary:
        cols += ["ho_n_trades", "ho_win_rate", "ho_expectancy_pips", "ho_p_adj"]
    summary_labels = {"strategy": "戦略", "verdict": "判定", "wf_n_trades": "WF取引数", "wf_win_rate": "WF勝率",
                      "wf_expectancy_pips": "WF期待値(pips)", "wf_profit_factor": "WF PF",
                      "wf_max_drawdown_pips": "WF最大DD(pips)", "wf_p_value": "WF p値", "wf_p_adj": "WF 補正p値",
                      "ho_n_trades": "HO取引数", "ho_win_rate": "HO勝率", "ho_expectancy_pips": "HO期待値(pips)",
                      "ho_p_adj": "HO 補正p値"}
    lines += [md_table(summary[cols] if not summary.empty else summary, summary_labels),
              "WF = ウォークフォワード（アウトオブサンプル）、HO = ホールドアウト、補正p値 = Bonferroni 補正後", ""]
    if not summary.empty:
        best_wr = summary["wf_win_rate"].max()
        lines += [
            "### 勝率について",
            "",
            f"ウォークフォワードで観測された最高の勝率は {best_wr:.1%} です。"
            + ("目標とされた 90% には届いていません。" if not best_wr >= 0.9 else "")
            + "勝率は利確幅と損切り幅の比でいくらでも変わる（利確を近くすれば勝率は上がり、1 回の損失が大きくなる）ため、"
              "優位性の判断には期待値と損益分岐勝率の比較を用いています。", "",
        ]
    lines += [
        "### 注意事項", "",
        "- バックテストの結果は将来の利益を保証しません。相場環境・スプレッド・スワップは変化します。",
        "- ウォークフォワードの全検証窓、全年を省略せず掲載しています。成績の良い期間だけを抜き出していません。",
        f"- パラメーター探索は戦略ごとに小さな格子（{', '.join(f'{k}: {len(param_grid(v.get('grid')))} 通り' for k, v in cfg['strategies'].items())}）"
        "に限定しています。格子を広げる・条件を足し引きするほど、偶然よく見える組を選ぶ危険が増えます。",
        "- 「開発期間・設定値」の行は、条件を検討したのと同じ期間での成績なので楽観的です。判断にはウォークフォワードとホールドアウトを使ってください。",
        "- p 値は両側検定（H0: 期待値 = 0）です。期待値が負でも p 値は小さくなるため、必ず期待値の符号と合わせて読んでください。",
        "- 本ツールは実口座・デモ口座への発注機能を持ちません。",
        "",
    ]
    if vreport.issues:
        lines += ["## データ検査の検出事項", ""]
        lines += [f"- [{i.level}] {i.timeframe} `{i.check}`: {i.message}" for i in vreport.issues] + [""]
    lines += sections
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
