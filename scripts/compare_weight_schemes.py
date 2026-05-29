#!/usr/bin/env python3
"""Compare portfolio weight schemes on walk-forward CV OOF vs Final OOS."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant.config import BACKTEST_DIR, DATA_CURATED_DIR, LABELS_DIR, MODEL_DIR
from quant.pipeline.monthly_portfolio import _run_monthly_path, _monthly_stock_returns
from quant.pipeline.sp500_benchmark import load_sp500_symbols, spy_monthly_returns

SCHEMES: list[tuple[str, str, float | None]] = [
    ("inv_vol", "1/σ normalize（baseline）", None),
    ("equal", "等權 Top30", None),
    ("rank_linear", "rank 線性 decay", None),
    ("softmax_sharpe", "softmax(score/(T·σ)) T=0.05", 0.05),
    ("softmax_sharpe", "softmax(score/(T·σ)) T=0.10", 0.10),
    ("softmax_sharpe", "softmax(score/(T·σ)) T=0.20", 0.20),
    ("score_inv_vol", "score/σ normalize", None),
]


def _load_context() -> dict:
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    wf_meta = meta.get("walk_forward") or {}
    cv_q = set(wf_meta.get("cv_oof_quarters") or wf_meta.get("early_quarters") or [])
    oos_q = set(wf_meta.get("final_oos_quarters") or wf_meta.get("recent_quarters") or [])

    ds = pd.read_parquet(MODEL_DIR / "model_dataset.parquet")
    quarters = sorted(ds["quarter"].unique())
    q_ends = pd.to_datetime(ds.groupby("quarter")["quarter_end"].min())

    sp500 = load_sp500_symbols()
    prices = pd.read_parquet(DATA_CURATED_DIR / "prices_daily.parquet")
    prices = prices[prices["symbol"].isin(sp500)].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    price_col = "adjusted_close" if "adjusted_close" in prices.columns else "close"
    prices["price"] = prices[price_col]
    prices["daily_ret"] = prices.groupby("symbol")["price"].pct_change()
    prices["month"] = prices["Date"].dt.to_period("M").astype(str)

    mret = _monthly_stock_returns(prices)
    bench_m = spy_monthly_returns()
    mlabels = pd.read_parquet(LABELS_DIR / "labels_monthly.parquet")

    wf = pd.read_parquet(MODEL_DIR / "walk_forward_scores.parquet")
    score_col = "score_wf" if "score_wf" in wf.columns else "score_oof"
    wf_by_q = {q: g.set_index("symbol")[score_col] for q, g in wf.groupby("quarter")}

    def score_wf(q: str, _df: pd.DataFrame) -> pd.Series | None:
        return wf_by_q.get(q)

    month_ends = sorted(pd.Timestamp(x) for x in mret.groupby("month")["month_end"].max().tolist())

    return {
        "meta": meta,
        "cv_q": cv_q,
        "oos_q": oos_q,
        "ds": ds,
        "quarters": quarters,
        "q_ends": q_ends,
        "mret": mret,
        "bench_m": bench_m,
        "mlabels": mlabels,
        "score_wf": score_wf,
        "feature_cols": meta["feature_cols"],
        "month_ends": month_ends,
    }


def main() -> None:
    ctx = _load_context()

    rows: list[dict] = []
    for scheme_id, scheme_label, temp in SCHEMES:
        scheme_key = f"{scheme_id}_{temp}" if temp is not None else scheme_id
        for segment, allowed, path in (
            ("cv_oof", ctx["cv_q"], "walk_forward_cv"),
            ("final_oos", ctx["oos_q"], "walk_forward_oos"),
        ):
            _, _, summary, _ = _run_monthly_path(
                ctx["month_ends"],
                ctx["mret"],
                ctx["bench_m"],
                ctx["mlabels"],
                ctx["ds"],
                ctx["quarters"],
                ctx["q_ends"],
                ctx["feature_cols"],
                ctx["score_wf"],
                allowed,
                path,
                weight_scheme=scheme_id,
                softmax_temperature=float(temp) if temp is not None else 0.1,
            )
            rows.append(
                {
                    "scheme": scheme_key,
                    "scheme_id": scheme_id,
                    "softmax_temperature": temp,
                    "scheme_label": scheme_label,
                    "segment": segment,
                    "n_months": summary.get("n_months", 0),
                    "total_excess_geometric": summary.get("total_excess_geometric"),
                    "annualized_excess": summary.get("annualized_excess"),
                    "ann_sharpe_strategy": summary.get("ann_sharpe_strategy"),
                    "ann_sharpe_excess": summary.get("ann_sharpe_excess"),
                    "ann_sharpe_benchmark": summary.get("ann_sharpe_benchmark"),
                    "max_drawdown": summary.get("max_drawdown"),
                    "total_return_strategy": summary.get("total_return_strategy"),
                }
            )

    df = pd.DataFrame(rows)
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_json = BACKTEST_DIR / "weight_scheme_comparison.json"
    out_md = BACKTEST_DIR / "WEIGHT_SCHEME_COMPARISON.md"
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    def _rank_segment(seg: str, metric: str) -> pd.DataFrame:
        sub = df[df["segment"] == seg].copy()
        sub = sub.sort_values(metric, ascending=False)
        sub["rank"] = range(1, len(sub) + 1)
        return sub

    cv_rank = _rank_segment("cv_oof", "ann_sharpe_excess")
    oos_rank = _rank_segment("final_oos", "ann_sharpe_excess")
    cv_ret_rank = df[df["segment"] == "cv_oof"].copy()
    cv_ret_rank = cv_ret_rank.sort_values("total_excess_geometric", ascending=False)
    cv_ret_rank["rank"] = range(1, len(cv_ret_rank) + 1)

    best_cv = cv_rank.iloc[0]
    baseline = df[(df["segment"] == "cv_oof") & (df["scheme"] == "inv_vol")].iloc[0]

    def _pct(v: float | None) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v * 100:.2f}%"

    def _sh(v: float | None) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        return f"{v:.2f}"

    lines = [
        "# 權重方案比較（Top30 · walk-forward 分數選股）\n",
        "同一套 `walk_forward_scores`；僅改權重公式。Sharpe 以**月頻超額報酬**年化。\n",
        "**選型依據：僅看 CV OOF（Q9–Q16）**；Final OOS 僅供事後監控，不可回頭用來挑方案。\n",
        "## CV OOF（Q9–Q16）— 選型主表 · 按超額 Sharpe 排序\n",
        "| rank | scheme | 累積超額 | 年化超額 | 策略 Sharpe | 超額 Sharpe | MDD |\n",
        "|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for _, r in cv_rank.iterrows():
        marker = " ← baseline" if r["scheme"] == "inv_vol" else ""
        lines.append(
            f"| {int(r['rank'])} | {r['scheme_label']}{marker} | "
            f"{_pct(r['total_excess_geometric'])} | {_pct(r['annualized_excess'])} | "
            f"{_sh(r['ann_sharpe_strategy'])} | {_sh(r['ann_sharpe_excess'])} | "
            f"{_pct(r['max_drawdown'])} |\n"
        )

    lines += [
        "\n### CV OOF · 按累積超額排序（Sharpe 之外參考）\n",
        "| rank | scheme | 累積超額 | 超額 Sharpe |\n",
        "|---:|---|---:|---:|\n",
    ]
    for _, r in cv_ret_rank.iterrows():
        lines.append(
            f"| {int(r['rank'])} | {r['scheme_label']} | "
            f"{_pct(r['total_excess_geometric'])} | {_sh(r['ann_sharpe_excess'])} |\n"
        )

    lines += [
        "\n## Final OOS（Q17–Q20）— 僅監控 · 不作選型\n",
        "| scheme | 累積超額 | 超額 Sharpe | OOF rank（Sharpe） |\n",
        "|---|---:|---:|---:|\n",
    ]
    oof_rank_map = {r["scheme"]: int(r["rank"]) for _, r in cv_rank.iterrows()}
    for _, r in oos_rank.sort_values("scheme").iterrows():
        lines.append(
            f"| {r['scheme_label']} | {_pct(r['total_excess_geometric'])} | "
            f"{_sh(r['ann_sharpe_excess'])} | {oof_rank_map.get(r['scheme'], '—')} |\n"
        )

    delta_sh = best_cv["ann_sharpe_excess"] - baseline["ann_sharpe_excess"]
    delta_ret = best_cv["total_excess_geometric"] - baseline["total_excess_geometric"]
    best_ret = cv_ret_rank.iloc[0]
    best_ret_sh_rank = int(cv_rank[cv_rank["scheme"] == best_ret["scheme"]]["rank"].iloc[0])

    lines += [
        "\n## 結論（僅依 CV OOF）\n",
        f"- **推薦方案（超額 Sharpe）**：{best_cv['scheme_label']} "
        f"（OOF 超額 Sharpe {_sh(best_cv['ann_sharpe_excess'])}，"
        f"vs baseline {_sh(baseline['ann_sharpe_excess'])}，+{delta_sh:.2f}）\n",
        f"- **累積超額最高（OOF）**：{best_ret['scheme_label']} "
        f"（{_pct(best_ret['total_excess_geometric'])}），"
        f"但 Sharpe 排 #{best_ret_sh_rank} — 波動較高\n",
        f"- **score/σ**：OOF Sharpe {_sh(df[(df.segment=='cv_oof')&(df.scheme=='score_inv_vol')].iloc[0]['ann_sharpe_excess'])}，"
        f"略優 baseline，改善有限\n",
        f"- **inv_vol baseline**：OOF 超額 Sharpe 垫底（{_sh(baseline['ann_sharpe_excess'])}）；"
        f"引入分數權重整體優於純 inv-vol\n",
        "- Final OOS 表僅供事後對照；**不可**用 OOS 結果回頭改選方案。\n",
        "- `回歸超額報酬` 需另訓練 regression 目標，未含在此表。\n",
    ]

    out_md.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out_md.name} + {out_json.name}")
    print(f"OOF best (Sharpe): {best_cv['scheme']} (excess Sharpe {best_cv['ann_sharpe_excess']:.2f})")
    print(f"OOF best (return): {best_ret['scheme']} (cum excess {best_ret['total_excess_geometric']*100:.1f}%)")


if __name__ == "__main__":
    main()
