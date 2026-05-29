"""Compute in-sample (CV OOF) vs out-of-sample portfolio returns."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from quant.config import BENCHMARK_ID, HP_POOL_TOP_K, MODEL_DIR, TOP_N_HOLDINGS, WF_OOS_QUARTERS
from quant.pipeline.sp500_benchmark import spy_quarterly_returns

RET_CLIP_LOWER, RET_CLIP_UPPER = -0.9, 3.0


def _run_backtest_with_scores(
    ds: pd.DataFrame,
    quarters: list[str],
    rows_by_q: dict[str, pd.DataFrame],
    score_by_q: dict[str, pd.Series],
    bench_by_q: pd.Series,
    signal_quarters: set[str],
) -> list[dict]:
    rows: list[dict] = []
    strat_val = 1.0
    bench_val = 1.0
    for q in sorted(signal_quarters):
        if q not in rows_by_q or q not in score_by_q:
            continue
        i = quarters.index(q)
        if i >= len(quarters) - 1:
            continue
        next_q = quarters[i + 1]
        current = rows_by_q[q].copy()
        next_rows = rows_by_q[next_q]
        if current.empty or next_rows.empty:
            continue

        scores = score_by_q[q]
        current = current.set_index("symbol")
        current["score"] = current.index.map(scores)
        current = current.dropna(subset=["score"]).reset_index()
        if current.empty:
            continue

        top = current.nlargest(TOP_N_HOLDINGS, "score")
        selected = next_rows[next_rows["symbol"].isin(top["symbol"])]
        if selected.empty or next_q not in bench_by_q.index:
            continue

        strat_ret = float(selected["return_capped"].mean())
        bench_ret = float(np.clip(float(bench_by_q.loc[next_q]), RET_CLIP_LOWER, RET_CLIP_UPPER))
        strat_val *= 1 + strat_ret
        bench_val *= 1 + bench_ret
        excess_ret = strat_ret - bench_ret
        excess_nav = strat_val / bench_val - 1.0 if bench_val > 0 else float("nan")
        rows.append(
            {
                "signal_quarter": q,
                "realized_quarter": next_q,
                "strategy_ret": strat_ret,
                "benchmark_ret": bench_ret,
                "excess_ret": excess_ret,
                "strategy_nav": strat_val,
                "benchmark_nav": bench_val,
                "excess_nav": excess_nav,
            }
        )
    return rows


def _summarize(label: str, rows: list[dict], eval_note: str = "") -> dict:
    if not rows:
        return {"label": label, "n_quarters": 0, "eval_note": eval_note}
    n = len(rows)
    strat_total = rows[-1]["strategy_nav"] - 1
    bench_total = rows[-1]["benchmark_nav"] - 1
    excess_total_arith = strat_total - bench_total
    excess_total_geom = rows[-1]["excess_nav"]
    years = n / 4
    strat_ann = rows[-1]["strategy_nav"] ** (1 / years) - 1 if years > 0 else float("nan")
    bench_ann = rows[-1]["benchmark_nav"] ** (1 / years) - 1 if years > 0 else float("nan")
    excess_ann_arith = strat_ann - bench_ann if years > 0 else float("nan")
    excess_ann_geom = (1 + excess_total_geom) ** (1 / years) - 1 if years > 0 else float("nan")
    strat_rets = [r["strategy_ret"] for r in rows]
    bench_rets = [r["benchmark_ret"] for r in rows]
    strat_vol = float(np.std(strat_rets, ddof=1) * np.sqrt(4)) if n > 1 else float("nan")
    bench_vol = float(np.std(bench_rets, ddof=1) * np.sqrt(4)) if n > 1 else float("nan")
    strat_sharpe = float((np.mean(strat_rets) * 4) / strat_vol) if strat_vol > 0 else float("nan")
    bench_sharpe = float((np.mean(bench_rets) * 4) / bench_vol) if bench_vol > 0 else float("nan")
    return {
        "label": label,
        "eval_note": eval_note,
        "n_quarters": n,
        "signal_quarters": sorted({r["signal_quarter"] for r in rows}),
        "realized_quarters": [r["realized_quarter"] for r in rows],
        "total_return_strategy": strat_total,
        "total_return_benchmark": bench_total,
        "total_return_excess": excess_total_arith,
        "total_return_excess_geometric": excess_total_geom,
        "annualized_strategy": strat_ann,
        "annualized_benchmark": bench_ann,
        "annualized_excess": excess_ann_arith,
        "annualized_excess_geometric": excess_ann_geom,
        "benchmark_label": BENCHMARK_ID,
        "ann_vol_strategy": strat_vol,
        "ann_vol_benchmark": bench_vol,
        "ann_sharpe_strategy": strat_sharpe,
        "ann_sharpe_benchmark": bench_sharpe,
        "quarterly": rows,
    }


def _render_excess_md(payload: dict) -> str:
    lines = [
        "# 策略 vs SPY（S&P 500 市值加權）— 超額報酬",
        "",
        f"- **基準**：SPY 季報酬（`{BENCHMARK_ID}`）",
        f"- **策略**：每季 Top{TOP_N_HOLDINGS} 等權；季報酬 clip [-90%, +300%]",
        "- **單季超額（算術）**：`策略季報酬 − 基準季報酬`",
        "- **累積超額（幾何）**：`策略 NAV / 基準 NAV − 1`",
        "",
    ]
    for key, title in (("in_sample", "CV OOF（expanding walk-forward）"), ("out_of_sample", "Final OOS（凍結 hyperparam）")):
        s = payload[key]
        lines.append(f"## {title}")
        lines.append("")
        if s.get("eval_note"):
            lines.append(f"_{s['eval_note']}_")
            lines.append("")
        lines.append("| 指標 | 策略 | SPY | 超額 |")
        lines.append("|------|-----:|----------:|-----:|")
        lines.append(
            f"| 累積報酬 | {s['total_return_strategy']*100:.2f}% | "
            f"{s['total_return_benchmark']*100:.2f}% | "
            f"{s['total_return_excess']*100:.2f}%（算術差）|"
        )
        lines.append(
            f"| 累積超額（幾何） | — | — | **{s.get('total_return_excess_geometric', 0)*100:.2f}%** |"
        )
        lines.append(
            f"| 年化報酬 | {s['annualized_strategy']*100:.2f}% | "
            f"{s['annualized_benchmark']*100:.2f}% | "
            f"{s['annualized_excess']*100:.2f}%（年化差）|"
        )
        lines.append("")
        lines.append("| 實現季 | 策略 | S&P500 | 超額（單季） | 累積超額 NAV |")
        lines.append("|--------|-----:|-------:|-------------:|-------------:|")
        for r in s.get("quarterly", []):
            lines.append(
                f"| {r['realized_quarter']} | {r['strategy_ret']*100:+.2f}% | "
                f"{r['benchmark_ret']*100:+.2f}% | {r['excess_ret']*100:+.2f}% | "
                f"{r['excess_nav']*100:+.2f}% |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    wf_meta = meta.get("walk_forward") or {}

    ds = pd.read_parquet(MODEL_DIR / "model_dataset.parquet").copy()
    ds["return_capped"] = ds["return"].clip(lower=RET_CLIP_LOWER, upper=RET_CLIP_UPPER)
    quarters = sorted(ds["quarter"].unique())
    bench_by_q = spy_quarterly_returns()

    rows_by_q = {q: ds[ds["quarter"] == q].copy() for q in quarters}

    wf_path = MODEL_DIR / "walk_forward_scores.parquet"
    if not wf_path.exists():
        wf_path = MODEL_DIR / "cv_oof_scores.parquet"
    if not wf_path.exists():
        raise FileNotFoundError(f"Missing walk-forward scores; run train_model(force=True) first.")
    wf = pd.read_parquet(wf_path)
    score_col = "score_wf" if "score_wf" in wf.columns else "score_oof"
    score_by_q: dict[str, pd.Series] = {}
    for q, g in wf.groupby("quarter"):
        score_by_q[q] = g.set_index("symbol")[score_col]

    wf_quarters = set(wf_meta.get("predicted_quarters") or score_by_q.keys())
    oos_q = set(wf_meta.get("final_oos_quarters") or wf_meta.get("recent_quarters") or list(sorted(wf_quarters))[-WF_OOS_QUARTERS:])
    cv_q = set(wf_meta.get("cv_oof_quarters") or wf_meta.get("early_quarters") or (wf_quarters - oos_q))

    is_rows = _run_backtest_with_scores(ds, quarters, rows_by_q, score_by_q, bench_by_q, cv_q)
    oos_rows = _run_backtest_with_scores(ds, quarters, rows_by_q, score_by_q, bench_by_q, oos_q)

    min_q = wf_meta.get("min_train_quarters") or meta.get("cv", {}).get("min_train_quarters", 8)
    is_note = (
        f"CV OOF: top-{HP_POOL_TOP_K} hyperparam pool, each config evaluated on 8-fold expanding OOF "
        f"(mean fold log_loss); scores from selected config."
    )
    oos_note = (
        f"Final OOS: last {len(oos_q)} quarters, frozen hyperparameters from CV selection, "
        "expanding retrain each fold."
    )

    is_sum = _summarize("walk_forward_cv", is_rows, is_note)
    oos_sum = _summarize("walk_forward_oos", oos_rows, oos_note)

    out_path = MODEL_DIR / "is_oos_returns.json"
    payload = {
        "benchmark": BENCHMARK_ID,
        "eval_method": meta.get("eval_method", "expanding_walk_forward"),
        "in_sample": is_sum,
        "out_of_sample": oos_sum,
        "walk_forward": wf_meta,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    md_path = MODEL_DIR / "EXCESS_VS_SP500.md"
    md_path.write_text(_render_excess_md(payload), encoding="utf-8")

    def pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    for s in (is_sum, oos_sum):
        print(f"\n=== {s['label'].upper()} ({s.get('n_quarters', 0)} realized quarters) ===")
        if s.get("eval_note"):
            print(s["eval_note"])
        if s.get("n_quarters", 0) == 0:
            continue
        print(
            f"Total return: strategy {pct(s['total_return_strategy'])} | "
            f"SPY {pct(s['total_return_benchmark'])} | "
            f"excess(arith) {pct(s['total_return_excess'])} | "
            f"excess(geom) {pct(s.get('total_return_excess_geometric', 0))}"
        )
        print(
            f"Annualized:   strategy {pct(s['annualized_strategy'])} | "
            f"benchmark {pct(s['annualized_benchmark'])} | "
            f"excess {pct(s['annualized_excess'])}"
        )
        print(
            f"Ann Sharpe:   strategy {s['ann_sharpe_strategy']:.2f} | "
            f"benchmark {s['ann_sharpe_benchmark']:.2f}"
        )
        print("Per-quarter (signal -> realized):")
        for r in s["quarterly"]:
            print(
                f"  {r['signal_quarter']} -> {r['realized_quarter']}: "
                f"strat {r['strategy_ret']*100:+.2f}% | SP500 {r['benchmark_ret']*100:+.2f}% | "
                f"excess {r['excess_ret']*100:+.2f}%"
            )

    cv = meta.get("walk_forward") or meta.get("cv") or {}
    if cv.get("wf_log_loss") is not None or cv.get("oof_log_loss") is not None:
        print(
            f"\nWalk-forward (all predicted quarters): "
            f"log_loss={cv.get('wf_log_loss') or cv.get('oof_log_loss')} "
            f"AUC={cv.get('wf_roc_auc') or cv.get('oof_roc_auc')}"
        )

    print(f"\nWrote {out_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
