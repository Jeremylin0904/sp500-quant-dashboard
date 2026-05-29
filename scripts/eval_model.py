#!/usr/bin/env python3
"""Expanding walk-forward evaluation: CV OOF + final OOS (Top-k, confusion matrix, Spearman)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant.config import TOP_N_HOLDINGS, TOP_N_LABEL, WF_OOS_QUARTERS


def _prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec}


def _confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)
    return {
        "tn": int((~yt & ~yp).sum()),
        "fp": int((~yt & yp).sum()),
        "fn": int((yt & ~yp).sum()),
        "tp": int((yt & yp).sum()),
    }


def _cm_metrics(cm: dict[str, int]) -> dict[str, float]:
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and (prec + rec) else float("nan")
    return {"accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1, "n": total}


def _pred_topk_by_quarter(df: pd.DataFrame, k: int) -> np.ndarray:
    rank = df.groupby("quarter")["pred"].rank(ascending=False, method="first")
    return (rank <= k).to_numpy(dtype=bool)


def _render_cm_md(title: str, cm: dict[str, int], metrics: dict[str, float]) -> list[str]:
    lines = [
        f"### {title}\n\n",
        "|  | Pred 0 | Pred 1 |\n",
        "|---|---:|---:|\n",
        f"| **Actual 0** | {cm['tn']} | {cm['fp']} |\n",
        f"| **Actual 1** | {cm['fn']} | {cm['tp']} |\n\n",
        f"- Accuracy {metrics['accuracy']*100:.2f}% · Precision {metrics['precision']*100:.2f}% · "
        f"Recall {metrics['recall']*100:.2f}% · F1 {metrics['f1']*100:.2f}% · "
        f"Specificity {metrics['specificity']*100:.2f}%\n\n",
    ]
    return lines


def _eval_slice(df: pd.DataFrame, label: str) -> dict:
    yt = df["is_top30_next"].fillna(False).astype(bool).to_numpy()
    cm_threshold = _confusion_counts(yt, df["pred"].to_numpy() >= 0.5)
    cm_top = _confusion_counts(yt, _pred_topk_by_quarter(df, TOP_N_HOLDINGS))
    cm_top30 = _confusion_counts(yt, _pred_topk_by_quarter(df, TOP_N_LABEL))

    cm_by_quarter: dict[str, dict] = {}
    for q, g in df.groupby("quarter"):
        yt_q = g["is_top30_next"].fillna(False).astype(bool).to_numpy()
        cm_q = _confusion_counts(yt_q, _pred_topk_by_quarter(g, TOP_N_HOLDINGS))
        cm_by_quarter[q] = {**cm_q, **_cm_metrics(cm_q)}

    top_ks = [10, 20, 30]
    by_q = []
    agg = {k: {"tp": 0, "fp": 0, "fn": 0} for k in top_ks}
    spearmans = []

    for q, g in df.groupby("quarter"):
        g = g.sort_values("pred", ascending=False).reset_index(drop=True)
        n = len(g)
        n_pos = int(g["is_top30_next"].sum())
        rho = float(spearmanr(g["pred"], g["y_next"], nan_policy="omit").correlation)
        spearmans.append({"quarter": q, "spearman": rho, "n": n})
        row = {"quarter": q, "n": n, "n_pos_top30": n_pos, "pos_rate": n_pos / n if n else 0, "spearman": rho}
        for k in top_ks:
            kk = min(k, n)
            top = g.head(kk)
            tp = int(top["is_top30_next"].sum())
            agg[k]["tp"] += tp
            agg[k]["fp"] += kk - tp
            agg[k]["fn"] += n_pos - tp
            row[f"precision@{k}"] = tp / kk if kk else float("nan")
            row[f"recall@{k}"] = tp / n_pos if n_pos else float("nan")
        by_q.append(row)

    return {
        "label": label,
        "quarters": sorted(df["quarter"].unique()),
        "n_rows": int(len(df)),
        "positive_rate": float(df["is_top30_next"].mean()),
        "confusion_matrix": {
            "threshold_0.5": {**cm_threshold, **_cm_metrics(cm_threshold)},
            f"top{TOP_N_HOLDINGS}_per_quarter": {**cm_top, **_cm_metrics(cm_top)},
            f"top{TOP_N_LABEL}_per_quarter": {**cm_top30, **_cm_metrics(cm_top30)},
            "per_quarter_top30": cm_by_quarter,
        },
        "topk_agg": {f"top{k}": _prf(v["tp"], v["fp"], v["fn"]) for k, v in agg.items()},
        "spearman_mean": float(pd.DataFrame(spearmans)["spearman"].mean()) if spearmans else float("nan"),
        "per_quarter": by_q,
    }


def _quarter_sets(wf_meta: dict, wf_quarters: list[str]) -> tuple[list[str], list[str]]:
    cv_q = wf_meta.get("cv_oof_quarters") or wf_meta.get("early_quarters")
    oos_q = wf_meta.get("final_oos_quarters") or wf_meta.get("recent_quarters")
    if not cv_q or not oos_q:
        oos_q = wf_quarters[-WF_OOS_QUARTERS:]
        cv_q = [q for q in wf_quarters if q not in oos_q]
    return cv_q, oos_q


def main() -> None:
    model_dir = ROOT / "quant" / "model"
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    ds = pd.read_parquet(model_dir / "model_dataset.parquet").copy()

    wf_path = model_dir / "walk_forward_scores.parquet"
    if not wf_path.exists():
        wf_path = model_dir / "cv_oof_scores.parquet"
    wf = pd.read_parquet(wf_path)
    score_col = "score_wf" if "score_wf" in wf.columns else "score_oof"
    pred_map = wf.set_index(["quarter", "symbol"])[score_col]

    ds = ds.sort_values(["quarter", "symbol"]).reset_index(drop=True)
    ds["pred"] = ds.apply(lambda r: pred_map.get((r["quarter"], r["symbol"]), np.nan), axis=1)
    ds = ds.dropna(subset=["pred"]).reset_index(drop=True)

    wf_meta = meta.get("walk_forward") or {}
    wf_quarters = wf_meta.get("predicted_quarters") or sorted(wf["quarter"].unique())
    cv_q, oos_q = _quarter_sets(wf_meta, wf_quarters)

    wf_all = ds[ds["quarter"].isin(wf_quarters)].copy()
    wf_cv = ds[ds["quarter"].isin(cv_q)].copy()
    wf_oos = ds[ds["quarter"].isin(oos_q)].copy()

    report = {
        "eval_method": meta.get("eval_method", "expanding_walk_forward"),
        "walk_forward_quarters": wf_quarters,
        "cv_oof_quarters": cv_q,
        "final_oos_quarters": oos_q,
        "all": _eval_slice(wf_all, "all_walk_forward"),
        "cv_oof": _eval_slice(wf_cv, "cv_oof_expanding") if len(wf_cv) else {},
        "final_oos": _eval_slice(wf_oos, "final_oos_frozen") if len(wf_oos) else {},
        "early": _eval_slice(wf_cv, "cv_oof_expanding") if len(wf_cv) else {},
        "recent": _eval_slice(wf_oos, "final_oos_frozen"),
        "val_quarters": oos_q,
        "n_oos": int(len(wf_oos)),
        "positive_rate": float(wf_oos["is_top30_next"].mean()) if len(wf_oos) else None,
        "confusion_matrix_oos": _eval_slice(wf_oos, "final_oos")["confusion_matrix"] if len(wf_oos) else {},
        "topk_agg": _eval_slice(wf_oos, "final_oos")["topk_agg"] if len(wf_oos) else {},
        "spearman_mean": _eval_slice(wf_oos, "final_oos")["spearman_mean"] if len(wf_oos) else None,
        "per_quarter": _eval_slice(wf_oos, "final_oos")["per_quarter"] if len(wf_oos) else [],
    }

    def _json_safe(o):
        if isinstance(o, float) and (np.isnan(o) or np.isinf(o)):
            return None
        raise TypeError

    (model_dir / "eval_report.json").write_text(
        json.dumps(report, indent=2, default=_json_safe), encoding="utf-8"
    )

    oos = report["final_oos"]
    cm = oos.get("confusion_matrix", {})
    cm30 = cm.get(f"top{TOP_N_LABEL}_per_quarter", {})
    frozen = (wf_meta.get("frozen") or {})
    frozen_est = frozen.get("frozen_estimator") or meta.get("final_oos", {}).get("frozen_estimator")
    best_rank = frozen.get("selected_pool_rank")
    pool_evals = frozen.get("pool_oof_evaluations") or []

    lines = [
        "# Expanding Walk-Forward 評估（Top-30 超額報酬標籤）\n",
        f"- **Hyperparam pool**：Q1–Q16 AutoML {meta.get('automl_settings', {}).get('pool_search_time_budget', 120)}s → top-{meta.get('automl_settings', {}).get('hp_pool_top_k', 5)} configs\n",
        "- **CV OOF**：每個 config 固定 hyperparam 跑 8-fold expanding OOF，取 **mean fold log_loss** 最佳\n",
        f"- **Final OOS**：凍結 hyperparam（{frozen_est or '—'}，pool rank #{best_rank}），Q17–Q20 expanding retrain\n",
        f"- **全部預測季**：{', '.join(wf_quarters)}\n",
        f"- **CV OOF 季**：{', '.join(cv_q)}\n",
        f"- **Final OOS 季**：{', '.join(oos_q)}\n",
        f"- **Final OOS 列數**：{oos.get('n_rows', 0)}\n",
        f"- **Final OOS mean Spearman**：{oos.get('spearman_mean', float('nan')):.4f}\n\n",
    ]
    if pool_evals:
        lines.append("## Hyperparam pool — 8-fold OOF 平均 log_loss\n\n")
        lines.append("| rank | estimator | mean fold log_loss | pooled OOF AUC |\n")
        lines.append("|---:|---|---:|---:|\n")
        for e in sorted(pool_evals, key=lambda x: x["mean_fold_log_loss"]):
            mark = " **← selected**" if e.get("pool_rank") == best_rank else ""
            auc = e.get("pooled_oof_roc_auc")
            auc_s = f"{auc:.3f}" if auc is not None else "—"
            lines.append(
                f"| {e['pool_rank']} | {e['estimator']}{mark} | {e['mean_fold_log_loss']:.4f} | {auc_s} |\n"
            )
        lines.append("\n")

    lines.append("## Final OOS — Confusion matrix（Top30 / 季）\n\n")
    if cm30:
        lines += _render_cm_md(
            f"Pred：每季分數 **Top {TOP_N_LABEL}**",
            {k: cm30[k] for k in ("tn", "fp", "fn", "tp")},
            cm30,
        )

    lines.append("## 每季 Spearman / Top-k（Final OOS）\n\n")
    lines.append("| quarter | n | n_pos | spearman | P@10 | R@10 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for r in oos.get("per_quarter", []):
        lines.append(
            f"| {r['quarter']} | {r['n']} | {r['n_pos_top30']} | {r['spearman']:.4f} | "
            f"{r.get('precision@10', 0)*100:.2f}% | {r.get('recall@10', 0)*100:.2f}% |\n"
        )

    (model_dir / "EVAL_REPORT.md").write_text("".join(lines), encoding="utf-8")
    print("Wrote EVAL_REPORT.md + eval_report.json")


if __name__ == "__main__":
    main()
