from __future__ import annotations

import json
import re
from functools import lru_cache

from quant.config import BACKTEST_DIR, BENCHMARK_ID, BENCHMARK_SYMBOL, MODEL_DIR, TOP_N_HOLDINGS, TOP_N_LABEL


@lru_cache(maxsize=1)
def get_backtest_curve() -> dict:
    path = BACKTEST_DIR / "backtest_curve.json"
    if not path.exists():
        return {"quarterly": [], "daily": []}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_backtest_monthly() -> dict:
    path = BACKTEST_DIR / "backtest_monthly.json"
    if not path.exists():
        return {"monthly_in_sample": [], "monthly_out_of_sample": []}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_backtest_daily() -> dict:
    path = BACKTEST_DIR / "backtest_daily.json"
    if not path.exists():
        return {"daily_in_sample": [], "daily_out_of_sample": []}
    return _load_json_lenient(path)


@lru_cache(maxsize=1)
def get_drawdown_news() -> dict:
    path = BACKTEST_DIR / "drawdown_news.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_performance_report() -> dict:
    path = BACKTEST_DIR / "performance_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_holdings_by_quarter() -> dict:
    path = BACKTEST_DIR / "holdings_by_quarter.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_holdings_by_month() -> dict:
    path = BACKTEST_DIR / "holdings_by_month.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_holdings_by_signal_quarter() -> dict:
    path = BACKTEST_DIR / "holdings_by_signal_quarter.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def get_model_meta() -> dict:
    path = MODEL_DIR / "model_meta.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json_lenient(path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\bNaN\b", "null", text)
    text = re.sub(r"\bInfinity\b", "null", text)
    text = re.sub(r"\b-Infinity\b", "null", text)
    return json.loads(text)


@lru_cache(maxsize=1)
def get_eval_report() -> dict:
    return _load_json_lenient(MODEL_DIR / "eval_report.json")


@lru_cache(maxsize=1)
def get_universe_summary() -> dict:
    """Distinct investable companies overall and per quarter (from model dataset)."""
    import pandas as pd

    from quant.config import MODEL_DIR

    path = MODEL_DIR / "model_dataset.parquet"
    if not path.exists():
        return {}
    ds = pd.read_parquet(path, columns=["symbol", "quarter"])
    per_q = ds.groupby("quarter")["symbol"].nunique().to_dict()
    return {
        "n_companies": int(ds["symbol"].nunique()),
        "per_quarter": {str(q): int(n) for q, n in per_q.items()},
    }


@lru_cache(maxsize=1)
def get_model_variables_doc() -> str:
    path = MODEL_DIR / "MODEL_VARIABLES.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def get_feature_stats() -> list[dict]:
    import csv

    path = MODEL_DIR / "feature_stats.csv"
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out: dict = {"col": r.get("col")}
            for k, v in r.items():
                if k == "col":
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = None
            rows.append(out)
    return rows


def _parse_feature_doc(md: str, stats: list[dict]) -> tuple[str, list[dict]]:
    """Single pass over MODEL_VARIABLES.md section-4 tables:
    - fill live miss%/inf% into the markdown ('(see feature_stats.csv)' / stale values)
    - emit a structured grouped feature list for a clean UI view
    Returns (filled_markdown, groups) where groups = [{title, items:[{feature, formula,
    miss_pct, inf_pct}]}].
    """
    by = {s.get("col"): s for s in stats if s.get("col")}
    feat_re = re.compile(r"`([A-Za-z0-9_]+)`")
    label_re = re.compile(r"^[A-Z]\.\s*")
    out: list[str] = []
    groups: list[dict] = []
    cur_title: str | None = None
    cur_items: list[dict] = []

    def flush():
        nonlocal cur_items
        if cur_title and cur_items:
            groups.append({"title": cur_title, "items": cur_items})
        cur_items = []

    for line in md.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("### "):
            flush()
            cur_title = label_re.sub("", stripped[4:].strip())
        elif stripped.startswith("|") and "`" in line:
            cells = line.split("|")
            if len(cells) >= 6:
                m = feat_re.search(cells[1])
                if m and m.group(1) in by:
                    s = by[m.group(1)]
                    miss = s.get("miss_pct")
                    inf = s.get("inf_pct")
                    if miss is not None:
                        cells[3] = f" {miss:.2f}% "
                    if inf is not None:
                        cells[4] = f" {inf:.4f}% "
                    line = "|".join(cells)
                    cur_items.append(
                        {
                            "feature": m.group(1),
                            "formula": cells[2].strip().strip("`").strip(),
                            "miss_pct": miss,
                            "inf_pct": inf,
                        }
                    )
        out.append(line)
    flush()
    return "\n".join(out), groups


# Model features whose stats are not in feature_stats.csv (still real model inputs).
_EXTRA_FEATURE_FORMULAS = {
    "daily_vol": "該股當季每日報酬標準差（設下限 VOL_FLOOR_DAILY）；同時供 softmax_sharpe 配重的風險項",
}

# Columns that live in model_dataset.parquet but are NOT fed to the model
# (IDs / dates / labels / leakage-prone targets). Mirrors MODEL_VARIABLES.md §3.
_NOT_IN_MODEL_COLS = [
    "symbol",
    "company_name",
    "sector",
    "industry",
    "quarter",
    "quarter_end",
    "as_of_date",
    "operating_cash_flow",
    "capital_expenditure",
    "free_cash_flow",
    "y_t",
    "y_next",
    "return",
    "excess_return",
    "sharpe",
    "is_top30_next",
    "close",
    "adjusted_close",
]

# Which grouped categories are direct levels vs engineered/derived features.
_RAW_GROUP_HINTS = ("原始基本面", "市場 / 價格", "市場/價格")


@lru_cache(maxsize=1)
def build_model_variables() -> dict:
    meta = get_model_meta()
    stats = get_feature_stats()
    filled_md, groups = _parse_feature_doc(get_model_variables_doc(), stats)
    feature_cols = list(meta.get("feature_cols") or [])

    # Tag each group as raw level input vs engineered, and ensure every actual
    # model feature (feature_cols) is represented even if it has no stats row.
    by = {s.get("col"): s for s in stats if s.get("col")}
    for g in groups:
        g["kind"] = "raw" if any(h in g["title"] for h in _RAW_GROUP_HINTS) else "engineered"

    grouped_names = {it["feature"] for g in groups for it in g["items"]}
    extra = [c for c in feature_cols if c not in grouped_names]
    if extra:
        target = next((g for g in groups if "動量" in g["title"]), None)
        bucket = target or {"title": "波動 / 風險", "items": [], "kind": "engineered"}
        for c in extra:
            s = by.get(c, {})
            bucket["items"].append(
                {
                    "feature": c,
                    "formula": _EXTRA_FEATURE_FORMULAS.get(c, ""),
                    "miss_pct": s.get("miss_pct"),
                    "inf_pct": s.get("inf_pct"),
                }
            )
        if target is None and bucket["items"]:
            groups.append(bucket)

    n_in_model = sum(len(g["items"]) for g in groups)
    return {
        "markdown": filled_md,
        "groups": groups,
        "n_grouped_features": n_in_model,
        "not_in_model": _NOT_IN_MODEL_COLS,
        "feature_stats": stats,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_rows": meta.get("n_rows"),
        "target_col": meta.get("target_col", "y_next"),
        "best_estimator": meta.get("best_estimator"),
    }


@lru_cache(maxsize=1)
def get_factor_analysis() -> dict:
    from quant.config import ROOT

    path = ROOT / "factor" / "factor_analysis.json"
    if not path.exists():
        return {}
    return _load_json_lenient(path)


@lru_cache(maxsize=1)
def build_model_summary() -> dict:
    meta = get_model_meta()
    perf = get_performance_report()
    eval_r = get_eval_report()
    wf = meta.get("walk_forward") or {}
    cv = meta.get("cv") or {}
    final_oos = meta.get("final_oos") or {}
    holdout = meta.get("holdout") or {}
    loss = meta.get("loss") or {}
    frozen = wf.get("frozen") or {}
    automl = meta.get("automl_settings") or {}

    cv_eval = eval_r.get("cv_oof") or eval_r.get("early") or {}
    oos_eval = eval_r.get("final_oos") or eval_r.get("recent") or {}

    cv_cm = (cv_eval.get("confusion_matrix") or {}).get(f"top{TOP_N_LABEL}_per_quarter") or {}
    oos_cm = (oos_eval.get("confusion_matrix") or {}).get(f"top{TOP_N_LABEL}_per_quarter") or {}
    top30_cm = oos_cm or (eval_r.get("confusion_matrix_oos") or {}).get(f"top{TOP_N_LABEL}_per_quarter") or {}

    pool_evals = frozen.get("pool_oof_evaluations") or []
    pool_search = frozen.get("pool_search") or {}

    return {
        "model": meta.get("model"),
        "task": meta.get("task", "classification"),
        "eval_method": meta.get("eval_method", "expanding_walk_forward"),
        "walk_forward_method": wf.get("method"),
        "best_estimator": meta.get("best_estimator"),
        "target_col": meta.get("target_col", "y_next"),
        "loss": loss,
        "n_features": len(meta.get("feature_cols") or []),
        "n_rows": meta.get("n_rows"),
        "all_quarters": meta.get("all_quarters"),
        "automl_settings": automl,
        "hp_pool": {
            "top_k": automl.get("hp_pool_top_k") or wf.get("hp_pool_top_k"),
            "pool_search_time_budget_sec": pool_search.get("search_time_budget_sec")
            or wf.get("pool_search_time_budget_sec"),
            "n_pool": pool_search.get("n_pool"),
            "selected_pool_rank": frozen.get("selected_pool_rank"),
            "selected_mean_fold_log_loss": frozen.get("selected_mean_fold_log_loss"),
            "selected_estimator": frozen.get("frozen_estimator") or meta.get("best_estimator"),
            "evaluations": [
                {
                    "pool_rank": e.get("pool_rank"),
                    "estimator": e.get("estimator"),
                    "weight_scheme": e.get("weight_scheme"),
                    "weight_spec_label": e.get("weight_spec_label"),
                    "softmax_temperature": e.get("softmax_temperature"),
                    "mean_fold_log_loss": e.get("mean_fold_log_loss"),
                    "oof_portfolio_ann_sharpe_excess": e.get("oof_portfolio_ann_sharpe_excess"),
                    "pooled_oof_roc_auc": e.get("pooled_oof_roc_auc"),
                    "portfolio_select_value": e.get("portfolio_select_value"),
                    "selected": (
                        e.get("pool_rank") == frozen.get("selected_pool_rank")
                        and e.get("weight_scheme") == frozen.get("selected_weight_scheme")
                        and e.get("softmax_temperature") == frozen.get("frozen_softmax_temperature")
                    ),
                }
                for e in sorted(
                    pool_evals,
                    key=lambda x: x.get("portfolio_select_value") or float("-inf"),
                    reverse=True,
                )
            ],
            "selected_weight_scheme": frozen.get("selected_weight_scheme") or frozen.get("weight_scheme"),
            "selected_softmax_temperature": frozen.get("frozen_softmax_temperature"),
            "weight_scheme": frozen.get("weight_scheme"),
            "weight_spec_label": frozen.get("weight_spec_label"),
            "weight_scheme_pool": frozen.get("weight_scheme_pool"),
            "portfolio_select_metric": frozen.get("portfolio_select_metric"),
        },
        "walk_forward": {
            "min_train_quarters": wf.get("min_train_quarters"),
            "feature_warmup_quarters": wf.get("feature_warmup_quarters")
            or frozen.get("feature_warmup_quarters"),
            "feature_warmup_excluded": frozen.get("feature_warmup_excluded"),
            "selection_train_quarters": frozen.get("selection_train_quarters"),
            "pool_search_time_budget_sec": (pool_search.get("search_time_budget_sec")
            or wf.get("pool_search_time_budget_sec")),
            "hp_pool_top_k": automl.get("hp_pool_top_k") or wf.get("hp_pool_top_k"),
            "n_cv_folds": wf.get("n_cv_folds"),
            "n_oos_folds": wf.get("n_oos_folds"),
            "folds": [
                {
                    "fold": f.get("fold"),
                    "phase": f.get("phase"),
                    "n_train_quarters": f.get("n_train_quarters"),
                    "train_start": (f.get("train_quarters") or [None])[0],
                    "train_end": (f.get("train_quarters") or [None])[-1],
                    "pred_quarter": f.get("pred_quarter"),
                    "pred_roc_auc": f.get("pred_roc_auc"),
                }
                for f in (wf.get("folds") or [])
            ],
            "predicted_quarters": wf.get("predicted_quarters"),
            "cv_oof_quarters": wf.get("cv_oof_quarters") or wf.get("early_quarters"),
            "final_oos_quarters": wf.get("final_oos_quarters") or wf.get("recent_quarters"),
            "early_quarters": wf.get("early_quarters"),
            "recent_quarters": wf.get("recent_quarters"),
            "wf_log_loss": wf.get("wf_log_loss") or cv.get("oof_log_loss"),
            "wf_roc_auc": wf.get("wf_roc_auc") or cv.get("oof_roc_auc"),
            "cv_oof_roc_auc": wf.get("cv_oof_roc_auc") or cv.get("oof_roc_auc"),
            "final_oos_roc_auc": wf.get("final_oos_roc_auc") or final_oos.get("roc_auc"),
            "recent_roc_auc": wf.get("final_oos_roc_auc") or final_oos.get("roc_auc"),
            "frozen_estimator": frozen.get("frozen_estimator") or final_oos.get("frozen_estimator"),
        },
        "cv": cv,
        "cv_eval": {
            "spearman_mean": cv_eval.get("spearman_mean"),
            "positive_rate": cv_eval.get("positive_rate"),
            "n_rows": cv_eval.get("n_rows"),
            "quarters": cv_eval.get("quarters"),
            "top30_precision": cv_cm.get("precision"),
            "top30_recall": cv_cm.get("recall"),
            "top30_f1": cv_cm.get("f1"),
            "confusion_matrix": cv_cm,
            "per_quarter": cv_eval.get("per_quarter") or [],
        },
        "final_oos": final_oos,
        "holdout": holdout,
        "oos_eval": {
            "spearman_mean": oos_eval.get("spearman_mean") or eval_r.get("spearman_mean"),
            "positive_rate": oos_eval.get("positive_rate") or eval_r.get("positive_rate"),
            "n_oos": oos_eval.get("n_rows") or eval_r.get("n_oos"),
            "final_oos_quarters": wf.get("final_oos_quarters") or eval_r.get("final_oos_quarters"),
            "recent_quarters": wf.get("final_oos_quarters") or eval_r.get("final_oos_quarters"),
            "top30_precision": top30_cm.get("precision"),
            "top30_recall": top30_cm.get("recall"),
            "top30_f1": top30_cm.get("f1"),
            "confusion_matrix": top30_cm,
            "per_quarter": oos_eval.get("per_quarter") or eval_r.get("per_quarter"),
        },
        "portfolio": {
            "benchmark": perf.get("benchmark", BENCHMARK_ID),
            "benchmark_symbol": BENCHMARK_SYMBOL,
            "top_n_holdings": perf.get("top_n_holdings", TOP_N_HOLDINGS),
            "label_top_n": perf.get("label_top_n", TOP_N_LABEL),
            "weighting": perf.get("weighting", "inverse_daily_vol"),
            "rebalance": "monthly",
            "signal": "quarterly (latest completed quarter at month-end)",
            "in_sample": perf.get("in_sample") or {},
            "out_of_sample": perf.get("out_of_sample") or {},
        },
        "backtest_eval_note": meta.get("backtest_eval_note"),
        "universe": get_universe_summary(),
    }


def clear_cache() -> None:
    get_backtest_curve.cache_clear()
    get_backtest_monthly.cache_clear()
    get_backtest_daily.cache_clear()
    get_drawdown_news.cache_clear()
    get_performance_report.cache_clear()
    get_holdings_by_quarter.cache_clear()
    get_holdings_by_month.cache_clear()
    get_holdings_by_signal_quarter.cache_clear()
    get_model_meta.cache_clear()
    get_model_variables_doc.cache_clear()
    get_feature_stats.cache_clear()
    build_model_variables.cache_clear()
    get_eval_report.cache_clear()
    get_factor_analysis.cache_clear()
    get_universe_summary.cache_clear()
    build_model_summary.cache_clear()
