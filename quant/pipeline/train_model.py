"""Train FLAML classifier for y_next with weighted BCE (sample_weight + log_loss)."""

from __future__ import annotations

import copy
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from quant.config import (
    AUTOML_TIME_BUDGET_FINAL,
    AUTOML_TIME_BUDGET_POOL_SEARCH,
    BCE_POS_WEIGHT_MODE,
    CV_MIN_TRAIN_QUARTERS,
    FEATURE_WARMUP_QUARTERS,
    HP_POOL_TOP_K,
    MODEL_DIR,
    PORTFOLIO_HP_SELECT_METRIC,
    RANDOM_STATE,
    SOFTMAX_TEMPERATURE_POOL,
    TARGET_COL,
    WEIGHT_SCHEME,
    WEIGHT_SCHEME_POOL,
    WF_OOS_QUARTERS,
)


def _first_cv_fold_index() -> int:
    """First quarter index predicted in CV OOF (after TTM warmup + min train window)."""
    return FEATURE_WARMUP_QUARTERS + CV_MIN_TRAIN_QUARTERS
from quant.pipeline.build_dataset import build_model_dataset


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Feature cleaning for training/predict. NaNs kept for tree models."""
    X = df[feature_cols].astype("float64").replace([np.inf, -np.inf], np.nan)
    X = X.clip(lower=-1e6, upper=1e6)
    return X.to_numpy(dtype="float64", copy=False)


def weighted_bce_sample_weights(y: np.ndarray, mode: str | None = None) -> np.ndarray | None:
    """Per-sample weights for weighted BCE: w_neg=1, w_pos=n_neg/n_pos on the train fold."""
    mode = mode or BCE_POS_WEIGHT_MODE
    if mode == "none":
        return None
    y = np.asarray(y, dtype=np.int64).ravel()
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    w = np.ones(len(y), dtype=np.float64)
    w[y == 1] = n_neg / n_pos
    return w


def predict_score(model, X: np.ndarray) -> np.ndarray:
    """P(y=1) for ranking / Top-K; falls back to predict for legacy regressors."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float64)
        return np.asarray(proba, dtype=np.float64).ravel()
    return np.asarray(model.predict(X), dtype=np.float64).ravel()


def _feature_cols(ds: pd.DataFrame) -> list[str]:
    drop_cols = {
        "symbol",
        "company_name",
        "sector",
        "industry",
        "quarter",
        "as_of_date",
        "report_date",
        "publish_date",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
        "y_t",
        "y_next",
        "return",
        "excess_return",
        "is_top30_next",
        "quarter_end",
    }
    return [c for c in ds.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(ds[c])]


def _automl_settings(time_budget: int) -> dict:
    return {
        "time_budget": time_budget,
        "task": "classification",
        "metric": "log_loss",
        "seed": RANDOM_STATE,
        "verbose": 0,
        "skip_transform": True,
        "estimator_list": ["xgb_limitdepth", "xgboost", "lgbm"],
    }


def _fit_automl(X_train: np.ndarray, y_train: np.ndarray, time_budget: int):
    from flaml import AutoML

    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    sw = weighted_bce_sample_weights(y_train)
    automl = AutoML()
    fit_kw: dict = {"X_train": X_train, "y_train": y_train, **_automl_settings(time_budget)}
    if sw is not None:
        fit_kw["sample_weight"] = sw
    automl.fit(**fit_kw)
    return automl


def _fit_frozen(estimator: str, config: dict, X_train: np.ndarray, y_train: np.ndarray):
    """Retrain with frozen hyperparameters (no search)."""
    from flaml import AutoML

    y_train = np.asarray(y_train, dtype=np.int64).ravel()
    sw = weighted_bce_sample_weights(y_train)
    automl = AutoML()
    fit_kw: dict = {
        "X_train": X_train,
        "y_train": y_train,
        **_automl_settings(1),
        "estimator_list": [estimator],
        "starting_points": {estimator: copy.deepcopy(config)},
    }
    if sw is not None:
        fit_kw["sample_weight"] = sw
    automl.fit(**fit_kw)
    return automl


def _config_key(estimator: str, config: dict) -> str:
    return f"{estimator}:{json.dumps(config, sort_keys=True, default=str)}"


def _build_hp_pool(
    X_sel: np.ndarray,
    y_sel: np.ndarray,
    top_k: int,
    time_budget: int,
) -> tuple[list[dict], dict]:
    """
    Run AutoML on post-warmup pre-OOS quarters, extract top-K unique (estimator, config) trials.
    """
    automl = _fit_automl(X_sel, y_sel, time_budget)
    candidates: list[dict] = []
    seen: set[str] = set()

    def _add(estimator: str, config: dict, source: str, internal_loss: float | None) -> None:
        key = _config_key(estimator, config)
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "estimator": estimator,
                "config": copy.deepcopy(config),
                "pool_source": source,
                "internal_val_loss": float(internal_loss) if internal_loss is not None else None,
            }
        )

    history = getattr(automl, "config_history", None) or {}
    for item in history.values():
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        est, cfg = item[0], item[1]
        internal = float(item[2]) if len(item) > 2 and item[2] is not None else None
        _add(str(est), cfg, "config_history", internal)

    per_est = getattr(automl, "best_config_per_estimator", None) or {}
    for est, cfg in per_est.items():
        _add(str(est), cfg, "best_config_per_estimator", None)

    best_est = getattr(automl, "best_estimator", None)
    best_cfg = getattr(automl, "best_config", None)
    if best_est and best_cfg:
        _add(str(best_est), best_cfg, "best_config", getattr(automl, "best_loss", None))

    candidates.sort(key=lambda c: c.get("internal_val_loss") if c.get("internal_val_loss") is not None else float("inf"))
    pool = candidates[:top_k] if len(candidates) > top_k else candidates

    pool_meta = {
        "search_time_budget_sec": time_budget,
        "n_trials_total": len(candidates),
        "n_pool": len(pool),
        "top_k": top_k,
        "best_estimator_from_search": best_est,
        "pool": [
            {
                "rank": i + 1,
                "estimator": p["estimator"],
                "config": p["config"],
                "pool_source": p["pool_source"],
                "internal_val_loss": p["internal_val_loss"],
            }
            for i, p in enumerate(pool)
        ],
    }
    return pool, pool_meta


def _classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.int64).ravel()
    p = np.clip(np.asarray(proba, dtype=np.float64).ravel(), 1e-15, 1 - 1e-15)
    out: dict[str, float | None] = {
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    sw = weighted_bce_sample_weights(y)
    if sw is not None:
        out["log_loss_weighted"] = float(log_loss(y, p, sample_weight=sw, labels=[0, 1]))
    else:
        out["log_loss_weighted"] = out["log_loss"]
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
    else:
        out["roc_auc"] = None
    return out


def _fold_record(
    *,
    fold: int,
    phase: str,
    train_quarters: list[str],
    pred_quarter: str,
    n_train_rows: int,
    n_pred_rows: int,
    y_tr: np.ndarray,
    y_va: np.ndarray,
    pred_va: np.ndarray,
    frozen_estimator: str | None = None,
    pool_rank: int | None = None,
) -> dict:
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    rec = {
        "fold": fold,
        "phase": phase,
        "train_quarters": train_quarters,
        "n_train_quarters": len(train_quarters),
        "pred_quarter": pred_quarter,
        "n_train_rows": n_train_rows,
        "n_pred_rows": n_pred_rows,
        "train_pos_weight": float(n_neg / n_pos) if n_pos else None,
        **{f"pred_{k}": v for k, v in _classification_metrics(y_va, pred_va).items()},
    }
    if frozen_estimator:
        rec["frozen_estimator"] = frozen_estimator
    if pool_rank is not None:
        rec["pool_rank"] = pool_rank
    return rec


def _run_expanding_oof(
    ds: pd.DataFrame,
    feature_cols: list[str],
    quarters: list[str],
    oos_start: int,
    estimator: str,
    config: dict,
    *,
    phase: str,
    fold_offset: int = 0,
    pool_rank: int | None = None,
) -> tuple[pd.DataFrame, list[dict], list[float]]:
    """Expanding walk-forward with a fixed (estimator, config). Returns OOF scores + per-fold log_loss."""
    if phase == "final_oos":
        fold_range = range(oos_start, len(quarters))
    else:
        fold_range = range(_first_cv_fold_index(), oos_start)

    wf_parts: list[pd.DataFrame] = []
    fold_metrics: list[dict] = []
    fold_losses: list[float] = []

    for i in fold_range:
        tr_q = set(quarters[FEATURE_WARMUP_QUARTERS:i])
        va_q = quarters[i]
        tr = ds[ds["quarter"].isin(tr_q)]
        va = ds[ds["quarter"] == va_q]
        if tr.empty or va.empty:
            continue

        X_tr = prepare_feature_matrix(tr, feature_cols)
        y_tr = tr[TARGET_COL].to_numpy(dtype=np.int64)
        X_va = prepare_feature_matrix(va, feature_cols)
        y_va = va[TARGET_COL].to_numpy(dtype=np.int64)

        model = _fit_frozen(estimator, config, X_tr, y_tr)
        pred_va = predict_score(model, X_va)
        cls = _classification_metrics(y_va, pred_va)
        fold_losses.append(float(cls["log_loss"]))

        part = va[["symbol", "quarter", TARGET_COL]].copy()
        part["score_wf"] = pred_va
        part["score_oof"] = pred_va
        part["wf_phase"] = phase
        wf_parts.append(part)

        fold_metrics.append(
            _fold_record(
                fold=fold_offset + len(fold_metrics) + 1,
                phase=phase,
                train_quarters=sorted(tr_q),
                pred_quarter=va_q,
                n_train_rows=int(len(tr)),
                n_pred_rows=int(len(va)),
                y_tr=y_tr,
                y_va=y_va,
                pred_va=pred_va,
                frozen_estimator=estimator,
                pool_rank=pool_rank,
            )
        )

    wf = pd.concat(wf_parts, ignore_index=True) if wf_parts else pd.DataFrame()
    return wf, fold_metrics, fold_losses


def _weight_pool_entries() -> list[dict]:
    """Normalize WEIGHT_SCHEME_POOL entries for grid search."""
    out: list[dict] = []
    for entry in WEIGHT_SCHEME_POOL:
        scheme = str(entry["weight_scheme"])
        temp = entry.get("softmax_temperature")
        out.append(
            {
                "weight_scheme": scheme,
                "softmax_temperature": float(temp) if temp is not None else None,
            }
        )
    return out


def _weight_spec_label(scheme: str, temp: float | None) -> str:
    if scheme in ("softmax", "softmax_sharpe") and temp is not None:
        suffix = "score/(T·σ)" if scheme == "softmax_sharpe" else "score/T"
        return f"{scheme}({suffix}, T={temp})"
    return scheme


def _select_config_and_weighting(
    ds: pd.DataFrame,
    feature_cols: list[str],
    quarters: list[str],
    oos_start: int,
    pool: list[dict],
    weight_pool: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    """
    For each model config in pool × each weight-scheme candidate:
    - expanding OOF classifier scores
    - CV OOF monthly portfolio backtest
    Pick best (config, weight_scheme, T?) by CV OOF portfolio ann_sharpe_excess.
    """
    from quant.pipeline.monthly_portfolio import eval_portfolio_on_wf_scores

    weight_pool = weight_pool or _weight_pool_entries()
    cv_quarters = set(quarters[_first_cv_fold_index():oos_start])
    evals: list[dict] = []
    best: dict | None = None

    for rank, cand in enumerate(pool, start=1):
        wf, fold_metrics, fold_losses = _run_expanding_oof(
            ds,
            feature_cols,
            quarters,
            oos_start,
            cand["estimator"],
            cand["config"],
            phase="cv_oof_pool_eval",
            pool_rank=rank,
        )
        if not fold_losses:
            continue

        y_oof = wf[TARGET_COL].to_numpy(dtype=np.int64) if len(wf) else np.array([], dtype=np.int64)
        pred_oof = wf["score_wf"].to_numpy() if len(wf) else np.array([])
        pooled = _classification_metrics(y_oof, pred_oof) if len(wf) else {}

        for wspec in weight_pool:
            scheme = wspec["weight_scheme"]
            temp = wspec.get("softmax_temperature")
            temp_arg = float(temp) if temp is not None else 0.1

            port = eval_portfolio_on_wf_scores(
                wf,
                cv_quarters,
                feature_cols=feature_cols,
                weight_scheme=scheme,
                softmax_temperature=temp_arg,
            )
            port_metric = port.get(PORTFOLIO_HP_SELECT_METRIC)
            if port_metric is None or (isinstance(port_metric, float) and np.isnan(port_metric)):
                port_metric = float("-inf")

            rec = {
                "pool_rank": rank,
                "estimator": cand["estimator"],
                "config": cand["config"],
                "pool_source": cand.get("pool_source"),
                "internal_val_loss": cand.get("internal_val_loss"),
                "weight_scheme": scheme,
                "weight_spec_label": _weight_spec_label(scheme, temp),
                "softmax_temperature": temp,
                "mean_fold_log_loss": float(np.mean(fold_losses)),
                "std_fold_log_loss": float(np.std(fold_losses, ddof=0)),
                "pooled_oof_log_loss": pooled.get("log_loss"),
                "pooled_oof_roc_auc": pooled.get("roc_auc"),
                "n_cv_folds": len(fold_losses),
                "fold_log_losses": fold_losses,
                "oof_portfolio_n_months": port.get("n_months"),
                "oof_portfolio_ann_sharpe_excess": port.get("ann_sharpe_excess"),
                "oof_portfolio_ann_sharpe_strategy": port.get("ann_sharpe_strategy"),
                "oof_portfolio_total_excess_geometric": port.get("total_excess_geometric"),
                "portfolio_select_metric": PORTFOLIO_HP_SELECT_METRIC,
                "portfolio_select_value": float(port_metric),
            }
            evals.append(rec)

            if best is None or rec["portfolio_select_value"] > best["portfolio_select_value"]:
                best = {**rec, "wf": wf, "fold_metrics": fold_metrics}

    if best is None:
        raise RuntimeError("Hyperparameter pool OOF evaluation produced no valid (model, weight) pairs.")

    evals.sort(key=lambda x: x.get("portfolio_select_value") or float("-inf"), reverse=True)
    return best, evals


def _expanding_walk_forward(
    ds: pd.DataFrame,
    feature_cols: list[str],
    quarters: list[str],
) -> tuple[pd.DataFrame, list[dict], dict]:
    """
    1) AutoML on Q1..Q(N-OOS) → top-K config pool
    2) Each config × weight-scheme pool: OOF + CV OOF portfolio Sharpe → pick best (config, weight)
    3) Best pair: CV OOF scores + Final OOS (frozen config + weight scheme)
    """
    quarters = sorted(quarters)
    n_q = len(quarters)
    oos_start = n_q - WF_OOS_QUARTERS

    min_q = _first_cv_fold_index() + 1
    if n_q < min_q:
        raise ValueError(
            f"Need at least {min_q} quarters (warmup={FEATURE_WARMUP_QUARTERS} + "
            f"min_train={CV_MIN_TRAIN_QUARTERS} + 1 pred); got {n_q}"
        )
    if oos_start <= _first_cv_fold_index():
        raise ValueError(
            f"Not enough quarters for CV OOF before {WF_OOS_QUARTERS}-quarter final OOS; got {n_q}"
        )

    sel_quarters = quarters[FEATURE_WARMUP_QUARTERS:oos_start]
    tr_sel = ds[ds["quarter"].isin(sel_quarters)]
    X_sel = prepare_feature_matrix(tr_sel, feature_cols)
    y_sel = tr_sel[TARGET_COL].to_numpy(dtype=np.int64)

    pool, pool_meta = _build_hp_pool(X_sel, y_sel, HP_POOL_TOP_K, AUTOML_TIME_BUDGET_POOL_SEARCH)
    best, pool_evals = _select_config_and_weighting(ds, feature_cols, quarters, oos_start, pool)

    frozen_estimator = best["estimator"]
    frozen_config = copy.deepcopy(best["config"])
    frozen_weight_scheme = str(best["weight_scheme"])
    frozen_temperature = best.get("softmax_temperature")
    if frozen_temperature is not None:
        frozen_temperature = float(frozen_temperature)
    wf_cv = best["wf"].copy()
    wf_cv["wf_phase"] = "cv_oof"
    fold_metrics = best["fold_metrics"]
    for fm in fold_metrics:
        fm["phase"] = "cv_oof"

    wf_oos, oos_fold_metrics, _ = _run_expanding_oof(
        ds,
        feature_cols,
        quarters,
        oos_start,
        frozen_estimator,
        frozen_config,
        phase="final_oos",
        fold_offset=len(fold_metrics),
        pool_rank=best["pool_rank"],
    )
    fold_metrics.extend(oos_fold_metrics)

    wf = pd.concat([wf_cv, wf_oos], ignore_index=True) if len(wf_oos) else wf_cv

    frozen_meta = {
        "selection_method": "automl_pool_x_weight_scheme_oof_portfolio_sharpe",
        "feature_warmup_quarters": FEATURE_WARMUP_QUARTERS,
        "feature_warmup_excluded": quarters[:FEATURE_WARMUP_QUARTERS],
        "selection_train_quarters": sel_quarters,
        "frozen_estimator": frozen_estimator,
        "frozen_config": frozen_config,
        "weight_scheme": frozen_weight_scheme,
        "weight_spec_label": best.get("weight_spec_label"),
        "frozen_softmax_temperature": frozen_temperature,
        "weight_scheme_pool": _weight_pool_entries(),
        "portfolio_select_metric": PORTFOLIO_HP_SELECT_METRIC,
        "selected_pool_rank": best["pool_rank"],
        "selected_weight_scheme": frozen_weight_scheme,
        "selected_softmax_temperature": frozen_temperature,
        "selected_mean_fold_log_loss": best["mean_fold_log_loss"],
        "selected_oof_portfolio_ann_sharpe_excess": best.get("oof_portfolio_ann_sharpe_excess"),
        "selected_pooled_oof_roc_auc": best.get("pooled_oof_roc_auc"),
        "pool_search": pool_meta,
        "pool_oof_evaluations": pool_evals,
    }
    return wf, fold_metrics, frozen_meta


def train_model(force: bool = False) -> dict:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "model.pkl"
    meta_path = MODEL_DIR / "model_meta.json"
    wf_path = MODEL_DIR / "walk_forward_scores.parquet"
    oof_path = MODEL_DIR / "cv_oof_scores.parquet"

    if model_path.exists() and meta_path.exists() and wf_path.exists() and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    ds = build_model_dataset(force=force)
    quarters = sorted(ds["quarter"].unique())
    feature_cols = _feature_cols(ds)

    wf, fold_metrics, frozen_meta = _expanding_walk_forward(ds, feature_cols, quarters)
    wf.to_parquet(wf_path, index=False)
    cv_only = wf[wf["wf_phase"] == "cv_oof"] if "wf_phase" in wf.columns else wf
    cv_only.to_parquet(oof_path, index=False)

    wf_quarters = sorted(wf["quarter"].unique()) if len(wf) else []
    cv_q = sorted(wf.loc[wf["wf_phase"] == "cv_oof", "quarter"].unique()) if len(wf) else []
    oos_q = sorted(wf.loc[wf["wf_phase"] == "final_oos", "quarter"].unique()) if len(wf) else []

    wf_cv = wf[wf["wf_phase"] == "cv_oof"] if len(wf) else wf.iloc[0:0]
    wf_oos = wf[wf["wf_phase"] == "final_oos"] if len(wf) else wf.iloc[0:0]

    y_wf = wf[TARGET_COL].to_numpy(dtype=np.int64) if len(wf) else np.array([], dtype=np.int64)
    pred_wf = wf["score_wf"].to_numpy() if len(wf) else np.array([])
    wf_cls = _classification_metrics(y_wf, pred_wf) if len(wf) else {}

    cv_cls = (
        _classification_metrics(
            wf_cv[TARGET_COL].to_numpy(dtype=np.int64),
            wf_cv["score_wf"].to_numpy(),
        )
        if len(wf_cv)
        else {}
    )
    oos_cls = (
        _classification_metrics(
            wf_oos[TARGET_COL].to_numpy(dtype=np.int64),
            wf_oos["score_wf"].to_numpy(),
        )
        if len(wf_oos)
        else {}
    )

    X_all = prepare_feature_matrix(ds, feature_cols)
    y_all = ds[TARGET_COL].to_numpy(dtype=np.int64)
    automl = _fit_frozen(frozen_meta["frozen_estimator"], frozen_meta["frozen_config"], X_all, y_all)

    n_pos = int(y_all.sum())
    n_neg = len(y_all) - n_pos

    meta = {
        "model": "FLAML_AutoML",
        "task": "classification",
        "eval_method": "expanding_walk_forward",
        "loss": {
            "name": "weighted_bce",
            "description": "Binary cross-entropy with sample_weight: w_neg=1, w_pos=n_neg/n_pos per train fold",
            "pos_weight_mode": BCE_POS_WEIGHT_MODE,
            "final_fit_pos_weight": float(n_neg / n_pos) if n_pos else None,
            "flaml_metric": "log_loss",
        },
        "best_estimator": frozen_meta["frozen_estimator"],
        "best_config": frozen_meta["frozen_config"],
        "automl_settings": {
            "pool_search_time_budget": AUTOML_TIME_BUDGET_POOL_SEARCH,
            "hp_pool_top_k": HP_POOL_TOP_K,
            "task": "classification",
            "metric": "log_loss",
            "skip_transform": True,
            "estimator_list": ["xgb_limitdepth", "xgboost", "lgbm"],
        },
        "walk_forward": {
            "method": "pool_search_x_weight_scheme_oof_portfolio_select_then_frozen_oos",
            "feature_warmup_quarters": FEATURE_WARMUP_QUARTERS,
            "min_train_quarters": CV_MIN_TRAIN_QUARTERS,
            "first_cv_fold_index": _first_cv_fold_index(),
            "hp_pool_top_k": HP_POOL_TOP_K,
            "pool_search_time_budget_sec": AUTOML_TIME_BUDGET_POOL_SEARCH,
            "n_folds": len(fold_metrics),
            "n_cv_folds": len([f for f in fold_metrics if f["phase"] == "cv_oof"]),
            "n_oos_folds": len([f for f in fold_metrics if f["phase"] == "final_oos"]),
            "predicted_quarters": wf_quarters,
            "cv_oof_quarters": cv_q,
            "final_oos_quarters": oos_q,
            "early_quarters": cv_q,
            "recent_quarters": oos_q,
            "frozen": frozen_meta,
            "n_wf_rows": int(len(wf)),
            "wf_log_loss": wf_cls.get("log_loss"),
            "wf_log_loss_weighted": wf_cls.get("log_loss_weighted"),
            "wf_roc_auc": wf_cls.get("roc_auc"),
            "cv_oof_log_loss": cv_cls.get("log_loss"),
            "cv_oof_roc_auc": cv_cls.get("roc_auc"),
            "final_oos_log_loss": oos_cls.get("log_loss"),
            "final_oos_roc_auc": oos_cls.get("roc_auc"),
            "folds": fold_metrics,
        },
        "cv": {
            "method": "pool_oof_8fold_expanding",
            "feature_warmup_quarters": FEATURE_WARMUP_QUARTERS,
            "min_train_quarters": CV_MIN_TRAIN_QUARTERS,
            "first_cv_fold_index": _first_cv_fold_index(),
            "oof_quarters": cv_q,
            "n_oof_rows": int(len(wf_cv)),
            "oof_log_loss": cv_cls.get("log_loss"),
            "oof_log_loss_weighted": cv_cls.get("log_loss_weighted"),
            "oof_roc_auc": cv_cls.get("roc_auc"),
        },
        "final_oos": {
            "method": "expanding_walk_forward_frozen",
            "n_oos_quarters": WF_OOS_QUARTERS,
            "oos_quarters": oos_q,
            "n_oos_rows": int(len(wf_oos)),
            "log_loss": oos_cls.get("log_loss"),
            "roc_auc": oos_cls.get("roc_auc"),
            "frozen_estimator": frozen_meta.get("frozen_estimator"),
            "frozen_config": frozen_meta.get("frozen_config"),
        },
        "holdout": {
            "val_quarters": oos_q,
            "val_log_loss": oos_cls.get("log_loss"),
            "val_roc_auc": oos_cls.get("roc_auc"),
        },
        "feature_cols": feature_cols,
        "target_col": TARGET_COL,
        "all_quarters": quarters,
        "n_rows": int(len(ds)),
        "final_model_note": (
            "model.pkl is fit on all quarters for deployment only; "
            "backtest/eval must use walk_forward_scores.parquet (never final model scores for historical quarters)."
        ),
        "backtest_eval_note": (
            f"Exclude first {FEATURE_WARMUP_QUARTERS} quarters (TTM warmup). "
            f"Pool: AutoML {AUTOML_TIME_BUDGET_POOL_SEARCH}s → top-{HP_POOL_TOP_K} model configs × "
            f"{len(WEIGHT_SCHEME_POOL)} weight schemes; CV OOF from fold {_first_cv_fold_index()} "
            f"(metric={PORTFOLIO_HP_SELECT_METRIC}); frozen for Final OOS."
        ),
    }

    joblib.dump({"model": automl, "feature_cols": feature_cols}, model_path)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
