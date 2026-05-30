"""Ablation: does dropping raw price levels (close / adjusted_close) help CV OOF?

Read-only experiment. Does NOT touch any production artifact (model.pkl,
model_meta.json, walk_forward_scores, backtest). It freezes the *selected*
(estimator, config, weight_scheme, T) from model_meta.json and only varies the
feature set fed to the model, so the comparison isolates the feature effect.

Selection rule stays the same as production: CV OOF portfolio annualized excess
Sharpe (ann_sharpe_excess); we also report pooled OOF ROC-AUC and mean fold
log-loss.

Run:  $env:PYTHONPATH="d:\\interview2"; python scripts/exp_price_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import MODEL_DIR, DATA_DIR  # type: ignore
from quant.pipeline.train_model import _run_expanding_oof, _classification_metrics
from quant.pipeline.monthly_portfolio import eval_portfolio_on_wf_scores

TARGET_COL = "y_next"

# Pure raw price levels.
PRICE_LEVELS = ["close", "adjusted_close"]

# All pure scale/level raw inputs (kind="raw" on the dashboard): absolute dollar
# amounts / share counts / price levels. Their cross-sectional signal is weak;
# the value is in the derived ratios/growth/momentum/ranks built from them.
RAW_LEVELS = [
    # group A: 原始基本面 (absolute $ amounts)
    "revenue", "gross_profit", "operating_income", "net_income", "eps_diluted",
    "total_assets", "total_liabilities", "total_equity", "cash_and_equivalents", "total_debt",
    # group B: 市場 / 價格 (levels / scale)
    "shares_outstanding", "close", "adjusted_close", "volume", "market_cap", "enterprise_value",
]

SCENARIOS = [
    ("baseline (all)", []),
    ("drop price levels (2)", PRICE_LEVELS),
    ("drop ALL raw levels (16)", RAW_LEVELS),
]


def _load_dataset() -> pd.DataFrame:
    for p in [
        MODEL_DIR / "model_dataset.parquet",
        DATA_DIR / "model_dataset.parquet",
    ]:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError("model_dataset.parquet not found")


def _run(ds: pd.DataFrame, feature_cols: list[str], quarters: list[str], oos_start: int,
         estimator: str, config: dict, weight_scheme: str, temp: float,
         portfolio_feature_cols: list[str]) -> dict:
    wf, _folds, fold_losses = _run_expanding_oof(
        ds, feature_cols, quarters, oos_start, estimator, config, phase="cv_oof",
    )
    cv_quarters = set(wf["quarter"].unique().tolist()) if len(wf) else set()
    y = wf[TARGET_COL].to_numpy(dtype=np.int64) if len(wf) else np.array([], dtype=np.int64)
    p = wf["score_wf"].to_numpy() if len(wf) else np.array([])
    cls = _classification_metrics(y, p) if len(wf) else {}
    port = eval_portfolio_on_wf_scores(
        wf, cv_quarters, feature_cols=portfolio_feature_cols,
        weight_scheme=weight_scheme, softmax_temperature=temp,
    )
    return {
        "n_features": len(feature_cols),
        "cv_quarters": sorted(cv_quarters),
        "mean_fold_log_loss": float(np.mean(fold_losses)) if fold_losses else None,
        "pooled_oof_roc_auc": cls.get("roc_auc"),
        "pooled_oof_log_loss": cls.get("log_loss"),
        "ann_sharpe_excess": port.get("ann_sharpe_excess"),
        "ann_sharpe_strategy": port.get("ann_sharpe_strategy"),
        "total_excess_geometric": port.get("total_excess_geometric"),
        "n_months": port.get("n_months"),
    }


def main() -> None:
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    frozen = meta["walk_forward"]["frozen"]
    estimator = frozen["frozen_estimator"]
    config = frozen["frozen_config"]
    weight_scheme = frozen["selected_weight_scheme"]
    temp = float(frozen.get("selected_softmax_temperature") or 0.1)

    quarters = list(meta["all_quarters"])
    cv_oof = list(meta["walk_forward"]["cv_oof_quarters"])
    oos_start = quarters.index(meta["walk_forward"]["final_oos_quarters"][0])

    base_feats = list(meta["feature_cols"])
    ds = _load_dataset()

    print("=" * 92)
    print("Ablation: drop pure raw scale/level features (read-only; production model untouched)")
    print(f"frozen: estimator={estimator} | weight={weight_scheme} T={temp}")
    print(f"CV OOF quarters={cv_oof} | oos_start_idx={oos_start}")
    print("(selection metric = CV OOF portfolio ann_sharpe_excess; higher is better)")
    print("=" * 92)

    results: list[tuple[str, dict]] = []
    base_metrics: dict | None = None
    for name, drop in SCENARIOS:
        feats = [c for c in base_feats if c not in set(drop)]
        r = _run(ds, feats, quarters, oos_start, estimator, config, weight_scheme, temp, base_feats)
        results.append((name, r))
        if base_metrics is None:
            base_metrics = r

    hdr = f"{'scenario':<26} {'n':>3} | {'ann_sharpe_exc':>14} {'Δsharpe':>9} | {'OOF_AUC':>8} | {'logloss':>8} | {'tot_exc_geo':>11}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in results:
        d_sharpe = (r["ann_sharpe_excess"] or 0) - (base_metrics["ann_sharpe_excess"] or 0)
        print(
            f"{name:<26} {r['n_features']:>3} | "
            f"{r['ann_sharpe_excess']:>14.4f} {d_sharpe:>+9.4f} | "
            f"{r['pooled_oof_roc_auc']:>8.4f} | {r['mean_fold_log_loss']:>8.4f} | "
            f"{r['total_excess_geometric']:>11.4f}"
        )
    print("=" * 92)


if __name__ == "__main__":
    main()
