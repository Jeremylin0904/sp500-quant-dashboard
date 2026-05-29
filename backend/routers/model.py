from __future__ import annotations

from fastapi import APIRouter

from backend.services.data_service import (
    build_model_summary,
    build_model_variables,
    get_eval_report,
    get_model_meta,
)

router = APIRouter(prefix="/api/model", tags=["model"])


@router.get("/summary")
def model_summary():
    return build_model_summary()


@router.get("/variables")
def model_variables():
    return build_model_variables()


@router.get("/features")
def model_features():
    meta = get_model_meta()
    wf = meta.get("walk_forward") or {}
    frozen = wf.get("frozen") or {}
    return {
        "model": meta.get("model"),
        "task": meta.get("task"),
        "target_col": meta.get("target_col"),
        "best_estimator": meta.get("best_estimator"),
        "feature_cols": meta.get("feature_cols", []),
        "loss": meta.get("loss"),
        "cv": meta.get("cv"),
        "final_oos": meta.get("final_oos"),
        "holdout": meta.get("holdout"),
        "hp_pool": {
            "selected_pool_rank": frozen.get("selected_pool_rank"),
            "selected_mean_fold_log_loss": frozen.get("selected_mean_fold_log_loss"),
            "evaluations": frozen.get("pool_oof_evaluations"),
        },
        "walk_forward_method": wf.get("method"),
    }


@router.get("/eval")
def model_eval():
    return get_eval_report()
