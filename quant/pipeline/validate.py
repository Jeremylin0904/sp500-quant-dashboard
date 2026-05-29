"""Validation checks for interview MVP pipeline outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from quant.config import BACKTEST_DIR, FEATURES_DIR, LABELS_DIR, MODEL_DIR


def run_validation(force: bool = False) -> dict:
    errors: list[str] = []
    warnings: list[str] = []

    feat_path = FEATURES_DIR / "features_quarterly.parquet"
    lab_path = LABELS_DIR / "labels_quarterly.parquet"
    ds_path = MODEL_DIR / "model_dataset.parquet"
    curve_path = BACKTEST_DIR / "backtest_curve.json"

    if not feat_path.exists():
        errors.append("Missing features_quarterly.parquet")
    if not lab_path.exists():
        errors.append("Missing labels_quarterly.parquet")
    if not ds_path.exists():
        errors.append("Missing model_dataset.parquet")
    if not curve_path.exists():
        warnings.append("Missing backtest_curve.json (backtest not run?)")

    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        # PIT check
        if "publish_date" in feat.columns and "as_of_date" in feat.columns:
            bad = (pd.to_datetime(feat["publish_date"]) > pd.to_datetime(feat["as_of_date"])).sum()
            if bad:
                errors.append(f"PIT violation: publish_date > as_of_date rows={int(bad)}")
        # PK uniqueness
        dup = feat.duplicated(["symbol", "quarter"]).sum()
        if dup:
            errors.append(f"Duplicate (symbol, quarter) in features: {int(dup)}")

    if lab_path.exists():
        lab = pd.read_parquet(lab_path)
        dup = lab.duplicated(["symbol", "quarter"]).sum()
        if dup:
            errors.append(f"Duplicate (symbol, quarter) in labels: {int(dup)}")

    passed = len(errors) == 0
    report = {
        "passed": passed,
        "errors": errors,
        "warnings": warnings,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (MODEL_DIR / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
