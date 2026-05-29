"""Merge features with lagged labels into a modeling dataset."""

from __future__ import annotations

import json

import pandas as pd

from quant.config import FEATURES_DIR, LABELS_DIR, MODEL_DIR, TARGET_COL


def build_model_dataset(force: bool = False) -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODEL_DIR / "model_dataset.parquet"
    meta_path = MODEL_DIR / "model_dataset_meta.json"
    if out_path.exists() and meta_path.exists() and not force:
        return pd.read_parquet(out_path)

    feat = pd.read_parquet(FEATURES_DIR / "features_quarterly.parquet")
    lab = pd.read_parquet(LABELS_DIR / "labels_quarterly.parquet")

    merged = feat.merge(
        lab[
            [
                "symbol",
                "quarter",
                "y_t",
                "y_next",
                "return",
                "excess_return",
                "daily_vol",
                "quarter_end",
                "is_top30_next",
            ]
        ],
        on=["symbol", "quarter"],
        how="inner",
    )

    # Keep only rows where target exists
    merged = merged.dropna(subset=[TARGET_COL]).sort_values(["quarter", "symbol"]).reset_index(drop=True)

    merged.to_parquet(out_path, index=False)
    meta_path.write_text(json.dumps({"rows": int(len(merged)), "cols": list(merged.columns)}, indent=2), encoding="utf-8")
    return merged

