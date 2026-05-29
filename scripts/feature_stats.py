#!/usr/bin/env python3
"""Compute missing rate and percentiles for model feature columns."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "quant" / "model"


def main() -> None:
    df = pd.read_parquet(MODEL_DIR / "model_dataset.parquet")
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    feature_cols = meta["feature_cols"]
    n = len(df)

    rows = []
    for c in feature_cols:
        s = df[c]
        miss = float(s.isna().mean())
        arr_all = s.to_numpy(dtype="float64")
        inf_pct = float(np.isinf(arr_all).mean() * 100)
        finite = s.replace([np.inf, -np.inf], np.nan).dropna()
        if len(finite) == 0:
            rows.append(
                {
                    "col": c,
                    "n": n,
                    "miss_pct": miss * 100,
                    "inf_pct": inf_pct,
                    "p01": None,
                    "p25": None,
                    "p50": None,
                    "p75": None,
                    "p99": None,
                    "min": None,
                    "max": None,
                }
            )
        else:
            arr = finite.to_numpy(dtype="float64")
            rows.append(
                {
                    "col": c,
                    "n": n,
                    "miss_pct": miss * 100,
                    "inf_pct": inf_pct,
                    "p01": float(np.percentile(arr, 1)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p99": float(np.percentile(arr, 99)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            )

    out = pd.DataFrame(rows)
    out_path = MODEL_DIR / "feature_stats.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path} ({len(out)} features, n={n})")
    print("\n--- Highest missing rate ---")
    for _, r in out.sort_values("miss_pct", ascending=False).head(12).iterrows():
        print(f"  {r['col']}: miss={r['miss_pct']:.1f}%")
    inf_rows = out[out["inf_pct"] > 0].sort_values("inf_pct", ascending=False)
    if len(inf_rows):
        print("\n--- Columns with inf ---")
        for _, r in inf_rows.iterrows():
            print(f"  {r['col']}: inf={r['inf_pct']:.2f}%")


if __name__ == "__main__":
    main()
