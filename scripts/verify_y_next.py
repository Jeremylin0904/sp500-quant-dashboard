#!/usr/bin/env python3
"""Verify y_next aligns with next-quarter top-30 excess return."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant.config import LABELS_DIR, MODEL_DIR, TOP_N_LABEL


def main() -> None:
    ds = pd.read_parquet(MODEL_DIR / "model_dataset.parquet")
    lab = pd.read_parquet(LABELS_DIR / "labels_quarterly.parquet")

    ds["next_quarter"] = ds["quarter"].map(lambda q: str(pd.Period(q, freq="Q") + 1))
    lab_n = lab.rename(columns={"quarter": "next_quarter", "y_t": "y_t_next", "excess_rank": "rank_next"})
    chk = ds.merge(lab_n[["symbol", "next_quarter", "y_t_next", "rank_next"]], on=["symbol", "next_quarter"], how="inner")

    print(f"rows: {len(chk)}")
    print(f"y_next mismatch vs y_t_next: {(chk['y_next'] - chk['y_t_next']).abs().max()}")
    pos = chk[chk["y_next"] >= 1.0]
    ok = pos["rank_next"] <= TOP_N_LABEL
    print(f"y_next=1 count: {len(pos)}; rank<={TOP_N_LABEL}: {ok.sum()}/{len(pos)} ({ok.mean()*100:.1f}%)")


if __name__ == "__main__":
    main()
