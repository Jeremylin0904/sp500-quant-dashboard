#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quant.pipeline.build_raw_samples import ensure_sample_raw_data
from quant.pipeline.fetch_simfin import fetch_simfin_real_data
from quant.pipeline.build_curated import build_curated
from quant.pipeline.build_features import build_features
from quant.pipeline.build_labels import build_labels
from quant.pipeline.build_dataset import build_model_dataset
from quant.pipeline.train_model import train_model
from quant.pipeline.backtest import run_backtest
from quant.pipeline.validate import run_validation


def main(force: bool = False) -> None:
    # If user provided SIMFIN_API_KEY, fetch real data; otherwise fall back to reproducible sample data.
    try:
        fetch_simfin_real_data(force=force)
        print("OK: fetched real data from SimFin into quant/data_raw/")
    except Exception as e:
        print(f"INFO: SimFin fetch skipped ({e}). Using sample raw data instead.")
        ensure_sample_raw_data(force=force)
    print("=== Step 1: Curate raw → parquet ===")
    build_curated(force=force)

    print("=== Step 2: Features (PIT aligned) ===")
    build_features(force=force)

    print("=== Step 3: Labels (y_t, y_next) ===")
    build_labels(force=force)

    print("=== Step 4: Model dataset ===")
    build_model_dataset(force=force)

    print("=== Step 5: Train model ===")
    train_model(force=force)

    print("=== Step 6: Backtest ===")
    run_backtest(force=force)

    print("=== Step 7: Validate ===")
    report = run_validation(force=force)
    if not report["passed"]:
        raise SystemExit(f"Validation failed: {report['errors']}")

    print("OK: build_all finished.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    main(force=args.force)

