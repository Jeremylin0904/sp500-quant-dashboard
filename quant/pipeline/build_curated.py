"""Build curated parquet tables from raw CSVs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant.config import (
    BALANCE_Q_CSV,
    CASHFLOW_Q_CSV,
    CONSTITUENTS_CSV,
    DATA_CURATED_DIR,
    INCOME_Q_CSV,
    PRICES_DAILY_CSV,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_curated(force: bool = False) -> dict:
    DATA_CURATED_DIR.mkdir(parents=True, exist_ok=True)

    out_prices = DATA_CURATED_DIR / "prices_daily.parquet"
    out_const = DATA_CURATED_DIR / "constituents.parquet"
    out_inc = DATA_CURATED_DIR / "income_quarterly.parquet"
    out_bal = DATA_CURATED_DIR / "balance_quarterly.parquet"
    out_cfs = DATA_CURATED_DIR / "cashflow_quarterly.parquet"
    meta_path = DATA_CURATED_DIR / "curated_meta.json"

    # cashflow is optional; only require it if raw exists.
    must = [out_prices, out_const, out_inc, out_bal, meta_path]
    if CASHFLOW_Q_CSV.exists():
        must.append(out_cfs)
    if all(p.exists() for p in must) and not force:
        return json.loads(meta_path.read_text(encoding="utf-8"))

    prices = pd.read_csv(PRICES_DAILY_CSV)
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    if "adjusted_close" in prices.columns:
        prices["adjusted_close"] = pd.to_numeric(prices["adjusted_close"], errors="coerce")
    prices["volume"] = pd.to_numeric(prices["volume"], errors="coerce")
    if "shares_outstanding" in prices.columns:
        prices["shares_outstanding"] = pd.to_numeric(prices["shares_outstanding"], errors="coerce")
    prices = prices.dropna(subset=["Date", "symbol", "close"]).sort_values(["symbol", "Date"])

    const = pd.read_csv(CONSTITUENTS_CSV)
    const = const.drop_duplicates("symbol")

    inc = pd.read_csv(INCOME_Q_CSV)
    bal = pd.read_csv(BALANCE_Q_CSV)
    for df in (inc, bal):
        df["report_date"] = pd.to_datetime(df["report_date"])
        df["publish_date"] = pd.to_datetime(df["publish_date"])

    cfs = None
    if CASHFLOW_Q_CSV.exists():
        cfs = pd.read_csv(CASHFLOW_Q_CSV)
        cfs["report_date"] = pd.to_datetime(cfs["report_date"])
        cfs["publish_date"] = pd.to_datetime(cfs["publish_date"])

    prices.to_parquet(out_prices, index=False)
    const.to_parquet(out_const, index=False)
    inc.to_parquet(out_inc, index=False)
    bal.to_parquet(out_bal, index=False)
    if cfs is not None:
        cfs.to_parquet(out_cfs, index=False)

    meta = {
        "sources": {
            "prices_daily.csv": {"path": str(PRICES_DAILY_CSV), "sha256": _sha256(PRICES_DAILY_CSV)},
            "constituents.csv": {"path": str(CONSTITUENTS_CSV), "sha256": _sha256(CONSTITUENTS_CSV)},
            "income_quarterly.csv": {"path": str(INCOME_Q_CSV), "sha256": _sha256(INCOME_Q_CSV)},
            "balance_quarterly.csv": {"path": str(BALANCE_Q_CSV), "sha256": _sha256(BALANCE_Q_CSV)},
            **(
                {"cashflow_quarterly.csv": {"path": str(CASHFLOW_Q_CSV), "sha256": _sha256(CASHFLOW_Q_CSV)}}
                if CASHFLOW_Q_CSV.exists()
                else {}
            ),
        },
        "rows": {
            "prices_daily": int(len(prices)),
            "constituents": int(len(const)),
            "income_quarterly": int(len(inc)),
            "balance_quarterly": int(len(bal)),
            **({"cashflow_quarterly": int(len(cfs))} if cfs is not None else {}),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta

