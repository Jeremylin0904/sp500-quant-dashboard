"""Quarterly + monthly labels: top-N excess return surge (binary)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant.config import DATA_CURATED_DIR, LABELS_DIR, TOP_N_LABEL, VOL_FLOOR_DAILY
from quant.config import BENCHMARK_ID
from quant.pipeline.sp500_benchmark import load_sp500_symbols, spy_monthly_returns, spy_quarterly_returns


def _load_sp500_prices() -> pd.DataFrame:
    sp500 = load_sp500_symbols()
    prices = pd.read_parquet(DATA_CURATED_DIR / "prices_daily.parquet").copy()
    prices = prices[prices["symbol"].isin(sp500)].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    prices = prices.sort_values(["symbol", "Date"])
    price_col = "adjusted_close" if "adjusted_close" in prices.columns else "close"
    prices["daily_ret"] = prices.groupby("symbol")[price_col].pct_change()
    prices["price"] = prices[price_col]
    return prices, price_col


def build_labels(force: bool = False) -> pd.DataFrame:
    """Quarterly label: y_t = 1 iff excess_return rank <= TOP_N_LABEL within quarter (SP500)."""
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    out_q = LABELS_DIR / "labels_quarterly.parquet"
    out_m = LABELS_DIR / "labels_monthly.parquet"
    meta_path = LABELS_DIR / "labels_meta.json"
    if out_q.exists() and out_m.exists() and meta_path.exists() and not force:
        return pd.read_parquet(out_q)

    prices, price_col = _load_sp500_prices()
    prices["quarter"] = prices["Date"].dt.to_period("Q").astype(str)
    prices["month"] = prices["Date"].dt.to_period("M").astype(str)

    # --- Quarterly ---
    q_recs: list[dict] = []
    for (sym, q), grp in prices.groupby(["symbol", "quarter"]):
        grp = grp.sort_values("Date")
        if len(grp) < 2:
            continue
        start, end = float(grp["price"].iloc[0]), float(grp["price"].iloc[-1])
        if start <= 0 or np.isnan(start) or np.isnan(end):
            continue
        daily_vol = float(grp["daily_ret"].std())
        if np.isnan(daily_vol):
            daily_vol = 0.0
        q_recs.append(
            {
                "symbol": sym,
                "quarter": q,
                "quarter_start": grp["Date"].iloc[0],
                "quarter_end": grp["Date"].iloc[-1],
                "return": (end - start) / start,
                "daily_vol": daily_vol,
            }
        )

    df = pd.DataFrame(q_recs)
    if df.empty:
        raise ValueError("No quarterly labels could be built.")

    bench_q = spy_quarterly_returns()
    df["bench_return"] = df["quarter"].map(bench_q)
    df["excess_return"] = df["return"] - df["bench_return"]
    df["excess_rank"] = df.groupby("quarter")["excess_return"].rank(ascending=False, method="first")
    df["y_t"] = (df["excess_rank"] <= TOP_N_LABEL).astype(float)
    df["is_top30"] = df["y_t"] >= 1.0

    df = df.sort_values(["symbol", "quarter"])
    df["y_next"] = df.groupby("symbol")["y_t"].shift(-1)
    df["is_top30_next"] = df.groupby("symbol")["is_top30"].shift(-1)

    # --- Monthly (for frontend "actual" top-30 each month) ---
    m_recs: list[dict] = []
    for (sym, mo), grp in prices.groupby(["symbol", "month"]):
        grp = grp.sort_values("Date")
        if len(grp) < 2:
            continue
        start, end = float(grp["price"].iloc[0]), float(grp["price"].iloc[-1])
        if start <= 0 or np.isnan(start) or np.isnan(end):
            continue
        daily_vol = float(grp["daily_ret"].std())
        if np.isnan(daily_vol):
            daily_vol = 0.0
        m_recs.append(
            {
                "symbol": sym,
                "month": mo,
                "month_start": grp["Date"].iloc[0],
                "month_end": grp["Date"].iloc[-1],
                "return": (end - start) / start,
                "daily_vol": max(daily_vol, VOL_FLOOR_DAILY),
            }
        )

    mdf = pd.DataFrame(m_recs)
    bench_m = spy_monthly_returns()
    mdf["bench_return"] = mdf["month"].map(bench_m)
    mdf["excess_return"] = mdf["return"] - mdf["bench_return"]
    mdf["excess_rank"] = mdf.groupby("month")["excess_return"].rank(ascending=False, method="first")
    mdf["y_month"] = (mdf["excess_rank"] <= TOP_N_LABEL).astype(float)
    mdf["is_top30"] = mdf["y_month"] >= 1.0

    meta = {
        "rows_quarterly": int(len(df)),
        "rows_monthly": int(len(mdf)),
        "universe": "sp500_only",
        "benchmark": BENCHMARK_ID,
        "benchmark_symbol": "SPY",
        "top_n_label": TOP_N_LABEL,
        "label_spec": {
            "y_t": f"1 if excess_return rank <= {TOP_N_LABEL} within quarter (vs SPY cap-weight)",
            "y_next": "next quarter y_t",
            "y_month": f"same rule at monthly frequency for realized-month comparison",
        },
    }
    df.to_parquet(out_q, index=False)
    mdf.to_parquet(out_m, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df
