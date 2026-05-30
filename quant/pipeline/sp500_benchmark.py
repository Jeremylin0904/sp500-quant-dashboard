"""S&P 500 market-cap benchmark via SPY (tracks cap-weighted S&P 500 index)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import BENCHMARK_SYMBOL, DATA_CURATED_DIR, SP500_CONSTITUENTS_PATH

_prices_cache: pd.DataFrame | None = None


def load_sp500_symbols() -> set[str]:
    """Load S&P 500 tickers from quant/data/sp500_constituents.csv."""
    path = SP500_CONSTITUENTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"S&P 500 constituents not found: {path}")
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else "symbol"
    syms = set(df[col].astype(str).str.strip().dropna())
    syms = {s.replace(".", "-") for s in syms}
    return syms


def _load_spy_prices() -> pd.DataFrame:
    global _prices_cache
    if _prices_cache is not None:
        return _prices_cache
    path = DATA_CURATED_DIR / "prices_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Curated prices not found: {path}")
    prices = pd.read_parquet(path)
    spy = prices[prices["symbol"] == BENCHMARK_SYMBOL].copy()
    if spy.empty:
        raise ValueError(
            f"{BENCHMARK_SYMBOL} not in curated prices; rebuild data_curated with ETF prices."
        )
    spy["Date"] = pd.to_datetime(spy["Date"])
    price_col = "adjusted_close" if "adjusted_close" in spy.columns else "close"
    spy = spy.sort_values("Date")
    spy["price"] = spy[price_col]
    _prices_cache = spy
    return spy


def _spy_period_returns(freq: str) -> pd.Series:
    """Period return for SPY: first/last close in each quarter or month (same as build_labels)."""
    spy = _load_spy_prices()
    period_col = "quarter" if freq.upper().startswith("Q") else "month"
    if period_col == "quarter":
        spy[period_col] = spy["Date"].dt.to_period("Q").astype(str)
    else:
        spy[period_col] = spy["Date"].dt.to_period("M").astype(str)

    rows: list[dict] = []
    for period, grp in spy.groupby(period_col):
        grp = grp.sort_values("Date")
        if len(grp) < 2:
            continue
        start, end = float(grp["price"].iloc[0]), float(grp["price"].iloc[-1])
        if start <= 0 or np.isnan(start) or np.isnan(end):
            continue
        rows.append({period_col: period, "return": (end - start) / start})

    if not rows:
        raise ValueError(f"No {BENCHMARK_SYMBOL} returns for freq={freq}")
    df = pd.DataFrame(rows)
    return df.set_index(period_col)["return"].rename("bench_return")


def spy_quarterly_returns() -> pd.Series:
    return _spy_period_returns("Q")


def spy_monthly_returns() -> pd.Series:
    return _spy_period_returns("M")


def spy_daily_returns() -> pd.Series:
    """Daily SPY simple returns indexed by Date (pct change of adjusted close)."""
    spy = _load_spy_prices()
    s = spy.sort_values("Date").set_index("Date")["price"]
    return s.pct_change().rename("bench_return")


def spy_daily_prices() -> pd.Series:
    """Daily SPY adjusted-close price level indexed by Date (for within-month NAV)."""
    spy = _load_spy_prices()
    return spy.sort_values("Date").set_index("Date")["price"].rename("spy_price")

