"""Build PIT-aligned quarterly feature table (clean rebuild).

Single source of truth for the new pipeline.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from quant.config import DATA_CURATED_DIR, FEATURES_DIR
from quant.pipeline.sp500_benchmark import load_sp500_symbols


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def build_features(force: bool = False) -> pd.DataFrame:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FEATURES_DIR / "features_quarterly.parquet"
    meta_path = FEATURES_DIR / "features_meta.json"
    if out_path.exists() and meta_path.exists() and not force:
        return pd.read_parquet(out_path)

    sp500 = load_sp500_symbols()

    prices = pd.read_parquet(DATA_CURATED_DIR / "prices_daily.parquet")
    prices = prices[prices["symbol"].isin(sp500)].copy()
    prices["quarter"] = prices["Date"].dt.to_period("Q").astype(str)
    q_end = (
        prices.sort_values(["symbol", "Date"])
        .groupby(["symbol", "quarter"], as_index=False)
        .agg(
            as_of_date=("Date", "max"),
            close=("close", "last"),
            adjusted_close=("adjusted_close", "last") if "adjusted_close" in prices.columns else ("close", "last"),
            volume=("volume", "last"),
            px_shares_outstanding=("shares_outstanding", "last") if "shares_outstanding" in prices.columns else ("volume", "size"),
        )
    )

    const = pd.read_parquet(DATA_CURATED_DIR / "constituents.parquet")
    const = const[const["symbol"].isin(sp500)].copy()
    inc = pd.read_parquet(DATA_CURATED_DIR / "income_quarterly.parquet")
    inc = inc[inc["symbol"].isin(sp500)].copy()
    bal = pd.read_parquet(DATA_CURATED_DIR / "balance_quarterly.parquet")
    bal = bal[bal["symbol"].isin(sp500)].copy()
    cfs_path = DATA_CURATED_DIR / "cashflow_quarterly.parquet"
    cfs = pd.read_parquet(cfs_path) if cfs_path.exists() else None
    if cfs is not None:
        cfs = cfs[cfs["symbol"].isin(sp500)].copy()

    fund = inc.merge(bal, on=["symbol", "report_date", "publish_date"], how="inner")
    if cfs is not None:
        fund = fund.merge(cfs, on=["symbol", "report_date", "publish_date"], how="left")
    fund = fund.merge(const, on="symbol", how="left")
    fund = fund.sort_values(["symbol", "report_date"])

    for col in ["revenue", "gross_profit", "operating_income", "net_income"]:
        fund[f"{col}_ttm"] = fund.groupby("symbol")[col].transform(
            lambda s: s.rolling(4, min_periods=4).sum()
        )
    if cfs is not None:
        for col in ["operating_cash_flow", "capital_expenditure", "free_cash_flow"]:
            if col in fund.columns:
                fund[f"{col}_ttm"] = fund.groupby("symbol")[col].transform(
                    lambda s: s.rolling(4, min_periods=4).sum()
                )

    fund["gross_margin"] = _safe_div(fund["gross_profit_ttm"], fund["revenue_ttm"])
    fund["operating_margin"] = _safe_div(fund["operating_income_ttm"], fund["revenue_ttm"])
    fund["net_margin"] = _safe_div(fund["net_income_ttm"], fund["revenue_ttm"])
    fund["roe"] = _safe_div(fund["net_income_ttm"], fund["total_equity"])
    fund["roa"] = _safe_div(fund["net_income_ttm"], fund["total_assets"])

    fund["shares_outstanding"] = fund["shares_diluted"].combine_first(fund["shares_basic"])

    # PIT join: latest publish_date <= as_of_date
    rows: list[pd.Series] = []
    fund = fund.sort_values(["symbol", "publish_date", "report_date"])
    for sym, grp in q_end.groupby("symbol"):
        f = fund[fund["symbol"] == sym]
        if f.empty:
            continue
        for _, pr in grp.iterrows():
            avail = f[f["publish_date"] <= pr["as_of_date"]]
            if avail.empty:
                continue
            r = avail.iloc[-1].copy()
            r["quarter"] = pr["quarter"]
            r["as_of_date"] = pr["as_of_date"]
            r["close"] = pr["close"]
            r["adjusted_close"] = pr.get("adjusted_close", pr["close"])
            r["volume"] = pr["volume"]
            # Prefer shares outstanding from prices (PIT at as_of_date) when present.
            px_sh = pr.get("px_shares_outstanding")
            if "shares_outstanding" in r.index:
                if pd.notna(px_sh) and float(px_sh) > 0:
                    r["shares_outstanding"] = px_sh
            rows.append(r)

    feat = pd.DataFrame(rows)
    if feat.empty:
        raise ValueError("No PIT-aligned features could be built")

    # Use adjusted close for market cap to reduce split artifacts.
    feat["market_cap"] = feat["adjusted_close"] * feat["shares_outstanding"]
    feat["enterprise_value"] = feat["market_cap"] + feat["total_debt"] - feat["cash_and_equivalents"]

    feat["pb_ratio"] = _safe_div(feat["market_cap"], feat["total_equity"])
    feat["pe_ratio"] = _safe_div(feat["market_cap"], feat["net_income_ttm"])
    feat["earnings_yield"] = _safe_div(feat["net_income_ttm"], feat["market_cap"])
    if "free_cash_flow_ttm" in feat.columns:
        feat["fcf_yield"] = _safe_div(feat["free_cash_flow_ttm"], feat["market_cap"])
    else:
        feat["fcf_yield"] = _safe_div(feat["net_income_ttm"], feat["market_cap"])  # fallback proxy
    feat["debt_to_equity"] = _safe_div(feat["total_debt"], feat["total_equity"])
    feat["debt_to_assets"] = _safe_div(feat["total_debt"], feat["total_assets"])
    feat["ebit_to_tev"] = _safe_div(feat["operating_income_ttm"], feat["enterprise_value"])
    feat["ev_to_ebit"] = _safe_div(feat["enterprise_value"], feat["operating_income_ttm"])

    # Growth
    feat = feat.sort_values(["symbol", "report_date"])
    feat["revenue_growth_yoy"] = feat.groupby("symbol")["revenue_ttm"].pct_change(4, fill_method=None)
    feat["eps_growth_yoy"] = feat.groupby("symbol")["eps_diluted"].pct_change(4, fill_method=None)

    # Momentum/liquidity
    feat = feat.sort_values(["symbol", "quarter"])
    feat["return_3m"] = feat.groupby("symbol")["close"].pct_change(1, fill_method=None)
    feat["return_6m"] = feat.groupby("symbol")["close"].pct_change(2, fill_method=None)
    feat["return_12m"] = feat.groupby("symbol")["close"].pct_change(4, fill_method=None)
    feat["volume_turnover"] = _safe_div(feat["volume"], feat["shares_outstanding"])

    def _rank(col: str) -> pd.Series:
        return feat.groupby(["quarter", "sector"])[col].rank(pct=True)

    feat["sector"] = feat["sector"].fillna("Unknown")
    feat["sector_rank_revenue_growth"] = _rank("revenue_growth_yoy")
    feat["sector_rank_eps_growth"] = _rank("eps_growth_yoy")
    feat["sector_rank_roe"] = _rank("roe")
    feat["sector_rank_operating_margin"] = _rank("operating_margin")
    feat["sector_rank_ebit_to_tev"] = _rank("ebit_to_tev")
    feat["sector_rank_fcf_yield"] = _rank("fcf_yield")
    feat["sector_rank_return_12m"] = _rank("return_12m")
    feat["sector_rank_low_debt"] = 1 - _rank("debt_to_assets")

    keep = [
        "symbol",
        "company_name",
        "sector",
        "industry",
        "quarter",
        "as_of_date",
        "report_date",
        "publish_date",
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "cash_and_equivalents",
        "total_debt",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
        "shares_outstanding",
        "close",
        "adjusted_close",
        "volume",
        "market_cap",
        "enterprise_value",
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roe",
        "roa",
        "pe_ratio",
        "pb_ratio",
        "ev_to_ebit",
        "ebit_to_tev",
        "fcf_yield",
        "earnings_yield",
        "debt_to_equity",
        "debt_to_assets",
        "return_3m",
        "return_6m",
        "return_12m",
        "volume_turnover",
        "sector_rank_revenue_growth",
        "sector_rank_eps_growth",
        "sector_rank_roe",
        "sector_rank_operating_margin",
        "sector_rank_ebit_to_tev",
        "sector_rank_fcf_yield",
        "sector_rank_return_12m",
        "sector_rank_low_debt",
    ]
    feat = feat[[c for c in keep if c in feat.columns]].sort_values(["quarter", "symbol"]).reset_index(drop=True)

    feat.to_parquet(out_path, index=False)
    meta_path.write_text(json.dumps({"rows": int(len(feat)), "cols": list(feat.columns)}, indent=2), encoding="utf-8")
    return feat

