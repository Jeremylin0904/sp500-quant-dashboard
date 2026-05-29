"""Fetch real US data from SimFin into quant/data_raw/ following our contract.

Requires SIMFIN_API_KEY in project root .env (or environment).
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from quant.config import (
    BALANCE_Q_CSV,
    CASHFLOW_Q_CSV,
    CONSTITUENTS_CSV,
    DATA_RAW_DIR,
    INCOME_Q_CSV,
    PRICES_DAILY_CSV,
)


def fetch_simfin_real_data(force: bool = False) -> None:
    """Download SimFin quarterly statements + daily shareprices and write CSVs.

    Notes:
    - SimFin's datasets may contain more columns than we need. We select a minimal subset.
    - cashflow_quarterly.csv is optional; if unavailable in the user's SimFin tier, we skip it.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if (
        PRICES_DAILY_CSV.exists()
        and INCOME_Q_CSV.exists()
        and BALANCE_Q_CSV.exists()
        and CONSTITUENTS_CSV.exists()
        and not force
    ):
        return

    # Load .env if present
    load_dotenv()
    key = os.getenv("SIMFIN_API_KEY")
    if not key:
        raise RuntimeError("SIMFIN_API_KEY is not set. Put it in .env or environment.")

    import simfin as sf

    sf.set_api_key(key)
    sf.set_data_dir(str(DATA_RAW_DIR / "_simfin_cache"))

    # Fundamentals
    inc = sf.load_income(variant="quarterly", market="us")
    bal = sf.load_balance(variant="quarterly", market="us")
    try:
        cfs = sf.load_cashflow(variant="quarterly", market="us")
    except Exception:
        cfs = None

    # Shareprices (daily) – includes OHLCV; we only keep close+volume
    px = sf.load_shareprices(market="us", variant="daily")

    # SimFin returns MultiIndex frames; normalize to columns
    def _reset(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex):
            return df.reset_index()
        return df.reset_index(drop=False)

    inc = _reset(inc)
    bal = _reset(bal)
    px = _reset(px)
    if cfs is not None:
        cfs = _reset(cfs)

    # Constituents: company_name + sector/industry via SimFin IndustryId → industries lookup
    try:
        companies = sf.load_companies(market="us")
        companies = _reset(companies)
        comp = companies.rename(
            columns={
                "Ticker": "symbol",
                "Company Name": "company_name",
                "IndustryId": "industry_id",
            }
        )
        for c in ["symbol", "company_name", "industry_id"]:
            if c not in comp.columns:
                comp[c] = pd.NA
        comp = comp[["symbol", "company_name", "industry_id"]].drop_duplicates("symbol")

        try:
            industries = sf.load_industries()
            if isinstance(industries.index, pd.Index) and industries.index.name == "IndustryId":
                ind = industries.reset_index()
            else:
                ind = industries.reset_index(drop=False)
            ind = ind.rename(columns={"IndustryId": "industry_id", "Sector": "sector", "Industry": "industry"})
            comp = comp.merge(ind[["industry_id", "sector", "industry"]], on="industry_id", how="left")
        except Exception:
            comp["sector"] = pd.NA
            comp["industry"] = pd.NA

        comp = comp.drop(columns=["industry_id"], errors="ignore")
    except Exception:
        comp = pd.DataFrame(columns=["symbol", "company_name", "sector", "industry"])

    # Map column names to our contract
    def pick(df: pd.DataFrame, mapping: dict[str, str], required: list[str]) -> pd.DataFrame:
        out = df.rename(columns=mapping)
        for r in required:
            if r not in out.columns:
                out[r] = pd.NA
        return out[required]

    # Income
    inc_map = {
        "Ticker": "symbol",
        "Report Date": "report_date",
        "Publish Date": "publish_date",
        "Revenue": "revenue",
        "Gross Profit": "gross_profit",
        "Operating Income (Loss)": "operating_income",
        "Net Income": "net_income",
        "Shares (Basic)": "shares_basic",
        "Shares (Diluted)": "shares_diluted",
        "Net Income (Common)": "net_income_common",
    }
    inc_req = [
        "symbol",
        "report_date",
        "publish_date",
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "shares_basic",
        "shares_diluted",
        "net_income_common",
    ]
    inc_out = pick(inc, inc_map, inc_req)
    inc_out["report_date"] = pd.to_datetime(inc_out["report_date"])
    inc_out["publish_date"] = pd.to_datetime(inc_out["publish_date"])
    # SimFin income quarterly doesn't always include EPS; derive a PIT-consistent proxy:
    # eps_diluted ≈ Net Income (Common) / Shares (Diluted)
    inc_out["eps_diluted"] = pd.to_numeric(inc_out["net_income_common"], errors="coerce") / pd.to_numeric(
        inc_out["shares_diluted"], errors="coerce"
    ).replace(0, pd.NA)
    inc_out = inc_out.drop(columns=["net_income_common"])
    inc_out.to_csv(INCOME_Q_CSV, index=False)

    # Balance
    bal_map = {
        "Ticker": "symbol",
        "Report Date": "report_date",
        "Publish Date": "publish_date",
        "Total Assets": "total_assets",
        "Total Liabilities": "total_liabilities",
        "Total Equity": "total_equity",
        "Cash, Cash Equivalents & Short Term Investments": "cash_and_equivalents",
        "Short Term Debt": "short_term_debt",
        "Long Term Debt": "long_term_debt",
    }
    bal_req = [
        "symbol",
        "report_date",
        "publish_date",
        "total_assets",
        "total_liabilities",
        "total_equity",
        "cash_and_equivalents",
        "short_term_debt",
        "long_term_debt",
    ]
    bal_out = pick(bal, bal_map, bal_req)
    bal_out["report_date"] = pd.to_datetime(bal_out["report_date"])
    bal_out["publish_date"] = pd.to_datetime(bal_out["publish_date"])
    bal_out["total_debt"] = pd.to_numeric(bal_out["short_term_debt"], errors="coerce").fillna(0) + pd.to_numeric(
        bal_out["long_term_debt"], errors="coerce"
    ).fillna(0)
    bal_out = bal_out.drop(columns=["short_term_debt", "long_term_debt"])
    bal_out.to_csv(BALANCE_Q_CSV, index=False)

    # Cashflow (optional)
    # SimFin US quarterly cashflow often does NOT include explicit "Capital Expenditures" / "Free Cash Flow".
    # Proxy (documented in README): use "Change in Fixed Assets & Intangibles" as the capex-like cash line.
    # - capital_expenditure: max(0, -change) when outflows are negative (typical for purchases).
    # - free_cash_flow: operating_cash_flow + change_in_fixed_assets (same sign convention as SimFin export).
    if cfs is not None:
        cfs_map = {
            "Ticker": "symbol",
            "Report Date": "report_date",
            "Publish Date": "publish_date",
            "Net Cash from Operating Activities": "operating_cash_flow",
            "Change in Fixed Assets & Intangibles": "change_in_fixed_assets",
        }
        cfs_req = [
            "symbol",
            "report_date",
            "publish_date",
            "operating_cash_flow",
            "change_in_fixed_assets",
        ]
        cfs_out = pick(cfs, cfs_map, cfs_req)
        cfs_out["report_date"] = pd.to_datetime(cfs_out["report_date"])
        cfs_out["publish_date"] = pd.to_datetime(cfs_out["publish_date"])
        ocf = pd.to_numeric(cfs_out["operating_cash_flow"], errors="coerce")
        dfa = pd.to_numeric(cfs_out["change_in_fixed_assets"], errors="coerce")
        cfs_out["capital_expenditure"] = (-dfa).clip(lower=0)
        cfs_out["free_cash_flow"] = ocf + dfa
        cfs_out = cfs_out.drop(columns=["change_in_fixed_assets"])
        cfs_out.to_csv(CASHFLOW_Q_CSV, index=False)

    # Prices
    px_map = {
        "Ticker": "symbol",
        "Date": "Date",
        "Close": "close",
        "Adj. Close": "adjusted_close",
        "Volume": "volume",
        "Shares Outstanding": "shares_outstanding",
    }
    px_req = ["Date", "symbol", "close", "adjusted_close", "volume", "shares_outstanding"]
    px_out = pick(px, px_map, px_req)
    px_out["Date"] = pd.to_datetime(px_out["Date"])
    px_out.to_csv(PRICES_DAILY_CSV, index=False)

    # Constituents
    if comp.empty:
        comp = pd.DataFrame({"symbol": sorted(inc_out["symbol"].dropna().unique())})
        comp["company_name"] = None
        comp["sector"] = "Unknown"
        comp["industry"] = None
    else:
        comp["sector"] = comp["sector"].fillna("Unknown")
        comp["industry"] = comp["industry"].fillna("Unknown")
    comp.to_csv(CONSTITUENTS_CSV, index=False)


if __name__ == "__main__":
    fetch_simfin_real_data(force=True)
    print("Done:", date.today().isoformat())

