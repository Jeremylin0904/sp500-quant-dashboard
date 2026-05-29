"""Generate tiny raw CSVs for a reproducible interview MVP.

If the user provides real raw data later, they can replace files under quant/data_raw/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.config import (
    DATA_RAW_DIR,
    PRICES_DAILY_CSV,
    CONSTITUENTS_CSV,
    INCOME_Q_CSV,
    BALANCE_Q_CSV,
)


def ensure_sample_raw_data(force: bool = False) -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    if PRICES_DAILY_CSV.exists() and CONSTITUENTS_CSV.exists() and INCOME_Q_CSV.exists() and BALANCE_Q_CSV.exists() and not force:
        return

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    company = ["Alpha Inc", "Beta Corp", "Gamma Ltd", "Delta PLC"]
    sector = ["Tech", "Tech", "Industrials", "Health Care"]
    industry = ["Software", "Hardware", "Machinery", "Biotech"]

    constituents = pd.DataFrame(
        {"symbol": symbols, "company_name": company, "sector": sector, "industry": industry}
    )
    constituents.to_csv(CONSTITUENTS_CSV, index=False)

    # Prices: daily for 6 quarters
    dates = pd.date_range("2022-01-03", "2023-06-30", freq="B")
    rows = []
    rng = np.random.default_rng(0)
    for sym in symbols:
        price = 100 + 10 * rng.normal()
        vol = 1_000_000
        for d in dates:
            price *= float(1 + rng.normal(0, 0.01))
            vol = int(max(1000, vol * float(1 + rng.normal(0, 0.05))))
            rows.append({"Date": d.strftime("%Y-%m-%d"), "symbol": sym, "close": price, "volume": vol})
    pd.DataFrame(rows).to_csv(PRICES_DAILY_CSV, index=False)

    # Quarterly fundamentals: report_date at quarter end, publish 30 days later
    q_ends = pd.period_range("2021Q4", "2023Q1", freq="Q").to_timestamp("Q").normalize()
    inc_rows = []
    bal_rows = []
    for sym in symbols:
        base_rev = float(1e9 * (1 + 0.2 * rng.normal()))
        shares = float(200e6 * (1 + 0.1 * rng.normal()))
        equity = float(5e9 * (1 + 0.2 * rng.normal()))
        debt = float(2e9 * (1 + 0.2 * rng.normal()))
        cash = float(1e9 * (1 + 0.2 * rng.normal()))
        assets = equity + debt + cash
        for rd in q_ends:
            pub = (rd + pd.Timedelta(days=30)).normalize()
            rev = base_rev * float(1 + 0.03 * rng.normal())
            gp = rev * float(0.55 + 0.05 * rng.normal())
            op = rev * float(0.20 + 0.05 * rng.normal())
            ni = rev * float(0.12 + 0.04 * rng.normal())
            eps = ni / shares
            inc_rows.append(
                {
                    "symbol": sym,
                    "report_date": rd.strftime("%Y-%m-%d"),
                    "publish_date": pub.strftime("%Y-%m-%d"),
                    "revenue": rev,
                    "gross_profit": gp,
                    "operating_income": op,
                    "net_income": ni,
                    "eps_diluted": eps,
                    "shares_basic": shares,
                    "shares_diluted": shares * 1.01,
                }
            )
            equity *= float(1 + 0.02 * rng.normal())
            debt *= float(1 + 0.01 * rng.normal())
            cash *= float(1 + 0.03 * rng.normal())
            assets = equity + debt + cash
            bal_rows.append(
                {
                    "symbol": sym,
                    "report_date": rd.strftime("%Y-%m-%d"),
                    "publish_date": pub.strftime("%Y-%m-%d"),
                    "total_assets": assets,
                    "total_liabilities": debt,
                    "total_equity": equity,
                    "cash_and_equivalents": cash,
                    "total_debt": debt,
                }
            )

    pd.DataFrame(inc_rows).to_csv(INCOME_Q_CSV, index=False)
    pd.DataFrame(bal_rows).to_csv(BALANCE_Q_CSV, index=False)

