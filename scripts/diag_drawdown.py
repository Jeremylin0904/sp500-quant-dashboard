"""Diagnostic: decompose the worst single-day portfolio returns by stock.

Read-only. Rebuilds the daily NAV exactly like monthly_portfolio._build_daily_nav
(fixed Top-30 weights within each realized month, daily_ret = adjusted_close
pct_change), finds the worst single-day returns per path (cv / oos / merged) and
prints the per-stock contribution (weight x daily_ret) so we can see whether a
-10% day is broad-based or driven by one bad/illiquid name.

Run: $env:PYTHONPATH="d:\\interview2"; python scripts/diag_drawdown.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from quant.config import BACKTEST_DIR, DATA_CURATED_DIR
from quant.pipeline.sp500_benchmark import load_sp500_symbols

TOP_N_DAYS = 5


def _prices() -> pd.DataFrame:
    p = pd.read_parquet(DATA_CURATED_DIR / "prices_daily.parquet")
    sp = set(load_sp500_symbols())
    p = p[p["symbol"].isin(sp)].copy()
    p["Date"] = pd.to_datetime(p["Date"])
    col = "adjusted_close" if "adjusted_close" in p.columns else "close"
    p["price"] = p[col]
    p["daily_ret"] = p.groupby("symbol")["price"].pct_change()
    p["month"] = p["Date"].dt.to_period("M").astype(str)
    return p


def _daily_contribs(holdings: dict, prices: pd.DataFrame, paths: set[str]) -> pd.DataFrame:
    """Per (date) portfolio return + the full per-stock contribution table."""
    recs: list[dict] = []
    for realized_month, h in sorted(holdings.items()):
        if h.get("path") not in paths:
            continue
        weights = {x["symbol"]: x["weight"] for x in h["selected"]}
        if not weights:
            continue
        grp = prices[prices["month"] == realized_month]
        for d, day in grp.groupby("Date"):
            dmap = day.set_index("symbol")["daily_ret"]
            for sym, w in weights.items():
                r = dmap.get(sym)
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                recs.append({"Date": d, "month": realized_month, "symbol": sym,
                             "weight": w, "daily_ret": float(r), "contrib": w * float(r)})
    return pd.DataFrame(recs)


def _report(name: str, contribs: pd.DataFrame) -> None:
    if contribs.empty:
        print(f"\n### {name}: no data")
        return
    day_ret = contribs.groupby("Date")["contrib"].sum().sort_values()
    print(f"\n{'='*88}\n### {name}  (days={day_ret.size}, worst single-day={day_ret.min()*100:.2f}%)\n{'='*88}")
    for d in day_ret.head(TOP_N_DAYS).index:
        tot = day_ret.loc[d]
        sub = contribs[contribs["Date"] == d].sort_values("contrib")
        n_held = sub["symbol"].nunique()
        print(f"\n{d.date()}  portfolio={tot*100:+.2f}%  (held {n_held} names)")
        print(f"  {'symbol':<8} {'weight%':>8} {'day_ret%':>9} {'contrib%':>9}")
        for _, r in sub.head(6).iterrows():
            print(f"  {r['symbol']:<8} {r['weight']*100:>8.2f} {r['daily_ret']*100:>9.2f} {r['contrib']*100:>9.3f}")
        worst = sub.iloc[0]
        share = worst["contrib"] / tot * 100 if tot != 0 else 0
        print(f"  -> worst name {worst['symbol']} = {share:.0f}% of the day's loss; "
              f"min day_ret among held = {sub['daily_ret'].min()*100:.1f}%")


def main() -> None:
    holdings = json.loads((BACKTEST_DIR / "holdings_by_month.json").read_text(encoding="utf-8"))
    prices = _prices()
    cv = _daily_contribs(holdings, prices, {"walk_forward_cv"})
    oos = _daily_contribs(holdings, prices, {"walk_forward_oos"})
    merged = _daily_contribs(holdings, prices, {"walk_forward_cv", "walk_forward_oos"})
    _report("CV OOF (in-sample)", cv)
    _report("Final OOS", oos)
    _report("Merged (CV+OOS)", merged)


if __name__ == "__main__":
    main()
