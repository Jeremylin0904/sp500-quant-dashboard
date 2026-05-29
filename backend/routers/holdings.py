from __future__ import annotations

from fastapi import APIRouter

from backend.services.data_service import (
    get_holdings_by_month,
    get_holdings_by_quarter,
    get_holdings_by_signal_quarter,
)

router = APIRouter(prefix="/api/holdings", tags=["holdings"])


@router.get("/signal-quarters")
def list_signal_quarters():
    h = get_holdings_by_signal_quarter()
    return {"signal_quarters": sorted(h.keys())}


@router.get("/signal/{signal_quarter}")
def by_signal_quarter(signal_quarter: str):
    """Holdings vs actual Top30 in the *next* quarter after signal (aligns with y_next)."""
    h = get_holdings_by_signal_quarter()
    entry = h.get(signal_quarter, {})
    return {
        "signal_quarter": signal_quarter,
        "realized_quarter": entry.get("realized_quarter"),
        "path": entry.get("path"),
        "selected": entry.get("selected", []),
        "actual_top30": entry.get("actual_top30", []),
        "hit_count": entry.get("hit_count"),
        "hit_rate": entry.get("hit_rate"),
        "label_note": entry.get("label_note"),
    }


@router.get("/{period}")
def by_period(period: str):
    """period: YYYY-MM (monthly detail) or legacy YYYYQ# quarterly list."""
    if period == "signal-quarters":
        return list_signal_quarters()
    if "Q" in period and len(period) <= 7:
        h = get_holdings_by_quarter()
        return {"period": period, "holdings": h.get(period, [])}
    h = get_holdings_by_month()
    entry = h.get(period, {})
    return {
        "period": period,
        "selected": entry.get("selected", []),
        "actual_top30": entry.get("actual_top30", []),
        "signal_quarter": entry.get("signal_quarter"),
        "realized_month": entry.get("realized_month"),
        "note": "actual_top30 is monthly; use /api/holdings/signal/{q} for quarterly y_next alignment",
    }
