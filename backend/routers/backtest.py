from __future__ import annotations

from fastapi import APIRouter

from backend.services.data_service import (
    get_backtest_curve,
    get_backtest_daily,
    get_backtest_monthly,
    get_drawdown_news,
    get_holdings_by_month,
    get_performance_report,
)

router = APIRouter(prefix="/api/backtest", tags=["backtest"])


@router.get("/curve")
def curve():
    return get_backtest_curve()


@router.get("/monthly")
def monthly():
    return get_backtest_monthly()


@router.get("/daily")
def daily():
    return get_backtest_daily()


@router.get("/drawdown-news")
def drawdown_news():
    return get_drawdown_news()


@router.get("/report")
def report():
    r = get_performance_report()
    return {
        "benchmark": r.get("benchmark"),
        "benchmark_symbol": "SPY",
        "top_n_holdings": r.get("top_n_holdings"),
        "label_top_n": r.get("label_top_n"),
        "weighting": r.get("weighting"),
        "in_sample": r.get("in_sample"),
        "out_of_sample": r.get("out_of_sample"),
    }


@router.get("/quarters")
def quarters():
    curve = get_backtest_curve()
    qs = sorted({p.get("quarter") for p in curve.get("quarterly", []) if p.get("quarter")})
    return {"quarters": qs}


@router.get("/months")
def months():
    h = get_holdings_by_month()
    return {"months": sorted(h.keys())}
