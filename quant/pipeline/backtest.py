"""Run monthly portfolio backtest (inverse-vol) and legacy quarterly curve."""

from __future__ import annotations

from quant.pipeline.monthly_portfolio import run_monthly_backtest


def run_backtest(force: bool = False) -> dict:
    return run_monthly_backtest(force=force)
