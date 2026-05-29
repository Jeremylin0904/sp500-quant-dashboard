#!/usr/bin/env python3
"""Pre-render every read-only API endpoint to static JSON so the dashboard can
be served as a fully static site (e.g. GitHub Pages) with no running backend.

Output mirrors the live API paths under frontend/public/api/<path>.json:
  /api/backtest/report          -> frontend/public/api/backtest/report.json
  /api/holdings/signal/2024Q1   -> frontend/public/api/holdings/signal/2024Q1.json

Run:  python scripts/export_static_api.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.services.data_service import (  # noqa: E402
    build_model_summary,
    build_model_variables,
    clear_cache,
    get_backtest_daily,
    get_backtest_monthly,
    get_drawdown_news,
    get_factor_analysis,
    get_holdings_by_signal_quarter,
    get_performance_report,
)

OUT = ROOT / "frontend" / "public" / "api"


def _report() -> dict:
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


def _signal_entry(q: str, entry: dict) -> dict:
    return {
        "signal_quarter": q,
        "realized_quarter": entry.get("realized_quarter"),
        "path": entry.get("path"),
        "selected": entry.get("selected", []),
        "actual_top30": entry.get("actual_top30", []),
        "hit_count": entry.get("hit_count"),
        "hit_rate": entry.get("hit_rate"),
        "label_note": entry.get("label_note"),
    }


def _write(rel: str, payload) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote api/{rel}")


def main() -> None:
    clear_cache()
    _write("backtest/report.json", _report())
    _write("backtest/monthly.json", get_backtest_monthly())
    _write("backtest/daily.json", get_backtest_daily())
    _write("backtest/drawdown-news.json", get_drawdown_news())
    _write("model/summary.json", build_model_summary())
    _write("model/variables.json", build_model_variables())
    _write("factor/analysis.json", get_factor_analysis())

    signal = get_holdings_by_signal_quarter()
    quarters = sorted(signal.keys())
    _write("holdings/signal-quarters.json", {"signal_quarters": quarters})
    for q in quarters:
        _write(f"holdings/signal/{q}.json", _signal_entry(q, signal.get(q, {})))

    print(f"\nOK: exported {7 + 1 + len(quarters)} files to {OUT}")


if __name__ == "__main__":
    main()
