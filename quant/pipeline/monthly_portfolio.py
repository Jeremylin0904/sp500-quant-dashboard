"""Monthly rebalance backtest: inverse-vol weights, expanding walk-forward scores."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from quant.config import (
    BACKTEST_DIR,
    BENCHMARK_ID,
    DATA_CURATED_DIR,
    LABELS_DIR,
    MODEL_DIR,
    TOP_N_HOLDINGS,
    TOP_N_LABEL,
    VOL_FLOOR_DAILY,
    WEIGHT_SCHEME,
)
from quant.pipeline.sp500_benchmark import (
    load_sp500_symbols,
    spy_daily_returns,
    spy_monthly_returns,
)
from quant.pipeline.train_model import prepare_feature_matrix

RET_CLIP_LOWER, RET_CLIP_UPPER = -0.9, 3.0


def _monthly_stock_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """symbol x month -> return, daily_vol, month_end."""
    price_col = "price"
    rows: list[dict] = []
    for (sym, mo), grp in prices.groupby(["symbol", "month"]):
        grp = grp.sort_values("Date")
        if len(grp) < 2:
            continue
        s, e = float(grp[price_col].iloc[0]), float(grp[price_col].iloc[-1])
        if s <= 0 or np.isnan(s) or np.isnan(e):
            continue
        vol = float(grp["daily_ret"].std())
        if np.isnan(vol):
            vol = VOL_FLOOR_DAILY
        rows.append(
            {
                "symbol": sym,
                "month": mo,
                "month_end": grp["Date"].iloc[-1],
                "return": (e - s) / s,
                "daily_vol": max(vol, VOL_FLOOR_DAILY),
            }
        )
    return pd.DataFrame(rows)


def _normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    total = sum(raw.values())
    if total <= 0 or not raw:
        n = len(raw)
        return {s: 1.0 / n for s in raw} if n else {}
    return {s: w / total for s, w in raw.items()}


def _inverse_vol_weights(symbols: list[str], vols: pd.Series) -> dict[str, float]:
    inv = {s: 1.0 / max(float(vols.get(s, VOL_FLOOR_DAILY)), VOL_FLOOR_DAILY) for s in symbols}
    return _normalize_weights(inv)


def portfolio_weights(
    top: pd.DataFrame,
    vols: pd.Series,
    scheme: str = "inv_vol",
    *,
    softmax_temperature: float = 0.1,
) -> dict[str, float]:
    """Top-K rows must include symbol, score; sorted by score desc preferred."""
    n = len(top)
    if n == 0:
        return {}

    if scheme == "equal":
        return {s: 1.0 / n for s in top["symbol"]}

    if scheme == "inv_vol":
        return _inverse_vol_weights(top["symbol"].tolist(), vols)

    if scheme == "rank_linear":
        raw = {row.symbol: float(n - i) for i, row in enumerate(top.itertuples(index=False))}
        return _normalize_weights(raw)

    if scheme == "softmax_sharpe":
        score_by_sym = top.set_index("symbol")["score"]
        logits: list[tuple[str, float]] = []
        for s in top["symbol"]:
            sigma = max(float(vols.get(s, VOL_FLOOR_DAILY)), VOL_FLOOR_DAILY)
            logits.append((s, float(score_by_sym[s]) / (softmax_temperature * sigma)))
        max_logit = max(v for _, v in logits)
        raw = {s: float(np.exp(v - max_logit)) for s, v in logits}
        return _normalize_weights(raw)

    if scheme.startswith("softmax"):
        if scheme != "softmax":
            # legacy: softmax_0.1 → score-only softmax
            softmax_temperature = float(scheme.split("_", 1)[1])
        score_by_sym = top.set_index("symbol")["score"]
        logits = [(s, float(score_by_sym[s]) / softmax_temperature) for s in top["symbol"]]
        max_logit = max(v for _, v in logits)
        raw = {s: float(np.exp(v - max_logit)) for s, v in logits}
        return _normalize_weights(raw)

    if scheme == "score_inv_vol":
        score_by_sym = top.set_index("symbol")["score"]
        raw = {
            s: float(score_by_sym[s]) / max(float(vols.get(s, VOL_FLOOR_DAILY)), VOL_FLOOR_DAILY)
            for s in top["symbol"]
        }
        return _normalize_weights(raw)

    raise ValueError(f"Unknown weight scheme: {scheme}")


def _next_quarter(signal_q: str, quarters: list[str]) -> str | None:
    qs = sorted(quarters)
    if signal_q not in qs:
        return None
    i = qs.index(signal_q)
    if i >= len(qs) - 1:
        return None
    return qs[i + 1]


def _build_quarterly_holdings(
    holdings_monthly: dict[str, dict],
    labels_q: pd.DataFrame,
    quarters: list[str],
) -> dict[str, dict]:
    """One row per signal quarter: selected vs actual Top30 in the *next* quarter (aligns with y_next)."""
    by_sig: dict[str, dict] = {}
    for _realized, h in holdings_monthly.items():
        sig = h.get("signal_quarter")
        if not sig or sig in by_sig:
            continue
        by_sig[sig] = h

    out: dict[str, dict] = {}
    for sig_q, h in sorted(by_sig.items()):
        realized_q = _next_quarter(sig_q, quarters)
        if not realized_q:
            continue
        selected = h.get("selected") or []
        realized_rows = labels_q[labels_q["quarter"] == realized_q]
        realized_by_sym = realized_rows.set_index("symbol")
        actual = realized_rows[realized_rows["is_top30"]].sort_values(
            "excess_return", ascending=False
        )
        sel_syms = {x["symbol"] for x in selected}
        act_syms = set(actual["symbol"])
        hit_syms = sel_syms & act_syms
        # Enrich each selected stock with its realized next-quarter return (vs predicted score/weight).
        selected_enriched = []
        for x in selected:
            sym = x["symbol"]
            row = realized_by_sym.loc[sym] if sym in realized_by_sym.index else None
            selected_enriched.append(
                {
                    **x,
                    "realized_return": float(row["return"]) if row is not None else None,
                    "realized_excess": float(row["excess_return"]) if row is not None else None,
                    "is_top30_actual": bool(sym in act_syms),
                }
            )
        out[sig_q] = {
            "signal_quarter": sig_q,
            "realized_quarter": realized_q,
            "path": h.get("path"),
            "selected": selected_enriched,
            "actual_top30": [
                {
                    "symbol": r["symbol"],
                    "excess_return": float(r["excess_return"]),
                    "return": float(r["return"]),
                    "y_label": 1,
                }
                for _, r in actual.iterrows()
            ],
            "hit_count": len(hit_syms),
            "hit_rate": len(hit_syms) / len(selected) if selected else 0.0,
            "label_note": "actual_top30 = excess rank Top30 within realized quarter (same as y_next label)",
        }
    return out


def _signal_quarter_for_month(month_end: pd.Timestamp, q_ends: pd.Series) -> str | None:
    eligible = q_ends[q_ends <= month_end]
    if eligible.empty:
        return None
    return str(eligible.index[-1])


def _build_portfolio_context(feature_cols: list[str] | None = None) -> dict:
    """Load prices, monthly returns, labels for portfolio simulation."""
    ds = pd.read_parquet(MODEL_DIR / "model_dataset.parquet")
    quarters = sorted(ds["quarter"].unique())
    q_ends = pd.to_datetime(ds.groupby("quarter")["quarter_end"].min())

    sp500 = load_sp500_symbols()
    prices = pd.read_parquet(DATA_CURATED_DIR / "prices_daily.parquet")
    prices = prices[prices["symbol"].isin(sp500)].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    price_col = "adjusted_close" if "adjusted_close" in prices.columns else "close"
    prices["price"] = prices[price_col]
    prices["daily_ret"] = prices.groupby("symbol")["price"].pct_change()
    prices["month"] = prices["Date"].dt.to_period("M").astype(str)

    mret = _monthly_stock_returns(prices)
    bench_m = spy_monthly_returns()
    mlabels = pd.read_parquet(LABELS_DIR / "labels_monthly.parquet")
    month_ends = sorted(pd.Timestamp(x) for x in mret.groupby("month")["month_end"].max().tolist())

    if feature_cols is None:
        meta_path = MODEL_DIR / "model_meta.json"
        if meta_path.exists():
            feature_cols = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_cols") or []
        else:
            feature_cols = []

    return {
        "ds": ds,
        "quarters": quarters,
        "q_ends": q_ends,
        "mret": mret,
        "bench_m": bench_m,
        "mlabels": mlabels,
        "feature_cols": feature_cols,
        "month_ends": month_ends,
    }


_portfolio_ctx_cache: dict | None = None


def get_portfolio_context(force_reload: bool = False, feature_cols: list[str] | None = None) -> dict:
    global _portfolio_ctx_cache
    if _portfolio_ctx_cache is None or force_reload or feature_cols is not None:
        ctx = _build_portfolio_context(feature_cols)
        if feature_cols is None:
            _portfolio_ctx_cache = ctx
        return ctx
    return _portfolio_ctx_cache


def eval_portfolio_on_wf_scores(
    wf: pd.DataFrame,
    allowed_signal_quarters: set[str],
    *,
    feature_cols: list[str] | None = None,
    weight_scheme: str = WEIGHT_SCHEME,
    softmax_temperature: float = 0.1,
) -> dict:
    """Monthly CV OOF portfolio metrics using per-quarter scores from walk-forward output."""
    ctx = get_portfolio_context(feature_cols=feature_cols)
    score_col = "score_wf" if "score_wf" in wf.columns else "score_oof"
    wf_by_q = {q: g.set_index("symbol")[score_col] for q, g in wf.groupby("quarter")}

    def score_fn(q: str, _df: pd.DataFrame) -> pd.Series | None:
        return wf_by_q.get(q)

    _, _, summary, _ = _run_monthly_path(
        ctx["month_ends"],
        ctx["mret"],
        ctx["bench_m"],
        ctx["mlabels"],
        ctx["ds"],
        ctx["quarters"],
        ctx["q_ends"],
        ctx["feature_cols"],
        score_fn,
        allowed_signal_quarters,
        "pool_eval",
        weight_scheme=weight_scheme,
        softmax_temperature=softmax_temperature,
    )
    return summary


def _run_monthly_path(
    month_ends: list[pd.Timestamp],
    mret: pd.DataFrame,
    bench_m: pd.Series,
    mlabels: pd.DataFrame,
    ds: pd.DataFrame,
    quarters: list[str],
    q_ends: pd.Series,
    feature_cols: list[str],
    score_fn,
    allowed_signal_quarters: set[str],
    path_name: str,
    weight_scheme: str = "inv_vol",
    softmax_temperature: float = 0.1,
) -> tuple[list[dict], list[dict], dict, pd.DataFrame]:
    """Rebalance at each month-end in month_ends; realize next calendar month."""
    mret = mret.copy()
    mret["return_capped"] = mret["return"].clip(RET_CLIP_LOWER, RET_CLIP_UPPER)
    months_sorted = sorted(mret["month"].unique())
    month_to_idx = {m: i for i, m in enumerate(months_sorted)}

    rows_by_q = {q: ds[ds["quarter"] == q].copy() for q in quarters}
    vol_by_sym_q = ds.set_index(["quarter", "symbol"])["daily_vol"].to_dict() if "daily_vol" in ds.columns else {}

    monthly_stats: list[dict] = []
    holdings: dict[str, dict] = {}
    daily_nav_rows: list[dict] = []

    nav = 1.0
    bench_nav = 1.0

    for me in sorted(month_ends):
        sig_q = _signal_quarter_for_month(me, q_ends)
        if sig_q is None or sig_q not in allowed_signal_quarters:
            continue
        if sig_q not in rows_by_q:
            continue

        mo = str(me.to_period("M"))
        if mo not in month_to_idx or month_to_idx[mo] >= len(months_sorted) - 1:
            continue
        next_mo = months_sorted[month_to_idx[mo] + 1]
        realized = str(pd.Period(next_mo, freq="M").strftime("%Y-%m"))

        current = rows_by_q[sig_q].copy()
        scores = score_fn(sig_q, current)
        if scores is None or scores.empty:
            continue
        current = current.set_index("symbol")
        current["score"] = current.index.map(scores)
        current = current.dropna(subset=["score"]).reset_index()
        if len(current) < 1:
            continue

        top = current.nlargest(TOP_N_HOLDINGS, "score").reset_index(drop=True)
        syms = top["symbol"].tolist()
        vols = pd.Series({s: vol_by_sym_q.get((sig_q, s), VOL_FLOOR_DAILY) for s in syms})
        weights = portfolio_weights(
            top, vols, weight_scheme, softmax_temperature=softmax_temperature
        )

        next_slice = mret[mret["month"] == next_mo]
        sel = next_slice[next_slice["symbol"].isin(syms)]
        if sel.empty:
            continue

        rets = sel.set_index("symbol")["return_capped"]
        port_ret = sum(weights.get(s, 0.0) * float(rets.get(s, 0.0)) for s in syms if s in rets.index)
        if next_mo not in bench_m.index:
            continue
        bench_ret = float(np.clip(float(bench_m.loc[next_mo]), RET_CLIP_LOWER, RET_CLIP_UPPER))

        # Monthly portfolio vol: weighted std proxy (diagonal, no correlation)
        port_vol = float(
            np.sqrt(sum((weights.get(s, 0.0) ** 2) * (float(vols.get(s, VOL_FLOOR_DAILY)) ** 2) for s in syms))
        )

        nav *= 1 + port_ret
        bench_nav *= 1 + bench_ret

        actual30 = mlabels[(mlabels["month"] == next_mo) & (mlabels["is_top30"])].sort_values(
            "excess_return", ascending=False
        )

        holdings[realized] = {
            "signal_quarter": sig_q,
            "signal_month": mo,
            "realized_month": realized,
            "path": path_name,
            "selected": [
                {
                    "symbol": s,
                    "weight": round(weights.get(s, 0.0), 6),
                    "score": float(top.loc[top["symbol"] == s, "score"].iloc[0]),
                    "daily_vol": float(vols.get(s, VOL_FLOOR_DAILY)),
                }
                for s in syms
            ],
            "actual_top30": [
                {
                    "symbol": r["symbol"],
                    "excess_return": float(r["excess_return"]),
                    "return": float(r["return"]),
                    "y_label": 1,
                }
                for _, r in actual30.iterrows()
            ],
        }

        monthly_stats.append(
            {
                "month": realized,
                "signal_quarter": sig_q,
                "path": path_name,
                "portfolio_return": port_ret,
                "benchmark_return": bench_ret,
                "excess_return": port_ret - bench_ret,
                "portfolio_vol": port_vol,
                "strategy_nav": nav,
                "benchmark_nav": bench_nav,
                "excess_nav": nav / bench_nav - 1.0 if bench_nav > 0 else float("nan"),
                "n_selected": len(syms),
            }
        )

    daily_df = pd.DataFrame(daily_nav_rows)
    summary = _summarize_path(monthly_stats, daily_df)
    return monthly_stats, holdings, summary, daily_df


def _summarize_path(monthly_stats: list[dict], daily_df: pd.DataFrame) -> dict:
    if not monthly_stats:
        return {"n_months": 0}
    ms = pd.DataFrame(monthly_stats)
    total_s = float(ms["strategy_nav"].iloc[-1] - 1)
    total_b = float(ms["benchmark_nav"].iloc[-1] - 1)
    n = len(ms)
    years = n / 12
    ann_s = ms["strategy_nav"].iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")
    ann_b = ms["benchmark_nav"].iloc[-1] ** (1 / years) - 1 if years > 0 else float("nan")

    strat_rets = ms["portfolio_return"].astype(float).tolist()
    bench_rets = ms["benchmark_return"].astype(float).tolist()
    excess_rets = ms["excess_return"].astype(float).tolist()
    strat_vol = float(np.std(strat_rets, ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    bench_vol = float(np.std(bench_rets, ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    excess_vol = float(np.std(excess_rets, ddof=1) * np.sqrt(12)) if n > 1 else float("nan")
    strat_sharpe = float((np.mean(strat_rets) * 12) / strat_vol) if strat_vol > 0 else float("nan")
    bench_sharpe = float((np.mean(bench_rets) * 12) / bench_vol) if bench_vol > 0 else float("nan")
    excess_sharpe = float((np.mean(excess_rets) * 12) / excess_vol) if excess_vol > 0 else float("nan")

    mdd = float("nan")
    max_day_up = float("nan")
    if len(daily_df) and "strategy_nav" in daily_df.columns:
        nav = daily_df["strategy_nav"].astype(float)
        peak = nav.cummax()
        dd = nav / peak - 1.0
        mdd = float(dd.min())
        if "daily_return" in daily_df.columns:
            max_day_up = float(daily_df["daily_return"].max())

    return {
        "n_months": n,
        "total_return_strategy": total_s,
        "total_return_benchmark": total_b,
        "total_excess_arithmetic": total_s - total_b,
        "total_excess_geometric": float(ms["excess_nav"].iloc[-1]) if len(ms) else float("nan"),
        "annualized_strategy": float(ann_s),
        "annualized_benchmark": float(ann_b),
        "annualized_excess": float(ann_s - ann_b),
        "ann_vol_strategy": strat_vol,
        "ann_vol_benchmark": bench_vol,
        "ann_vol_excess": excess_vol,
        "ann_sharpe_strategy": strat_sharpe,
        "ann_sharpe_benchmark": bench_sharpe,
        "ann_sharpe_excess": excess_sharpe,
        "mean_monthly_portfolio_vol": float(ms["portfolio_vol"].mean()),
        "max_drawdown": mdd,
        "max_single_day_gain": max_day_up,
    }


def _mdd_with_dates(daily: pd.DataFrame) -> dict:
    """Max drawdown defined as the worst single-day return, plus its date.

    Also keeps the peak-to-trough (cumulative) drawdown for reference.
    """
    df = daily.sort_values("Date").reset_index(drop=True)
    dr = df["daily_return"].astype(float)
    dates = pd.to_datetime(df["Date"])
    worst_i = int(dr.idxmin())
    # peak-to-trough (cumulative) kept for reference
    nav = df["strategy_nav"].astype(float)
    dd = nav / nav.cummax() - 1.0
    return {
        "max_drawdown": float(dr.min()),
        "max_drawdown_date": str(dates.iloc[worst_i].date()),
        "max_drawdown_peak_to_trough": float(dd.min()),
        "max_single_day_gain": float(dr.max()),
    }


def _build_daily_series(daily_df: pd.DataFrame, spy_ret: pd.Series) -> list[dict]:
    """Daily strategy + benchmark NAV (both rebased to 1.0 at segment start)."""
    if daily_df is None or daily_df.empty:
        return []
    df = daily_df.sort_values("Date").reset_index(drop=True)
    bnav = 1.0
    out: list[dict] = []
    for _, r in df.iterrows():
        d = pd.Timestamp(r["Date"])
        rb = spy_ret.get(d, 0.0)
        rb = 0.0 if rb is None or (isinstance(rb, float) and np.isnan(rb)) else float(rb)
        bnav *= 1 + rb
        out.append(
            {
                "date": str(d.date()),
                "strategy_nav": float(r["strategy_nav"]),
                "benchmark_nav": bnav,
                "daily_return": float(r.get("daily_return", 0.0)),
            }
        )
    return out


def _build_daily_nav(
    holdings: dict[str, dict],
    prices: pd.DataFrame,
    path_name: str,
) -> pd.DataFrame:
    """Approximate daily NAV while holding fixed weights within each realized month."""
    rows: list[dict] = []
    nav = 1.0
    for realized_month, h in sorted(holdings.items(), key=lambda x: x[0]):
        if h.get("path") != path_name:
            continue
        weights = {x["symbol"]: x["weight"] for x in h["selected"]}
        if not weights:
            continue
        mo = realized_month
        grp_all = prices[prices["month"] == mo].copy()
        if grp_all.empty:
            continue
        dates = sorted(grp_all["Date"].unique())
        for d in dates:
            day = grp_all[grp_all["Date"] == d]
            day_rets = []
            for sym, w in weights.items():
                r = day.loc[day["symbol"] == sym, "daily_ret"]
                if len(r):
                    day_rets.append(w * float(r.iloc[0]))
            if not day_rets:
                continue
            dr = float(np.sum(day_rets))
            nav *= 1 + dr
            rows.append({"Date": d, "month": mo, "daily_return": dr, "strategy_nav": nav, "path": path_name})
    return pd.DataFrame(rows)


def run_monthly_backtest(force: bool = False) -> dict:
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BACKTEST_DIR / "backtest_monthly.json"
    holdings_path = BACKTEST_DIR / "holdings_by_month.json"
    holdings_q_path = BACKTEST_DIR / "holdings_by_signal_quarter.json"
    report_path = BACKTEST_DIR / "performance_report.json"

    if out_path.exists() and holdings_path.exists() and holdings_q_path.exists() and report_path.exists() and not force:
        return {
            "monthly": json.loads(out_path.read_text(encoding="utf-8")),
            "holdings": json.loads(holdings_path.read_text(encoding="utf-8")),
            "report": json.loads(report_path.read_text(encoding="utf-8")),
        }

    meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
    feature_cols: list[str] = meta["feature_cols"]
    wf_meta = meta.get("walk_forward") or {}
    frozen = wf_meta.get("frozen") or {}
    weight_scheme = frozen.get("weight_scheme") or frozen.get("selected_weight_scheme") or WEIGHT_SCHEME
    temp_raw = frozen.get("frozen_softmax_temperature")
    softmax_temperature = float(temp_raw) if temp_raw is not None else 0.1
    weight_label = frozen.get("weight_spec_label")
    if not weight_label:
        if weight_scheme == "softmax_sharpe" and temp_raw is not None:
            weight_label = f"softmax(score/(T*σ)), T={softmax_temperature}"
        elif weight_scheme == "softmax" and temp_raw is not None:
            weight_label = f"softmax(score/T), T={softmax_temperature}"
        else:
            weight_label = weight_scheme
    wf_quarters = set(wf_meta.get("predicted_quarters") or meta.get("cv", {}).get("oof_quarters") or [])
    cv_q = set(wf_meta.get("cv_oof_quarters") or wf_meta.get("early_quarters") or [])
    oos_q = set(wf_meta.get("final_oos_quarters") or wf_meta.get("recent_quarters") or [])

    ds = pd.read_parquet(MODEL_DIR / "model_dataset.parquet")
    quarters = sorted(ds["quarter"].unique())
    q_ends = ds.groupby("quarter")["quarter_end"].min()
    q_ends = pd.to_datetime(q_ends)

    sp500 = load_sp500_symbols()
    prices = pd.read_parquet(DATA_CURATED_DIR / "prices_daily.parquet")
    prices = prices[prices["symbol"].isin(sp500)].copy()
    prices["Date"] = pd.to_datetime(prices["Date"])
    price_col = "adjusted_close" if "adjusted_close" in prices.columns else "close"
    prices["price"] = prices[price_col]
    prices["daily_ret"] = prices.groupby("symbol")["price"].pct_change()
    prices["month"] = prices["Date"].dt.to_period("M").astype(str)

    mret = _monthly_stock_returns(prices)
    bench_m = spy_monthly_returns()
    mlabels = pd.read_parquet(LABELS_DIR / "labels_monthly.parquet")
    qlabels = pd.read_parquet(LABELS_DIR / "labels_quarterly.parquet")

    wf_path = MODEL_DIR / "walk_forward_scores.parquet"
    if not wf_path.exists():
        wf_path = MODEL_DIR / "cv_oof_scores.parquet"
    wf = pd.read_parquet(wf_path) if wf_path.exists() else None
    wf_by_q: dict[str, pd.Series] = {}
    if wf is not None and len(wf):
        score_col = "score_wf" if "score_wf" in wf.columns else "score_oof"
        wf_by_q = {q: g.set_index("symbol")[score_col] for q, g in wf.groupby("quarter")}

    def score_wf(q: str, _df: pd.DataFrame) -> pd.Series | None:
        if q not in wf_by_q:
            return None
        return wf_by_q[q]

    month_ends = sorted(mret.groupby("month")["month_end"].max().tolist())
    month_ends = [pd.Timestamp(x) for x in month_ends]

    is_stats, is_hold_items, is_sum, _ = _run_monthly_path(
        month_ends,
        mret,
        bench_m,
        mlabels,
        ds,
        quarters,
        q_ends,
        feature_cols,
        score_wf,
        cv_q if cv_q else wf_quarters - oos_q,
        "walk_forward_cv",
        weight_scheme=weight_scheme,
        softmax_temperature=softmax_temperature,
    )
    oos_stats, oos_hold_items, oos_sum, _ = _run_monthly_path(
        month_ends,
        mret,
        bench_m,
        mlabels,
        ds,
        quarters,
        q_ends,
        feature_cols,
        score_wf,
        oos_q,
        "walk_forward_oos",
        weight_scheme=weight_scheme,
        softmax_temperature=softmax_temperature,
    )

    holdings_all = {**is_hold_items, **oos_hold_items}
    holdings_quarterly = _build_quarterly_holdings(holdings_all, qlabels, quarters)
    daily_is = _build_daily_nav(holdings_all, prices, "walk_forward_cv")
    daily_oos = _build_daily_nav(holdings_all, prices, "walk_forward_oos")
    if len(daily_is):
        is_sum.update(_mdd_with_dates(daily_is))
    if len(daily_oos):
        oos_sum.update(_mdd_with_dates(daily_oos))

    report = {
        "benchmark": BENCHMARK_ID,
        "weighting": weight_label,
        "weight_scheme": weight_scheme,
        "softmax_temperature": softmax_temperature,
        "top_n_holdings": TOP_N_HOLDINGS,
        "label_top_n": TOP_N_LABEL,
        "in_sample": is_sum,
        "out_of_sample": oos_sum,
        "monthly_in_sample": is_stats,
        "monthly_out_of_sample": oos_stats,
    }

    out = {
        "monthly_in_sample": is_stats,
        "monthly_out_of_sample": oos_stats,
        "meta": {
            "benchmark": BENCHMARK_ID,
            "weighting": weight_label,
            "weight_scheme": weight_scheme,
            "softmax_temperature": softmax_temperature,
            "top_n": TOP_N_HOLDINGS,
        },
    }
    spy_ret = spy_daily_returns()
    daily_out = {
        "daily_in_sample": _build_daily_series(daily_is, spy_ret),
        "daily_out_of_sample": _build_daily_series(daily_oos, spy_ret),
    }
    (BACKTEST_DIR / "backtest_daily.json").write_text(json.dumps(daily_out, indent=2), encoding="utf-8")

    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    holdings_path.write_text(json.dumps(holdings_all, indent=2), encoding="utf-8")
    holdings_q_path.write_text(json.dumps(holdings_quarterly, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = _render_performance_md(report)
    (BACKTEST_DIR / "PERFORMANCE_REPORT.md").write_text(md, encoding="utf-8")

    # Keep quarterly curve for legacy frontend (aggregate monthly to last point per quarter)
    _write_legacy_quarterly_curve(is_stats, oos_stats)

    return {"monthly": out, "holdings": holdings_all, "report": report}


def _write_legacy_quarterly_curve(is_stats: list[dict], oos_stats: list[dict]) -> None:
    def to_curve(stats: list[dict]) -> list[dict]:
        if not stats:
            return []
        df = pd.DataFrame(stats)
        df["period"] = df["month"].apply(lambda m: str(pd.Period(m, freq="M").asfreq("Q")))
        rows = []
        for q, g in df.groupby("period"):
            last = g.iloc[-1]
            rows.append(
                {
                    "date": str(pd.Period(q, freq="Q").end_time.date()),
                    "quarter": q,
                    "strategy": float(last["strategy_nav"]),
                    "benchmark": float(last["benchmark_nav"]),
                }
            )
        return rows

    curve = {
        "quarterly": to_curve(oos_stats),
        "quarterly_is_oof": to_curve(is_stats),
        "daily": [],
        "meta": {"source": "monthly_backtest_aggregated"},
    }
    (BACKTEST_DIR / "backtest_curve.json").write_text(json.dumps(curve, indent=2), encoding="utf-8")


def _render_performance_md(report: dict) -> str:
    def pct(x: float) -> str:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NA"
        return f"{x * 100:.2f}%"

    weight_desc = report.get("weighting") or "inverse_daily_vol"
    lines = [
        "# 投組績效報告（月度調倉）\n",
        f"- **標籤**：當季/當月超額報酬排名前 **{report['label_top_n']}** → `y=1`\n",
        f"- **持股**：模型分數 Top **{report['top_n_holdings']}**；權重 **{weight_desc}**（normalize，sum=1）\n",
        f"- **基準**：SPY（S&P 500 市值加權）\n",
        "## CV OOF（expanding walk-forward · post-TTM-warmup）\n",
        _section_md(report["in_sample"], pct),
        "\n## Final OOS（凍結 hyperparam · Q17–Q20）\n",
        _section_md(report["out_of_sample"], pct),
        "\n## 月度明細（Final OOS）\n",
        "\n| 月 | 組合報酬 | 基準 | 超額 | 組合波動(月) |\n|---|---:|---:|---:|---:|\n",
    ]
    for r in report.get("monthly_out_of_sample") or []:
        lines.append(
            f"| {r['month']} | {r['portfolio_return']*100:.2f}% | {r['benchmark_return']*100:.2f}% | "
            f"{r['excess_return']*100:.2f}% | {r['portfolio_vol']*100:.2f}% |\n"
        )
    lines.append(
        "\n持股對照：`holdings_by_signal_quarter.json`（信號季 → **下一季** Top30，對齊 `y_next`）；"
        "`holdings_by_month.json` 為月度實現明細。\n"
    )
    return "".join(lines)


def _section_md(s: dict, pct) -> str:
    if not s.get("n_months"):
        return "_無資料_\n"
    return (
        f"- 月數：{s['n_months']}\n"
        f"- 累積報酬（策略 / 基準）：{pct(s.get('total_return_strategy'))} / {pct(s.get('total_return_benchmark'))}\n"
        f"- 累積超額（幾何）：{pct(s.get('total_excess_geometric'))}\n"
        f"- 年化超額（算術差）：{pct(s.get('annualized_excess'))}\n"
        f"- **Sharpe（年化）**：策略 {s.get('ann_sharpe_strategy', float('nan')):.2f} · "
        f"基準 {s.get('ann_sharpe_benchmark', float('nan')):.2f} · "
        f"超額 {s.get('ann_sharpe_excess', float('nan')):.2f}\n"
        f"- 年化波動：策略 {pct(s.get('ann_vol_strategy'))} · 基準 {pct(s.get('ann_vol_benchmark'))}\n"
        f"- 平均月度組合波動：{pct(s.get('mean_monthly_portfolio_vol'))}\n"
        f"- 最大回撤：{pct(s.get('max_drawdown'))}\n"
        f"- 單日最大漲幅：{pct(s.get('max_single_day_gain'))}\n"
    )
