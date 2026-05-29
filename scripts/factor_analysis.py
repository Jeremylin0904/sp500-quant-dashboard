#!/usr/bin/env python3
"""Fama-French 5-factor + Momentum regression on strategy monthly returns.

Factor sourcing (all from Kenneth R. French Data Library, monthly, USD):
  - 5 factors  : F-F_Research_Data_5_Factors_2x3   (Mkt-RF, SMB, HML, RMW, CMA, RF)
  - Momentum   : F-F_Momentum_Factor               (Mom)
Both are downloaded as the official CSV zips and cached under quant/data/french/.

Factor construction (per Fama-French / French data library):
  - Mkt-RF : value-weighted return of all CRSP US equities minus 1-month T-bill (RF).
  - SMB    : Small-Minus-Big — avg return of small-cap minus big-cap portfolios (size).
  - HML    : High-Minus-Low — high book-to-market (value) minus low (growth).
  - RMW    : Robust-Minus-Weak — high minus low operating profitability.
  - CMA    : Conservative-Minus-Aggressive — low minus high asset growth (investment).
  - Mom    : winners-minus-losers on prior 2-12 month returns (momentum / UMD).
  - RF     : 1-month US Treasury bill rate.
All factor values are monthly percentages in the source files (divided by 100 here).

Regression (per segment): (R_strategy - RF) = alpha + b1*MktRF + b2*SMB + b3*HML
                                              + b4*RMW + b5*CMA + b6*Mom + e
alpha = factor-adjusted excess return (annualized = (1+alpha_m)^12 - 1 approx, reported both).
"""

from __future__ import annotations

import io
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from quant.config import BACKTEST_DIR, MODEL_DIR  # noqa: E402

FRENCH_DIR = ROOT / "quant" / "data" / "french"
FACTOR_DIR = ROOT / "quant" / "factor"

FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
MOM_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def _download_zip_csv(url: str, cache_path: Path) -> str:
    """Download Ken French CSV zip, cache raw CSV text locally."""
    if cache_path.exists():
        return cache_path.read_text(encoding="latin-1")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60, context=ctx).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode("latin-1")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="latin-1")
    return text


def _parse_monthly_block(text: str, value_cols: list[str]) -> pd.DataFrame:
    """Parse Ken French CSV: keep rows whose first token is YYYYMM (6 digits)."""
    rows: list[dict] = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        key = parts[0]
        if len(key) == 6 and key.isdigit():
            try:
                vals = [float(p) for p in parts[1 : 1 + len(value_cols)]]
            except ValueError:
                continue
            if len(vals) != len(value_cols):
                continue
            rec = {"yyyymm": key}
            rec.update(dict(zip(value_cols, vals)))
            rows.append(rec)
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No monthly rows parsed from Ken French CSV")
    for c in value_cols:
        df[c] = df[c] / 100.0  # percent -> decimal
    df["month"] = df["yyyymm"].str[:4] + "-" + df["yyyymm"].str[4:6]
    return df.set_index("month")[value_cols]


def load_factors() -> pd.DataFrame:
    ff5_text = _download_zip_csv(FF5_URL, FRENCH_DIR / "F-F_Research_Data_5_Factors_2x3.csv")
    mom_text = _download_zip_csv(MOM_URL, FRENCH_DIR / "F-F_Momentum_Factor.csv")
    ff5 = _parse_monthly_block(ff5_text, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    mom = _parse_monthly_block(mom_text, ["Mom"])
    fac = ff5.join(mom, how="inner")
    return fac


def _strategy_monthly() -> dict[str, pd.DataFrame]:
    data = json.loads((BACKTEST_DIR / "backtest_monthly.json").read_text(encoding="utf-8"))

    def to_df(rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["month", "strat", "bench"]).set_index("month")
        df = pd.DataFrame(rows)[["month", "portfolio_return", "benchmark_return"]]
        df = df.rename(columns={"portfolio_return": "strat", "benchmark_return": "bench"})
        return df.set_index("month")

    is_df = to_df(data.get("monthly_in_sample") or [])
    oos_df = to_df(data.get("monthly_out_of_sample") or [])
    all_df = pd.concat([is_df, oos_df]).sort_index()
    all_df = all_df[~all_df.index.duplicated(keep="first")]
    return {"cv_oof": is_df, "final_oos": oos_df, "all": all_df}


def _regress(strat: pd.DataFrame, fac: pd.DataFrame) -> dict | None:
    df = strat.join(fac, how="inner").dropna()
    if len(df) < len(FACTORS) + 2:
        # Not enough points for a stable 6-factor fit; fall back to report n only.
        if df.empty:
            return None
    y = (df["strat"] - df["RF"]).to_numpy()
    X = df[FACTORS].to_numpy()
    Xc = sm.add_constant(X)
    model = sm.OLS(y, Xc).fit()

    names = ["alpha"] + FACTORS
    coefs = model.params
    tvals = model.tvalues
    pvals = model.pvalues
    se = model.bse

    alpha_m = float(coefs[0])
    factor_rows = []
    for i, nm in enumerate(FACTORS, start=1):
        factor_rows.append(
            {
                "factor": nm,
                "beta": float(coefs[i]),
                "t_stat": float(tvals[i]),
                "p_value": float(pvals[i]),
                "std_err": float(se[i]),
            }
        )
    # Mean factor premia over the window (annualized arithmetic) for context.
    contrib = {nm: float(df[nm].mean() * 12) for nm in FACTORS}

    return {
        "n_months": int(len(df)),
        "months": list(df.index),
        "alpha_monthly": alpha_m,
        "alpha_annualized_compound": float((1 + alpha_m) ** 12 - 1),
        "alpha_annualized_arith": float(alpha_m * 12),
        "alpha_t_stat": float(tvals[0]),
        "alpha_p_value": float(pvals[0]),
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "factors": factor_rows,
        "mean_factor_premium_annualized": contrib,
        "mean_excess_return_monthly": float(y.mean()),
    }


def main() -> None:
    FACTOR_DIR.mkdir(parents=True, exist_ok=True)
    fac = load_factors()
    strat = _strategy_monthly()

    segments = {}
    for key, label in (("cv_oof", "CV OOF (in-sample)"), ("final_oos", "Final OOS"), ("all", "全期間")):
        res = _regress(strat[key], fac)
        if res is not None:
            res["segment_label"] = label
        segments[key] = res

    factor_window = {
        "start": min(fac.index),
        "end": max(fac.index),
    }
    out = {
        "model": "Fama-French 5-Factor + Momentum (UMD)",
        "source": "Kenneth R. French Data Library (monthly, USD)",
        "factor_definitions": {
            "Mkt-RF": "市場超額：全美股市值加權報酬 − 一個月國庫券(RF)",
            "SMB": "Small-Minus-Big：小型股 − 大型股（規模）",
            "HML": "High-Minus-Low：高淨值市價比(價值) − 低(成長)",
            "RMW": "Robust-Minus-Weak：高 − 低營業獲利能力",
            "CMA": "Conservative-Minus-Aggressive：低 − 高資產成長（投資）",
            "Mom": "Momentum/UMD：前 2–12 月贏家 − 輸家",
            "RF": "一個月美國國庫券利率",
        },
        "regression_spec": "(R_strategy - RF) = alpha + b·[Mkt-RF, SMB, HML, RMW, CMA, Mom] + e",
        "factor_data_window": factor_window,
        "segments": segments,
    }

    (FACTOR_DIR / "factor_analysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_md(out)
    print("Wrote factor_analysis.json + FACTOR_ANALYSIS.md")
    for k in ("cv_oof", "final_oos", "all"):
        r = segments.get(k)
        if r:
            print(
                f"{k:10s} n={r['n_months']:2d} alpha(ann)={r['alpha_annualized_compound']*100:6.2f}% "
                f"t={r['alpha_t_stat']:.2f} R2={r['r_squared']:.2f}"
            )


def _write_md(out: dict) -> None:
    lines = [
        "# 因子分析：Fama-French 五因子 + 動量\n",
        f"- **資料來源**：{out['source']}\n",
        "- **五因子**：`F-F_Research_Data_5_Factors_2x3`（Mkt-RF, SMB, HML, RMW, CMA, RF）\n",
        "- **動量**：`F-F_Momentum_Factor`（Mom / UMD）\n",
        f"- **因子資料區間**：{out['factor_data_window']['start']} ~ {out['factor_data_window']['end']}\n",
        f"- **回歸式**：`{out['regression_spec']}`\n\n",
        "## 因子定義與來源\n\n",
        "| 因子 | 定義 |\n|---|---|\n",
    ]
    for k, v in out["factor_definitions"].items():
        lines.append(f"| `{k}` | {v} |\n")
    lines.append("\n> 因子皆為 Ken French 官方月頻資料（百分比，已轉小數）。市場/規模/價值來自三因子；")
    lines.append("RMW、CMA 為五因子新增之獲利能力與投資因子；動量(Mom)為另一獨立檔案。\n\n")

    for key in ("cv_oof", "final_oos", "all"):
        r = out["segments"].get(key)
        if not r:
            continue
        lines.append(f"## {r.get('segment_label', key)}（n={r['n_months']} 月）\n\n")
        lines.append(
            f"- **Alpha（年化）**：{r['alpha_annualized_compound']*100:.2f}%　"
            f"(月 {r['alpha_monthly']*100:.3f}%, t={r['alpha_t_stat']:.2f}, p={r['alpha_p_value']:.3f})\n"
        )
        lines.append(f"- **R² / adj-R²**：{r['r_squared']:.3f} / {r['adj_r_squared']:.3f}\n\n")
        lines.append("| 因子 | beta | t 值 | p 值 |\n|---|---:|---:|---:|\n")
        for f in r["factors"]:
            lines.append(
                f"| {f['factor']} | {f['beta']:.3f} | {f['t_stat']:.2f} | {f['p_value']:.3f} |\n"
            )
        lines.append("\n")

    (FACTOR_DIR / "FACTOR_ANALYSIS.md").write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
