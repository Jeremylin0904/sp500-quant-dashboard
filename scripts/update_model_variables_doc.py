from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Paths:
    root: Path
    model_meta: Path
    feature_stats: Path
    model_variables_md: Path


def _repo_paths() -> Paths:
    root = Path(__file__).resolve().parents[1]
    return Paths(
        root=root,
        model_meta=root / "quant" / "model" / "model_meta.json",
        feature_stats=root / "quant" / "model" / "feature_stats.csv",
        model_variables_md=root / "quant" / "model" / "MODEL_VARIABLES.md",
    )


def _read_feature_cols(model_meta_path: Path) -> list[str]:
    meta = json.loads(model_meta_path.read_text(encoding="utf-8"))
    cols = meta.get("feature_cols")
    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError("model_meta.json missing valid feature_cols list")
    return cols


def _format_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.2f}%"


def _format_inf_pct(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.4f}%"


def build_top_missing_table(
    feature_stats_path: Path, feature_cols: list[str], top_n: int = 15
) -> str:
    df = pd.read_csv(feature_stats_path)
    feature_col_name = "feature" if "feature" in df.columns else ("col" if "col" in df.columns else None)
    if feature_col_name is None:
        raise ValueError("feature_stats.csv missing 'feature' (or legacy 'col') column")

    expected = {"miss_pct", "inf_pct"}
    missing_cols = expected - set(df.columns)
    if missing_cols:
        raise ValueError(f"feature_stats.csv missing columns: {sorted(missing_cols)}")

    df = df[df[feature_col_name].isin(feature_cols)].copy()
    df = df.sort_values(["miss_pct", "inf_pct", feature_col_name], ascending=[False, False, True]).head(top_n)

    lines = []
    lines.append("| feature | miss_pct | inf_pct |")
    lines.append("|---|---:|---:|")
    for _, r in df.iterrows():
        feat = str(r[feature_col_name])
        miss = _format_pct(r["miss_pct"])
        infp = _format_inf_pct(r["inf_pct"])
        lines.append(f"| `{feat}` | {miss} | {infp} |")
    return "\n".join(lines) + "\n"


def _replace_block(text: str, start_marker: str, end_marker: str, new_block: str) -> str:
    pattern = re.compile(
        rf"({re.escape(start_marker)}\n)(.*?)(\n{re.escape(end_marker)})",
        flags=re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Cannot find auto block markers: {start_marker} ... {end_marker}")
    return text[: m.start(2)] + new_block.rstrip("\n") + text[m.end(2) :]


def update_model_variables_md(paths: Paths, top_n: int = 15) -> None:
    feature_cols = _read_feature_cols(paths.model_meta)
    table = build_top_missing_table(paths.feature_stats, feature_cols, top_n=top_n)

    md = paths.model_variables_md.read_text(encoding="utf-8")
    md = _replace_block(
        md,
        "<!-- AUTO:TOP_MISSING_START -->",
        "<!-- AUTO:TOP_MISSING_END -->",
        table,
    )

    today = date.today().isoformat()
    updated_line = (
        f"*最後更新：{today}（SP500-only universe；label 在 SP500 內排名；AutoML + NaN 不補值；"
        "OOS 評估報告已產生）*\n"
    )
    md = _replace_block(
        md,
        "<!-- AUTO:LAST_UPDATED_START -->",
        "<!-- AUTO:LAST_UPDATED_END -->",
        updated_line,
    )

    paths.model_variables_md.write_text(md, encoding="utf-8")


def main() -> None:
    paths = _repo_paths()
    update_model_variables_md(paths, top_n=15)
    print("OK: updated quant/model/MODEL_VARIABLES.md")


if __name__ == "__main__":
    main()

