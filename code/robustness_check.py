"""
Outlier-robustness check for the SIR pipeline.

Given one or more per-row SIR result CSVs (produced by sir_pipeline.py),
computes Pearson R and MAE between leader_hours and ai_estimated_hours
under three settings:

    (1) full_sample                    : no trimming, no winsorization.
    (2) trim_top1pct_by_SIR            : drop the top 1% of rows by
                                         inflation_rate, then recompute.
    (3) winsor_99pct_both              : cap ai_estimated_hours and
                                         leader_hours each at their
                                         99th percentile, then recompute.
    (4) trim_top1pct_SIR + winsor_99pct: (2) and (3) combined.

When several model result files are supplied (e.g., Q2 / Q4 / Q8 variants
of the same base model), the script also reports whether the ordering
across models is preserved under each setting.

This script does NOT call the LLM and does NOT need a local inference
server; it only consumes CSV outputs.
"""

import argparse
import sys
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


REQUIRED_COLUMNS = ("leader_hours", "ai_estimated_hours")
SIR_COLUMN       = "inflation_rate"


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    if np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def summarize(label: str, leader: Iterable[float],
              ai: Iterable[float]) -> dict:
    leader = np.asarray(list(leader), dtype=float)
    ai     = np.asarray(list(ai),     dtype=float)
    mask = np.isfinite(leader) & np.isfinite(ai)
    leader, ai = leader[mask], ai[mask]
    r   = safe_corr(leader, ai)
    mae = mean_absolute_error(leader, ai) if len(leader) else float("nan")
    return {"setting": label, "N": int(len(leader)), "R": r, "MAE": mae}


def check_file(path: str) -> Tuple[str, pd.DataFrame]:
    print(f"\n=== {path} ===")
    df = pd.read_csv(path, encoding_errors="ignore")
    df = df.drop_duplicates().copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    df["leader_hours"]       = pd.to_numeric(df["leader_hours"],       errors="coerce")
    df["ai_estimated_hours"] = pd.to_numeric(df["ai_estimated_hours"], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    df = df[df["leader_hours"] > 0]

    rows = []
    rows.append(summarize("full_sample", df["leader_hours"], df["ai_estimated_hours"]))

    if SIR_COLUMN in df.columns:
        sir = pd.to_numeric(df[SIR_COLUMN], errors="coerce")
        keep = sir <= sir.quantile(0.99)
        rows.append(summarize("trim_top1pct_by_SIR",
                              df.loc[keep, "leader_hours"],
                              df.loc[keep, "ai_estimated_hours"]))
    else:
        rows.append({"setting": "trim_top1pct_by_SIR",
                     "N": np.nan, "R": np.nan, "MAE": np.nan})
        print(f"[warn] column '{SIR_COLUMN}' not found; skipping trimming.",
              file=sys.stderr)

    leader_99 = df["leader_hours"].quantile(0.99)
    ai_99     = df["ai_estimated_hours"].quantile(0.99)
    rows.append(summarize("winsor_99pct_both",
                          df["leader_hours"].clip(upper=leader_99),
                          df["ai_estimated_hours"].clip(upper=ai_99)))

    if SIR_COLUMN in df.columns:
        sir = pd.to_numeric(df[SIR_COLUMN], errors="coerce")
        keep = sir <= sir.quantile(0.99)
        rows.append(summarize("trim_top1pct_SIR + winsor_99pct",
                              df.loc[keep, "leader_hours"].clip(upper=leader_99),
                              df.loc[keep, "ai_estimated_hours"].clip(upper=ai_99)))
    else:
        rows.append({"setting": "trim_top1pct_SIR + winsor_99pct",
                     "N": np.nan, "R": np.nan, "MAE": np.nan})

    out = pd.DataFrame(rows)
    print(out.to_string(index=False, formatters={
        "R":   lambda v: f"{v:.4f}" if pd.notna(v) else "nan",
        "MAE": lambda v: f"{v:.4f}" if pd.notna(v) else "nan",
    }))
    return (path, out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", required=True,
                   help="One or more per-row result CSVs produced by "
                        "sir_pipeline.py. Repeat for each model configuration "
                        "you want to compare.")
    args = p.parse_args()

    all_results = [check_file(path) for path in args.results]
    if len(all_results) < 2:
        return

    # Cross-model summary table
    pivot = {}
    for name, df in all_results:
        pivot[name] = {row["setting"]: row["R"] for _, row in df.iterrows()}
    pivot_df = pd.DataFrame(pivot).T

    print("\n" + "=" * 70)
    print("SUMMARY: Pearson R across models under each robustness setting")
    print("=" * 70)
    print(pivot_df.to_string(float_format=lambda v: f"{v:.4f}"
                             if pd.notna(v) else "nan"))

    print("\n" + "=" * 70)
    print("Max |delta R| vs full_sample for each model")
    print("=" * 70)
    for name, df in all_results:
        full = df.loc[df["setting"] == "full_sample", "R"].iloc[0]
        max_delta = (df["R"] - full).abs().max()
        print(f"  {name}: max |delta R| = {max_delta:.4f}")

    # Ordering check: does model order stay the same under each setting?
    print("\n" + "=" * 70)
    print("Ordering check: is the R-based ranking of models preserved")
    print("across all settings? (compared against full_sample order)")
    print("=" * 70)
    full_order = pivot_df["full_sample"].sort_values(ascending=False).index.tolist()
    for setting in pivot_df.columns:
        order = pivot_df[setting].sort_values(ascending=False).index.tolist()
        preserved = (order == full_order)
        print(f"  {setting:40s}  preserved={preserved}")


if __name__ == "__main__":
    main()
