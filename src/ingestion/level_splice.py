"""
level_splice.py — corrects the Nov 2019 RBI reporting definition change.

The RBI changed how it counts cards outstanding in Nov 2019 (the
"non-financial transactions reclassification" called out in the project
brief). This is a *measurement* change, not a market event — modelling it
as a structural break attributes a definitional jump to a structural cause
and contaminates coefficient estimates.

The fix: estimate the level shift at the break using a local linear
regression each side (regression-discontinuity style), then additively
correct all pre-break observations so the series is commensurate with the
post-Nov-2019 measurement convention.

Limitation: assumes the change is a level shift, not a slope shift. The
script reports both pre and post slopes so you can verify visually that
this assumption is reasonable before relying on the spliced series.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# RBI reporting definition change.
BREAK_DATE = pd.Timestamp("2019-11-01")

# Local windows. Pre stops at break; post stops before COVID (Mar 2020).
# Using 4 months each side gives a symmetric, COVID-free comparison —
# extend pre only if the series looks too noisy on a 4-month fit.
PRE_WINDOW_MONTHS = 4
POST_WINDOW_MONTHS = 4


@dataclass
class SpliceResult:
    series_name: str
    break_date: pd.Timestamp
    pre_intercept: float
    pre_slope: float           # units of series / month
    post_intercept: float
    post_slope: float
    pre_extrapolated_at_break: float
    post_at_break: float
    additive_shift: float       # add this to all pre-break observations
    relative_shift_pct: float
    n_pre: int
    n_post: int


def _fit_local_linear(y: np.ndarray) -> tuple[float, float]:
    """Fit y = a + b*t. Returns (intercept, slope)."""
    if np.isnan(y).any():
        raise ValueError("NaN in fit window")
    t = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(t, y, 1)
    return float(intercept), float(slope)


def estimate_break_shift(
    df: pd.DataFrame,
    col: str,
    date_col: str = "date",
    break_date: pd.Timestamp = BREAK_DATE,
    pre_months: int = PRE_WINDOW_MONTHS,
    post_months: int = POST_WINDOW_MONTHS,
) -> SpliceResult:
    """Estimate the level shift at break_date via local linear fits."""
    df = df.sort_values(date_col).reset_index(drop=True)

    pre_start = break_date - pd.DateOffset(months=pre_months)
    pre_end = break_date - pd.DateOffset(months=1)
    post_start = break_date
    post_end = break_date + pd.DateOffset(months=post_months - 1)

    pre = df[(df[date_col] >= pre_start) & (df[date_col] <= pre_end)]
    post = df[(df[date_col] >= post_start) & (df[date_col] <= post_end)]

    if len(pre) < 3 or len(post) < 3:
        raise ValueError(
            f"Insufficient data around break: pre={len(pre)}, post={len(post)}. "
            f"Need >=3 each."
        )

    pre_y = pre[col].to_numpy(dtype=float)
    post_y = post[col].to_numpy(dtype=float)

    pre_a, pre_b = _fit_local_linear(pre_y)
    post_a, post_b = _fit_local_linear(post_y)

    # Pre window has t = 0..len(pre)-1. The break sits at t = len(pre).
    pre_at_break = pre_a + pre_b * len(pre)
    # Post window has t = 0 at the break itself.
    post_at_break_val = post_a

    shift = post_at_break_val - pre_at_break
    rel = (shift / pre_at_break) * 100 if pre_at_break else float("nan")

    return SpliceResult(
        series_name=col,
        break_date=break_date,
        pre_intercept=pre_a, pre_slope=pre_b,
        post_intercept=post_a, post_slope=post_b,
        pre_extrapolated_at_break=pre_at_break,
        post_at_break=post_at_break_val,
        additive_shift=shift,
        relative_shift_pct=rel,
        n_pre=len(pre), n_post=len(post),
    )


def apply_splice(
    df: pd.DataFrame,
    col: str,
    shift: float,
    date_col: str = "date",
    break_date: pd.Timestamp = BREAK_DATE,
) -> pd.DataFrame:
    """Return a copy of df with `shift` added to pre-break values of `col`."""
    out = df.copy()
    mask = out[date_col] < break_date
    out.loc[mask, col] = out.loc[mask, col] + shift
    return out


def splice_series(
    df: pd.DataFrame,
    col: str,
    date_col: str = "date",
    break_date: pd.Timestamp = BREAK_DATE,
    pre_months: int = PRE_WINDOW_MONTHS,
    post_months: int = POST_WINDOW_MONTHS,
) -> tuple[pd.DataFrame, SpliceResult]:
    """Estimate and apply the splice in one call."""
    result = estimate_break_shift(
        df, col, date_col, break_date, pre_months, post_months
    )
    return apply_splice(df, col, result.additive_shift, date_col, break_date), result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("input", type=Path, help="processed PSI csv/parquet")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--col", required=True,
                   help="column to splice (e.g. debit_outstanding)")
    p.add_argument("--date-col", default="date")
    p.add_argument("--break-date", default=str(BREAK_DATE.date()),
                   help="override the break date (YYYY-MM-DD)")
    p.add_argument("--pre-months", type=int, default=PRE_WINDOW_MONTHS)
    p.add_argument("--post-months", type=int, default=POST_WINDOW_MONTHS)
    p.add_argument("--report", type=Path,
                   help="optional JSON path for the splice report")
    args = p.parse_args()

    if args.input.suffix == ".parquet":
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input, parse_dates=[args.date_col])

    spliced, result = splice_series(
        df, args.col, date_col=args.date_col,
        break_date=pd.Timestamp(args.break_date),
        pre_months=args.pre_months, post_months=args.post_months,
    )

    if args.out.suffix == ".parquet":
        spliced.to_parquet(args.out, index=False)
    else:
        spliced.to_csv(args.out, index=False)

    print(f"\nLevel splice on '{args.col}' at {result.break_date.date()}")
    print(f"  pre  fit: intercept={result.pre_intercept:,.0f}  slope={result.pre_slope:+,.0f}/mo  (n={result.n_pre})")
    print(f"  post fit: intercept={result.post_intercept:,.0f}  slope={result.post_slope:+,.0f}/mo  (n={result.n_post})")
    print(f"  extrapolated pre-value at break:  {result.pre_extrapolated_at_break:,.0f}")
    print(f"  actual       post-value at break: {result.post_at_break:,.0f}")
    print(f"  additive shift applied to pre-break: {result.additive_shift:+,.0f} ({result.relative_shift_pct:+.2f}%)")
    print(f"  -> output: {args.out}")

    # Slope sanity warning
    if result.pre_slope != 0:
        ratio = result.post_slope / result.pre_slope
        if not (0.5 <= ratio <= 2.0):
            print(
                f"\nWARNING: Pre/post slopes differ materially (ratio={ratio:.2f}). "
                f"This may not be a pure level shift -- verify visually before relying on the splice."
            )

    if args.report:
        d = dataclasses.asdict(result)
        d["break_date"] = str(d["break_date"])
        with open(args.report, "w") as f:
            json.dump(d, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())