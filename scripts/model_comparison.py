"""
ETS vs Prophet comparison on stable CC and DC banks.

Runs rolling CV (identical folds) for:
  - Prophet (current config, log1p)
  - Holt-Winters ETS (additive trend + seasonality)
  - AutoARIMA (via pmdarima if available, else skipped)

Reports MAPE on ORIGINAL scale for each bank.
Only run on clean-series banks (no merger events in training window).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

cc_bw = pd.read_parquet(PROCESSED / "bankwise_cards_cc.parquet")
cc_bw["date"] = pd.to_datetime(cc_bw["date"]).dt.to_period("M").dt.to_timestamp()
dc_bw = pd.read_parquet(PROCESSED / "bankwise_cards_dc.parquet")
dc_bw["date"] = pd.to_datetime(dc_bw["date"]).dt.to_period("M").dt.to_timestamp()

INITIAL = 48   # months of training for first fold
HORIZON = 6    # forecast horizon
STEP    = 6    # rolling step

# Only clean-series banks (no merger in training window)
CC_STABLE = ["Axis Bank", "IndusInd Bank", "HSBC", "ICICI Bank"]
DC_STABLE = ["HDFC Bank", "Axis Bank", "ICICI Bank", "Bank of India",
             "Central Bank of India", "UCO Bank", "Indian Overseas Bank"]


def prophet_cv(series: pd.Series) -> float:
    """Rolling CV MAPE for Prophet on original scale (log1p internally)."""
    vals = series.values
    n = len(vals)
    mapes = []
    pos = INITIAL
    while pos + HORIZON <= n:
        tr_y = vals[:pos]
        te_y = vals[pos:pos + HORIZON]
        dates = series.index
        tr_ds = dates[:pos]
        te_ds = dates[pos:pos + HORIZON]

        tr_df = pd.DataFrame({"ds": tr_ds, "y": np.log1p(tr_y)})
        te_df = pd.DataFrame({"ds": te_ds})

        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode="additive",
                    changepoint_prior_scale=0.05, interval_width=0.90)
        m.fit(tr_df)
        fc = m.predict(te_df)
        y_pred = np.expm1(fc["yhat"].values)
        mask = te_y != 0
        if mask.any():
            mapes.append(np.mean(np.abs((te_y[mask] - y_pred[mask]) / te_y[mask])) * 100)
        pos += STEP
    return float(np.mean(mapes)) if mapes else float("nan")


def ets_cv(series: pd.Series) -> float:
    """Rolling CV MAPE for Holt-Winters ETS (additive trend + seasonality)."""
    vals = series.values
    n = len(vals)
    mapes = []
    pos = INITIAL
    while pos + HORIZON <= n:
        tr_y = vals[:pos]
        te_y = vals[pos:pos + HORIZON]
        try:
            model = ExponentialSmoothing(
                tr_y,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="heuristic",
            )
            fit = model.fit(optimized=True)
            y_pred = fit.forecast(HORIZON)
            mask = (te_y != 0) & np.isfinite(y_pred)
            if mask.any():
                mapes.append(np.mean(np.abs((te_y[mask] - y_pred[mask]) / te_y[mask])) * 100)
        except Exception as exc:
            pass
        pos += STEP
    return float(np.mean(mapes)) if mapes else float("nan")


def arima_cv(series: pd.Series) -> float:
    """Rolling CV MAPE for AutoARIMA (requires pmdarima)."""
    try:
        from pmdarima import auto_arima
    except ImportError:
        return float("nan")
    vals = series.values
    n = len(vals)
    mapes = []
    pos = INITIAL
    while pos + HORIZON <= n:
        tr_y = vals[:pos]
        te_y = vals[pos:pos + HORIZON]
        try:
            model = auto_arima(tr_y, seasonal=True, m=12, suppress_warnings=True,
                               error_action="ignore", stepwise=True)
            y_pred = model.predict(HORIZON)
            mask = te_y != 0
            if mask.any():
                mapes.append(np.mean(np.abs((te_y[mask] - y_pred[mask]) / te_y[mask])) * 100)
        except Exception:
            pass
        pos += STEP
    return float(np.mean(mapes)) if mapes else float("nan")


def run_comparison(bw: pd.DataFrame, col: str, banks: list[str], label: str,
                   start: str = "2017-01-01") -> None:
    print(f"\n{'='*80}")
    print(f"{label} — Prophet vs ETS vs AutoARIMA (MAPE on original scale)")
    print(f"Folds: {INITIAL}m initial / {HORIZON}m horizon / {STEP}m step")
    print(f"{'='*80}")
    print(f"  {'Bank':<28} {'Prophet':>9} {'ETS':>9} {'AutoARIMA':>11} {'Winner':>9}")
    print("  " + "-" * 70)

    for bank in banks:
        bdf = bw[(bw.bank_name == bank) & bw[col].notna() & (bw[col] > 0)]
        bdf = bdf[bdf.date >= start].sort_values("date").set_index("date")[col]
        if len(bdf) < INITIAL + HORIZON:
            print(f"  {bank:<28} {'(skip — short)':>42}")
            continue

        p  = prophet_cv(bdf)
        e  = ets_cv(bdf)
        a  = arima_cv(bdf)

        scores = {"Prophet": p, "ETS": e}
        if not np.isnan(a):
            scores["AutoARIMA"] = a
        winner = min(scores, key=scores.get)

        a_str = f"{a:.2f}%" if not np.isnan(a) else "  N/A   "
        print(f"  {bank:<28} {p:>7.2f}%  {e:>7.2f}%  {a_str:>10}  {winner:>9}")


if __name__ == "__main__":
    run_comparison(cc_bw, "cc_outstanding", CC_STABLE, "CC STABLE BANKS")
    run_comparison(dc_bw, "dc_outstanding", DC_STABLE, "DC STABLE BANKS")
    print("\nDone.")
