"""
qr_ablation.py -- ablation test for the UPI QR codes regressor on the
credit cards outstanding model.

Fits the credit model with and without `upi_qr_codes`, then compares:
  - rolling-CV MAPE on both specs (same folds, same seed)
  - the 12-month forward forecast point values
  - per-month and overall % deltas

Decision rule (set in CLAUDE.md):
  - mean |d%| < 1.0%  ->  drop QR (regressor is duplicating trend)
  - otherwise          ->  escalate to Granger causality on first-differenced
                          series, QR saturation scenario, and RuPay credit-
                          on-UPI as a cleaner alternative

Note on the forward-forecast regressors: this script naïvely carries
forward the last observed value of each regressor for the 12-month horizon.
This is fine for an ablation (both models get the same future), but
NOT what you want for the headline forecast -- for that, pass proper
scenarios via `future_regressors_df`.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from prophet import Prophet


DROP_THRESHOLD_PCT = 1.0
HORIZON_MONTHS = 12
CV_HORIZON = 6
CV_STEP = 6
CV_INITIAL = 48


@dataclass
class AblationResult:
    forecast_with: pd.DataFrame
    forecast_without: pd.DataFrame
    per_month_pct_delta: pd.Series
    mean_abs_pct_delta: float
    max_abs_pct_delta: float
    cv_mape_with: float
    cv_mape_without: float
    decision: str

    def summary(self) -> str:
        lines = [
            "UPI QR regressor ablation -- credit cards outstanding",
            "",
            f"  CV MAPE with QR:    {self.cv_mape_with:.2%}",
            f"  CV MAPE without QR: {self.cv_mape_without:.2%}",
            f"  dCV MAPE:           {(self.cv_mape_without - self.cv_mape_with):+.2%}",
            "",
            f"  12-month forecast mean |d%|: {self.mean_abs_pct_delta:.2f}%",
            f"  12-month forecast max  |d%|: {self.max_abs_pct_delta:.2f}%",
            "",
            f"  Decision: {self.decision}",
            "",
            "Per-month forecast delta (with vs without QR):",
        ]
        for ds, pct in self.per_month_pct_delta.items():
            lines.append(f"  {pd.Timestamp(ds).strftime('%Y-%m')}: {pct:+.2f}%")
        return "\n".join(lines)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def _fit(df: pd.DataFrame, regressors: list[str]) -> Prophet:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        for r in regressors:
            m.add_regressor(r)
        m.fit(df[["ds", "y"] + regressors])
    return m


def _forecast(
    df: pd.DataFrame,
    regressors: list[str],
    horizon: int,
    future_regressors_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit and produce a `horizon`-month forward forecast."""
    m = _fit(df, regressors)
    last_ds = df["ds"].iloc[-1]
    future_dates = pd.date_range(
        last_ds + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )
    future = pd.DataFrame({"ds": future_dates})
    if future_regressors_df is not None:
        future_regressors_df = future_regressors_df.set_index("ds")
        for r in regressors:
            future[r] = future["ds"].map(future_regressors_df[r])
            if future[r].isna().any():
                raise ValueError(
                    f"future_regressors_df is missing values for '{r}'. "
                    f"Provide all horizon months."
                )
    else:
        # Naïve carry-forward
        for r in regressors:
            future[r] = df[r].iloc[-1]

    full = pd.concat(
        [df[["ds"] + regressors], future],
        ignore_index=True,
    )
    pred = m.predict(full)
    fc = pred.iloc[-horizon:][["ds", "yhat", "yhat_lower", "yhat_upper"]]
    return fc.reset_index(drop=True)


def _rolling_cv_mape(
    df: pd.DataFrame,
    regressors: list[str],
    initial: int = CV_INITIAL,
    horizon: int = CV_HORIZON,
    step: int = CV_STEP,
) -> float:
    n = len(df)
    mapes = []
    pos = initial
    while pos + horizon <= n:
        tr = df.iloc[:pos]
        te = df.iloc[pos:pos + horizon]
        try:
            m = _fit(tr, regressors)
            pred = m.predict(te[["ds"] + regressors])
            mapes.append(_mape(te["y"].to_numpy(), pred["yhat"].to_numpy()))
        except Exception:
            pass
        pos += step
    return float(np.mean(mapes)) if mapes else float("nan")


def ablate(
    df: pd.DataFrame,
    y_col: str,
    qr_col: str,
    other_regressors: list[str],
    horizon: int = HORIZON_MONTHS,
    date_col: str = "date",
    threshold_pct: float = DROP_THRESHOLD_PCT,
    future_regressors_df: pd.DataFrame | None = None,
) -> AblationResult:
    df = (df.sort_values(date_col)
            .reset_index(drop=True)
            .rename(columns={date_col: "ds", y_col: "y"}))

    with_qr = other_regressors + [qr_col]
    without_qr = list(other_regressors)

    f_with = _forecast(df, with_qr, horizon, future_regressors_df)
    f_without = _forecast(df, without_qr, horizon, future_regressors_df)

    pct = (
        (f_with["yhat"].to_numpy() - f_without["yhat"].to_numpy())
        / f_without["yhat"].to_numpy()
        * 100
    )
    pct_s = pd.Series(pct, index=f_with["ds"].to_numpy())
    mean_abs = float(np.mean(np.abs(pct)))
    max_abs = float(np.max(np.abs(pct)))

    cv_with = _rolling_cv_mape(df, with_qr)
    cv_without = _rolling_cv_mape(df, without_qr)

    if mean_abs < threshold_pct:
        decision = (
            f"DROP QR  (mean |d%| = {mean_abs:.2f}% < {threshold_pct:.1f}% threshold; "
            "QR is duplicating Prophet's trend component)"
        )
    else:
        decision = (
            f"ESCALATE  (mean |d%| = {mean_abs:.2f}% >= {threshold_pct:.1f}% threshold; "
            "run Granger on first-differenced series, test QR saturation scenario, "
            "evaluate RuPay credit-on-UPI volume as alternative)"
        )

    return AblationResult(
        forecast_with=f_with,
        forecast_without=f_without,
        per_month_pct_delta=pct_s,
        mean_abs_pct_delta=mean_abs,
        max_abs_pct_delta=max_abs,
        cv_mape_with=cv_with,
        cv_mape_without=cv_without,
        decision=decision,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--y", default="credit_outstanding")
    p.add_argument("--qr-col", default="upi_qr_codes")
    p.add_argument("--other-regressors", nargs="+", default=[
        "credit_pos_volume", "credit_online_volume", "pos_terminals", "repo_rate",
    ])
    p.add_argument("--horizon", type=int, default=HORIZON_MONTHS)
    p.add_argument("--date-col", default="date")
    p.add_argument("--threshold-pct", type=float, default=DROP_THRESHOLD_PCT)
    p.add_argument("--future-regressors",
                   type=Path, default=None,
                   help="optional csv/parquet with future ds + regressor values "
                        "for the horizon. If omitted, last values are carried forward.")
    args = p.parse_args()

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=[args.date_col])

    fut = None
    if args.future_regressors:
        if args.future_regressors.suffix == ".parquet":
            fut = pd.read_parquet(args.future_regressors)
        else:
            fut = pd.read_csv(args.future_regressors, parse_dates=["ds"])

    result = ablate(
        df, y_col=args.y, qr_col=args.qr_col,
        other_regressors=args.other_regressors,
        horizon=args.horizon, date_col=args.date_col,
        threshold_pct=args.threshold_pct,
        future_regressors_df=fut,
    )
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())