"""
repo_lag_cv.py -- nested rolling cross-validation for repo rate lag
selection on the credit cards outstanding model.

Outer loop: rolling test folds (the honest performance estimate).
Inner loop: within each outer fold's TRAINING data alone, pick the lag in
{3, 6, 9} months that minimises inner-loop MAPE.

The reported headline = mean outer-fold MAPE where each fold uses the lag
chosen on that fold's training data alone -- no information leakage from
the test set into the hyperparameter choice. The frequency distribution
of inner-loop lag picks tells you whether the choice is stable across
folds or whether the data simply can't discriminate.

Anchor on prior: Kapur & Behera (RBI WP 2012) and the RBI Monetary
Transmission Report estimate bank lending rate pass-through lags in the
3-12 month range. Credit cards are NOT on EBLR (fixed APR), so the channel
runs through bank risk appetite rather than direct cost-of-funds -- the
prior is weaker than the transmission literature suggests. Don't override
the CV result with the literature; report both.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from prophet import Prophet


# Project defaults per CLAUDE.md
DEFAULT_LAGS = (3, 6, 9)
INITIAL_TRAIN_MONTHS = 48
HORIZON_MONTHS = 6
STEP_MONTHS = 6
INNER_FOLDS = 3


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

@dataclass
class OuterFoldResult:
    fold_idx: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    chosen_lag: int
    inner_mape_by_lag: dict[int, float]
    outer_mape: float


@dataclass
class NestedCVResult:
    folds: list[OuterFoldResult] = field(default_factory=list)

    @property
    def outer_mapes(self) -> list[float]:
        return [f.outer_mape for f in self.folds if not np.isnan(f.outer_mape)]

    @property
    def mean_outer_mape(self) -> float:
        m = self.outer_mapes
        return float(np.mean(m)) if m else float("nan")

    @property
    def std_outer_mape(self) -> float:
        m = self.outer_mapes
        return float(np.std(m)) if m else float("nan")

    @property
    def lag_pick_distribution(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for f in self.folds:
            out[f.chosen_lag] = out.get(f.chosen_lag, 0) + 1
        return out

    def summary(self) -> str:
        lines = [
            f"Nested rolling CV: {len(self.folds)} outer folds",
            f"  Honest MAPE: {self.mean_outer_mape:.2%} +/- {self.std_outer_mape:.2%}",
            f"  Lag picks:   {self.lag_pick_distribution}",
            "",
            "Per-fold detail:",
        ]
        for f in self.folds:
            inner = "  ".join(
                f"L{k}={v:.2%}" if not np.isinf(v) else f"L{k}=fail"
                for k, v in sorted(f.inner_mape_by_lag.items())
            )
            lines.append(
                f"  fold {f.fold_idx}: "
                f"train->{f.train_end.date()}  "
                f"test {f.test_start.date()}..{f.test_end.date()}  "
                f"chose lag={f.chosen_lag}  "
                f"outer_MAPE={f.outer_mape:.2%}  "
                f"[inner: {inner}]"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shift(df: pd.DataFrame, col: str, lag: int) -> pd.DataFrame:
    """Lag a single regressor column by `lag` months in-place on a copy."""
    out = df.copy()
    out[col] = out[col].shift(lag)
    return out


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = yt != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    regressors: list[str],
) -> np.ndarray:
    """Fit Prophet on train, predict on test, return yhat array."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )
        for r in regressors:
            m.add_regressor(r)
        m.fit(train[["ds", "y"] + regressors])
        pred = m.predict(test[["ds"] + regressors])
    return pred["yhat"].to_numpy()


def _inner_cv_for_lag(
    train_df: pd.DataFrame,
    other_regressors: list[str],
    repo_col: str,
    lag: int,
    horizon: int,
    step: int,
    n_folds: int,
) -> float:
    """Mean MAPE across rolling folds inside train_df for a specific lag."""
    lagged = _shift(train_df, repo_col, lag).dropna(subset=[repo_col])
    regressors = other_regressors + [repo_col]

    mapes = []
    n = len(lagged)
    # Carve folds from the end of train_df working backwards.
    for k in range(n_folds):
        test_end_pos = n - k * step
        test_start_pos = test_end_pos - horizon
        if test_start_pos < 36:  # need >=36 months train for inner fold
            break
        tr = lagged.iloc[:test_start_pos]
        te = lagged.iloc[test_start_pos:test_end_pos]
        if len(te) < horizon:
            continue
        try:
            yhat = _fit_predict(tr, te, regressors)
            mapes.append(_mape(te["y"].to_numpy(), yhat))
        except Exception:
            continue
    return float(np.mean(mapes)) if mapes else float("inf")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def nested_cv(
    df: pd.DataFrame,
    y_col: str,
    repo_col: str,
    other_regressors: list[str],
    lags: tuple[int, ...] = DEFAULT_LAGS,
    initial_train: int = INITIAL_TRAIN_MONTHS,
    horizon: int = HORIZON_MONTHS,
    step: int = STEP_MONTHS,
    inner_folds: int = INNER_FOLDS,
    date_col: str = "date",
    verbose: bool = True,
) -> NestedCVResult:
    """
    Nested rolling CV. Returns NestedCVResult with the honest outer-fold MAPEs.
    """
    df = (df.sort_values(date_col)
            .reset_index(drop=True)
            .rename(columns={date_col: "ds", y_col: "y"}))
    n = len(df)

    result = NestedCVResult()
    fold_idx = 0
    train_end_pos = initial_train

    while train_end_pos + horizon <= n:
        train = df.iloc[:train_end_pos]
        test = df.iloc[train_end_pos:train_end_pos + horizon]

        if verbose:
            print(f"  outer fold {fold_idx}: train->{train['ds'].iloc[-1].date()}, "
                  f"test {test['ds'].iloc[0].date()}..{test['ds'].iloc[-1].date()}")

        # Inner CV: pick the best lag using training data only
        inner = {
            lag: _inner_cv_for_lag(
                train, other_regressors, repo_col, lag,
                horizon=horizon, step=step, n_folds=inner_folds,
            )
            for lag in lags
        }
        chosen = min(inner, key=lambda k: inner[k])

        # Score outer test fold with the chosen lag
        all_so_far = df.iloc[:train_end_pos + horizon]
        lagged = _shift(all_so_far, repo_col, chosen).dropna(subset=[repo_col])
        tr_outer = lagged[lagged["ds"] <= train["ds"].iloc[-1]]
        te_outer = lagged[lagged["ds"] > train["ds"].iloc[-1]]

        if len(te_outer) >= horizon:
            try:
                yhat = _fit_predict(tr_outer, te_outer, other_regressors + [repo_col])
                outer_mape = _mape(te_outer["y"].to_numpy(), yhat)
            except Exception as e:
                if verbose:
                    print(f"    outer fold {fold_idx} fit failed: {e}")
                outer_mape = float("nan")
        else:
            outer_mape = float("nan")

        result.folds.append(OuterFoldResult(
            fold_idx=fold_idx,
            train_end=train["ds"].iloc[-1],
            test_start=test["ds"].iloc[0],
            test_end=test["ds"].iloc[-1],
            chosen_lag=chosen,
            inner_mape_by_lag=inner,
            outer_mape=outer_mape,
        ))
        fold_idx += 1
        train_end_pos += step

    return result


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path, help="master parquet/csv with all regressors")
    p.add_argument("--y", default="credit_outstanding")
    p.add_argument("--repo-col", default="repo_rate")
    p.add_argument("--other-regressors", nargs="+", default=[
        "credit_pos_volume", "credit_online_volume", "pos_terminals", "upi_qr_codes",
    ])
    p.add_argument("--lags", type=int, nargs="+", default=list(DEFAULT_LAGS))
    p.add_argument("--initial-train", type=int, default=INITIAL_TRAIN_MONTHS)
    p.add_argument("--horizon", type=int, default=HORIZON_MONTHS)
    p.add_argument("--step", type=int, default=STEP_MONTHS)
    p.add_argument("--inner-folds", type=int, default=INNER_FOLDS)
    p.add_argument("--date-col", default="date")
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args()

    if args.data.suffix == ".parquet":
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, parse_dates=[args.date_col])

    result = nested_cv(
        df, y_col=args.y, repo_col=args.repo_col,
        other_regressors=args.other_regressors,
        lags=tuple(args.lags),
        initial_train=args.initial_train,
        horizon=args.horizon, step=args.step,
        inner_folds=args.inner_folds,
        date_col=args.date_col,
        verbose=not args.quiet,
    )
    print()
    print(result.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())