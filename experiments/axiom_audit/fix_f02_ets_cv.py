"""
FIX F2: Implement rolling CV for ETS models
============================================
ETS banks currently skip CV entirely during production runs.
This implements walk-forward CV matching Prophet's config:
  initial=36 months, horizon=6 months, step=6 months.

MAPE computed on original scale (expm1 if log-transformed).
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from loguru import logger


def run_ets_cv(
    bank_df: pd.DataFrame,
    bank_name: str,
    card_type: str,
    initial_months: int = 36,
    horizon_months: int = 6,
    step_months: int = 6,
    use_log_transform: bool = True,
) -> dict:
    """Walk-forward cross-validation for Holt-Winters ETS.

    Args:
        bank_df: Prophet-format DataFrame (ds, y). y is log1p if use_log_transform.
        bank_name: Bank identifier.
        card_type: 'cc' or 'dc'.
        initial_months: Minimum training window size.
        horizon_months: Forecast horizon per fold.
        step_months: Step between fold starts.
        use_log_transform: Whether y is in log1p space.

    Returns:
        Dict with mape_mean, mape_median, mape_min, mape_max, cv_windows.
    """
    y_all = bank_df["y"].values
    n = len(y_all)

    if n < initial_months + horizon_months:
        logger.warning(f"  {bank_name}: insufficient data for ETS CV ({n} < {initial_months + horizon_months})")
        return {"bank_name": bank_name, "card_type": card_type, "mape_mean": None, "mape_median": None}

    fold_mapes = []
    start = initial_months

    while start + horizon_months <= n:
        y_train = y_all[:start]
        y_test = y_all[start:start + horizon_months]

        try:
            model = ExponentialSmoothing(
                y_train,
                trend="add",
                seasonal="add",
                seasonal_periods=12,
                initialization_method="heuristic",
            )
            fit = model.fit(optimized=True)
            y_pred = fit.forecast(horizon_months)

            # Back-transform for MAPE on original scale
            if use_log_transform:
                y_test_orig = np.expm1(y_test)
                y_pred_orig = np.expm1(y_pred)
            else:
                y_test_orig = y_test
                y_pred_orig = y_pred

            # Per-step MAPE, capped at 100%
            with np.errstate(divide="ignore", invalid="ignore"):
                ape = np.abs(y_test_orig - y_pred_orig) / np.abs(y_test_orig)
                ape = np.clip(ape, 0, 1.0)
                fold_mape = np.mean(ape)

            fold_mapes.append(fold_mape)
        except Exception as e:
            logger.debug(f"  {bank_name}: ETS CV fold at {start} failed: {e}")

        start += step_months

    if not fold_mapes:
        return {"bank_name": bank_name, "card_type": card_type, "mape_mean": None, "mape_median": None}

    mapes = np.array(fold_mapes) * 100
    result = {
        "bank_name": bank_name,
        "card_type": card_type,
        "mape_mean": float(np.mean(mapes)),
        "mape_median": float(np.median(mapes)),
        "mape_min": float(np.min(mapes)),
        "mape_max": float(np.max(mapes)),
        "cv_windows": len(mapes),
        "n_capped": int(np.sum(np.array(fold_mapes) >= 1.0)),
    }
    logger.info(
        f"  [{card_type.upper()}] {bank_name} (ETS CV): "
        f"median {result['mape_median']:.2f}% (mean {result['mape_mean']:.2f}%) "
        f"[{result['mape_min']:.2f}%–{result['mape_max']:.2f}%] "
        f"({result['cv_windows']} folds)"
    )
    return result


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 80
    t = np.arange(n)
    y = np.log1p(1000 + 10 * t + 50 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 20, n))

    df = pd.DataFrame({
        "ds": pd.date_range("2018-01-01", periods=n, freq="MS"),
        "y": y,
    })

    result = run_ets_cv(df, "Test Bank", "cc", use_log_transform=True)
    print(f"\nResult: {result}")
