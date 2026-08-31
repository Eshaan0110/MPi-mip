"""
MIP Modelling — Forecast Accuracy Metrics
===========================================
Scale-independent metrics alongside MAPE for honest model evaluation.

MAPE is asymmetric and flatters declining series (debit cards).
MASE/RMSSE are scale-free and symmetric — defensible to any reviewer.
Pinball loss scores interval forecasts, not just point forecasts.

Usage:
    from src.modelling.metrics import score_forecast, score_intervals
    scores = score_forecast(actual, predicted, seasonal_period=12)
    interval_scores = score_intervals(actual, lower, upper, alpha=0.10)
"""
from __future__ import annotations

import numpy as np


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_series: np.ndarray,
    seasonal_period: int = 12,
) -> float:
    """Mean Absolute Scaled Error.

    Scaled by the in-sample seasonal naive MAE. MASE < 1 means the
    forecast beats the seasonal naive baseline on average.
    """
    n = len(training_series)
    if n <= seasonal_period:
        return float("nan")
    naive_errors = np.abs(training_series[seasonal_period:] - training_series[:-seasonal_period])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def rmsse(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_series: np.ndarray,
    seasonal_period: int = 12,
) -> float:
    """Root Mean Squared Scaled Error.

    Like MASE but uses squared errors — penalises large misses more heavily.
    """
    n = len(training_series)
    if n <= seasonal_period:
        return float("nan")
    naive_errors = training_series[seasonal_period:] - training_series[:-seasonal_period]
    scale_sq = np.mean(naive_errors ** 2)
    if scale_sq == 0:
        return float("nan")
    return float(np.sqrt(np.mean((actual - predicted) ** 2) / scale_sq))


def pinball_loss(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """Pinball (quantile) loss for prediction intervals.

    Scores the lower bound at quantile alpha/2 and the upper bound at
    quantile 1 - alpha/2. Lower is better. A perfectly calibrated 90%
    interval has minimal pinball loss.
    """
    tau_lo = alpha / 2
    tau_hi = 1 - alpha / 2

    loss_lo = np.where(
        actual < lower,
        (1 - tau_lo) * (lower - actual),
        tau_lo * (actual - lower),
    )
    loss_hi = np.where(
        actual > upper,
        tau_hi * (actual - upper),
        (1 - tau_hi) * (upper - actual),
    )
    return float(np.mean(loss_lo + loss_hi))


def empirical_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Fraction of actuals that fall within [lower, upper]."""
    covered = (actual >= lower) & (actual <= upper)
    return float(np.mean(covered))


def score_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    training_series: np.ndarray,
    seasonal_period: int = 12,
) -> dict[str, float]:
    """Compute all point-forecast metrics in one call."""
    return {
        "mape": mape(actual, predicted),
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mase": mase(actual, predicted, training_series, seasonal_period),
        "rmsse": rmsse(actual, predicted, training_series, seasonal_period),
    }


def score_intervals(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.10,
) -> dict[str, float]:
    """Compute all interval metrics in one call."""
    return {
        "coverage": empirical_coverage(actual, lower, upper),
        "pinball_loss": pinball_loss(actual, lower, upper, alpha),
        "mean_width": float(np.mean(upper - lower)),
    }
