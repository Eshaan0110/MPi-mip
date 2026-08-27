"""
P2 Accuracy Improvement Tests
==============================
Tests for P2.2 (ARIMAX ensemble member) and P2.7 (metrics module).

Run:  uv run pytest tests/test_p2_improvements.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_monthly_series(n: int = 84, trend: float = 2.0, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 1, n)
    return 100 + trend * t + seasonal + noise


# ===================================================================
# P2.2: ARIMAX as fourth ensemble member
# ===================================================================

class TestP2_2_ARIMAX:
    def test_arimax_function_exists(self):
        from src.modelling.aggregate_model import _fit_arimax_forecast
        assert callable(_fit_arimax_forecast)

    def test_arimax_returns_array(self):
        from src.modelling.aggregate_model import _fit_arimax_forecast
        y = _make_monthly_series(84)
        exog = np.random.RandomState(99).randn(84, 1)
        exog_future = np.random.RandomState(99).randn(24, 1)
        result = _fit_arimax_forecast(y, exog, exog_future, 24)
        assert result is not None, "ARIMAX returned None on valid input"
        assert len(result) == 24

    def test_arimax_in_ensemble_weights(self):
        from src.modelling.aggregate_model import ENSEMBLE_WEIGHTS
        for key in ("cc", "dc"):
            assert "arimax" in ENSEMBLE_WEIGHTS[key], f"No arimax key in {key} weights"

    def test_arimax_in_weight_estimation(self):
        from src.modelling.aggregate_model import _estimate_ensemble_weights
        from src.modelling.model_config import CC_CONFIG
        y = _make_monthly_series(84)
        weights = _estimate_ensemble_weights(y, CC_CONFIG, 24)
        assert "arimax" in weights, "arimax not in re-estimated weights"

    def test_run_forecast_includes_arimax(self):
        import inspect
        from src.modelling.aggregate_model import run_forecast
        src = inspect.getsource(run_forecast)
        assert "arimax" in src, "run_forecast doesn't reference arimax"

    def test_get_regressor_cols_exists(self):
        from src.modelling.aggregate_model import _get_regressor_cols
        assert callable(_get_regressor_cols)


# ===================================================================
# P2.7: Scale-independent metrics
# ===================================================================

class TestP2_7_Metrics:
    def test_mape(self):
        from src.modelling.metrics import mape
        actual = np.array([100.0, 200.0, 300.0])
        pred = np.array([110.0, 190.0, 310.0])
        result = mape(actual, pred)
        assert 0 < result < 10

    def test_mae(self):
        from src.modelling.metrics import mae
        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 190.0])
        assert mae(actual, pred) == 10.0

    def test_rmse(self):
        from src.modelling.metrics import rmse
        actual = np.array([100.0, 200.0])
        pred = np.array([110.0, 190.0])
        assert rmse(actual, pred) == 10.0

    def test_mase_below_one_beats_naive(self):
        from src.modelling.metrics import mase
        rng = np.random.RandomState(42)
        training = 100 + np.arange(60, dtype=float) + rng.normal(0, 2, 60)
        actual = training[-12:]
        pred = actual + rng.normal(0, 0.5, 12)
        result = mase(actual, pred, training, seasonal_period=12)
        assert result < 1.0, f"MASE {result} should be < 1 for a good forecast"

    def test_rmsse(self):
        from src.modelling.metrics import rmsse
        training = np.arange(60, dtype=float)
        actual = np.array([60.0, 61.0, 62.0])
        pred = np.array([60.5, 61.5, 62.5])
        result = rmsse(actual, pred, training)
        assert result > 0

    def test_pinball_loss(self):
        from src.modelling.metrics import pinball_loss
        actual = np.array([100.0, 200.0, 300.0])
        lower = np.array([90.0, 180.0, 280.0])
        upper = np.array([110.0, 220.0, 320.0])
        result = pinball_loss(actual, lower, upper, alpha=0.10)
        assert result >= 0

    def test_empirical_coverage(self):
        from src.modelling.metrics import empirical_coverage
        actual = np.array([100.0, 200.0, 300.0, 400.0])
        lower = np.array([90.0, 180.0, 310.0, 380.0])
        upper = np.array([110.0, 220.0, 290.0, 420.0])
        cov = empirical_coverage(actual, lower, upper)
        assert cov == 0.75  # 3 of 4 within bounds (300 is outside [310,290])

    def test_score_forecast_returns_all_keys(self):
        from src.modelling.metrics import score_forecast
        training = np.arange(60, dtype=float)
        actual = np.array([60.0, 61.0, 62.0])
        pred = np.array([60.5, 61.5, 62.5])
        scores = score_forecast(actual, pred, training)
        for key in ("mape", "mae", "rmse", "mase", "rmsse"):
            assert key in scores, f"Missing key: {key}"

    def test_score_intervals_returns_all_keys(self):
        from src.modelling.metrics import score_intervals
        actual = np.array([100.0, 200.0])
        lower = np.array([90.0, 190.0])
        upper = np.array([110.0, 210.0])
        scores = score_intervals(actual, lower, upper)
        for key in ("coverage", "pinball_loss", "mean_width"):
            assert key in scores, f"Missing key: {key}"
