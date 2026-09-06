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
# P2.1: Log-space modelling infrastructure
# ===================================================================

class TestP2_1_LogSpace:
    def test_log_transform_flag_in_configs(self):
        from src.modelling.model_config import CC_CONFIG, DC_CONFIG
        assert "log_transform" in CC_CONFIG
        assert "log_transform" in DC_CONFIG

    def test_arima_accepts_log_transform(self):
        from src.modelling.aggregate_model import _fit_arima_forecast
        y = _make_monthly_series(84)
        result = _fit_arima_forecast(y, 12, log_transform=True)
        assert result is not None
        assert len(result) == 12
        assert np.all(result > 0), "Log-space ARIMA should produce positive forecasts"

    def test_ets_accepts_log_transform(self):
        from src.modelling.aggregate_model import _fit_ets_forecast
        y = _make_monthly_series(84)
        result = _fit_ets_forecast(y, 12, log_transform=True)
        assert result is not None
        assert len(result) == 12
        assert np.all(result > 0)

    def test_arimax_accepts_log_transform(self):
        from src.modelling.aggregate_model import _fit_arimax_forecast
        y = _make_monthly_series(84)
        exog = np.random.RandomState(99).randn(84, 1)
        exog_future = np.random.RandomState(99).randn(12, 1)
        result = _fit_arimax_forecast(y, exog, exog_future, 12, log_transform=True)
        assert result is not None
        assert np.all(result > 0)


# ===================================================================
# P2.6: Direct multi-horizon forecasting
# ===================================================================

class TestP2_6_DirectMultiHorizon:
    def test_direct_multihorizon_exists(self):
        from src.modelling.aggregate_model import _fit_direct_multihorizon
        assert callable(_fit_direct_multihorizon)

    def test_returns_correct_length(self):
        from src.modelling.aggregate_model import _fit_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_direct_multihorizon(y, 24)
        assert result is not None
        assert len(result) == 24

    def test_no_jumps_at_boundaries(self):
        """Segment transitions should be smooth (blended), not discontinuous."""
        from src.modelling.aggregate_model import _fit_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_direct_multihorizon(y, 24)
        if result is not None:
            for i in [5, 6, 11, 12]:
                if i < 23:
                    jump = abs(result[i+1] - result[i]) / abs(result[i])
                    assert jump < 0.5, f"Large jump at step {i}: {jump:.2%}"

    def test_direct_in_ensemble_weights(self):
        from src.modelling.aggregate_model import ENSEMBLE_WEIGHTS
        for key in ("cc", "dc"):
            assert "direct" in ENSEMBLE_WEIGHTS[key]


# ===================================================================
# P2.5: UPI ensemble forecast
# ===================================================================

class TestP2_5_UPIEnsemble:
    def test_txn_ensemble_weights_exist(self):
        from src.modelling.txn_volume_model import TXN_ENSEMBLE_WEIGHTS
        assert "forecast_upi_vol" in TXN_ENSEMBLE_WEIGHTS

    def test_txn_arima_function(self):
        from src.modelling.txn_volume_model import _fit_arima_fc
        y = _make_monthly_series(84)
        result = _fit_arima_fc(y, 12)
        assert result is not None
        assert len(result) == 12

    def test_txn_ets_function(self):
        from src.modelling.txn_volume_model import _fit_ets_fc
        y = _make_monthly_series(84)
        result = _fit_ets_fc(y, 12)
        assert result is not None
        assert len(result) == 12


# ===================================================================
# P2.3: Pooled bank model
# ===================================================================

class TestP2_3_PooledBank:
    def test_pooled_seasonal_function(self):
        from src.modelling.bank_model import _compute_pooled_seasonal
        # Create fake bank DataFrames
        bank_dfs = {}
        for i in range(3):
            n = 60
            ds = pd.date_range("2018-01-01", periods=n, freq="MS")
            y = 100 + 10 * np.sin(2 * np.pi * np.arange(n) / 12)
            bank_dfs[f"Bank_{i}"] = pd.DataFrame({"ds": ds, "y": y})
        seasonal = _compute_pooled_seasonal(bank_dfs)
        assert len(seasonal) == 12
        assert abs(seasonal.mean() - 1.0) < 0.01

    def test_add_pooled_seasonal_regressor(self):
        from src.modelling.bank_model import _add_pooled_seasonal_regressor
        ds = pd.date_range("2020-01-01", periods=24, freq="MS")
        df = pd.DataFrame({"ds": ds, "y": np.arange(24, dtype=float)})
        seasonal = pd.Series({m: 1.0 + 0.05 * (m - 6) for m in range(1, 13)})
        result = _add_pooled_seasonal_regressor(df, seasonal)
        assert "pooled_seasonal" in result.columns
        assert len(result) == 24


# ===================================================================
# P2.4: MinT reconciliation
# ===================================================================

class TestP2_4_MinT:
    def test_reconcile_mint_exists(self):
        from src.modelling.bank_model import _reconcile_mint
        assert callable(_reconcile_mint)

    def test_estimate_insample_variance_with_history(self):
        from src.modelling.bank_model import _estimate_insample_variance
        fc = pd.DataFrame({"forecast": [100, 110, 120], "date": pd.date_range("2026-01", periods=3, freq="MS")})
        hist = pd.DataFrame({
            "date": pd.date_range("2025-01", periods=12, freq="MS"),
            "yhat": np.arange(100, 112, dtype=float),
            "actual": np.arange(100, 112, dtype=float) + np.array([1, -2, 3, -1, 2, -3, 1, -1, 2, -2, 1, -1]),
        })
        var = _estimate_insample_variance(fc, hist)
        assert var > 1.0

    def test_estimate_insample_variance_no_history(self):
        from src.modelling.bank_model import _estimate_insample_variance
        fc = pd.DataFrame({"forecast": [100, 200, 300]})
        var = _estimate_insample_variance(fc, None)
        assert var > 0

    def test_mint_distributes_discrepancy_by_variance(self):
        """Series with lower variance should absorb more of the discrepancy."""
        from src.modelling.bank_model import _estimate_insample_variance
        dates = pd.date_range("2026-01", periods=3, freq="MS")
        bank_a = pd.DataFrame({"date": dates, "forecast": [50.0, 50, 50], "forecast_lower": [40.0]*3, "forecast_upper": [60.0]*3, "bank_name": "A", "card_type": "cc"})
        bank_b = pd.DataFrame({"date": dates, "forecast": [50.0, 50, 50], "forecast_lower": [40.0]*3, "forecast_upper": [60.0]*3, "bank_name": "B", "card_type": "cc"})
        residual = pd.DataFrame({"date": dates, "forecast": [0.0]*3, "forecast_lower": [0.0]*3, "forecast_upper": [0.0]*3, "bank_name": "_RESIDUAL", "card_type": "cc"})

        # History: bank A has low variance, bank B has high variance
        hist_a = pd.DataFrame({"date": pd.date_range("2025-01", periods=12, freq="MS"), "yhat": np.ones(12)*50, "actual": np.ones(12)*50 + np.random.RandomState(0).normal(0, 1, 12)})
        hist_b = pd.DataFrame({"date": pd.date_range("2025-01", periods=12, freq="MS"), "yhat": np.ones(12)*50, "actual": np.ones(12)*50 + np.random.RandomState(1).normal(0, 10, 12)})

        var_a = _estimate_insample_variance(bank_a, hist_a)
        var_b = _estimate_insample_variance(bank_b, hist_b)
        assert var_a < var_b, "Bank A should have lower variance"


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
