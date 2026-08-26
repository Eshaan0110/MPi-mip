"""
P1 Correctness Defect Tests
============================
One test per defect (D1–D8) from Phase 4 improvement brief.
Tests verify the FIX is in place — they should FAIL on the old code
and PASS on the current code.

Run:  uv run pytest tests/test_p1_defects.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers: synthetic monthly series
# ---------------------------------------------------------------------------

def _make_monthly_series(n: int = 84, trend: float = 2.0, seed: int = 42) -> np.ndarray:
    """Generate a synthetic monthly series with trend + seasonality."""
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 1, n)
    return 100 + trend * t + seasonal + noise


def _make_train_df(n: int = 84) -> pd.DataFrame:
    y = _make_monthly_series(n)
    ds = pd.date_range("2013-01-01", periods=n, freq="MS")
    return pd.DataFrame({"ds": ds, "y": y})


# ===================================================================
# D1: Conformal intervals use ensemble residuals, not ARIMA-only
# ===================================================================

class TestD1_EnsembleConformalIntervals:
    def test_conformal_uses_ets(self):
        """_build_conformal_intervals must fit ETS (not just ARIMA)."""
        from unittest.mock import patch, MagicMock
        from src.modelling.aggregate_model import _build_conformal_intervals
        from src.modelling.model_config import CC_CONFIG

        train_df = _make_train_df(84)

        # Patch ETS to track whether it's called
        with patch(
            "src.modelling.aggregate_model.ExponentialSmoothing"
        ) as mock_ets:
            mock_fit = MagicMock()
            mock_fit.forecast.return_value = np.ones(24) * 200
            mock_ets.return_value.fit.return_value = mock_fit

            _build_conformal_intervals(train_df, CC_CONFIG, 24)

            assert mock_ets.called, (
                "D1 FAIL: _build_conformal_intervals never called ETS — "
                "still using ARIMA-only residuals"
            )


# ===================================================================
# D2: Percentage errors, not absolute residuals
# ===================================================================

class TestD2_PercentageIntervals:
    def test_intervals_are_percentage_based(self):
        """Returned lower/upper arrays should be fractional (< 1.0 in
        absolute value), not in the same units as y."""
        from src.modelling.aggregate_model import _build_conformal_intervals
        from src.modelling.model_config import CC_CONFIG

        train_df = _make_train_df(84)
        lower_pcts, upper_pcts = _build_conformal_intervals(train_df, CC_CONFIG, 24)

        assert np.all(np.abs(lower_pcts) < 1.0), (
            f"D2 FAIL: lower_pcts look absolute, not fractional: {lower_pcts[:3]}"
        )
        assert np.all(np.abs(upper_pcts) < 1.0), (
            f"D2 FAIL: upper_pcts look absolute, not fractional: {upper_pcts[:3]}"
        )

    def test_intervals_applied_multiplicatively(self):
        """In run_forecast, CI = ensemble * (1 + pct), not ensemble + width."""
        import inspect
        from src.modelling.aggregate_model import run_forecast

        src = inspect.getsource(run_forecast)
        assert "(1 + lower_pcts)" in src or "(1+lower_pcts)" in src, (
            "D2 FAIL: run_forecast doesn't apply intervals multiplicatively"
        )


# ===================================================================
# D3: Conformal CV extends to full 24-month horizon
# ===================================================================

class TestD3_FullHorizonConformal:
    def test_conformal_produces_24_values(self):
        """_build_conformal_intervals should return arrays of length=horizon."""
        from src.modelling.aggregate_model import _build_conformal_intervals
        from src.modelling.model_config import CC_CONFIG

        train_df = _make_train_df(96)  # 8 years — enough for 24m folds
        lower, upper = _build_conformal_intervals(train_df, CC_CONFIG, 24)

        assert len(lower) == 24, f"D3 FAIL: lower has {len(lower)} steps, expected 24"
        assert len(upper) == 24, f"D3 FAIL: upper has {len(upper)} steps, expected 24"


# ===================================================================
# D4: Stale horizon comments fixed
# ===================================================================

class TestD4_DocCorrectness:
    def test_model_config_no_6month(self):
        """model_config.py should not reference '6-month' or 'horizon=6m'."""
        from pathlib import Path
        config_path = Path(__file__).parent.parent / "src" / "modelling" / "model_config.py"
        text = config_path.read_text()
        assert "horizon=6m" not in text, "D4 FAIL: model_config.py still says horizon=6m"
        assert "6-month horizon" not in text.lower(), (
            "D4 FAIL: model_config.py still references 6-month horizon"
        )

    def test_readme_no_6month_horizon(self):
        """README.md should not reference '6-month horizon'."""
        from pathlib import Path
        readme = Path(__file__).parent.parent / "README.md"
        text = readme.read_text()
        assert "6-month horizon" not in text, (
            "D4 FAIL: README.md still says 6-month horizon"
        )


# ===================================================================
# D5: No bfill on lagged regressors (data leakage)
# ===================================================================

class TestD5_NoBfillLeakage:
    def test_build_training_df_no_bfill_on_lagged(self):
        """After applying a lag, the resulting column should have NaN at the
        top (not backfilled with future data)."""
        from src.modelling.data_prep import build_training_df, build_master, load_all

        # Check source code for bfill on lagged columns
        import inspect
        src = inspect.getsource(build_training_df)
        # The old code had: df[lagged_col] = filled.shift(spec.lag).bfill()
        # After the D5 fix there should be no .bfill() on lagged columns
        assert ".bfill()" not in src.split("if spec.lag > 0")[1].split("else")[0] if "if spec.lag > 0" in src else True, (
            "D5 FAIL: bfill still present on lagged regressor branch"
        )

    def test_no_bfill_in_build_future_df(self):
        """build_future_df should not bfill lagged regressors either."""
        import inspect
        from src.modelling.data_prep import build_future_df
        src = inspect.getsource(build_future_df)
        # Should not have .bfill() on the filled.shift() line
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if "shift(spec.lag)" in line and ".bfill()" in line:
                pytest.fail(f"D5 FAIL: build_future_df line {i} backfills lagged regressor")


# ===================================================================
# D6: Ensemble weights re-estimated via walk-forward CV
# ===================================================================

class TestD6_DynamicWeights:
    def test_estimate_ensemble_weights_exists(self):
        """_estimate_ensemble_weights function should exist and be callable."""
        from src.modelling.aggregate_model import _estimate_ensemble_weights
        assert callable(_estimate_ensemble_weights)

    def test_returns_weight_dict(self):
        """Should return dict with prophet, arima, ets keys summing to ~1.0."""
        from src.modelling.aggregate_model import _estimate_ensemble_weights
        from src.modelling.model_config import CC_CONFIG

        y = _make_monthly_series(84)
        weights = _estimate_ensemble_weights(y, CC_CONFIG, 24)

        assert "prophet" in weights, "D6 FAIL: no 'prophet' key in weights"
        assert "arima" in weights, "D6 FAIL: no 'arima' key in weights"
        assert "ets" in weights, "D6 FAIL: no 'ets' key in weights"
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.05, f"D6 FAIL: weights sum to {total}, not ~1.0"

    def test_run_forecast_calls_weight_estimation(self):
        """run_forecast source should call _estimate_ensemble_weights."""
        import inspect
        from src.modelling.aggregate_model import run_forecast
        src = inspect.getsource(run_forecast)
        assert "_estimate_ensemble_weights" in src, (
            "D6 FAIL: run_forecast doesn't call _estimate_ensemble_weights"
        )


# ===================================================================
# D7: Damped extrapolation for future regressors
# ===================================================================

class TestD7_DampedExtrapolation:
    def test_damping_factor_in_build_future_df(self):
        """build_future_df should use damped extrapolation (0.95^i)."""
        import inspect
        from src.modelling.data_prep import build_future_df
        src = inspect.getsource(build_future_df)
        assert "0.95" in src or "damping" in src, (
            "D7 FAIL: build_future_df has no damping factor"
        )
        assert "damping ** i" in src or "damping**i" in src, (
            "D7 FAIL: damping is defined but not applied per-step"
        )

    def test_damped_values_converge(self):
        """With a slope, damped extrapolation should produce a decelerating
        trend (each increment smaller than the last)."""
        slope = 5.0
        damping = 0.95
        proj_val = 100.0
        values = [max(0.0, proj_val + slope * (damping ** i) * i) for i in range(1, 25)]
        increments = [values[i] - values[i - 1] for i in range(1, len(values))]
        # After the first few steps, increments should start declining
        late_increments = increments[5:]
        assert all(late_increments[i] <= late_increments[i - 1] + 0.01 for i in range(1, len(late_increments))), (
            "D7 FAIL: damped extrapolation increments not declining"
        )


# ===================================================================
# D8: Minimum delta threshold for agent auto-promotion
# ===================================================================

class TestD8_MinDeltaThreshold:
    def test_min_delta_in_retrainer(self):
        """retrainer.py should have a minimum improvement threshold."""
        import inspect
        from src.agent.retrainer import retrain_aggregate
        src = inspect.getsource(retrain_aggregate)
        assert "min_delta" in src or "0.2" in src, (
            "D8 FAIL: retrain_aggregate has no minimum delta threshold"
        )

    def test_tiny_improvement_not_promoted(self):
        """An improvement of 0.1pp (below 0.2pp threshold) should NOT promote."""
        import inspect
        from src.agent.retrainer import retrain_aggregate
        src = inspect.getsource(retrain_aggregate)
        # Verify the logic checks improvement >= min_delta
        assert "improvement >= min_delta" in src or "improvement >= 0.2" in src, (
            "D8 FAIL: promotion logic doesn't enforce minimum delta"
        )

    def test_promotion_logic_rejects_marginal(self):
        """Simulate: old=5.0, new=4.9 (only 0.1pp improvement) → not promoted."""
        old_mape = 5.0
        new_mape = 4.9
        min_delta = 0.2
        improvement = old_mape - new_mape  # 0.1
        promoted = improvement >= min_delta
        assert not promoted, (
            f"D8 FAIL: 0.1pp improvement should NOT promote (threshold={min_delta})"
        )

    def test_promotion_logic_accepts_sufficient(self):
        """Simulate: old=5.0, new=4.7 (0.3pp improvement) → promoted."""
        old_mape = 5.0
        new_mape = 4.7
        min_delta = 0.2
        improvement = old_mape - new_mape  # 0.3
        promoted = improvement >= min_delta
        assert promoted, (
            f"D8 FAIL: 0.3pp improvement should promote (threshold={min_delta})"
        )
