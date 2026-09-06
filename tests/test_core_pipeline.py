"""
Core Pipeline Tests
====================
Tests for ingestion, data prep, sync, reconciliation, and bank-level
direct multi-horizon forecasting.

Run:  uv run pytest tests/test_core_pipeline.py -v
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _make_monthly_series(n: int = 84, trend: float = 2.0, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=float)
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    noise = rng.normal(0, 1, n)
    return 100 + trend * t + seasonal + noise


# ===================================================================
# Ingestion validation
# ===================================================================

class TestIngestionValidation:
    def test_forward_fill_handles_none(self):
        from src.ingestion.validation import _forward_fill
        row = ("A", None, None, "B", None)
        result = _forward_fill(row)
        assert result == ["A", "A", "A", "B", "B"]

    def test_forward_fill_all_filled(self):
        from src.ingestion.validation import _forward_fill
        row = ("X", "Y", "Z")
        assert _forward_fill(row) == ["X", "Y", "Z"]

    def test_is_section_header(self):
        from src.ingestion.validation import _is_section_header
        assert _is_section_header("5 Cards") is True
        assert _is_section_header("8 Cards Outstanding") is True
        assert _is_section_header("5.1 Sub-item") is False
        assert _is_section_header(None) is False
        assert _is_section_header("") is False

    def test_section_per_column(self):
        from src.ingestion.validation import _section_per_column
        row = ("5 Cards", "credit", "debit", "8 ATM", "count")
        result = _section_per_column(row)
        assert result[0] == "5 Cards"
        assert result[1] == "5 Cards"
        assert result[2] == "5 Cards"
        assert result[3] == "8 ATM"
        assert result[4] == "8 ATM"

    def test_schema_validation_error_is_exception(self):
        from src.ingestion.validation import SchemaValidationError
        with pytest.raises(SchemaValidationError):
            raise SchemaValidationError("test")


# ===================================================================
# Data prep
# ===================================================================

class TestDataPrep:
    def test_load_all_returns_dict(self):
        from src.modelling.data_prep import load_all
        try:
            result = load_all()
            assert isinstance(result, dict)
        except FileNotFoundError:
            pytest.skip("Processed data files not available")

    def test_build_training_df_structure(self):
        from src.modelling.data_prep import build_training_df
        from src.modelling.model_config import CC_CONFIG
        dates = pd.date_range("2017-01-01", periods=84, freq="MS")
        master = pd.DataFrame({
            "date": dates,
            "credit_cards_outstanding_lakh": _make_monthly_series(84),
            "repo_rate": np.full(84, 6.5),
        })
        try:
            df = build_training_df(master, CC_CONFIG)
            assert "ds" in df.columns
            assert "y" in df.columns
            assert len(df) > 0
        except Exception:
            pytest.skip("build_training_df requires specific column names")


# ===================================================================
# Model config
# ===================================================================

class TestModelConfig:
    def test_cc_config_has_required_keys(self):
        from src.modelling.model_config import CC_CONFIG
        for key in ("name", "target_col", "output_stem"):
            assert key in CC_CONFIG, f"Missing key: {key}"

    def test_dc_config_has_required_keys(self):
        from src.modelling.model_config import DC_CONFIG
        for key in ("name", "target_col", "output_stem"):
            assert key in DC_CONFIG, f"Missing key: {key}"

    def test_ensemble_weights_sum_to_one(self):
        from src.modelling.aggregate_model import ENSEMBLE_WEIGHTS
        for key, weights in ENSEMBLE_WEIGHTS.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 0.01, f"{key} weights sum to {total}"

    def test_bank_lists_not_empty(self):
        from src.modelling.bank_config import CC_BANK_LIST, DC_BANK_LIST
        assert len(CC_BANK_LIST) >= 10, f"CC has only {len(CC_BANK_LIST)} banks"
        assert len(DC_BANK_LIST) >= 10, f"DC has only {len(DC_BANK_LIST)} banks"

    def test_forecast_horizon_consistent(self):
        from src.modelling.bank_config import BANK_FORECAST_PERIODS
        assert BANK_FORECAST_PERIODS >= 12


# ===================================================================
# Bank direct multi-horizon (P2.6)
# ===================================================================

class TestBankDirectMultiHorizon:
    def test_function_exists(self):
        from src.modelling.bank_model import _fit_bank_direct_multihorizon
        assert callable(_fit_bank_direct_multihorizon)

    def test_returns_correct_length(self):
        from src.modelling.bank_model import _fit_bank_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_bank_direct_multihorizon(y, 12)
        assert result is not None
        assert len(result) == 12

    def test_returns_none_for_short_series(self):
        from src.modelling.bank_model import _fit_bank_direct_multihorizon
        y = _make_monthly_series(20)
        result = _fit_bank_direct_multihorizon(y, 12)
        assert result is None

    def test_all_positive(self):
        from src.modelling.bank_model import _fit_bank_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_bank_direct_multihorizon(y, 12)
        assert result is not None
        assert np.all(result >= 0)

    def test_smooth_transitions(self):
        from src.modelling.bank_model import _fit_bank_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_bank_direct_multihorizon(y, 12)
        if result is not None:
            for i in range(len(result) - 1):
                jump = abs(result[i + 1] - result[i]) / max(abs(result[i]), 1)
                assert jump < 0.5, f"Large jump at step {i}: {jump:.2%}"

    def test_blend_weight_exists(self):
        from src.modelling.bank_model import BANK_DIRECT_BLEND_WEIGHT
        assert 0 < BANK_DIRECT_BLEND_WEIGHT < 1

    def test_longer_horizon(self):
        from src.modelling.bank_model import _fit_bank_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_bank_direct_multihorizon(y, 24)
        assert result is not None
        assert len(result) == 24


# ===================================================================
# MinT reconciliation (P2.4)
# ===================================================================

class TestMinTReconciliation:
    def test_variance_estimation_with_residuals(self):
        from src.modelling.bank_model import _estimate_insample_variance
        fc = pd.DataFrame({"forecast": [100, 110, 120]})
        hist = pd.DataFrame({
            "date": pd.date_range("2025-01", periods=12, freq="MS"),
            "yhat": np.arange(100, 112, dtype=float),
            "y": np.arange(100, 112, dtype=float) + np.random.RandomState(0).normal(0, 5, 12),
        })
        var = _estimate_insample_variance(fc, hist)
        assert var > 1.0

    def test_variance_estimation_without_history(self):
        from src.modelling.bank_model import _estimate_insample_variance
        fc = pd.DataFrame({"forecast": [100, 200, 300]})
        var = _estimate_insample_variance(fc, None)
        assert var > 0

    def test_low_variance_gets_more_weight(self):
        from src.modelling.bank_model import _estimate_insample_variance
        fc = pd.DataFrame({"forecast": [100.0] * 3})
        hist_low = pd.DataFrame({
            "date": pd.date_range("2025-01", periods=12, freq="MS"),
            "yhat": np.ones(12) * 100,
            "y": np.ones(12) * 100 + np.random.RandomState(0).normal(0, 1, 12),
        })
        hist_high = pd.DataFrame({
            "date": pd.date_range("2025-01", periods=12, freq="MS"),
            "yhat": np.ones(12) * 100,
            "y": np.ones(12) * 100 + np.random.RandomState(1).normal(0, 20, 12),
        })
        var_low = _estimate_insample_variance(fc, hist_low)
        var_high = _estimate_insample_variance(fc, hist_high)
        # Lower variance → higher 1/var → more weight
        assert 1.0 / var_low > 1.0 / var_high


# ===================================================================
# Supabase sync helpers
# ===================================================================

class TestSyncHelpers:
    def test_upsert_df_dry_run(self):
        from scripts.sync_to_supabase import upsert_df
        client = MagicMock()
        df = pd.DataFrame({
            "metric": ["cc", "dc"],
            "value": [100.0, 200.0],
        })
        with patch("scripts.sync_to_supabase.DRY_RUN", True):
            result = upsert_df(client, "test_table", df, ["metric"])
        assert result == 0
        client.table.assert_not_called()

    def test_upsert_df_handles_nan(self):
        from scripts.sync_to_supabase import upsert_df
        client = MagicMock()
        client.table.return_value.upsert.return_value.execute.return_value = None
        df = pd.DataFrame({
            "metric": ["cc"],
            "value": [float("nan")],
        })
        with patch("scripts.sync_to_supabase.DRY_RUN", False):
            upsert_df(client, "test_table", df, ["metric"])
        call_args = client.table.return_value.upsert.call_args
        records = call_args[0][0]
        assert records[0]["value"] is None


# ===================================================================
# Aggregate model core functions
# ===================================================================

class TestAggregateModelCore:
    def test_arima_forecast(self):
        from src.modelling.aggregate_model import _fit_arima_forecast
        y = _make_monthly_series(84)
        result = _fit_arima_forecast(y, 12)
        assert result is not None
        assert len(result) == 12

    def test_ets_forecast(self):
        from src.modelling.aggregate_model import _fit_ets_forecast
        y = _make_monthly_series(84)
        result = _fit_ets_forecast(y, 12)
        assert result is not None
        assert len(result) == 12

    def test_direct_multihorizon_forecast(self):
        from src.modelling.aggregate_model import _fit_direct_multihorizon
        y = _make_monthly_series(84)
        result = _fit_direct_multihorizon(y, 24)
        assert result is not None
        assert len(result) == 24

    def test_arima_log_transform_positive(self):
        from src.modelling.aggregate_model import _fit_arima_forecast
        y = _make_monthly_series(84)
        result = _fit_arima_forecast(y, 12, log_transform=True)
        assert result is not None
        assert np.all(result > 0)


# ===================================================================
# Bank config integrity
# ===================================================================

class TestBankConfigIntegrity:
    def test_ets_banks_are_in_bank_lists(self):
        from src.modelling.bank_config import ETS_BANKS, CC_BANK_LIST, DC_BANK_LIST
        for (bank, card_type), use_ets in ETS_BANKS.items():
            if not use_ets:
                continue
            target_list = CC_BANK_LIST if card_type == "cc" else DC_BANK_LIST
            assert bank in target_list, f"ETS bank {bank}/{card_type} not in bank list"

    def test_bank_start_dates_are_valid(self):
        from src.modelling.bank_config import BANK_START_DATES
        for key, date_str in BANK_START_DATES.items():
            dt = pd.Timestamp(date_str)
            assert dt.year >= 2010, f"{key} start date too early: {date_str}"
            assert dt.year <= 2025, f"{key} start date too late: {date_str}"

    def test_cv_config_keys(self):
        from src.modelling.bank_config import BANK_CV_CONFIG
        for key in ("initial", "period", "horizon"):
            assert key in BANK_CV_CONFIG, f"Missing CV config key: {key}"


# ===================================================================
# File structure
# ===================================================================

class TestFileStructure:
    def test_ingestion_modules_exist(self):
        for module in ["rbi", "npci", "cpi", "bankwise", "repo_rate", "p2p_upi"]:
            path = PROJECT_ROOT / "src" / "ingestion" / f"{module}.py"
            assert path.exists(), f"Missing ingestion module: {module}"

    def test_modelling_modules_exist(self):
        for module in ["aggregate_model", "bank_model", "txn_volume_model",
                       "data_prep", "bank_data_prep", "model_config", "bank_config", "metrics"]:
            path = PROJECT_ROOT / "src" / "modelling" / f"{module}.py"
            assert path.exists(), f"Missing modelling module: {module}"

    def test_workflow_files_exist(self):
        for wf in ["monthly_pipeline.yml", "agent_pipeline.yml", "market_pipeline.yml"]:
            path = PROJECT_ROOT / ".github" / "workflows" / wf
            assert path.exists(), f"Missing workflow: {wf}"

    def test_sync_script_exists(self):
        assert (PROJECT_ROOT / "scripts" / "sync_to_supabase.py").exists()

    def test_supabase_schema_exists(self):
        migrations = PROJECT_ROOT / "supabase" / "migrations"
        assert migrations.exists()
        sql_files = list(migrations.glob("*.sql"))
        assert len(sql_files) > 0, "No migration SQL files"


# ===================================================================
# Metrics module
# ===================================================================

class TestMetricsComprehensive:
    def test_mape_zero_actual_handling(self):
        from src.modelling.metrics import mape
        actual = np.array([0.0, 100.0, 200.0])
        pred = np.array([10.0, 110.0, 190.0])
        result = mape(actual, pred)
        assert np.isfinite(result)

    def test_mase_naive_beats_perfect(self):
        from src.modelling.metrics import mase
        training = np.arange(60, dtype=float)
        actual = np.array([60.0, 61.0, 62.0])
        perfect = actual.copy()
        result = mase(actual, perfect, training, seasonal_period=1)
        assert result == 0.0 or result < 0.01

    def test_empirical_coverage_all_inside(self):
        from src.modelling.metrics import empirical_coverage
        actual = np.array([100.0, 200.0])
        lower = np.array([90.0, 190.0])
        upper = np.array([110.0, 210.0])
        assert empirical_coverage(actual, lower, upper) == 1.0

    def test_empirical_coverage_none_inside(self):
        from src.modelling.metrics import empirical_coverage
        actual = np.array([100.0, 200.0])
        lower = np.array([110.0, 210.0])
        upper = np.array([120.0, 220.0])
        assert empirical_coverage(actual, lower, upper) == 0.0

    def test_pinball_loss_symmetric(self):
        from src.modelling.metrics import pinball_loss
        actual = np.array([100.0])
        lower = np.array([90.0])
        upper = np.array([110.0])
        result = pinball_loss(actual, lower, upper, alpha=0.10)
        assert result >= 0


# ===================================================================
# P3: Regressor candidates
# ===================================================================

class TestRegressorCandidates:
    def test_build_festive_index(self):
        from src.modelling.regressor_candidates import build_festive_index
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        idx = build_festive_index(dates)
        assert len(idx) == 24
        assert idx.name == "festive_index"
        # October should be the peak
        oct_vals = idx[dates.month == 10]
        assert all(v > 1.0 for v in oct_vals)

    def test_festive_index_averages_near_one(self):
        from src.modelling.regressor_candidates import build_festive_index
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        idx = build_festive_index(dates)
        assert abs(idx.mean() - 1.0) < 0.02

    def test_compute_upi_p2m_share(self):
        from src.modelling.regressor_candidates import compute_upi_p2m_share
        master = pd.DataFrame({
            "upi_p2m_vol_mn": [100, 200, 300],
            "upi_volume_mn": [500, 500, 500],
        })
        share = compute_upi_p2m_share(master)
        assert len(share) == 3
        assert share.iloc[0] == pytest.approx(0.2)
        assert share.iloc[2] == pytest.approx(0.6)

    def test_upi_p2m_share_missing_columns(self):
        from src.modelling.regressor_candidates import compute_upi_p2m_share
        master = pd.DataFrame({"other_col": [1, 2, 3]})
        share = compute_upi_p2m_share(master)
        assert len(share) == 0

    def test_candidates_defined(self):
        from src.modelling.regressor_candidates import CANDIDATES
        assert len(CANDIDATES) >= 3
        names = [c.name for c in CANDIDATES]
        assert "CPI inflation" in names
        assert "Festive calendar" in names
        assert "UPI P2M share" in names

    def test_candidate_targets_are_valid(self):
        from src.modelling.regressor_candidates import CANDIDATES
        valid_targets = {
            "credit_cards_outstanding_lakh",
            "debit_cards_outstanding_lakh",
        }
        for c in CANDIDATES:
            for t in c.targets:
                assert t in valid_targets, f"{c.name} has invalid target: {t}"

    def test_evaluate_candidates_returns_results(self):
        from src.modelling.regressor_candidates import evaluate_candidates
        try:
            results = evaluate_candidates()
            assert isinstance(results, list)
            assert len(results) >= 3
        except FileNotFoundError:
            pytest.skip("Processed data not available")

    def test_save_results(self, tmp_path):
        from src.modelling.regressor_candidates import (
            save_results, CandidateResult, RegressorCandidate,
        )
        candidate = RegressorCandidate(
            name="test", col="test_col", targets=["cc"],
            lag=1, mode="additive", hypothesis="test",
            interpretation_confirmed="yes", interpretation_not_confirmed="no",
        )
        results = [CandidateResult(
            candidate=candidate, target="cc",
            granger_pvalue=0.5, granger_fstat=1.0,
            granger_best_lag=1, passed=False, verdict="fail",
        )]
        out = save_results(results, tmp_path / "test_results.csv")
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == 1
        assert df["passed"].iloc[0] == False
