"""
P3 Trust Tests
===============
Tests for P3.1 (vintage handling), P3.2 (agent ablation),
P3.3 (regressor candidates), P3.4 (data versioning).

Run:  uv run pytest tests/test_p3_trust.py -v
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ===================================================================
# P3.4: Data versioning
# ===================================================================

class TestP3_4_DataVersioning:
    def test_snapshot_creates_directory(self, tmp_path):
        import src.modelling.data_versioning as dv
        # Patch paths
        orig_processed = dv._PROCESSED
        orig_vintages = dv._VINTAGES
        dv._PROCESSED = tmp_path / "processed"
        dv._VINTAGES = tmp_path / "vintages"
        dv._PROCESSED.mkdir()

        # Create fake files
        pd.DataFrame({"a": [1, 2]}).to_parquet(dv._PROCESSED / "test.parquet")
        pd.DataFrame({"b": [3, 4]}).to_csv(dv._PROCESSED / "test.csv", index=False)

        try:
            result = dv.snapshot_processed("2026-08-28")
            assert result.exists()
            assert (result / "test.parquet").exists()
            assert (result / "test.csv").exists()
        finally:
            dv._PROCESSED = orig_processed
            dv._VINTAGES = orig_vintages

    def test_list_vintages(self, tmp_path):
        import src.modelling.data_versioning as dv
        orig = dv._VINTAGES
        dv._VINTAGES = tmp_path / "vintages"
        dv._VINTAGES.mkdir()
        (dv._VINTAGES / "2026-07-01").mkdir()
        (dv._VINTAGES / "2026-08-01").mkdir()

        try:
            vintages = dv.list_vintages()
            assert vintages == ["2026-08-01", "2026-07-01"]
        finally:
            dv._VINTAGES = orig

    def test_load_vintage(self, tmp_path):
        import src.modelling.data_versioning as dv
        orig = dv._VINTAGES
        dv._VINTAGES = tmp_path / "vintages"
        vintage_dir = dv._VINTAGES / "2026-08-01"
        vintage_dir.mkdir(parents=True)
        pd.DataFrame({"x": [10, 20]}).to_parquet(vintage_dir / "test.parquet")

        try:
            df = dv.load_vintage("2026-08-01", "test")
            assert len(df) == 2
            assert df["x"].tolist() == [10, 20]
        finally:
            dv._VINTAGES = orig


# ===================================================================
# P3.1: Vintage scoring
# ===================================================================

class TestP3_1_VintageScoring:
    def test_save_forecast_vintage(self, tmp_path):
        import src.modelling.vintage_scoring as vs
        orig = vs._FORECAST_VINTAGES
        vs._FORECAST_VINTAGES = tmp_path / "fc_vintages"

        try:
            fc = pd.DataFrame({
                "date": pd.date_range("2026-06-01", periods=12, freq="MS"),
                "forecast": np.arange(100, 112, dtype=float),
            })
            result = vs.save_forecast_vintage(fc, "forecast_cc", "2026-08-28")
            assert result.exists()
            assert (result / "forecast_cc.parquet").exists()
        finally:
            vs._FORECAST_VINTAGES = orig

    def test_save_with_metadata(self, tmp_path):
        import src.modelling.vintage_scoring as vs
        orig = vs._FORECAST_VINTAGES
        vs._FORECAST_VINTAGES = tmp_path / "fc_vintages"

        try:
            fc = pd.DataFrame({
                "date": pd.date_range("2026-06-01", periods=6, freq="MS"),
                "forecast": [100.0] * 6,
            })
            meta = {"weights": {"prophet": 0.3, "arima": 0.7}}
            result = vs.save_forecast_vintage(fc, "forecast_cc", "2026-08-28", meta)
            meta_path = result / "forecast_cc_meta.json"
            assert meta_path.exists()
            loaded = json.loads(meta_path.read_text())
            assert loaded["weights"]["prophet"] == 0.3
        finally:
            vs._FORECAST_VINTAGES = orig

    def test_score_all_vintages_returns_list(self, tmp_path):
        import src.modelling.vintage_scoring as vs
        orig = vs._FORECAST_VINTAGES
        vs._FORECAST_VINTAGES = tmp_path / "fc_vintages"

        try:
            results = vs.score_all_vintages("forecast_cc", "credit_cards_outstanding_lakh")
            assert isinstance(results, list)
        finally:
            vs._FORECAST_VINTAGES = orig


# ===================================================================
# P3.3: Regressor candidates
# ===================================================================

class TestP3_3_RegressorCandidates:
    def test_festive_index_shape(self):
        from src.modelling.regressor_candidates import build_festive_index
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        idx = build_festive_index(dates)
        assert len(idx) == 24
        assert idx.mean() == pytest.approx(1.0, abs=0.02)

    def test_festive_october_peak(self):
        from src.modelling.regressor_candidates import build_festive_index
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        idx = build_festive_index(dates)
        assert idx.iloc[9] == 1.15  # October is the peak

    def test_upi_p2m_share_computation(self):
        from src.modelling.regressor_candidates import compute_upi_p2m_share
        master = pd.DataFrame({
            "upi_p2m_vol_mn": [100.0, 200.0, 300.0],
            "upi_volume_mn": [500.0, 500.0, 500.0],
        })
        share = compute_upi_p2m_share(master)
        assert len(share) == 3
        assert share.iloc[0] == pytest.approx(0.2)
        assert share.iloc[2] == pytest.approx(0.6)

    def test_candidates_list_nonempty(self):
        from src.modelling.regressor_candidates import CANDIDATES
        assert len(CANDIDATES) >= 3

    def test_candidate_result_dataclass(self):
        from src.modelling.regressor_candidates import CandidateResult, CANDIDATES
        cr = CandidateResult(
            candidate=CANDIDATES[0],
            target="credit_cards_outstanding_lakh",
            granger_pvalue=0.03,
            granger_fstat=5.2,
            granger_best_lag=3,
            passed=True,
            verdict="CONFIRMED",
        )
        assert cr.passed is True


# ===================================================================
# P3.2: Agent ablation
# ===================================================================

class TestP3_2_AgentAblation:
    def test_protocol_registered(self):
        from src.modelling.agent_ablation import ABLATION_PROTOCOL
        assert ABLATION_PROTOCOL["min_months"] == 12
        assert ABLATION_PROTOCOL["decision_threshold_pp"] == 0.2
        assert ABLATION_PROTOCOL["primary_metric"] == "mape"

    def test_ablation_arm_dataclass(self):
        from src.modelling.agent_ablation import AblationArm
        arm = AblationArm(name="full", regressors=["repo_rate"], mape=4.5)
        assert arm.name == "full"
        assert arm.mape == 4.5

    def test_registry_io(self, tmp_path):
        from src.modelling.agent_ablation import _load_registry, _save_registry, _REGISTRY
        import src.modelling.agent_ablation as aa
        orig = aa._REGISTRY
        aa._REGISTRY = tmp_path / "test_registry.json"

        try:
            _save_registry([{"test": True}])
            loaded = _load_registry()
            assert len(loaded) == 1
            assert loaded[0]["test"] is True
        finally:
            aa._REGISTRY = orig

    def test_ablation_status_not_started(self, tmp_path):
        import src.modelling.agent_ablation as aa
        orig = aa._REGISTRY
        aa._REGISTRY = tmp_path / "nonexistent.json"

        try:
            status = aa.get_ablation_status()
            assert status["status"] == "not_started"
            assert status["runs"] == 0
        finally:
            aa._REGISTRY = orig

    def test_walk_forward_cv(self):
        from src.modelling.agent_ablation import _walk_forward_cv
        rng = np.random.RandomState(42)
        y = 100 + np.arange(84, dtype=float) + rng.normal(0, 2, 84)
        metrics = _walk_forward_cv(y, None, initial=48, step=6, horizon=12)
        assert "mape" in metrics
        assert "n_folds" in metrics
        assert metrics["n_folds"] >= 2
        assert metrics["mape"] > 0
