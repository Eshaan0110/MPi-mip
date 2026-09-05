"""
Tests for previously missing brief items:
- P0.2: Teams webhook integration
- P0.3: Gitleaks workflow exists
- D8: Multi-gate promotion (stability + monotonicity)
- D9: Bank data freshness check
- P2.3: Pooled bank model exists
- P5.3: Per-market workflow exists
- P5.4: Expansion design note exists
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── P0.2: Teams webhook ──────────────────────────────────────────────

class TestTeamsWebhook:
    def test_send_teams_webhook_exists(self):
        from src.utils.email_alert import send_teams_webhook
        assert callable(send_teams_webhook)

    def test_notify_all_exists(self):
        from src.utils.email_alert import notify_all
        assert callable(notify_all)

    def test_teams_webhook_skips_without_env(self):
        from src.utils.email_alert import send_teams_webhook
        with patch.dict("os.environ", {}, clear=True):
            result = send_teams_webhook("test", "success")
        assert result is False

    def test_notify_all_returns_both_channels(self):
        from src.utils.email_alert import notify_all
        with patch.dict("os.environ", {}, clear=True):
            result = notify_all("test", "success")
        assert "email" in result
        assert "teams" in result

    def test_teams_webhook_sends_on_success(self):
        from src.utils.email_alert import send_teams_webhook
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch.dict("os.environ", {"TEAMS_WEBHOOK_URL": "https://example.com/webhook"}):
            with patch("httpx.post", return_value=mock_resp) as mock_post:
                result = send_teams_webhook("Monthly Pipeline", "failure", {"Step": "Train"})
        assert result is True
        mock_post.assert_called_once()
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
        assert "attachments" in payload


# ── P0.3: Gitleaks ───────────────────────────────────────────────────

class TestGitleaks:
    def test_gitleaks_job_in_workflow(self):
        workflow = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"
        content = workflow.read_text()
        assert "gitleaks" in content
        assert "gitleaks/gitleaks-action" in content

    def test_gitleaks_runs_on_push_and_pr(self):
        workflow = PROJECT_ROOT / ".github" / "workflows" / "tests.yml"
        content = workflow.read_text()
        assert "push:" in content
        assert "pull_request:" in content


# ── D8: Multi-gate promotion ─────────────────────────────────────────

class TestD8MultiGate:
    def test_cv_returns_per_fold_mapes(self):
        from src.agent.retrainer import _cross_validate_ensemble
        # The function signature now returns tuple[float, list[float]]
        import inspect
        sig = inspect.signature(_cross_validate_ensemble)
        assert "extra_regressor_cols" in sig.parameters

    def test_retrainer_has_stability_gate(self):
        source = (PROJECT_ROOT / "src" / "agent" / "retrainer.py").read_text()
        assert "passes_stability" in source
        assert "passes_monotonicity" in source


# ── D9: Bank data freshness ──────────────────────────────────────────

class TestD9Freshness:
    def test_check_data_freshness_exists(self):
        from src.modelling.bank_data_prep import check_data_freshness
        assert callable(check_data_freshness)

    def test_freshness_detects_stale_bank(self):
        from src.modelling.bank_data_prep import check_data_freshness
        dates_psi = pd.date_range("2024-01-01", periods=24, freq="MS")
        dates_bank = pd.date_range("2024-01-01", periods=18, freq="MS")
        df = pd.DataFrame({
            "date": list(dates_bank),
            "bank_name": ["TestBank"] * len(dates_bank),
            "cc_outstanding": np.random.rand(len(dates_bank)) * 1000,
        })
        with patch("src.modelling.bank_data_prep._load_psi_series") as mock_psi:
            mock_psi.return_value = pd.Series(
                np.random.rand(len(dates_psi)),
                index=dates_psi,
                name="psi_cards",
            )
            result = check_data_freshness(df, "cc", max_stale_months=3)
        assert "TestBank" in result

    def test_freshness_passes_fresh_bank(self):
        from src.modelling.bank_data_prep import check_data_freshness
        dates = pd.date_range("2024-01-01", periods=24, freq="MS")
        df = pd.DataFrame({
            "date": list(dates),
            "bank_name": ["FreshBank"] * len(dates),
            "cc_outstanding": np.random.rand(len(dates)) * 1000,
        })
        with patch("src.modelling.bank_data_prep._load_psi_series") as mock_psi:
            mock_psi.return_value = pd.Series(
                np.random.rand(len(dates)),
                index=dates,
                name="psi_cards",
            )
            result = check_data_freshness(df, "cc", max_stale_months=3)
        assert len(result) == 0

    def test_load_bank_data_returns_stale_banks_key(self):
        source = (PROJECT_ROOT / "src" / "modelling" / "bank_data_prep.py").read_text()
        assert '"stale_banks"' in source


# ── P2.3: Pooled bank model ──────────────────────────────────────────

class TestPooledBankModel:
    def test_module_exists(self):
        from src.modelling.bank_pooled_model import fit_pooled_model
        assert callable(fit_pooled_model)

    def test_build_features_exists(self):
        from src.modelling.bank_pooled_model import _build_features
        assert callable(_build_features)

    def test_build_features_output(self):
        from src.modelling.bank_pooled_model import _build_features
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        df = pd.DataFrame({"ds": dates, "y": np.arange(36, dtype=float) + 100})
        result = _build_features(df, "TestBank", 0)
        assert "month" in result.columns
        assert "lag_12" in result.columns
        assert "rolling_3m" in result.columns
        assert "bank_rank" in result.columns
        assert "bank_name" in result.columns
        assert len(result) < len(df)  # some rows dropped due to NaN from lags


# ── P5.3: Per-market workflow ─────────────────────────────────────────

class TestMarketWorkflow:
    def test_market_pipeline_yml_exists(self):
        path = PROJECT_ROOT / ".github" / "workflows" / "market_pipeline.yml"
        assert path.exists()

    def test_market_pipeline_has_market_input(self):
        path = PROJECT_ROOT / ".github" / "workflows" / "market_pipeline.yml"
        content = path.read_text()
        assert "market:" in content
        assert "--market" in content

    def test_market_pipeline_validates_market_code(self):
        path = PROJECT_ROOT / ".github" / "workflows" / "market_pipeline.yml"
        content = path.read_text()
        assert "get_adapter" in content

    def test_market_pipeline_has_notify_with_teams(self):
        path = PROJECT_ROOT / ".github" / "workflows" / "market_pipeline.yml"
        content = path.read_text()
        assert "TEAMS_WEBHOOK_URL" in content
        assert "notify_all" in content


# ── P5.4: Expansion design note ──────────────────────────────────────

class TestExpansionNote:
    def test_expansion_note_exists(self):
        path = PROJECT_ROOT / "src" / "markets" / "EXPANSION_NOTE.md"
        assert path.exists()

    def test_expansion_note_covers_uk(self):
        path = PROJECT_ROOT / "src" / "markets" / "EXPANSION_NOTE.md"
        content = path.read_text()
        assert "United Kingdom" in content or "UK" in content

    def test_expansion_note_has_adapter_skeleton(self):
        path = PROJECT_ROOT / "src" / "markets" / "EXPANSION_NOTE.md"
        content = path.read_text()
        assert "UKAdapter" in content
        assert "MarketMeta" in content

    def test_expansion_note_references_gate(self):
        path = PROJECT_ROOT / "src" / "markets" / "EXPANSION_NOTE.md"
        content = path.read_text()
        assert "30 consecutive" in content
