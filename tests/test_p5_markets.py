"""
P5 — International readiness tests.
Validates MarketAdapter contract, India adapter, and the registry.
"""

import pytest
from src.markets.base import MarketAdapter, MarketMeta, CardType, DataSource, StructuralEvent
from src.markets.registry import get_adapter, list_markets
from src.markets.india import IndiaAdapter


class TestMarketRegistry:
    def test_india_registered(self):
        assert "IN" in list_markets()

    def test_get_india(self):
        adapter = get_adapter("IN")
        assert isinstance(adapter, MarketAdapter)
        assert adapter.meta.code == "IN"

    def test_case_insensitive(self):
        assert get_adapter("in").meta.code == "IN"

    def test_unknown_market_raises(self):
        with pytest.raises(KeyError, match="Unknown market"):
            get_adapter("XX")

    def test_list_markets_sorted(self):
        markets = list_markets()
        assert markets == sorted(markets)


class TestIndiaAdapter:
    @pytest.fixture
    def adapter(self):
        return IndiaAdapter()

    def test_meta_fields(self, adapter):
        m = adapter.meta
        assert m.code == "IN"
        assert m.currency == "INR"
        assert m.currency_symbol == "₹"
        assert m.unit_label == "Lakhs"
        assert m.unit_divisor == 10.0
        assert m.timezone == "Asia/Kolkata"

    def test_card_types(self, adapter):
        codes = [ct.code for ct in adapter.meta.card_types]
        assert "CC" in codes
        assert "DC" in codes

    def test_data_sources_non_empty(self, adapter):
        assert len(adapter.meta.data_sources) >= 4

    def test_structural_events(self, adapter):
        events = adapter.meta.structural_events
        assert len(events) >= 6
        names = [e.name for e in events]
        assert "Demonetisation" in names

    def test_cc_bank_allowlist(self, adapter):
        banks = adapter.get_bank_allowlist("CC")
        assert len(banks) == 12
        assert "HDFC Bank" in banks

    def test_dc_bank_allowlist(self, adapter):
        banks = adapter.get_bank_allowlist("DC")
        assert len(banks) == 16
        assert "State Bank of India" in banks

    def test_unknown_card_type_returns_empty(self, adapter):
        assert adapter.get_bank_allowlist("XX") == []

    def test_training_start_cc(self, adapter):
        assert adapter.get_training_start("cc_outstanding") == "2013-01-01"

    def test_training_start_dc_is_none(self, adapter):
        assert adapter.get_training_start("dc_outstanding") is None

    def test_model_config_has_ensemble(self, adapter):
        cfg = adapter.get_model_config("cc_outstanding")
        assert "ensemble_members" in cfg
        assert len(cfg["ensemble_members"]) == 5
        assert "prophet" in cfg["ensemble_members"]

    def test_model_config_has_cv(self, adapter):
        cfg = adapter.get_model_config("cc_outstanding")
        assert "cv" in cfg
        assert "initial" in cfg["cv"]

    def test_regressors(self, adapter):
        regs = adapter.get_regressor_columns("cc_outstanding")
        assert "repo_rate" in regs

    def test_pipeline_steps(self, adapter):
        steps = adapter.get_pipeline_steps()
        assert len(steps) >= 5
        names = [s["name"] for s in steps]
        assert "Ingestion" in names
        assert any("Aggregate" in n for n in names)

    def test_pipeline_steps_have_required_keys(self, adapter):
        for step in adapter.get_pipeline_steps():
            assert "name" in step
            assert "cv_flag" in step
            assert "skip_flag" in step
            assert "module" in step or "script" in step

    def test_bank_lists_return_copies(self, adapter):
        """Ensure returned lists are copies, not mutable references."""
        a = adapter.get_bank_allowlist("CC")
        b = adapter.get_bank_allowlist("CC")
        assert a == b
        a.append("Test Bank")
        assert "Test Bank" not in adapter.get_bank_allowlist("CC")


class TestMarketAdapterContract:
    """Verify all registered adapters satisfy the full contract."""

    @pytest.fixture(params=list_markets())
    def adapter(self, request):
        return get_adapter(request.param)

    def test_meta_is_frozen(self, adapter):
        m = adapter.meta
        assert isinstance(m, MarketMeta)
        with pytest.raises(AttributeError):
            m.code = "XX"

    def test_card_types_non_empty(self, adapter):
        assert len(adapter.meta.card_types) > 0

    def test_bank_allowlist_per_card_type(self, adapter):
        for ct in adapter.meta.card_types:
            banks = adapter.get_bank_allowlist(ct.code)
            assert isinstance(banks, list)

    def test_model_config_returns_dict(self, adapter):
        for ct in adapter.meta.card_types:
            cfg = adapter.get_model_config(ct.metric)
            assert isinstance(cfg, dict)
            assert "ensemble_members" in cfg

    def test_pipeline_steps_returns_list(self, adapter):
        steps = adapter.get_pipeline_steps()
        assert isinstance(steps, list)
        assert len(steps) > 0
