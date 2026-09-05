"""
India market adapter — all India-specific configuration in one place.
"""

from __future__ import annotations

import sys

from src.markets.base import (
    CardType,
    DataSource,
    MarketAdapter,
    MarketMeta,
    StructuralEvent,
)


_META = MarketMeta(
    code="IN",
    name="India",
    currency="INR",
    currency_symbol="₹",
    unit_label="Lakhs",
    unit_divisor=10.0,
    regulator="Reserve Bank of India",
    timezone="Asia/Kolkata",
    card_types=(
        CardType(code="CC", label="Credit Card", metric="cc_outstanding"),
        CardType(code="DC", label="Debit Card", metric="dc_outstanding"),
    ),
    data_sources=(
        DataSource(name="rbi_psi", label="RBI Payment System Indicators", frequency="monthly", raw_subdir="rbi_psi"),
        DataSource(name="rbi_bankwise", label="RBI Bankwise ATM/POS/Card", frequency="monthly", raw_subdir="rbi_bankwise"),
        DataSource(name="npci_upi", label="NPCI UPI Statistics", frequency="monthly", raw_subdir="npci_upi"),
        DataSource(name="rbi_repo", label="RBI Repo Rate", frequency="monthly", raw_subdir="rbi_repo_rate"),
    ),
    structural_events=(
        StructuralEvent("2014-08-01", "PMJDY launch", "positive"),
        StructuralEvent("2016-11-01", "Demonetisation", "positive"),
        StructuralEvent("2020-04-01", "COVID lockdown", "negative"),
        StructuralEvent("2022-01-01", "UPI inflection", "negative_debit"),
        StructuralEvent("2022-07-01", "RBI card conduct directions", "positive_dc"),
        StructuralEvent("2023-11-01", "RBI unsecured lending tightening", "negative_credit"),
    ),
)

_CC_BANKS = [
    "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
    "Kotak Mahindra Bank", "RBL Bank", "IDFC First Bank",
    "IndusInd Bank", "Bank of Baroda", "Yes Bank", "Canara Bank", "HSBC",
]

_DC_BANKS = [
    "State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
    "Union Bank of India", "Punjab National Bank", "Axis Bank",
    "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
    "ICICI Bank", "Paytm Payments Bank", "Central Bank of India",
    "India Post Payments Bank", "Indian Overseas Bank", "UCO Bank",
]

_TRAINING_STARTS = {
    "cc_outstanding": "2013-01-01",
    "dc_outstanding": None,
    "dc_vol": "2022-01-01",
}

_REGRESSORS = {
    "cc_outstanding": ["repo_rate", "credit_card_vol_lakh"],
    "dc_outstanding": ["repo_rate", "upi_p2m_vol_cr"],
}


class IndiaAdapter(MarketAdapter):

    @property
    def meta(self) -> MarketMeta:
        return _META

    def get_bank_allowlist(self, card_code: str) -> list[str]:
        if card_code == "CC":
            return list(_CC_BANKS)
        if card_code == "DC":
            return list(_DC_BANKS)
        return []

    def get_training_start(self, metric: str) -> str | None:
        return _TRAINING_STARTS.get(metric)

    def get_model_config(self, metric: str) -> dict:
        return {
            "ensemble_members": ["prophet", "arima", "arimax", "ets", "direct"],
            "cv": {
                "initial": "1461 days",
                "period": "182 days",
                "horizon": "365 days",
            },
            "forecast": {
                "periods": 24,
                "freq": "MS",
            },
            "prophet_kwargs": {
                "yearly_seasonality": True,
                "weekly_seasonality": False,
                "daily_seasonality": False,
                "seasonality_mode": "multiplicative",
                "interval_width": 0.90,
            },
        }

    def get_regressor_columns(self, metric: str) -> list[str]:
        return _REGRESSORS.get(metric, [])

    def get_pipeline_steps(self) -> list[dict]:
        return [
            {"name": "Ingestion", "module": "src.ingestion", "cv_flag": False, "skip_flag": "skip_ingestion"},
            {"name": "Aggregate Model (CC + DC Outstanding)", "module": "src.modelling.aggregate_model", "cv_flag": True, "skip_flag": None},
            {"name": "Bank-Level Model", "module": "src.modelling.bank_model", "cv_flag": True, "skip_flag": None},
            {"name": "Transaction Volume Models", "module": "src.modelling.txn_volume_model", "cv_flag": True, "skip_flag": None},
            {"name": "UPI Displacement Analysis", "module": "src.modelling.upi_analysis", "cv_flag": False, "skip_flag": None},
            {"name": "Rebuild Dashboard", "module": None, "script": "scripts/rebuild_dashboard.py", "cv_flag": False, "skip_flag": None},
        ]
