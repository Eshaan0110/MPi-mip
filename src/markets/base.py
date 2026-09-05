"""
MarketAdapter — abstract base defining what each country market provides.

Every new market implements this interface. The pipeline, scorecard, and
dashboard read market-specific details from the adapter, not from
hard-coded constants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CardType:
    """One card category tracked in this market."""
    code: str          # e.g. "CC", "DC"
    label: str         # e.g. "Credit Card", "Debit Card"
    metric: str        # e.g. "cc_outstanding", "dc_outstanding"


@dataclass(frozen=True)
class StructuralEvent:
    """A market-level event that affects forecasts."""
    date: str          # ISO date, e.g. "2016-11-08"
    name: str
    direction: str     # "positive", "negative", "definitional_break", etc.


@dataclass(frozen=True)
class DataSource:
    """Metadata for one raw data source the market uses."""
    name: str          # e.g. "rbi_psi"
    label: str         # e.g. "RBI Payment System Indicators"
    frequency: str     # "monthly", "quarterly", "daily"
    raw_subdir: str    # relative to data/raw/


@dataclass(frozen=True)
class MarketMeta:
    """Static metadata for a market — no logic, just facts."""
    code: str                                # ISO 3166-1 alpha-2, e.g. "IN"
    name: str                                # "India"
    currency: str                            # "INR"
    currency_symbol: str                     # "₹"
    unit_label: str                          # "Lakhs" — the unit raw data arrives in
    unit_divisor: float                      # to convert raw → millions (e.g. 10.0 for lakhs→millions)
    regulator: str                           # "Reserve Bank of India"
    timezone: str                            # "Asia/Kolkata"
    card_types: tuple[CardType, ...]         # card categories tracked
    data_sources: tuple[DataSource, ...]     # raw data feeds
    structural_events: tuple[StructuralEvent, ...] = ()


class MarketAdapter(ABC):
    """Base class every market adapter must implement.

    Adapters are stateless — they expose configuration and factory
    methods but hold no mutable state. One adapter per market.
    """

    @property
    @abstractmethod
    def meta(self) -> MarketMeta:
        """Return static market metadata."""

    @abstractmethod
    def get_bank_allowlist(self, card_code: str) -> list[str]:
        """Return the ordered list of bank names modelled for a card type."""

    @abstractmethod
    def get_training_start(self, metric: str) -> str | None:
        """Return the earliest training date (ISO) for a metric, or None for 'use all data'."""

    @abstractmethod
    def get_model_config(self, metric: str) -> dict:
        """Return model hyperparameters for a given metric.

        At minimum: {"ensemble_members": [...], "cv": {...}, "forecast": {...}}
        """

    @abstractmethod
    def get_pipeline_steps(self) -> list[dict]:
        """Return the ordered list of pipeline step definitions.

        Each step: {"name": str, "module": str, "cv_flag": bool, "skip_flag": str | None}
        """

    def get_regressor_columns(self, metric: str) -> list[str]:
        """Return external regressor column names for a metric. Override per market."""
        return []

    def resolve_data_dir(self, project_root: Path) -> Path:
        """Return the market-specific processed data directory."""
        return project_root / "data" / "processed"
