"""
Market registry — discovers and loads MarketAdapter subclasses.

Adapters are registered by placing a module in src/markets/ that defines
a subclass of MarketAdapter. The registry auto-discovers them at import
time by scanning this package.

Usage:
    from src.markets import get_adapter, list_markets

    adapter = get_adapter("IN")
    print(adapter.meta.name)  # "India"
    print(list_markets())     # ["IN"]
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from src.markets.base import MarketAdapter

_registry: dict[str, MarketAdapter] = {}


def _discover() -> None:
    """Scan this package for MarketAdapter subclasses and register them."""
    if _registry:
        return

    package_dir = Path(__file__).resolve().parent

    for info in pkgutil.iter_modules([str(package_dir)]):
        if info.name in ("base", "registry", "__init__"):
            continue
        module = importlib.import_module(f"src.markets.{info.name}")
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, MarketAdapter)
                and attr is not MarketAdapter
            ):
                instance = attr()
                _registry[instance.meta.code] = instance


def get_adapter(market_code: str) -> MarketAdapter:
    """Return the adapter for a market code (e.g. 'IN'). Raises KeyError if unknown."""
    _discover()
    code = market_code.upper()
    if code not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(f"Unknown market '{code}'. Available: {available}")
    return _registry[code]


def list_markets() -> list[str]:
    """Return sorted list of registered market codes."""
    _discover()
    return sorted(_registry.keys())
