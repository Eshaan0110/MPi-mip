"""
Feature Engineer — Signals → Model Regressors
===============================================
Converts structured findings from the extractor into quantitative
regressor columns that the ensemble models can consume.

Signal-to-regressor mapping:
    new_card_launch       → new_products_<bank>  (count in month)
    card_discontinuation  → discontinued_<bank>  (count, negative)
    regulatory_change     → rbi_regulatory       (direction: +1/-1)
    partnership           → partnerships_<bank>   (count in month)
    growth_target         → bank_bullish_<bank>   (1 if positive target)
    infrastructure_change → infra_change          (direction: +1/-1)
    macro_policy          → macro_signal           (direction: +1/-1)

Output: a DataFrame with month index and one column per regressor,
ready to merge into the model training DataFrame.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import pandas as pd
from loguru import logger


SIGNAL_TO_REGRESSOR = {
    "new_card_launch": {"prefix": "new_products", "aggregation": "count", "weight": 1},
    "card_discontinuation": {"prefix": "discontinued", "aggregation": "count", "weight": -1},
    "regulatory_change": {"prefix": "rbi_regulatory", "aggregation": "direction", "weight": 1},
    "partnership": {"prefix": "partnerships", "aggregation": "count", "weight": 1},
    "growth_target": {"prefix": "bank_bullish", "aggregation": "flag", "weight": 1},
    "infrastructure_change": {"prefix": "infra_change", "aggregation": "direction", "weight": 1},
    "macro_policy": {"prefix": "macro_signal", "aggregation": "direction", "weight": 1},
    "market_event": {"prefix": "market_event", "aggregation": "direction", "weight": 1},
}

DIRECTION_MAP = {"positive": 1, "negative": -1, "neutral": 0}
MAGNITUDE_MAP = {"high": 3, "medium": 2, "low": 1}


def _parse_month(date_str: str | None) -> str | None:
    """Convert a date string to YYYY-MM format."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m")
    except (ValueError, AttributeError):
        return None


def signals_to_regressors(
    signals: list[dict],
    start_month: str = "2020-01",
    end_month: str | None = None,
) -> pd.DataFrame:
    """Convert a list of signal dicts into a regressor DataFrame.

    Args:
        signals: list of finding dicts from the extractor
        start_month: first month for the output index
        end_month: last month (default: current month + 24)

    Returns:
        DataFrame with DatetimeIndex (monthly) and regressor columns.
    """
    if end_month is None:
        end_month = pd.Timestamp.now().strftime("%Y-%m")

    months = pd.date_range(start_month, end_month, freq="MS")
    regressors: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for sig in signals:
        sig_type = sig.get("signal_type", "")
        mapping = SIGNAL_TO_REGRESSOR.get(sig_type)
        if not mapping:
            continue

        month = _parse_month(sig.get("effective_date") or sig.get("discovered_at"))
        if not month:
            continue

        bank = sig.get("bank_name")
        direction = DIRECTION_MAP.get(sig.get("impact_direction", "neutral"), 0)
        magnitude = MAGNITUDE_MAP.get(sig.get("impact_magnitude", "low"), 1)

        if mapping["aggregation"] == "count":
            col = f"{mapping['prefix']}_{bank}" if bank else mapping["prefix"]
            regressors[col][month] += mapping["weight"] * magnitude
        elif mapping["aggregation"] == "direction":
            col = mapping["prefix"]
            regressors[col][month] += direction * magnitude
        elif mapping["aggregation"] == "flag":
            col = f"{mapping['prefix']}_{bank}" if bank else mapping["prefix"]
            regressors[col][month] = 1.0 if direction >= 0 else 0.0

    if not regressors:
        logger.info("No regressors generated from signals")
        return pd.DataFrame(index=months)

    df = pd.DataFrame(index=months)
    for col, month_vals in regressors.items():
        clean_col = col.replace(" ", "_").replace(".", "").lower()
        df[clean_col] = 0.0
        for m, val in month_vals.items():
            ts = pd.Timestamp(m + "-01")
            if ts in df.index:
                df.loc[ts, clean_col] = val

    logger.info(f"Generated {len(df.columns)} regressor columns from {len(signals)} signals")
    return df


def select_top_regressors(
    regressors_df: pd.DataFrame,
    target_series: pd.Series,
    max_regressors: int = 5,
) -> list[str]:
    """Select the top N regressors by absolute correlation with target."""
    if regressors_df.empty:
        return []

    aligned = regressors_df.reindex(target_series.index).fillna(0)
    correlations = {}
    for col in aligned.columns:
        if aligned[col].std() > 0:
            corr = aligned[col].corr(target_series)
            if pd.notna(corr):
                correlations[col] = abs(corr)

    ranked = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    selected = [col for col, _ in ranked[:max_regressors]]
    logger.info(f"Selected regressors: {selected}")
    return selected
