"""
MIP — Live Scorecard (P4.1)
============================
Compares past forecasts against actual data as new RBI months land.
Produces per-model, per-month APE scores that the dashboard shows as
rolling 12-month audited accuracy.

This runs during each sync — it reads forecasts_aggregate and
processed_aggregate from Supabase, finds months where both a forecast
and an actual exist, computes APE, and upserts into scorecard_scores.

For bank-level models it reads forecasts_bank vs processed_bank_series.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED = _PROJECT_ROOT / "data" / "processed"


def generate_aggregate_scores() -> list[dict]:
    """Score aggregate forecasts against actuals from local parquet files.

    Matches forecast rows to actuals by month. Returns list of dicts
    ready for Supabase upsert into scorecard_scores.
    """
    actuals_path = _PROCESSED / "rbi_psi_cards.parquet"
    if not actuals_path.exists():
        logger.info("No actuals file for scorecard — skipping aggregate scoring")
        return []

    actuals = pd.read_parquet(actuals_path)
    actuals["date"] = pd.to_datetime(actuals["date"]).dt.to_period("M").dt.to_timestamp()

    metric_map = {
        "cc_outstanding": ("credit_cards_outstanding_lakh", "forecast_cc.parquet"),
        "dc_outstanding": ("debit_cards_outstanding_lakh", "forecast_dc.parquet"),
    }

    scores = []

    for model_name, (actual_col, forecast_file) in metric_map.items():
        fc_path = _PROCESSED / forecast_file
        if not fc_path.exists():
            continue
        if actual_col not in actuals.columns:
            continue

        fc = pd.read_parquet(fc_path)
        date_col = "ds" if "ds" in fc.columns else "date"
        yhat_col = "yhat" if "yhat" in fc.columns else "forecast"

        if date_col not in fc.columns or yhat_col not in fc.columns:
            continue

        fc["date"] = pd.to_datetime(fc[date_col]).dt.to_period("M").dt.to_timestamp()
        fc = fc.rename(columns={yhat_col: "forecast"})

        actual_series = actuals[["date", actual_col]].dropna(subset=[actual_col])
        actual_series = actual_series.rename(columns={actual_col: "actual"})

        merged = fc[["date", "forecast"]].merge(actual_series, on="date", how="inner")

        for _, row in merged.iterrows():
            if row["actual"] == 0:
                continue
            ape = abs(row["actual"] - row["forecast"]) / abs(row["actual"]) * 100
            scores.append({
                "model_name": model_name,
                "forecast_month": str(row["date"])[:10],
                "forecast_value": round(float(row["forecast"]), 2),
                "actual_value": round(float(row["actual"]), 2),
                "ape": round(float(ape), 2),
            })

    logger.info(f"Scorecard: generated {len(scores)} aggregate scores")
    return scores


def generate_bank_scores() -> list[dict]:
    """Score bank-level forecasts against actuals from local parquet files."""
    scores = []

    for card_type in ["cc", "dc"]:
        actuals_path = _PROCESSED / f"bankwise_cards_{card_type}.parquet"
        forecasts_dir = _PROCESSED / "bankwise_forecasts"

        if not actuals_path.exists() or not forecasts_dir.exists():
            continue

        actuals = pd.read_parquet(actuals_path)
        actuals["date"] = pd.to_datetime(actuals["date"]).dt.to_period("M").dt.to_timestamp()
        target_col = f"{card_type}_outstanding"

        if target_col not in actuals.columns:
            continue

        for fc_path in sorted(forecasts_dir.glob(f"{card_type}_*_forecast.parquet")):
            fc = pd.read_parquet(fc_path)
            date_col = "ds" if "ds" in fc.columns else "date"
            yhat_col = "yhat" if "yhat" in fc.columns else "forecast"

            if date_col not in fc.columns or yhat_col not in fc.columns:
                continue

            bank_name_raw = fc_path.stem.replace("_forecast", "").replace(f"{card_type}_", "", 1)
            bank_name = bank_name_raw.replace("_", " ").title()

            fc["date"] = pd.to_datetime(fc[date_col]).dt.to_period("M").dt.to_timestamp()
            fc = fc.rename(columns={yhat_col: "forecast"})

            bank_actuals = actuals[actuals["bank_name"] == bank_name][["date", target_col]].dropna()
            bank_actuals = bank_actuals.rename(columns={target_col: "actual"})

            merged = fc[["date", "forecast"]].merge(bank_actuals, on="date", how="inner")

            model_label = f"{card_type}_{bank_name_raw}"

            for _, row in merged.iterrows():
                if row["actual"] == 0:
                    continue
                ape = abs(row["actual"] - row["forecast"]) / abs(row["actual"]) * 100
                scores.append({
                    "model_name": model_label,
                    "forecast_month": str(row["date"])[:10],
                    "forecast_value": round(float(row["forecast"]), 2),
                    "actual_value": round(float(row["actual"]), 2),
                    "ape": round(float(ape), 2),
                })

    logger.info(f"Scorecard: generated {len(scores)} bank-level scores")
    return scores


def generate_all_scores() -> list[dict]:
    """Generate all scorecard scores (aggregate + bank-level)."""
    return generate_aggregate_scores() + generate_bank_scores()


def rolling_accuracy(scores: list[dict], window: int = 12) -> dict[str, float]:
    """Compute rolling MAPE per model from the last `window` scored months.

    Returns {model_name: rolling_mape}.
    """
    if not scores:
        return {}

    df = pd.DataFrame(scores)
    df["forecast_month"] = pd.to_datetime(df["forecast_month"])

    result = {}
    for model, grp in df.groupby("model_name"):
        latest = grp.nlargest(window, "forecast_month")
        result[str(model)] = round(float(latest["ape"].mean()), 2)

    return result
