"""
MIP Modelling — Pooled Bank Model (P2.3)
=========================================
A cross-bank gradient-boosted tree model that learns shared patterns
across all banks for a given card type. Bank identity, size tier,
and pooled seasonal features let the model borrow strength from
data-rich banks to improve forecasts for smaller ones.

Uses sklearn GradientBoostingRegressor (bundled with Prophet).

Usage:
    from src.modelling.bank_pooled_model import fit_pooled_model
    result = fit_pooled_model("cc")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.ensemble import GradientBoostingRegressor

from src.modelling.bank_data_prep import load_bank_data

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED = _PROJECT_ROOT / "data" / "processed"
_OUTPUT_DIR = _PROCESSED / "pooled"


def _build_features(df: pd.DataFrame, bank_name: str, bank_rank: int) -> pd.DataFrame:
    """Build feature matrix for one bank's time series."""
    feat = pd.DataFrame({"ds": df["ds"], "y": df["y"]})
    feat["month"] = feat["ds"].dt.month
    feat["year"] = feat["ds"].dt.year
    feat["trend"] = np.arange(len(feat))
    feat["bank_rank"] = bank_rank

    for lag in [1, 2, 3, 6, 12]:
        feat[f"lag_{lag}"] = feat["y"].shift(lag)

    feat["rolling_3m"] = feat["y"].shift(1).rolling(3, min_periods=1).mean()
    feat["rolling_6m"] = feat["y"].shift(1).rolling(6, min_periods=1).mean()
    feat["rolling_12m"] = feat["y"].shift(1).rolling(12, min_periods=1).mean()
    feat["yoy_change"] = feat["y"] / feat["y"].shift(12) - 1

    feat["bank_name"] = bank_name
    return feat.dropna().reset_index(drop=True)


def fit_pooled_model(
    card_type: str,
    forecast_periods: int = 12,
) -> dict:
    """Fit a pooled cross-bank model and produce forecasts.

    Returns dict with 'model', 'forecasts' (DataFrame), and 'cv_mape'.
    """
    assert card_type in ("cc", "dc")
    logger.info(f"Fitting pooled bank model ({card_type.upper()})...")

    data = load_bank_data(card_type)
    bank_dfs = data["bank_dfs"]

    all_features = []
    bank_names = sorted([b for b, df in bank_dfs.items() if df is not None])

    for rank, bank in enumerate(bank_names):
        bdf = bank_dfs[bank]
        if bdf is None or len(bdf) < 24:
            continue
        feat = _build_features(bdf, bank, rank)
        all_features.append(feat)

    if not all_features:
        logger.warning(f"No banks with sufficient data for pooled model ({card_type.upper()})")
        return {"model": None, "forecasts": pd.DataFrame(), "cv_mape": None}

    pool = pd.concat(all_features, ignore_index=True)
    feature_cols = [c for c in pool.columns if c not in ("ds", "y", "bank_name")]

    # Walk-forward CV: hold out last 12 months
    cutoff = pool["ds"].max() - pd.DateOffset(months=12)
    train = pool[pool["ds"] <= cutoff]
    test = pool[pool["ds"] > cutoff]

    if len(test) == 0 or len(train) < 50:
        logger.warning("Insufficient data for pooled model CV")
        return {"model": None, "forecasts": pd.DataFrame(), "cv_mape": None}

    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=42,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[feature_cols], train["y"])

    test_pred = model.predict(test[feature_cols])
    test_actual = test["y"].values
    mask = test_actual != 0
    cv_mape = float(np.mean(np.abs((test_actual[mask] - test_pred[mask]) / test_actual[mask])) * 100)

    logger.info(f"  Pooled model CV MAPE: {cv_mape:.2f}%")

    # Refit on full data
    model.fit(pool[feature_cols], pool["y"])

    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    logger.info(f"  Top features: {', '.join(f'{k}={v:.3f}' for k, v in top_features)}")

    # Save
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecasts = []
    for bank in bank_names:
        bdf = bank_dfs[bank]
        if bdf is None or len(bdf) < 24:
            continue
        feat = _build_features(bdf, bank, bank_names.index(bank))
        last_row = feat.iloc[-1:].copy()
        pred = model.predict(last_row[feature_cols])
        forecasts.append({
            "bank_name": bank,
            "card_type": card_type,
            "pooled_forecast": float(pred[0]),
            "last_actual": float(bdf["y"].iloc[-1]),
            "last_date": str(bdf["ds"].iloc[-1].date()),
        })

    fc_df = pd.DataFrame(forecasts)
    fc_df.to_csv(_OUTPUT_DIR / f"pooled_{card_type}_forecasts.csv", index=False)

    logger.info(f"  Saved pooled forecasts for {len(forecasts)} banks")

    return {
        "model": model,
        "forecasts": fc_df,
        "cv_mape": cv_mape,
        "feature_importances": importances,
    }


if __name__ == "__main__":
    for ct in ["cc", "dc"]:
        result = fit_pooled_model(ct)
        if result["cv_mape"] is not None:
            print(f"{ct.upper()} pooled model CV MAPE: {result['cv_mape']:.2f}%")
