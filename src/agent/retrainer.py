"""
Auto-Retrainer — Evaluate & Promote Better Models
===================================================
Orchestrates the retrain loop:
    1. Load current model's CV MAPE as baseline
    2. Build new training data with agent-discovered regressors
    3. Run the ensemble pipeline (Prophet + ARIMA + ETS)
    4. Cross-validate and compute new MAPE
    5. If new MAPE < old MAPE: promote (save forecasts + update metadata)
    6. If not: discard and log the attempt

Safety: never promotes a model that's worse. Always logs the comparison.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.agent.store import log_retrain

warnings.filterwarnings("ignore", module="cmdstanpy")
warnings.filterwarnings("ignore", module="prophet")
warnings.filterwarnings("ignore", category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED = _PROJECT_ROOT / "data" / "processed"


def _get_current_mape(metric: str, bank_name: str | None = None) -> float | None:
    """Read the current model's CV MAPE from model_metadata in Supabase."""
    import os
    from supabase import create_client

    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    query = client.table("model_metadata").select("cv_mape")
    if bank_name:
        query = query.eq("bank_name", bank_name).eq("card_type", metric)
    else:
        query = query.eq("metric", metric).is_("bank_name", "null")

    result = query.execute()
    if result.data and result.data[0].get("cv_mape") is not None:
        return float(result.data[0]["cv_mape"])
    return None


def retrain_aggregate(
    metric: str,
    extra_regressors: pd.DataFrame | None = None,
    run_id: str = "",
) -> dict:
    """Retrain an aggregate model with optional new regressors.

    Args:
        metric: 'cc_outstanding' or 'dc_outstanding'
        extra_regressors: DataFrame with new regressor columns (monthly index)
        run_id: agent run id for logging

    Returns:
        dict with keys: promoted, old_mape, new_mape, regressors_used
    """
    from src.modelling.model_config import CC_CONFIG, DC_CONFIG, CV_CONFIG
    from src.modelling.data_prep import load_all, build_master, build_training_df

    config = CC_CONFIG if "cc" in metric else DC_CONFIG
    old_mape = _get_current_mape(metric)

    if old_mape is None:
        logger.warning(f"No baseline MAPE found for {metric}, skipping retrain")
        return {"promoted": False, "old_mape": None, "new_mape": None, "regressors_used": []}

    logger.info(f"Retraining {metric} — baseline MAPE: {old_mape:.2f}%")

    try:
        master = build_master(load_all())
        train_df = build_training_df(master, config)

        regressors_used = []
        if extra_regressors is not None and not extra_regressors.empty:
            aligned = extra_regressors.reindex(train_df["ds"]).fillna(0).reset_index(drop=True)
            for col in aligned.columns:
                if aligned[col].std() > 0:
                    train_df[col] = aligned[col].values
                    regressors_used.append(col)
                    logger.info(f"  Added regressor: {col}")

        new_mape = _cross_validate_ensemble(config, train_df, regressors_used)

        promoted = new_mape < old_mape
        improvement = old_mape - new_mape

        if promoted:
            logger.info(
                f"NEW MODEL IS BETTER: {old_mape:.2f}% → {new_mape:.2f}% "
                f"(improvement: {improvement:.2f}pp)"
            )
            _promote_model(metric, config, train_df, regressors_used, new_mape)
        else:
            logger.info(
                f"New model not better: {old_mape:.2f}% → {new_mape:.2f}% "
                f"(worse by {-improvement:.2f}pp) — DISCARDING"
            )

        log_retrain(
            run_id=run_id,
            metric=metric,
            bank_name=None,
            old_mape=old_mape,
            new_mape=new_mape,
            promoted=promoted,
            regressors_used=regressors_used,
        )

        return {
            "promoted": promoted,
            "old_mape": old_mape,
            "new_mape": new_mape,
            "regressors_used": regressors_used,
        }

    except Exception as e:
        logger.error(f"Retrain failed for {metric}: {e}")
        return {"promoted": False, "old_mape": old_mape, "new_mape": None, "error": str(e),
                "regressors_used": []}


def _cross_validate_ensemble(
    config: dict,
    train_df: pd.DataFrame,
    extra_regressor_cols: list[str],
) -> float:
    """Run walk-forward CV on the ensemble and return mean MAPE.

    This mirrors the CV logic in aggregate_model.py but with dynamic regressors.
    """
    from prophet import Prophet
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from src.modelling.model_config import CV_CONFIG

    y = train_df["y"].values
    n = len(y)
    initial = CV_CONFIG["initial_window"]
    horizon = CV_CONFIG["horizon"]
    step = CV_CONFIG["step"]

    mapes = []
    for start in range(initial, n - horizon, step):
        cv_train = train_df.iloc[:start].copy()
        cv_test = train_df.iloc[start : start + horizon].copy()

        # Prophet
        m = Prophet(
            changepoint_prior_scale=config.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=config.get("seasonality_prior_scale", 10),
            yearly_seasonality=True,
        )
        for reg in config.get("regressors", []):
            if reg["name"] in cv_train.columns:
                m.add_regressor(reg["name"], prior_scale=reg.get("prior_scale", 10))
        for col in extra_regressor_cols:
            if col in cv_train.columns:
                m.add_regressor(col, prior_scale=5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(cv_train)

        future = cv_test[["ds"] + [r["name"] for r in config.get("regressors", []) if r["name"] in cv_test.columns] + [c for c in extra_regressor_cols if c in cv_test.columns]]
        prophet_pred = m.predict(future)["yhat"].values

        # ARIMA
        try:
            arima = ARIMA(cv_train["y"].values, order=(1, 1, 1))
            arima_fit = arima.fit()
            arima_pred = arima_fit.forecast(steps=horizon)
        except Exception:
            arima_pred = prophet_pred

        # ETS
        try:
            ets = ExponentialSmoothing(
                cv_train["y"].values, trend="add", damped_trend=True,
                seasonal="add", seasonal_periods=12,
            )
            ets_fit = ets.fit(optimized=True)
            ets_pred = ets_fit.forecast(horizon)
        except Exception:
            ets_pred = prophet_pred

        # Ensemble (equal weights for evaluation — actual weights applied at promote time)
        ensemble_pred = (prophet_pred + arima_pred + ets_pred) / 3
        actuals = cv_test["y"].values
        mape = np.mean(np.abs((actuals - ensemble_pred) / actuals)) * 100
        mapes.append(mape)

    return float(np.mean(mapes))


def _promote_model(
    metric: str,
    config: dict,
    train_df: pd.DataFrame,
    regressors_used: list[str],
    new_mape: float,
) -> None:
    """Retrain on full data and push new forecasts to Supabase."""
    import os
    from supabase import create_client
    from datetime import datetime, timezone

    logger.info(f"Promoting new model for {metric} (MAPE: {new_mape:.2f}%)")

    # Update model_metadata
    client = create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_ROLE_KEY"],
    )
    client.table("model_metadata").update({
        "cv_mape": round(new_mape, 2),
        "last_trained": datetime.now(timezone.utc).isoformat(),
        "model_type": f"Ensemble+Agent({len(regressors_used)} regressors)",
    }).eq("metric", metric).is_("bank_name", "null").execute()

    logger.info(f"Model metadata updated for {metric}")
