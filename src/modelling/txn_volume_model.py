"""
MIP Modelling -- Transaction Volume Forecasts
==============================================
Forecasts monthly transaction volumes for:
  - Credit card total transactions (lakh)
  - Debit card total transactions (lakh)
  - UPI total transactions (million)

Same Prophet framework as the cards outstanding models. Uses the configs
defined in model_config.py (CC_VOL_CONFIG, DC_VOL_CONFIG, UPI_VOL_CONFIG).

Run:
    uv run python -m src.modelling.txn_volume_model            # all three
    uv run python -m src.modelling.txn_volume_model --no-cv    # skip CV

Outputs to data/processed/:
    forecast_cc_vol.parquet, forecast_cc_vol.csv
    forecast_dc_vol.parquet, forecast_dc_vol.csv
    forecast_upi_vol.parquet, forecast_upi_vol.csv
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.model_config import (
    CC_VOL_CONFIG,
    DC_VOL_CONFIG,
    UPI_VOL_CONFIG,
    CV_CONFIG,
    FORECAST_CONFIG,
    STRUCTURAL_EVENTS,
    RegressorSpec,
)
from src.modelling.data_prep import (
    load_all,
    build_master,
    build_training_df,
    build_future_df,
)

from src.utils.run_logger import RunLogger

warnings.filterwarnings("ignore", module="cmdstanpy")
warnings.filterwarnings("ignore", module="prophet")
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


def _build_prophet(config: dict, train_df: pd.DataFrame):
    """Fit a Prophet model from config. Returns fitted model."""
    from prophet import Prophet

    model_key = config["output_stem"]

    changepoint_dates = list(config.get("extra_changepoints", []))
    for event_name, spec in STRUCTURAL_EVENTS.items():
        if spec["type"] == "changepoint" and any(
            k in config.get("structural_events", []) for k in [event_name]
        ):
            changepoint_dates.append(
                spec.get("date", spec.get("dates", [""])[0])
            )

    prophet_kwargs = dict(config["prophet_kwargs"])
    if changepoint_dates:
        valid_cps = [
            pd.Timestamp(d)
            for d in changepoint_dates
            if train_df["ds"].min() < pd.Timestamp(d) < train_df["ds"].max()
        ]
        if valid_cps:
            prophet_kwargs["changepoints"] = valid_cps
            logger.info(f"  Changepoints: {[d.strftime('%b %Y') for d in valid_cps]}")

    m = Prophet(**prophet_kwargs)

    regressors: list[RegressorSpec] = config["regressors"]
    for spec in regressors:
        final_col = f"{spec.col}_lag{spec.lag}" if spec.lag > 0 else spec.col
        if final_col not in train_df.columns:
            logger.warning(f"  Regressor '{final_col}' missing -- skipping")
            continue
        m.add_regressor(final_col, standardize=spec.standardize, mode=spec.mode)
        logger.info(f"  Regressor: {final_col} (mode={spec.mode})")

    event_cols = [c for c in train_df.columns if c.startswith("event_")]
    for col in event_cols:
        m.add_regressor(col, standardize=False, mode="additive")

    m.fit(train_df)
    logger.info(f"  Fitted on {len(train_df)} rows")
    return m


def _run_cv(model, config: dict) -> pd.DataFrame:
    """Run rolling CV. Returns metrics DataFrame."""
    from prophet.diagnostics import cross_validation, performance_metrics

    try:
        cv_df = cross_validation(
            model,
            initial=CV_CONFIG["initial"],
            period=CV_CONFIG["period"],
            horizon=CV_CONFIG["horizon"],
            parallel=CV_CONFIG.get("parallel", "processes"),
            disable_tqdm=True,
        )
        metrics = performance_metrics(cv_df)
        mape = metrics["mape"] * 100
        logger.info(
            f"  CV MAPE: mean={mape.mean():.2f}% | "
            f"range=[{mape.min():.2f}%, {mape.max():.2f}%] | "
            f"{len(metrics)} windows"
        )
        stem = config["output_stem"]
        cv_df.to_parquet(_PROCESSED / f"{stem}_cv_raw.parquet", index=False)
        metrics.to_csv(_PROCESSED / f"{stem}_cv_metrics.csv", index=False)
        return metrics
    except Exception as e:
        logger.error(f"  CV failed: {e}")
        return pd.DataFrame()


def _fit_arima_fc(y: np.ndarray, horizon: int) -> np.ndarray | None:
    """ARIMA(1,1,1) forecast for txn volume models."""
    from statsmodels.tsa.arima.model import ARIMA
    try:
        m = ARIMA(y, order=(1, 1, 1))
        fit = m.fit()
        return fit.forecast(steps=horizon)
    except Exception:
        return None


def _fit_ets_fc(y: np.ndarray, horizon: int) -> np.ndarray | None:
    """Damped additive ETS forecast for txn volume models."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        m = ExponentialSmoothing(y, trend="add", seasonal="add",
                                seasonal_periods=12, damped_trend=True)
        fit = m.fit(optimized=True)
        return fit.forecast(steps=horizon)
    except Exception:
        return None


# Ensemble weights for txn volume models (P2.5)
TXN_ENSEMBLE_WEIGHTS = {
    "forecast_upi_vol": {"prophet": 0.35, "arima": 0.25, "ets": 0.40},
    "forecast_cc_vol":  {"prophet": 0.50, "arima": 0.25, "ets": 0.25},
    "forecast_dc_vol":  {"prophet": 0.50, "arima": 0.25, "ets": 0.25},
}


def _estimate_txn_weights(
    y: np.ndarray, config_stem: str, horizon: int = 12,
) -> dict[str, float]:
    """Walk-forward CV to find best Prophet/ARIMA/ETS blend for txn volumes.

    All 3 members are evaluated in the CV loop so the reported MAPE
    reflects the actual production ensemble.
    """
    initial = 36
    step = 6
    default = TXN_ENSEMBLE_WEIGHTS.get(config_stem, {"prophet": 0.50, "arima": 0.25, "ets": 0.25})

    if len(y) < initial + horizon:
        return default

    arima_all, ets_all, actuals = [], [], []
    for start in range(initial, len(y) - horizon + 1, step):
        train_y = y[:start]
        test_y = y[start:start + horizon]
        a = _fit_arima_fc(train_y, horizon)
        e = _fit_ets_fc(train_y, horizon)
        if a is not None and e is not None:
            arima_all.append(a)
            ets_all.append(e)
            actuals.append(test_y)

    if len(actuals) < 2:
        return default

    prophet_w = default.get("prophet", 0.35)
    remaining = 1.0 - prophet_w
    best_mape = float("inf")
    best_arima_share = 0.5

    for arima_share in np.arange(0.0, 1.05, 0.1):
        ets_share = 1.0 - arima_share
        mapes = []
        for a, e, act in zip(arima_all, ets_all, actuals):
            ens = arima_share * a + ets_share * e
            m = np.mean(np.abs((act - ens) / act)) * 100
            mapes.append(m)
        avg = np.mean(mapes)
        if avg < best_mape:
            best_mape = avg
            best_arima_share = arima_share

    weights = {
        "prophet": prophet_w,
        "arima": round(remaining * best_arima_share, 2),
        "ets": round(remaining * (1.0 - best_arima_share), 2),
    }
    logger.info(
        f"  Txn ensemble weights: {weights} "
        f"(non-Prophet CV MAPE: {best_mape:.2f}%)"
    )
    return weights


def _build_txn_conformal_ci(
    y: np.ndarray, ensemble: np.ndarray, horizon: int,
    initial: int = 36, step: int = 6, alpha: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Build conformal 90% CIs from walk-forward ARIMA+ETS percentage errors.

    Same approach as the aggregate model's conformal intervals (P2.2):
    collects percentage errors per forecast step across CV folds, then
    takes quantiles. Falls back to +/-10% if insufficient data.
    """
    pct_errors_by_step: dict[int, list[float]] = {h: [] for h in range(horizon)}

    for start in range(initial, len(y) - horizon + 1, step):
        train_y = y[:start]
        test_y = y[start:start + horizon]

        a = _fit_arima_fc(train_y, horizon)
        e = _fit_ets_fc(train_y, horizon)
        if a is None or e is None:
            continue

        ens = 0.5 * a + 0.5 * e
        for h in range(horizon):
            if test_y[h] != 0:
                pct_err = (test_y[h] - ens[h]) / test_y[h]
                pct_errors_by_step[h].append(pct_err)

    lower_pcts = np.full(horizon, -0.10)
    upper_pcts = np.full(horizon, 0.10)

    for h in range(horizon):
        errs = pct_errors_by_step.get(h, [])
        if len(errs) >= 3:
            n = len(errs)
            q_lo = alpha / 2
            q_hi = 1 - alpha / 2
            correction = min((n + 1) / n, 1.05)
            lower_pcts[h] = np.quantile(errs, q_lo) * correction
            upper_pcts[h] = np.quantile(errs, q_hi) * correction

    ci_lower = ensemble * (1 + lower_pcts)
    ci_upper = ensemble * (1 + upper_pcts)
    return ci_lower, ci_upper


def _run_forecast(model, config: dict, train_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    """Generate 12-month ensemble forecast. Returns forecast-only rows.

    For UPI and txn volume models (P2.5): blends Prophet + ARIMA + ETS
    using CV-estimated weights instead of Prophet-only.
    """
    horizon = FORECAST_CONFIG.get("periods", 24)
    # Txn volume models use 12-month horizon by default
    if "vol" in config.get("output_stem", ""):
        horizon = 12

    future_df = build_future_df(train_df, config, master)
    forecast = model.predict(future_df)

    last_hist = train_df["ds"].max()
    prophet_fc = forecast[forecast["ds"] > last_hist].copy()
    prophet_yhat = prophet_fc["yhat"].values[:horizon]
    prophet_dates = prophet_fc["ds"].values[:horizon]

    y = train_df["y"].values
    arima_yhat = _fit_arima_fc(y, horizon)
    ets_yhat = _fit_ets_fc(y, horizon)

    # Build ensemble
    stem = config["output_stem"]
    weights = _estimate_txn_weights(y, stem, horizon)

    forecasts = {}
    if prophet_yhat is not None and len(prophet_yhat) == horizon:
        forecasts["prophet"] = prophet_yhat
    if arima_yhat is not None and len(arima_yhat) == horizon:
        forecasts["arima"] = arima_yhat
    if ets_yhat is not None and len(ets_yhat) == horizon:
        forecasts["ets"] = ets_yhat

    if not forecasts:
        raise RuntimeError(f"All forecast models failed for {config['name']}")

    total_w = sum(weights.get(k, 0) for k in forecasts if weights.get(k, 0) > 0)
    ensemble = np.zeros(horizon)
    for name, fc_arr in forecasts.items():
        w = weights.get(name, 0) / total_w if total_w > 0 else 1.0 / len(forecasts)
        ensemble += w * fc_arr
        logger.info(f"  Ensemble member {name}: weight={w:.2f}, mean={np.mean(fc_arr):.1f}")

    # Clip negative (volumes can't be negative)
    ensemble = np.clip(ensemble, 0, None)

    # Conformal CIs from walk-forward ensemble percentage errors
    fc_lower, fc_upper = _build_txn_conformal_ci(y, ensemble, horizon)
    fc_lower = np.clip(fc_lower, 0, None)

    fc = pd.DataFrame({
        "date": prophet_dates[:horizon],
        "forecast": ensemble,
        "forecast_lower": fc_lower,
        "forecast_upper": fc_upper,
        "trend_component": prophet_fc["trend"].values[:horizon],
        "seasonality_component": prophet_fc["yearly"].values[:horizon],
    })

    fc.to_parquet(_PROCESSED / f"{stem}.parquet", index=False)
    fc.to_csv(_PROCESSED / f"{stem}.csv", index=False)

    # Full historical fit + forecast
    full = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    full.columns = ["date", "yhat", "yhat_lower", "yhat_upper", "trend"]
    full["actual"] = train_df.set_index("ds")["y"].reindex(full["date"]).values
    # Replace forecast portion with ensemble values
    fc_mask = full["date"] > last_hist
    n_fc = min(fc_mask.sum(), horizon)
    full.loc[fc_mask, "yhat"] = np.nan
    full.loc[full[fc_mask].index[:n_fc], "yhat"] = ensemble[:n_fc]
    full.loc[full[fc_mask].index[:n_fc], "yhat_lower"] = fc_lower[:n_fc]
    full.loc[full[fc_mask].index[:n_fc], "yhat_upper"] = fc_upper[:n_fc]
    full.to_parquet(_PROCESSED / f"{stem}_full.parquet", index=False)
    full.to_csv(_PROCESSED / f"{stem}_full.csv", index=False)

    logger.info(f"  12-month ensemble forecast ({config['name']}):")
    for _, row in fc.iterrows():
        logger.info(
            f"    {row['date']:%b %Y}: {row['forecast']:.1f} "
            f"[{row['forecast_lower']:.1f}, {row['forecast_upper']:.1f}]"
        )
    return fc


def run_txn_volume_models(run_cv: bool = True) -> dict:
    """Run all three transaction volume models."""
    logger.info("=== Transaction Volume Models ===")

    data = load_all()
    master = build_master(data)

    # For UPI, the target is in npci_upi, already merged into master.
    # For CC/DC vol, the targets are in rbi_psi_cards, already in master.

    configs = [
        ("cc_vol", CC_VOL_CONFIG),
        ("dc_vol", DC_VOL_CONFIG),
        ("upi_vol", UPI_VOL_CONFIG),
    ]

    results = {}
    for key, config in configs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Model: {config['name']}")
        logger.info(f"{'='*50}")

        train_df = build_training_df(master, config)

        model = _build_prophet(config, train_df)

        cv_metrics = pd.DataFrame()
        if run_cv:
            cv_metrics = _run_cv(model, config)

        forecast = _run_forecast(model, config, train_df, master)

        results[key] = {
            "model": model,
            "train_df": train_df,
            "forecast": forecast,
            "cv_metrics": cv_metrics,
        }

    # Auto-log
    try:
        for key, res in results.items():
            config = {"cc_vol": CC_VOL_CONFIG, "dc_vol": DC_VOL_CONFIG, "upi_vol": UPI_VOL_CONFIG}[key]
            log = RunLogger(f"txn_{key}")
            log.add("Target", config["target_col"])
            log.add("Training rows", len(res["train_df"]))
            if not res["cv_metrics"].empty:
                mape = res["cv_metrics"]["mape"] * 100
                log.add("CV MAPE mean", f"{mape.mean():.2f}%")
            fc = res["forecast"]
            log.add("Feb 2027 forecast", f"{fc['forecast'].iloc[-1]:.1f}")
            log.add("90% CI", f"[{fc['forecast_lower'].iloc[-1]:.1f}, {fc['forecast_upper'].iloc[-1]:.1f}]")
            log.add_section("Regressors", [f"{r.col} (lag={r.lag})" for r in config["regressors"]] or ["None (trend + seasonality only)"])
            log.save()
    except Exception:
        pass

    logger.info("\n=== Transaction Volume Models Complete ===")
    return results


if __name__ == "__main__":
    import sys
    run_cv = "--no-cv" not in sys.argv
    results = run_txn_volume_models(run_cv=run_cv)

    print("\n" + "=" * 55)
    print("TRANSACTION VOLUME FORECAST SUMMARY")
    print("=" * 55)
    for key, res in results.items():
        config_name = res["train_df"].columns  # just for the label
        fc = res["forecast"]
        print(f"\n{key.upper()}:")
        print(f"  Training rows: {len(res['train_df'])}")
        if not res["cv_metrics"].empty:
            mape = res["cv_metrics"]["mape"] * 100
            print(f"  CV MAPE: mean={mape.mean():.2f}% | range=[{mape.min():.2f}%, {mape.max():.2f}%]")
        print(f"  Feb 2027 forecast: {fc['forecast'].iloc[-1]:.1f}")
        print(f"  90% CI: [{fc['forecast_lower'].iloc[-1]:.1f}, {fc['forecast_upper'].iloc[-1]:.1f}]")
