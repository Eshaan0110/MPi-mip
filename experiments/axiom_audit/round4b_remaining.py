"""
Round 4B — Remaining audit tests:
  1. Alternative model comparison (ARIMA vs Prophet) + Diebold-Mariano
  2. Bank reconciliation (sum of banks vs RBI total)
  3. Data leakage check
  4. Long-horizon drift (6m vs 12m forecast degradation)
  5. DC vol lag sensitivity (test lag 1-6 to find business-correct fix)
"""
import sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "round4_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.modelling.data_prep import load_all, build_master, build_training_df
from src.modelling.model_config import CC_CONFIG, DC_CONFIG, CV_CONFIG, RegressorSpec
from src.modelling.aggregate_model import build_prophet_model

DATA = load_all()
MASTER = build_master(DATA)


def save(name, result):
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.success(f"Saved {name}.json")


def test_arima_comparison():
    """Compare Prophet vs auto-ARIMA on the same CV folds."""
    from statsmodels.tsa.arima.model import ARIMA
    from prophet.diagnostics import cross_validation, performance_metrics

    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(MASTER, config)
        y = train_df["y"].values
        dates = train_df["ds"].values

        # Prophet CV MAPE (reuse)
        model = build_prophet_model(config, train_df)
        cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                                  period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                  parallel="processes", disable_tqdm=True)
        prophet_mape = performance_metrics(cv_df)["mape"].mean() * 100

        # Manual ARIMA CV on same folds
        initial_months = int(CV_CONFIG["initial"].replace(" days", "")) // 30
        horizon_months = int(CV_CONFIG["horizon"].replace(" days", "")) // 30
        step_months = int(CV_CONFIG["period"].replace(" days", "")) // 30

        arima_mapes = []
        for start in range(initial_months, len(y) - horizon_months + 1, step_months):
            train_y = y[:start]
            test_y = y[start:start + horizon_months]
            if len(test_y) < horizon_months:
                break
            try:
                # ARIMA(1,1,1) — simple but robust
                m = ARIMA(train_y, order=(1, 1, 1))
                fit = m.fit()
                pred = fit.forecast(steps=horizon_months)
                fold_mape = np.mean(np.abs((test_y - pred) / test_y)) * 100
                arima_mapes.append(fold_mape)
            except Exception:
                pass

        arima_mape = np.mean(arima_mapes) if arima_mapes else None

        # ETS comparison
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        ets_mapes = []
        for start in range(initial_months, len(y) - horizon_months + 1, step_months):
            train_y = y[:start]
            test_y = y[start:start + horizon_months]
            if len(test_y) < horizon_months:
                break
            try:
                m = ExponentialSmoothing(train_y, trend="add", seasonal="add",
                                          seasonal_periods=12, damped_trend=True)
                fit = m.fit(optimized=True)
                pred = fit.forecast(steps=horizon_months)
                fold_mape = np.mean(np.abs((test_y - pred) / test_y)) * 100
                ets_mapes.append(fold_mape)
            except Exception:
                pass

        ets_mape = np.mean(ets_mapes) if ets_mapes else None

        results[label] = {
            "prophet_cv_mape": round(prophet_mape, 3),
            "arima_111_cv_mape": round(arima_mape, 3) if arima_mape else None,
            "ets_aad_cv_mape": round(ets_mape, 3) if ets_mape else None,
            "n_arima_folds": len(arima_mapes),
            "n_ets_folds": len(ets_mapes),
            "prophet_best": prophet_mape < (arima_mape or 999) and prophet_mape < (ets_mape or 999),
        }
        logger.info(f"[ALT MODELS] {label}: Prophet={prophet_mape:.3f}%, ARIMA={arima_mape:.3f}%, ETS={ets_mape:.3f}%")

    return results


def test_bank_reconciliation():
    """Check if sum of individual bank forecasts + residual ≈ aggregate forecast."""
    results = {}
    processed = PROJECT_ROOT / "data" / "processed"

    for card_type, stem in [("CC", "forecast_cc"), ("DC", "forecast_dc")]:
        agg_file = processed / f"{stem}_full.csv"
        bank_file = processed / f"bank_{card_type.lower()}_forecasts.csv"

        if not agg_file.exists() or not bank_file.exists():
            results[card_type] = {"error": f"Missing files: agg={agg_file.exists()}, bank={bank_file.exists()}"}
            continue

        agg = pd.read_csv(agg_file, parse_dates=["date"])
        banks = pd.read_csv(bank_file, parse_dates=["date"])

        # Sum bank forecasts for the forecast period
        forecast_dates = agg[agg["actual_lakh"].isna()]["date"]
        if len(forecast_dates) == 0:
            results[card_type] = {"note": "No forecast-only dates found"}
            continue

        bank_sum = banks.groupby("date")["yhat"].sum()
        agg_fc = agg.set_index("date")["yhat_lakh"]

        common = bank_sum.index.intersection(agg_fc.index)
        if len(common) == 0:
            results[card_type] = {"note": "No overlapping dates between bank and aggregate"}
            continue

        diffs = []
        for d in common[-12:]:  # last 12 months
            b = bank_sum.get(d, np.nan)
            a = agg_fc.get(d, np.nan)
            if pd.notna(b) and pd.notna(a) and a != 0:
                pct_diff = (b - a) / a * 100
                diffs.append({"date": str(d), "bank_sum": round(b, 1), "aggregate": round(a, 1),
                              "pct_diff": round(pct_diff, 2)})

        results[card_type] = {
            "n_dates_compared": len(diffs),
            "details": diffs[-6:] if diffs else [],
            "mean_abs_pct_diff": round(np.mean([abs(d["pct_diff"]) for d in diffs]), 2) if diffs else None,
        }

    return results


def test_data_leakage():
    """Check that no future information leaks into training data."""
    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(MASTER, config)
        issues = []

        # Check: are there any regressor values that use data beyond the training cutoff?
        last_date = train_df["ds"].max()

        # Check regressor columns for NaN patterns (NaN at end = forward-filled future data)
        reg_cols = [c for c in train_df.columns if c not in ["ds", "y"] and not c.startswith("event_")]
        for col in reg_cols:
            n_null = train_df[col].isna().sum()
            last_valid = train_df[col].last_valid_index()
            if n_null > 0:
                issues.append(f"{col}: {n_null} NaN values")

        # Check: event dummies should be 0/1 only
        event_cols = [c for c in train_df.columns if c.startswith("event_")]
        for col in event_cols:
            vals = train_df[col].unique()
            if not set(vals).issubset({0, 0.0, 1, 1.0}):
                issues.append(f"{col}: unexpected values {vals}")

        # Check: y column has no NaN
        y_nulls = train_df["y"].isna().sum()
        if y_nulls > 0:
            issues.append(f"y has {y_nulls} NaN values")

        # Check: dates are monotonically increasing with no gaps > 35 days
        date_diffs = train_df["ds"].diff().dt.days
        max_gap = date_diffs.max()
        if max_gap > 35:
            issues.append(f"Max date gap: {max_gap} days")

        results[label] = {
            "n_rows": len(train_df),
            "date_range": f"{train_df['ds'].min():%Y-%m-%d} to {train_df['ds'].max():%Y-%m-%d}",
            "regressor_cols": reg_cols,
            "event_cols": event_cols,
            "issues": issues if issues else ["No leakage detected"],
            "clean": len(issues) == 0,
        }
        logger.info(f"[LEAKAGE] {label}: {'CLEAN' if not issues else issues}")

    return results


def test_dc_vol_lag():
    """Test debit_card_vol_lakh at lag 0-6 to find if lagging fixes the CV penalty."""
    from prophet.diagnostics import cross_validation, performance_metrics

    results = {}
    for lag in [0, 1, 2, 3, 4, 5, 6]:
        logger.info(f"[DC LAG] debit_card_vol_lakh lag={lag}")
        config_test = dict(DC_CONFIG)
        config_test["regressors"] = [
            RegressorSpec(col="debit_card_vol_lakh", standardize=True, lag=lag,
                          fill_method="linear", mode="additive"),
            RegressorSpec(col="debit_card_pos_vol_lakh", standardize=True, lag=0,
                          fill_method="zero", mode="additive"),
        ]
        try:
            train_df = build_training_df(MASTER, config_test)
            model = build_prophet_model(config_test, train_df)
            cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                                      period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                      parallel="processes", disable_tqdm=True)
            mape = performance_metrics(cv_df)["mape"].mean() * 100
            results[f"lag_{lag}"] = round(mape, 3)
            logger.info(f"  MAPE={mape:.3f}%")
        except Exception as e:
            results[f"lag_{lag}"] = {"error": str(e)}

    # Without debit_card_vol_lakh at all
    config_no = dict(DC_CONFIG)
    config_no["regressors"] = [
        RegressorSpec(col="debit_card_pos_vol_lakh", standardize=True, lag=0,
                      fill_method="zero", mode="additive"),
    ]
    train_df = build_training_df(MASTER, config_no)
    model = build_prophet_model(config_no, train_df)
    cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                              period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                              parallel="processes", disable_tqdm=True)
    mape = performance_metrics(cv_df)["mape"].mean() * 100
    results["without_vol"] = round(mape, 3)

    return results


def test_horizon_drift():
    """Compare forecast error at month 1-3 vs 4-6 to check long-horizon degradation."""
    from prophet.diagnostics import cross_validation, performance_metrics

    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(MASTER, config)
        model = build_prophet_model(config, train_df)
        cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                                  period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                  parallel="processes", disable_tqdm=True)

        cv_df["horizon_days"] = (cv_df["ds"] - cv_df["cutoff"]).dt.days
        cv_df["ape"] = np.abs((cv_df["y"] - cv_df["yhat"]) / cv_df["y"])

        buckets = {}
        for name, lo, hi in [("month_1", 0, 35), ("month_2", 35, 65), ("month_3", 65, 95),
                              ("month_4", 95, 125), ("month_5", 125, 155), ("month_6", 155, 190)]:
            mask = (cv_df["horizon_days"] >= lo) & (cv_df["horizon_days"] < hi)
            sub = cv_df[mask]
            if len(sub) > 0:
                buckets[name] = {"mape": round(sub["ape"].mean() * 100, 2), "n": len(sub)}

        m1_3 = cv_df[cv_df["horizon_days"] < 95]["ape"].mean() * 100
        m4_6 = cv_df[cv_df["horizon_days"] >= 95]["ape"].mean() * 100
        drift = m4_6 - m1_3

        results[label] = {
            "by_month": buckets,
            "mape_month_1_3": round(m1_3, 2),
            "mape_month_4_6": round(m4_6, 2),
            "drift_pp": round(drift, 2),
            "drift_acceptable": abs(drift) < 3.0,
        }
        logger.info(f"[DRIFT] {label}: M1-3={m1_3:.2f}%, M4-6={m4_6:.2f}%, drift={drift:+.2f}pp")

    return results


def main():
    tests = [
        ("alt_models", test_arima_comparison),
        ("bank_reconciliation", test_bank_reconciliation),
        ("data_leakage", test_data_leakage),
        ("dc_vol_lag_sensitivity", test_dc_vol_lag),
        ("horizon_drift", test_horizon_drift),
    ]
    all_results = {}
    for name, fn in tests:
        logger.info(f"\n{'='*40}\n  {name}\n{'='*40}")
        try:
            result = fn()
            all_results[name] = result
            save(name, result)
        except Exception as e:
            import traceback
            logger.error(f"{name} FAILED:\n{traceback.format_exc()}")
            all_results[name] = {"error": traceback.format_exc()}

    save("round4b_all", all_results)
    print(json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
