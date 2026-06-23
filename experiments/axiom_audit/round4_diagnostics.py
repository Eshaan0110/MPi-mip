"""
AXIOM Round 4 — Deep Quantitative Diagnostics
Runs sequentially, saves after each test so partial results survive.
"""
import sys, json, warnings, traceback
from pathlib import Path
import numpy as np, pandas as pd
from loguru import logger

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = Path(__file__).parent / "round4_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

from src.modelling.data_prep import load_all, build_master, build_training_df
from src.modelling.model_config import CC_CONFIG, DC_CONFIG, CV_CONFIG, STRUCTURAL_EVENTS, RegressorSpec
from src.modelling.aggregate_model import build_prophet_model

DATA = None
MASTER = None

def get_data():
    global DATA, MASTER
    if DATA is None:
        DATA = load_all()
        MASTER = build_master(DATA)
    return DATA, MASTER

def save(name, result):
    with open(RESULTS_DIR / f"{name}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.success(f"Saved {name}.json")


def test_stationarity():
    from statsmodels.tsa.stattools import adfuller, kpss
    _, master = get_data()
    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(master, config)
        y = train_df["y"].values
        adf_stat, adf_p, *_ = adfuller(y, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(y, regression="ct", nlags="auto")
        dy = np.diff(y)
        adf_d_stat, adf_d_p, *_ = adfuller(dy, autolag="AIC")
        results[label] = {
            "level_adf_stat": round(adf_stat, 4), "level_adf_p": round(adf_p, 4),
            "level_kpss_stat": round(kpss_stat, 4), "level_kpss_p": round(kpss_p, 4),
            "diff_adf_stat": round(adf_d_stat, 4), "diff_adf_p": round(adf_d_p, 4),
            "interpretation": ("non-stationary in levels" if adf_p > 0.05 else "stationary in levels")
                + "; " + ("stationary after differencing" if adf_d_p < 0.05 else "STILL non-stationary"),
        }
        logger.info(f"[STATIONARITY] {label}: ADF p={adf_p:.4f}, KPSS p={kpss_p:.4f}, diff ADF p={adf_d_p:.4f}")
    return results


def test_residuals():
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import acorr_ljungbox
    _, master = get_data()
    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(master, config)
        model = build_prophet_model(config, train_df)
        future = model.make_future_dataframe(periods=0, freq="MS")
        for col in train_df.columns:
            if col not in ["ds", "y"] and col not in future.columns:
                future = future.merge(train_df[["ds", col]], on="ds", how="left")
        pred = model.predict(future)
        pred = pred.merge(train_df[["ds", "y"]], on="ds")
        residuals = (pred["y"] - pred["yhat"]).values
        dw = durbin_watson(residuals)
        lb = acorr_ljungbox(residuals, lags=[6, 12], return_df=True)
        skew = float(pd.Series(residuals).skew())
        kurt = float(pd.Series(residuals).kurtosis())
        results[label] = {
            "n": len(residuals), "mean": round(np.mean(residuals), 4),
            "std": round(np.std(residuals), 4),
            "skewness": round(skew, 4), "kurtosis": round(kurt, 4),
            "durbin_watson": round(dw, 4),
            "dw_interp": "positive autocorrelation" if dw < 1.5 else "ok" if dw < 2.5 else "negative autocorrelation",
            "ljung_box_lag6_p": round(float(lb.iloc[0]["lb_pvalue"]), 4),
            "ljung_box_lag12_p": round(float(lb.iloc[1]["lb_pvalue"]), 4),
        }
        logger.info(f"[RESID] {label}: DW={dw:.3f}, LB6 p={lb.iloc[0]['lb_pvalue']:.4f}, skew={skew:.3f}")
    return results


def test_regressor_ablation():
    from prophet.diagnostics import cross_validation, performance_metrics
    _, master = get_data()
    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(master, config)
        model_full = build_prophet_model(config, train_df)
        cv_full = cross_validation(model_full, initial=CV_CONFIG["initial"],
                                    period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                    parallel="processes", disable_tqdm=True)
        mape_full = performance_metrics(cv_full)["mape"].mean() * 100
        ablation = {"baseline_mape": round(mape_full, 3)}

        # Regressor ablation
        for spec in config["regressors"]:
            col = f"{spec.col}_lag{spec.lag}" if spec.lag > 0 else spec.col
            logger.info(f"[ABLATION] {label}: without {col}")
            config_no = dict(config)
            config_no["regressors"] = [r for r in config["regressors"] if r is not spec]
            try:
                model_no = build_prophet_model(config_no, train_df)
                cv_no = cross_validation(model_no, initial=CV_CONFIG["initial"],
                                          period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                          parallel="processes", disable_tqdm=True)
                mape_no = performance_metrics(cv_no)["mape"].mean() * 100
                delta = mape_no - mape_full
                ablation[f"without_{col}"] = {
                    "mape": round(mape_no, 3), "delta_pp": round(delta, 3),
                    "verdict": "KEEP" if delta > 0.1 else "REMOVE" if delta < -0.1 else "MARGINAL"
                }
                logger.info(f"  -> delta={delta:+.3f}pp")
            except Exception as e:
                ablation[f"without_{col}"] = {"error": str(e)}

        # Event dummy ablation
        event_cols = [c for c in train_df.columns if c.startswith("event_")]
        for ecol in event_cols:
            logger.info(f"[ABLATION] {label}: without {ecol}")
            train_no = train_df.drop(columns=[ecol])
            try:
                model_no = build_prophet_model(config, train_no)
                cv_no = cross_validation(model_no, initial=CV_CONFIG["initial"],
                                          period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                          parallel="processes", disable_tqdm=True)
                mape_no = performance_metrics(cv_no)["mape"].mean() * 100
                delta = mape_no - mape_full
                ablation[f"without_{ecol}"] = {
                    "mape": round(mape_no, 3), "delta_pp": round(delta, 3),
                    "verdict": "KEEP" if delta > 0.1 else "REMOVE" if delta < -0.1 else "MARGINAL"
                }
                logger.info(f"  -> delta={delta:+.3f}pp")
            except Exception as e:
                ablation[f"without_{ecol}"] = {"error": str(e)}

        results[label] = ablation
    return results


def test_ci_calibration():
    from prophet.diagnostics import cross_validation
    _, master = get_data()
    results = {}
    for label, config in [("CC", CC_CONFIG), ("DC", DC_CONFIG)]:
        train_df = build_training_df(master, config)
        model = build_prophet_model(config, train_df)
        cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                                  period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                  parallel="processes", disable_tqdm=True)
        cv_df["horizon_days"] = (cv_df["ds"] - cv_df["cutoff"]).dt.days
        cv_df["in_ci"] = (cv_df["y"] >= cv_df["yhat_lower"]) & (cv_df["y"] <= cv_df["yhat_upper"])
        cal = {"nominal": 90.0}
        for bname, lo, hi in [("month_1_2", 0, 62), ("month_3_4", 62, 124), ("month_5_6", 124, 190)]:
            mask = (cv_df["horizon_days"] >= lo) & (cv_df["horizon_days"] < hi)
            sub = cv_df[mask]
            if len(sub) > 0:
                emp = sub["in_ci"].mean() * 100
                cal[bname] = {"empirical_pct": round(emp, 1), "n": len(sub)}
                logger.info(f"[CI] {label} {bname}: {emp:.1f}% ({len(sub)} obs)")
        overall = cv_df["in_ci"].mean() * 100
        cal["overall"] = round(overall, 1)
        cal["interpretation"] = "well-calibrated" if 85 <= overall <= 95 else "over-confident" if overall < 85 else "conservative"
        results[label] = cal
    return results


def test_lag_sensitivity():
    from prophet.diagnostics import cross_validation, performance_metrics
    _, master = get_data()
    results = {}
    for lag in [0, 3, 6, 9, 12]:
        logger.info(f"[LAG] repo_rate lag={lag}")
        config_test = dict(CC_CONFIG)
        config_test["regressors"] = [
            RegressorSpec(col="repo_rate", standardize=True, lag=lag, fill_method="ffill", mode="additive"),
        ]
        train_df = build_training_df(master, config_test)
        model = build_prophet_model(config_test, train_df)
        cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                                  period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                  parallel="processes", disable_tqdm=True)
        mape = performance_metrics(cv_df)["mape"].mean() * 100
        results[f"lag_{lag}"] = round(mape, 3)
        logger.info(f"  MAPE={mape:.3f}%")

    config_none = dict(CC_CONFIG)
    config_none["regressors"] = []
    train_df = build_training_df(master, config_none)
    model = build_prophet_model(config_none, train_df)
    cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                              period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                              parallel="processes", disable_tqdm=True)
    mape = performance_metrics(cv_df)["mape"].mean() * 100
    results["no_regressor"] = round(mape, 3)
    return results


def test_event_sensitivity():
    from prophet.diagnostics import cross_validation, performance_metrics
    _, master = get_data()
    results = {}
    event = "rbi_credit_tightening"
    orig_date = STRUCTURAL_EVENTS[event]["date"]
    for shift in [-2, -1, 0, 1, 2]:
        shifted = pd.Timestamp(orig_date) + pd.DateOffset(months=shift)
        logger.info(f"[EVENT] {event} shift {shift:+d}m -> {shifted:%Y-%m-%d}")
        STRUCTURAL_EVENTS[event]["date"] = shifted.strftime("%Y-%m-%d")
        train_df = build_training_df(master, CC_CONFIG)
        model = build_prophet_model(CC_CONFIG, train_df)
        cv_df = cross_validation(model, initial=CV_CONFIG["initial"],
                                  period=CV_CONFIG["period"], horizon=CV_CONFIG["horizon"],
                                  parallel="processes", disable_tqdm=True)
        mape = performance_metrics(cv_df)["mape"].mean() * 100
        results[f"shift_{shift:+d}m"] = round(mape, 3)
        STRUCTURAL_EVENTS[event]["date"] = orig_date
    return results


def test_dc_regressors():
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _, master = get_data()
    train_df = build_training_df(master, DC_CONFIG)
    reg_cols = [c for c in ["debit_card_vol_lakh", "debit_card_pos_vol_lakh"] if c in train_df.columns]
    results = {}
    if len(reg_cols) >= 2:
        X = train_df[reg_cols].dropna()
        corr = float(X[reg_cols[0]].corr(X[reg_cols[1]]))
        results["correlation"] = round(corr, 4)
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        Xs = np.column_stack([np.ones(len(Xs)), Xs])
        for i, col in enumerate(reg_cols, 1):
            vif = variance_inflation_factor(Xs, i)
            results[f"vif_{col}"] = round(vif, 2)
        results["multicollinearity_risk"] = "HIGH" if corr > 0.8 else "MODERATE" if corr > 0.6 else "LOW"
    return results


def test_granger():
    from statsmodels.tsa.stattools import grangercausalitytests
    _, master = get_data()
    results = {}
    # CC: repo_rate -> credit_cards_outstanding
    train_cc = build_training_df(master, CC_CONFIG)
    y = np.diff(train_cc["y"].values)
    if "repo_rate_lag9" in train_cc.columns:
        x = np.diff(train_cc["repo_rate_lag9"].values)
    else:
        x = np.diff(train_cc.get("repo_rate", train_cc["y"]).values)
    df_test = pd.DataFrame({"y": y, "x": x})
    df_test = df_test.dropna()
    try:
        gc = grangercausalitytests(df_test[["y", "x"]], maxlag=4, verbose=False)
        results["cc_repo_rate"] = {}
        for lag in range(1, 5):
            p = gc[lag][0]["ssr_ftest"][1]
            results["cc_repo_rate"][f"lag_{lag}_p"] = round(p, 4)
        results["cc_repo_rate"]["significant_at_005"] = any(gc[l][0]["ssr_ftest"][1] < 0.05 for l in range(1, 5))
    except Exception as e:
        results["cc_repo_rate"] = {"error": str(e)}

    # DC: debit_card_vol -> debit_cards_outstanding
    train_dc = build_training_df(master, DC_CONFIG)
    if "debit_card_vol_lakh" in train_dc.columns:
        y = np.diff(train_dc["y"].values)
        x = np.diff(train_dc["debit_card_vol_lakh"].values)
        df_test = pd.DataFrame({"y": y, "x": x}).dropna()
        try:
            gc = grangercausalitytests(df_test[["y", "x"]], maxlag=4, verbose=False)
            results["dc_vol"] = {}
            for lag in range(1, 5):
                p = gc[lag][0]["ssr_ftest"][1]
                results["dc_vol"][f"lag_{lag}_p"] = round(p, 4)
            results["dc_vol"]["significant_at_005"] = any(gc[l][0]["ssr_ftest"][1] < 0.05 for l in range(1, 5))
        except Exception as e:
            results["dc_vol"] = {"error": str(e)}
    return results


TESTS = [
    ("stationarity", test_stationarity),
    ("residuals", test_residuals),
    ("granger", test_granger),
    ("dc_regressors", test_dc_regressors),
    ("regressor_ablation", test_regressor_ablation),
    ("ci_calibration", test_ci_calibration),
    ("lag_sensitivity", test_lag_sensitivity),
    ("event_sensitivity", test_event_sensitivity),
]


def main():
    all_results = {}
    for name, fn in TESTS:
        logger.info(f"\n{'='*40}\n  TEST: {name}\n{'='*40}")
        try:
            result = fn()
            all_results[name] = result
            save(name, result)
        except Exception:
            logger.error(f"TEST {name} FAILED:\n{traceback.format_exc()}")
            all_results[name] = {"error": traceback.format_exc()}

    save("all_diagnostics", all_results)
    print("\n" + json.dumps(all_results, indent=2, default=str))


if __name__ == "__main__":
    main()
