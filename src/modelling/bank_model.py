"""
MIP Modelling — Bank-Level Ground-Up Model
==========================================
Runs individual Prophet models for the top 20 CC and DC issuers,
aggregates their forecasts with a residual bucket, and produces
the ground-up India total for cross-checking against PSI.

Architecture:
  1. For each top bank: fit Prophet (trend + seasonality + changepoints).
  2. For residual bucket (PSI − sum of top banks): fit a simpler Prophet.
  3. Aggregate all bank forecasts + residual = ground-up India total.
  4. Cross-check: ground-up total vs PSI aggregate (last overlap period).
  5. Save individual bank forecasts + aggregated ground-up outputs.

Run:
    uv run python -m src.modelling.bank_model          # CC + DC
    uv run python -m src.modelling.bank_model --cc     # CC only
    uv run python -m src.modelling.bank_model --dc     # DC only
    uv run python -m src.modelling.bank_model --no-cv  # skip CV

Outputs (all to data/processed/):
    bankwise_forecasts/
        cc_{bank_name}_forecast.parquet   — 12-month forecast per bank
        cc_{bank_name}_forecast.csv
        cc_residual_forecast.parquet
        dc_{bank_name}_forecast.parquet
        dc_{bank_name}_forecast.csv
        dc_residual_forecast.parquet
    groundup/
        groundup_cc.parquet               — aggregated ground-up CC
        groundup_cc.csv
        groundup_dc.parquet
        groundup_dc.csv
        groundup_summary.csv              — coverage + cross-check vs PSI
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.bank_config import (
    BANK_PROPHET_CONFIG,
    RESIDUAL_PROPHET_CONFIG,
    CC_BANK_CHANGEPOINTS,
    DC_BANK_CHANGEPOINTS,
    BANK_FORECAST_PERIODS,
    BANK_FORECAST_FREQ,
    BANK_CV_CONFIG,
    BANK_OUTPUT_DIR,
    GROUNDUP_OUTPUT_DIR,
)
from src.modelling.bank_data_prep import load_bank_data

from src.utils.run_logger import RunLogger

warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


# ── Output directory setup ─────────────────────────────────────────────────

def _ensure_output_dirs() -> tuple[Path, Path]:
    bank_dir    = _PROCESSED / BANK_OUTPUT_DIR
    groundup_dir = _PROCESSED / GROUNDUP_OUTPUT_DIR
    bank_dir.mkdir(parents=True, exist_ok=True)
    groundup_dir.mkdir(parents=True, exist_ok=True)
    return bank_dir, groundup_dir


# ── Individual bank model ──────────────────────────────────────────────────

def _fit_bank_model(
    bank_df: pd.DataFrame,
    changepoints: list[str],
    prophet_config: dict,
    bank_name: str,
) -> object:
    """Fit Prophet model for one bank. Returns fitted model."""
    from prophet import Prophet

    # Filter changepoints to those within the training window
    valid_cps = [
        pd.Timestamp(cp)
        for cp in changepoints
        if bank_df["ds"].min() < pd.Timestamp(cp) < bank_df["ds"].max()
    ]

    cfg = dict(prophet_config)
    if valid_cps:
        cfg["changepoints"] = valid_cps

    m = Prophet(**cfg)
    m.fit(bank_df)
    return m


def _run_bank_cv(
    model,
    bank_name: str,
    card_type: str,
) -> dict:
    """Run CV for one bank model. Returns MAPE stats.

    Caps per-fold MAPE at 100% before aggregating. A fold predicting 10x
    or 100x the actual value is a fit failure, not a precision signal —
    letting it through dominates the mean and produces uninterpretable
    six-figure "MAPE" values. Median is reported as the headline.
    """
    from prophet.diagnostics import cross_validation, performance_metrics

    try:
        cv_df  = cross_validation(
            model,
            initial=BANK_CV_CONFIG["initial"],
            period=BANK_CV_CONFIG["period"],
            horizon=BANK_CV_CONFIG["horizon"],
            parallel=BANK_CV_CONFIG.get("parallel", "threads"),
            disable_tqdm=True,
        )
        metrics = performance_metrics(cv_df)

        # Cap fit-failure folds at 100%
        n_capped = int((metrics["mape"] > 1.0).sum())
        metrics["mape"] = metrics["mape"].clip(upper=1.0)

        mape_mean   = metrics["mape"].mean()   * 100
        mape_median = metrics["mape"].median() * 100
        mape_min    = metrics["mape"].min()    * 100
        mape_max    = metrics["mape"].max()    * 100

        capped_note = f", {n_capped} folds capped" if n_capped else ""
        logger.info(
            f"  [{card_type.upper()}] {bank_name}: "
            f"CV MAPE median {mape_median:.2f}% (mean {mape_mean:.2f}%) "
            f"[{mape_min:.2f}%–{mape_max:.2f}%] "
            f"({len(metrics)} windows{capped_note})"
        )
        return {
            "bank_name":   bank_name,
            "card_type":   card_type,
            "mape_mean":   mape_mean,
            "mape_median": mape_median,
            "mape_min":    mape_min,
            "mape_max":    mape_max,
            "cv_windows":  len(metrics),
            "n_capped":    n_capped,
        }
    except Exception as e:
        logger.warning(f"  [{card_type.upper()}] {bank_name}: CV failed — {e}")
        return {
            "bank_name":   bank_name,
            "card_type":   card_type,
            "mape_mean":   None,
            "mape_median": None,
        }


def _forecast_bank(
    model,
    bank_df: pd.DataFrame,
    bank_name: str,
    card_type: str,
    bank_dir: Path,
) -> pd.DataFrame:
    """Generate forecast for one bank. Saves and returns forecast df."""
    future = model.make_future_dataframe(
        periods=BANK_FORECAST_PERIODS,
        freq=BANK_FORECAST_FREQ,
    )
    forecast = model.predict(future)

    last_hist = bank_df["ds"].max()
    fc = forecast[forecast["ds"] > last_hist][[
        "ds", "yhat", "yhat_lower", "yhat_upper", "trend"
    ]].copy()
    fc.columns = ["date", "forecast", "forecast_lower", "forecast_upper", "trend"]
    fc["bank_name"]  = bank_name
    fc["card_type"]  = card_type

    # Also save the full historical fit + forecast (for dashboard)
    full = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    full.columns = ["date", "yhat", "yhat_lower", "yhat_upper"]
    full["actual"] = bank_df.set_index("ds")["y"].reindex(full["date"]).values
    full["bank_name"] = bank_name
    full["card_type"] = card_type

    # Sanitise bank name for filename
    safe_name = bank_name.lower().replace(" ", "_").replace(".", "").replace("/", "_")
    stem = f"{card_type}_{safe_name}"

    fc.to_parquet(bank_dir / f"{stem}_forecast.parquet", index=False)
    fc.to_csv(bank_dir / f"{stem}_forecast.csv", index=False)
    full.to_parquet(bank_dir / f"{stem}_full.parquet", index=False)

    return fc


# ── Residual model ─────────────────────────────────────────────────────────

def _run_residual_model(
    residual_df: pd.DataFrame,
    card_type: str,
    bank_dir: Path,
) -> pd.DataFrame:
    """Fit residual model and return 12-month forecast.

    The residual_df here is PSI − sum(top banks), not a sum of small
    bankwise banks. See build_residual_prophet_df.
    """
    from prophet import Prophet

    logger.info(f"  [{card_type.upper()}] Fitting residual bucket model...")
    m = Prophet(**RESIDUAL_PROPHET_CONFIG)
    m.fit(residual_df)

    future = m.make_future_dataframe(
        periods=BANK_FORECAST_PERIODS,
        freq=BANK_FORECAST_FREQ,
    )
    forecast = m.predict(future)

    last_hist = residual_df["ds"].max()
    fc = forecast[forecast["ds"] > last_hist][[
        "ds", "yhat", "yhat_lower", "yhat_upper"
    ]].copy()
    fc.columns = ["date", "forecast", "forecast_lower", "forecast_upper"]
    fc["bank_name"] = "_RESIDUAL"
    fc["card_type"] = card_type

    fc.to_parquet(bank_dir / f"{card_type}_residual_forecast.parquet", index=False)
    fc.to_csv(bank_dir / f"{card_type}_residual_forecast.csv", index=False)
    logger.info(
        f"  [{card_type.upper()}] Residual forecast: "
        f"{fc['forecast'].iloc[0]:,.0f} → {fc['forecast'].iloc[-1]:,.0f}"
    )
    return fc


# ── Aggregation + cross-check ──────────────────────────────────────────────

def _aggregate_groundup(
    bank_forecasts: list[pd.DataFrame],
    residual_fc: pd.DataFrame,
    card_type: str,
    groundup_dir: Path,
) -> pd.DataFrame:
    """Sum all bank forecasts + residual = ground-up India total."""
    all_fc = bank_forecasts + [residual_fc]
    combined = pd.concat(all_fc, ignore_index=True)

    groundup = (
        combined.groupby("date")
        .agg(
            forecast=       ("forecast",       "sum"),
            forecast_lower= ("forecast_lower", "sum"),
            forecast_upper= ("forecast_upper", "sum"),
            n_banks=        ("bank_name",      "count"),
        )
        .reset_index()
    )
    groundup["card_type"] = card_type

    groundup.to_parquet(groundup_dir / f"groundup_{card_type}.parquet", index=False)
    groundup.to_csv(groundup_dir / f"groundup_{card_type}.csv", index=False)
    logger.info(
        f"  [{card_type.upper()}] Ground-up aggregate: "
        f"{groundup['forecast'].iloc[0]:,.0f} → {groundup['forecast'].iloc[-1]:,.0f} "
        f"(across {groundup['n_banks'].iloc[0]} components)"
    )
    return groundup


def _cross_check_vs_psi(
    groundup: pd.DataFrame,
    card_type: str,
    groundup_dir: Path,
) -> pd.DataFrame:
    """Compare ground-up total against PSI aggregate for the overlap period."""
    psi_path = _PROCESSED / "rbi_psi_cards.parquet"
    if not psi_path.exists():
        logger.warning("PSI parquet not found — skipping cross-check.")
        return pd.DataFrame()

    psi = pd.read_parquet(psi_path)
    psi["date"] = pd.to_datetime(psi["date"]).dt.to_period("M").dt.to_timestamp()

    target_col = (
        "credit_cards_outstanding_lakh"
        if card_type == "cc"
        else "debit_cards_outstanding_lakh"
    )
    psi_series = psi[["date", target_col]].dropna().rename(
        columns={target_col: "psi_lakh"}
    )

    # Convert ground-up to same unit (individual cards → lakh)
    gu = groundup[["date", "forecast"]].rename(columns={"forecast": "groundup_cards"})
    gu["groundup_lakh"] = gu["groundup_cards"] / 1e5

    merged = psi_series.merge(gu, on="date", how="inner")
    merged["pct_diff"] = (merged["groundup_lakh"] - merged["psi_lakh"]) / merged["psi_lakh"] * 100

    if merged.empty:
        logger.warning(f"  [{card_type.upper()}] No overlap between ground-up and PSI dates.")
        return pd.DataFrame()

    # Note: bankwise ends May 2025, PSI ends Feb 2026
    # Cross-check is on the overlap period (bankwise forecast vs PSI actual)
    logger.info(f"\n  [{card_type.upper()}] GROUND-UP vs PSI CROSS-CHECK:")
    logger.info(f"  {'Date':<12} {'PSI (lakh)':>14} {'Ground-up (lakh)':>18} {'Diff %':>8}")
    logger.info(f"  {'-'*56}")
    for _, row in merged.tail(6).iterrows():
        logger.info(
            f"  {row['date']:%b %Y}     {row['psi_lakh']:>12.1f}     "
            f"{row['groundup_lakh']:>14.1f}     {row['pct_diff']:>+6.1f}%"
        )

    avg_diff = merged["pct_diff"].abs().mean()
    logger.info(
        f"\n  [{card_type.upper()}] Average absolute divergence vs PSI: {avg_diff:.1f}%"
    )
    if avg_diff < 10:
        logger.success(f"  [{card_type.upper()}] Cross-check PASSED (divergence <10%)")
    elif avg_diff < 20:
        logger.warning(f"  [{card_type.upper()}] Cross-check marginal (divergence 10–20%)")
    else:
        logger.error(
            f"  [{card_type.upper()}] Cross-check FAILED (divergence >20%) — "
            f"investigate coverage gaps or unit mismatches"
        )

    merged.to_csv(groundup_dir / f"crosscheck_{card_type}.csv", index=False)
    return merged


# ── Coverage % vs PSI ─────────────────────────────────────────────────────

def _log_coverage_vs_psi(data: dict, card_type: str) -> float | None:
    """Compute and log: sum(top 20 latest outstanding) / PSI total * 100.

    Returns coverage percentage, or None if PSI is unavailable.
    """
    try:
        psi_path = _PROCESSED / "rbi_psi_cards.parquet"
        if not psi_path.exists():
            logger.warning("PSI parquet not found -- cannot compute coverage %.")
            return None

        psi = pd.read_parquet(psi_path)
        psi["date"] = pd.to_datetime(psi["date"]).dt.to_period("M").dt.to_timestamp()
        psi_col = (
            "credit_cards_outstanding_lakh"
            if card_type == "cc"
            else "debit_cards_outstanding_lakh"
        )

        df = data["df"]
        target_col = data["target_col"]
        top_banks = data["top_banks"]

        # Latest date in bankwise
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]

        top_sum_cards = latest[latest["bank_name"].isin(top_banks)][target_col].sum()
        top_sum_lakh = top_sum_cards / 1e5

        psi_latest = psi[psi["date"] == latest_date]
        if psi_latest.empty:
            # Try closest date
            psi_latest = psi[psi["date"] <= latest_date].tail(1)

        if psi_latest.empty or psi_latest[psi_col].isna().all():
            logger.warning(f"  [{card_type.upper()}] No PSI data at {latest_date.date()} for coverage calc.")
            return None

        psi_val = psi_latest[psi_col].iloc[0]
        coverage = (top_sum_lakh / psi_val) * 100

        logger.info(
            f"\n  [{card_type.upper()}] Ground-up coverage: {coverage:.1f}% of PSI total "
            f"(top {len(top_banks)} banks = {top_sum_lakh:,.1f} lakh, "
            f"PSI = {psi_val:,.1f} lakh, as of {latest_date:%b %Y})"
        )
        return coverage
    except Exception as e:
        logger.warning(f"  [{card_type.upper()}] Coverage calc failed: {e}")
        return None


# ── Main runner ────────────────────────────────────────────────────────────

def run_bank_model(
    card_type: str,
    run_cv: bool = True,
) -> dict:
    """Run the full bank-level ground-up pipeline for 'cc' or 'dc'.

    Returns dict with keys:
        bank_forecasts  — list of per-bank forecast DataFrames
        residual_fc     — residual forecast DataFrame
        groundup        — aggregated ground-up India total
        crosscheck      — comparison vs PSI
        cv_results      — list of CV MAPE dicts per bank
    """
    assert card_type in ("cc", "dc")
    changepoints = CC_BANK_CHANGEPOINTS if card_type == "cc" else DC_BANK_CHANGEPOINTS

    bank_dir, groundup_dir = _ensure_output_dirs()

    logger.info(f"\n{'═'*55}")
    logger.info(f"BANK-LEVEL GROUND-UP MODEL — {card_type.upper()}")
    logger.info(f"{'═'*55}")

    # Load and prepare data
    data = load_bank_data(card_type)
    top_banks   = data["top_banks"]
    bank_dfs    = data["bank_dfs"]
    residual_df = data["residual_df"]

    bank_forecasts: list[pd.DataFrame] = []
    cv_results:     list[dict]         = []

    # Fit individual bank models
    for bank_name in top_banks:
        df = bank_dfs.get(bank_name)
        if df is None:
            logger.warning(f"  [{card_type.upper()}] {bank_name}: skipped (insufficient data)")
            continue

        logger.info(f"\n  [{card_type.upper()}] {bank_name} ({len(df)} months)")

        model = _fit_bank_model(
            df, changepoints, BANK_PROPHET_CONFIG, bank_name
        )

        if run_cv and len(df) >= 72:  # CV needs enough data — skip if <72m
            cv_result = _run_bank_cv(model, bank_name, card_type)
            cv_results.append(cv_result)
        elif run_cv:
            logger.info(
                f"  [{card_type.upper()}] {bank_name}: CV skipped "
                f"(<72 months, not enough for 36m initial + 6m horizon)"
            )

        fc = _forecast_bank(model, df, bank_name, card_type, bank_dir)
        bank_forecasts.append(fc)

    # Residual model
    residual_fc = _run_residual_model(residual_df, card_type, bank_dir)

    # Aggregate
    groundup = _aggregate_groundup(bank_forecasts, residual_fc, card_type, groundup_dir)

    # Cross-check vs PSI
    crosscheck = _cross_check_vs_psi(groundup, card_type, groundup_dir)

    # Save CV summary — use median as the headline (mean dominated by fit failures)
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        cv_df.to_csv(groundup_dir / f"bank_cv_summary_{card_type}.csv", index=False)
        valid = cv_df[cv_df["mape_median"].notna()]
        if not valid.empty:
            logger.info(
                f"\n  [{card_type.upper()}] Bank-level CV summary: "
                f"median across banks {valid['mape_median'].median():.2f}% | "
                f"range [{valid['mape_median'].min():.2f}%–{valid['mape_median'].max():.2f}%]"
            )

    # Coverage %: top banks' latest outstanding / PSI total
    coverage_pct = _log_coverage_vs_psi(data, card_type)

    # Auto-log
    try:
        log = RunLogger(f"bank_{card_type}")
        log.add("Card type", card_type.upper())
        log.add("Top banks modelled", len(top_banks))
        gu_latest = groundup[groundup["date"] == groundup["date"].max()]
        log.add("Ground-up forecast (Feb 2027)", f"{gu_latest['forecast'].sum():,.0f} cards")
        log.add("90% CI", f"[{gu_latest['forecast_lower'].sum():,.0f}, {gu_latest['forecast_upper'].sum():,.0f}]")
        if coverage_pct is not None:
            log.add("Coverage vs PSI", f"{coverage_pct:.1f}%")
        if cv_results:
            valid = [r for r in cv_results if r.get("mape_median") is not None]
            if valid:
                medians = [r["mape_median"] for r in valid]
                log.add("Bank CV MAPE median", f"{pd.Series(medians).median():.2f}%")
                log.add("Bank CV MAPE range", f"[{min(medians):.2f}%, {max(medians):.2f}%]")
        log.save()
    except Exception:
        pass

    logger.success(f"\n  [{card_type.upper()}] Ground-up model complete.")
    return {
        "bank_forecasts": bank_forecasts,
        "residual_fc":    residual_fc,
        "groundup":       groundup,
        "crosscheck":     crosscheck,
        "cv_results":     cv_results,
        "coverage_pct":   coverage_pct,
    }


def run_all_bank_models(run_cv: bool = True) -> dict:
    """Run ground-up models for both CC and DC."""
    results = {}
    for ct in ["cc", "dc"]:
        results[ct] = run_bank_model(ct, run_cv=run_cv)

    # Combined ground-up summary
    logger.info("\n" + "═" * 55)
    logger.info("GROUND-UP SUMMARY — CC + DC")
    logger.info("═" * 55)
    for ct in ["cc", "dc"]:
        gu = results[ct]["groundup"]
        logger.info(
            f"{ct.upper()} Feb 2027 forecast: "
            f"{gu[gu['date'] == gu['date'].max()]['forecast'].sum():,.0f} cards"
        )
    return results


if __name__ == "__main__":
    import sys
    run_cv  = "--no-cv" not in sys.argv
    run_cc  = "--dc" not in sys.argv
    run_dc  = "--cc" not in sys.argv

    if run_cc and run_dc:
        results = run_all_bank_models(run_cv=run_cv)
    elif run_cc:
        results = {"cc": run_bank_model("cc", run_cv=run_cv)}
    else:
        results = {"dc": run_bank_model("dc", run_cv=run_cv)}

    print("\n" + "=" * 55)
    print("GROUND-UP MODEL COMPLETE")
    print("=" * 55)
    for ct, res in results.items():
        gu = res["groundup"]
        latest = gu[gu["date"] == gu["date"].max()]
        print(f"\n{ct.upper()} ground-up Feb 2027:")
        print(f"  Forecast: {latest['forecast'].sum():,.0f} cards")
        print(f"  90% CI:   [{latest['forecast_lower'].sum():,.0f} – {latest['forecast_upper'].sum():,.0f}]")

        cc = res["crosscheck"]
        if not cc.empty:
            print(f"  PSI divergence (avg): {cc['pct_diff'].abs().mean():.1f}%")

        if res["cv_results"]:
            cv_valid = [r for r in res["cv_results"] if r.get("mape_median")]
            if cv_valid:
                medians = [r["mape_median"] for r in cv_valid]
                print(f"  Bank CV MAPE (median across banks): {pd.Series(medians).median():.2f}%")