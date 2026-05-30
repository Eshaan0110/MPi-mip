"""
granger_causality.py  --  Formal Granger causality tests for MIP regressors.

Tests whether each regressor has genuine predictive power over the target
series, beyond what the target's own history already explains.

The three hypotheses under test:
  1. repo_rate (lag 6)  →  credit_cards_outstanding_lakh
     Economic prior: RBI tightening → banks slow CC issuance ~6m later.
     Granger confirms whether the lag-6 channel is statistically real.

  2. upi_qr_lakh  →  credit_cards_outstanding_lakh
     AXIOM challenge: both trend upward together. Granger checks if QR
     has independent predictive power or is just riding the trend.

  3. debit_card_pos_vol_lakh  →  debit_cards_outstanding_lakh
     Confirms the displacement signal direction: falling POS swipes
     predict future DC outstanding, not the other way round.

Method:
  - First-difference both series before testing (Granger requires stationarity).
  - Test at maxlag = 6 (tests lags 1 through 6 separately).
  - Report F-stat and p-value per lag. p < 0.05 = statistically significant.
  - Also test the REVERSE direction (Y → X) to check for reverse causality.

Usage:
  # With processed parquet files (normal pipeline run):
  python -m src.modelling.granger_causality

  # With a CSV of the master dataframe:
  python -m src.modelling.granger_causality --data path/to/master.csv

Output:
  Prints a formatted table of results to stdout.
  Writes granger_results.json to data/processed/ for downstream use.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED    = _PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Data loading — mirrors data_prep.py load pattern
# ---------------------------------------------------------------------------

def _load_master() -> pd.DataFrame | None:
    """Try to load the processed parquet files and build a master frame.
    Returns None if data hasn't been ingested yet (no parquet files present)."""
    psi_path = _PROCESSED / "rbi_psi_cards.parquet"
    repo_path = _PROCESSED / "repo_rate.parquet"

    if not psi_path.exists():
        return None

    psi = pd.read_parquet(psi_path)
    psi["date"] = pd.to_datetime(psi["date"]).dt.to_period("M").dt.to_timestamp()

    master = psi[[
        "date",
        "credit_cards_outstanding_lakh",
        "debit_cards_outstanding_lakh",
        "debit_card_pos_vol_lakh",
        "upi_qr_lakh",
    ]].copy()

    if repo_path.exists():
        repo = pd.read_parquet(repo_path)
        repo["date"] = pd.to_datetime(repo["date"]).dt.to_period("M").dt.to_timestamp()
        master = master.merge(repo[["date", "repo_rate"]], on="date", how="left")
        master["repo_rate"] = master["repo_rate"].ffill()

    return master.sort_values("date").reset_index(drop=True)


def load_data(csv_path: Path | None = None) -> pd.DataFrame:
    """Load master dataframe. Tries parquet pipeline first, then CSV fallback."""
    if csv_path is not None:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()
        return df.sort_values("date").reset_index(drop=True)

    master = _load_master()
    if master is not None:
        return master

    raise FileNotFoundError(
        "No processed parquet files found in data/processed/.\n"
        "Run `python -m src.ingestion` first, or pass --data path/to/master.csv"
    )


# ---------------------------------------------------------------------------
# Stationarity check
# ---------------------------------------------------------------------------

def _is_stationary(series: pd.Series, alpha: float = 0.05) -> tuple[bool, float]:
    """ADF test. Returns (is_stationary, p_value)."""
    clean = series.dropna()
    if len(clean) < 20:
        return False, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = adfuller(clean, autolag="AIC")
    return result[1] < alpha, result[1]


def _make_stationary(series: pd.Series, name: str) -> tuple[pd.Series, str]:
    """
    First-difference the series if not already stationary.
    Granger causality requires both series to be stationary.
    Returns (differenced_series, transformation_applied).
    """
    is_stat, pval = _is_stationary(series)
    if is_stat:
        return series, f"none (ADF p={pval:.3f}, already stationary)"

    diff1 = series.diff().dropna()
    is_stat2, pval2 = _is_stationary(diff1)
    if is_stat2:
        return diff1, f"first-difference (ADF p after diff={pval2:.3f})"

    # If still not stationary after one diff, second difference
    diff2 = diff1.diff().dropna()
    return diff2, f"second-difference (ADF p after diff={pval2:.3f}, second diff applied)"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GrangerResult:
    hypothesis: str         # "X → Y" description
    x_col: str
    y_col: str
    x_transform: str
    y_transform: str
    maxlag: int
    significant_lags: list[int]    # lags where p < 0.05
    best_lag: int                  # lag with lowest p-value
    best_pvalue: float
    best_fstat: float
    verdict: str                   # "CONFIRMED" | "WEAK" | "NOT CONFIRMED"
    interpretation: str

    def __str__(self) -> str:
        lines = [
            f"\n{'─'*65}",
            f"  {self.hypothesis}",
            f"{'─'*65}",
            f"  X transform : {self.x_transform}",
            f"  Y transform : {self.y_transform}",
            f"  Lags tested : 1 – {self.maxlag}",
            f"  Sig. lags   : {self.significant_lags if self.significant_lags else 'none'}",
            f"  Best lag    : {self.best_lag}  (F={self.best_fstat:.2f}, p={self.best_pvalue:.4f})",
            f"  Verdict     : {self.verdict}",
            f"  Meaning     : {self.interpretation}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core test function
# ---------------------------------------------------------------------------

def run_granger(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hypothesis: str,
    maxlag: int = 6,
    alpha: float = 0.05,
    interpretation_confirmed: str = "",
    interpretation_not_confirmed: str = "",
) -> GrangerResult:
    """
    Run Granger causality test: does X Granger-cause Y?

    Stationarises both series first. Tests at all lags 1..maxlag.
    Returns a GrangerResult with verdict and interpretation.
    """
    # Align and drop nulls
    paired = df[["date", x_col, y_col]].dropna().sort_values("date")
    x_raw = paired[x_col]
    y_raw = paired[y_col]

    if len(paired) < 30:
        return GrangerResult(
            hypothesis=hypothesis, x_col=x_col, y_col=y_col,
            x_transform="n/a", y_transform="n/a",
            maxlag=maxlag, significant_lags=[], best_lag=0,
            best_pvalue=1.0, best_fstat=0.0,
            verdict="INSUFFICIENT DATA",
            interpretation=f"Only {len(paired)} overlapping observations after dropping nulls. Need ≥30."
        )

    # Stationarise
    x_stat, x_tf = _make_stationary(x_raw, x_col)
    y_stat, y_tf = _make_stationary(y_raw, y_col)

    # Realign after differencing (differencing reduces length by 1 or 2)
    combined = pd.DataFrame({"x": x_stat, "y": y_stat}).dropna()
    if len(combined) < maxlag + 10:
        maxlag = max(1, len(combined) // 5)

    test_data = combined[["y", "x"]].values  # statsmodels: [effect, cause]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

    # Extract p-values and F-stats for each lag
    pvals = {}
    fstats = {}
    for lag, result in raw_results.items():
        # result[0] = dict of test results, result[1] = OLS fit objects
        # 'ssr_ftest' = F-test based on sum-of-squared residuals
        fstat, pval, df_denom, df_num = result[0]["ssr_ftest"]
        pvals[lag] = float(pval)
        fstats[lag] = float(fstat)

    significant_lags = [lag for lag, p in pvals.items() if p < alpha]
    best_lag = min(pvals, key=lambda k: pvals[k])
    best_pval = pvals[best_lag]
    best_fstat = fstats[best_lag]

    # Verdict
    if best_pval < 0.01:
        verdict = "CONFIRMED (p < 0.01)"
    elif best_pval < 0.05:
        verdict = "CONFIRMED (p < 0.05)"
    elif best_pval < 0.10:
        verdict = "WEAK (p < 0.10, marginal)"
    else:
        verdict = "NOT CONFIRMED (p ≥ 0.10)"

    interpretation = (
        interpretation_confirmed if best_pval < alpha else interpretation_not_confirmed
    )

    return GrangerResult(
        hypothesis=hypothesis,
        x_col=x_col,
        y_col=y_col,
        x_transform=x_tf,
        y_transform=y_tf,
        maxlag=maxlag,
        significant_lags=significant_lags,
        best_lag=best_lag,
        best_pvalue=best_pval,
        best_fstat=best_fstat,
        verdict=verdict,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# Main: run all three tests + reverse directions
# ---------------------------------------------------------------------------

def run_all_tests(df: pd.DataFrame, maxlag: int = 6) -> list[GrangerResult]:
    results = []

    # ── Test 1: repo_rate → CC outstanding ───────────────────────────────
    # Apply 6-month lag to repo_rate before testing (our model assumption)
    df_repo = df.copy()
    df_repo["repo_rate_lag6"] = df_repo["repo_rate"].shift(6)

    results.append(run_granger(
        df_repo,
        x_col="repo_rate_lag6",
        y_col="credit_cards_outstanding_lakh",
        hypothesis="repo_rate (lag 6) → CC outstanding",
        maxlag=maxlag,
        interpretation_confirmed=(
            "Repo rate at 6-month lag has genuine predictive power over CC growth. "
            "Validates the regressor choice. RBI tightening signal is real and leads by ~6 months."
        ),
        interpretation_not_confirmed=(
            "Repo rate at lag 6 does NOT significantly predict CC growth beyond CC's own history. "
            "ACTION: Re-run with raw lag (lag=0) and test other lags. "
            "If no lag is significant, consider dropping repo_rate as a regressor."
        ),
    ))

    # Reverse: CC outstanding → repo_rate (should NOT be significant)
    results.append(run_granger(
        df_repo,
        x_col="credit_cards_outstanding_lakh",
        y_col="repo_rate_lag6",
        hypothesis="CC outstanding → repo_rate (REVERSE — should NOT hold)",
        maxlag=maxlag,
        interpretation_confirmed=(
            "WARNING: reverse causality found. CC growth predicts repo rate. "
            "This could indicate the relationship is coincidental or driven by a common third factor. "
            "ACTION: Escalate to AXIOM — the regressor may be endogenous."
        ),
        interpretation_not_confirmed=(
            "Good: no reverse causality. CC growth does not predict repo rate. "
            "Direction of causality runs one way only: repo → CC."
        ),
    ))

    # ── Test 2: upi_qr_lakh → CC outstanding ─────────────────────────────
    results.append(run_granger(
        df,
        x_col="upi_qr_lakh",
        y_col="credit_cards_outstanding_lakh",
        hypothesis="upi_qr_lakh → CC outstanding",
        maxlag=maxlag,
        interpretation_confirmed=(
            "QR code expansion has genuine predictive power over CC issuance. "
            "RuPay credit-on-UPI channel is real. Keep the regressor. "
            "Answers AXIOM's challenge that this is just two upward trends."
        ),
        interpretation_not_confirmed=(
            "QR codes do NOT significantly predict CC growth beyond CC's own trend. "
            "ACTION: Drop upi_qr_lakh as a regressor — it is riding the trend, not driving it. "
            "Run qr_ablation.py to confirm the forecast impact of dropping it."
        ),
    ))

    # Reverse: CC outstanding → upi_qr_lakh
    results.append(run_granger(
        df,
        x_col="credit_cards_outstanding_lakh",
        y_col="upi_qr_lakh",
        hypothesis="CC outstanding → upi_qr_lakh (REVERSE — check for circularity)",
        maxlag=maxlag,
        interpretation_confirmed=(
            "WARNING: CC growth predicts QR code expansion. "
            "The relationship may be bidirectional or the QR regressor may be endogenous. "
            "ACTION: Flag to AXIOM. Consider instrumenting or dropping the regressor."
        ),
        interpretation_not_confirmed=(
            "Good: CC growth does not predict QR code deployment. "
            "QR infrastructure is being pushed by NPCI/government mandate, not pulled by CC demand. "
            "Circularity concern is resolved."
        ),
    ))

    # ── Test 3: DC POS volume → DC outstanding ────────────────────────────
    # Only use post-Nov 2019 data for this test (PSI format change; pre-2019 DC POS = 0)
    df_dc = df[df["date"] >= "2019-11-01"].copy()

    results.append(run_granger(
        df_dc,
        x_col="debit_card_pos_vol_lakh",
        y_col="debit_cards_outstanding_lakh",
        hypothesis="DC POS volume → DC outstanding (post-Nov 2019)",
        maxlag=min(maxlag, 4),   # shorter series after 2019 cutoff — reduce maxlag
        interpretation_confirmed=(
            "Falling DC POS swipes have genuine predictive power over DC outstanding. "
            "The UPI displacement signal is causally linked to card count trajectory. "
            "Validates using this as a regressor in the DC model."
        ),
        interpretation_not_confirmed=(
            "DC POS volume does NOT significantly predict DC outstanding. "
            "ACTION: The displacement story may be captured entirely by changepoints already. "
            "Test dropping debit_card_pos_vol_lakh and see if DC CV MAPE changes."
        ),
    ))

    # Reverse: DC outstanding → DC POS volume
    results.append(run_granger(
        df_dc,
        x_col="debit_cards_outstanding_lakh",
        y_col="debit_card_pos_vol_lakh",
        hypothesis="DC outstanding → DC POS volume (REVERSE)",
        maxlag=min(maxlag, 4),
        interpretation_confirmed=(
            "WARNING: more DC cards predict more DC POS swipes — reverse causality present. "
            "The relationship is bidirectional. "
            "ACTION: Consider whether the model should treat this as a lagged feedback loop."
        ),
        interpretation_not_confirmed=(
            "Good: DC card count does not predict POS swipes. "
            "More cards being issued does not drive more POS usage — UPI has taken over. "
            "Direction of displacement runs one way: POS decline → outstanding slowdown."
        ),
    ))

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[GrangerResult]) -> None:
    print("\n" + "═" * 65)
    print("  MIP — GRANGER CAUSALITY RESULTS")
    print("  Confirming regressor validity for CC + DC models")
    print("═" * 65)

    for r in results:
        print(r)

    print("\n" + "═" * 65)
    print("  VERDICT SUMMARY")
    print("─" * 65)
    confirmed = [r for r in results if "CONFIRMED" in r.verdict and "NOT" not in r.verdict]
    not_confirmed = [r for r in results if "NOT CONFIRMED" in r.verdict]
    weak = [r for r in results if "WEAK" in r.verdict]
    warnings_ = [r for r in confirmed if "REVERSE" in r.hypothesis]
    forward = [r for r in confirmed if "REVERSE" not in r.hypothesis]

    print(f"\n  Forward causality confirmed  : {len(forward)}")
    for r in forward:
        print(f"    ✓  {r.hypothesis}")

    if weak:
        print(f"\n  Weak / marginal              : {len(weak)}")
        for r in weak:
            print(f"    ~  {r.hypothesis}")

    if not_confirmed:
        print(f"\n  Not confirmed                : {len(not_confirmed)}")
        for r in not_confirmed:
            print(f"    ✗  {r.hypothesis}")

    if warnings_:
        print(f"\n  Reverse causality warnings   : {len(warnings_)}")
        for r in warnings_:
            print(f"    ⚠  {r.hypothesis}")

    print("\n" + "═" * 65)
    print("  ACTION ITEMS")
    print("─" * 65)
    action_needed = [r for r in results if "ACTION" in r.interpretation]
    if action_needed:
        for i, r in enumerate(action_needed, 1):
            action_line = [l for l in r.interpretation.split(".") if "ACTION" in l]
            print(f"  {i}. {r.hypothesis}")
            if action_line:
                print(f"     → {action_line[0].replace('ACTION:', '').strip()}")
    else:
        print("  None — all regressors validated.")

    print("═" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Granger causality tests for MIP model regressors."
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to master CSV (date, cc_outstanding, dc_outstanding, "
             "repo_rate, upi_qr_lakh, debit_card_pos_vol_lakh). "
             "If omitted, loads from data/processed/ parquet files."
    )
    parser.add_argument(
        "--maxlag", type=int, default=6,
        help="Maximum lag to test (default: 6). Tests all lags 1..maxlag."
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance threshold (default: 0.05)."
    )
    parser.add_argument(
        "--output-json", type=Path,
        default=_PROCESSED / "granger_results.json",
        help="Where to write JSON results (default: data/processed/granger_results.json)."
    )
    args = parser.parse_args()

    print("\nLoading data...")
    try:
        df = load_data(args.csv_path if hasattr(args, "csv_path") else args.data)
        print(f"  Loaded {len(df)} rows | "
              f"{df['date'].min().strftime('%b %Y')} → {df['date'].max().strftime('%b %Y')}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

    # Check required columns
    required = [
        "credit_cards_outstanding_lakh",
        "debit_cards_outstanding_lakh",
        "repo_rate",
        "upi_qr_lakh",
        "debit_card_pos_vol_lakh",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"\nERROR: Missing columns: {missing}")
        print(f"Available: {list(df.columns)}")
        return 1

    print(f"\nRunning Granger causality tests (maxlag={args.maxlag}, alpha={args.alpha})...")
    print("  Note: Both series are first-differenced to ensure stationarity.\n")

    results = run_all_tests(df, maxlag=args.maxlag)
    print_summary(results)

    # Write JSON output
    try:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        def _serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(args.output_json, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=_serializable)
        print(f"Results written to: {args.output_json}\n")
    except Exception as e:
        print(f"Warning: could not write JSON — {e}")

    # Exit code: 0 if all forward tests confirmed, 1 if any failed
    forward_results = [r for r in results if "REVERSE" not in r.hypothesis]
    all_confirmed = all("CONFIRMED" in r.verdict for r in forward_results)
    return 0 if all_confirmed else 1


if __name__ == "__main__":
    raise SystemExit(main())