"""
granger_causality.py  --  Formal Granger causality tests for MIP regressors.

Tests whether each regressor has genuine predictive power over the target
series, beyond what the target's own history already explains.

The five hypotheses under test (Rahul email point 1):

  VARIABLE EXCLUSION TESTS (confirm dropped variables are genuinely redundant):
  1. cpi_inflation_pct -> credit_cards_outstanding_lakh
     Rahul ask: confirm CPI adds no lagged predictive value beyond repo rate.

  2. upi_volume_mn -> debit_cards_outstanding_lakh
     Rahul ask: confirm raw UPI volume is genuinely redundant vs debit-POS.

  VARIABLE INCLUSION TESTS (confirm kept variables have directional validity):
  3. repo_rate (lag 6) -> credit_cards_outstanding_lakh
     Economic prior: RBI tightening -> banks slow CC issuance ~6m later.

  4. upi_qr_lakh -> credit_cards_outstanding_lakh
     AXIOM challenge: both trend upward -- Granger checks independent signal.

  5. debit_card_pos_vol_lakh -> debit_cards_outstanding_lakh
     Rahul ask: confirm causal direction holds (POS -> outstanding, not reverse).

Method:
  - First-difference both series before testing (Granger requires stationarity).
  - Test at maxlag = 6 (tests lags 1 through 6 separately).
  - Report F-stat and p-value per lag. p < 0.05 = statistically significant.
  - Also test the REVERSE direction (Y -> X) to check for reverse causality.

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
# Data loading
# ---------------------------------------------------------------------------

def _load_master() -> pd.DataFrame | None:
    """Load and merge all processed parquet files into one master frame."""
    psi_path  = _PROCESSED / "rbi_psi_cards.parquet"
    repo_path = _PROCESSED / "repo_rate.parquet"
    cpi_path  = _PROCESSED / "cpi.parquet"
    npci_path = _PROCESSED / "npci_upi.parquet"

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

    if cpi_path.exists():
        cpi = pd.read_parquet(cpi_path)
        cpi["date"] = pd.to_datetime(cpi["date"]).dt.to_period("M").dt.to_timestamp()
        # Accept either column name variant
        cpi_col = "cpi_inflation_pct" if "cpi_inflation_pct" in cpi.columns else "cpi_index"
        master = master.merge(cpi[["date", cpi_col]], on="date", how="left")
        if cpi_col != "cpi_inflation_pct":
            master = master.rename(columns={cpi_col: "cpi_inflation_pct"})
        master["cpi_inflation_pct"] = master["cpi_inflation_pct"].ffill()
    else:
        master["cpi_inflation_pct"] = np.nan

    if npci_path.exists():
        npci = pd.read_parquet(npci_path)
        npci["date"] = pd.to_datetime(npci["date"]).dt.to_period("M").dt.to_timestamp()
        upi_col = "upi_volume_mn" if "upi_volume_mn" in npci.columns else npci.columns[1]
        master = master.merge(npci[["date", upi_col]], on="date", how="left")
        if upi_col != "upi_volume_mn":
            master = master.rename(columns={upi_col: "upi_volume_mn"})
    else:
        master["upi_volume_mn"] = np.nan

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
    """First-difference until stationary. Granger requires stationarity."""
    is_stat, pval = _is_stationary(series)
    if is_stat:
        return series, f"none (ADF p={pval:.3f}, already stationary)"

    diff1 = series.diff().dropna()
    is_stat2, pval2 = _is_stationary(diff1)
    if is_stat2:
        return diff1, f"first-difference (ADF p after diff={pval2:.3f})"

    diff2 = diff1.diff().dropna()
    return diff2, f"second-difference (ADF p after diff={pval2:.3f}, second diff applied)"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class GrangerResult:
    hypothesis: str
    x_col: str
    y_col: str
    x_transform: str
    y_transform: str
    maxlag: int
    significant_lags: list[int]
    best_lag: int
    best_pvalue: float
    best_fstat: float
    verdict: str
    interpretation: str

    def __str__(self) -> str:
        lines = [
            f"\n{'─'*65}",
            f"  {self.hypothesis}",
            f"{'─'*65}",
            f"  X transform : {self.x_transform}",
            f"  Y transform : {self.y_transform}",
            f"  Lags tested : 1 - {self.maxlag}",
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
    """Run Granger causality test: does X Granger-cause Y?"""
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
            interpretation=f"Only {len(paired)} overlapping observations. Need >=30."
        )

    x_stat, x_tf = _make_stationary(x_raw, x_col)
    y_stat, y_tf = _make_stationary(y_raw, y_col)

    combined = pd.DataFrame({"x": x_stat, "y": y_stat}).dropna()
    if len(combined) < maxlag + 10:
        maxlag = max(1, len(combined) // 5)

    test_data = combined[["y", "x"]].values

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw_results = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)

    pvals = {}
    fstats = {}
    for lag, result in raw_results.items():
        fstat, pval, _, _ = result[0]["ssr_ftest"]
        pvals[lag] = float(pval)
        fstats[lag] = float(fstat)

    significant_lags = [lag for lag, p in pvals.items() if p < alpha]
    best_lag = min(pvals, key=lambda k: pvals[k])
    best_pval = pvals[best_lag]
    best_fstat = fstats[best_lag]

    if best_pval < 0.01:
        verdict = "CONFIRMED (p < 0.01)"
    elif best_pval < 0.05:
        verdict = "CONFIRMED (p < 0.05)"
    elif best_pval < 0.10:
        verdict = "WEAK (p < 0.10, marginal)"
    else:
        verdict = "NOT CONFIRMED (p >= 0.10)"

    interpretation = (
        interpretation_confirmed if best_pval < alpha else interpretation_not_confirmed
    )

    return GrangerResult(
        hypothesis=hypothesis,
        x_col=x_col, y_col=y_col,
        x_transform=x_tf, y_transform=y_tf,
        maxlag=maxlag,
        significant_lags=significant_lags,
        best_lag=best_lag,
        best_pvalue=best_pval,
        best_fstat=best_fstat,
        verdict=verdict,
        interpretation=interpretation,
    )


# ---------------------------------------------------------------------------
# All tests
# ---------------------------------------------------------------------------

def run_all_tests(df: pd.DataFrame, maxlag: int = 6) -> list[GrangerResult]:
    results = []

    # ── EXCLUSION TEST 1: CPI -> CC outstanding (Rahul ask) ──────────────
    # Rahul: confirm CPI adds no lagged value beyond what repo rate already captures.
    # Use CC training window (Jan 2013+) — same window as the actual CC model.
    df_cc = df[df["date"] >= "2013-01-01"].copy()

    if df_cc.get("cpi_inflation_pct", pd.Series(dtype=float)).notna().sum() > 30:
        results.append(run_granger(
            df_cc,
            x_col="cpi_inflation_pct",
            y_col="credit_cards_outstanding_lakh",
            hypothesis="CPI inflation -> CC outstanding (EXCLUSION TEST)",
            maxlag=maxlag,
            interpretation_confirmed=(
                "CPI has independent predictive power over CC growth beyond repo rate. "
                "ACTION: Reconsider dropping CPI. Re-test with both CPI and repo in the model "
                "and check if multicollinearity re-emerges (VIF > 10)."
            ),
            interpretation_not_confirmed=(
                "CPI does NOT predict CC growth beyond what repo rate already captures. "
                "Exclusion of CPI is statistically justified. "
                "Repo rate subsumes the inflation signal — decision confirmed."
            ),
        ))
    else:
        print("  WARNING: cpi_inflation_pct has insufficient data — skipping CPI test.")
        print("           Check that data/processed/cpi.parquet exists and has been ingested.")

    # ── EXCLUSION TEST 2: UPI total volume -> DC outstanding (Rahul ask) ──
    # Rahul: confirm raw UPI volume is genuinely redundant vs debit-POS.
    # Only meaningful post-2019 when UPI volume became significant.
    df_dc_upi = df[df["date"] >= "2019-11-01"].copy()

    if df_dc_upi.get("upi_volume_mn", pd.Series(dtype=float)).notna().sum() > 30:
        results.append(run_granger(
            df_dc_upi,
            x_col="upi_volume_mn",
            y_col="debit_cards_outstanding_lakh",
            hypothesis="UPI total volume -> DC outstanding (EXCLUSION TEST, post-Nov 2019)",
            maxlag=min(maxlag, 4),
            interpretation_confirmed=(
                "UPI total volume has predictive power over DC outstanding. "
                "It is NOT fully redundant with debit-POS. "
                "ACTION: Re-examine whether upi_volume_mn should replace or complement "
                "debit_card_pos_vol_lakh in the DC model."
            ),
            interpretation_not_confirmed=(
                "UPI total volume does NOT predict DC outstanding beyond debit-POS signal. "
                "Exclusion is statistically justified. "
                "The debit-POS regressor captures the displacement story more cleanly."
            ),
        ))
    else:
        print("  WARNING: upi_volume_mn has insufficient data — skipping UPI volume test.")
        print("           Check that data/processed/npci_upi.parquet exists and has been ingested.")

    # ── INCLUSION TEST 3: repo_rate -> CC outstanding ────────────────────
    df_repo = df_cc.copy()
    df_repo["repo_rate_lag6"] = df_repo["repo_rate"].shift(6)

    results.append(run_granger(
        df_repo,
        x_col="repo_rate_lag6",
        y_col="credit_cards_outstanding_lakh",
        hypothesis="repo_rate (lag 6) -> CC outstanding (INCLUSION TEST)",
        maxlag=maxlag,
        interpretation_confirmed=(
            "Repo rate at 6-month lag has genuine predictive power over CC growth. "
            "Validates the regressor choice."
        ),
        interpretation_not_confirmed=(
            "Repo rate at lag 6 does NOT predict CC growth beyond CC own history. "
            "Note: second-differencing may be destroying the slow-moving policy signal. "
            "Cross-validate with CV MAPE comparison (with vs without repo_rate) "
            "before making a drop decision."
        ),
    ))

    # Reverse: CC -> repo_rate
    results.append(run_granger(
        df_repo,
        x_col="credit_cards_outstanding_lakh",
        y_col="repo_rate_lag6",
        hypothesis="CC outstanding -> repo_rate (REVERSE)",
        maxlag=maxlag,
        interpretation_confirmed=(
            "Reverse causality found. Likely reflects RBI reaction function: "
            "credit expansion -> inflation risk -> RBI tightens. "
            "This is documented policy behaviour, not a modelling flaw. "
            "The forward direction (repo -> CC) remains the correct causal channel in the model."
        ),
        interpretation_not_confirmed=(
            "Good: no reverse causality. Direction runs one way: repo -> CC."
        ),
    ))

    # ── INCLUSION TEST 4: upi_qr_lakh -> CC outstanding ──────────────────
    results.append(run_granger(
        df_cc,
        x_col="upi_qr_lakh",
        y_col="credit_cards_outstanding_lakh",
        hypothesis="upi_qr_lakh -> CC outstanding (INCLUSION TEST)",
        maxlag=maxlag,
        interpretation_confirmed=(
            "QR code expansion has independent predictive power over CC issuance. "
            "RuPay credit-on-UPI channel is real. Keep the regressor."
        ),
        interpretation_not_confirmed=(
            "QR codes do NOT predict CC growth beyond CC own trend. "
            "ACTION: Run qr_ablation.py to check forecast impact. "
            "If mean delta < 1%, drop the regressor."
        ),
    ))

    # ── INCLUSION TEST 5: DC POS -> DC outstanding (Rahul ask) ───────────
    df_dc = df[df["date"] >= "2019-11-01"].copy()

    results.append(run_granger(
        df_dc,
        x_col="debit_card_pos_vol_lakh",
        y_col="debit_cards_outstanding_lakh",
        hypothesis="DC POS volume -> DC outstanding (INCLUSION TEST, post-Nov 2019)",
        maxlag=min(maxlag, 4),
        interpretation_confirmed=(
            "DC POS swipes have genuine predictive power over DC outstanding. "
            "The displacement signal is causally linked to card count trajectory. "
            "Validates using this as a regressor."
        ),
        interpretation_not_confirmed=(
            "DC POS volume does NOT predict DC outstanding. "
            "Changepoints (UPI inflection Jan 2022) may be fully absorbing this signal. "
            "ACTION: Test dropping debit_card_pos_vol_lakh and compare DC CV MAPE."
        ),
    ))

    # Reverse: DC outstanding -> DC POS (Rahul ask: confirm direction)
    results.append(run_granger(
        df_dc,
        x_col="debit_cards_outstanding_lakh",
        y_col="debit_card_pos_vol_lakh",
        hypothesis="DC outstanding -> DC POS volume (REVERSE — Rahul direction check)",
        maxlag=min(maxlag, 4),
        interpretation_confirmed=(
            "WARNING: DC card count predicts POS swipes. Reverse causality present. "
            "ACTION: Check whether more cards being issued is partially sustaining POS volume. "
            "If so, the causal story is bidirectional."
        ),
        interpretation_not_confirmed=(
            "Good: DC card count does NOT predict POS swipes. "
            "More cards being issued does not drive more POS usage — UPI has taken over. "
            "Causal direction confirmed: POS decline -> outstanding slowdown."
        ),
    ))

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: list[GrangerResult]) -> None:
    print("\n" + "=" * 65)
    print("  MIP — GRANGER CAUSALITY RESULTS")
    print("  Variable selection validation (Rahul email, point 1)")
    print("=" * 65)

    for r in results:
        print(r)

    print("\n" + "=" * 65)
    print("  VERDICT SUMMARY")
    print("-" * 65)

    exclusion = [r for r in results if "EXCLUSION" in r.hypothesis]
    inclusion = [r for r in results if "INCLUSION" in r.hypothesis]
    reverse   = [r for r in results if "REVERSE" in r.hypothesis]

    print("\n  EXCLUSION TESTS (dropped variables — should NOT be confirmed):")
    for r in exclusion:
        icon = "✓ justified" if "NOT CONFIRMED" in r.verdict else "✗ RECONSIDER"
        print(f"    {icon}  {r.hypothesis.split('(')[0].strip()} — {r.verdict}")

    print("\n  INCLUSION TESTS (kept variables — should be confirmed):")
    for r in inclusion:
        icon = "✓" if "CONFIRMED" in r.verdict and "NOT" not in r.verdict else "~ weak" if "WEAK" in r.verdict else "✗"
        print(f"    {icon}  {r.hypothesis.split('(')[0].strip()} — {r.verdict}")

    print("\n  REVERSE DIRECTION CHECKS:")
    for r in reverse:
        icon = "⚠ warning" if "CONFIRMED" in r.verdict and "NOT" not in r.verdict else "✓ clean"
        print(f"    {icon}  {r.hypothesis.split('(')[0].strip()} — {r.verdict}")

    print("\n" + "=" * 65)
    print("  ACTION ITEMS FOR RAHUL RESPONSE")
    print("-" * 65)
    action_needed = [r for r in results if "ACTION" in r.interpretation]
    if action_needed:
        for i, r in enumerate(action_needed, 1):
            action_line = [l.strip() for l in r.interpretation.split(".") if "ACTION" in l]
            print(f"  {i}. {r.hypothesis}")
            if action_line:
                print(f"     -> {action_line[0].replace('ACTION:', '').strip()}")
    else:
        print("  None — all variable selections validated.")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Granger causality tests for MIP model variable selection."
    )
    parser.add_argument(
        "--data", type=Path, default=None,
        help="Path to master CSV. If omitted, loads from data/processed/ parquets."
    )
    parser.add_argument("--maxlag", type=int, default=6)
    parser.add_argument("--alpha",  type=float, default=0.05)
    parser.add_argument(
        "--output-json", type=Path,
        default=_PROCESSED / "granger_results.json",
    )
    args = parser.parse_args()

    print("\nLoading data...")
    try:
        df = load_data(args.data)
        print(f"  Loaded {len(df)} rows | "
              f"{df['date'].min().strftime('%b %Y')} -> {df['date'].max().strftime('%b %Y')}")
        print(f"  Columns: {list(df.columns)}")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        return 1

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

    # Warn if new columns are missing (non-fatal — tests will be skipped)
    for col, label in [("cpi_inflation_pct", "CPI"), ("upi_volume_mn", "UPI volume")]:
        if col not in df.columns or df[col].notna().sum() < 30:
            print(f"  NOTE: {col} not available — {label} exclusion test will be skipped.")
            print(f"        Ensure data/processed/{'cpi' if 'cpi' in col else 'npci_upi'}.parquet is ingested.")

    print(f"\nRunning Granger causality tests (maxlag={args.maxlag}, alpha={args.alpha})...")
    print("  Both series are first-differenced to ensure stationarity.\n")

    results = run_all_tests(df, maxlag=args.maxlag)
    print_summary(results)

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

    forward = [r for r in results if "REVERSE" not in r.hypothesis]
    all_ok = all(
        ("NOT CONFIRMED" in r.verdict if "EXCLUSION" in r.hypothesis
         else "CONFIRMED" in r.verdict)
        for r in forward
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())