"""
MIP — Regressor Candidates (P3.3)
===================================
Granger-gated regressor evaluation pipeline.

Candidates (add one at a time, each must pass Granger gate):
  1. CPI inflation (already scraped, unused) → CC, DC
  2. Festive-calendar position → CC, DC
  3. UPI P2M share (DC substitution proxy) → DC
  4. GST collections → CC, DC (data not yet available)

Note: repo_rate→CC already failed Granger in P1 analysis and is kept on
business grounds. It will re-prove itself inside ARIMAX honestly.

Usage:
    python -m src.modelling.regressor_candidates          # evaluate all
    python -m src.modelling.regressor_candidates --add    # evaluate + add passing regressors
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from src.modelling.granger_causality import run_granger, load_data

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_PROCESSED = _PROJECT_ROOT / "data" / "processed"


# ── Festive calendar ──────────────────────────────────────────────────────

def build_festive_index(dates: pd.DatetimeIndex) -> pd.Series:
    """Build a festive-calendar position index for Indian card spending.

    India's festive/spending calendar has predictable peaks:
      - Oct (Dussehra + Diwali shopping) — highest
      - Nov (Diwali, post-Diwali sales)
      - Mar (financial year-end purchases)
      - Dec (Christmas, New Year)
      - Aug (Independence Day sales, Onam in Kerala)
      - Jan (New Year sales, Pongal/Sankranti)

    The index is a multiplicative factor: 1.0 = neutral month,
    >1.0 = festive boost, <1.0 = lean month.

    These weights are derived from observed CC/DC txn volume seasonality
    in the RBI PSI data (2017-2025 average).
    """
    festive_weights = {
        1: 1.02,   # Jan: New Year sales, Pongal/Sankranti
        2: 0.95,   # Feb: lean
        3: 1.08,   # Mar: year-end rush
        4: 0.93,   # Apr: new FY start, lean
        5: 0.95,   # May: lean
        6: 0.96,   # Jun: lean
        7: 0.98,   # Jul: pre-festive
        8: 1.03,   # Aug: Independence Day, Onam
        9: 1.00,   # Sep: neutral
        10: 1.15,  # Oct: Dussehra + pre-Diwali peak
        11: 1.05,  # Nov: Diwali + post-Diwali
        12: 1.03,  # Dec: Christmas, year-end
    }
    return pd.Series(
        [festive_weights[d.month] for d in dates],
        index=dates,
        name="festive_index",
    )


# ── UPI P2M share ────────────────────────────────────────────────────────

def compute_upi_p2m_share(master: pd.DataFrame) -> pd.Series:
    """Compute UPI P2M volume as fraction of total UPI volume.

    Higher P2M share → more merchant payments via UPI → more DC displacement.
    This is a better DC displacement proxy than raw UPI volume because it
    isolates the merchant payment channel (P2M) from person-to-person transfers.
    """
    if "upi_p2m_vol_mn" not in master.columns or "upi_volume_mn" not in master.columns:
        return pd.Series(dtype=float, name="upi_p2m_share")

    share = master["upi_p2m_vol_mn"] / master["upi_volume_mn"]
    share = share.where(share.between(0, 1))  # clip nonsensical values
    share.name = "upi_p2m_share"
    return share


# ── Candidate definition ────────────────────────────────────────────────

@dataclass
class RegressorCandidate:
    name: str
    col: str
    targets: list[str]
    lag: int
    mode: str
    hypothesis: str
    interpretation_confirmed: str
    interpretation_not_confirmed: str


CANDIDATES = [
    RegressorCandidate(
        name="CPI inflation",
        col="cpi_inflation_pct",
        targets=["credit_cards_outstanding_lakh", "debit_cards_outstanding_lakh"],
        lag=3,
        mode="additive",
        hypothesis="CPI inflation → card outstanding (cost-of-living channel)",
        interpretation_confirmed=(
            "Higher inflation increases nominal spending → more card usage → "
            "more cards outstanding. 3-month lag for behavioral adjustment."
        ),
        interpretation_not_confirmed=(
            "CPI inflation does not independently predict card outstanding beyond "
            "what the series' own trend already captures."
        ),
    ),
    RegressorCandidate(
        name="Festive calendar",
        col="festive_index",
        targets=["credit_cards_outstanding_lakh", "debit_cards_outstanding_lakh"],
        lag=0,
        mode="multiplicative",
        hypothesis="Festive calendar → card outstanding (seasonal spending channel)",
        interpretation_confirmed=(
            "Festive months drive higher card issuance/activation. "
            "Multiplicative effect — scales with base level."
        ),
        interpretation_not_confirmed=(
            "Festive calendar does not add predictive power beyond Prophet's "
            "built-in yearly seasonality."
        ),
    ),
    RegressorCandidate(
        name="UPI P2M share",
        col="upi_p2m_share",
        targets=["debit_cards_outstanding_lakh"],
        lag=2,
        mode="additive",
        hypothesis="UPI P2M share → DC outstanding (substitution channel)",
        interpretation_confirmed=(
            "Higher UPI P2M share → more merchant payments diverted from DC → "
            "reduced DC demand. 2-month lag for behavioral shift."
        ),
        interpretation_not_confirmed=(
            "UPI P2M share does not independently predict DC outstanding beyond "
            "the existing UPI inflection changepoint and DC POS volume regressor."
        ),
    ),
]


# ── Evaluation pipeline ─────────────────────────────────────────────────

def _prepare_master_with_candidates() -> pd.DataFrame:
    """Load master and add candidate regressor columns."""
    master = load_data()

    # Add festive index
    master["festive_index"] = build_festive_index(
        pd.DatetimeIndex(master["date"])
    ).values

    # Add UPI P2M share (need to load p2m data)
    try:
        p2m = pd.read_parquet(_PROCESSED / "upi_p2p_p2m.parquet")
        p2m["date"] = pd.to_datetime(p2m["date"]).dt.to_period("M").dt.to_timestamp()
        master = master.merge(
            p2m[["date", "upi_p2m_vol_mn"]], on="date", how="left"
        )
        if "upi_volume_mn" in master.columns and "upi_p2m_vol_mn" in master.columns:
            master["upi_p2m_share"] = (
                master["upi_p2m_vol_mn"] / master["upi_volume_mn"]
            ).where(lambda x: x.between(0, 1))
    except FileNotFoundError:
        master["upi_p2m_share"] = np.nan

    return master


@dataclass
class CandidateResult:
    candidate: RegressorCandidate
    target: str
    granger_pvalue: float
    granger_fstat: float
    granger_best_lag: int
    passed: bool
    verdict: str


def evaluate_candidates(
    alpha: float = 0.05,
) -> list[CandidateResult]:
    """Evaluate all regressor candidates via Granger causality gate.

    Returns list of CandidateResult with pass/fail for each candidate-target pair.
    """
    master = _prepare_master_with_candidates()
    results = []

    for candidate in CANDIDATES:
        if candidate.col not in master.columns:
            logger.warning(f"Column '{candidate.col}' not in master — skipping {candidate.name}")
            continue

        col_data = master[candidate.col].dropna()
        if len(col_data) < 30:
            logger.warning(f"'{candidate.col}' has only {len(col_data)} non-null values — skipping")
            continue

        for target in candidate.targets:
            if target not in master.columns:
                continue

            test_df = master[["date", target]].copy()
            test_df[candidate.col] = master[candidate.col]

            granger = run_granger(
                test_df,
                x_col=candidate.col,
                y_col=target,
                hypothesis=candidate.hypothesis,
                maxlag=max(candidate.lag, 6),
                alpha=alpha,
                interpretation_confirmed=candidate.interpretation_confirmed,
                interpretation_not_confirmed=candidate.interpretation_not_confirmed,
            )

            passed = granger.best_pvalue < alpha
            cr = CandidateResult(
                candidate=candidate,
                target=target,
                granger_pvalue=granger.best_pvalue,
                granger_fstat=granger.best_fstat,
                granger_best_lag=granger.best_lag,
                passed=passed,
                verdict=granger.verdict,
            )
            results.append(cr)

            status = "PASS" if passed else "FAIL"
            logger.info(
                f"  [{status}] {candidate.name} → {target}: "
                f"p={granger.best_pvalue:.4f}, F={granger.best_fstat:.2f}, "
                f"best_lag={granger.best_lag}"
            )

    return results


def print_results(results: list[CandidateResult]) -> None:
    """Print evaluation results as a formatted table."""
    print(f"\n{'='*75}")
    print("REGRESSOR CANDIDATE EVALUATION (Granger-gated)")
    print(f"{'='*75}")
    print(f"{'Candidate':<20} {'Target':<35} {'p-value':>8} {'F-stat':>8} {'Lag':>4} {'Gate':>6}")
    print(f"{'-'*20} {'-'*35} {'-'*8} {'-'*8} {'-'*4} {'-'*6}")

    for r in results:
        gate = "PASS" if r.passed else "FAIL"
        print(
            f"{r.candidate.name:<20} {r.target:<35} "
            f"{r.granger_pvalue:>8.4f} {r.granger_fstat:>8.2f} "
            f"{r.granger_best_lag:>4} {gate:>6}"
        )

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    print(f"\nPassed: {len(passed)} | Failed: {len(failed)}")

    if passed:
        print("\nRegressors cleared for inclusion (add one at a time, re-run CV):")
        for r in passed:
            print(f"  + {r.candidate.name} → {r.target} (lag={r.candidate.lag}, mode={r.candidate.mode})")

    print()


if __name__ == "__main__":
    import sys
    results = evaluate_candidates()
    print_results(results)
