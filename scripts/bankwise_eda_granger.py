"""
Bankwise EDA + Granger Causality
=================================
For each candidate per-bank variable (infrastructure + transaction volumes),
runs:
  1. Data availability summary (how many banks, date range)
  2. Spearman correlation vs outstanding (per bank, then median across banks)
  3. Granger causality: X -> cc_outstanding / dc_outstanding per bank

All metrics on original scale. First-differenced before Granger.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# ── Load data ─────────────────────────────────────────────────────────────
cc_bw = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc_bw = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
cc_bw["date"] = pd.to_datetime(cc_bw["date"]).dt.to_period("M").dt.to_timestamp()
dc_bw["date"] = pd.to_datetime(dc_bw["date"]).dt.to_period("M").dt.to_timestamp()

CC_BANKS = [
    "HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
    "Kotak Mahindra Bank", "IndusInd Bank", "Bank of Baroda",
    "Yes Bank", "Canara Bank", "HSBC",
]
DC_BANKS = [
    "State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
    "Union Bank of India", "Punjab National Bank", "Axis Bank",
    "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
    "Central Bank of India", "UCO Bank", "ICICI Bank",
    "Indian Overseas Bank", "Paytm Payments Bank",
]

CC_VARS = {
    "atm_onsite":      "ATMs On-site",
    "atm_offsite":     "ATMs Off-site",
    "pos_terminals":   "PoS Terminals",
    "micro_atm":       "Micro ATMs",
    "bharat_qr":       "Bharat QR",
    "upi_qr":          "UPI QR Codes",
    "cc_pos_vol":      "CC PoS Txn Vol",
    "cc_online_vol":   "CC Online Txn Vol",
    "cc_others_vol":   "CC Others Txn Vol",
    "cc_atm_cash_vol": "CC ATM Cash Vol",
}
DC_VARS = {
    "atm_onsite":      "ATMs On-site",
    "atm_offsite":     "ATMs Off-site",
    "pos_terminals":   "PoS Terminals",
    "micro_atm":       "Micro ATMs",
    "bharat_qr":       "Bharat QR",
    "upi_qr":          "UPI QR Codes",
    "dc_pos_vol":      "DC PoS Txn Vol",
    "dc_online_vol":   "DC Online Txn Vol",
    "dc_others_vol":   "DC Others Txn Vol",
    "dc_atm_cash_vol": "DC ATM Cash Vol",
}


def granger_pval(y, x, max_lag=3):
    """Granger causality p-value (best across lags 1..max_lag) on differenced data."""
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < 20:
        return None
    dy = df["y"].diff().dropna()
    dx = df["x"].diff().dropna()
    data = pd.DataFrame({"y": dy.values[:len(dx)], "x": dx.values[:len(dy)]}).dropna()
    if len(data) < 15:
        return None
    best_p = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= len(data) // 3:
            continue
        try:
            res = grangercausalitytests(data[["y", "x"]], maxlag=lag, verbose=False)
            p = res[lag][0]["ssr_ftest"][1]
            best_p = min(best_p, p)
        except Exception:
            pass
    return best_p


def run_analysis(bw, banks, target_col, cand_vars, label):
    print()
    print("=" * 90)
    print(f"{label}")
    print("=" * 90)

    # ── Section 1: Data availability ──────────────────────────────────────
    print(f"\n--- DATA AVAILABILITY ---")
    print(f"  {'Variable':<22} {'Label':<22} {'Banks w/ data':>14} {'Date range':>22} {'Null%':>6}")
    print("  " + "-" * 90)
    for col, lbl in cand_vars.items():
        if col not in bw.columns:
            print(f"  {col:<22} {lbl:<22} {'NOT IN DATA':>14}")
            continue
        sub = bw[bw["bank_name"].isin(banks) & bw[col].notna()]
        n_banks = sub["bank_name"].nunique()
        if n_banks == 0:
            print(f"  {col:<22} {lbl:<22} {'0':>14}")
            continue
        dr = f"{sub['date'].min():%b %Y} - {sub['date'].max():%b %Y}"
        null_pct = bw[bw["bank_name"].isin(banks)][col].isna().mean() * 100
        print(f"  {col:<22} {lbl:<22} {n_banks:>14} {dr:>22} {null_pct:>5.1f}%")

    # ── Section 2: Spearman correlation (per-bank median) ─────────────────
    print(f"\n--- SPEARMAN CORRELATION vs {target_col.upper()} (median across banks) ---")
    print(f"  {'Variable':<22} {'Median rho':>11} {'Min rho':>8} {'Max rho':>8} {'n_banks':>8} {'Sig banks':>10}")
    print("  " + "-" * 75)
    for col, lbl in cand_vars.items():
        if col not in bw.columns:
            continue
        rhos = []
        n_sig = 0
        for bank in banks:
            bdf = bw[bw["bank_name"] == bank][[target_col, col]].dropna()
            if len(bdf) < 10:
                continue
            rho, p = stats.spearmanr(bdf[target_col], bdf[col])
            rhos.append(rho)
            if p < 0.05:
                n_sig += 1
        if not rhos:
            continue
        med = float(np.median(rhos))
        print(f"  {col:<22} {med:>+11.3f} {min(rhos):>+8.3f} {max(rhos):>+8.3f} {len(rhos):>8} {n_sig:>10}")

    # ── Section 3: Granger causality (per-bank, report best p-value) ──────
    print(f"\n--- GRANGER CAUSALITY: X -> {target_col.upper()} (per bank, lags 1-3) ---")
    print(f"  {'Variable':<22} ", end="")
    for bank in banks[:8]:
        print(f"{bank[:10]:>11}", end="")
    print(f"  {'Verdict':>12}")
    print("  " + "-" * 110)

    for col, lbl in cand_vars.items():
        if col not in bw.columns:
            continue
        pvals = []
        row = f"  {col:<22} "
        for bank in banks[:8]:
            bdf = bw[bw["bank_name"] == bank][[target_col, col]].dropna()
            p = granger_pval(bdf[target_col], bdf[col])
            pvals.append(p)
            if p is None:
                row += f"{'  N/A':>11}"
            else:
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                row += f"{p:>8.3f}{sig:>3}"
        n_sig = sum(1 for p in pvals if p is not None and p < 0.05)
        n_valid = sum(1 for p in pvals if p is not None)
        verdict = f"PREDICTIVE ({n_sig}/{n_valid})" if n_sig > 0 else f"weak ({n_sig}/{n_valid})"
        print(row + f"  {verdict:>12}")

    print()


if __name__ == "__main__":
    run_analysis(cc_bw, CC_BANKS, "cc_outstanding", CC_VARS,
                 "CREDIT CARDS — Per-bank variable EDA + Granger")
    run_analysis(dc_bw, DC_BANKS, "dc_outstanding", DC_VARS,
                 "DEBIT CARDS — Per-bank variable EDA + Granger")
    print("Done.")
