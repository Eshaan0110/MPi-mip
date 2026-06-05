"""
Per-Bank EDA for bankwise regressors
======================================
For each bank, shows EDA of the candidate regressors
within the STABLE REGIME window (respects BANK_START_DATES exactly).

Outputs:
  - Summary stats (mean, trend, % change over window)
  - Spearman correlation vs outstanding
  - Granger causality p-values (lags 1, 3, 6)
  - Data quality (null months, gaps)
  - Verdict: USE / CONSIDER / DROP for each variable per bank
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# ── Config (mirrors bank_config.py — do NOT change) ───────────────────────
CC_LIVE_BANK_TRAIN_START = pd.Timestamp("2013-01-01")
DC_LIVE_BANK_TRAIN_START = pd.Timestamp("2017-01-01")

BANK_START_DATES = {
    ("HDFC Bank",           "cc"): pd.Timestamp("2017-01-01"),
    ("State Bank of India", "cc"): pd.Timestamp("2017-04-01"),
    ("ICICI Bank",          "cc"): pd.Timestamp("2017-01-01"),
    ("Kotak Mahindra Bank", "cc"): pd.Timestamp("2018-01-01"),
    ("Bank of Baroda",      "cc"): pd.Timestamp("2019-04-01"),
    ("Yes Bank",            "cc"): pd.Timestamp("2020-06-01"),
    ("Canara Bank",         "cc"): pd.Timestamp("2020-04-01"),
    ("State Bank of India", "dc"): pd.Timestamp("2017-04-01"),
    ("Bank of Baroda",      "dc"): pd.Timestamp("2019-04-01"),
    ("Canara Bank",         "dc"): pd.Timestamp("2020-04-01"),
    ("Union Bank of India", "dc"): pd.Timestamp("2020-04-01"),
    ("Punjab National Bank","dc"): pd.Timestamp("2020-04-01"),
    ("Indian Bank",         "dc"): pd.Timestamp("2020-04-01"),
    ("Paytm Payments Bank", "dc"): pd.Timestamp("2018-04-01"),
}

def get_start(bank, card_type):
    key = (bank, card_type)
    if key in BANK_START_DATES:
        return BANK_START_DATES[key]
    return CC_LIVE_BANK_TRAIN_START if card_type == "cc" else DC_LIVE_BANK_TRAIN_START

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

# Variables with long history (>= 2011) — useful for modelling
CC_CANDIDATES = {
    "atm_offsite":     "ATMs Off-site",
    "pos_terminals":   "PoS Terminals",
    "cc_pos_vol":      "CC PoS Txn Vol",
    "cc_atm_cash_vol": "CC ATM Cash Vol",
}
DC_CANDIDATES = {
    "atm_onsite":      "ATMs On-site",
    "pos_terminals":   "PoS Terminals",
    "dc_pos_vol":      "DC PoS Txn Vol",
    "dc_atm_cash_vol": "DC ATM Cash Vol",
}

# ── Load ──────────────────────────────────────────────────────────────────
cc_bw = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc_bw = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
cc_bw["date"] = pd.to_datetime(cc_bw["date"]).dt.to_period("M").dt.to_timestamp()
dc_bw["date"] = pd.to_datetime(dc_bw["date"]).dt.to_period("M").dt.to_timestamp()


def granger_p(y, x, lag):
    df = pd.DataFrame({"y": y, "x": x}).dropna()
    if len(df) < lag * 3 + 10:
        return None
    dy = df["y"].diff().dropna()
    dx = df["x"].diff().dropna()
    data = pd.DataFrame({"y": dy.values[:len(dx)], "x": dx.values[:len(dy)]}).dropna()
    if len(data) < lag * 3 + 5:
        return None
    try:
        res = grangercausalitytests(data[["y", "x"]], maxlag=lag, verbose=False)
        return res[lag][0]["ssr_ftest"][1]
    except Exception:
        return None


def fmt_p(p):
    if p is None: return "  N/A "
    s = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "  "
    return f"{p:.3f}{s}"


def trend_label(pct):
    if pct > 50:   return f"+{pct:.0f}% (strong growth)"
    if pct > 10:   return f"+{pct:.0f}% (growth)"
    if pct > -10:  return f"{pct:+.0f}% (flat)"
    if pct > -50:  return f"{pct:.0f}% (declining)"
    return f"{pct:.0f}% (strong decline)"


def run_bank_eda(bw, banks, target_col, candidates, card_type, label):
    print()
    print("=" * 100)
    print(f"{label}")
    print("=" * 100)

    for bank in banks:
        start = get_start(bank, card_type)
        bdf = bw[(bw["bank_name"] == bank) & (bw["date"] >= start)].sort_values("date")

        if bdf.empty or bdf[target_col].isna().all():
            print(f"\n  {bank}: NO DATA in stable window")
            continue

        n_months = len(bdf)
        window_start = bdf["date"].min()
        window_end   = bdf["date"].max()
        outstanding_latest = bdf[target_col].dropna().iloc[-1]

        print(f"\n{'─'*100}")
        print(f"  BANK: {bank}  |  Stable window: {window_start:%b %Y} -> {window_end:%b %Y}  ({n_months} months)")
        print(f"  Latest {target_col}: {outstanding_latest:,.0f} cards")
        print(f"{'─'*100}")
        print(f"  {'Variable':<22} {'Coverage':>9} {'Latest val':>12} {'Trend':>25}  {'Spearman':>9}  {'Granger L1':>10} {'L3':>8} {'L6':>8}  Verdict")
        print(f"  {'-'*96}")

        for col, lbl in candidates.items():
            if col not in bdf.columns:
                print(f"  {lbl:<22} {'NOT IN DATA':>9}")
                continue

            series = bdf[col].copy()
            n_null = series.isna().sum()
            n_valid = series.notna().sum()
            coverage = f"{n_valid}/{n_months}"

            if n_valid < 6:
                print(f"  {lbl:<22} {coverage:>9}  (insufficient data)")
                continue

            latest_val  = series.dropna().iloc[-1]
            first_val   = series.dropna().iloc[0]
            pct_change  = ((latest_val - first_val) / first_val * 100) if first_val != 0 else 0
            trend_str   = trend_label(pct_change)

            # Spearman vs outstanding
            pair = bdf[[target_col, col]].dropna()
            if len(pair) >= 8:
                rho, p_rho = stats.spearmanr(pair[target_col], pair[col])
                rho_str = f"{rho:+.3f}{'*' if p_rho < 0.05 else ' '}"
            else:
                rho_str = "  N/A "

            # Granger at lags 1, 3, 6
            y_s = bdf[target_col]
            x_s = bdf[col]
            p1 = granger_p(y_s, x_s, 1)
            p3 = granger_p(y_s, x_s, 3)
            p6 = granger_p(y_s, x_s, 6)

            # Verdict
            best_p = min(p for p in [p1, p3, p6] if p is not None) if any(p is not None for p in [p1, p3, p6]) else 1.0
            if best_p < 0.01 and n_valid >= 36:
                verdict = "USE"
            elif best_p < 0.05 and n_valid >= 24:
                verdict = "CONSIDER"
            elif best_p < 0.10 and n_valid >= 24:
                verdict = "WEAK"
            else:
                verdict = "drop"

            print(
                f"  {lbl:<22} {coverage:>9}  {latest_val:>12,.0f}  {trend_str:<25}  "
                f"{rho_str:>9}  {fmt_p(p1):>10} {fmt_p(p3):>8} {fmt_p(p6):>8}  {verdict}"
            )

    print()


if __name__ == "__main__":
    run_bank_eda(cc_bw, CC_BANKS, "cc_outstanding", CC_CANDIDATES, "cc",
                 "CREDIT CARDS — Per-bank EDA (stable regime windows only)")
    run_bank_eda(dc_bw, DC_BANKS, "dc_outstanding", DC_CANDIDATES, "dc",
                 "DEBIT CARDS — Per-bank EDA (stable regime windows only)")
    print("Done.")
