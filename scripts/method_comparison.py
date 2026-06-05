"""
Three-Method Comparison: Fixed Share vs Trend Share vs Ground-Up Models
========================================================================
Tests all three bank forecast methodologies against actual March + April 2026 data.

Method 1 — Fixed Share:    bank_forecast = bank_share_at_May2025 * actual_total
Method 2 — Trend Share:    bank_forecast = projected_share (linear trend) * actual_total
Method 3 — Ground-Up:      bank_forecast = individual Prophet model forecast

For fairness, Methods 1 and 2 use the ACTUAL March/April 2026 total
(so errors reflect only the allocation methodology, not total forecast error).
Then we also show the end-to-end comparison using our PSI model's forecast total.

Output: per-bank table + summary MAPE for each method.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from src.ingestion.bankwise import ingest, canonical_bank

# ── Load actuals ───────────────────────────────────────────────────────────
ACTUALS_FILES = [
    ("March 2026", "2026-03", Path(r"C:\Users\ASUS\Downloads\ATMMARCH20265F48EF7056E84759B802285873DD3FB3.XLSX")),
    ("April 2026", "2026-04", Path(r"C:\Users\ASUS\Downloads\ATMAPRIL20265AF3EE208AF746979B772EED805CDCA8.XLSX")),
]

# ── Load training history (up to May 2025) ────────────────────────────────
cc_bw = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc_bw = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
for df in [cc_bw, dc_bw]:
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

FC_DIR = Path("data/processed/bankwise_forecasts")

CC_BANKS = ["HDFC Bank","State Bank of India","ICICI Bank","Axis Bank",
            "Kotak Mahindra Bank","IndusInd Bank","Bank of Baroda",
            "Yes Bank","Canara Bank","HSBC"]
DC_BANKS = ["State Bank of India","Bank of Baroda","Canara Bank","HDFC Bank",
            "Union Bank of India","Punjab National Bank","Axis Bank",
            "Bank of India","Kotak Mahindra Bank","Indian Bank",
            "Central Bank of India","UCO Bank","ICICI Bank",
            "Indian Overseas Bank","Paytm Payments Bank"]


def compute_fixed_share(bw, col, banks, ref_date="2025-05-01"):
    """Method 1: each bank's share at ref_date (last known month)."""
    ref = pd.Timestamp(ref_date)
    latest = bw[bw.date == ref][[col, "bank_name"]].dropna()
    total = latest[col].sum()
    shares = {}
    for b in banks:
        v = latest[latest.bank_name == b][col]
        shares[b] = float(v.iloc[0]) / total if len(v) > 0 and total > 0 else 0.0
    return shares


def compute_trend_share(bw, col, banks, n_months=16, ref_date="2025-05-01"):
    """Method 2: linear trend in each bank's share over last n_months → project to target months."""
    ref = pd.Timestamp(ref_date)
    window = bw[bw.date <= ref].sort_values("date")
    dates = sorted(window.date.unique())[-n_months:]

    # monthly total across all banks
    totals = (window[window.date.isin(dates)]
              .groupby("date")[col].sum().rename("total"))

    share_series = {}
    for b in banks:
        bs = (window[window.bank_name == b & window.date.isin(dates)]
              if False else
              window[(window.bank_name == b) & (window.date.isin(dates))]
              .set_index("date")[col])
        if len(bs) < 4:
            share_series[b] = None
            continue
        df = pd.DataFrame({"share": bs / totals}).dropna()
        if len(df) < 4:
            share_series[b] = None
            continue
        t = np.arange(len(df))
        m, c = np.polyfit(t, df.share.values, 1)
        share_series[b] = (m, c, len(df))  # slope, intercept, n

    def project(bank, months_ahead):
        """Project share n months after end of training window."""
        if share_series.get(bank) is None:
            return compute_fixed_share(bw, col, [bank], ref_date).get(bank, 0)
        m, c, n = share_series[bank]
        proj = c + m * (n - 1 + months_ahead)
        return max(0.0, proj)  # clamp negatives

    return project


def load_ground_up_forecast(card_type, banks, month_str):
    """Method 3: load from individual Prophet model forecast CSVs."""
    forecasts = {}
    for bank in banks:
        safe = bank.lower().replace(" ","_").replace(".","").replace("/","_")
        fc_path = FC_DIR / f"{card_type}_{safe}_forecast.csv"
        if not fc_path.exists():
            continue
        fc = pd.read_csv(fc_path)
        row = fc[fc.date.str.startswith(month_str)]
        if row.empty: continue
        forecasts[bank] = float(row.forecast.iloc[0])
    return forecasts


def run_comparison(card_type, banks, bw, col, label):
    print()
    print("=" * 100)
    print(f"{label}")
    print("=" * 100)

    # Pre-compute shares once
    fixed_shares = compute_fixed_share(bw, col, banks)
    trend_project = compute_trend_share(bw, col, banks)

    all_results = []

    for month_label, month_str, filepath in ACTUALS_FILES:
        actual_df = ingest(filepath, verbose=False)
        actual_df["bank"] = actual_df["bank"].apply(canonical_bank)

        actual_col = "credit_outstanding" if card_type == "cc" else "debit_outstanding"
        actual_total = actual_df[actual_col].sum()

        # Months ahead from May 2025
        months_ahead = 10 if "March" in month_label else 11

        # Ground-up forecasts
        gu_forecasts = load_ground_up_forecast(card_type, banks, month_str)

        print(f"\n{month_label}  |  Actual total: {actual_total/1e6:.1f}M cards")
        print(f"\n  {'Bank':<28} {'Actual':>9} {'M1 Fixed':>9} {'M1 APE':>7} {'M2 Trend':>9} {'M2 APE':>7} {'M3 GndUp':>9} {'M3 APE':>7}")
        print("  " + "-" * 92)

        m1_apes, m2_apes, m3_apes = [], [], []

        for bank in banks:
            actual_row = actual_df[actual_df.bank == bank][actual_col]
            if actual_row.empty or actual_row.isna().all(): continue
            actual_val = float(actual_row.iloc[0])
            if actual_val <= 0: continue

            # Method 1: fixed share * actual total
            m1 = fixed_shares.get(bank, 0) * actual_total
            m1_ape = abs(m1 - actual_val) / actual_val * 100

            # Method 2: trend share * actual total
            proj_share = trend_project(bank, months_ahead)
            m2 = proj_share * actual_total
            m2_ape = abs(m2 - actual_val) / actual_val * 100

            # Method 3: ground-up
            m3_val = gu_forecasts.get(bank)
            if m3_val is not None:
                m3_ape = abs(m3_val - actual_val) / actual_val * 100
                m3_str = f"{m3_val/1e6:>7.2f}M"
                m3_ape_str = f"{m3_ape:>6.1f}%"
                m3_apes.append(m3_ape)
            else:
                m3_str = "   N/A  "
                m3_ape_str = "  N/A  "

            m1_apes.append(m1_ape)
            m2_apes.append(m2_ape)

            print(f"  {bank:<28} {actual_val/1e6:>7.2f}M  {m1/1e6:>7.2f}M {m1_ape:>6.1f}%  {m2/1e6:>7.2f}M {m2_ape:>6.1f}%  {m3_str} {m3_ape_str}")

            all_results.append({
                "month": month_label, "bank": bank, "card_type": card_type,
                "actual": actual_val, "m1": m1, "m2": m2, "m3": m3_val,
                "m1_ape": m1_ape, "m2_ape": m2_ape, "m3_ape": m3_ape if m3_val else None,
            })

        print(f"\n  {'SUMMARY':<28} {'':>9}  {'':>7}  {'Median':>7}  {'':>7}  {'Median':>7}  {'':>7}  {'Median':>7}")
        m1_med = float(np.median(m1_apes)) if m1_apes else float("nan")
        m2_med = float(np.median(m2_apes)) if m2_apes else float("nan")
        m3_med = float(np.median(m3_apes)) if m3_apes else float("nan")
        m1_mean = float(np.mean(m1_apes)) if m1_apes else float("nan")
        m2_mean = float(np.mean(m2_apes)) if m2_apes else float("nan")
        m3_mean = float(np.mean(m3_apes)) if m3_apes else float("nan")

        print(f"  {'Median APE':<28} {'':>18} {m1_med:>6.1f}%  {'':>9} {m2_med:>6.1f}%  {'':>9} {m3_med:>6.1f}%")
        print(f"  {'Mean APE':<28} {'':>18} {m1_mean:>6.1f}%  {'':>9} {m2_mean:>6.1f}%  {'':>9} {m3_mean:>6.1f}%")

    # Combined summary across both months
    df = pd.DataFrame(all_results)
    print(f"\n{'─'*100}")
    print(f"  COMBINED (Mar + Apr 2026, {len(df[df.m3_ape.notna()])} bank-month observations)")
    print(f"{'─'*100}")
    print(f"  {'Metric':<30} {'Method 1 Fixed Share':>22} {'Method 2 Trend Share':>22} {'Method 3 Ground-Up':>22}")
    print(f"  {'-'*96}")

    for metric, fn in [("Median APE", np.median), ("Mean APE", np.mean),
                        ("% banks within 5%", None), ("% banks within 15%", None)]:
        if fn:
            m1v = fn(df.m1_ape.dropna()) if len(df.m1_ape.dropna()) else float("nan")
            m2v = fn(df.m2_ape.dropna()) if len(df.m2_ape.dropna()) else float("nan")
            m3v = fn(df.m3_ape.dropna()) if len(df.m3_ape.dropna()) else float("nan")
            print(f"  {metric:<30} {m1v:>20.1f}%  {m2v:>20.1f}%  {m3v:>20.1f}%")
        else:
            thr = 5 if "5%" in metric else 15
            m1v = (df.m1_ape < thr).mean() * 100
            m2v = (df.m2_ape < thr).mean() * 100
            m3v = (df.m3_ape.dropna() < thr).mean() * 100 if df.m3_ape.notna().any() else float("nan")
            print(f"  {metric:<30} {m1v:>20.1f}%  {m2v:>20.1f}%  {m3v:>20.1f}%")

    return df


if __name__ == "__main__":
    cc_results = run_comparison("cc", CC_BANKS, cc_bw, "cc_outstanding", "CREDIT CARDS")
    dc_results = run_comparison("dc", DC_BANKS, dc_bw, "dc_outstanding", "DEBIT CARDS")

    print("\n" + "=" * 100)
    print("OVERALL VERDICT")
    print("=" * 100)
    all_df = pd.concat([cc_results, dc_results])
    for method, col in [("Method 1 — Fixed Share", "m1_ape"),
                         ("Method 2 — Trend Share", "m2_ape"),
                         ("Method 3 — Ground-Up",   "m3_ape")]:
        vals = all_df[col].dropna()
        if len(vals):
            print(f"  {method:<30}  Median {np.median(vals):.1f}%  |  Mean {np.mean(vals):.1f}%  |  <5%: {(vals<5).mean()*100:.0f}%  |  <15%: {(vals<15).mean()*100:.0f}%")
