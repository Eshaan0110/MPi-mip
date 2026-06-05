"""
FAIR Three-Method Comparison
==============================
All methods trained on data available BEFORE March 2026.
No method sees the actual March/April 2026 totals.

Method 1 — Fixed Share:
  Share = bank / total at May 2025 (last bankwise month)
  Forecast = share * PSI_model_forecast_for_that_month

Method 2 — Trend Share:
  Fit linear trend to each bank's share over Jan 2024 - May 2025 (16 months)
  Project share to March/April 2026
  Forecast = projected_share * PSI_model_forecast_for_that_month

Method 3 — Ground-Up:
  Individual Prophet model forecast per bank (trained up to May 2025)

PSI model was trained on PSI data up to Feb 2026 — completely independent
of bankwise data, so no leakage.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import warnings; warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from src.ingestion.bankwise import ingest, canonical_bank

# ── PSI model forecasts (trained on PSI up to Feb 2026) ───────────────────
PSI_FORECAST = {
    "cc": {"2026-03": 1168.536319 * 1e5, "2026-04": 1177.617898 * 1e5},
    "dc": {"2026-03": 10362.417665 * 1e5, "2026-04": 10376.779021 * 1e5},
}

# ── Actuals ────────────────────────────────────────────────────────────────
ACTUALS_FILES = [
    ("March 2026", "2026-03", Path(r"C:\Users\ASUS\Downloads\ATMMARCH20265F48EF7056E84759B802285873DD3FB3.XLSX")),
    ("April 2026", "2026-04", Path(r"C:\Users\ASUS\Downloads\ATMAPRIL20265AF3EE208AF746979B772EED805CDCA8.XLSX")),
]

# ── Training data (up to May 2025) ─────────────────────────────────────────
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


def get_fixed_shares(bw, col, banks):
    """Share at May 2025 — last known bankwise month."""
    ref = pd.Timestamp("2025-05-01")
    latest = bw[bw.date == ref][["bank_name", col]].dropna()
    # total across ALL banks at that date (not just top banks)
    total = bw[bw.date == ref][col].sum()
    shares = {}
    for b in banks:
        v = latest[latest.bank_name == b][col]
        shares[b] = float(v.iloc[0]) / total if len(v) > 0 and total > 0 else 0.0
    return shares, total


def get_trend_shares(bw, col, banks, n_months=16):
    """Linear trend on each bank's share over last 16 months of training data."""
    ref = pd.Timestamp("2025-05-01")
    all_dates = sorted(bw[bw.date <= ref].date.unique())
    window_dates = all_dates[-n_months:]

    # Total across ALL banks per month
    monthly_total = (bw[bw.date.isin(window_dates)]
                     .groupby("date")[col].sum())

    trends = {}
    for b in banks:
        bs = (bw[(bw.bank_name == b) & (bw.date.isin(window_dates))]
              .set_index("date")[col])
        aligned = bs.reindex(window_dates)
        shares = aligned / monthly_total
        shares = shares.dropna()
        if len(shares) < 4:
            trends[b] = None
            continue
        t = np.arange(len(shares))
        m, c = np.polyfit(t, shares.values, 1)
        trends[b] = (m, c, len(shares))
    return trends


def project_share(trends, fixed_shares, bank, months_ahead):
    """Project share n months beyond end of training (May 2025)."""
    if trends.get(bank) is None:
        return fixed_shares.get(bank, 0.0)
    m, c, n = trends[bank]
    proj = c + m * (n - 1 + months_ahead)
    return max(0.0, proj)


def get_ground_up(card_type, banks, month_str):
    forecasts = {}
    for bank in banks:
        safe = bank.lower().replace(" ","_").replace(".","").replace("/","_")
        p = FC_DIR / f"{card_type}_{safe}_forecast.csv"
        if not p.exists(): continue
        fc = pd.read_csv(p)
        row = fc[fc.date.str.startswith(month_str)]
        if row.empty: continue
        forecasts[bank] = float(row.forecast.iloc[0])
    return forecasts


def run_fair_comparison(card_type, banks, bw, col, label):
    print()
    print("=" * 100)
    print(f"{label}  [FAIR — all methods use only data available before March 2026]")
    print("=" * 100)

    fixed_shares, bankwise_total_may25 = get_fixed_shares(bw, col, banks)
    trend_data = get_trend_shares(bw, col, banks)

    # Months from May 2025 to target: Mar 2026 = 10, Apr 2026 = 11
    months_ahead_map = {"2026-03": 10, "2026-04": 11}

    all_rows = []

    for month_label, month_str, filepath in ACTUALS_FILES:
        psi_total = PSI_FORECAST[card_type][month_str]
        months_ahead = months_ahead_map[month_str]

        actual_df = ingest(filepath, verbose=False)
        actual_df["bank"] = actual_df["bank"].apply(canonical_bank)
        actual_col = "credit_outstanding" if card_type == "cc" else "debit_outstanding"
        actual_grand_total = actual_df[actual_col].sum()
        psi_error_pct = (psi_total - actual_grand_total) / actual_grand_total * 100

        gu = get_ground_up(card_type, banks, month_str)

        print(f"\n{month_label}")
        print(f"  PSI model forecast total : {psi_total/1e6:,.1f}M cards")
        print(f"  Actual total             : {actual_grand_total/1e6:,.1f}M cards")
        print(f"  PSI model total error    : {psi_error_pct:+.1f}%")
        print()
        print(f"  {'Bank':<28} {'Actual':>9} {'M1 Fixed':>9} {'APE':>6} {'M2 Trend':>9} {'APE':>6} {'M3 GndUp':>9} {'APE':>6}")
        print("  " + "-" * 90)

        m1_apes, m2_apes, m3_apes = [], [], []

        for bank in banks:
            row = actual_df[actual_df.bank == bank][actual_col]
            if row.empty or row.isna().all(): continue
            actual_val = float(row.iloc[0])
            if actual_val <= 0: continue

            # Method 1
            m1 = fixed_shares.get(bank, 0) * psi_total
            m1_ape = abs(m1 - actual_val) / actual_val * 100

            # Method 2
            ps = project_share(trend_data, fixed_shares, bank, months_ahead)
            m2 = ps * psi_total
            m2_ape = abs(m2 - actual_val) / actual_val * 100

            # Method 3
            m3_val = gu.get(bank)
            if m3_val is not None:
                m3_ape = abs(m3_val - actual_val) / actual_val * 100
                m3_apes.append(m3_ape)
                m3_str = f"{m3_val/1e6:>7.2f}M"
                m3_ape_str = f"{m3_ape:>5.1f}%"
            else:
                m3_str = "   N/A  "
                m3_ape_str = "  N/A "

            m1_apes.append(m1_ape)
            m2_apes.append(m2_ape)

            print(f"  {bank:<28} {actual_val/1e6:>7.2f}M  {m1/1e6:>7.2f}M {m1_ape:>5.1f}%  {m2/1e6:>7.2f}M {m2_ape:>5.1f}%  {m3_str} {m3_ape_str}")

            all_rows.append({"month": month_label, "bank": bank,
                             "actual": actual_val, "m1": m1, "m2": m2, "m3": m3_val,
                             "m1_ape": m1_ape, "m2_ape": m2_ape,
                             "m3_ape": m3_ape if m3_val else None})

        print(f"\n  Median APE  {np.median(m1_apes):>27.1f}%  {np.median(m2_apes):>17.1f}%  {np.median(m3_apes) if m3_apes else float('nan'):>17.1f}%")
        print(f"  Mean APE    {np.mean(m1_apes):>27.1f}%  {np.mean(m2_apes):>17.1f}%  {np.mean(m3_apes) if m3_apes else float('nan'):>17.1f}%")

    # Combined
    df = pd.DataFrame(all_rows)
    print(f"\n{'─'*100}")
    print(f"  COMBINED ({len(df)} bank-month observations — no method saw the actual totals)")
    print(f"{'─'*100}")
    print(f"  {'Metric':<28} {'Method 1 — Fixed Share':>24} {'Method 2 — Trend Share':>24} {'Method 3 — Ground-Up':>22}")
    print(f"  {'-'*98}")

    for metric, fn, thr in [
        ("Median APE",          np.median, None),
        ("Mean APE",            np.mean,   None),
        ("Banks within 5%",     None,      5),
        ("Banks within 10%",    None,      10),
        ("Banks within 20%",    None,      20),
    ]:
        if fn:
            v1 = fn(df.m1_ape.dropna());  v2 = fn(df.m2_ape.dropna());  v3 = fn(df.m3_ape.dropna())
            print(f"  {metric:<28} {v1:>22.1f}%  {v2:>22.1f}%  {v3:>20.1f}%")
        else:
            v1 = (df.m1_ape < thr).mean()*100
            v2 = (df.m2_ape < thr).mean()*100
            v3 = (df.m3_ape.dropna() < thr).mean()*100
            print(f"  {metric:<28} {v1:>22.1f}%  {v2:>22.1f}%  {v3:>20.1f}%")

    return df


if __name__ == "__main__":
    cc = run_fair_comparison("cc", CC_BANKS, cc_bw, "cc_outstanding", "CREDIT CARDS")
    dc = run_fair_comparison("dc", DC_BANKS, dc_bw, "dc_outstanding", "DEBIT CARDS")

    all_df = pd.concat([cc, dc])
    print("\n" + "=" * 100)
    print("FINAL SCORECARD — All Banks, Both Card Types, March + April 2026")
    print("(All methods used only data available before March 2026)")
    print("=" * 100)
    for name, col in [("Method 1 — Fixed Share", "m1_ape"),
                       ("Method 2 — Trend Share", "m2_ape"),
                       ("Method 3 — Ground-Up",   "m3_ape")]:
        v = all_df[col].dropna()
        print(f"  {name:<30}  Median {np.median(v):.1f}%  |  Mean {np.mean(v):.1f}%  "
              f"|  <5%: {(v<5).mean()*100:.0f}%  |  <10%: {(v<10).mean()*100:.0f}%  "
              f"|  <20%: {(v<20).mean()*100:.0f}%")
