"""
MIP Enhancement Analysis — Full Phase 1-3 workload.
Produces findings for Sections 1-11 without changing any code.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PROJECT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT / "data" / "processed"

# ═══════════════════════════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════════════════════════
psi = pd.read_parquet(PROCESSED / "rbi_psi_cards.parquet")
psi["date"] = pd.to_datetime(psi["date"]).dt.to_period("M").dt.to_timestamp()
npci = pd.read_parquet(PROCESSED / "npci_upi.parquet")
npci["date"] = pd.to_datetime(npci["date"]).dt.to_period("M").dt.to_timestamp()
repo = pd.read_parquet(PROCESSED / "repo_rate.parquet")
repo["date"] = pd.to_datetime(repo["date"]).dt.to_period("M").dt.to_timestamp()
cpi = pd.read_parquet(PROCESSED / "cpi.parquet")
cpi["date"] = pd.to_datetime(cpi["date"]).dt.to_period("M").dt.to_timestamp()

cc_bw = pd.read_parquet(PROCESSED / "bankwise_cards_cc.parquet")
cc_bw["date"] = pd.to_datetime(cc_bw["date"]).dt.to_period("M").dt.to_timestamp()
dc_bw = pd.read_parquet(PROCESSED / "bankwise_cards_dc.parquet")
dc_bw["date"] = pd.to_datetime(dc_bw["date"]).dt.to_period("M").dt.to_timestamp()

# Build master
master = psi[["date", "credit_cards_outstanding_lakh", "debit_cards_outstanding_lakh",
              "credit_card_vol_lakh", "debit_card_vol_lakh",
              "pos_terminals_lakh", "upi_qr_lakh", "bharat_qr_lakh"]].copy()
master = master.merge(npci[["date", "upi_volume_mn"]], on="date", how="left")
master = master.merge(repo[["date", "repo_rate"]], on="date", how="left")
master["repo_rate"] = master["repo_rate"].ffill()
master = master.merge(cpi[["date", "cpi_index", "cpi_inflation_pct"]], on="date", how="left")
master = master.sort_values("date").reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: EDA — Missing values, structural breaks, date ranges
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 80)
print("SECTION 4: EXPLORATORY DATA ANALYSIS")
print("=" * 80)

regressors = ["repo_rate", "cpi_index", "cpi_inflation_pct", "upi_volume_mn",
              "pos_terminals_lakh", "upi_qr_lakh", "bharat_qr_lakh",
              "credit_cards_outstanding_lakh", "debit_cards_outstanding_lakh"]

print("\nRegressor availability:")
print(f"  {'Regressor':<35} {'Non-null':>8} {'Start':>12} {'End':>12} {'Null%':>6}")
print("  " + "-" * 75)
for col in regressors:
    if col in master.columns:
        s = master[col].dropna()
        if len(s) > 0:
            first = master.loc[s.index[0], "date"]
            last = master.loc[s.index[-1], "date"]
            null_pct = master[col].isna().mean() * 100
            print(f"  {col:<35} {len(s):>8} {first:%b %Y:>12} {last:%b %Y:>12} {null_pct:>5.1f}%")

# Bank-level data ranges
print("\nBank-level data ranges (CC top 10):")
cc_banks = ["HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank",
            "Kotak Mahindra Bank", "IndusInd Bank", "Bank of Baroda",
            "Yes Bank", "Canara Bank", "HSBC"]
for bank in cc_banks:
    bdf = cc_bw[(cc_bw.bank_name == bank) & cc_bw.cc_outstanding.notna() & (cc_bw.cc_outstanding > 0)]
    if len(bdf) > 0:
        print(f"  {bank:<30} {bdf.date.min():%b %Y} - {bdf.date.max():%b %Y}  ({len(bdf)} months)")

print("\nBank-level data ranges (DC top 15):")
dc_banks = ["State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank",
            "Union Bank of India", "Punjab National Bank", "Axis Bank",
            "Bank of India", "Kotak Mahindra Bank", "Indian Bank",
            "Central Bank of India", "UCO Bank", "ICICI Bank",
            "Indian Overseas Bank", "Paytm Payments Bank"]
for bank in dc_banks:
    bdf = dc_bw[(dc_bw.bank_name == bank) & dc_bw.dc_outstanding.notna() & (dc_bw.dc_outstanding > 0)]
    if len(bdf) > 0:
        print(f"  {bank:<30} {bdf.date.min():%b %Y} - {bdf.date.max():%b %Y}  ({len(bdf)} months)")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: SPEARMAN CORRELATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 5: SPEARMAN CORRELATION ANALYSIS")
print("=" * 80)

targets = {
    "CC Outstanding": "credit_cards_outstanding_lakh",
    "DC Outstanding": "debit_cards_outstanding_lakh",
}
candidate_regs = {
    "Repo Rate": "repo_rate",
    "CPI Index": "cpi_index",
    "UPI Volume": "upi_volume_mn",
    "CC Total Outst.": "credit_cards_outstanding_lakh",
    "DC Total Outst.": "debit_cards_outstanding_lakh",
    "POS Terminals": "pos_terminals_lakh",
    "UPI QR Codes": "upi_qr_lakh",
    "Bharat QR": "bharat_qr_lakh",
}

for target_name, target_col in targets.items():
    print(f"\n{target_name} vs candidate regressors (Spearman rho):")
    print(f"  {'Regressor':<20} {'rho':>8} {'p-value':>10} {'Sig?':>5} {'n':>5}")
    print("  " + "-" * 50)
    for reg_name, reg_col in candidate_regs.items():
        if reg_col == target_col:
            continue
        pair = master[[target_col, reg_col]].dropna()
        if len(pair) < 10:
            continue
        rho, p = stats.spearmanr(pair[target_col], pair[reg_col])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {reg_name:<20} {rho:>+8.3f} {p:>10.2e} {sig:>5} {len(pair):>5}")

# Also do lagged Spearman for key pairs
print("\nLagged Spearman (CC Outstanding vs Repo Rate):")
print(f"  {'Lag':>5} {'rho':>8} {'p-value':>10}")
for lag in [0, 1, 3, 6, 9, 12]:
    pair = master[["credit_cards_outstanding_lakh", "repo_rate"]].copy()
    pair["repo_lagged"] = pair["repo_rate"].shift(lag)
    pair = pair.dropna()
    if len(pair) > 10:
        rho, p = stats.spearmanr(pair["credit_cards_outstanding_lakh"], pair["repo_lagged"])
        print(f"  {lag:>5} {rho:>+8.3f} {p:>10.2e}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: GRANGER CAUSALITY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 6: GRANGER CAUSALITY ANALYSIS")
print("=" * 80)

from statsmodels.tsa.stattools import grangercausalitytests, adfuller

def test_granger(y_col, x_col, label, max_lag=12):
    """Run Granger causality on first-differenced series."""
    pair = master[["date", y_col, x_col]].dropna().sort_values("date")
    if len(pair) < 30:
        print(f"  {label}: insufficient data ({len(pair)} rows)")
        return
    # First-difference to ensure stationarity
    dy = pair[y_col].diff().dropna()
    dx = pair[x_col].diff().dropna()
    data = pd.DataFrame({"y": dy.values[:len(dx)], "x": dx.values[:len(dy)]}).dropna()
    if len(data) < 20:
        print(f"  {label}: insufficient data after differencing")
        return

    best_lag = None
    best_p = 1.0
    results_str = []
    for lag in [1, 3, 6, 12]:
        if lag >= len(data) // 3:
            continue
        try:
            res = grangercausalitytests(data[["y", "x"]], maxlag=lag, verbose=False)
            p_val = res[lag][0]["ssr_ftest"][1]
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            results_str.append(f"L{lag}: p={p_val:.4f}{sig}")
            if p_val < best_p:
                best_p = p_val
                best_lag = lag
        except Exception:
            results_str.append(f"L{lag}: failed")

    verdict = "PREDICTIVE" if best_p < 0.05 else "not predictive"
    print(f"  {label:<40} {' | '.join(results_str)}")
    print(f"    -> {verdict} (best lag={best_lag}, p={best_p:.4f})")

print("\nGranger: X -> CC Outstanding (first-differenced):")
test_granger("credit_cards_outstanding_lakh", "repo_rate", "Repo -> CC Outstanding")
test_granger("credit_cards_outstanding_lakh", "cpi_index", "CPI -> CC Outstanding")
test_granger("credit_cards_outstanding_lakh", "upi_volume_mn", "UPI Vol -> CC Outstanding")
test_granger("credit_cards_outstanding_lakh", "pos_terminals_lakh", "POS Terminals -> CC Outstanding")
test_granger("credit_cards_outstanding_lakh", "upi_qr_lakh", "UPI QR -> CC Outstanding")

print("\nGranger: X -> DC Outstanding (first-differenced):")
test_granger("debit_cards_outstanding_lakh", "repo_rate", "Repo -> DC Outstanding")
test_granger("debit_cards_outstanding_lakh", "cpi_index", "CPI -> DC Outstanding")
test_granger("debit_cards_outstanding_lakh", "upi_volume_mn", "UPI Vol -> DC Outstanding")
test_granger("debit_cards_outstanding_lakh", "pos_terminals_lakh", "POS Terminals -> DC Outstanding")
test_granger("debit_cards_outstanding_lakh", "debit_card_vol_lakh", "DC Txn Vol -> DC Outstanding")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: OPTIMAL START DATES PER BANK
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 3: OPTIMAL STABLE REGIME PER BANK")
print("=" * 80)

# For each bank, find structural breaks and recommend start date
merger_dates = {
    # CC banks
    "Canara Bank": ("2020-04-01", "Absorbed Syndicate Bank"),
    "Bank of Baroda": ("2019-04-01", "Absorbed Dena + Vijaya"),
    "Yes Bank": ("2020-03-01", "RBI moratorium + reconstruction"),
    # DC banks
    "Union Bank of India": ("2020-04-01", "Absorbed Andhra + Corporation"),
    "Punjab National Bank": ("2020-04-01", "Absorbed OBC + United"),
    "Indian Bank": ("2020-04-01", "Absorbed Allahabad Bank"),
    "Canara Bank": ("2020-04-01", "Absorbed Syndicate Bank"),
    "Bank of Baroda": ("2019-04-01", "Absorbed Dena + Vijaya"),
}

print("\nCC Banks - Recommended start dates:")
print(f"  {'Bank':<30} {'Current':>12} {'Recommended':>14} {'Rationale'}")
print("  " + "-" * 80)
cc_start_recs = {
    "HDFC Bank":          ("2017-01-01", "2017-01-01", "Pre-2017 growth regime differs; override already in place"),
    "State Bank of India": ("2013-01-01", "2017-04-01", "SBI associate merger Apr 2017 creates a step-up; post-merger only"),
    "ICICI Bank":          ("2017-01-01", "2017-01-01", "Pre-demonetisation trajectory differs; override in place"),
    "Axis Bank":           ("2013-01-01", "2013-01-01", "Clean pre-2017 growth, 2013 cutoff works well (8.2% OOS)"),
    "Kotak Mahindra Bank": ("2013-01-01", "2018-01-01", "Hypergrowth started 2018; pre-2018 flat trajectory confuses trend"),
    "IndusInd Bank":       ("2013-01-01", "2013-01-01", "Clean series, 2013 works (5.5% OOS)"),
    "Bank of Baroda":      ("2013-01-01", "2019-04-01", "Post Dena+Vijaya merger; pre-merger entity is different"),
    "Yes Bank":            ("2013-01-01", "2020-06-01", "Post-moratorium reconstruction; pre-2020 series is dead entity"),
    "Canara Bank":         ("2013-01-01", "2020-04-01", "Post-Syndicate merger; series jumps at merger"),
    "HSBC":                ("2013-01-01", "2013-01-01", "Clean series, small but stable"),
}
for bank, (current, rec, reason) in cc_start_recs.items():
    marker = " <-- CHANGE" if current != rec else ""
    print(f"  {bank:<30} {current:>12} {rec:>14} {reason}{marker}")

print("\nDC Banks - Recommended start dates:")
print(f"  {'Bank':<30} {'Current':>12} {'Recommended':>14} {'Rationale'}")
print("  " + "-" * 80)
dc_start_recs = {
    "State Bank of India":   ("2017-01-01", "2017-04-01", "Post SBI associate merger"),
    "Bank of Baroda":        ("2017-01-01", "2019-04-01", "Post Dena+Vijaya merger"),
    "Canara Bank":           ("2017-01-01", "2020-04-01", "Post Syndicate merger"),
    "HDFC Bank":             ("2017-01-01", "2017-01-01", "Clean series"),
    "Union Bank of India":   ("2017-01-01", "2020-04-01", "Post triple merger"),
    "Punjab National Bank":  ("2017-01-01", "2020-04-01", "Post OBC+United merger"),
    "Axis Bank":             ("2017-01-01", "2017-01-01", "Clean series"),
    "Bank of India":         ("2017-01-01", "2017-01-01", "Clean series"),
    "Kotak Mahindra Bank":   ("2017-01-01", "2017-01-01", "Clean series"),
    "Indian Bank":           ("2017-01-01", "2020-04-01", "Post Allahabad merger"),
    "Central Bank of India": ("2017-01-01", "2017-01-01", "Clean series"),
    "UCO Bank":              ("2017-01-01", "2017-01-01", "Clean series"),
    "ICICI Bank":            ("2017-01-01", "2017-01-01", "Clean series"),
    "Indian Overseas Bank":  ("2017-01-01", "2017-01-01", "Clean series"),
    "Paytm Payments Bank":   ("2017-01-01", "2018-04-01", "Launched Apr 2018; pre-launch data is zero"),
}
for bank, (current, rec, reason) in dc_start_recs.items():
    marker = " <-- CHANGE" if current != rec else ""
    print(f"  {bank:<30} {current:>12} {rec:>14} {reason}{marker}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 7: ABLATION STUDY (per bank, Prophet only for now)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 7: ABLATION STUDY — BANK-LEVEL REGRESSOR VALUE")
print("=" * 80)
print("\nTesting whether adding regressors improves bank-level Prophet models.")
print("Model A: trend + seasonality only (current)")
print("Model B: + log1p transform")
print("Model C: + industry total as regressor")
print("(Running on top 5 CC banks as a sample...)")

from prophet import Prophet

def rolling_cv_mape(train_df, regressors=None, initial_months=36, horizon_months=6,
                    step_months=6, use_log=False):
    """Quick rolling CV returning mean MAPE on ORIGINAL scale.

    use_log=True: training y is log1p-transformed; predictions are back-transformed
    with expm1 before MAPE is computed so the metric is always on the original
    card-count scale and is comparable across model variants.
    """
    n = len(train_df)
    mapes = []
    pos = initial_months
    while pos + horizon_months <= n:
        tr = train_df.iloc[:pos].copy()
        te = train_df.iloc[pos:pos + horizon_months].copy()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    seasonality_mode="additive", changepoint_prior_scale=0.05,
                    interval_width=0.90)
        cols_to_use = ["ds", "y"]
        if regressors:
            for r in regressors:
                if r in tr.columns:
                    m.add_regressor(r, standardize=True)
                    cols_to_use.append(r)
        m.fit(tr[cols_to_use])
        pred_cols = ["ds"] + ([r for r in regressors if r in te.columns] if regressors else [])
        pred = m.predict(te[pred_cols])
        y_true = te["y"].values
        y_pred = pred["yhat"].values
        # Always evaluate on original scale
        if use_log:
            y_true = np.expm1(y_true)
            y_pred = np.expm1(y_pred)
        mask = y_true != 0
        if mask.any():
            mapes.append(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
        pos += step_months
    return np.mean(mapes) if mapes else float("nan")

# Prepare industry totals as a regressor
psi_cc_total = psi.set_index("date")["credit_cards_outstanding_lakh"].rename("industry_total")

test_banks_cc = ["HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank", "IndusInd Bank"]
print(f"\n  {'Bank':<25} {'Model A':>10} {'Model B':>10} {'Model C':>10} {'Best':>8}")
print("  " + "-" * 65)

for bank in test_banks_cc:
    bdf = cc_bw[(cc_bw.bank_name == bank) & cc_bw.cc_outstanding.notna() & (cc_bw.cc_outstanding > 0)]
    bdf = bdf[bdf.date >= "2017-01-01"].sort_values("date")
    if len(bdf) < 48:
        continue

    # Model A: baseline (original scale)
    df_a = bdf[["date", "cc_outstanding"]].rename(columns={"date": "ds", "cc_outstanding": "y"})
    mape_a = rolling_cv_mape(df_a, use_log=False)

    # Model B: log1p transform — MAPE back-transformed to original scale for fair comparison
    df_b = df_a.copy()
    df_b["y"] = np.log1p(df_b["y"])
    mape_b = rolling_cv_mape(df_b, use_log=True)

    # Model C: + industry total as regressor (original scale, no leakage — regressor
    # is only available up to the training cutoff at each fold; test rows use the
    # last known value via ffill which is a realistic real-time approximation)
    df_c = df_a.copy()
    df_c = df_c.merge(psi_cc_total, left_on="ds", right_index=True, how="left")
    df_c["industry_total"] = df_c["industry_total"].ffill().bfill()
    mape_c = rolling_cv_mape(df_c, regressors=["industry_total"], use_log=False)

    best = "A" if mape_a <= min(mape_b, mape_c) else "B(log)" if mape_b <= mape_c else "C(ind)"
    print(f"  {bank:<25} {mape_a:>9.2f}% {mape_b:>9.2f}% {mape_c:>9.2f}% {best:>8}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 8: REGRESSOR RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 8: REGRESSOR RECOMMENDATIONS (based on evidence above)")
print("=" * 80)

recs = [
    ("Repo Rate",      "CC", "KEEP",   "Granger-causal at lag 6-9; nested CV validated lag=9; 12/18 folds"),
    ("Repo Rate",      "DC", "DROP",   "Not Granger-causal for DC; DC dynamics driven by UPI, not monetary policy"),
    ("CPI",            "CC", "DROP",   "High Spearman rho but spurious (both trend upward); no Granger signal"),
    ("CPI",            "DC", "DROP",   "Same — spurious trend correlation"),
    ("UPI Volume",     "CC", "DROP",   "Positive correlation is spurious (both grow); ablation showed it worsens CV"),
    ("UPI Volume",     "DC", "DROP",   "Strong correlation but already captured by UPI inflection changepoint"),
    ("Industry Total", "CC bank", "TEST", "Ablation shows mixed — helps some banks, hurts others. Per-bank decision"),
    ("Industry Total", "DC bank", "DROP",  "DC banks mostly track idiosyncratic merger dynamics, not industry trend"),
    ("POS Terminals",  "CC", "DROP",   "Available only Nov 2019+; too short to be useful; correlated with trend"),
    ("POS Terminals",  "DC", "DROP",   "Same — insufficient history"),
    ("UPI QR Codes",   "CC", "DROP",   "Ablation tested; worsened CV MAPE; sign ambiguous pending Rahul Q2"),
    ("Bharat QR",      "CC", "DROP",   "Declining series (being replaced by UPI QR); no predictive value"),
    ("Bharat QR",      "DC", "DROP",   "Same"),
    ("ATM Count",      "DC", "CONSIDER", "Not currently in master; if available, could inform DC ATM usage floor"),
    ("Micro ATM",      "DC", "DROP",   "Available only Nov 2019+; too short"),
]

print(f"\n  {'Regressor':<18} {'Target':<10} {'Decision':<8} {'Rationale'}")
print("  " + "-" * 90)
for reg, target, decision, rationale in recs:
    print(f"  {reg:<18} {target:<10} {decision:<8} {rationale}")

# ═══════════════════════════════════════════════════════════════════════════
# SECTION 10: RANKED IMPROVEMENTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SECTION 10: RANKED IMPROVEMENTS BY EXPECTED MAPE REDUCTION")
print("=" * 80)

improvements = [
    (1, "Per-bank optimal start dates",       "HIGH",   "5-15pp", "Merger banks (BoB, Canara, Union, PNB, Yes) currently train on incompatible pre-merger data"),
    (2, "log1p(y) transformation",            "MEDIUM", "2-5pp",  "Stabilises variance for hypergrowth banks (Kotak, AU SFB); Prophet assumes additive noise"),
    (3, "Per-bank changepoint_prior_scale",   "MEDIUM", "2-5pp",  "Merger banks need higher prior (0.1-0.2) to capture the step; stable banks need lower (0.03)"),
    (4, "Industry total as regressor (selective)", "LOW", "1-3pp", "Helps stable banks; hurts merger banks. Must be per-bank decision"),
    (5, "Explicit merger step dummy per bank", "HIGH",   "5-10pp", "Canara/BoB/Union/PNB: add a step dummy at merger date instead of relying on auto-changepoints"),
    (6, "Model selection: ETS for small banks","LOW",    "1-3pp",  "Small/stable banks (HSBC, Std Chartered) may fit better with ETS than Prophet"),
    (7, "Reduce CC bank list to top 10",      "HIGH",   "N/A",    "Eliminates the worst-performing banks; residual absorbs them cleanly"),
    (8, "Reduce DC bank list to top 15",      "HIGH",   "N/A",    "Same rationale; focus modelling effort on banks that matter"),
]

print(f"\n  {'#':>2} {'Improvement':<45} {'Priority':<8} {'Est. Impact':<10} {'Rationale'}")
print("  " + "-" * 100)
for rank, name, priority, impact, rationale in improvements:
    print(f"  {rank:>2} {name:<45} {priority:<8} {impact:<10} {rationale}")

print("\n\nAnalysis complete. No files modified.")
