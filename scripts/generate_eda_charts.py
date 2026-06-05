"""Generate per-bank EDA charts as PNGs for the Word doc."""
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path

CC_DEFAULT_START = pd.Timestamp("2013-01-01")
DC_DEFAULT_START = pd.Timestamp("2017-01-01")

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

def get_start(bank, ct):
    return BANK_START_DATES.get((bank, ct), CC_DEFAULT_START if ct == "cc" else DC_DEFAULT_START)

CC_BANKS = ["HDFC Bank","State Bank of India","ICICI Bank","Axis Bank",
            "Kotak Mahindra Bank","IndusInd Bank","Bank of Baroda",
            "Yes Bank","Canara Bank","HSBC"]
DC_BANKS = ["State Bank of India","Bank of Baroda","Canara Bank","HDFC Bank",
            "Union Bank of India","Punjab National Bank","Axis Bank",
            "Bank of India","Kotak Mahindra Bank","Indian Bank",
            "Central Bank of India","UCO Bank","ICICI Bank",
            "Indian Overseas Bank","Paytm Payments Bank"]

CC_VARS = {"atm_offsite":"ATMs Off-site","pos_terminals":"PoS Terminals",
           "cc_pos_vol":"CC PoS Txn Vol","cc_atm_cash_vol":"CC ATM Cash Vol"}
DC_VARS = {"atm_onsite":"ATMs On-site","pos_terminals":"PoS Terminals",
           "dc_pos_vol":"DC PoS Txn Vol","dc_atm_cash_vol":"DC ATM Cash Vol"}

cc_bw = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc_bw = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
for df in [cc_bw, dc_bw]:
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

out_dir = Path("reports/eda_charts")
out_dir.mkdir(parents=True, exist_ok=True)

COLORS = {"outstanding":"#c0392b","var":"#2980b9"}

def granger_p(y, x, lag):
    df = pd.DataFrame({"y":y.values,"x":x.values},index=y.index).dropna()
    if len(df) < lag*3+10: return None
    dy = df["y"].diff().dropna(); dx = df["x"].diff().dropna()
    data = pd.DataFrame({"y":dy.values[:len(dx)],"x":dx.values[:len(dy)]}).dropna()
    if len(data) < lag*3+5: return None
    try:
        res = grangercausalitytests(data[["y","x"]], maxlag=lag, verbose=False)
        return res[lag][0]["ssr_ftest"][1]
    except: return None

def make_chart(bdf, target_col, col, var_label, bank, card_type):
    bdf2 = bdf.set_index("date")[[target_col, col]].dropna()
    if len(bdf2) < 6: return None
    y, x = bdf2[target_col], bdf2[col]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle(f"{bank}  —  {var_label}", fontsize=11, fontweight="bold", y=1.01)

    # Left: dual-axis time series
    ax1 = axes[0]
    ax1r = ax1.twinx()
    l1, = ax1.plot(x.index, x.values, color=COLORS["var"], lw=1.8, label=var_label)
    l2, = ax1r.plot(y.index, y.values, color=COLORS["outstanding"], lw=1.8,
                    linestyle="--", label="Cards Outstanding")
    ax1.set_ylabel(var_label, color=COLORS["var"], fontsize=8)
    ax1r.set_ylabel("Cards Outstanding", color=COLORS["outstanding"], fontsize=8)
    ax1.tick_params(axis="y", labelcolor=COLORS["var"], labelsize=7)
    ax1r.tick_params(axis="y", labelcolor=COLORS["outstanding"], labelsize=7)
    ax1.tick_params(axis="x", labelsize=7)
    ax1.set_title("Time Series", fontsize=9)
    ax1.legend(handles=[l1,l2], fontsize=7, loc="upper left")

    # Right: scatter
    ax2 = axes[1]
    rho, p_rho = stats.spearmanr(x.values, y.values)
    ax2.scatter(x.values, y.values, alpha=0.55, s=18, color=COLORS["var"])
    try:
        m, b = np.polyfit(x.values, y.values, 1)
        xl = np.linspace(x.min(), x.max(), 100)
        ax2.plot(xl, m*xl+b, color=COLORS["outstanding"], lw=1.8)
    except: pass
    ax2.set_xlabel(var_label, fontsize=8)
    ax2.set_ylabel("Cards Outstanding", fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.set_title(f"Scatter  (Spearman ρ = {rho:+.3f}, p={p_rho:.3f})", fontsize=9)

    plt.tight_layout()
    safe = bank.lower().replace(" ","_").replace(".","")
    safe_col = col.replace("_","")
    path = out_dir / f"{card_type}_{safe}_{safe_col}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")
    return str(path)

print("Generating CC charts...")
for bank in CC_BANKS:
    start = get_start(bank, "cc")
    bdf = cc_bw[(cc_bw.bank_name==bank)&(cc_bw.date>=start)].sort_values("date").reset_index(drop=True)
    for col, lbl in CC_VARS.items():
        if col in bdf.columns and bdf[col].notna().sum() >= 6:
            make_chart(bdf, "cc_outstanding", col, lbl, bank, "cc")

print("Generating DC charts...")
for bank in DC_BANKS:
    start = get_start(bank, "dc")
    bdf = dc_bw[(dc_bw.bank_name==bank)&(dc_bw.date>=start)].sort_values("date").reset_index(drop=True)
    for col, lbl in DC_VARS.items():
        if col in bdf.columns and bdf[col].notna().sum() >= 6:
            make_chart(bdf, "dc_outstanding", col, lbl, bank, "dc")

print(f"\nAll charts saved to {out_dir}")
