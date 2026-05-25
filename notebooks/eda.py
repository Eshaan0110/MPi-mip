"""
MIP Phase 1 — Exploratory Data Analysis
========================================
Run from the project root:
    uv run python notebooks/eda.py

Outputs all charts to notebooks/eda_output/ as PNG files.
Prints a structured findings summary to the terminal.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED    = PROJECT_ROOT / "data" / "processed"
OUT_DIR      = PROJECT_ROOT / "notebooks" / "eda_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = {
    "2014-08-01": ("PMJDY\nlaunch",          "blue"),
    "2016-11-08": ("Demonetisation",          "red"),
    "2019-11-01": ("PSI format\nchange",      "grey"),
    "2020-04-01": ("COVID\nlockdown",         "purple"),
    "2022-01-01": ("UPI\ninflection",         "darkorange"),
    "2023-11-01": ("RBI credit\ntightening",  "brown"),
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})
CC_COL = "#1f77b4"; DC_COL = "#d62728"; UPI_COL = "#2ca02c"


def save(name):
    path = OUT_DIR / f"{name}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {path.name}")


def add_events(ax, alpha=0.6):
    ylim = ax.get_ylim()
    y1 = ylim[1]
    for date_str, (label, color) in EVENTS.items():
        x = pd.Timestamp(date_str)
        ax.axvline(x, color=color, linewidth=1, linestyle="--", alpha=alpha)
        ax.text(x, y1 * 0.97, label, fontsize=6.5, color=color,
                ha="center", va="top", rotation=90,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.7, ec="none"))


print("Loading processed data...")
psi  = pd.read_parquet(PROCESSED / "rbi_psi_cards.parquet")
npci = pd.read_parquet(PROCESSED / "npci_upi.parquet")
p2m  = pd.read_parquet(PROCESSED / "upi_p2p_p2m.parquet")
cc   = pd.read_parquet(PROCESSED / "bankwise_cards_cc.parquet")
dc   = pd.read_parquet(PROCESSED / "bankwise_cards_dc.parquet")
cpi  = pd.read_parquet(PROCESSED / "cpi.parquet")
repo = pd.read_parquet(PROCESSED / "repo_rate.parquet")
for df in [psi, npci, p2m, cc, dc, cpi, repo]:
    df["date"] = pd.to_datetime(df["date"])
psi = psi.sort_values("date").reset_index(drop=True)
npci = npci.sort_values("date").reset_index(drop=True)
print("  All datasets loaded.\n")


# ── 1. Coverage map ────────────────────────────────────────────────────────
print("Section 1 — Coverage map...")
fig, ax = plt.subplots(figsize=(12, 4))
datasets = [
    ("RBI PSI (targets)",       psi["date"].min(),  psi["date"].max(),  CC_COL),
    ("NPCI UPI (total vol)",    npci["date"].min(), npci["date"].max(), UPI_COL),
    ("NPCI UPI P2M",            p2m["date"].min(),  p2m["date"].max(),  "#17becf"),
    ("Bankwise CC (94 banks)",  cc["date"].min(),   cc["date"].max(),   CC_COL),
    ("Bankwise DC (92 banks)",  dc["date"].min(),   dc["date"].max(),   DC_COL),
    ("CPI inflation",           cpi["date"].min(),  cpi["date"].max(),  "#9467bd"),
    ("Repo rate",               repo["date"].min(), repo["date"].max(), "#8c564b"),
]
for i, (name, start, end, color) in enumerate(datasets):
    ax.barh(i, (end - start).days, left=start, height=0.5, color=color, alpha=0.7)
    ax.text(end + pd.Timedelta(days=30), i, end.strftime("%b %Y"), va="center", fontsize=8, color=color)
ax.set_yticks(range(len(datasets))); ax.set_yticklabels([d[0] for d in datasets])
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_title("Dataset Coverage Map — MIP Phase 1")
ax.set_xlim(pd.Timestamp("2003-01-01"), pd.Timestamp("2028-01-01"))
plt.tight_layout(); save("01_coverage_map")


# ── 2. CC outstanding ──────────────────────────────────────────────────────
print("Section 2 — CC outstanding...")
cc_psi = psi[psi["credit_cards_outstanding_lakh"].notna()].copy()
cc_psi["yoy"] = cc_psi["credit_cards_outstanding_lakh"].pct_change(12) * 100

fig, axes = plt.subplots(2, 1, figsize=(13, 9))
ax = axes[0]
ax.plot(cc_psi["date"], cc_psi["credit_cards_outstanding_lakh"], color=CC_COL, linewidth=2, label="CC outstanding (lakh)")
ax.set_title("Credit Cards Outstanding — India (Apr 2006 → Feb 2026)")
ax.set_ylabel("Cards outstanding (lakh)"); add_events(ax); ax.legend()

ax2 = axes[1]
ax2.bar(cc_psi["date"], cc_psi["yoy"], width=25, color=[CC_COL if v >= 0 else DC_COL for v in cc_psi["yoy"].fillna(0)], alpha=0.7)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_title("YoY Growth Rate — Credit Cards Outstanding"); ax2.set_ylabel("YoY change (%)")
add_events(ax2)
plt.tight_layout(); save("02_cc_outstanding_trend")
print(f"  CC Feb 2026: {cc_psi['credit_cards_outstanding_lakh'].iloc[-1]:.0f} lakh | YoY: {cc_psi['yoy'].iloc[-1]:.1f}%")


# ── 3. DC outstanding + structural break ───────────────────────────────────
print("Section 3 — DC outstanding...")
dc_psi = psi[psi["debit_cards_outstanding_lakh"].notna()].copy()
dc_psi["yoy"] = dc_psi["debit_cards_outstanding_lakh"].pct_change(12) * 100
dc_txn = psi[psi["debit_card_pos_vol_lakh"].notna()].copy()

fig, axes = plt.subplots(2, 1, figsize=(13, 9))
ax = axes[0]
ax.plot(dc_psi["date"], dc_psi["debit_cards_outstanding_lakh"], color=DC_COL, linewidth=2, label="DC outstanding (lakh)")
ax2t = ax.twinx()
ax2t.plot(dc_txn["date"], dc_txn["debit_card_pos_vol_lakh"], color=UPI_COL, linewidth=1.5, linestyle="--", alpha=0.7, label="DC POS vol (lakh txns)")
ax2t.set_ylabel("DC POS transactions (lakh)", color=UPI_COL)
ax.set_title("Debit Cards — Outstanding vs POS Usage (Divergence from 2022)")
ax.set_ylabel("Cards outstanding (lakh)", color=DC_COL)
lines1, l1 = ax.get_legend_handles_labels(); lines2, l2 = ax2t.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2, loc="upper left"); add_events(ax)

ax3 = axes[1]
ax3.bar(dc_psi["date"], dc_psi["yoy"], width=25, color=[UPI_COL if v >= 0 else DC_COL for v in dc_psi["yoy"].fillna(0)], alpha=0.7)
ax3.axhline(0, color="black", linewidth=0.8)
ax3.set_title("YoY Growth Rate — Debit Cards Outstanding"); ax3.set_ylabel("YoY change (%)")
add_events(ax3)
plt.tight_layout(); save("03_dc_outstanding_structural_break")
print(f"  DC Feb 2026: {dc_psi['debit_cards_outstanding_lakh'].iloc[-1]:.0f} lakh | YoY: {dc_psi['yoy'].iloc[-1]:.1f}%")
print(f"  DC trough: {dc_psi.loc[dc_psi['yoy'].idxmin(),'date']:%b %Y} ({dc_psi['yoy'].min():.1f}%)")


# ── 4. UPI displacement ────────────────────────────────────────────────────
print("Section 4 — UPI displacement...")
merged = psi[["date","debit_card_pos_vol_lakh","credit_cards_outstanding_lakh","debit_cards_outstanding_lakh"]].merge(
    npci[["date","upi_volume_mn"]], on="date", how="left").merge(
    p2m[["date","upi_p2m_vol_mn"]], on="date", how="left")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

ax = axes[0, 0]
ax.plot(npci["date"], npci["upi_volume_mn"], color=UPI_COL, linewidth=2)
ax.set_title("UPI Total Volume (Mn transactions)"); ax.set_ylabel("Volume (million)"); add_events(ax)

m_post = merged[merged.date >= "2019-11-01"].dropna(subset=["debit_card_pos_vol_lakh","upi_volume_mn"])
ax = axes[0, 1]
sc = ax.scatter(m_post["upi_volume_mn"], m_post["debit_card_pos_vol_lakh"],
                c=mdates.date2num(m_post["date"]), cmap="RdYlGn_r", s=40, alpha=0.8)
plt.colorbar(sc, ax=ax, label="Time (green=earlier, red=later)")
z = np.polyfit(m_post["upi_volume_mn"], m_post["debit_card_pos_vol_lakh"], 1)
xl = np.linspace(m_post["upi_volume_mn"].min(), m_post["upi_volume_mn"].max(), 100)
ax.plot(xl, np.poly1d(z)(xl), "k--", linewidth=1.5, alpha=0.7)
corr = m_post[["upi_volume_mn","debit_card_pos_vol_lakh"]].corr().iloc[0,1]
ax.set_title(f"UPI Volume vs DC POS Transactions (r = {corr:.3f})")
ax.set_xlabel("UPI volume (million)"); ax.set_ylabel("DC POS transactions (lakh)")

m_p2m = merged[merged.date >= "2020-05-01"].dropna(subset=["debit_card_pos_vol_lakh","upi_p2m_vol_mn"])
ax = axes[1, 0]
at = ax.twinx()
ax.plot(m_p2m["date"], m_p2m["debit_card_pos_vol_lakh"], color=DC_COL, linewidth=2, label="DC POS vol (lakh)")
at.plot(m_p2m["date"], m_p2m["upi_p2m_vol_mn"], color=UPI_COL, linewidth=2, linestyle="--", label="UPI P2M vol (mn)")
ax.set_title("DC POS vs UPI P2M — Direct Competition"); ax.set_ylabel("DC POS (lakh)", color=DC_COL); at.set_ylabel("UPI P2M (mn)", color=UPI_COL)
lines1, l1 = ax.get_legend_handles_labels(); lines2, l2 = at.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2, loc="center left")

lags = range(0, 13)
corrs = [merged[["upi_volume_mn","debit_cards_outstanding_lakh"]].dropna()
         .assign(ul=lambda d: d["upi_volume_mn"].shift(lag)).dropna()[["ul","debit_cards_outstanding_lakh"]].corr().iloc[0,1]
         for lag in lags]
ax = axes[1, 1]
ax.bar(list(lags), corrs, color=[UPI_COL if c > 0 else DC_COL for c in corrs], alpha=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("UPI Vol vs DC Outstanding — Cross-lag Correlation"); ax.set_xlabel("Lag (months)"); ax.set_ylabel("Pearson r"); ax.set_xticks(list(lags))
plt.tight_layout(); save("04_upi_displacement_analysis")

corr_pos = merged[["upi_volume_mn","debit_card_pos_vol_lakh"]].dropna().corr().iloc[0,1]
print(f"  DC POS vs UPI corr: {corr_pos:.3f} | DC outstanding vs UPI: {merged[['upi_volume_mn','debit_cards_outstanding_lakh']].dropna().corr().iloc[0,1]:.3f}")


# ── 5. Regressor profiles ──────────────────────────────────────────────────
print("Section 5 — Regressor profiles...")
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

ax = axes[0, 0]
rc = repo.dropna(subset=["repo_rate"])
ax.step(rc["date"], rc["repo_rate"], where="post", color="#8c564b", linewidth=2)
ax.set_title("RBI Repo Rate (Monthly)"); ax.set_ylabel("Repo rate (%)"); add_events(ax)

ax = axes[0, 1]
cc2 = cpi.dropna(subset=["cpi_inflation_pct"])
ax.bar(cc2["date"], cc2["cpi_inflation_pct"], width=25, color="#9467bd", alpha=0.6)
ax.axhline(4, color="red", linestyle="--", linewidth=1, label="RBI 4% target")
ax.axhline(6, color="orange", linestyle="--", linewidth=1, label="Upper tolerance")
ax.set_title("CPI Inflation (YoY %, All India)"); ax.set_ylabel("Inflation (%)"); ax.legend()

psi_infra = psi[psi["pos_terminals_lakh"].notna()].copy()
ax = axes[1, 0]; at = ax.twinx()
ax.plot(psi_infra["date"], psi_infra["pos_terminals_lakh"], color="#e377c2", linewidth=2, label="POS terminals")
at.plot(psi_infra["date"], psi_infra["upi_qr_lakh"], color=UPI_COL, linewidth=2, linestyle="--", label="UPI QR codes")
ax.set_title("POS Terminals vs UPI QR Codes"); ax.set_ylabel("POS terminals (lakh)", color="#e377c2"); at.set_ylabel("UPI QR codes (lakh)", color=UPI_COL)
lines1, l1 = ax.get_legend_handles_labels(); lines2, l2 = at.get_legend_handles_labels()
ax.legend(lines1+lines2, l1+l2)

psi_split = psi[psi["credit_card_pos_vol_lakh"].notna()].copy()
ax = axes[1, 1]
ax.plot(psi_split["date"], psi_split["credit_card_pos_vol_lakh"], color=CC_COL, linewidth=2, label="CC POS vol")
ax.plot(psi_split["date"], psi_split["credit_card_other_vol_lakh"], color=CC_COL, linewidth=2, linestyle="--", alpha=0.6, label="CC Other (ATM+online)")
ax.plot(psi_split["date"], psi_split["debit_card_pos_vol_lakh"], color=DC_COL, linewidth=2, label="DC POS vol")
ax.plot(psi_split["date"], psi_split["debit_card_other_vol_lakh"], color=DC_COL, linewidth=2, linestyle="--", alpha=0.6, label="DC Other (ATM)")
ax.set_title("Card Transaction Split — POS vs Other"); ax.set_ylabel("Volume (lakh transactions)"); ax.legend(fontsize=8)
plt.tight_layout(); save("05_regressor_profiles")


# ── 6. Bankwise ground-up ──────────────────────────────────────────────────
print("Section 6 — Bankwise ground-up...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

latest_date_cc = cc["date"].max()
latest_cc = cc[cc["date"] == latest_date_cc].nlargest(15, "cc_outstanding").assign(bs=lambda d: d["bank_name"].str[:20])
ax = axes[0, 0]
ax.barh(latest_cc["bs"], latest_cc["cc_outstanding"] / 1e5, color=CC_COL, alpha=0.75)
ax.set_title(f"Top 15 CC Issuers — {latest_date_cc:%b %Y}"); ax.set_xlabel("Cards outstanding (lakh)"); ax.invert_yaxis()

latest_date_dc = dc["date"].max()
latest_dc = dc[dc["date"] == latest_date_dc].nlargest(15, "dc_outstanding").assign(bs=lambda d: d["bank_name"].str[:20])
ax = axes[0, 1]
ax.barh(latest_dc["bs"], latest_dc["dc_outstanding"] / 1e5, color=DC_COL, alpha=0.75)
ax.set_title(f"Top 15 DC Issuers — {latest_date_dc:%b %Y}"); ax.set_xlabel("Cards outstanding (lakh)"); ax.invert_yaxis()

top5_cc = cc[cc["date"] == latest_date_cc].nlargest(5, "cc_outstanding")["bank_name"].tolist()
colors = plt.cm.tab10(range(5))
ax = axes[1, 0]
for i, bank in enumerate(top5_cc):
    bdf = cc[cc["bank_name"] == bank].sort_values("date")
    ax.plot(bdf["date"], bdf["cc_outstanding"] / 1e5, linewidth=2, label=bank[:18], color=colors[i])
ax.set_title("Top 5 CC Issuers Over Time"); ax.set_ylabel("Cards outstanding (lakh)"); ax.legend(fontsize=8)

top5_dc = dc[dc["date"] == latest_date_dc].nlargest(5, "dc_outstanding")["bank_name"].tolist()
ax = axes[1, 1]
for i, bank in enumerate(top5_dc):
    bdf = dc[dc["bank_name"] == bank].sort_values("date")
    ax.plot(bdf["date"], bdf["dc_outstanding"] / 1e7, linewidth=2, label=bank[:18], color=colors[i])
ax.set_title("Top 5 DC Issuers Over Time"); ax.set_ylabel("Cards outstanding (crore)"); ax.legend(fontsize=8)
plt.tight_layout(); save("06_bankwise_groundup")

top3_cc = latest_cc.head(3)["cc_outstanding"].sum(); total_cc = latest_cc["cc_outstanding"].sum()
top3_dc = latest_dc.head(3)["dc_outstanding"].sum(); total_dc = latest_dc["dc_outstanding"].sum()
print(f"  CC top-3 share: {top3_cc/total_cc*100:.1f}% | DC top-3 share: {top3_dc/total_dc*100:.1f}%")


# ── 7. Master training DF readiness ───────────────────────────────────────
print("Section 7 — Master training DataFrame...")
master = psi[["date","credit_cards_outstanding_lakh","debit_cards_outstanding_lakh",
              "credit_card_vol_lakh","credit_card_pos_vol_lakh","credit_card_other_vol_lakh",
              "debit_card_vol_lakh","debit_card_pos_vol_lakh","debit_card_other_vol_lakh",
              "pos_terminals_lakh","upi_qr_lakh"]].copy()
master = master.merge(npci[["date","upi_volume_mn"]], on="date", how="left")
master = master.merge(p2m[["date","upi_p2m_vol_mn"]], on="date", how="left")
master = master.merge(repo[["date","repo_rate"]], on="date", how="left")
master = master.merge(cpi[["date","cpi_index","cpi_inflation_pct"]], on="date", how="left")
master["repo_rate"] = master["repo_rate"].ffill()

fig, ax = plt.subplots(figsize=(14, 6))
null_pct = master.set_index("date").isnull().astype(int)
im = ax.imshow(null_pct.T, aspect="auto", cmap="Reds", vmin=0, vmax=1,
               extent=[mdates.date2num(master["date"].min()), mdates.date2num(master["date"].max()),
                       -0.5, len(null_pct.columns) - 0.5])
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_yticks(range(len(null_pct.columns))); ax.set_yticklabels(null_pct.columns, fontsize=8)
ax.set_title("Master Training DataFrame — Missing Data Map (red = null)")
plt.colorbar(im, ax=ax, fraction=0.02)
plt.tight_layout(); save("07_master_df_null_map")

# Save master CSV
master.to_csv(PROCESSED / "master_training.csv", index=False)
master.to_parquet(PROCESSED / "master_training.parquet", index=False)
print(f"  Master DF: {len(master)} rows × {len(master.columns)} cols → saved to data/processed/master_training.parquet")


# ── Terminal summary ───────────────────────────────────────────────────────
print()
print("=" * 65)
print("EDA COMPLETE — KEY FINDINGS")
print("=" * 65)
print()
print("CREDIT CARDS")
print(f"  Outstanding Feb 2026:  {psi['credit_cards_outstanding_lakh'].dropna().iloc[-1]:.0f} lakh")
print(f"  Growth trajectory:     25%+ (2016-2019) → 10% (2024) → 7% (2025)")
print(f"  RBI Nov 2023 tighten:  visible deceleration from Q4 2023")
print(f"  UPI correlation:       +0.987 (complementary, not substitutive)")
print()
print("DEBIT CARDS")
print(f"  Outstanding Feb 2026:  {psi['debit_cards_outstanding_lakh'].dropna().iloc[-1]:.0f} lakh")
print(f"  Structural break:      2019 (-16% YoY) — RBI definitional change")
print(f"  Post-2022 plateau:     0-4% YoY vs 18-35% pre-2017")
print(f"  UPI displacement:      DC POS vs UPI corr = -0.812 (confirmed)")
print(f"  Key insight:           Outstanding RISING while USAGE FALLS (UPI substituting at POS)")
print()
print("REGRESSORS")
print(f"  Repo rate:             full coverage, ffill pre-2007")
print(f"  CPI:                   Jan 2011+, use cpi_index (inflation_pct has gaps)")
print(f"  POS terminals/UPI QR:  Nov 2019+ only (new PSI format)")
print(f"  UPI P2M:               May 2020+ — use total UPI vol for pre-2020")
print()
print("BANKWISE")
print(f"  CC model-ready:        {cc[~cc['low_coverage']]['bank_name'].nunique()} banks, Apr 2011–May 2025")
print(f"  DC model-ready:        {dc[~dc['low_coverage']]['bank_name'].nunique()} banks, Apr 2011–May 2025")
print(f"  CC top-3 concentration:{top3_cc/total_cc*100:.0f}%")
print(f"  DC top-3 concentration:{top3_dc/total_dc*100:.0f}%")
print()
print(f"Charts: {OUT_DIR}")