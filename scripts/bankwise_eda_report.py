"""
Per-Bank Regressor EDA Report
=================================
Generates a self-contained HTML report with:
  - Time series plots (variable + outstanding on same chart)
  - Scatter plots (variable vs outstanding)
  - Distribution + trend stats
  - Granger causality results with written interpretation
  - Final verdict with documented reasoning per bank per variable

Output: reports/bankwise_eda_report.html
"""
import sys, warnings, base64, io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from pathlib import Path

# ── Config — mirrors bank_config.py exactly ───────────────────────────────
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

def get_start(bank, card_type):
    return BANK_START_DATES.get(
        (bank, card_type),
        CC_DEFAULT_START if card_type == "cc" else DC_DEFAULT_START,
    )

CC_BANKS = [
    "HDFC Bank","State Bank of India","ICICI Bank","Axis Bank",
    "Kotak Mahindra Bank","IndusInd Bank","Bank of Baroda",
    "Yes Bank","Canara Bank","HSBC",
]
DC_BANKS = [
    "State Bank of India","Bank of Baroda","Canara Bank","HDFC Bank",
    "Union Bank of India","Punjab National Bank","Axis Bank",
    "Bank of India","Kotak Mahindra Bank","Indian Bank",
    "Central Bank of India","UCO Bank","ICICI Bank",
    "Indian Overseas Bank","Paytm Payments Bank",
]

CC_VARS = {
    "atm_offsite":     "ATMs Off-site",
    "pos_terminals":   "PoS Terminals",
    "cc_pos_vol":      "CC PoS Txn Volume",
    "cc_atm_cash_vol": "CC ATM Cash Volume",
}
DC_VARS = {
    "atm_onsite":      "ATMs On-site",
    "pos_terminals":   "PoS Terminals",
    "dc_pos_vol":      "DC PoS Txn Volume",
    "dc_atm_cash_vol": "DC ATM Cash Volume",
}

# ── Load ──────────────────────────────────────────────────────────────────
cc_bw = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc_bw = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
for df in [cc_bw, dc_bw]:
    df["date"] = pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()


# ── Analytics helpers ─────────────────────────────────────────────────────
def granger_results(y, x, lags=(1, 3, 6)):
    out = {}
    df = pd.DataFrame({"y": y.values, "x": x.values}, index=y.index).dropna()
    if len(df) < 20:
        return {lag: None for lag in lags}
    dy = df["y"].diff().dropna()
    dx = df["x"].diff().dropna()
    data = pd.DataFrame({"y": dy.values[:len(dx)], "x": dx.values[:len(dy)]}).dropna()
    for lag in lags:
        if lag >= len(data) // 3:
            out[lag] = None
            continue
        try:
            res = grangercausalitytests(data[["y","x"]], maxlag=lag, verbose=False)
            out[lag] = res[lag][0]["ssr_ftest"][1]
        except Exception:
            out[lag] = None
    return out

def adf_pval(series):
    try:
        return adfuller(series.dropna())[1]
    except Exception:
        return None

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# ── Plot builder ──────────────────────────────────────────────────────────
def make_bank_var_plot(bdf, target_col, col, var_label, bank, card_type):
    """4-panel figure: time series, MoM change, scatter, autocorr of residuals."""
    series = bdf[[target_col, col]].dropna()
    if len(series) < 6:
        return None

    y = series[target_col]
    x = series[col]
    dates = series.index if isinstance(series.index, pd.DatetimeIndex) else bdf.loc[series.index, "date"] if "date" in bdf.columns else series.index

    # rebuild with date index
    bdf2 = bdf.set_index("date")[[target_col, col]].dropna()
    y = bdf2[target_col]
    x = bdf2[col]

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(f"{bank}  |  {var_label}  →  {target_col.replace('_',' ').title()}", fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Panel 1: Dual-axis time series
    ax1 = fig.add_subplot(gs[0, 0])
    color1, color2 = "#1f77b4", "#d62728"
    ax1.set_xlabel("Date"); ax1.set_ylabel(var_label, color=color1)
    ax1.plot(x.index, x.values, color=color1, lw=1.5, label=var_label)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Cards Outstanding", color=color2)
    ax2.plot(y.index, y.values, color=color2, lw=1.5, linestyle="--", label="Outstanding")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax1.set_title("Time Series (dual axis)", fontsize=10)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, fontsize=7, loc="upper left")

    # Panel 2: Year-over-year % change of variable
    ax3 = fig.add_subplot(gs[0, 1])
    yoy = x.pct_change(12) * 100
    colors_bar = ["#2ca02c" if v >= 0 else "#d62728" for v in yoy.fillna(0)]
    ax3.bar(yoy.index, yoy.values, color=colors_bar, width=25, alpha=0.7)
    ax3.axhline(0, color="black", lw=0.8)
    ax3.set_title(f"{var_label} — YoY % Change", fontsize=10)
    ax3.set_ylabel("YoY %")
    ax3.set_xlabel("Date")

    # Panel 3: Scatter with regression line
    ax4 = fig.add_subplot(gs[1, 0])
    rho, p_rho = stats.spearmanr(x.values, y.values)
    ax4.scatter(x.values, y.values, alpha=0.5, s=20, color="#1f77b4")
    # OLS trend line
    try:
        m, b = np.polyfit(x.values, y.values, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax4.plot(x_line, m*x_line+b, color="#d62728", lw=1.5)
    except Exception:
        pass
    ax4.set_xlabel(var_label, fontsize=9)
    ax4.set_ylabel("Cards Outstanding", fontsize=9)
    ax4.set_title(f"Scatter  |  Spearman ρ = {rho:+.3f}  (p={p_rho:.3f})", fontsize=10)

    # Panel 4: Distribution of MoM changes in variable
    ax5 = fig.add_subplot(gs[1, 1])
    mom = x.diff().dropna()
    ax5.hist(mom.values, bins=20, color="#1f77b4", alpha=0.7, edgecolor="white")
    ax5.axvline(mom.mean(), color="#d62728", lw=1.5, linestyle="--", label=f"Mean={mom.mean():.0f}")
    ax5.axvline(0, color="black", lw=0.8)
    ax5.set_title(f"{var_label} — MoM Change Distribution", fontsize=10)
    ax5.set_xlabel("Month-on-month change")
    ax5.set_ylabel("Frequency")
    ax5.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig_to_base64(fig), plt.close(fig) or True


# ── HTML builder ──────────────────────────────────────────────────────────
VERDICT_STYLE = {
    "USE":      "background:#1a7a1a;color:white;padding:3px 8px;border-radius:4px;font-weight:bold",
    "CONSIDER": "background:#c47a00;color:white;padding:3px 8px;border-radius:4px;font-weight:bold",
    "WEAK":     "background:#888;color:white;padding:3px 8px;border-radius:4px",
    "drop":     "background:#bbb;color:#444;padding:3px 8px;border-radius:4px",
}

def sig_badge(p):
    if p is None: return "<span style='color:#999'>N/A</span>"
    if p < 0.001: return f"<b style='color:#1a7a1a'>{p:.4f} ***</b>"
    if p < 0.01:  return f"<b style='color:#2a8a2a'>{p:.4f} **</b>"
    if p < 0.05:  return f"<b style='color:#c47a00'>{p:.4f} *</b>"
    return f"<span style='color:#c00'>{p:.4f}</span>"

def granger_verdict_text(gp, n_months, var_label, target_label):
    best_p = min((p for p in gp.values() if p is not None), default=1.0)
    best_lag = min(gp, key=lambda k: gp[k] if gp[k] is not None else 1.0)

    if best_p < 0.01 and n_months >= 36:
        return (
            f"<b style='color:#1a7a1a'>STRONG GRANGER SIGNAL.</b> "
            f"{var_label} Granger-causes {target_label} at lag {best_lag} (p={best_p:.4f}). "
            f"Past values of {var_label} provide statistically significant predictive information "
            f"about future card issuance beyond what the outstanding series itself can predict. "
            f"<b>Recommended: include as regressor.</b>"
        )
    elif best_p < 0.05 and n_months >= 24:
        return (
            f"<b style='color:#c47a00'>MODERATE GRANGER SIGNAL.</b> "
            f"{var_label} shows marginal Granger causality at lag {best_lag} (p={best_p:.4f}). "
            f"Signal exists but is not robust — consider including with caution and monitor "
            f"whether it helps or hurts OOS performance. "
            f"<b>Recommended: test in ablation before committing.</b>"
        )
    elif best_p < 0.10:
        return (
            f"<b style='color:#888'>WEAK SIGNAL (p={best_p:.4f}).</b> "
            f"Marginally significant at lag {best_lag} but below the 5% threshold. "
            f"Not recommended as a production regressor without further validation."
        )
    else:
        return (
            f"<b style='color:#c00'>NO GRANGER SIGNAL (best p={best_p:.4f}).</b> "
            f"{var_label} does not Granger-cause {target_label} in this bank's stable window. "
            f"High Spearman correlation (if present) is likely spurious trend co-movement — "
            f"both series growing together over time does not imply causality. "
            f"<b>Recommendation: drop this variable.</b>"
        )

def build_section(bw, banks, target_col, cand_vars, card_type, title):
    html = f"<h1 style='color:#1a3a6a;border-bottom:3px solid #1a3a6a;padding-bottom:8px'>{title}</h1>"

    for bank in banks:
        start = get_start(bank, card_type)
        bdf = bw[(bw["bank_name"] == bank) & (bw["date"] >= start)].sort_values("date").reset_index(drop=True)

        if bdf.empty or bdf[target_col].isna().all():
            html += f"<h2>{bank}</h2><p style='color:gray'>No data in stable window.</p>"
            continue

        n_months = len(bdf)
        window_start = bdf["date"].min()
        window_end   = bdf["date"].max()
        latest_out   = bdf[target_col].dropna().iloc[-1]
        first_out    = bdf[target_col].dropna().iloc[0]
        out_pct      = (latest_out - first_out) / first_out * 100

        html += f"""
        <h2 style='color:#1a3a6a;margin-top:40px'>{bank}</h2>
        <div style='background:#f0f4ff;border-left:5px solid #1a3a6a;padding:12px;margin-bottom:16px;border-radius:4px'>
          <b>Stable regime:</b> {window_start:%b %Y} → {window_end:%b %Y} &nbsp;|&nbsp;
          <b>Months:</b> {n_months} &nbsp;|&nbsp;
          <b>Latest outstanding:</b> {latest_out:,.0f} cards &nbsp;|&nbsp;
          <b>Growth over window:</b> {out_pct:+.1f}%
          <br><small style='color:#666'>Start date from BANK_START_DATES config — stable post-merger/reconstruction regime only.</small>
        </div>
        """

        for col, var_label in cand_vars.items():
            if col not in bdf.columns:
                continue
            series = bdf[col].copy()
            n_valid = series.notna().sum()
            n_null  = series.isna().sum()

            if n_valid < 6:
                html += f"<h3>{var_label}</h3><p style='color:gray'>Insufficient data ({n_valid} observations).</p>"
                continue

            latest_val = series.dropna().iloc[-1]
            first_val  = series.dropna().iloc[0]
            mean_val   = series.mean()
            std_val    = series.std()
            pct_change = ((latest_val - first_val) / first_val * 100) if first_val != 0 else 0
            mom = series.diff().dropna()
            mom_mean = mom.mean()
            mom_std  = mom.std()

            # Stationarity
            adf_p = adf_pval(series.dropna())
            stationarity = "Stationary (ADF p<0.05)" if (adf_p is not None and adf_p < 0.05) else f"Non-stationary (ADF p={adf_p:.3f})" if adf_p is not None else "N/A"

            # Spearman
            pair = bdf[[target_col, col]].dropna()
            if len(pair) >= 8:
                rho, p_rho = stats.spearmanr(pair[target_col], pair[col])
                rho_str = f"{rho:+.3f} (p={p_rho:.4f})"
            else:
                rho, p_rho, rho_str = 0, 1, "N/A"

            # Granger
            gp = granger_results(bdf.set_index("date")[target_col], bdf.set_index("date")[col])
            best_p = min((p for p in gp.values() if p is not None), default=1.0)

            # Verdict
            if best_p < 0.01 and n_valid >= 36:   verdict = "USE"
            elif best_p < 0.05 and n_valid >= 24:  verdict = "CONSIDER"
            elif best_p < 0.10 and n_valid >= 24:  verdict = "WEAK"
            else:                                   verdict = "drop"

            # Plot
            plot_result = make_bank_var_plot(bdf, target_col, col, var_label, bank, card_type)
            img_tag = ""
            if plot_result:
                b64, _ = plot_result
                img_tag = f"<img src='data:image/png;base64,{b64}' style='width:100%;max-width:900px;margin:10px 0'>"

            granger_txt = granger_verdict_text(gp, n_valid, var_label, target_col.replace("_"," ").title())

            html += f"""
            <div style='border:1px solid #ddd;border-radius:6px;padding:16px;margin:16px 0;background:#fafafa'>
              <h3 style='margin-top:0'>{var_label}
                &nbsp;<span style='{VERDICT_STYLE[verdict]}'>{verdict}</span>
              </h3>

              <table style='border-collapse:collapse;font-size:13px;margin-bottom:12px'>
                <tr style='background:#e8edf5'>
                  <th style='padding:6px 12px;text-align:left'>Stat</th>
                  <th style='padding:6px 12px;text-align:left'>Value</th>
                  <th style='padding:6px 12px;text-align:left'>Stat</th>
                  <th style='padding:6px 12px;text-align:left'>Value</th>
                </tr>
                <tr><td style='padding:4px 12px'>Coverage</td><td><b>{n_valid}/{n_months}</b> ({n_null} nulls)</td>
                    <td style='padding:4px 12px'>Latest value</td><td><b>{latest_val:,.0f}</b></td></tr>
                <tr style='background:#f5f5f5'><td style='padding:4px 12px'>Mean</td><td>{mean_val:,.0f}</td>
                    <td style='padding:4px 12px'>Std dev</td><td>{std_val:,.0f}</td></tr>
                <tr><td style='padding:4px 12px'>Trend over window</td><td><b>{pct_change:+.1f}%</b></td>
                    <td style='padding:4px 12px'>Avg MoM change</td><td>{mom_mean:+,.0f} ± {mom_std:,.0f}</td></tr>
                <tr style='background:#f5f5f5'><td style='padding:4px 12px'>Spearman ρ vs outstanding</td><td><b>{rho_str}</b></td>
                    <td style='padding:4px 12px'>Stationarity</td><td>{stationarity}</td></tr>
              </table>

              <b>Granger Causality: {var_label} → {target_col.replace("_"," ").title()}</b>
              <table style='border-collapse:collapse;font-size:13px;margin:8px 0 12px 0'>
                <tr style='background:#e8edf5'>
                  <th style='padding:5px 14px'>Lag 1 month</th>
                  <th style='padding:5px 14px'>Lag 3 months</th>
                  <th style='padding:5px 14px'>Lag 6 months</th>
                  <th style='padding:5px 14px'>Interpretation</th>
                </tr>
                <tr>
                  <td style='padding:5px 14px;text-align:center'>{sig_badge(gp.get(1))}</td>
                  <td style='padding:5px 14px;text-align:center'>{sig_badge(gp.get(3))}</td>
                  <td style='padding:5px 14px;text-align:center'>{sig_badge(gp.get(6))}</td>
                  <td style='padding:5px 14px;font-size:12px'>*** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05</td>
                </tr>
              </table>

              <div style='background:#fff8e8;border-left:4px solid #c47a00;padding:10px 14px;border-radius:4px;font-size:13px;margin-bottom:12px'>
                {granger_txt}
              </div>

              {img_tag}
            </div>
            """

    return html


# ── Main ──────────────────────────────────────────────────────────────────
print("Building CC section...")
cc_html = build_section(cc_bw, CC_BANKS, "cc_outstanding", CC_VARS, "cc",
                        "Credit Cards — Per-Bank Regressor EDA")

print("Building DC section...")
dc_html = build_section(dc_bw, DC_BANKS, "dc_outstanding", DC_VARS, "dc",
                        "Debit Cards — Per-Bank Regressor EDA")

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>MIP Bankwise Regressor EDA</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            max-width: 1100px; margin: 0 auto; padding: 30px 20px; color: #222; }}
    h1 {{ font-size: 1.6em; }} h2 {{ font-size: 1.3em; }} h3 {{ font-size: 1.1em; }}
    table {{ width: 100%; margin-bottom: 8px; }}
    td, th {{ border: 1px solid #ddd; }}
    .toc {{ background:#f0f4ff; padding:16px; border-radius:6px; margin-bottom:30px; }}
    .toc a {{ display:block; margin:2px 0; color:#1a3a6a; text-decoration:none; }}
    .toc a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <h1 style="color:#1a3a6a">MIP — Bankwise Per-Bank Regressor EDA</h1>
  <p style="color:#555">
    All analysis uses the <b>stable regime window only</b> (from <code>BANK_START_DATES</code> config).<br>
    Variables tested: ATMs Off-site / On-site, PoS Terminals, CC/DC PoS Transaction Volume, CC/DC ATM Cash Volume.<br>
    Granger causality tested on first-differenced series at lags 1, 3, and 6 months.
  </p>
  <div class="toc">
    <b>Table of Contents</b><br>
    {"".join(f'<a href="#cc_{b.lower().replace(" ","_")}">CC — {b}</a>' for b in CC_BANKS)}
    {"".join(f'<a href="#dc_{b.lower().replace(" ","_")}">DC — {b}</a>' for b in DC_BANKS)}
  </div>

  {cc_html}
  <hr style="margin:40px 0">
  {dc_html}

  <p style="color:#aaa;font-size:11px;margin-top:40px">
    Generated by scripts/bankwise_eda_report.py — MIP Phase 1
  </p>
</body>
</html>"""

out_path = Path("reports/bankwise_eda_report.html")
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(full_html, encoding="utf-8")
print(f"\nDone. Report saved to: {out_path.resolve()}")
