"""Compute all Granger + stats per bank per variable, output as JSON for docx builder."""
import json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path

CC_DEFAULT_START = pd.Timestamp("2013-01-01")
DC_DEFAULT_START = pd.Timestamp("2017-01-01")
BANK_START_DATES = {
    ("HDFC Bank","cc"):pd.Timestamp("2017-01-01"),("State Bank of India","cc"):pd.Timestamp("2017-04-01"),
    ("ICICI Bank","cc"):pd.Timestamp("2017-01-01"),("Kotak Mahindra Bank","cc"):pd.Timestamp("2018-01-01"),
    ("Bank of Baroda","cc"):pd.Timestamp("2019-04-01"),("Yes Bank","cc"):pd.Timestamp("2020-06-01"),
    ("Canara Bank","cc"):pd.Timestamp("2020-04-01"),("State Bank of India","dc"):pd.Timestamp("2017-04-01"),
    ("Bank of Baroda","dc"):pd.Timestamp("2019-04-01"),("Canara Bank","dc"):pd.Timestamp("2020-04-01"),
    ("Union Bank of India","dc"):pd.Timestamp("2020-04-01"),("Punjab National Bank","dc"):pd.Timestamp("2020-04-01"),
    ("Indian Bank","dc"):pd.Timestamp("2020-04-01"),("Paytm Payments Bank","dc"):pd.Timestamp("2018-04-01"),
}
def get_start(b,ct): return BANK_START_DATES.get((b,ct), CC_DEFAULT_START if ct=="cc" else DC_DEFAULT_START)

CC_BANKS=["HDFC Bank","State Bank of India","ICICI Bank","Axis Bank","Kotak Mahindra Bank","IndusInd Bank","Bank of Baroda","Yes Bank","Canara Bank","HSBC"]
DC_BANKS=["State Bank of India","Bank of Baroda","Canara Bank","HDFC Bank","Union Bank of India","Punjab National Bank","Axis Bank","Bank of India","Kotak Mahindra Bank","Indian Bank","Central Bank of India","UCO Bank","ICICI Bank","Indian Overseas Bank","Paytm Payments Bank"]
CC_VARS={"atm_offsite":"ATMs Off-site","pos_terminals":"PoS Terminals","cc_pos_vol":"CC PoS Txn Vol","cc_atm_cash_vol":"CC ATM Cash Vol"}
DC_VARS={"atm_onsite":"ATMs On-site","pos_terminals":"PoS Terminals","dc_pos_vol":"DC PoS Txn Vol","dc_atm_cash_vol":"DC ATM Cash Vol"}

cc_bw = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc_bw = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
for df in [cc_bw,dc_bw]: df["date"]=pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()

def gp(y,x,lag):
    d=pd.DataFrame({"y":y.values,"x":x.values},index=y.index).dropna()
    if len(d)<lag*3+10: return None
    dy=d["y"].diff().dropna(); dx=d["x"].diff().dropna()
    data=pd.DataFrame({"y":dy.values[:len(dx)],"x":dx.values[:len(dy)]}).dropna()
    if len(data)<lag*3+5: return None
    try:
        res=grangercausalitytests(data[["y","x"]],maxlag=lag,verbose=False)
        return round(res[lag][0]["ssr_ftest"][1],4)
    except: return None

VERDICTS = {
    "ATMs Off-site": "Tracks bank's outreach infrastructure. Growth leads CC issuance by 3–6 months.",
    "PoS Terminals": "Merchant network expansion precedes card acquisition drives.",
    "CC PoS Txn Vol": "Rising swipe activity signals card utility — reduces churn, encourages issuance.",
    "CC ATM Cash Vol": "Cash withdrawal volume on CC reflects credit utilisation intensity.",
    "ATMs On-site": "Branch ATM growth is a proxy for branch expansion, which drives DC account opening.",
    "DC PoS Txn Vol": "Declining POS swipes (UPI displacement) is a leading indicator of DC attrition.",
    "DC ATM Cash Vol": "ATM cash usage decline rate predicts pace of DC card base erosion.",
}

REASONS = {
    "USE": "Strong Granger signal (p<0.01) with sufficient history. Include as regressor.",
    "CONSIDER": "Moderate signal (p<0.05). Test in ablation before committing to production.",
    "WEAK": "Marginal signal (p<0.10). Not recommended for production without further data.",
    "drop": "No Granger signal. High correlation is spurious trend co-movement, not causal.",
}

def analyse(bw, banks, target_col, cand_vars, card_type):
    results = []
    for bank in banks:
        start = get_start(bank, card_type)
        bdf = bw[(bw.bank_name==bank)&(bw.date>=start)].sort_values("date").reset_index(drop=True)
        if bdf.empty or bdf[target_col].isna().all(): continue
        n = len(bdf); ws = bdf.date.min(); we = bdf.date.max()
        out_latest = float(bdf[target_col].dropna().iloc[-1])
        out_first  = float(bdf[target_col].dropna().iloc[0])
        out_pct    = round((out_latest-out_first)/out_first*100,1)
        bank_entry = {"bank":bank,"card_type":card_type,"n_months":n,
                      "window_start":ws.strftime("%b %Y"),"window_end":we.strftime("%b %Y"),
                      "outstanding_latest":int(out_latest),"outstanding_growth_pct":out_pct,"variables":[]}
        for col, lbl in cand_vars.items():
            if col not in bdf.columns: continue
            s = bdf[col].copy(); n_valid = int(s.notna().sum())
            if n_valid < 6: continue
            latest_val = float(s.dropna().iloc[-1]); first_val = float(s.dropna().iloc[0])
            pct = round((latest_val-first_val)/first_val*100,1) if first_val!=0 else 0
            pair = bdf[[target_col,col]].dropna()
            rho, p_rho = (stats.spearmanr(pair[target_col],pair[col]) if len(pair)>=8 else (0,1))
            idx = bdf.set_index("date")
            g1 = gp(idx[target_col],idx[col],1)
            g3 = gp(idx[target_col],idx[col],3)
            g6 = gp(idx[target_col],idx[col],6)
            best_p = min(p for p in [g1,g3,g6] if p is not None) if any(p is not None for p in [g1,g3,g6]) else 1.0
            if best_p<0.01 and n_valid>=36:   verdict="USE"
            elif best_p<0.05 and n_valid>=24: verdict="CONSIDER"
            elif best_p<0.10 and n_valid>=24: verdict="WEAK"
            else:                              verdict="drop"
            safe = bank.lower().replace(" ","_").replace(".","")
            safe_col = col.replace("_","")
            chart = f"reports/eda_charts/{card_type}_{safe}_{safe_col}.png"
            bank_entry["variables"].append({
                "col":col,"label":lbl,"n_valid":n_valid,"n_months":n,
                "latest_val":round(latest_val),"pct_change":pct,
                "spearman_rho":round(float(rho),3),"spearman_p":round(float(p_rho),4),
                "granger_l1":g1,"granger_l3":g3,"granger_l6":g6,
                "verdict":verdict,
                "economic_rationale":VERDICTS.get(lbl,""),
                "verdict_reason":REASONS[verdict],
                "chart_path":chart if Path(chart).exists() else None
            })
        results.append(bank_entry)
    return results

data = {"cc": analyse(cc_bw, CC_BANKS, "cc_outstanding", CC_VARS, "cc"),
        "dc": analyse(dc_bw, DC_BANKS, "dc_outstanding", DC_VARS, "dc")}

Path("reports").mkdir(exist_ok=True)
Path("reports/eda_data.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
print("Saved reports/eda_data.json")
