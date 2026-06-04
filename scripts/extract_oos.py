"""Out-of-sample accuracy test: March + April 2026 vs forecasts."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path
from src.ingestion.bankwise import ingest

# Extract actuals from bankwise files
actuals = {}
files = [
    ("March 2026", "2026-03", Path(r"C:\Users\ASUS\Downloads\ATMMARCH20265F48EF7056E84759B802285873DD3FB3.XLSX")),
    ("April 2026", "2026-04", Path(r"C:\Users\ASUS\Downloads\ATMAPRIL20265AF3EE208AF746979B772EED805CDCA8.XLSX")),
]

for label, month, path in files:
    df = ingest(path, verbose=False)
    cc = df["credit_outstanding"].sum() / 1e5
    dc = df["debit_outstanding"].sum() / 1e5
    actuals[label] = {"month": month, "cc": cc, "dc": dc}
    print(f"{label}: CC={cc:.2f} lakh, DC={dc:.2f} lakh ({df.bank.nunique()} banks)")

# Load forecasts
cc_fc = pd.read_csv("data/processed/forecast_cc.csv")
dc_fc = pd.read_csv("data/processed/forecast_dc.csv")
gu_cc = pd.read_csv("data/processed/groundup/groundup_cc.csv")
gu_dc = pd.read_csv("data/processed/groundup/groundup_dc.csv")

print()
print("=" * 75)
print("OUT-OF-SAMPLE ACCURACY TEST (Mar-Apr 2026)")
print("Model trained through Feb 2026. These months were NEVER seen.")
print("=" * 75)

all_cc_ape = []
all_dc_ape = []

for label, vals in actuals.items():
    m = vals["month"]
    print(f"\n{label}:")
    print(f"  {'Model':<26} {'Actual':>10} {'Forecast':>10} {'Error':>9} {'APE':>7}  CI?")
    print("  " + "-" * 72)

    # CC aggregate
    cc_row = cc_fc[cc_fc.date.str.startswith(m)].iloc[0]
    cc_err = cc_row.forecast_lakh - vals["cc"]
    cc_ape = abs(cc_err / vals["cc"]) * 100
    cc_in = "YES" if cc_row.forecast_lower_lakh <= vals["cc"] <= cc_row.forecast_upper_lakh else "NO"
    all_cc_ape.append(cc_ape)
    print(f"  CC Outstanding (agg)     {vals['cc']:>9.1f}  {cc_row.forecast_lakh:>9.1f}  {cc_err:>+8.1f}  {cc_ape:>5.2f}%  {cc_in}")

    # DC aggregate
    dc_row = dc_fc[dc_fc.date.str.startswith(m)].iloc[0]
    dc_err = dc_row.forecast_lakh - vals["dc"]
    dc_ape = abs(dc_err / vals["dc"]) * 100
    dc_in = "YES" if dc_row.forecast_lower_lakh <= vals["dc"] <= dc_row.forecast_upper_lakh else "NO"
    all_dc_ape.append(dc_ape)
    print(f"  DC Outstanding (agg)     {vals['dc']:>9.1f}  {dc_row.forecast_lakh:>9.1f}  {dc_err:>+8.1f}  {dc_ape:>5.2f}%  {dc_in}")

    # CC ground-up
    gu_row = gu_cc[gu_cc.date.str.startswith(m)]
    if not gu_row.empty:
        pred = gu_row.forecast.iloc[0] / 1e5
        err = pred - vals["cc"]
        ape = abs(err / vals["cc"]) * 100
        print(f"  CC Ground-up (20 banks)  {vals['cc']:>9.1f}  {pred:>9.1f}  {err:>+8.1f}  {ape:>5.2f}%")

    # DC ground-up
    gu_row = gu_dc[gu_dc.date.str.startswith(m)]
    if not gu_row.empty:
        pred = gu_row.forecast.iloc[0] / 1e5
        err = pred - vals["dc"]
        ape = abs(err / vals["dc"]) * 100
        print(f"  DC Ground-up (20 banks)  {vals['dc']:>9.1f}  {pred:>9.1f}  {err:>+8.1f}  {ape:>5.2f}%")

print()
print("=" * 75)
print("SUMMARY")
print("=" * 75)
avg_cc = sum(all_cc_ape) / len(all_cc_ape)
avg_dc = sum(all_dc_ape) / len(all_dc_ape)
print(f"  CC Outstanding avg OOS error: {avg_cc:.2f}%  (CV MAPE was 3.46%)")
print(f"  DC Outstanding avg OOS error: {avg_dc:.2f}%  (CV MAPE was 7.08%)")
print()

for name, oos, cv in [("CC", avg_cc, 3.46), ("DC", avg_dc, 7.08)]:
    if oos <= cv:
        verdict = "OOS BETTER than CV -- model generalises excellently"
    elif oos <= cv * 1.5:
        verdict = "OOS within 1.5x CV -- model generalises well"
    else:
        verdict = "OOS > 1.5x CV -- investigate"
    print(f"  {name}: {verdict}")
