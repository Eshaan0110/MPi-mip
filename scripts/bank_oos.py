"""Per-bank out-of-sample accuracy: March + April 2026."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path
from src.ingestion.bankwise import ingest, canonical_bank

months = [
    ("March 2026", "2026-03", Path(r"C:\Users\ASUS\Downloads\ATMMARCH20265F48EF7056E84759B802285873DD3FB3.XLSX")),
    ("April 2026", "2026-04", Path(r"C:\Users\ASUS\Downloads\ATMAPRIL20265AF3EE208AF746979B772EED805CDCA8.XLSX")),
]

fc_dir = Path("data/processed/bankwise_forecasts")

for card_type, card_label, col in [("cc", "CREDIT CARD", "credit_outstanding"), ("dc", "DEBIT CARD", "debit_outstanding")]:
    print("=" * 80)
    print(f"{card_label} - PER-BANK OUT-OF-SAMPLE ACCURACY")
    print("=" * 80)

    all_apes = []

    for label, month, path in months:
        df = ingest(path, verbose=False)
        df["bank"] = df["bank"].apply(canonical_bank)

        print(f"\n{label}:")
        header = f"  {'Bank':<32} {'Actual':>10} {'Forecast':>10} {'Error':>9} {'APE':>7}"
        print(header)
        print("  " + "-" * 70)

        bank_rows = df[df[col].notna() & (df[col] > 0)].sort_values(col, ascending=False)

        for _, row in bank_rows.head(20).iterrows():
            bank = row["bank"]
            actual = row[col]

            safe = bank.lower().replace(" ", "_").replace(".", "").replace("/", "_")
            fc_path = fc_dir / f"{card_type}_{safe}_forecast.csv"
            if not fc_path.exists():
                continue

            fc = pd.read_csv(str(fc_path))
            fc_row = fc[fc.date.str.startswith(month)]
            if fc_row.empty:
                continue

            pred = fc_row.forecast.iloc[0]
            err = pred - actual
            ape = abs(err / actual) * 100
            all_apes.append((bank, label, ape))

            a_str = f"{actual/1e6:.2f}M"
            p_str = f"{pred/1e6:.2f}M"
            e_str = f"{err/1e6:+.2f}M"
            print(f"  {bank:<32} {a_str:>10} {p_str:>10} {e_str:>9} {ape:>5.1f}%")

    if all_apes:
        ape_vals = [a for _, _, a in all_apes]
        ape_vals_sorted = sorted(ape_vals)
        median = ape_vals_sorted[len(ape_vals_sorted) // 2]
        mean = sum(ape_vals) / len(ape_vals)
        print(f"\n{card_label} SUMMARY ({len(all_apes)} bank-month observations):")
        print(f"  Median OOS APE: {median:.1f}%")
        print(f"  Mean OOS APE:   {mean:.1f}%")
        # Count by quality
        good = sum(1 for a in ape_vals if a < 5)
        ok = sum(1 for a in ape_vals if 5 <= a < 15)
        bad = sum(1 for a in ape_vals if a >= 15)
        print(f"  <5% error: {good}/{len(ape_vals)} observations")
        print(f"  5-15% error: {ok}/{len(ape_vals)} observations")
        print(f"  >15% error: {bad}/{len(ape_vals)} observations")
    print()
