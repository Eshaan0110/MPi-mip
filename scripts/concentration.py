"""Show market concentration: what % do top N banks represent."""
import pandas as pd

cc = pd.read_parquet("data/processed/bankwise_cards_cc.parquet")
dc = pd.read_parquet("data/processed/bankwise_cards_dc.parquet")
psi = pd.read_parquet("data/processed/rbi_psi_cards.parquet")

latest = cc.date.max()
psi_row = psi[psi.date == latest]
psi_cc = psi_row.credit_cards_outstanding_lakh.iloc[0] * 1e5
psi_dc = psi_row.debit_cards_outstanding_lakh.iloc[0] * 1e5

for card, df_bw, psi_total, col in [
    ("CREDIT CARD", cc, psi_cc, "cc_outstanding"),
    ("DEBIT CARD", dc, psi_dc, "dc_outstanding"),
]:
    bw = df_bw[df_bw.date == latest].sort_values(col, ascending=False)

    print("=" * 70)
    print(f"{card} MARKET CONCENTRATION (as of {latest.strftime('%b %Y')})")
    print("=" * 70)

    cumsum = 0
    print(f"  Rank  Bank                             Cards       Share   Cumul")
    print(f"  ----  ----                             -----       -----   -----")
    for i, (_, r) in enumerate(bw.head(20).iterrows(), 1):
        cards = r[col]
        share = cards / psi_total * 100
        cumsum += share
        marker = " <<<" if i in [5, 10] else ""
        print(f"  {i:>4}  {r.bank_name:<30}  {cards/1e6:>7.1f}M   {share:>5.1f}%  {cumsum:>5.1f}%{marker}")

    print(f"\n  PSI India Total: {psi_total/1e6:.1f}M cards")
    for n in [3, 5, 7, 10, 15, 20]:
        top_n_sum = bw.head(n)[col].sum()
        pct = top_n_sum / psi_total * 100
        print(f"  Top {n:>2} banks = {pct:>5.1f}% of India total")
    print()
