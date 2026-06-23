"""
FIX F8+F9: Flag low-accuracy bank models in dashboard
======================================================
Kotak CC (20.2%), BoB CC (21.9%), Paytm DC (27.5%) have MAPE >15%.
These forecasts are directional only — not suitable for quantitative
decisions. The dashboard should surface this.

Two approaches:
1. Add accuracy_tier to model_metadata (green/amber/red)
2. Attempt ensemble Prophet+ETS to improve accuracy

This script implements both.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# === APPROACH 1: Accuracy tiering ===

def classify_accuracy(mape: float) -> dict:
    """Classify MAPE into tiers with display guidance."""
    if mape <= 7:
        return {"tier": "green", "label": "High confidence", "usable_for": "quantitative decisions"}
    elif mape <= 15:
        return {"tier": "amber", "label": "Moderate confidence", "usable_for": "directional guidance with CI bands"}
    else:
        return {"tier": "red", "label": "Low confidence", "usable_for": "directional only — not for quantitative use"}


# Apply to current CV results
BANK_ACCURACY_TIERS = {
    # CC
    ("ICICI Bank", "CC"):           classify_accuracy(4.54),
    ("HDFC Bank", "CC"):            classify_accuracy(5.44),
    ("SBI", "CC"):                  classify_accuracy(5.89),
    ("IndusInd Bank", "CC"):        classify_accuracy(5.90),
    ("Axis Bank", "CC"):            classify_accuracy(10.48),
    ("HSBC", "CC"):                 classify_accuracy(13.99),
    ("Kotak Mahindra Bank", "CC"):  classify_accuracy(20.24),   # RED
    ("Bank of Baroda", "CC"):       classify_accuracy(21.87),   # RED
    # DC
    ("HDFC Bank", "DC"):            classify_accuracy(3.00),
    ("SBI", "DC"):                  classify_accuracy(4.48),
    ("Bank of Baroda", "DC"):       classify_accuracy(5.74),
    ("Axis Bank", "DC"):            classify_accuracy(5.95),
    ("ICICI Bank", "DC"):           classify_accuracy(6.38),
    ("Central Bank of India", "DC"):classify_accuracy(7.37),
    ("Bank of India", "DC"):        classify_accuracy(9.30),
    ("UCO Bank", "DC"):             classify_accuracy(9.76),
    ("Kotak Mahindra Bank", "DC"):  classify_accuracy(11.39),
    ("Indian Overseas Bank", "DC"): classify_accuracy(12.25),
    ("Paytm Payments Bank", "DC"):  classify_accuracy(27.52),   # RED
}


# === APPROACH 2: Prophet+ETS ensemble for red-tier banks ===

def ensemble_forecast(
    prophet_forecast: pd.DataFrame,
    ets_forecast: pd.DataFrame,
    prophet_mape: float,
    ets_mape: float,
) -> pd.DataFrame:
    """Inverse-MAPE weighted ensemble of Prophet and ETS forecasts.

    Weight = 1/MAPE normalised. Lower MAPE → higher weight.
    """
    w_prophet = (1 / prophet_mape) / (1 / prophet_mape + 1 / ets_mape)
    w_ets = 1 - w_prophet

    result = prophet_forecast.copy()
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        if col in prophet_forecast.columns and col in ets_forecast.columns:
            result[col] = w_prophet * prophet_forecast[col] + w_ets * ets_forecast[col]

    return result


if __name__ == "__main__":
    print("Bank Accuracy Tiers:")
    print(f"{'Bank':<30} {'Type':>4} {'MAPE':>8} {'Tier':>6} {'Guidance'}")
    print("-" * 85)
    for (bank, ct), info in sorted(BANK_ACCURACY_TIERS.items(), key=lambda x: x[1]["tier"]):
        mape_val = {
            ("ICICI Bank", "CC"): 4.54, ("HDFC Bank", "CC"): 5.44,
            ("Kotak Mahindra Bank", "CC"): 20.24, ("Bank of Baroda", "CC"): 21.87,
            ("Paytm Payments Bank", "DC"): 27.52,
        }.get((bank, ct), 0)
        print(f"  {bank:<28} {ct:>4} {info['tier']:>6}   {info['usable_for']}")

    # Count by tier
    tiers = [v["tier"] for v in BANK_ACCURACY_TIERS.values()]
    print(f"\nSummary: {tiers.count('green')} green, {tiers.count('amber')} amber, {tiers.count('red')} red")
    print("\nRed-tier banks should either:")
    print("  1. Use Prophet+ETS ensemble (this script)")
    print("  2. Be moved to residual bucket (Paytm)")
    print("  3. Be flagged with disclaimer in dashboard")
