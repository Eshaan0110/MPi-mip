"""
FIX F6: Dynamic logistic growth caps
=====================================
Current caps are hardcoded (e.g., Kotak CC = 6.5M). As actuals approach
the cap, forecasts flatten prematurely. If growth re-accelerates, the
cap biases forecasts downward.

Fix: Compute caps dynamically from recent data:
  cap = max(
      last_actual * 1.3,                          # minimum headroom
      last_actual + trailing_12m_growth * 24,      # 24-month projection
  )
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_dynamic_cap(
    series: pd.Series,
    min_headroom: float = 1.3,
    horizon_months: int = 24,
) -> float:
    """Compute a dynamic logistic cap from recent data.

    Args:
        series: Monthly outstanding values (raw, not log-transformed).
        min_headroom: Minimum cap as multiple of last value (default 1.3x).
        horizon_months: How far ahead to project for cap calculation.

    Returns:
        Cap value.
    """
    last_val = series.iloc[-1]

    # Trailing 12-month average monthly growth
    if len(series) >= 13:
        growth_12m = series.diff().iloc[-12:].mean()
    else:
        growth_12m = series.diff().mean()

    # Two cap candidates
    headroom_cap = last_val * min_headroom
    projection_cap = last_val + growth_12m * horizon_months

    cap = max(headroom_cap, projection_cap)

    # Safety floor: cap should never be below current value
    cap = max(cap, last_val * 1.1)

    return cap


if __name__ == "__main__":
    # Compare static vs dynamic caps for Kotak CC
    from src.modelling.bank_data_prep import _load_bankwise
    from src.modelling.bank_config import BANK_GROWTH_CAPS

    for card_type in ["cc", "dc"]:
        df = _load_bankwise(card_type)
        target = f"{card_type}_outstanding"

        print(f"\n{'='*60}")
        print(f"Dynamic Cap Analysis — {card_type.upper()}")
        print(f"{'='*60}")

        for (bank, ct), static_cap in sorted(BANK_GROWTH_CAPS.items()):
            if ct != card_type:
                continue
            bank_data = df[df["bank_name"] == bank].sort_values("date")
            if bank_data.empty:
                continue

            series = bank_data[target].dropna()
            dynamic_cap = compute_dynamic_cap(series)
            last_val = series.iloc[-1]

            print(f"\n  {bank}:")
            print(f"    Last actual:  {last_val:>12,.0f}")
            print(f"    Static cap:   {static_cap:>12,.0f} ({static_cap/last_val:.2f}x)")
            print(f"    Dynamic cap:  {dynamic_cap:>12,.0f} ({dynamic_cap/last_val:.2f}x)")
            print(f"    Difference:   {(dynamic_cap - static_cap):>+12,.0f}")
