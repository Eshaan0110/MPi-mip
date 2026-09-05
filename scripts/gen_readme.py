"""
MIP — README Generator (P4.4)
================================
Auto-generates the Models section of README.md from pipeline outputs
so accuracy numbers, ensemble weights, and training details stay in
sync with the code. Kills D4 (docs drift) permanently.

Usage:
    uv run python scripts/gen_readme.py          # update README.md in place
    uv run python scripts/gen_readme.py --dry-run # preview without writing
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.toml"
README_PATH = PROJECT_ROOT / "README.md"

DRY_RUN = "--dry-run" in sys.argv

START_MARKER = "<!-- AUTO-GENERATED:MODELS:START -->"
END_MARKER = "<!-- AUTO-GENERATED:MODELS:END -->"


def _load_config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def _load_cv_metrics() -> dict[str, float]:
    """Load CV MAPE from pipeline output files."""
    metrics = {}
    for stem, label in [
        ("forecast_cc_cv_metrics", "cc_outstanding"),
        ("forecast_dc_cv_metrics", "dc_outstanding"),
    ]:
        path = PROCESSED / f"{stem}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "mape" in df.columns:
                metrics[label] = round(df["mape"].mean() * 100, 2)
    return metrics


def _load_ensemble_weights() -> dict[str, dict[str, float]]:
    """Load ensemble weights from cached JSON or weight CSV files."""
    weights = {}
    for key in ["cc", "dc"]:
        weight_path = PROCESSED / f"ensemble_weights_{key}.json"
        if weight_path.exists():
            with open(weight_path) as f:
                weights[key] = json.load(f)
            continue

        csv_path = PROCESSED / f"forecast_{key}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            w = {}
            for col in df.columns:
                if col.startswith("forecast_") and col.endswith("_lakh") and col != "forecast_lakh":
                    member = col.replace("forecast_", "").replace("_lakh", "")
                    w[member] = None
            if w:
                weights[key] = w

    return weights


def _load_bank_cv() -> dict[str, str]:
    """Load median bank-level MAPE."""
    result = {}
    for ct in ["cc", "dc"]:
        path = PROCESSED / "groundup" / f"bank_cv_summary_{ct}.csv"
        if path.exists():
            df = pd.read_csv(path)
            if "mape_mean" in df.columns:
                lo = df["mape_mean"].quantile(0.25)
                hi = df["mape_mean"].quantile(0.75)
                result[ct] = f"~{lo:.0f}–{hi:.0f}%"
    return result


def _format_weight(w: float | None) -> str:
    if w is None:
        return "—"
    return f"{w * 100:.0f}%" if w <= 1.0 else f"{w:.0f}%"


def generate_models_section() -> str:
    """Generate the Models section from pipeline outputs."""
    cv = _load_cv_metrics()
    weights = _load_ensemble_weights()
    bank_cv = _load_bank_cv()
    config = _load_config()

    events = config.get("structural_events", [])
    event_lines = []
    for e in events:
        event_lines.append(f"- {e['name']} ({e['date']}) — {e.get('direction', '')}")

    cc_w = weights.get("cc", {})
    dc_w = weights.get("dc", {})

    members = ["prophet", "arima", "arimax", "ets", "direct"]
    member_labels = {"prophet": "Prophet", "arima": "ARIMA", "arimax": "ARIMAX", "ets": "ETS", "direct": "Direct"}

    cc_row = " | ".join([_format_weight(cc_w.get(m)) for m in members])
    dc_row = " | ".join([_format_weight(dc_w.get(m)) for m in members])
    header = " | ".join([member_labels.get(m, m) for m in members])

    cc_mape = f"~{cv['cc_outstanding']:.1f}%" if "cc_outstanding" in cv else "—"
    dc_mape = f"~{cv['dc_outstanding']:.1f}%" if "dc_outstanding" in cv else "—"

    cc_bank_range = bank_cv.get("cc", "—")
    dc_bank_range = bank_cv.get("dc", "—")

    section = f"""### Aggregate — India Total (CC & DC Outstanding)

**Architecture:** Weighted ensemble of 5 models (Prophet + ARIMA + ARIMAX + ETS + Direct multi-horizon)

| Series | {header} | CV MAPE |
|--------|{"---|".join(["---"] * len(members))}---|---------|
| CC Outstanding | {cc_row} | {cc_mape} |
| DC Outstanding | {dc_row} | {dc_mape} |

**Confidence intervals:** Conformal prediction intervals from walk-forward CV residual quantiles (5th/95th percentile). Distribution-free — no normality assumption.

### Bank-Level (~80 models)

Each bank x card type gets its own model:
- **Large/complex banks** → Prophet with logistic growth caps
- **Small/stable banks** → Holt-Winters ETS

Bank forecasts are summed and reconciled against the aggregate total (residual adjustment).

**Median CV MAPE:** CC banks {cc_bank_range}, DC banks {dc_bank_range}

### Cross-Validation

Walk-forward CV: 48-month initial window, 12-month horizon, 6-month step. Model is always tested on data it has never seen.

**Structural events coded:**
{chr(10).join(event_lines)}"""

    return section


def update_readme() -> bool:
    """Update README.md between markers, or report if markers are missing."""
    readme = README_PATH.read_text(encoding="utf-8")
    section = generate_models_section()

    if START_MARKER in readme and END_MARKER in readme:
        before = readme[: readme.index(START_MARKER) + len(START_MARKER)]
        after = readme[readme.index(END_MARKER) :]
        new_readme = f"{before}\n\n{section}\n\n{after}"
    else:
        print(f"Markers not found in README.md.")
        print(f"Add these markers around the Models section:")
        print(f"  {START_MARKER}")
        print(f"  {END_MARKER}")
        print(f"\nGenerated section:\n")
        print(section)
        return False

    if DRY_RUN:
        print("[DRY RUN] Would update README.md with:")
        print(section)
        return True

    README_PATH.write_text(new_readme, encoding="utf-8")
    print("README.md updated with latest model data.")
    return True


if __name__ == "__main__":
    update_readme()
