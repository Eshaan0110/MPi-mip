"""Rebuild the dashboard data JSON and re-embed it into dashboard.html.

Run after any model changes:
    uv run python scripts/rebuild_dashboard.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
DASHBOARD = PROJECT_ROOT / "dashboard.html"


def _nan_to_none(d: dict) -> dict:
    """Replace float NaN with None in dict-of-lists (JSON-safe)."""
    import math
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = [None if isinstance(x, float) and math.isnan(x) else x for x in v]
        else:
            out[k] = v
    return out


def _load_full(stem: str) -> dict:
    df = pd.read_parquet(PROCESSED / f"{stem}_full.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return _nan_to_none(df.to_dict(orient="list"))


def _load_bank(card_type: str, bank_name: str) -> dict:
    safe = bank_name.lower().replace(" ", "_").replace(".", "")
    path = PROCESSED / "bankwise_forecasts" / f"{card_type}_{safe}_full.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return _nan_to_none(df.to_dict(orient="list"))


def build_data() -> dict:
    data = {
        "cc": _load_full("forecast_cc"),
        "dc": _load_full("forecast_dc"),
        "cc_vol": _load_full("forecast_cc_vol"),
        "dc_vol": _load_full("forecast_dc_vol"),
        "upi_vol": _load_full("forecast_upi_vol"),
        "forecast_start": "2026-03-01",
        "events": {
            "Demonetisation": "2016-11-01",
            "PSI redef.": "2019-11-01",
            "COVID shock": "2020-04-01",
            "UPI inflection": "2022-01-01",
            "RBI tightening": "2023-11-01",
        },
        "top5_cc_banks": {},
        "top5_dc_banks": {},
        "all_cc_banks": {},
        "all_dc_banks": {},
        # CV MAPE on original scale (corrected Jun 2026 — previously log-scale, now proper)
        "cv_mape": {
            "cc_outstanding": 3.46,
            "dc_outstanding": 7.08,
            "cc_vol": 13.63,
            "dc_vol": "N/A",
            "upi_vol": 12.31,
            "cc_bank_median": 7.93,
            "dc_bank_median": 7.44,
        },
    }

    for b in ["HDFC Bank", "State Bank of India", "ICICI Bank", "Axis Bank", "Kotak Mahindra Bank"]:
        data["top5_cc_banks"][b] = _load_bank("cc", b)

    for b in ["State Bank of India", "Bank of Baroda", "Canara Bank", "HDFC Bank", "Union Bank of India"]:
        data["top5_dc_banks"][b] = _load_bank("dc", b)

    # Load ALL bank forecasts for the interactive selector
    import glob
    bank_dir = PROCESSED / "bankwise_forecasts"

    for card_type, key in [("cc", "all_cc_banks"), ("dc", "all_dc_banks")]:
        seen = set()
        for f in sorted(bank_dir.glob(f"{card_type}_*_full.parquet")):
            # Extract bank name from filename
            stem = f.stem  # e.g. "cc_hdfc_bank_full"
            bank_part = stem[len(card_type) + 1 : -len("_full")]  # "hdfc_bank"
            # Read the parquet to get the actual bank_name column
            try:
                df = pd.read_parquet(f)
                if "bank_name" in df.columns:
                    bank_name = df["bank_name"].iloc[0]
                else:
                    bank_name = bank_part.replace("_", " ").title()
                if bank_name in seen:
                    continue
                seen.add(bank_name)
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                data[key][bank_name] = _nan_to_none(df.to_dict(orient="list"))
            except Exception as e:
                print(f"  Warning: could not load {f.name}: {e}")

    # Collect forecast months for the month selector (Jul 2026 → Jul 2027)
    MONTH_START = "2026-07-01"
    MONTH_END   = "2027-07-01"
    try:
        # Aggregate forecast months (CC as reference)
        fc_cc = pd.read_csv(PROCESSED / "forecast_cc.csv")
        fc_cc["date"] = pd.to_datetime(fc_cc["date"]).dt.strftime("%Y-%m-%d")
        agg_months = sorted(fc_cc["date"].tolist())

        # Bank-level forecast months
        gu_cc = pd.read_parquet(PROCESSED / "groundup" / "groundup_cc.parquet")
        gu_cc["date"] = pd.to_datetime(gu_cc["date"]).dt.strftime("%Y-%m-%d")
        bank_months = sorted(gu_cc["date"].tolist())

        # Union of both, filtered to Jul 2026 - Jul 2027
        all_months = sorted(set(agg_months + bank_months))
        data["forecast_months"] = [m for m in all_months if MONTH_START <= m <= MONTH_END]
    except Exception:
        data["forecast_months"] = []

    # Load CV summaries for bank MAPE display
    for card_type in ["cc", "dc"]:
        cv_path = PROCESSED / "groundup" / f"bank_cv_summary_{card_type}.csv"
        if cv_path.exists():
            cv_df = pd.read_csv(cv_path)
            data[f"{card_type}_bank_cv"] = cv_df.where(cv_df.notna(), None).to_dict(orient="list")

    # Load aggregate forecast CSVs for month-selector table
    for key, stem, unit in [
        ("fc_cc",      "forecast_cc",      "lakh"),
        ("fc_dc",      "forecast_dc",      "lakh"),
        ("fc_cc_vol",  "forecast_cc_vol",  "lakh_txn"),
        ("fc_dc_vol",  "forecast_dc_vol",  "lakh_txn"),
        ("fc_upi_vol", "forecast_upi_vol", "mn_txn"),
    ]:
        try:
            fc = pd.read_csv(PROCESSED / f"{stem}.csv")
            fc["date"] = pd.to_datetime(fc["date"]).dt.strftime("%Y-%m-%d")
            data[key] = _nan_to_none(fc.to_dict(orient="list"))
        except Exception as e:
            print(f"  Warning: could not load {stem}.csv: {e}")

    print(f"  CC banks loaded: {len(data['all_cc_banks'])}")
    print(f"  DC banks loaded: {len(data['all_dc_banks'])}")
    print(f"  Forecast months: {len(data['forecast_months'])} ({data['forecast_months'][0] if data['forecast_months'] else 'none'} -> {data['forecast_months'][-1] if data['forecast_months'] else 'none'})")

    return data


def embed_in_dashboard(data: dict) -> None:
    html = DASHBOARD.read_text(encoding="utf-8")

    # Find the embedded data block and replace it
    start_marker = "/*DATA_START*/"
    end_marker = "/*DATA_END*/"

    # Find the markers (or the existing const MIP_DATA = ... block)
    if start_marker in html:
        before = html[: html.index(start_marker)]
        after_end = html[html.index(end_marker) + len(end_marker) :]
        # Remove the build(MIP_DATA); line that follows
        if after_end.strip().startswith("build(MIP_DATA);"):
            after_end = after_end[after_end.index("build(MIP_DATA);") + len("build(MIP_DATA);") :]
    elif "const MIP_DATA" in html:
        # Find the const declaration and the build() call
        idx = html.index("const MIP_DATA")
        before = html[:idx]
        # Find the end of the JSON (the ;) then the build call
        build_idx = html.index("build(MIP_DATA);", idx)
        after_end = html[build_idx + len("build(MIP_DATA);") :]
    else:
        print("ERROR: Could not find data embed markers in dashboard.html")
        return

    js_block = (
        f"{start_marker}\n"
        f"const MIP_DATA = {json.dumps(data)};\n"
        f"{end_marker}\n"
        f"build(MIP_DATA);"
    )

    new_html = before + js_block + after_end
    DASHBOARD.write_text(new_html, encoding="utf-8")
    print(f"Dashboard updated: {len(new_html) / 1024:.1f} KB")


def main():
    print("Building dashboard data...")
    data = build_data()

    json_path = PROCESSED / "mip_dashboard_data.json"
    with open(json_path, "w") as f:
        json.dump(data, f)
    print(f"  JSON saved: {json_path.name} ({json_path.stat().st_size / 1024:.1f} KB)")

    print("Embedding in dashboard.html...")
    embed_in_dashboard(data)
    print("Done. Open dashboard.html in any browser (no server needed).")


if __name__ == "__main__":
    main()
