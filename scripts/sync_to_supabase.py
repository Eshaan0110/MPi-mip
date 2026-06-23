"""
Sync local Parquet data to Supabase PostgreSQL.
================================================
Reads processed data and forecasts from local Parquet files and
upserts them into Supabase tables.

Requires environment variables:
    SUPABASE_URL — your Supabase project URL
    SUPABASE_SERVICE_KEY — service role key (NOT anon key)

Run:
    uv run python scripts/sync_to_supabase.py
    uv run python scripts/sync_to_supabase.py --dry-run   # preview without writing
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

try:
    from supabase import create_client
except ImportError:
    print("Install supabase-py: uv add supabase")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
FORECASTS = PROCESSED / "bankwise_forecasts"
GROUNDUP = PROCESSED / "groundup"

DRY_RUN = "--dry-run" in sys.argv


def get_client():
    url = os.environ.get("SUPABASE_URL", "https://nwevrclikkiuemttovih.supabase.co")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im53ZXZyY2xpa2tpdWVtdHRvdmloIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4MjA1Njc1MSwiZXhwIjoyMDk3NjMyNzUxfQ.YmEpjJsXPilnZ_WA2w74G91oGknziXJTMrizvMsta2o")
    if not url or not key:
        logger.error("Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")
        sys.exit(1)
    return create_client(url, key)


def upsert_df(client, table: str, df: pd.DataFrame, conflict_cols: list[str]):
    """Upsert a DataFrame to a Supabase table."""
    df = df.where(df.notnull(), None)
    records = df.to_dict("records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (v != v):  # NaN check
                rec[k] = None
    if DRY_RUN:
        logger.info(f"  [DRY RUN] Would upsert {len(records)} rows to {table}")
        return 0

    batch_size = 500
    total = 0
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        client.table(table).upsert(batch, on_conflict=",".join(conflict_cols)).execute()
        total += len(batch)

    logger.info(f"  Upserted {total} rows to {table}")
    return total


def sync_bankwise_forecasts(client):
    """Sync per-bank forecast parquets to forecasts_bank table."""
    logger.info("Syncing bank forecasts...")
    all_rows = []

    for f in sorted(FORECASTS.glob("*_forecast.parquet")):
        name = f.stem  # e.g. cc_hdfc_bank_forecast
        parts = name.replace("_forecast", "").split("_", 1)
        if len(parts) != 2:
            continue
        card_type = parts[0].upper()
        bank_name = parts[1].replace("_", " ").title()

        df = pd.read_parquet(f)

        # Handle different column naming conventions
        date_col = "ds" if "ds" in df.columns else "date"
        yhat_col = "yhat" if "yhat" in df.columns else "forecast"
        lower_col = "yhat_lower" if "yhat_lower" in df.columns else "forecast_lower"
        upper_col = "yhat_upper" if "yhat_upper" in df.columns else "forecast_upper"

        if date_col not in df.columns or yhat_col not in df.columns:
            continue

        # Use bank_name from parquet if available, else from filename
        for _, row in df.iterrows():
            bn = row.get("bank_name", bank_name) if "bank_name" in df.columns else bank_name
            ct = row.get("card_type", card_type).upper() if "card_type" in df.columns else card_type
            all_rows.append({
                "bank_name": bn,
                "card_type": ct,
                "forecast_month": str(row[date_col])[:10],
                "yhat": float(row[yhat_col]),
                "yhat_lower": float(row[lower_col]) if pd.notna(row.get(lower_col)) else None,
                "yhat_upper": float(row[upper_col]) if pd.notna(row.get(upper_col)) else None,
                "model_type": "Prophet",
            })

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        return upsert_df(client, "forecasts_bank", df_all, ["bank_name", "card_type", "forecast_month"])
    return 0


def sync_aggregate_forecasts(client):
    """Sync aggregate forecast parquets to forecasts_aggregate table."""
    logger.info("Syncing aggregate forecasts...")
    all_rows = []

    metric_files = {
        "cc_outstanding": PROCESSED / "forecast_cc.parquet",
        "dc_outstanding": PROCESSED / "forecast_dc.parquet",
    }

    # Also check for txn volume forecasts
    for name in ["forecast_cc_vol", "forecast_dc_vol", "forecast_upi_vol"]:
        p = PROCESSED / f"{name}.parquet"
        if p.exists():
            metric = name.replace("forecast_", "").replace("_vol", "_txn_vol")
            if "upi" in name:
                metric = "upi_vol"
            metric_files[metric] = p

    for metric, path in metric_files.items():
        if not path.exists():
            logger.debug(f"  Skipping {metric}: {path.name} not found")
            continue

        df = pd.read_parquet(path)

        # Handle different column naming conventions
        date_col = next((c for c in ["ds", "date"] if c in df.columns), None)
        yhat_col = next((c for c in ["yhat", "forecast_lakh", "forecast"] if c in df.columns), None)
        lower_col = next((c for c in ["yhat_lower", "forecast_lower_lakh", "forecast_lower"] if c in df.columns), None)
        upper_col = next((c for c in ["yhat_upper", "forecast_upper_lakh", "forecast_upper"] if c in df.columns), None)

        if not date_col or not yhat_col:
            logger.debug(f"  Skipping {metric}: no date/forecast columns found in {df.columns.tolist()}")
            continue

        for _, row in df.iterrows():
            all_rows.append({
                "metric": metric,
                "forecast_month": str(row[date_col])[:10],
                "yhat": float(row[yhat_col]),
                "yhat_lower": float(row[lower_col]) if lower_col and pd.notna(row.get(lower_col)) else None,
                "yhat_upper": float(row[upper_col]) if upper_col and pd.notna(row.get(upper_col)) else None,
                "model_type": "Prophet",
            })

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        return upsert_df(client, "forecasts_aggregate", df_all, ["metric", "forecast_month"])
    return 0


def sync_raw_npci(client):
    """Sync NPCI UPI JSON data to raw_npci_upi table."""
    logger.info("Syncing NPCI UPI data...")
    json_path = PROJECT_ROOT / "data" / "raw" / "npci_upi" / "npci_upi_monthly.json"
    if not json_path.exists():
        logger.debug("  No NPCI UPI JSON found")
        return 0

    records = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    for r in records:
        if not r.get("date"):
            continue
        rows.append({
            "month": r["date"],
            "banks_live": int(r["banks_live"]) if r.get("banks_live") else None,
            "volume_mn": r.get("volume_mn"),
            "value_cr": r.get("value_cr"),
        })

    if rows:
        df = pd.DataFrame(rows)
        return upsert_df(client, "raw_npci_upi", df, ["month"])
    return 0


def sync_model_metadata(client):
    """Sync per-bank CV MAPE from local CSV summaries + aggregate MAPE to model_metadata."""
    logger.info("Syncing model metadata...")
    rows = []

    # Bank-level CV MAPE from groundup summaries
    for card_type in ["cc", "dc"]:
        cv_path = GROUNDUP / f"bank_cv_summary_{card_type}.csv"
        if not cv_path.exists():
            continue
        cv = pd.read_csv(cv_path)
        for _, r in cv.iterrows():
            rows.append({
                "bank_name": r["bank_name"],
                "card_type": card_type.upper(),
                "metric": None,
                "model_type": "Prophet",
                "cv_mape": round(float(r["mape_mean"]), 2),
                "params_json": json.dumps({
                    "cv_windows": int(r["cv_windows"]),
                    "mape_median": round(float(r["mape_median"]), 2),
                    "mape_min": round(float(r["mape_min"]), 2),
                    "mape_max": round(float(r["mape_max"]), 2),
                }),
            })

    # Aggregate-level MAPE (hardcoded from rebuild_dashboard.py)
    agg_mape = {
        "cc_outstanding": 3.46,
        "dc_outstanding": 7.08,
        "cc_txn_vol": 13.63,
        "upi_vol": 12.31,
    }
    for metric, mape in agg_mape.items():
        rows.append({
            "bank_name": None,
            "card_type": None,
            "metric": metric,
            "model_type": "Prophet",
            "cv_mape": mape,
        })

    if rows:
        df = pd.DataFrame(rows)
        return upsert_df(client, "model_metadata", df, ["bank_name", "card_type", "metric"])
    return 0


def seed_scraper_runs(client):
    """Seed scraper_runs with initial records so Data Status page isn't empty."""
    logger.info("Seeding scraper runs...")
    sources = {
        "rbi_bankwise": 948,
        "npci_upi": 102,
    }
    rows = []
    for source, count in sources.items():
        rows.append({
            "source": source,
            "status": "success",
            "records_written": count,
        })

    if DRY_RUN:
        logger.info(f"  [DRY RUN] Would insert {len(rows)} scraper_run records")
        return 0

    for row in rows:
        client.table("scraper_runs").insert(row).execute()
    logger.info(f"  Inserted {len(rows)} scraper_run records")
    return len(rows)


def main():
    logger.info(f"{'[DRY RUN] ' if DRY_RUN else ''}Syncing local data to Supabase...")

    client = get_client()
    total = 0

    total += sync_bankwise_forecasts(client)
    total += sync_aggregate_forecasts(client)
    total += sync_raw_npci(client)
    total += sync_model_metadata(client)
    total += seed_scraper_runs(client)

    logger.info(f"Done. Total rows synced: {total}")


if __name__ == "__main__":
    main()
