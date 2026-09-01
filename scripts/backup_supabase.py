"""
MIP — Supabase Backup (P4.5)
===============================
Exports all Supabase tables to JSON files in data/backups/<date>/.
Designed to run nightly via GitHub Actions or locally.

Usage:
    uv run python scripts/backup_supabase.py
    uv run python scripts/backup_supabase.py --dry-run

Restore:
    uv run python scripts/backup_supabase.py --restore 2026-09-01

Environment:
    SUPABASE_URL — your Supabase project URL
    SUPABASE_SERVICE_KEY — service role key
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from loguru import logger

try:
    from supabase import create_client
except ImportError:
    print("Install supabase-py: uv add supabase")
    sys.exit(1)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = PROJECT_ROOT / "data" / "backups"

TABLES = [
    "forecasts_bank",
    "forecasts_aggregate",
    "processed_aggregate",
    "processed_bank_series",
    "model_metadata",
    "raw_npci_upi",
    "scraper_runs",
    "pipeline_runs",
    "scorecard_scores",
    "scenarios",
    "agent_findings",
    "agent_runs",
    "agent_retrains",
    "agent_articles",
]

DRY_RUN = "--dry-run" in sys.argv


def get_client():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        logger.error("Set SUPABASE_URL and SUPABASE_SERVICE_KEY")
        sys.exit(1)
    return create_client(url, key)


def backup(backup_date: str | None = None) -> Path:
    """Export all tables to JSON files."""
    label = backup_date or date.today().isoformat()
    dest = BACKUP_DIR / label
    dest.mkdir(parents=True, exist_ok=True)

    client = get_client()
    manifest = {"date": label, "tables": {}}

    for table in TABLES:
        try:
            result = client.table(table).select("*").execute()
            rows = result.data or []
            manifest["tables"][table] = len(rows)

            if DRY_RUN:
                logger.info(f"  [DRY RUN] {table}: {len(rows)} rows")
                continue

            out_path = dest / f"{table}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2, default=str)
            logger.info(f"  {table}: {len(rows)} rows -> {out_path.name}")
        except Exception as e:
            logger.warning(f"  {table}: FAILED ({e})")
            manifest["tables"][table] = -1

    if not DRY_RUN:
        manifest_path = dest / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Backup complete: {dest}")

    return dest


def restore(backup_label: str) -> None:
    """Restore all tables from a backup."""
    src = BACKUP_DIR / backup_label
    if not src.exists():
        logger.error(f"Backup not found: {src}")
        sys.exit(1)

    client = get_client()

    for table in TABLES:
        json_path = src / f"{table}.json"
        if not json_path.exists():
            logger.info(f"  {table}: no backup file, skipping")
            continue

        with open(json_path, encoding="utf-8") as f:
            rows = json.load(f)

        if not rows:
            logger.info(f"  {table}: empty, skipping")
            continue

        if DRY_RUN:
            logger.info(f"  [DRY RUN] Would restore {len(rows)} rows to {table}")
            continue

        logger.info(f"  {table}: clearing and restoring {len(rows)} rows...")
        client.table(table).delete().neq("id", 0).execute()

        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            for row in batch:
                row.pop("id", None)
            client.table(table).insert(batch).execute()

        logger.info(f"  {table}: restored {len(rows)} rows")

    logger.info(f"Restore from {backup_label} complete")


def main():
    if "--restore" in sys.argv:
        idx = sys.argv.index("--restore")
        if idx + 1 >= len(sys.argv):
            print("Usage: --restore <date-label>")
            sys.exit(1)
        label = sys.argv[idx + 1]
        restore(label)
    else:
        backup()


if __name__ == "__main__":
    main()
