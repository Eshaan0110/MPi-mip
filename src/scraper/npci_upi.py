"""
NPCI UPI Product Statistics — Auto Downloader
===============================================
Fetches UPI monthly statistics from NPCI's public JSON API.

NPCI serves data via a REST API at:
  https://www.npci.org.in/api/product-statistic/tab/detail
  ?product_name=upi&tab_name=product-statistics-upi
  &year_range=YYYY-YY&excel_type=monthly&page_no=1&size=50&sort_by=asc

Data includes: month, no_of_banks_live_on_upi, volume_in_mn, value_in_cr

No browser automation needed — direct HTTP calls.

Run:
    uv run python -m src.scraper.npci_upi
    uv run python -m src.scraper.npci_upi --all   # fetch all years
"""
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "npci_upi"

API_URL = "https://www.npci.org.in/api/product-statistic/tab/detail"


def _year_ranges(all_years: bool = False) -> list[str]:
    """Generate fiscal year ranges to query. Indian FY runs Apr–Mar."""
    now = datetime.now()
    current_fy_start = now.year if now.month >= 4 else now.year - 1
    if all_years:
        return [f"{y}-{str(y+1)[-2:]}" for y in range(2016, current_fy_start + 1)]
    # Just current + previous FY
    return [
        f"{current_fy_start}-{str(current_fy_start+1)[-2:]}",
        f"{current_fy_start-1}-{str(current_fy_start)[-2:]}",
    ]


def _parse_number(s: str) -> float | None:
    """Parse Indian-formatted numbers like '22,641.11' or '29,52,542.05'."""
    if not s or s.strip() == "":
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


async def fetch_year(year_range: str) -> list[dict]:
    """Fetch monthly UPI stats for a fiscal year range like '2025-26'."""
    import httpx

    params = {
        "product_name": "upi",
        "tab_name": "product-statistics-upi",
        "year_range": year_range,
        "excel_type": "monthly",
        "page_no": 1,
        "size": 50,
        "sort_by": "asc",
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=30, verify=False) as client:
        resp = await client.get(API_URL, params=params)
        if resp.status_code != 200:
            logger.warning(f"  API returned {resp.status_code} for {year_range}")
            return []

        data = resp.json()
        results = data.get("data", {}).get("results", [])
        logger.info(f"  FY {year_range}: {len(results)} months")
        return results


def _month_to_date(month_str: str) -> str | None:
    """Convert 'May-2026' to '2026-05-01'."""
    try:
        dt = datetime.strptime(month_str, "%B-%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


async def scrape_and_download(download_all: bool = False) -> list[Path]:
    """Fetch NPCI UPI data via API and save as JSON files."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    year_ranges = _year_ranges(all_years=download_all)
    logger.info(f"Fetching NPCI UPI data for: {year_ranges}")

    all_records = []
    for yr in year_ranges:
        records = await fetch_year(yr)
        for rec in records:
            all_records.append({
                "month": rec.get("month", ""),
                "date": _month_to_date(rec.get("month", "")),
                "banks_live": _parse_number(rec.get("no_of_banks_live_on_upi", "")),
                "volume_mn": _parse_number(rec.get("volume_in_mn", "")),
                "value_cr": _parse_number(rec.get("value_in_cr", "")),
                "source": "npci_api",
                "fetched_at": datetime.now().isoformat(),
            })

    if not all_records:
        logger.warning("No records fetched from NPCI API")
        return []

    # Deduplicate by date
    seen = set()
    unique = []
    for r in all_records:
        if r["date"] and r["date"] not in seen:
            seen.add(r["date"])
            unique.append(r)

    unique.sort(key=lambda x: x["date"])

    # Save as JSON (structured data, not Excel)
    out_path = _RAW_DIR / "npci_upi_monthly.json"
    existing_data = []
    if out_path.exists():
        try:
            existing_data = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Merge: keep existing records, add/update new ones
    existing_dates = {r["date"] for r in existing_data if r.get("date")}
    new_count = 0
    for r in unique:
        if r["date"] not in existing_dates:
            existing_data.append(r)
            new_count += 1
        else:
            # Update existing record
            for i, er in enumerate(existing_data):
                if er.get("date") == r["date"]:
                    existing_data[i] = r
                    break

    existing_data.sort(key=lambda x: x.get("date", ""))
    out_path.write_text(json.dumps(existing_data, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Saved {len(existing_data)} total records ({new_count} new) to {out_path.name}")
    return [out_path] if new_count > 0 else []


def main():
    download_all = "--all" in sys.argv
    results = asyncio.run(scrape_and_download(download_all=download_all))
    if results:
        print(f"\nUpdated {len(results)} file(s) with new NPCI UPI data")
    else:
        print("\nNo new NPCI UPI data to download.")


if __name__ == "__main__":
    main()
