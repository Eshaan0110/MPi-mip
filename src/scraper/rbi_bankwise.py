"""
RBI Bankwise ATM/POS/Card Statistics — Auto Downloader
========================================================
Scrapes https://www.rbi.org.in/Scripts/ATMView.aspx to find the latest
bankwise Excel file, downloads it if not already present.

The page lists direct download links like:
  https://rbidocs.rbi.org.in/rdocs/ATM/DOCs/ATMAPRIL2026...XLSX

Run:
    uv run python -m src.scraper.rbi_bankwise
    uv run python -m src.scraper.rbi_bankwise --all   # download all available
"""
import asyncio
import re
import sys
from pathlib import Path

from playwright.async_api import async_playwright
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "rbi_bankwise"

PAGE_URL = "https://www.rbi.org.in/Scripts/ATMView.aspx"

# Extract month-year from URL like ATMAPRIL2026... or ATM122025...
_MONTH_RE = re.compile(
    r"ATM(?:CS|C|P|S)?(?:"
    r"(?P<mname>JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)"
    r"|(?P<mnum>\d{2}))"
    r"(?P<year>\d{4})",
    re.IGNORECASE,
)

MONTHS = {
    "JANUARY":1,"FEBRUARY":2,"MARCH":3,"APRIL":4,"MAY":5,"JUNE":6,
    "JULY":7,"AUGUST":8,"SEPTEMBER":9,"OCTOBER":10,"NOVEMBER":11,"DECEMBER":12,
}


def parse_month_from_url(url: str):
    m = _MONTH_RE.search(url)
    if not m:
        return None
    year = int(m.group("year"))
    if m.group("mname"):
        month = MONTHS[m.group("mname").upper()]
    else:
        month = int(m.group("mnum"))
    return (year, month)


def filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1]


async def scrape_download_links() -> list[dict]:
    """Return list of {url, filename, year, month} for all bankwise files on the page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Loading {PAGE_URL}")
        await page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => e.href).filter(h => h.match(/rdocs\\/ATM\\/DOCs/i) && h.match(/\\.xlsx?$/i))"
        )
        await browser.close()

    results = []
    for url in links:
        parsed = parse_month_from_url(url)
        if parsed:
            year, month = parsed
            results.append({
                "url": url,
                "filename": filename_from_url(url),
                "year": year,
                "month": month,
            })

    results.sort(key=lambda x: (x["year"], x["month"]), reverse=True)
    logger.info(f"Found {len(results)} bankwise files on RBI page")
    return results


async def download_file(url: str, dest: Path) -> bool:
    """Download a single file via HTTP (RBI serves files directly, no JS needed)."""
    import httpx
    async with httpx.AsyncClient(follow_redirects=True, timeout=60, verify=False) as client:
        resp = await client.get(url)
        if resp.status_code == 200 and len(resp.content) > 5000:
            dest.write_bytes(resp.content)
            return True
        else:
            raise RuntimeError(f"HTTP {resp.status_code}, size={len(resp.content)}")


async def run(download_all: bool = False):
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    links = await scrape_download_links()
    if not links:
        logger.error("No download links found. RBI page may have changed.")
        return []

    # Check what we already have
    existing = {f.name.upper() for f in _RAW_DIR.iterdir() if f.suffix.upper() in (".XLS", ".XLSX")}
    # Also check subdirectories
    for d in _RAW_DIR.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.upper() in (".XLS", ".XLSX"):
                    existing.add(f.name.upper())

    to_download = []
    for link in links:
        if link["filename"].upper() in existing:
            logger.debug(f"  Already have: {link['filename']}")
            continue
        to_download.append(link)
        if not download_all:
            break  # only download the latest new one

    if not to_download:
        logger.info("All files already downloaded. Nothing new.")
        return []

    downloaded = []
    for link in to_download:
        dest = _RAW_DIR / link["filename"]
        logger.info(f"Downloading: {link['filename']} ({link['month']}/{link['year']})")
        try:
            await download_file(link["url"], dest)
            logger.success(f"  Saved: {dest}")
            downloaded.append(dest)
        except Exception as e:
            logger.error(f"  Failed: {e}")

    return downloaded


def main():
    download_all = "--all" in sys.argv
    results = asyncio.run(run(download_all=download_all))
    if results:
        print(f"\nDownloaded {len(results)} new file(s):")
        for p in results:
            print(f"  {p}")
    else:
        print("\nNo new files to download.")


if __name__ == "__main__":
    main()
