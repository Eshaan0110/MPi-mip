"""
RBI Repo Rate — Auto Downloader
=================================
Downloads the Repo Rate from RBI's Handbook of Statistics or the
current rates page. The repo rate only changes ~6-8 times per year
at MPC meetings, so this doesn't need to run monthly.

Strategy:
  1. Try to download the Handbook of Statistics Table 43 Excel
  2. Fallback: scrape the current rate from rbi.org.in/Scripts/BS_ViewBulletin.aspx

Run:
    uv run python -m src.scraper.rbi_repo
"""
import asyncio
import re
from pathlib import Path

from playwright.async_api import async_playwright
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "rbi_repo_rate"

HANDBOOK_URL = "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=15817"
CURRENT_RATES_URL = "https://www.rbi.org.in/Scripts/BS_NSDPDisplay.aspx?param=4"


async def download_handbook_table():
    """Try to find and download Table 43 from Handbook of Statistics."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Loading Handbook page: {HANDBOOK_URL}")
        await page.goto(HANDBOOK_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        # Look for Table 43 or "Policy Rates" links
        links = await page.eval_on_selector_all(
            "a[href]",
            """els => els.map(e => ({
                text: e.innerText.trim(),
                href: e.href
            })).filter(l =>
                l.href.match(/\\.xlsx?$/i) ||
                l.text.match(/Table.*43|Policy.*Rate|Repo/i)
            )"""
        )

        logger.info(f"Found {len(links)} potential links")
        for l in links[:10]:
            logger.debug(f"  {l['text'][:50]:50s}  {l['href'][:80]}")

        existing = {f.name.upper() for f in _RAW_DIR.iterdir() if f.suffix.upper() in (".XLS", ".XLSX")}
        downloaded = []

        for link in links:
            if not link["href"].upper().endswith((".XLS", ".XLSX")):
                continue
            fname = link["href"].rsplit("/", 1)[-1]
            if fname.upper() in existing:
                logger.debug(f"  Already have: {fname}")
                continue
            dest = _RAW_DIR / fname
            try:
                import httpx
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(link["href"])
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        dest.write_bytes(resp.content)
                        logger.success(f"  Downloaded: {fname}")
                        downloaded.append(dest)
            except Exception as e:
                logger.error(f"  Failed: {e}")

        await browser.close()
        return downloaded


async def scrape_current_rate():
    """Scrape the current repo rate from RBI's policy rates page."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Scraping current rate from: {CURRENT_RATES_URL}")
        await page.goto(CURRENT_RATES_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        body = await page.inner_text("body")
        # Look for "Repo Rate" followed by a number
        match = re.search(r"Repo\s*Rate[^\d]*(\d+\.?\d*)%?", body, re.IGNORECASE)
        await browser.close()

        if match:
            rate = float(match.group(1))
            logger.info(f"Current Repo Rate: {rate}%")
            return rate
        else:
            logger.warning("Could not extract current repo rate from page")
            return None


async def run():
    downloaded = await download_handbook_table()
    rate = await scrape_current_rate()
    return downloaded, rate


def main():
    downloaded, rate = asyncio.run(run())
    if downloaded:
        print(f"Downloaded {len(downloaded)} file(s)")
    if rate:
        print(f"Current Repo Rate: {rate}%")
    if not downloaded and not rate:
        print("Could not get repo rate data automatically.")
        print("Manual: download RepoRate2007.XLSX from RBI Handbook of Statistics")


if __name__ == "__main__":
    main()
