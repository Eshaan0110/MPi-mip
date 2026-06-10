"""
NPCI UPI Ecosystem Statistics — Auto Downloader
=================================================
Scrapes the NPCI UPI stats page and downloads yearly Excel files.

The page at npci.org.in uses JavaScript to render content, so we use
Playwright to extract the actual download URLs.

Run:
    uv run python -m src.scraper.npci_upi
"""
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "npci_upi"

PAGE_URL = "https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics"


async def scrape_and_download():
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    existing = {f.name.upper() for f in _RAW_DIR.iterdir() if f.suffix.upper() in (".XLS", ".XLSX")}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Loading {PAGE_URL}")
        await page.goto(PAGE_URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(5000)

        # NPCI renders download links dynamically. Look for Excel links.
        links = await page.eval_on_selector_all(
            "a[href]",
            """els => els.map(e => ({
                text: e.innerText.trim(),
                href: e.href
            })).filter(l =>
                l.href.match(/\\.xlsx?$/i) ||
                l.href.match(/Product.Statistics.*UPI/i) ||
                l.href.match(/Ecosystem.Statistics.*UPI/i)
            )"""
        )

        if not links:
            # Try looking for download buttons that trigger JS downloads
            logger.info("No direct links found. Looking for download triggers...")
            # Some NPCI pages have year tabs that reveal download links
            year_tabs = await page.query_selector_all("[data-year], .year-tab, .nav-link")
            for tab in year_tabs[:6]:
                try:
                    await tab.click()
                    await page.wait_for_timeout(2000)
                except Exception:
                    pass

            # Re-check for links after clicking tabs
            links = await page.eval_on_selector_all(
                "a[href]",
                """els => els.map(e => ({
                    text: e.innerText.trim(),
                    href: e.href
                })).filter(l =>
                    l.href.match(/\\.xlsx?$/i) ||
                    l.href.match(/UPI/i)
                )"""
            )

        logger.info(f"Found {len(links)} potential download links")
        for l in links[:10]:
            logger.debug(f"  {l['text'][:40]:40s} {l['href'][:80]}")

        downloaded = []
        for link in links:
            url = link["href"]
            fname = url.rsplit("/", 1)[-1]
            if fname.upper() in existing:
                logger.debug(f"  Already have: {fname}")
                continue

            dest = _RAW_DIR / fname
            logger.info(f"Downloading: {fname}")
            try:
                async with page.expect_download(timeout=30000) as dl_info:
                    await page.evaluate(f"window.location.href = '{url}'")
                download = await dl_info.value
                await download.save_as(str(dest))
                logger.success(f"  Saved: {dest}")
                downloaded.append(dest)
            except Exception:
                # Fallback: direct HTTP download
                try:
                    import httpx
                    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                        resp = await client.get(url)
                        if resp.status_code == 200 and len(resp.content) > 1000:
                            dest.write_bytes(resp.content)
                            logger.success(f"  Saved (HTTP): {dest}")
                            downloaded.append(dest)
                        else:
                            logger.warning(f"  HTTP download failed: status={resp.status_code} size={len(resp.content)}")
                except Exception as e2:
                    logger.error(f"  Failed: {e2}")

        await browser.close()
        return downloaded


def main():
    results = asyncio.run(scrape_and_download())
    if results:
        print(f"\nDownloaded {len(results)} new file(s)")
    else:
        print("\nNo new files to download.")


if __name__ == "__main__":
    main()
