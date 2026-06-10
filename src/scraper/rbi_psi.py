"""
RBI Payment System Indicators — Auto Downloader
==================================================
Navigates the RBI DBIE portal to download the PSI Excel file.

DBIE is a JavaScript-heavy single-page app. The path is:
  dbie.rbi.org.in → Statistics → Financial Sector → Payment Systems

This scraper uses Playwright to navigate the portal and trigger the
Excel export. It's fragile by nature — RBI redesigns the portal
periodically. If it breaks, check the portal structure manually and
update the selectors.

Run:
    uv run python -m src.scraper.rbi_psi
"""
import asyncio
import sys
from pathlib import Path
from datetime import datetime

from playwright.async_api import async_playwright
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw" / "rbi_psi"

# DBIE portal URLs
DBIE_BASE = "https://dbie.rbi.org.in"
DBIE_STATS = f"{DBIE_BASE}/DBIE/dbie.rbi?site=statistics"

# Alternative: direct PSI page if DBIE restructures
ALT_PSI_URL = "https://www.rbi.org.in/Scripts/PSIUserView.aspx"


async def try_direct_psi_page() -> list[Path]:
    """Fallback: scrape the PSI user view page for direct Excel links."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        logger.info(f"Trying direct PSI page: {ALT_PSI_URL}")
        await page.goto(ALT_PSI_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(3000)

        # Look for Excel download links
        links = await page.eval_on_selector_all(
            "a[href]",
            """els => els.map(e => ({
                text: e.innerText.trim(),
                href: e.href
            })).filter(l =>
                l.href.match(/\\.xlsx?$/i) &&
                (l.href.match(/Payment.System/i) || l.text.match(/Payment.System/i))
            )"""
        )

        if not links:
            # Broader search
            links = await page.eval_on_selector_all(
                "a[href]",
                """els => els.map(e => ({
                    text: e.innerText.trim(),
                    href: e.href
                })).filter(l => l.href.match(/\\.xlsx?$/i))"""
            )

        logger.info(f"Found {len(links)} Excel links on PSI page")
        for l in links[:10]:
            logger.debug(f"  {l['text'][:50]:50s}  {l['href'][:80]}")

        existing = {f.name.upper() for f in _RAW_DIR.iterdir() if f.suffix.upper() in (".XLS", ".XLSX")}
        downloaded = []

        for link in links:
            url = link["href"]
            fname = url.rsplit("/", 1)[-1]
            if fname.upper() in existing:
                continue

            dest = _RAW_DIR / fname
            try:
                import httpx
                async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
                    resp = await client.get(url)
                    if resp.status_code == 200 and len(resp.content) > 5000:
                        dest.write_bytes(resp.content)
                        logger.success(f"  Downloaded: {fname} ({len(resp.content)//1024} KB)")
                        downloaded.append(dest)
            except Exception as e:
                logger.error(f"  Failed to download {fname}: {e}")

        await browser.close()
        return downloaded


async def try_dbie_portal() -> list[Path]:
    """Primary: navigate DBIE portal to export PSI data."""
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--ignore-certificate-errors"],  # DBIE has cert issues
        )
        page = await browser.new_page()

        try:
            logger.info(f"Loading DBIE portal: {DBIE_STATS}")
            await page.goto(DBIE_STATS, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(8000)

            # Screenshot for debugging
            await page.screenshot(path=str(_PROJECT_ROOT / "reports" / "dbie_portal.png"))

            # Try to find "Payment Systems" or "Financial Sector" links
            body = await page.inner_text("body")
            if "Payment System" in body:
                logger.info("Found 'Payment System' text on page")

                # Click through the navigation
                ps_link = await page.query_selector("text=Payment System")
                if ps_link:
                    await ps_link.click()
                    await page.wait_for_timeout(5000)
                    await page.screenshot(path=str(_PROJECT_ROOT / "reports" / "dbie_psi.png"))

                    # Look for export/download button
                    export_btn = await page.query_selector("text=Export") or \
                                 await page.query_selector("text=Download") or \
                                 await page.query_selector("text=Excel")
                    if export_btn:
                        async with page.expect_download(timeout=60000) as dl_info:
                            await export_btn.click()
                        download = await dl_info.value
                        dest = _RAW_DIR / f"PSI_export_{datetime.now():%Y%m%d}.xlsx"
                        await download.save_as(str(dest))
                        logger.success(f"Downloaded PSI via DBIE: {dest}")
                        await browser.close()
                        return [dest]

            logger.warning("DBIE portal navigation failed. Portal may have changed structure.")
            await browser.close()
            return []

        except Exception as e:
            logger.error(f"DBIE portal error: {e}")
            await browser.close()
            return []


async def run():
    # Try DBIE first, fall back to direct page
    results = await try_dbie_portal()
    if not results:
        logger.info("DBIE failed, trying direct PSI page...")
        results = await try_direct_psi_page()
    return results


def main():
    results = asyncio.run(run())
    if results:
        print(f"\nDownloaded {len(results)} PSI file(s)")
    else:
        print("\nCould not download PSI data automatically.")
        print("Manual download: https://www.rbi.org.in/Scripts/PSIUserView.aspx")
        print("Save to: data/raw/rbi_psi/")


if __name__ == "__main__":
    main()
