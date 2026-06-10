"""Probe all data source websites to understand their structure for scraping."""
import asyncio
import sys
from pathlib import Path
from playwright.async_api import async_playwright

TARGETS = {
    "npci_upi": {
        "url": "https://www.npci.org.in/what-we-do/upi/upi-ecosystem-statistics",
        "goal": "Find Excel download links for UPI stats",
    },
    "rbi_bankwise": {
        "url": "https://www.rbi.org.in/Scripts/ATMView.aspx",
        "goal": "Find monthly bankwise Excel download links",
    },
    "rbi_psi": {
        "url": "https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics",
        "goal": "Navigate to Payment System Indicators and find export option",
    },
    "rbi_repo": {
        "url": "https://www.rbi.org.in/Scripts/PublicationsView.aspx?id=15817",
        "goal": "Find Handbook of Statistics Table 43 download",
    },
    "mospi_cpi": {
        "url": "https://cpi.mospi.gov.in/cpidownload",
        "goal": "Find CPI time series download",
    },
}

async def probe_site(name, info, browser):
    print(f"\n{'='*70}")
    print(f"PROBING: {name}")
    print(f"URL: {info['url']}")
    print(f"Goal: {info['goal']}")
    print(f"{'='*70}")

    page = await browser.new_page()
    try:
        await page.goto(info["url"], wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(5000)

        # Get page title
        title = await page.title()
        print(f"Title: {title}")

        # Find all download links (xlsx, xls, csv, zip)
        links = await page.eval_on_selector_all(
            "a[href]",
            """els => els.map(e => ({
                text: e.innerText.trim().substring(0, 80),
                href: e.href
            })).filter(l =>
                l.href.match(/\.(xlsx?|csv|zip)/i) ||
                l.text.match(/download|excel|export/i) ||
                l.href.match(/download|export/i)
            )"""
        )

        if links:
            print(f"\nDownload links found: {len(links)}")
            for l in links[:20]:
                print(f"  {l['text'][:50]:50s}  {l['href'][:80]}")
        else:
            print("\nNo direct download links found. Checking for buttons...")
            buttons = await page.eval_on_selector_all(
                "button, input[type=button], input[type=submit], a.btn, [role=button]",
                "els => els.map(e => ({text: e.innerText.trim().substring(0, 60), tag: e.tagName}))"
            )
            for b in buttons[:15]:
                print(f"  <{b['tag']}> {b['text']}")

        # Screenshot
        ss_path = f"reports/probe_{name}.png"
        Path("reports").mkdir(exist_ok=True)
        await page.screenshot(path=ss_path, full_page=False)
        print(f"\nScreenshot: {ss_path}")

    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        await page.close()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        for name, info in TARGETS.items():
            await probe_site(name, info, browser)
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
