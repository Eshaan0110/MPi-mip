"""
Playwright probe v3 — navigate into Statistics > Payment Systems
"""

import asyncio
from playwright.async_api import async_playwright

BASE_URL = "https://dbie.rbi.org.in"


async def probe():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        print("Loading portal...")
        await page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_timeout(8000)

        # Click "Statistics" in the left nav
        print("Clicking Statistics...")
        await page.click("text=Statistics")
        await page.wait_for_timeout(4000)
        await page.screenshot(path="rbi_statistics.png", full_page=True)

        body = await page.inner_text("body")
        print("\n--- Page text after clicking Statistics ---")
        print(body[:2000])

        links = await page.eval_on_selector_all(
            "a[href]",
            "els => els.map(e => ({text: e.innerText.trim(), href: e.href}))"
        )
        print(f"\nLinks found: {len(links)}")
        for link in links:
            print(f"  {link['text'][:60]:60s}  {link['href']}")

        await browser.close()


asyncio.run(probe())