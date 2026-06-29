"""
Research Agent — Web Crawler & Content Collector
=================================================
Crawls structured sources for market intelligence:
    1. RBI circulars & press releases (card/payment related)
    2. Bank newsrooms (top CC/DC issuers)
    3. NPCI announcements
    4. Financial news (card launches, partnerships, regulatory)

Uses Playwright for JS-heavy sites, httpx for simple pages.
Downloads PDFs for Claude extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from loguru import logger

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "agent_cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Source definitions ────────────────────────────────────────────────────

RBI_SOURCES = [
    {
        "name": "RBI Press Releases",
        "url": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
        "type": "html_list",
        "keywords": ["card", "payment", "UPI", "debit", "credit", "digital payment",
                      "NPCI", "interchange", "MDR", "tokenisation", "KYC"],
    },
    {
        "name": "RBI Circulars",
        "url": "https://www.rbi.org.in/Scripts/BS_CircularIndexDisplay.aspx",
        "type": "html_list",
        "keywords": ["card", "payment system", "prepaid", "credit card",
                      "debit card", "digital payment", "UPI"],
    },
    {
        "name": "RBI Notifications",
        "url": "https://www.rbi.org.in/Scripts/NotificationUser.aspx",
        "type": "html_list",
        "keywords": ["card", "payment", "credit", "debit"],
    },
]

BANK_NEWSROOMS = {
    "HDFC Bank": "https://www.hdfcbank.com/personal/about-us/news-room",
    "State Bank of India": "https://sbi.co.in/web/corporate-governance/press-release",
    "ICICI Bank": "https://www.icicibank.com/aboutus/article",
    "Axis Bank": "https://www.axisbank.com/about-us/press-releases",
    "Kotak Mahindra Bank": "https://www.kotak.com/en/about-us/media-center.html",
}

NPCI_SOURCE = {
    "name": "NPCI Announcements",
    "url": "https://www.npci.org.in/what-we-do/upi/announcements",
    "type": "html_list",
}

NEWS_SOURCES = [
    {
        "name": "Livemint Payments",
        "url": "https://www.livemint.com/industry/banking",
        "type": "news",
        "keywords": ["credit card", "debit card", "card launch", "RBI card",
                      "bank cards", "payment card"],
    },
]


@dataclass
class RawArticle:
    """A scraped article/circular/press release."""
    source: str
    title: str
    url: str
    date: str | None = None
    content: str = ""
    pdf_path: str | None = None
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash and self.content:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ── HTTP fetcher ──────────────────────────────────────────────────────────

async def fetch_page(url: str, timeout: int = 30) -> str | None:
    """Fetch a page via httpx. Returns HTML or None on error."""
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "MIP-ResearchAgent/1.0 (market-intelligence)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


async def download_pdf(url: str, filename: str) -> Path | None:
    """Download a PDF to the agent cache directory."""
    dest = _CACHE_DIR / filename
    if dest.exists():
        return dest
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=60,
            headers={"User-Agent": "MIP-ResearchAgent/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            logger.info(f"Downloaded PDF: {filename} ({len(resp.content)} bytes)")
            return dest
    except Exception as e:
        logger.warning(f"PDF download failed {url}: {e}")
        return None


# ── HTML parsers ──────────────────────────────────────────────────────────

def _extract_rbi_links(html: str, base_url: str, keywords: list[str]) -> list[dict]:
    """Extract links from RBI-style listing pages filtered by keywords."""
    from html.parser import HTMLParser

    class LinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links: list[dict] = []
            self._in_a = False
            self._href = ""
            self._text = ""

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self._in_a = True
                self._text = ""
                for name, val in attrs:
                    if name == "href":
                        self._href = val or ""

        def handle_data(self, data):
            if self._in_a:
                self._text += data

        def handle_endtag(self, tag):
            if tag == "a" and self._in_a:
                self._in_a = False
                title = self._text.strip()
                if title and self._href:
                    self.links.append({"title": title, "href": self._href})

    parser = LinkParser()
    parser.feed(html)

    pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    relevant = []
    for link in parser.links:
        if pattern.search(link["title"]):
            href = link["href"]
            if not href.startswith("http"):
                if href.startswith("/"):
                    from urllib.parse import urlparse
                    parsed = urlparse(base_url)
                    href = f"{parsed.scheme}://{parsed.netloc}{href}"
                else:
                    href = base_url.rsplit("/", 1)[0] + "/" + href
            relevant.append({"title": link["title"], "url": href})

    return relevant


# ── Main research pipeline ────────────────────────────────────────────────

async def crawl_rbi_sources() -> list[RawArticle]:
    """Crawl all RBI sources for card/payment related content."""
    articles = []
    for source in RBI_SOURCES:
        logger.info(f"Crawling: {source['name']}")
        html = await fetch_page(source["url"])
        if not html:
            continue
        links = _extract_rbi_links(html, source["url"], source["keywords"])
        logger.info(f"  Found {len(links)} relevant links")

        for link in links[:20]:
            content_html = await fetch_page(link["url"])
            content = ""
            if content_html:
                content = re.sub(r"<[^>]+>", " ", content_html)
                content = re.sub(r"\s+", " ", content).strip()
                content = content[:8000]

            articles.append(RawArticle(
                source=source["name"],
                title=link["title"],
                url=link["url"],
                content=content,
            ))

    return articles


async def crawl_bank_newsrooms() -> list[RawArticle]:
    """Crawl bank newsroom pages for card-related announcements."""
    articles = []
    card_keywords = ["card", "credit", "debit", "launch", "payment", "partner",
                     "co-brand", "contactless", "rewards", "discontinu"]
    pattern = re.compile("|".join(card_keywords), re.IGNORECASE)

    for bank, url in BANK_NEWSROOMS.items():
        logger.info(f"Crawling newsroom: {bank}")
        html = await fetch_page(url)
        if not html:
            continue
        links = _extract_rbi_links(html, url, card_keywords)
        for link in links[:10]:
            articles.append(RawArticle(
                source=f"Bank: {bank}",
                title=link["title"],
                url=link["url"],
                metadata={"bank_name": bank},
            ))

    return articles


async def run_research() -> list[RawArticle]:
    """Full research sweep across all sources."""
    logger.info("Starting research sweep...")

    rbi_articles, bank_articles = await asyncio.gather(
        crawl_rbi_sources(),
        crawl_bank_newsrooms(),
    )

    all_articles = rbi_articles + bank_articles
    logger.info(f"Research complete: {len(all_articles)} articles collected")
    return all_articles


if __name__ == "__main__":
    articles = asyncio.run(run_research())
    for a in articles:
        print(f"[{a.source}] {a.title}")
        print(f"  URL: {a.url}")
        print()
