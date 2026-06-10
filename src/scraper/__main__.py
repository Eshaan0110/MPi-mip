"""
MIP Auto-Scraper — Downloads all datasets from source websites.
==================================================================
Run:
    uv run python -m src.scraper              # download only new files
    uv run python -m src.scraper --all        # download all available files
    uv run python -m src.scraper --only rbi_bankwise   # single source

Sources:
    rbi_bankwise  — RBI bank-wise ATM/POS/Card statistics (monthly Excel)
    npci_upi      — NPCI UPI ecosystem statistics (yearly Excel)
    rbi_psi       — RBI Payment System Indicators (DBIE portal)
    rbi_repo      — RBI Repo Rate (Handbook of Statistics)
"""
import asyncio
import sys
import time
from loguru import logger

from src.scraper.rbi_bankwise import run as run_bankwise
from src.scraper.npci_upi import scrape_and_download as run_npci
from src.scraper.rbi_psi import run as run_psi
from src.scraper.rbi_repo import run as run_repo


SCRAPERS = {
    "rbi_bankwise": ("RBI Bankwise (ATM/POS/Card Stats)", run_bankwise),
    "npci_upi":     ("NPCI UPI Statistics", run_npci),
    "rbi_psi":      ("RBI Payment System Indicators", run_psi),
    "rbi_repo":     ("RBI Repo Rate", run_repo),
}


async def run_all(only: str | None = None, download_all: bool = False):
    results = {}
    t0 = time.time()

    for key, (name, func) in SCRAPERS.items():
        if only and key != only:
            continue

        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        try:
            if key == "rbi_bankwise":
                r = await func(download_all=download_all)
            elif key == "rbi_repo":
                downloaded, rate = await func()
                r = downloaded
                if rate:
                    print(f"  Current Repo Rate: {rate}%")
            else:
                r = await func()

            results[key] = {
                "status": "OK" if r else "NO_NEW_DATA",
                "files": len(r) if isinstance(r, list) else 0,
            }
        except Exception as e:
            logger.error(f"  {name} FAILED: {e}")
            results[key] = {"status": "FAILED", "error": str(e)}

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SCRAPER COMPLETE — {elapsed:.1f}s")
    print(f"{'='*60}")
    for key, r in results.items():
        name = SCRAPERS[key][0]
        status = r["status"]
        icon = "OK" if status == "OK" else "--" if status == "NO_NEW_DATA" else "FAIL"
        files = r.get("files", 0)
        print(f"  [{icon:4s}] {name:<40s} {files} new file(s)" +
              (f" — {r.get('error', '')}" if status == "FAILED" else ""))

    return results


def main():
    only = None
    download_all = "--all" in sys.argv
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = sys.argv[idx + 1]

    asyncio.run(run_all(only=only, download_all=download_all))


if __name__ == "__main__":
    main()
