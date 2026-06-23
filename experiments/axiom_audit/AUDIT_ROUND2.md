# Axiom Technical Audit — Round 2: Production Readiness

## Scope
Beyond ML accuracy (covered in Round 1). This round covers: CI/CD pipeline robustness, web app error handling, data validation, monitoring, and deployment config.

---

## Finding P01 — CI/CD: Silent Failures (CRITICAL)
**File:** `.github/workflows/monthly_pipeline.yml`
**Problem:** Every step has `continue-on-error: true`. If scraping fails, ingestion runs on stale data, training runs on stale data, sync pushes stale results to Supabase. No notification — pipeline shows green even when broken.
**Impact:** Dashboard serves stale/wrong forecasts with no one knowing.
**Fix:** Remove `continue-on-error` from critical steps. Add a failure notification step. Chain steps with proper dependencies.

## Finding P02 — CI/CD: No Pipeline Status Tracking (HIGH)
**File:** `.github/workflows/monthly_pipeline.yml`
**Problem:** The `pipeline_runs` table exists in Supabase but nothing writes to it. The "Report pipeline status" step just echoes to console.
**Impact:** Data Status page has no visibility into pipeline health. No audit trail.
**Fix:** Add a step that writes pipeline status to Supabase on success/failure.

## Finding P03 — Web: Placeholder Supabase Client (HIGH)
**File:** `web/src/lib/supabase.ts`
**Problem:** If env vars are missing, falls through to `createClient("https://placeholder.supabase.co", "placeholder")`. This silently makes API calls to a non-existent host, returning cryptic errors.
**Impact:** Dev/deploy confusion. No clear error message.
**Fix:** Check `isConfigured` in pages and show a config error banner instead of loading forever.

## Finding P04 — Web: No Error States (MEDIUM)
**File:** All page.tsx files
**Problem:** Every page catches `error` from Supabase but ignores it — just checks `if (data)`. If the query fails, user sees "Loading..." forever.
**Impact:** Bad UX when Supabase is down or rate-limited.
**Fix:** Add error state handling with user-friendly message.

## Finding P05 — Web: Hardcoded Accuracy on About Page (LOW)
**File:** `web/src/app/about/page.tsx`
**Problem:** Shows "CC Bank Median CV MAPE 7.9%" etc. as hardcoded text. These will be wrong as models improve.
**Impact:** Misleading. Looks unprofessional when numbers don't match Models page.
**Fix:** Fetch from model_metadata or remove specific numbers.

## Finding P06 — Scrapers: No Retry / No Timeout Handling (MEDIUM)
**File:** `src/scraper/*.py`
**Problem:** HTTP requests have timeouts but no retry logic. A single transient failure (503, network blip) kills the entire scrape.
**Impact:** Monthly pipeline fails on transient errors, requires manual re-run.
**Fix:** Add simple retry with exponential backoff (3 attempts, 2s/4s/8s).

## Finding P07 — Pipeline: No Data Freshness Validation (HIGH)
**Problem:** After scraping, there's no check that new data actually arrived. If NPCI changes their API or RBI changes their page layout, scraper returns 0 rows silently.
**Impact:** Pipeline proceeds with stale data, retrains models on same data, pushes identical forecasts.
**Fix:** Add a validation step: compare latest data month vs. expected month. Fail loudly if data isn't fresh.

## Finding P08 — Web: Nav Uses `<a>` Tags Instead of Next.js `<Link>` (LOW)
**File:** `web/src/app/layout.tsx`
**Problem:** Navigation uses plain `<a>` tags. Each click does a full page reload instead of client-side navigation.
**Impact:** Slower navigation, poor UX. Loses state on page switch.
**Fix:** Use `next/link` `<Link>` component.

## Finding P09 — Security: Service Key Rotation Needed (MEDIUM)
**Problem:** Service role key was shared in plaintext during development. Should be rotated before production.
**Impact:** Compromised key = full database access.
**Fix:** Rotate in Supabase dashboard. Update GitHub secrets.

## Finding P10 — Web: Banks Page Fetches ALL Data at Once (MEDIUM)
**File:** `web/src/app/banks/page.tsx`
**Problem:** `SELECT *` from `forecasts_bank` with no limit. As data grows (11 banks × 24 months × 2 card types × monthly retrains), this query returns thousands of rows on every page load.
**Impact:** Slow page load, high bandwidth. Will get worse over time.
**Fix:** Filter by latest training run. Or paginate. Or add a `latest_only` view.

---

## Priority Matrix

| # | Severity | Effort | Fix |
|---|----------|--------|-----|
| P01 | CRITICAL | 30min | Remove continue-on-error, add failure handling |
| P07 | HIGH | 1hr | Data freshness check after scraping |
| P02 | HIGH | 30min | Write pipeline status to Supabase |
| P03 | HIGH | 15min | Config check + error banner |
| P04 | MEDIUM | 30min | Error states in all pages |
| P06 | MEDIUM | 30min | Retry logic for HTTP requests |
| P08 | LOW | 10min | next/link for navigation |
| P10 | MEDIUM | 20min | Filter forecasts_bank query |
| P05 | LOW | 15min | Dynamic accuracy on About page |
| P09 | MEDIUM | 5min | Key rotation (user action) |
