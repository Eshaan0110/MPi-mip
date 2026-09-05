# Expansion Design Note: United Kingdom (UK)

## Why UK first

1. **Data availability**: Bank of England publishes monthly card statistics
   (Table A5.2) in machine-readable CSV — no scraping needed.
2. **Regulatory clarity**: FCA publishes card fee caps, lending rules, and
   consumer credit data openly.
3. **Structural similarity to India**: UK has a concentrated issuer market
   (~6 banks hold >80% of outstanding) with clear seasonal patterns and
   well-documented structural events (Brexit, COVID, cost-of-living crisis).
4. **English-language sources**: No translation layer needed for the agent
   researcher pipeline.

## Data sources

| Source | URL pattern | Frequency | Format |
|--------|------------|-----------|--------|
| BoE Table A5.2 | bankofengland.co.uk/statistics/credit-card | Monthly | CSV |
| BoE Table A5.4 | bankofengland.co.uk/statistics/money-and-credit | Monthly | CSV |
| FCA Product Sales Data | fca.org.uk/data/product-sales-data | Quarterly | XLSX |
| UK Finance Card Payments | ukfinance.org.uk/data-and-research | Monthly | PDF/CSV |
| ONS CPI | ons.gov.uk/economy/inflationandpriceindices | Monthly | CSV |
| BoE Base Rate | bankofengland.co.uk/monetary-policy | As-changed | CSV |

## UKAdapter skeleton

```python
_META = MarketMeta(
    code="UK",
    name="United Kingdom",
    currency="GBP",
    currency_symbol="£",
    unit_label="Millions",
    unit_divisor=1.0,
    regulator="Financial Conduct Authority",
    timezone="Europe/London",
    card_types=(
        CardType(code="CC", label="Credit Card", metric="cc_outstanding"),
        CardType(code="DC", label="Debit Card", metric="dc_outstanding"),
    ),
    data_sources=(
        DataSource(name="boe_a52", label="BoE Table A5.2", frequency="monthly", raw_subdir="boe_cards"),
        DataSource(name="boe_base_rate", label="BoE Base Rate", frequency="monthly", raw_subdir="boe_rate"),
        DataSource(name="ons_cpi", label="ONS CPI", frequency="monthly", raw_subdir="ons_cpi"),
    ),
    structural_events=(
        StructuralEvent("2016-06-23", "Brexit referendum", "negative"),
        StructuralEvent("2020-03-23", "COVID lockdown", "negative"),
        StructuralEvent("2022-02-03", "BoE rate hike cycle start", "negative_credit"),
        StructuralEvent("2022-10-01", "Cost of living crisis peak", "negative"),
    ),
)
```

## Key differences from India

| Dimension | India (IN) | UK |
|-----------|-----------|-----|
| Unit | Lakhs (÷10 → millions) | Millions (native) |
| Regulator data | RBI PDF scraping + NPCI | BoE CSV download |
| Debit card dynamics | UPI displacing DC | Contactless/open-banking displacing DC |
| Seasonality | Festive (Oct–Nov) peak | Christmas (Dec) peak, summer dip |
| Interest rate | RBI repo rate | BoE base rate |
| Bank count (modelled) | 12 CC / 16 DC | ~6 CC / ~6 DC |

## Implementation checklist

1. [ ] Create `src/markets/uk.py` with `UKAdapter(MarketAdapter)`
2. [ ] Build BoE scraper (`src/scraper/boe_cards.py`) — CSV download, no Playwright needed
3. [ ] Map BoE columns to MIP schema (`cc_outstanding`, `dc_outstanding` in GBP millions)
4. [ ] Add BoE base rate as regressor (analogous to repo_rate)
5. [ ] Add ONS CPI as regressor candidate
6. [ ] Configure UK bank allowlist from BoE data (Barclays, HSBC, Lloyds, NatWest, Santander, Nationwide)
7. [ ] Set structural events and training start dates
8. [ ] Run `market_pipeline.yml` with `market=UK`
9. [ ] Validate: UK scorecard scores appear, dashboard renders UK data

## Estimated effort

- Adapter + scraper: 1–2 days
- Model tuning + CV: 2–3 days
- Dashboard integration: 1 day
- Total: ~1 week to first UK forecast

## Gate

Per Rahul's working agreement: expansion requires 30 consecutive green
pipeline days on IN + live scorecard + P2 comparison table showing no
regression. Expansion decisions are Rahul's.
