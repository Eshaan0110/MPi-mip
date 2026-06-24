const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel,
  BorderStyle, WidthType, ShadingType, PageNumber, PageBreak,
  TableOfContents
} = require("docx");

const border = { style: BorderStyle.SINGLE, size: 1, color: "999999" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 100, right: 100 };

function text(t, opts = {}) {
  return new TextRun({ text: t, font: "Arial", size: opts.size || 22, ...opts });
}

function para(t, opts = {}) {
  const runs = typeof t === "string" ? [text(t, opts)] : t;
  return new Paragraph({
    children: runs,
    spacing: { after: opts.after || 120, before: opts.before || 0, line: opts.line || 276 },
    alignment: opts.alignment || AlignmentType.LEFT,
    ...(opts.heading ? { heading: opts.heading } : {}),
    ...(opts.bullet ? { numbering: { reference: "bullets", level: opts.bulletLevel || 0 } } : {}),
  });
}

function h1(t) {
  return para(t, { heading: HeadingLevel.HEADING_1, size: 28, bold: true, after: 200, before: 300 });
}
function h2(t) {
  return para(t, { heading: HeadingLevel.HEADING_2, size: 24, bold: true, after: 160, before: 240 });
}
function h3(t) {
  return para(t, { heading: HeadingLevel.HEADING_3, size: 22, bold: true, after: 120, before: 200 });
}

function bullet(t, level = 0) {
  return para(t, { bullet: true, bulletLevel: level, size: 22 });
}

function makeCell(content, opts = {}) {
  const runs = typeof content === "string"
    ? [text(content, { size: 20, bold: opts.bold })]
    : content;
  return new TableCell({
    borders,
    width: { size: opts.width || 2340, type: WidthType.DXA },
    margins: cellMargins,
    shading: opts.header ? { fill: "F2F2F2", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ children: runs, alignment: opts.align || AlignmentType.LEFT })],
  });
}

function makeTable(headers, rows, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        children: headers.map((h, i) => makeCell(h, { width: colWidths[i], bold: true, header: true })),
      }),
      ...rows.map((row) =>
        new TableRow({
          children: row.map((cell, i) => makeCell(cell, { width: colWidths[i] })),
        })
      ),
    ],
  });
}

function spacer() { return para("", { after: 80 }); }
function pageBreak() { return new Paragraph({ children: [new PageBreak()] }); }

// ─── Build Document ────────────────────────────────────────────────────────

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 300, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [
        { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "-", alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 1440, hanging: 360 } } } },
      ]},
    ],
  },
  sections: [
    // ═══════════════════ COVER PAGE ═══════════════════
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
      },
      children: [
        para("", { after: 3000 }),
        para("MPi Market Intelligence Platform (MIP)", { alignment: AlignmentType.CENTER, size: 36, bold: true, after: 200 }),
        para("Phase 2 Final Project Report", { alignment: AlignmentType.CENTER, size: 28, after: 600 }),
        para("Automated Forecasting Platform for India's Credit Card, Debit Card, and Digital Payments Market", { alignment: AlignmentType.CENTER, size: 22, italics: true, after: 1200 }),
        para([text("Prepared by: ", { size: 22 }), text("Eshaan Adyanthaya", { size: 22, bold: true })], { alignment: AlignmentType.CENTER, after: 120 }),
        para([text("Organisation: ", { size: 22 }), text("MPi — Axiom Programme", { size: 22, bold: true })], { alignment: AlignmentType.CENTER, after: 120 }),
        para([text("Date: ", { size: 22 }), text("June 2026", { size: 22, bold: true })], { alignment: AlignmentType.CENTER, after: 120 }),
        para([text("Audit Score: ", { size: 22 }), text("88/100 (AXIOM Round 4)", { size: 22, bold: true })], { alignment: AlignmentType.CENTER, after: 120 }),
        para("Live Dashboard: https://web-mocha-kappa-71.vercel.app", { alignment: AlignmentType.CENTER, size: 22, after: 120 }),
      ],
    },

    // ═══════════════════ MAIN CONTENT ═══════════════════
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } },
      },
      headers: {
        default: new Header({ children: [
          para("MPi — MIP Phase 2 Project Report", { alignment: AlignmentType.RIGHT, size: 18, italics: true }),
        ]}),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [text("Page ", { size: 18 }), new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18 })],
        })] }),
      },
      children: [

        h1("Table of Contents"),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
        pageBreak(),

        // ═══════════════════ 1 ═══════════════════
        h1("1. Executive Summary"),
        para("The Market Intelligence Platform (MIP) is an end-to-end automated forecasting system that predicts India's credit card outstanding, debit card outstanding, and digital payment volumes 24 months into the future. It scrapes official data from the Reserve Bank of India (RBI) and National Payments Corporation of India (NPCI) every month, trains machine learning models, and publishes live forecasts through a web dashboard."),
        para("In simple terms: MIP answers the question \"How many credit cards and debit cards will India have in 2 years, and how much will people spend on them?\"", { italics: true }),

        h2("Key Achievements"),
        bullet("Ensemble forecasting engine combining three models (Prophet + ARIMA + ETS) with mathematically optimised weights per card type"),
        bullet("Credit card outstanding forecast accuracy: approximately 3.5% average error (MAPE) on unseen data"),
        bullet("Debit card outstanding forecast accuracy: approximately 5.7% average error after lag optimisation"),
        bullet("Approximately 80 individual bank-level models for granular per-bank predictions"),
        bullet("Fully automated monthly pipeline: scrape data, retrain models, update database, refresh dashboard"),
        bullet("Live web dashboard deployed at https://web-mocha-kappa-71.vercel.app"),
        bullet("Passed rigorous quantitative audit scoring 88/100 across 12 diagnostic test suites"),
        pageBreak(),

        // ═══════════════════ 2 ═══════════════════
        h1("2. Problem Statement and Business Context"),
        para("India's digital payments ecosystem is one of the fastest growing in the world. Credit card outstanding crossed 1,100 lakh (110 million) cards in 2025, while UPI processes over 18 billion transactions per month. Understanding where these numbers are heading is critical for:"),
        bullet("Banks planning credit card issuance targets and risk exposure"),
        bullet("Payment networks (Visa, Mastercard, RuPay) planning infrastructure investment"),
        bullet("Regulators assessing systemic credit risk and digital adoption trends"),
        bullet("Investors evaluating fintech and banking stocks"),
        spacer(),
        para("Before MIP, forecasting these numbers required manual analysis of RBI spreadsheets, ad-hoc statistical models, and subjective expert judgment. The process was slow (weeks), unreproducible, and could not incorporate the multiple structural breaks that India's payment landscape has experienced (demonetisation in 2016, COVID in 2020, UPI's explosive growth)."),
        para("MIP automates the entire process: from downloading the latest RBI data to publishing forecasts that anyone in the organisation can view on their phone."),
        pageBreak(),

        // ═══════════════════ 3 ═══════════════════
        h1("3. Data Sources"),
        para("All data comes from official government sources. No third-party or commercial data is used."),

        h2("3.1 RBI Payment System Indicators (PSI)"),
        para("This is the primary dataset. Published monthly by RBI, it contains India-total numbers for:"),
        bullet("Credit cards outstanding (how many cards exist in the country)"),
        bullet("Debit cards outstanding"),
        bullet("Credit card transaction volume (how many transactions per month)"),
        bullet("Debit card transaction volume"),
        bullet("POS terminal counts, ATM counts, and other infrastructure metrics"),
        para("Data range: April 2004 to present (updated monthly, typically with a 1-2 month lag)."),

        h2("3.2 RBI Bankwise Data"),
        para("Also from RBI's DBIE portal, this provides the same metrics broken down by individual bank. For example, how many credit cards does HDFC Bank have versus SBI versus ICICI. Data range: April 2011 to present."),

        h2("3.3 NPCI UPI Statistics"),
        para("UPI transaction volumes and values published by NPCI. This matters because UPI is actively displacing debit card transactions at merchant terminals. The debit card model needs to account for this displacement effect. Data range: April 2016 to present."),

        h2("3.4 RBI Repo Rate"),
        para("The RBI's policy interest rate. This is used as a predictor for credit card outstanding because when RBI raises rates, banks become more cautious about issuing unsecured credit (credit cards are unsecured lending). Data range: January 2007 to present."),

        h2("3.5 Why These Specific Datasets?"),
        para("We chose these datasets because they are the most authoritative sources for India's card and digital payments data. RBI is the regulator — all banks must report to RBI, so the PSI data represents the true market total, not estimates. NPCI runs the UPI network, so their volume numbers are exact, not surveyed."),
        pageBreak(),

        // ═══════════════════ 4 ═══════════════════
        h1("4. Data Pipeline"),

        h2("4.1 Automated Scraping"),
        para("MIP runs automated scrapers that download the latest data files from RBI and NPCI websites every month. This is done using a tool called Playwright, which acts like a robot browser — it navigates to the RBI website, clicks through to the right page, and downloads the Excel file, exactly as a human would. We built three scrapers:"),
        bullet("RBI PSI scraper — downloads the monthly Payment System Indicators Excel file"),
        bullet("RBI Bankwise scraper — downloads the per-bank ATM/card/POS Excel files"),
        bullet("NPCI UPI scraper — downloads UPI monthly statistics"),
        spacer(),
        para("Why automate this? Because RBI publishes data in messy Excel files with inconsistent formats across years. A manual process would require someone to download the file, find the right sheet, copy the right cells, and paste them into the analysis spreadsheet every month. This is error-prone and time-consuming. The scraper does it in under 60 seconds."),

        h2("4.2 Data Ingestion and Cleaning"),
        para("The raw Excel files from RBI are not analysis-ready. They have:"),
        bullet("Multiple sheets with different layouts across years"),
        bullet("Merged cells, footnotes, and headers embedded in data rows"),
        bullet("Unit changes (some years report in lakhs, others in absolute numbers)"),
        bullet("Missing months and definitional changes (e.g., RBI changed how it counts \"outstanding\" cards in November 2019)"),
        spacer(),
        para("The ingestion module parses all of this into a single clean table (we call it the \"master training\" dataset) stored in Parquet format — a fast, compressed file format used in data engineering. The master training dataset has one row per month and columns for every metric (CC outstanding, DC outstanding, repo rate, UPI volume, etc.)."),

        h2("4.3 Level Splicing"),
        para("In November 2019, RBI changed the definition of \"cards outstanding\" in the PSI data. The pre-November 2019 numbers and post-November 2019 numbers are not directly comparable — there is a level shift. If we just stitch them together, the model would see a sudden jump that is not real growth."),
        para("Our solution is called \"level splicing\": we calculate the ratio between the old and new series during the overlap period (when RBI briefly published both formats) and use that ratio to rescale the historical data to be consistent with the current definition. Think of it as converting old-definition numbers to new-definition numbers so the entire series is apples-to-apples."),
        pageBreak(),

        // ═══════════════════ 5 ═══════════════════
        h1("5. Forecasting Models"),
        para("MIP uses three different statistical/ML models and combines them into an \"ensemble\" — a weighted average that is more accurate than any single model alone. Think of it as getting three expert opinions and weighting them based on how accurate each expert has been historically."),

        h2("5.1 Prophet (Meta/Facebook)"),
        para("Prophet is an open-source forecasting model developed by Meta (formerly Facebook). It was designed for business time series that have strong seasonal patterns and multiple structural breaks. It works by decomposing a time series into three parts:"),
        bullet("Trend: the long-term direction (is the number going up or down over years?)"),
        bullet("Seasonality: repeating patterns within a year (e.g., credit card applications tend to spike before festivals)"),
        bullet("Events/holidays: one-off disruptions like COVID lockdowns or demonetisation"),
        spacer(),
        para("Why we chose Prophet:", { bold: true }),
        bullet("It handles structural breaks well — India's payment data has multiple regime changes"),
        bullet("It can incorporate external predictors (like the RBI repo rate) as \"regressors\""),
        bullet("It gives uncertainty bands (confidence intervals) around its predictions"),
        bullet("It is battle-tested on millions of time series at Facebook/Meta"),

        h2("5.2 ARIMA(1,1,1)"),
        para("ARIMA stands for Auto-Regressive Integrated Moving Average. It is a classical statistical model that has been the gold standard for time series forecasting for decades. The numbers (1,1,1) mean:"),
        bullet("1 autoregressive term: the forecast depends on the previous month's value"),
        bullet("1 differencing step: we forecast the change month-to-month, not the absolute level"),
        bullet("1 moving average term: the forecast also considers the previous month's forecast error"),
        spacer(),
        para("Why we include ARIMA: It is fundamentally different from Prophet. Prophet fits a smooth curve through the data; ARIMA focuses on short-term momentum. When Prophet overshoots, ARIMA often undershoots, and vice versa. Combining them cancels out individual errors."),

        h2("5.3 Damped Exponential Smoothing (ETS)"),
        para("ETS (Error, Trend, Seasonality) is another classical model. The \"damped\" version gradually flattens the trend over time, preventing runaway extrapolation. If the number of credit cards has been growing at 5% per month, damped ETS will predict that the growth rate gradually slows down rather than continuing at 5% forever."),
        para("Why we include ETS: It provides a conservative anchor. If Prophet and ARIMA both predict aggressive growth, ETS pulls the ensemble forecast back toward a more cautious projection. This is valuable for risk management."),

        h2("5.4 Ensemble: Combining the Three Models"),
        para("Rather than picking one model, we take a weighted average of all three. The weights are not arbitrary — they were mathematically optimised using cross-validation (testing on data the models have never seen):"),
        spacer(),
        makeTable(
          ["Card Type", "Prophet", "ARIMA", "ETS", "Combined MAPE"],
          [
            ["Credit Card", "35%", "39%", "26%", "~3.5%"],
            ["Debit Card", "35%", "65%", "0%", "~5.7%"],
          ],
          [2000, 1600, 1600, 1600, 2560]
        ),
        spacer(),
        para("For debit cards, ETS gets 0% weight. This means the optimiser found that ETS adds no value for DC — ARIMA alone (with Prophet) does better. This is because the debit card series has a complex structural break (UPI displacement starting 2022) that damped ETS cannot capture well."),
        para("A 3.5% MAPE means that on average, our credit card forecast is off by only 3.5% from the actual number. For a market of 1,100+ lakh cards, that is within approximately 38 lakh cards of the real value."),
        pageBreak(),

        // ═══════════════════ 6 ═══════════════════
        h1("6. Regressors"),
        para("Regressors are external variables that help explain why the number we are forecasting goes up or down. Instead of just looking at the trend and seasonality of credit cards, we also tell the model \"look at what the RBI interest rate is doing — that affects credit card issuance.\""),

        h2("6.1 Credit Card: Repo Rate (lagged 9 months)"),
        para("The RBI repo rate is the interest rate at which RBI lends to banks. When it goes up, banks face higher borrowing costs and tighten credit card issuance. When it comes down, banks loosen up and issue more cards."),
        para("This effect is not instant — our analysis found it takes about 9 months for an RBI rate change to show up in credit card outstanding numbers. This 9-month lag was selected objectively: we tested every lag from 0 to 12 months using 18 separate cross-validation windows, and lag 9 was selected as optimal in 12 out of 18 windows (67% of the time)."),

        h2("6.2 Debit Card: Transaction Volume (lagged 4 months)"),
        para("This is the number of debit card transactions per month. More transactions signal higher usage, which eventually drives banks to issue more cards. The 4-month lag reflects the reporting and settlement delay between a transaction happening and RBI capturing it in the outstanding data. Lag 4 gave 5.7% error versus 7.6% at lag 0 — a 1.9 percentage point improvement."),

        h2("6.3 Debit Card: POS Volume (no lag)"),
        para("This measures how many debit card swipes happen at physical merchant terminals. It is declining because UPI is replacing debit cards at shops. Including this gives the model a direct signal of the displacement happening — as POS volume drops, the model adjusts its debit card forecast downward."),
        pageBreak(),

        // ═══════════════════ 7 ═══════════════════
        h1("7. Structural Events"),
        para("India's payment landscape has been disrupted by several major events that cannot be predicted by trend or seasonality alone. We explicitly code these into the model:"),
        spacer(),
        makeTable(
          ["Event", "Date", "What Happened", "How We Model It"],
          [
            ["Demonetisation", "Nov 2016", "Rs 500/1000 notes banned. Forced cashless adoption. CC and DC issuance spiked.", "Changepoint (permanent growth rate shift)"],
            ["COVID Lockdown", "Apr-May 2020", "Nationwide lockdown. Card issuance froze for 2 months.", "Pulse dummy (ignore these 2 abnormal months)"],
            ["PMJDY / Jan Dhan", "Aug 2014", "Government opened 380M+ bank accounts with free RuPay debit cards.", "Changepoint (DC only)"],
            ["UPI Inflection", "Jan 2022", "UPI merchant payments overtook debit card POS. DC growth reversed.", "Changepoint (DC only)"],
            ["RBI Credit Tightening", "Nov 2023", "RBI raised risk weights on unsecured consumer credit.", "Step dummy (CC only, permanent slowdown)"],
          ],
          [1800, 1200, 3200, 3160]
        ),
        spacer(),
        para("Without these event codings, the models would either overfit to the disruption period or miss the structural shift entirely."),
        pageBreak(),

        // ═══════════════════ 8 ═══════════════════
        h1("8. Confidence Intervals"),
        para("Every forecast comes with a range showing the best and worst cases. For example: \"We forecast 1,200 lakh credit cards in March 2027, with a 90% confidence interval of 1,189 to 1,223 lakh.\" This means we are 90% confident the real number will fall within that range."),

        h2("8.1 Why Not Use Default Confidence Intervals?"),
        para("Most forecasting models compute confidence intervals by assuming the forecast errors follow a bell-curve (normal distribution). Our audit found that this assumption does not hold for our data — the errors are skewed and have heavy tails. Using the default CIs would give misleadingly narrow ranges."),

        h2("8.2 Conformal Prediction Intervals"),
        para("Instead, we use a technique called conformal prediction:"),
        bullet("We run the model many times on different historical windows (walk-forward cross-validation)"),
        bullet("Each time, we record how far off the prediction was from reality (the residual)"),
        bullet("We collect all these errors and take the 5th percentile (worst undershoot) and 95th percentile (worst overshoot)"),
        bullet("These actual historical errors become the confidence band width"),
        spacer(),
        para("This approach is distribution-free — we make no assumption about the shape of the errors. The intervals are based on how wrong the model actually was in the past, not how wrong a theoretical distribution says it should be."),
        pageBreak(),

        // ═══════════════════ 9 ═══════════════════
        h1("9. Bank-Level Forecasts"),
        para("In addition to India-total forecasts, MIP forecasts credit and debit card outstanding for approximately 80 individual banks."),

        h2("9.1 Model Selection Per Bank"),
        bullet("Large/complex banks (HDFC, SBI, ICICI, Axis, etc.) get Prophet models with logistic growth caps to prevent unrealistic extrapolation"),
        bullet("Small/stable banks get simpler Holt-Winters ETS models (they do not have enough data for Prophet)"),
        bullet("Merged banks (e.g., Syndicate Bank merged into Canara Bank) train only on post-merger data"),

        h2("9.2 Reconciliation"),
        para("The sum of all bank forecasts should approximately equal the India-total forecast from the aggregate model. We compute a \"residual\" bucket: the difference between the PSI total and the sum of the top banks. This residual captures all the small banks we do not model individually."),
        para("Median accuracy: CC banks approximately 4-6% MAPE, DC banks approximately 6-9% MAPE."),
        pageBreak(),

        // ═══════════════════ 10 ═══════════════════
        h1("10. Cross-Validation"),
        para("Cross-validation is how we measure whether the model actually works on future data it has never seen."),

        h2("10.1 Walk-Forward Design"),
        bullet("Start with 48 months of history (the initial window)"),
        bullet("Train the model on those 48 months"),
        bullet("Forecast the next 6 months and compare to what actually happened"),
        bullet("Slide the window forward by 6 months and repeat"),
        bullet("Do this across the entire historical period"),
        spacer(),
        para("This gives us 10-15 test windows. The average error across all windows is the MAPE we report."),

        h2("10.2 Accuracy Results"),
        spacer(),
        makeTable(
          ["Model", "Metric", "CV MAPE", "Interpretation"],
          [
            ["Aggregate CC", "CC Outstanding", "~3.5%", "Off by ~38 lakh cards on avg (of 1,100 lakh)"],
            ["Aggregate DC", "DC Outstanding", "~5.7%", "Off by ~590 lakh cards on avg (of 10,400 lakh)"],
            ["CC Txn Volume", "Monthly CC Transactions", "~13.6%", "Higher error due to COVID volatility"],
            ["DC Txn Volume", "Monthly DC Transactions", "~7%", "Trained on post-2022 decline only"],
            ["UPI Volume", "Monthly UPI Transactions", "~12.3%", "Hypergrowth series, harder to predict"],
          ],
          [1800, 2200, 1200, 4160]
        ),
        pageBreak(),

        // ═══════════════════ 11 ═══════════════════
        h1("11. Key Design Decisions"),

        h2("11.1 CC Training Starts January 2013"),
        para("Before 2013, India's credit card market was recovering from the 2008 Global Financial Crisis. Including that data would confuse the model — it would try to average a declining market (2008-2012) with a growing market (2013+). Since our goal is to forecast forward in a growth regime, we only train on the growth period."),

        h2("11.2 DC Volume Training Starts January 2022"),
        para("Debit card transactions were growing until 2019, then started declining as UPI replaced them. Mixing a growth regime with a decline regime gives the model contradictory signals — it produces absurdly wide confidence intervals (one test window gave a CI range of 14 to 846 lakh, a 35x range, completely useless). Training only on the post-2022 decline regime gives dramatically tighter, more useful forecasts."),

        h2("11.3 Ensemble Over Single Model"),
        para("No single model consistently wins on all series. ARIMA dominates for debit cards, but Prophet is stronger for credit cards where regressors and event dummies matter. The weighted ensemble reduces variance — when one model is wrong, the others compensate."),

        h2("11.4 Conformal CIs Over Parametric"),
        para("Residuals from our model are not normally distributed (the audit confirmed this). Parametric confidence intervals would be overconfident. Conformal intervals use actual historical errors — no distributional assumptions required."),

        h2("11.5 Keeping the DC Volume Regressor"),
        para("When we remove debit card transaction volume as a regressor, the MAPE actually improves by 1.3 percentage points. So why keep it? Because it is the core business driver of debit card issuance. The lag fix (0 to 4 months) recovered most of the accuracy penalty. A model that does not include transaction volume as a driver of card outstanding is not credible, even if the statistical fit is marginally better without it."),
        pageBreak(),

        // ═══════════════════ 12 ═══════════════════
        h1("12. Web Dashboard"),
        para("The forecasts are published through a live web application built with Next.js 14, Recharts (for charts), and Tailwind CSS (for styling). It reads data from a Supabase PostgreSQL cloud database."),
        para("Live at: https://web-mocha-kappa-71.vercel.app", { bold: true }),
        spacer(),
        makeTable(
          ["Page", "What It Shows"],
          [
            ["Dashboard (Home)", "India-level KPI cards, 24-month forecast charts with 90% CI bands, summary table"],
            ["Bank Explorer", "Per-bank forecasts for ~80 banks. Toggle CC/DC. Select any bank and month."],
            ["Data Status", "Scraper history: when data was last refreshed, record counts, errors"],
            ["Model Performance", "Every model's CV MAPE, colour-coded by accuracy tier"],
            ["About", "Methodology, data sources, accuracy summary, limitations"],
          ],
          [2200, 7160]
        ),
        pageBreak(),

        // ═══════════════════ 13 ═══════════════════
        h1("13. Automated Monthly Pipeline"),
        para("The entire system is automated using GitHub Actions. On the 15th of every month, a four-step pipeline runs:"),
        spacer(),
        makeTable(
          ["Step", "Job", "What It Does", "Duration"],
          [
            ["1", "Scrape", "Downloads latest RBI/NPCI Excel files", "~2 min"],
            ["2", "Train", "Runs ingestion + all models", "~20 min"],
            ["3", "Sync", "Pushes forecasts to Supabase cloud database", "~1 min"],
            ["4", "Notify", "Reports success/failure", "~10 sec"],
          ],
          [600, 1200, 5360, 1200]
        ),
        spacer(),
        para("This pipeline can also be triggered manually. The entire process requires zero human intervention."),
        pageBreak(),

        // ═══════════════════ 14 ═══════════════════
        h1("14. Technology Stack"),
        spacer(),
        makeTable(
          ["Layer", "Technology", "Why"],
          [
            ["Language (ML)", "Python 3.12", "Standard for data science"],
            ["Package Manager", "uv", "10-100x faster than pip; reproducible"],
            ["Forecasting", "Prophet + statsmodels", "Prophet for regressors; statsmodels for ARIMA/ETS"],
            ["Local Storage", "Parquet", "Fast, compressed columnar format"],
            ["Cloud Database", "Supabase (PostgreSQL)", "Free tier; REST API; real-time capable"],
            ["Web Framework", "Next.js 14 (React)", "Server-side rendering; easy deployment"],
            ["Charts", "Recharts", "React-native; supports CI bands"],
            ["Styling", "Tailwind CSS", "Utility-first; consistent"],
            ["Hosting", "Vercel", "Zero-config for Next.js; automatic HTTPS"],
            ["CI/CD", "GitHub Actions", "Free; runs monthly pipeline"],
            ["Scraping", "Playwright", "Handles JS-heavy government portals"],
          ],
          [1800, 2800, 4760]
        ),
        pageBreak(),

        // ═══════════════════ 15 ═══════════════════
        h1("15. Audit Results"),
        para("The system underwent a rigorous quantitative audit scoring 88/100 across 12 diagnostic test suites:"),
        spacer(),
        makeTable(
          ["Area", "Score", "Status"],
          [
            ["Stationarity and Data Properties", "9/10", "Pass"],
            ["Residual Diagnostics", "5/10", "Known Prophet limitation; mitigated by conformal CIs"],
            ["Regressor Validation (CC)", "9/10", "Pass; repo rate lag-9 validated"],
            ["Regressor Validation (DC)", "9/10", "Pass; DC vol lag fixed 0 to 4"],
            ["Structural Events", "9/10", "Pass"],
            ["CI Calibration", "8/10", "Pass; conformal intervals implemented"],
            ["Alternative Models", "10/10", "Pass; ensemble with optimised weights"],
            ["Scenario Testing", "9/10", "Pass; 4 CC + 3 DC scenarios"],
            ["Lag Sensitivity", "9/10", "Pass"],
            ["Horizon Drift", "8/10", "Acceptable"],
            ["Data Integrity", "8/10", "No leakage"],
            ["Cross-Validation Design", "8/10", "Pass"],
          ],
          [3200, 800, 5360]
        ),
        spacer(),
        para("The main area of weakness (5/10 on residual diagnostics) is a known characteristic of Prophet: in-sample residuals are autocorrelated because Prophet's piecewise-linear trend deliberately undersmooths. This does not affect out-of-sample forecast accuracy, and is addressed through conformal prediction intervals."),
        pageBreak(),

        // ═══════════════════ 16 ═══════════════════
        h1("16. Scenario Analysis"),
        para("For credit cards, MIP runs the forecast under four RBI interest rate scenarios:"),
        spacer(),
        makeTable(
          ["Scenario", "Repo Rate", "What It Simulates"],
          [
            ["Base Case", "6.25%", "Current rate maintained"],
            ["Dovish (-100bp)", "5.25%", "RBI cuts rate to boost growth; CC issuance accelerates"],
            ["Hawkish (+100bp)", "7.25%", "RBI raises rate to fight inflation; CC growth slows"],
            ["Emergency Tightening", "8.00%", "Extreme rate hike; CC issuance drops sharply"],
          ],
          [2200, 1400, 5760]
        ),
        spacer(),
        para("These scenarios isolate the impact of rate changes on credit card outstanding by overriding the repo rate regressor while keeping everything else constant."),
        pageBreak(),

        // ═══════════════════ 17 ═══════════════════
        h1("17. Limitations and Future Work"),

        h2("17.1 Current Limitations"),
        bullet("No forward-looking variables: The model uses only historical data and current-state regressors. It does not incorporate news sentiment, bank announcements, or policy signals that could affect future issuance."),
        bullet("Residual autocorrelation: Prophet's in-sample residuals are correlated. Mitigated by conformal CIs but not eliminated."),
        bullet("Debit card uncertainty: The DC series has a complex ongoing structural break (UPI displacement). Longer-horizon DC forecasts carry higher uncertainty."),
        bullet("Small bank models: Banks with limited history use simpler ETS models that cannot incorporate external regressors."),
        bullet("Scraper fragility: RBI's website changes periodically. Scrapers may break when the portal is redesigned."),

        h2("17.2 Planned Enhancements"),
        bullet("LLM-powered news monitoring: Integrate an AI model that reads bank/RBI news feeds in real time and flags events that could impact forecasts"),
        bullet("Forward-looking regressors: Incorporate RBI policy signals, credit bureau data, and digital adoption proxies"),
        bullet("Transaction value forecasting (not just volume)"),
        bullet("Merchant acceptance modelling"),
        bullet("Rural vs urban segmentation"),
        pageBreak(),

        // ═══════════════════ 18 ═══════════════════
        h1("18. Project Structure"),
        spacer(),
        makeTable(
          ["Directory / File", "Purpose"],
          [
            ["src/ingestion/", "Data parsers for RBI Excel, NPCI JSON, CPI, repo rate, level splicing"],
            ["src/modelling/model_config.py", "Single source of truth for all model parameters, events, regressors"],
            ["src/modelling/aggregate_model.py", "Ensemble forecasting engine (Prophet + ARIMA + ETS + conformal CIs)"],
            ["src/modelling/bank_model.py", "Per-bank Prophet/ETS models (~80 banks)"],
            ["src/modelling/bank_config.py", "Bank lists, growth caps, model type flags"],
            ["src/modelling/txn_volume_model.py", "CC/DC/UPI transaction volume forecasting"],
            ["src/modelling/data_prep.py", "Feature engineering, lag computation, master dataset builder"],
            ["src/scraper/", "Automated RBI/NPCI data downloaders (Playwright-based)"],
            ["scripts/sync_to_supabase.py", "Pushes forecast Parquet files to Supabase cloud database"],
            ["web/", "Next.js 14 web dashboard (React + Recharts + Tailwind CSS)"],
            ["supabase/migrations/", "PostgreSQL database schema (11 tables)"],
            [".github/workflows/", "Monthly CI/CD pipeline definition"],
            ["run_pipeline.py", "One-command pipeline runner"],
            ["experiments/axiom_audit/", "Audit diagnostic scripts and reports"],
          ],
          [3200, 6160]
        ),
        pageBreak(),

        // ═══════════════════ 19 ═══════════════════
        h1("19. Conclusion"),
        para("MIP Phase 2 delivers a production-grade, fully automated forecasting system for India's card and digital payments market. The system:"),
        bullet("Scrapes official RBI/NPCI data automatically every month"),
        bullet("Trains an ensemble of three complementary forecasting models"),
        bullet("Produces 24-month forecasts with honest, distribution-free confidence intervals"),
        bullet("Covers both India-total and approximately 80 individual banks"),
        bullet("Publishes results through a live web dashboard"),
        bullet("Passed a rigorous quantitative audit (88/100) covering 12 diagnostic areas"),
        bullet("Runs entirely on free/open-source tooling with zero ongoing infrastructure cost"),
        spacer(),
        para("The platform provides a reliable, data-driven foundation for MPi's market intelligence offerings. With the planned additions of news-based sentiment monitoring and forward-looking variables, it will evolve from a reactive forecasting tool to a proactive market intelligence platform."),
        spacer(),
        para("--- End of Report ---", { alignment: AlignmentType.CENTER, italics: true }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  const outPath = process.argv[2] || "MIP_Phase2_Final_Report.docx";
  fs.writeFileSync(outPath, buffer);
  console.log(`Report generated: ${outPath} (${(buffer.length / 1024).toFixed(0)} KB)`);
});
