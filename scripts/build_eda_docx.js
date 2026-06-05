const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, Header, Footer, AlignmentType, HeadingLevel, BorderStyle,
  WidthType, ShadingType, VerticalAlign, PageNumber, LevelFormat, PageBreak
} = require("docx");
const fs = require("fs");
const path = require("path");

const data = JSON.parse(fs.readFileSync("reports/eda_data.json", "utf8"));

// ── Colours ───────────────────────────────────────────────────────────────
const BLUE      = "1A3A6A";
const BLUE_LITE = "D5E8F0";
const GREEN     = "1A5C1A";
const GREEN_BG  = "D5EDD5";
const AMBER     = "7A4A00";
const AMBER_BG  = "FFF0CC";
const GREY_BG   = "F0F0F0";
const RED_BG    = "FADDDD";
const WHITE     = "FFFFFF";

const border = (color = "CCCCCC") => ({ style: BorderStyle.SINGLE, size: 1, color });
const borders = (color) => { const b = border(color); return { top:b, bottom:b, left:b, right:b }; };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top:noBorder, bottom:noBorder, left:noBorder, right:noBorder };

const VERDICT_COLOR = { USE: GREEN, CONSIDER: AMBER, WEAK: "888888", drop: "999999" };
const VERDICT_BG    = { USE: GREEN_BG, CONSIDER: AMBER_BG, WEAK: GREY_BG, drop: RED_BG };

// ── Helpers ───────────────────────────────────────────────────────────────
function cell(children, opts = {}) {
  return new TableCell({
    borders: borders(opts.borderColor || "CCCCCC"),
    width: opts.width ? { size: opts.width, type: WidthType.DXA } : undefined,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: Array.isArray(children) ? children : [children],
  });
}

function para(text, opts = {}) {
  return new Paragraph({
    heading: opts.heading,
    alignment: opts.center ? AlignmentType.CENTER : AlignmentType.LEFT,
    spacing: { before: opts.spaceBefore || 0, after: opts.spaceAfter || 60 },
    children: [new TextRun({
      text,
      bold: opts.bold,
      color: opts.color || "222222",
      size: opts.size || 20,
      font: "Arial",
    })],
  });
}

function sigLabel(p) {
  if (p === null || p === undefined) return "N/A";
  if (p < 0.001) return `${p.toFixed(4)} ***`;
  if (p < 0.01)  return `${p.toFixed(4)} **`;
  if (p < 0.05)  return `${p.toFixed(4)} *`;
  return `${p.toFixed(4)}`;
}

function fmtNum(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString("en-IN");
}

function loadImage(imgPath) {
  try { return fs.readFileSync(imgPath); } catch { return null; }
}

// ── Section builder ───────────────────────────────────────────────────────
function buildBankSection(bankData) {
  const blocks = [];

  blocks.push(new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 80 },
    children: [new TextRun({ text: bankData.bank, bold: true, size: 28, font: "Arial", color: BLUE })],
  }));

  // Info bar
  blocks.push(new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2340, 2340, 2340, 2340],
    rows: [new TableRow({ children: [
      cell(para(`Stable window`, { bold:true, size:18, color:"444444" }), { fill: BLUE_LITE, width:2340 }),
      cell(para(`${bankData.window_start} – ${bankData.window_end}`, { size:18 }), { fill: BLUE_LITE, width:2340 }),
      cell(para(`Months of data`, { bold:true, size:18, color:"444444" }), { fill: BLUE_LITE, width:2340 }),
      cell(para(`${bankData.n_months}`, { size:18 }), { fill: BLUE_LITE, width:2340 }),
    ]})],
  }));

  blocks.push(new Paragraph({ spacing: { after: 120 }, children: [] }));

  // Variables
  for (const v of bankData.variables) {
    const vColor = VERDICT_COLOR[v.verdict] || "888888";
    const vBg    = VERDICT_BG[v.verdict]    || GREY_BG;

    // Variable heading row
    blocks.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [6360, 3000],
      rows: [new TableRow({ children: [
        cell(new Paragraph({ children: [
          new TextRun({ text: v.label, bold: true, size: 22, font: "Arial", color: BLUE }),
        ]}), { fill: GREY_BG, width: 6360 }),
        cell(new Paragraph({ alignment: AlignmentType.CENTER, children: [
          new TextRun({ text: v.verdict, bold: true, size: 22, font: "Arial", color: WHITE }),
        ]}), { fill: vColor, width: 3000 }),
      ]})],
    }));

    // Stats table
    blocks.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [1560, 1800, 1560, 1800, 1560, 1080],
      rows: [
        new TableRow({ children: [
          cell(para("Coverage",   { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(`${v.n_valid} / ${v.n_months} months`, { size:17 }), { width:1800 }),
          cell(para("Latest value", { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(fmtNum(v.latest_val), { size:17 }), { width:1800 }),
          cell(para("Trend",  { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(`${v.pct_change > 0 ? "+" : ""}${v.pct_change}%`, { size:17, color: v.pct_change >= 0 ? GREEN : "C00000" }), { width:1080 }),
        ]}),
        new TableRow({ children: [
          cell(para("Spearman ρ", { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(`${v.spearman_rho >= 0 ? "+" : ""}${v.spearman_rho}  (p=${v.spearman_p})`, { size:17 }), { width:1800 }),
          cell(para("Granger L1", { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(sigLabel(v.granger_l1), { size:17, color: v.granger_l1 < 0.05 ? GREEN : "C00000" }), { width:1800 }),
          cell(para("Granger L3", { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(sigLabel(v.granger_l3), { size:17, color: v.granger_l3 < 0.05 ? GREEN : "C00000" }), { width:1080 }),
        ]}),
        new TableRow({ children: [
          cell(para("Granger L6", { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          cell(para(sigLabel(v.granger_l6), { size:17, color: v.granger_l6 < 0.05 ? GREEN : "C00000" }), { width:1800 }),
          cell(para("Why this variable?", { bold:true, size:17, color:"444444" }), { fill: GREY_BG, width:1560 }),
          new TableCell({
            borders: borders("CCCCCC"),
            columnSpan: 3,
            width: { size: 3720, type: WidthType.DXA },
            shading: { fill: WHITE, type: ShadingType.CLEAR },
            margins: { top: 60, bottom: 60, left: 100, right: 100 },
            children: [new Paragraph({ children: [
              new TextRun({ text: v.economic_rationale, size: 17, font: "Arial", italics: true, color: "444444" }),
            ]})],
          }),
        ]}),
      ],
    }));

    // Verdict box
    blocks.push(new Table({
      width: { size: 9360, type: WidthType.DXA },
      columnWidths: [9360],
      rows: [new TableRow({ children: [
        cell(new Paragraph({ children: [
          new TextRun({ text: `Decision: `, bold: true, size: 18, font: "Arial", color: vColor }),
          new TextRun({ text: v.verdict_reason, size: 18, font: "Arial", color: "333333" }),
        ]}), { fill: vBg, width: 9360 }),
      ]})],
    }));

    // Chart image
    if (v.chart_path) {
      const imgData = loadImage(v.chart_path);
      if (imgData) {
        blocks.push(new Paragraph({ spacing: { before: 80 }, children: [
          new ImageRun({
            type: "png",
            data: imgData,
            transformation: { width: 620, height: 213 },
            altText: { title: v.label, description: `${bankData.bank} ${v.label}`, name: v.label },
          }),
        ]}));
      }
    }

    blocks.push(new Paragraph({ spacing: { after: 180 }, children: [] }));
  }

  return blocks;
}

// ── Document assembly ─────────────────────────────────────────────────────
function hrPara() {
  return new Paragraph({
    spacing: { before: 160, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: BLUE } },
    children: [],
  });
}

const allChildren = [];

// Title
allChildren.push(
  new Paragraph({ spacing: { before: 200, after: 60 }, children: [
    new TextRun({ text: "MIP Phase 1 — Bankwise Regressor EDA", bold: true, size: 40, font: "Arial", color: BLUE }),
  ]}),
  new Paragraph({ spacing: { after: 80 }, children: [
    new TextRun({ text: "Per-Bank Variable Analysis: Infrastructure & Transaction Volumes", size: 24, font: "Arial", color: "555555", italics: true }),
  ]}),
  hrPara(),
);

// Methodology
allChildren.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing:{ before:200, after:80 }, children:[
    new TextRun({ text: "Methodology", bold:true, size:32, font:"Arial", color:BLUE }),
  ]}),
  para("Each bank is analysed within its stable regime window only — the post-merger or post-reconstruction period defined in BANK_START_DATES. Pre-merger data is excluded because it represents a structurally different entity.", { size: 19, spaceAfter: 80 }),
  para("Four candidate variables are tested per bank:", { size: 19, spaceAfter: 60 }),
);

const varDescriptions = [
  ["ATMs Off-site / On-site", "Number of ATMs the bank operates outside / inside its own branches."],
  ["PoS Terminals", "Number of point-of-sale machines the bank has deployed at merchant locations."],
  ["CC / DC PoS Txn Volume", "Monthly card swipe count at PoS terminals. For DC, this is declining (UPI displacement)."],
  ["CC / DC ATM Cash Volume", "Monthly ATM cash withdrawal count on the bank's cards."],
];

allChildren.push(new Table({
  width: { size: 9360, type: WidthType.DXA },
  columnWidths: [2800, 6560],
  rows: varDescriptions.map(([name, desc]) => new TableRow({ children: [
    cell(para(name, { bold:true, size:18, color:BLUE }), { fill: BLUE_LITE, width:2800 }),
    cell(para(desc, { size:18 }), { width:6560 }),
  ]})),
}));

allChildren.push(
  new Paragraph({ spacing: { after: 120 }, children: [] }),
  para("Two tests are run for each variable:", { size: 19, spaceAfter: 60 }),
  para("1. Spearman Correlation — measures rank correlation with card outstanding. A high value (>0.8) often reflects two series trending together over time, which can be spurious.", { size: 18, spaceAfter: 40 }),
  para("2. Granger Causality — tests whether past changes in the variable predict future changes in card outstanding, after removing the effect of the outstanding series itself. This is the decisive test. p < 0.05 means the variable adds genuine predictive value.", { size: 18, spaceAfter: 40 }),
  para("Verdicts: USE (p<0.01, ≥36 months)  |  CONSIDER (p<0.05, ≥24 months)  |  WEAK (p<0.10)  |  drop (no signal)", { bold: true, size: 18, spaceAfter: 0 }),
  hrPara(),
);

// CC Section
allChildren.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing:{before:200,after:80}, children:[
    new TextRun({ text: "Credit Cards — Top 10 Banks", bold:true, size:32, font:"Arial", color:BLUE }),
  ]}),
);
for (const bankData of data.cc) {
  allChildren.push(...buildBankSection(bankData));
}

allChildren.push(new Paragraph({ children: [new PageBreak()] }));

// DC Section
allChildren.push(
  new Paragraph({ heading: HeadingLevel.HEADING_1, spacing:{before:200,after:80}, children:[
    new TextRun({ text: "Debit Cards — Top 15 Banks", bold:true, size:32, font:"Arial", color:BLUE }),
  ]}),
);
for (const bankData of data.dc) {
  allChildren.push(...buildBankSection(bankData));
}

// ── Build + save ──────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id:"Heading1", name:"Heading 1", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:32, bold:true, font:"Arial", color:BLUE },
        paragraph:{ spacing:{before:240,after:120}, outlineLevel:0 } },
      { id:"Heading2", name:"Heading 2", basedOn:"Normal", next:"Normal", quickFormat:true,
        run:{ size:28, bold:true, font:"Arial", color:BLUE },
        paragraph:{ spacing:{before:200,after:80}, outlineLevel:1 } },
    ],
  },
  sections: [{
    properties: {
      page: { size:{ width:12240, height:15840 }, margin:{ top:1080, right:1080, bottom:1080, left:1080 } }
    },
    headers: { default: new Header({ children: [new Paragraph({
      alignment: AlignmentType.RIGHT,
      border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space:1 } },
      children: [new TextRun({ text: "MIP Phase 1 — Bankwise Regressor EDA", size:16, font:"Arial", color:"888888" })],
    })]}) },
    footers: { default: new Footer({ children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      border: { top: { style: BorderStyle.SINGLE, size: 4, color: BLUE, space:1 } },
      children: [
        new TextRun({ text: "Page ", size:16, font:"Arial", color:"888888" }),
        new TextRun({ children: [PageNumber.CURRENT], size:16, font:"Arial", color:"888888" }),
        new TextRun({ text: " of ", size:16, font:"Arial", color:"888888" }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], size:16, font:"Arial", color:"888888" }),
      ],
    })]}) },
    children: allChildren,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("reports/bankwise_eda_report.docx", buf);
  console.log("Saved: reports/bankwise_eda_report.docx");
});
