"use strict";
const pptxgen = require("pptxgenjs");

const pres = new pptxgen();

// A0 landscape: 33.11 x 23.39 inches
pres.defineLayout({ name: 'A0_LAND', width: 33.11, height: 23.39 });
pres.layout = 'A0_LAND';
pres.author = 'M. Aytunc Evlice';
pres.title  = 'AC-WLS Power System State Estimation with PMU Integration';

const slide = pres.addSlide();

// ── Palette ──────────────────────────────────────────────────────────────────
const NAVY_DARK  = '0A1628';
const NAVY_MID   = '0D2B5E';
const NAVY_LIGHT = '1A3A6B';
const ORANGE     = 'E87722';
const WHITE      = 'FFFFFF';
const CARD_BG    = 'F4F6FB';
const EQ_BG      = 'EEF2FF';
const EQ_BORDER  = 'C5CAE9';
const TEXT_DARK  = '0D1B3E';
const TEXT_MID   = '334E68';
const TEXT_LIGHT = 'B0C4DE';

// ── Dimensions ────────────────────────────────────────────────────────────────
const W = 33.11, H = 23.39;
const ML = 0.4, MR = 0.4;
const HEADER_H = 2.3;
const FOOTER_H = 0.55;
const CONTENT_Y = HEADER_H + 0.22;
const CONTENT_H = H - HEADER_H - FOOTER_H - 0.32;
const GAP = 0.22;
const COL_W = (W - ML - MR - GAP * 3) / 4;
const colX = i => ML + i * (COL_W + GAP);

// ── Helpers ───────────────────────────────────────────────────────────────────
const makeShadow = () => ({ type: 'outer', color: '000000', blur: 8, offset: 2, angle: 135, opacity: 0.15 });

function card(x, y, w, h) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x, y, w, h, fill: { color: WHITE },
    line: { color: WHITE, width: 0 }, shadow: makeShadow()
  });
}

function sectionHeader(x, y, w, label) {
  const SH = 0.52;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: SH, fill: { color: NAVY_MID }, line: { color: NAVY_MID, width: 0 } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.11, h: SH, fill: { color: ORANGE }, line: { color: ORANGE, width: 0 } });
  slide.addText(label.toUpperCase(), {
    x: x + 0.19, y, w: w - 0.22, h: SH,
    fontSize: 19, bold: true, fontFace: 'Calibri',
    color: WHITE, valign: 'middle', charSpacing: 2, margin: 0
  });
  return SH;
}

function eqBlock(x, y, w, h, lines) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: EQ_BG }, line: { color: EQ_BORDER, width: 1 } });
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w: 0.08, h, fill: { color: NAVY_LIGHT }, line: { color: NAVY_LIGHT, width: 0 } });
  slide.addText(
    lines.map((l, i) => ({ text: l, options: { breakLine: i < lines.length - 1, fontSize: 17, fontFace: 'Courier New', color: NAVY_MID } })),
    { x: x + 0.18, y, w: w - 0.22, h, valign: 'middle', margin: 0 }
  );
}

function subHead(x, y, w, label) {
  slide.addText(label, { x, y, w, h: 0.38, fontSize: 19, bold: true, fontFace: 'Calibri', color: NAVY_MID, margin: 0 });
  return 0.4;
}

function bodyText(x, y, w, h, text, opts = {}) {
  slide.addText(text, { x, y, w, h, fontSize: 17, fontFace: 'Calibri', color: TEXT_MID, valign: 'top', margin: 0, ...opts });
}

function bulletList(x, y, w, h, items, fontSize = 17) {
  slide.addText(
    items.map((item, i) => ({
      text: item.text || item,
      options: {
        bullet: true,
        breakLine: i < items.length - 1,
        fontSize,
        fontFace: 'Calibri',
        color: item.color || TEXT_DARK,
        bold: item.bold || false
      }
    })),
    { x, y, w, h, valign: 'top', margin: 0 }
  );
}

// ── HEADER ────────────────────────────────────────────────────────────────────
slide.background = { color: NAVY_DARK };

slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: HEADER_H, fill: { color: NAVY_MID }, line: { color: NAVY_MID, width: 0 } });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: W, h: 0.11, fill: { color: ORANGE }, line: { color: ORANGE, width: 0 } });

slide.addText('AC-WLS Power System State Estimation with PMU Integration', {
  x: ML, y: 0.18, w: W * 0.68, h: 1.05,
  fontSize: 48, bold: true, fontFace: 'Calibri',
  color: WHITE, valign: 'middle', margin: 0
});
slide.addText('EE574 Real-Time Power System Monitoring  —  Semester Project', {
  x: ML, y: 1.2, w: W * 0.68, h: 0.5,
  fontSize: 24, italic: true, fontFace: 'Calibri',
  color: TEXT_LIGHT, valign: 'middle', margin: 0
});
slide.addText('M. Aytunç Evlice', {
  x: ML, y: 1.7, w: W * 0.5, h: 0.45,
  fontSize: 21, fontFace: 'Calibri', color: ORANGE, valign: 'middle', margin: 0
});

// Key stat boxes in header (right side)
const stats = [
  { val: '14', lbl: 'Buses' },
  { val: '27', lbl: 'States' },
  { val: '72', lbl: 'Measurements' },
  { val: '8/8', lbl: 'Cases Passed' },
];
const SBW = 3.6, SBH = 1.7, SBX0 = W * 0.72, SBY = 0.28, SBGAP = 0.22;
stats.forEach((s, i) => {
  const sx = SBX0 + i * (SBW + SBGAP);
  slide.addShape(pres.shapes.RECTANGLE, { x: sx, y: SBY, w: SBW, h: SBH, fill: { color: NAVY_LIGHT }, line: { color: ORANGE, width: 2 } });
  slide.addText(s.val, { x: sx, y: SBY + 0.1, w: SBW, h: 1.0, fontSize: 42, bold: true, fontFace: 'Calibri', color: ORANGE, align: 'center', valign: 'middle', margin: 0 });
  slide.addText(s.lbl, { x: sx, y: SBY + 1.05, w: SBW, h: 0.5, fontSize: 17, fontFace: 'Calibri', color: WHITE, align: 'center', valign: 'middle', margin: 0 });
});

// ── COLUMN 1: Introduction + Network ─────────────────────────────────────────
const c1x = colX(0);
const introH = CONTENT_H * 0.54;
card(c1x, CONTENT_Y, COL_W, introH);
sectionHeader(c1x, CONTENT_Y, COL_W, 'Introduction');

bodyText(c1x + 0.15, CONTENT_Y + 0.62, COL_W - 0.3, 1.55,
  'State estimation is the core algorithm of every Energy Management System (EMS). ' +
  'It fuses noisy SCADA and PMU measurements into a consistent estimate of the power ' +
  'system operating point (bus voltages and angles), enabling real-time monitoring and control.'
);

bodyText(c1x + 0.15, CONTENT_Y + 2.2, COL_W - 0.3, 0.38,
  'This project implements a full AC-WLS state estimator from scratch:', { color: TEXT_DARK }
);

bulletList(c1x + 0.15, CONTENT_Y + 2.62, COL_W - 0.3, introH - 2.75, [
  'Newton-based WLS iterative solver',
  'Analytical Jacobian (all measurement types)',
  'Transformer π-model (off-nominal tap ratios)',
  'PMU phasor measurement integration',
  'Numerical observability analysis (SVD rank)',
  'Normalized-residual bad data detection',
]);

const netY = CONTENT_Y + introH + GAP;
const netH = CONTENT_H - introH - GAP;
card(c1x, netY, COL_W, netH);
sectionHeader(c1x, netY, COL_W, 'Network');

bulletList(c1x + 0.15, netY + 0.62, COL_W - 0.3, netH - 0.75, [
  'IEEE 14-bus test network (1961 AEP system)',
  '14 buses,  17 branches,  3 transformers',
  '5 generator/slack buses,  9 load buses',
  'Tap ratios: 6→9 (0.978), 6→10 (0.969), 4→12 (0.932)',
  'State vector:  x = [θ₂…θ₁₄, V₁…V₁₄]  →  27 states',
  'SCADA: σV = 0.005 pu,  σPQ = 0.01 pu',
  'PMU: σ = 0.0001 pu  (100× more accurate than SCADA)',
]);

// ── COLUMN 2: Methodology ─────────────────────────────────────────────────────
const c2x = colX(1);
card(c2x, CONTENT_Y, COL_W, CONTENT_H);
sectionHeader(c2x, CONTENT_Y, COL_W, 'Methodology');

let cy = CONTENT_Y + 0.62;
const EX = c2x + 0.15, EW = COL_W - 0.3;

cy += subHead(EX, cy, EW, 'State Vector');
eqBlock(EX, cy, EW, 0.58, ['x = [θ₂ … θN, V₁ … VN]  (2N−1 states, θ₁ = 0)']);
cy += 0.68;

cy += subHead(EX, cy, EW, 'WLS Objective');
eqBlock(EX, cy, EW, 1.0, ['J(x) = (z − h(x))ᵀ W (z − h(x))', 'W = diag(1/σᵢ²)']);
cy += 1.1;

cy += subHead(EX, cy, EW, 'Newton Normal Equations (per iteration)');
eqBlock(EX, cy, EW, 0.58, ['(HᵀWH) Δx = Hᵀ W (z − h(x))']);
cy += 0.68;
bodyText(EX, cy, EW, 0.42, 'H = dh/dx  is the analytical Jacobian (all measurement types).', { italic: true });
cy += 0.52;

cy += subHead(EX, cy, EW, 'Convergence');
bodyText(EX, cy, EW, 0.85, 'Backtracking line search: step α is halved until J decreases, preventing divergence. Warm start from load-flow solution → converges in 2–3 iterations.');
cy += 0.95;

cy += subHead(EX, cy, EW, 'Observability Check');
eqBlock(EX, cy, EW, 0.58, ['rank(H) = n_states  ⟺  system is observable']);
cy += 0.68;
bodyText(EX, cy, EW, 0.65, 'Computed via SVD at flat start. PMUs at buses {2,5,9,10,12,14} restore full observability: rank 19→27, adding 45 redundant degrees of freedom.');
cy += 0.75;

cy += subHead(EX, cy, EW, 'Bad Data Detection (Normalized Residual Test)');
eqBlock(EX, cy, EW, 1.15, ['r̄ᵢ = rᵢ / √Ωᵢᵢ', 'Ω = R − H(HᵀWH)⁻¹Hᵀ', 'Flag if |r̄ᵢ| > 3.0']);
cy += 1.25;
bodyText(EX, cy, EW, 1.0, 'Under clean conditions r̄ᵢ ~ N(0,1). Values exceeding 3.0 (3σ) indicate gross errors. The measurement with the largest |r̄ᵢ| is removed; the estimator re-runs without it.');

// ── COLUMN 3: Results ─────────────────────────────────────────────────────────
const c3x = colX(2);
card(c3x, CONTENT_Y, COL_W, CONTENT_H);
sectionHeader(c3x, CONTENT_Y, COL_W, 'Results — Three Scenarios');

cy = CONTENT_Y + 0.62;
const RX = c3x + 0.15, RW = COL_W - 0.3;

function scenBlock(x, y, w, titleText, titleColor, items) {
  const TH = 0.44;
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h: TH, fill: { color: titleColor }, line: { color: titleColor, width: 0 } });
  slide.addText(titleText, { x: x + 0.12, y, w: w - 0.15, h: TH, fontSize: 18, bold: true, fontFace: 'Calibri', color: WHITE, valign: 'middle', margin: 0 });
  const IH = items.length * 0.37 + 0.18;
  slide.addShape(pres.shapes.RECTANGLE, { x, y: y + TH, w, h: IH, fill: { color: CARD_BG }, line: { color: EQ_BORDER, width: 1 } });
  slide.addText(
    items.map((it, i) => ({
      text: it.text,
      options: { bullet: true, breakLine: i < items.length - 1, fontSize: 17, fontFace: 'Calibri', color: it.bold ? NAVY_MID : TEXT_DARK, bold: it.bold || false }
    })),
    { x: x + 0.12, y: y + TH + 0.08, w: w - 0.2, h: IH - 0.14, valign: 'top', margin: 0 }
  );
  return TH + IH + 0.16;
}

cy += scenBlock(RX, cy, RW, 'Scenario 1 — Original SCADA  (Unobservable)', '7B1818', [
  { text: '19 measurements from measure.dat  (sections 1–6)' },
  { text: 'Jacobian rank: 19/27  (deficit 8 — not observable!)', bold: true },
  { text: 'Unobservable buses: 9, 10, 12, 14' },
  { text: 'WLS objective J = 25,101  (model–data mismatch)', bold: true },
  { text: 'Normalized residuals |r̄| > 10,000 — estimation unreliable' },
]);

cy += scenBlock(RX, cy, RW, 'Scenario 2 — Synthetic SCADA + PMU  (Clean)', '1B5E20', [
  { text: '72 measurements  (58 SCADA + 14 PMU at buses {2,5,9,10,12,14})' },
  { text: 'Jacobian rank: 27/27  ✓  (+45 redundant degrees of freedom)', bold: true },
  { text: 'Converges in 2 iterations' },
  { text: 'RMSE voltage = 0.00055 pu   |   RMSE angle = 0.016°', bold: true },
  { text: 'WLS J = 42.4 ≈ degrees of freedom  ✓' },
  { text: '0 measurements flagged as bad data  ✓' },
]);

cy += scenBlock(RX, cy, RW, 'Scenario 3 — Bad Data Injected & Detected', '0C3D8F', [
  { text: '+0.5 pu gross error injected into Pflow(1→3)' },
  { text: 'WLS J rises to 2,236  (clear alarm)', bold: true },
  { text: 'Largest residual: Pflow(1→3),  |r̄| = 44.9  →  correctly identified  ✓', bold: true },
  { text: 'After removal: RMSE voltage = 0.00054 pu' },
  { text: 'State fully recovered to Scenario 2 accuracy  ✓' },
]);

// Summary table
slide.addText('Summary', { x: RX, y: cy, w: RW, h: 0.38, fontSize: 19, bold: true, fontFace: 'Calibri', color: NAVY_MID, margin: 0 });
cy += 0.4;

const mkHdr = t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY_MID }, fontSize: 15 } });
const tblH = CONTENT_Y + CONTENT_H - cy - 0.05;
slide.addTable([
  [mkHdr('Scenario'), mkHdr('Meas'), mkHdr('Rank'), mkHdr('J_WLS'), mkHdr('BD')],
  ['Sc1: Original SCADA', '19', '19/27', '25,101', 'N/A'],
  ['Sc2: SCADA + PMU (clean)', '72', '27/27', '42.4', '—'],
  ['Sc3: Bad data injected', '72', '27/27', '2,236', 'YES ✓'],
], {
  x: RX, y: cy, w: RW, h: tblH,
  border: { pt: 1, color: EQ_BORDER },
  fill: { color: CARD_BG },
  colW: [RW * 0.43, RW * 0.1, RW * 0.13, RW * 0.18, RW * 0.16],
  fontSize: 15, fontFace: 'Calibri', color: TEXT_DARK,
  align: 'center', valign: 'middle',
});

// ── COLUMN 4: Validation + Conclusions + References ────────────────────────────
const c4x = colX(3);
const VAL_H = CONTENT_H * 0.53;
card(c4x, CONTENT_Y, COL_W, VAL_H);
sectionHeader(c4x, CONTENT_Y, COL_W, 'Multi-Network Validation');

bodyText(c4x + 0.15, CONTENT_Y + 0.62, COL_W - 0.3, 0.75,
  'The estimator was validated on 8 networks from 3 to 40 buses using synthetically generated test cases. All 8/8 cases passed all four checks.'
);

const mkVH = t => ({ text: t, options: { bold: true, color: WHITE, fill: { color: NAVY_MID }, fontSize: 14 } });
const VW = COL_W - 0.3;
const vcols = [VW * 0.25, VW * 0.12, VW * 0.13, VW * 0.30, VW * 0.20];
slide.addTable([
  [mkVH('Network'), mkVH('Buses'), mkVH('States'), mkVH('RMSE_V (pu)'), mkVH('BD')],
  ['3-BUS',   '3',  '5',  '0.00023', 'YES'],
  ['5-BUS',   '5',  '9',  '0.00021', 'YES'],
  ['IEEE-14', '14', '27', '0.00087', 'YES'],
  ['RAND-4',  '4',  '7',  '0.00062', 'YES'],
  ['RAND-7',  '7',  '13', '0.00059', 'YES'],
  ['RAND-10', '10', '19', '0.00134', 'YES'],
  ['RAND-20', '20', '39', '0.00067', 'YES'],
  ['RAND-40', '40', '79', '0.00072', 'YES'],
], {
  x: c4x + 0.15, y: CONTENT_Y + 1.45,
  w: VW, h: VAL_H - 1.58,
  border: { pt: 1, color: EQ_BORDER },
  fill: { color: CARD_BG },
  colW: vcols,
  fontSize: 14, fontFace: 'Calibri', color: TEXT_DARK,
  align: 'center', valign: 'middle',
});

// Conclusions card
const concY = CONTENT_Y + VAL_H + GAP;
const concH = CONTENT_H - VAL_H - GAP - (CONTENT_H * 0.165) - GAP;
card(c4x, concY, COL_W, concH);
sectionHeader(c4x, concY, COL_W, 'Conclusions');

bulletList(c4x + 0.15, concY + 0.62, COL_W - 0.3, concH - 0.75, [
  { text: 'AC-WLS estimator built from scratch with analytical Jacobian and transformer π-model' },
  { text: 'PMU placement at 6 buses restores full observability (rank 19→27, +45 dof)' },
  { text: 'Normalized residual test correctly identifies injected errors (|r̄| = 44.9 vs threshold 3.0)' },
  { text: 'Estimator is robust across 3–40 bus topologies (8/8 PASS)' },
  { text: 'WLS J ≈ degrees of freedom confirms statistical consistency of clean estimates' },
]);

// References card
const refY = concY + concH + GAP;
const refH = CONTENT_H * 0.165;
card(c4x, refY, COL_W, refH);
sectionHeader(c4x, refY, COL_W, 'References');

bulletList(c4x + 0.15, refY + 0.62, COL_W - 0.3, refH - 0.72, [
  'A. Monticelli, State Estimation in Electric Power Systems, Kluwer, 1999',
  'A.J. Wood & B.F. Wollenberg, Power Generation Operation and Control, Wiley, 2013',
  'IEEE CDF format: IEEE Trans. PAS, Vol. PAS-92, No. 6, 1973',
], 15);

// ── FOOTER ────────────────────────────────────────────────────────────────────
const footY = H - FOOTER_H;
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: footY, w: W, h: FOOTER_H, fill: { color: NAVY_MID }, line: { color: NAVY_MID, width: 0 } });
slide.addShape(pres.shapes.RECTANGLE, { x: 0, y: footY, w: W, h: 0.07, fill: { color: ORANGE }, line: { color: ORANGE, width: 0 } });
slide.addText('EE574  |  Real-Time Power System Monitoring  |  AC-WLS State Estimator  |  M. Aytunç Evlice', {
  x: ML, y: footY + 0.1, w: W * 0.72, h: 0.38,
  fontSize: 14, fontFace: 'Calibri', color: TEXT_LIGHT, valign: 'middle', margin: 0
});
slide.addText('github.com/aytuncevlice/EE574_Project', {
  x: W * 0.74, y: footY + 0.1, w: W * 0.24, h: 0.38,
  fontSize: 14, fontFace: 'Calibri', color: ORANGE,
  align: 'right', valign: 'middle', margin: 0
});

// ── Write file ────────────────────────────────────────────────────────────────
const OUT = 'C:/Users/aytu_/Desktop/GitHub/EE574_Project/poster.pptx';
pres.writeFile({ fileName: OUT })
  .then(() => console.log('Saved: ' + OUT))
  .catch(e => { console.error(e); process.exit(1); });
