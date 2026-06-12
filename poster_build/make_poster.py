"""
Assemble poster.pptx (landscape A0) from poster_build/figures/*.png

Run:  python poster_build/make_poster.py
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, 'poster_build', 'figures')
OUT = os.path.join(ROOT, 'poster.pptx')

# palette (matches figures / GUI)
RED    = RGBColor(0xC0, 0x39, 0x2B)
RED_DK = RGBColor(0x92, 0x2B, 0x21)
DARK   = RGBColor(0x1C, 0x28, 0x33)
GRAY   = RGBColor(0x71, 0x7D, 0x7E)
LIGHT  = RGBColor(0xF4, 0xF6, 0xF8)
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)

PAGE_W, PAGE_H = 33.1, 23.4
MARGIN = 0.5
N_COLS = 4
COL_GAP = 0.45
COL_W = (PAGE_W - 2 * MARGIN - (N_COLS - 1) * COL_GAP) / N_COLS
HDR_H = 2.95
FOOT_H = 0.62
BODY_TOP = HDR_H + 0.38
BODY_BOT = PAGE_H - MARGIN - FOOT_H - 0.25

FONT = 'Calibri'
FONT_HDR = 'Georgia'

prs = Presentation()
prs.slide_width = Inches(PAGE_W)
prs.slide_height = Inches(PAGE_H)
slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank


def rect(x, y, w, h, fill, line=None):
    from pptx.enum.shapes import MSO_SHAPE
    sp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
                                Inches(w), Inches(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1)
    sp.shadow.inherit = False
    return sp


def textbox(x, y, w, h, runs_per_par, align=PP_ALIGN.LEFT,
            anchor=MSO_ANCHOR.TOP, space_after=6, line_spacing=1.12):
    """runs_per_par: list of paragraphs; each paragraph is a list of
    (text, size, bold, color, italic) run tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, runs in enumerate(runs_per_par):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        for text, size, bold, color, italic in runs:
            r = p.add_run()
            r.text = text
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.color.rgb = color
    return tb


def body(text, size=18, color=DARK, bold=False, italic=False):
    return [(text, size, bold, color, italic)]


def img_h(name, w):
    im = Image.open(os.path.join(FIG, name))
    return w * im.size[1] / im.size[0]


# ── header band ───────────────────────────────────────────────────────────────
rect(0, 0, PAGE_W, HDR_H, RED)
rect(0, HDR_H, PAGE_W, 0.07, RED_DK)
tb = textbox(MARGIN, 0.32, PAGE_W - 2 * MARGIN, 1.15,
             [[('AC-WLS Power System State Estimation with PMU Integration',
                64, True, WHITE, False)]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
for p in tb.text_frame.paragraphs:
    for r in p.runs:
        r.font.name = FONT_HDR
textbox(MARGIN, 1.55, PAGE_W - 2 * MARGIN, 0.55,
        [[('EE574  Power System Real-Time Monitoring & Control  '
           '—  Semester Project', 28, False, RGBColor(0xFA, 0xDB, 0xD8),
           False)]],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
textbox(MARGIN, 2.18, PAGE_W - 2 * MARGIN, 0.55,
        [[('Ogün Altun  (2165785)      ·      Aytunç Evlice  '
           '(2516177)', 26, True, WHITE, False)]],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

# ── footer band (references) ──────────────────────────────────────────────────
fy = PAGE_H - MARGIN - FOOT_H + 0.12
rect(0, fy - 0.10, PAGE_W, FOOT_H + MARGIN, LIGHT)
textbox(MARGIN, fy, PAGE_W - 2 * MARGIN, FOOT_H,
        [[('References:   ', 15, True, DARK, False),
          ('A. Monticelli, State Estimation in Electric Power Systems, '
           'Kluwer, 1999   ·   A.J. Wood & B.F. Wollenberg, Power '
           'Generation, Operation and Control, Wiley, 2013   ·   '
           'IEEE Common Data Format, IEEE Trans. PAS-92(6), 1973   ·   ',
           15, False, GRAY, False),
          ('Code:  github.com/aytunc-evlice/EE574_Project', 15, True, RED,
           False)]],
        align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# ── column engine ─────────────────────────────────────────────────────────────
def col_x(i):
    return MARGIN + i * (COL_W + COL_GAP)


def layout_column(i, items, gap=0.25):
    """items: list of ('header', text) | ('text', tb_height, paragraphs)
    | ('img', name, width or None). Stacks from BODY_TOP."""
    x = col_x(i)
    y = BODY_TOP
    for item in items:
        if item[0] == 'header':
            textbox(x, y, COL_W, 0.55,
                    [[(item[1], 30, True, RED, False)]],
                    anchor=MSO_ANCHOR.MIDDLE)
            y += 0.55 + 0.12
        elif item[0] == 'text':
            _, h, pars = item
            textbox(x, y, COL_W, h, pars)
            y += h + gap
        elif item[0] == 'img':
            _, name, w = item
            w = COL_W if w is None else w
            h = img_h(name, w)
            slide.shapes.add_picture(
                os.path.join(FIG, name),
                Inches(x + (COL_W - w) / 2), Inches(y),
                Inches(w), Inches(h))
            y += h + gap
    return y


# ── column 1: introduction & network ─────────────────────────────────────────
intro = [
    body('State estimation is the core algorithm of every Energy '
         'Management System: it fuses noisy SCADA and PMU measurements '
         'into a consistent estimate of all bus voltages and angles, '
         'enabling real-time monitoring and control.'),
    body('This project implements a complete AC weighted-least-squares '
         '(WLS) estimator from scratch and validates it on the IEEE '
         '14-bus network, randomized networks up to 40 buses, and '
         'time-series measurement streams:'),
    body('•  Newton-based AC-WLS solver with analytical Jacobian', 17),
    body('•  Numerical observability analysis & PMU placement', 17),
    body('•  Normalized-residual bad data detection and removal', 17),
    body('•  Time-series operation on SCADA (5 s) + PMU (1 s) '
         'streams', 17),
]
facts = [
    [('IEEE 14-bus network:  ', 16, True, DARK, False),
     ('14 buses, 17 branches incl. 3 off-nominal-tap transformers, '
      '27 states.  PMUs at buses 2, 5, 9, 10, 12, 14 '
      '(σ ≈ 10⁻⁴ pu, 50× more accurate than '
      'SCADA).', 16, False, GRAY, False)],
]
layout_column(0, [
    ('header', 'Introduction'),
    ('text', 4.62, intro),
    ('text', 1.00, facts),
    ('header', 'Network & Observability'),
    ('img', '07_network_ieee14.png', None),
    ('img', '10_rank_ladder.png', None),
])

# ── column 2: methodology ─────────────────────────────────────────────────────
impl = [
    [('Implementation:  ', 16, True, DARK, False),
     ('Python / NumPy from scratch — IEEE CDF parser, transformer '
      'π-model, analytical Jacobian, warm start from load flow, '
      'backtracking line search (2–3 Newton iterations), '
      'innovation pre-screen (300σ) for gross PMU errors, and an '
      'interactive GUI for running the estimator on any CDF + '
      'measurement set.', 16, False, GRAY, False)],
]
layout_column(1, [
    ('header', 'Methodology'),
    ('img', '01_eq_wls.png', None),
    ('img', '02_eq_observability.png', None),
    ('img', '03_eq_baddata.png', None),
    ('text', 1.75, impl),
])

# ── column 3: algorithm & detection ──────────────────────────────────────────
layout_column(2, [
    ('header', 'Algorithm'),
    ('img', '04_flowchart.png', 6.65),
    ('header', 'Bad Data Detection in Action'),
    ('img', '09_residuals_example.png', None),
    ('img', '06_blindspot.png', None),
], gap=0.22)

# ── column 4: results & conclusions ──────────────────────────────────────────
concl = [
    body('•  Full AC-WLS estimator built from scratch; converges in '
         '2–3 Newton iterations with warm start.', 17),
    body('•  6 PMUs restore full observability (rank 19 → 27) '
         '— but only PMUs at unobservable buses add rank: '
         'placement matters more than count.', 17),
    body('•  SCADA + PMU fusion gives ~4.5× lower voltage RMSE '
         'than SCADA-only (7.8×10⁻⁴ vs '
         '3.5×10⁻³ pu).', 17),
    body('•  The normalized-residual test catches every injected '
         'gross error under redundant metering.', 17),
    body('•  Detection capability is set by metering redundancy, '
         'not the algorithm: critical measurements remain '
         'unverifiable.', 17),
]
layout_column(3, [
    ('header', 'Results'),
    ('img', '05_mode_comparison.png', None),
    ('img', '08_timeseries.png', None),
    ('img', '11_critical_measurement.png', None),
    ('header', 'Conclusions'),
    ('text', 3.6, concl),
])

prs.save(OUT)
print('saved', os.path.relpath(OUT, ROOT))
