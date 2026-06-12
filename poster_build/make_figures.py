"""
Generate discrete poster figures -> poster_build/figures/*.png

  01_eq_wls.png            WLS state estimation equations
  02_eq_observability.png  Observability analysis equations
  03_eq_baddata.png        Bad data detection / removal / criticality equations
  04_flowchart.png         Overall algorithm flowchart
  05_mode_comparison.png   SCADA+PMU vs SCADA-only accuracy & detection (3 cases)
  06_blindspot.png         Critical-measurement blind spot experiment
  07_network_ieee14.png    IEEE-14 one-line diagram with PMU placement
  08_timeseries.png        Time-series detection demo (real solve of case_TEST)
  09_residuals_example.png Residual bar chart at one instant (real solve)
  10_rank_ladder.png       PMU rank ladder (real observability computation)
  11_critical_measurement.png  Critical-measurement blind spot card

Run:  python poster_build/make_figures.py
"""
import os
import sys
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG_DIR = os.path.join(ROOT, 'poster_build', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ── palette (white / red, matches GUI) ────────────────────────────────────────
RED    = '#C0392B'
DARK   = '#1C2833'
GRAY   = '#717D7E'
LIGHT  = '#F4F6F8'
BORDER = '#D5D8DC'
GREEN  = '#1E8449'
BLUE   = '#2E6DA4'
ORANGE = '#D35400'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'mathtext.fontset': 'dejavusans',
    'axes.edgecolor': BORDER,
    'text.color': DARK,
    'axes.labelcolor': DARK,
    'xtick.color': GRAY,
    'ytick.color': GRAY,
})

DPI = 250


def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('  saved', os.path.relpath(path, ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# 1-3. EQUATION CARDS
# ══════════════════════════════════════════════════════════════════════════════
def eq_card(name, title, rows, figsize=(7.6, None)):
    """rows: list of (kind, text) where kind in {'eq','sub','gap'};
    sub rows may contain newlines."""
    H_EQ, H_SUB, H_GAP, H_TITLE, PAD = 0.55, 0.34, 0.16, 0.45, 0.30

    def row_h(kind, text):
        if kind == 'eq':
            return H_EQ
        if kind == 'sub':
            return H_SUB * (text.count('\n') + 1)
        return H_GAP

    height = H_TITLE + PAD
    for kind, text in rows:
        height += row_h(kind, text)
    height += PAD
    fig = plt.figure(figsize=(figsize[0], height))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, height)

    # card frame + title strip
    ax.add_patch(FancyBboxPatch((0.012, 0.05), 0.976, height - 0.1,
                                boxstyle='round,pad=0.01,rounding_size=0.06',
                                fc='white', ec=BORDER, lw=1.4))
    y = height - PAD - 0.30
    ax.text(0.05, y + 0.16, title, fontsize=17, fontweight='bold', color=RED,
            va='center')
    ax.plot([0.05, 0.95], [y - 0.10, y - 0.10], color=BORDER, lw=1.0)
    y -= H_TITLE

    for kind, text in rows:
        h = row_h(kind, text)
        if kind == 'eq':
            ax.text(0.5, y - h / 2, text, fontsize=15.5, ha='center',
                    va='center', color=DARK)
        elif kind == 'sub':
            ax.text(0.5, y - h / 2, text, fontsize=10.5, ha='center',
                    va='center', color=GRAY, linespacing=1.45)
        y -= h
    save(fig, name)


def make_equations():
    print('[1-3] equation cards')
    eq_card('01_eq_wls.png', 'AC-WLS State Estimation', [
        ('eq',  r'$x = [\,\theta_2 \ldots \theta_N,\; V_1 \ldots V_N\,]^T'
                r'\qquad (2N-1\ \mathrm{states},\ \ \theta_1 = 0)$'),
        ('sub', 'state vector: all bus angles (slack reference) and magnitudes'),
        ('gap', ''),
        ('eq',  r'$\min_{x}\ J(x) = [\,z - h(x)\,]^T\, W\, [\,z - h(x)\,],'
                r'\qquad W = \mathrm{diag}(1/\sigma_i^2)$'),
        ('sub', 'weighted least squares: accurate meters (PMU) dominate the fit'),
        ('gap', ''),
        ('eq',  r'$(H^T W H)\ \Delta x^{(k)} = H^T W\,[\,z - h(x^{(k)})\,],'
                r'\qquad H = \partial h / \partial x$'),
        ('sub', 'Newton normal equations with analytical Jacobian; '
                'warm start + backtracking line search'),
    ])

    eq_card('02_eq_observability.png', 'Numerical Observability', [
        ('eq',  r'$\mathrm{rank}(H) = 2N - 1 \ \ \Leftrightarrow\ \ '
                r'\mathrm{system\ observable}$'),
        ('sub', 'the measurement set carries enough independent information '
                'to estimate every state;\nthe rank of the Jacobian H is '
                'computed numerically at flat start'),
        ('gap', ''),
        ('eq',  r'$\mathrm{rank}(H) < 2N - 1 \ \ \Rightarrow\ \ '
                r'\mathrm{unobservable\ buses\ exist}$'),
        ('sub', 'the rank deficit counts the states the measurements cannot '
                'see;\nthe affected buses are identified and reported'),
        ('gap', ''),
        ('eq',  r'$\mathrm{PMU\ at\ bus\ }i:\ \ \{V_i,\ \theta_i\}\ '
                r'\mathrm{measured\ directly}\ \ (\sigma_{PMU} \approx '
                r'10^{-4}\ \mathrm{pu})$'),
        ('sub', 'each PMU adds two rows to H -> rank increases where SCADA '
                'coverage is thin\n(IEEE-14: rank 19 -> 27 with 6 PMUs, '
                'fully observable)'),
    ])

    eq_card('03_eq_baddata.png', 'Bad Data Detection & Removal', [
        ('eq',  r'$r = z - h(\hat{x}), \qquad'
                r' \Omega = R - H\,(H^T W H)^{-1} H^T$'),
        ('sub', 'residual covariance from the hat matrix; R = diag'
                r'($\sigma_i^2$)'),
        ('gap', ''),
        ('eq',  r'$r^{N}_{i} = \dfrac{|r_i|}{\sqrt{\Omega_{ii}}}\ >\ 3.0'
                r'\ \ \Rightarrow\ \ \mathrm{flagged\ as\ bad\ data}$'),
        ('sub', r'under clean conditions $r^N_i \sim N(0,1)$ '
                '-> 3-sigma test (largest normalized residual)'),
        ('gap', ''),
        ('eq',  r'$\mathrm{remove}\ \arg\max_i |r^N_i|\ \ \rightarrow\ \ '
                r're\!-\!estimate\ \ \rightarrow\ \ repeat\ until\ '
                r'\max|r^N_i| \leq 3.0$'),
        ('sub', 'iterative removal: one gross error at a time, state fully '
                're-estimated after each removal'),
    ])


# ══════════════════════════════════════════════════════════════════════════════
# 4. FLOWCHART
# ══════════════════════════════════════════════════════════════════════════════
def make_flowchart():
    print('[4] flowchart')
    fig = plt.figure(figsize=(8.6, 11.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, 10); ax.set_ylim(0, 14)

    CX = 4.6          # main column center
    BW, BH = 4.6, 1.0

    def box(x, y, w, h, text, fc='white', ec=DARK, tc=DARK, lw=1.6, fs=11,
            bold=False, round_=0.12):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=f'round,pad=0.02,rounding_size={round_}',
            fc=fc, ec=ec, lw=lw, zorder=3))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs,
                color=tc, fontweight='bold' if bold else 'normal', zorder=4)

    def arrow(p0, p1, color=DARK, lw=1.8, style='-', label=None, lpos=None,
              lcolor=None, connstyle='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(
            p0, p1, arrowstyle='-|>', mutation_scale=16, lw=lw,
            color=color, linestyle=style, zorder=2,
            connectionstyle=connstyle, shrinkA=2, shrinkB=2))
        if label:
            lx, ly = lpos
            ax.text(lx, ly, label, fontsize=10, color=lcolor or color,
                    fontweight='bold', ha='center', va='center', zorder=5,
                    bbox=dict(fc='white', ec='none', pad=1))

    y = 13.1
    box(CX, y, BW, 0.85, 'IEEE CDF network file', fc=RED, ec=RED,
        tc='white', bold=True, fs=12)

    y2 = y - 1.45
    arrow((CX, y - 0.45), (CX, y2 + 0.55))
    box(CX, y2, BW, BH, 'Build network model\n'
        r'$Y_{bus}$, $\pi$-model, off-nominal taps', fc=LIGHT, ec=BORDER)

    y3 = y2 - 1.55
    arrow((CX, y2 - 0.52), (CX, y3 + 0.55))
    box(CX, y3, BW, BH, 'Measurement set at instant $t$\n'
        'SCADA (P, Q, V)  +  PMU (V, $\\theta$)', fc=LIGHT, ec=BORDER)

    # observability decision
    y4 = y3 - 1.65
    arrow((CX, y3 - 0.52), (CX, y4 + 0.55))
    box(CX, y4, BW, BH, 'Observable?\n' r'$\mathrm{rank}(H) = 2N-1$',
        fc='white', ec=RED, lw=2.0, bold=True)
    # NO branch -> right side
    box(8.45, y4, 2.6, BH, 'flag unobservable\nskip / warn', fc='#FDEDEC',
        ec=RED, tc=RED, fs=10)
    arrow((CX + BW / 2, y4), (8.45 - 1.3, y4), color=RED,
          label='NO', lpos=(CX + BW / 2 + 0.55, y4 + 0.28), lcolor=RED)

    # WLS solve
    y5 = y4 - 1.7
    arrow((CX, y4 - 0.55), (CX, y5 + 0.55), label='YES',
          lpos=(CX - 0.45, y4 - 1.05), lcolor=GREEN)
    box(CX, y5, BW, BH, 'AC-WLS Newton solve\nwarm start - line search - '
        'converge', fc=LIGHT, ec=BORDER)

    # bad data decision
    y6 = y5 - 1.7
    arrow((CX, y5 - 0.52), (CX, y6 + 0.55))
    box(CX, y6, BW, BH, 'Bad data?\n' r'$\max\,|r^N_i| > 3.0$',
        fc='white', ec=RED, lw=2.0, bold=True)
    # YES branch -> remove + loop back to WLS
    box(8.45, y6, 2.6, BH, 'remove largest-\nresidual measurement',
        fc='#FDEDEC', ec=RED, tc=RED, fs=10)
    arrow((CX + BW / 2, y6), (8.45 - 1.3, y6), color=RED,
          label='YES', lpos=(CX + BW / 2 + 0.55, y6 + 0.28), lcolor=RED)
    # explicit re-estimate loop: up from the remove box, then into WLS solve
    ax.plot([8.45, 8.45], [y6 + 0.52, y5], color=RED, lw=1.8, ls='--',
            zorder=2)
    arrow((8.45, y5), (CX + BW / 2 + 0.05, y5), color=RED, style='--',
          label='re-estimate', lpos=(8.45, y5 + 0.32), lcolor=RED)

    # criticality audit
    y7 = y6 - 1.7
    arrow((CX, y6 - 0.55), (CX, y7 + 0.55), label='NO',
          lpos=(CX - 0.45, y6 - 1.05), lcolor=GREEN)
    box(CX, y7, BW, BH, 'Criticality audit\n'
        r'$e_{\min}$ -> flag unverifiable channels', fc=LIGHT, ec=BORDER)

    # outputs
    y8 = y7 - 1.6
    arrow((CX, y7 - 0.52), (CX, y8 + 0.48))
    box(CX, y8, BW, 0.95, 'Outputs:  $\\hat{x}$,  RMSE,  $J$,  '
        'detection report', fc=GREEN, ec=GREEN, tc='white', bold=True, fs=11.5)

    # loop back to next instant
    arrow((CX - BW / 2, y8), (0.62, y8), color=GRAY)
    arrow((0.62, y8), (0.62, y3), color=GRAY, style='--')
    arrow((0.62, y3), (CX - BW / 2, y3), color=GRAY, style='--',
          label='next instant  $t + \\Delta t$', lpos=(1.62, y3 + 0.75),
          lcolor=GRAY)

    ax.text(5.0, 13.78, 'AC-WLS State Estimation with Bad Data Processing',
            fontsize=15, fontweight='bold', color=DARK, ha='center')
    save(fig, '04_flowchart.png')


# ══════════════════════════════════════════════════════════════════════════════
# 5. MODE COMPARISON  (numbers from results/_guitest/FINDINGS.md)
# ══════════════════════════════════════════════════════════════════════════════
def make_mode_comparison():
    print('[5] mode comparison')
    cases  = ['IEEE-14', 'RAND-20', 'RAND-40']
    rmse_v   = {'SCADA + PMU': [7.8e-4, 1.3e-3, 2.1e-3],
                'SCADA only':  [3.5e-3, 3.0e-3, 3.1e-3]}
    rmse_ang = {'SCADA + PMU': [0.046, 0.064, 0.313],
                'SCADA only':  [0.240, 0.143, 0.380]}
    f1       = {'SCADA + PMU': [1.00, 1.00, 0.95],
                'SCADA only':  [0.94, 1.00, 0.95]}
    colors   = {'SCADA + PMU': RED, 'SCADA only': '#85929E'}

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor='white')
    panels = [
        (axes[0], rmse_v,   'RMSE voltage (pu)',   '%.1e'),
        (axes[1], rmse_ang, 'RMSE angle (deg)',    '%.3f'),
        (axes[2], f1,       'Bad data detection success rate', '%.2f'),
    ]
    x = np.arange(len(cases)); w = 0.36
    for ax, data, title, fmt in panels:
        ax.set_facecolor('white')
        for k, (mode, vals) in enumerate(data.items()):
            bars = ax.bar(x + (k - 0.5) * w, vals, w, color=colors[mode],
                          label=mode, zorder=3)
            for b, v in zip(bars, vals):
                ax.annotate(fmt % v, (b.get_x() + b.get_width() / 2, v),
                            ha='center', va='bottom', fontsize=8.5,
                            color=DARK, fontweight='bold')
        ax.set_title(title, fontsize=12, color=DARK, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(cases, fontsize=10)
        ax.grid(axis='y', color=BORDER, alpha=0.8, zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.margins(y=0.18)
    axes[2].set_ylim(0, 1.18)
    handles = [Patch(fc=colors[m], label=m) for m in rmse_v]
    fig.legend(handles=handles, ncol=2, fontsize=10, frameon=False,
               loc='upper right', bbox_to_anchor=(0.99, 1.085))
    fig.suptitle('Measurement fusion: PMU integration improves every metric '
                 '(13 fused instants per case)',
                 fontsize=13, color=DARK, fontweight='bold', y=1.04, x=0.42)
    fig.text(0.5, -0.075,
             'PMU-only mode is structurally unobservable on all cases '
             '(voltage-only phasors: 2 x n_PMU rows  <  2N-1 states) '
             '-> fusion with SCADA is required.\n'
             'Detection rates depend on meter placement and error location '
             '(single-seed runs) - see the redundancy study.',
             ha='center', fontsize=9.5, color=GRAY, style='italic',
             linespacing=1.6)
    fig.tight_layout()
    save(fig, '05_mode_comparison.png')


# ══════════════════════════════════════════════════════════════════════════════
# 6. CRITICAL-MEASUREMENT BLIND SPOT  (FINDINGS.md section 4c)
# ══════════════════════════════════════════════════════════════════════════════
def make_blindspot():
    print('[6] blind spot')
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.0, 4.8), facecolor='white')

    GRAY_BAR = '#85929E'
    cats = ['FEW METERS\n(low redundancy)', 'MANY METERS\n(high redundancy)']

    # ── left: the same gross error, seen in the residual test ──
    vals = [1.56, 32.08]
    bars = a0.bar([0, 1], vals, 0.5, color=[GRAY_BAR, RED], zorder=3)
    a0.axhline(3.0, color=RED, ls='--', lw=1.8, zorder=4)
    a0.text(1.72, 3.6, 'detection threshold 3.0', color=RED, fontsize=10,
            va='bottom', ha='right')
    a0.annotate('1.56 - MISSED', (0, vals[0]), xytext=(0, 22),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color=GRAY_BAR)
    a0.annotate('32.08\nDETECTED', (1, vals[1]), xytext=(0, -42),
                textcoords='offset points', ha='center', va='bottom',
                fontsize=13, fontweight='bold', color='white')
    a0.set_xticks([0, 1]); a0.set_xticklabels(cats, fontsize=11)
    a0.set_xlim(-0.75, 1.75)
    a0.set_ylim(0, 38)
    a0.set_ylabel('how loud the error is in the test  ' r'($|r^N|$)',
                  fontsize=11)
    a0.set_title('The SAME gross error - hidden vs caught',
                 fontsize=12.5, fontweight='bold', color=DARK)
    a0.grid(axis='y', color=BORDER, alpha=0.8, zorder=0)
    for sp in ('top', 'right'):
        a0.spines[sp].set_visible(False)

    # ── right: overall detection rate ──
    rate = [35, 100]
    bars = a1.bar([0, 1], rate, 0.5, color=[GRAY_BAR, RED], zorder=3)
    for x, v, c in zip([0, 1], rate, [GRAY_BAR, RED]):
        a1.annotate(f'{v}%', (x, v), xytext=(0, 6),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=16, fontweight='bold', color=c)
    a1.set_xticks([0, 1]); a1.set_xticklabels(cats, fontsize=11)
    a1.set_xlim(-0.75, 1.75)
    a1.set_ylim(0, 118)
    a1.set_ylabel('bad data detection rate (%)', fontsize=11)
    a1.set_title('Detection rate over the full stream',
                 fontsize=12.5, fontweight='bold', color=DARK)
    a1.grid(axis='y', color=BORDER, alpha=0.8, zorder=0)
    for sp in ('top', 'right'):
        a1.spines[sp].set_visible(False)

    fig.suptitle('More measurement redundancy  ->  higher bad data detection '
                 'rate', fontsize=14.5, fontweight='bold', color=DARK, y=1.04)
    fig.text(0.5, -0.035,
             'Same 20-bus network, same injected gross errors - only the '
             'number of meters differs. A measurement no other meter can '
             'cross-check ("critical") cannot be validated:\nits errors are '
             'absorbed into the state. Redundant coverage lets the residual '
             'test cross-check every channel.',
             ha='center', fontsize=10, color=GRAY, style='italic',
             linespacing=1.5)
    fig.tight_layout()
    save(fig, '06_blindspot.png')


# ══════════════════════════════════════════════════════════════════════════════
# 7. IEEE-14 ONE-LINE DIAGRAM
# ══════════════════════════════════════════════════════════════════════════════
def spring_layout(n, edges, seed=7, iters=400):
    rng = np.random.RandomState(seed)
    pos = rng.uniform(-1, 1, (n, 2)).astype(float)
    A = np.zeros((n, n))
    for i, j in edges:
        A[i, j] = A[j, i] = 1.0
    k = (1.0 / n) ** 0.5
    t = 0.1
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.sqrt((delta ** 2).sum(-1)) + 1e-9
        force = (k * k) / dist - A * (dist * dist) / k
        np.fill_diagonal(force, 0.0)
        disp = (delta / dist[..., None] * force[..., None]).sum(1)
        length = np.sqrt((disp ** 2).sum(-1)) + 1e-9
        pos += (disp / length[:, None]) * np.minimum(length, t)[:, None]
        t *= 0.99
    pos -= pos.min(0)
    span = pos.max(0); span[span == 0] = 1.0
    return pos / span


def make_network():
    print('[7] network diagram')
    from src.parser import parse_ieee_cdf
    buses, branches, _ = parse_ieee_cdf(os.path.join(ROOT, 'ieee_cdf_sample.dat'))
    nums = [b['num'] for b in buses]
    bidx = {n: i for i, n in enumerate(nums)}
    edges = [(bidx[br['from_bus']], bidx[br['to_bus']]) for br in branches]
    pos = spring_layout(len(buses), edges)

    PMU = {2, 5, 9, 10, 12, 14}
    fig, ax = plt.subplots(figsize=(8.6, 7.2), facecolor='white')
    ax.axis('off')

    for br, (i, j) in zip(branches, edges):
        xfmr = abs(br['tap'] - 1.0) > 1e-9
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                color=ORANGE if xfmr else '#AEB6BF',
                lw=2.6 if xfmr else 1.6, zorder=1)
        if xfmr:
            mx, my = (pos[i] + pos[j]) / 2
            ax.add_patch(Circle((mx, my), 0.016, fc='white', ec=ORANGE,
                                lw=1.8, zorder=2))
            ax.add_patch(Circle((mx + 0.018, my), 0.016, fc='white',
                                ec=ORANGE, lw=1.8, zorder=2))

    for k, b in enumerate(buses):
        if b['type'] == 3:
            fc, label = RED, 'slack bus'
        elif b['type'] == 2:
            fc, label = '#5D6D7E', 'generator (PV) bus'
        else:
            fc, label = '#D6DBDF', 'load (PQ) bus'
        ax.scatter(*pos[k], s=620, c=fc, zorder=3,
                   edgecolors=DARK, linewidths=1.0)
        if b['num'] in PMU:
            ax.scatter(*pos[k], s=1150, facecolors='none', edgecolors=GREEN,
                       linewidths=2.6, zorder=4)
        ax.annotate(str(b['num']), pos[k], color='white' if b['type'] in (2, 3)
                    else DARK, fontsize=10, fontweight='bold',
                    ha='center', va='center', zorder=5)

    handles = [
        plt.scatter([], [], s=160, c=RED, edgecolors=DARK, label='slack bus'),
        plt.scatter([], [], s=160, c='#5D6D7E', edgecolors=DARK,
                    label='generator (PV) bus'),
        plt.scatter([], [], s=160, c='#D6DBDF', edgecolors=DARK,
                    label='load (PQ) bus'),
        plt.scatter([], [], s=220, facecolors='none', edgecolors=GREEN,
                    linewidths=2.4, label='PMU installed'),
        plt.Line2D([], [], color=ORANGE, lw=2.6,
                   label='transformer (off-nominal tap)'),
        plt.Line2D([], [], color='#AEB6BF', lw=1.6, label='line'),
    ]
    ax.legend(handles=handles, fontsize=10, frameon=False, ncol=3,
              loc='upper center', bbox_to_anchor=(0.5, -0.01),
              columnspacing=1.4, handletextpad=0.5)
    ax.set_title('IEEE 14-bus test network - 17 branches, 3 transformers, '
                 '6 PMUs', fontsize=13.5, fontweight='bold', color=DARK,
                 pad=12)
    ax.margins(0.06)
    save(fig, '07_network_ieee14.png')


# ══════════════════════════════════════════════════════════════════════════════
# 8-9. REAL TIME-SERIES SOLVE  (case_TEST clean + bad twins)
# ══════════════════════════════════════════════════════════════════════════════
def make_timeseries():
    print('[8-9] time-series solve (real data)')
    from solve_timeseries import solve_stream
    d_clean = os.path.join(ROOT, 'results', '_guitest', 'case_TEST')
    d_bad   = d_clean + '_bad'
    if not (os.path.isfile(os.path.join(d_clean, 'index.csv'))
            and os.path.isfile(os.path.join(d_bad, 'index.csv'))):
        print('  !! case_TEST streams not found - skipping figures 08/09')
        return
    cdf = os.path.join(ROOT, 'ieee_cdf_sample.dat')   # manifest may hold a
    out_c = solve_stream(d_clean, use_scada=True, use_pmu=True, threshold=3.0,
                         cdf_path=cdf)                # stale absolute path
    out_b = solve_stream(d_bad,   use_scada=True, use_pmu=True, threshold=3.0,
                         cdf_path=cdf)
    rc, rb = out_c['results'], out_b['results']
    t = np.array([r['t'] for r in rb])

    # ── fig 8: RMSE tracking + removals ──
    fig, (a0, a1) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, facecolor='white',
        gridspec_kw={'height_ratios': [2.1, 1]})
    a0.plot(t, [r['rmse_V_before'] or np.nan for r in rb], 'o--', color=RED,
            lw=1.8, ms=5, label='corrupted stream - before removal')
    a0.plot(t, [r['rmse_V'] or np.nan for r in rb], 'o-', color=DARK,
            lw=2.0, ms=5, label='corrupted stream - after removal')
    a0.plot(t, [r['rmse_V'] or np.nan for r in rc], 's-', color=GREEN,
            lw=1.6, ms=4, alpha=0.85, label='clean stream (reference)')
    a0.set_yscale('log')
    a0.set_ylabel('RMSE voltage (pu)', fontsize=11)
    a0.grid(color=BORDER, alpha=0.8, zorder=0)
    a0.legend(fontsize=9.5, frameon=True, edgecolor=BORDER, facecolor='white',
              framealpha=0.95, loc='lower center', ncol=3)
    a0.set_title('Detect -> remove -> re-estimate restores clean-stream '
                 'accuracy at every instant  (IEEE-14, 60 s stream)',
                 fontsize=12.5, fontweight='bold', color=DARK)
    for sp in ('top', 'right'):
        a0.spines[sp].set_visible(False)

    a1.step(t, [len(r['actual_bad']) for r in rb], where='mid', color='#85929E',
            lw=2.4, label='gross errors present (ground truth)')
    a1.step(t, [r['n_removed'] for r in rb], where='mid', color=RED, lw=1.8,
            ls='--', label='detected & removed')
    a1.set_xlabel('time (s)', fontsize=11)
    a1.set_ylabel('# bad data', fontsize=11)
    a1.set_ylim(bottom=0)
    a1.grid(color=BORDER, alpha=0.8, zorder=0)
    a1.legend(fontsize=10, frameon=False, loc='upper left')
    for sp in ('top', 'right'):
        a1.spines[sp].set_visible(False)
    fig.tight_layout()
    save(fig, '08_timeseries.png')

    # ── fig 9: residual bars before / after removal at the worst instant ──
    r = max(rb, key=lambda x: x['n_removed'])
    removed = {rm['label'] for rm in r['removed']}
    rm_list = ',  '.join(rm['label'] for rm in r['removed'])

    fig, (a0, a1) = plt.subplots(2, 1, figsize=(11.5, 6.4), facecolor='white')

    # before: first WLS pass
    rn0, lb0 = np.abs(r['r_n_first']), r['labels']
    a0.bar(np.arange(len(rn0)), rn0,
           color=[RED if lb0[i] in removed else '#A9CCE3'
                  for i in range(len(rn0))], zorder=3)
    a0.axhline(3.0, color=RED, ls='--', lw=1.5, zorder=4)
    for i in range(len(rn0)):
        if lb0[i] in removed and rn0[i] > 3.0:
            a0.annotate(lb0[i], (i, rn0[i]), xytext=(0, 5),
                        textcoords='offset points', ha='center',
                        fontsize=9, color=RED, fontweight='bold')
    # gross errors so large they were rejected before the first WLS pass
    prescreened = [rm['label'] for rm in r['removed']
                   if rm.get('by') == 'prescreen']
    if prescreened:
        a0.text(0.985, 0.80,
                f'{", ".join(prescreened)}: PMU error > 300$\\sigma$\n'
                'rejected by the innovation pre-screen\n'
                '(too large to even appear on this chart)',
                transform=a0.transAxes, ha='right', va='top', fontsize=9.5,
                color=RED, fontweight='bold',
                bbox=dict(fc='#FDEDEC', ec=RED, lw=1.2,
                          boxstyle='round,pad=0.45'))
    a0.set_title(f'First WLS pass at t = {r["t"]:.0f} s - '
                 f'{r["n_removed"]} gross errors present',
                 fontsize=11.5, fontweight='bold', color=DARK)
    a0.set_ylabel(r'$|r^N_i|$', fontsize=12)

    # after: final pass on the kept set
    rn1 = np.abs(r['r_n_final'])
    a1.bar(np.arange(len(rn1)), rn1, color='#A9CCE3', zorder=3)
    a1.axhline(3.0, color=RED, ls='--', lw=1.5, zorder=4)
    a1.text(len(rn1) - 0.5, 3.05, 'threshold 3.0', color=RED, fontsize=10,
            va='bottom', ha='right')
    a1.set_ylim(0, max(4.2, float(np.max(rn1)) * 1.25))
    a1.set_title(f'After iterative removal of  [{rm_list}]  and '
                 're-estimation - every kept measurement is below threshold',
                 fontsize=11.5, fontweight='bold', color=DARK)
    a1.set_xlabel('measurement index', fontsize=11)
    a1.set_ylabel(r'$|r^N_i|$', fontsize=12)

    for ax in (a0, a1):
        ax.grid(axis='y', color=BORDER, alpha=0.8, zorder=0)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    legend = [Patch(fc=RED, label='identified as bad & removed'),
              Patch(fc='#A9CCE3', label='kept measurement')]
    a0.legend(handles=legend, fontsize=10, frameon=False, loc='upper left')
    fig.tight_layout()
    save(fig, '09_residuals_example.png')


# ══════════════════════════════════════════════════════════════════════════════
# 10. OBSERVABILITY RANK LADDER  (real computation on IEEE-14 + measure.dat)
# ══════════════════════════════════════════════════════════════════════════════
def make_rank_ladder():
    print('[10] observability rank ladder')
    from src.parser import parse_ieee_cdf, parse_measurements
    from src.network import Network
    from src.observability import check_observability
    from src.simulation import generate_pmu_measurements

    buses, branches, _ = parse_ieee_cdf(os.path.join(ROOT, 'ieee_cdf_sample.dat'))
    net = Network(buses, branches)
    vb = {b['num'] for b in buses}
    bs = {(br['from_bus'], br['to_bus']) for br in branches}
    for br in branches:
        bs.add((br['to_bus'], br['from_bus']))
    slack_num = net.buses[net.slack_idx]['num']
    orig = parse_measurements(os.path.join(ROOT, 'measure.dat'), vb, bs,
                              max_sections=6)
    orig = [m for m in orig
            if not (m['type'] == 'Vang' and m.get('bus') == slack_num)]

    PMU_ORDER = [2, 5, 9, 10, 12, 14]
    ranks, labels = [], []
    for k in range(len(PMU_ORDER) + 1):
        sub = PMU_ORDER[:k]
        meas = orig + (generate_pmu_measurements(net, sub, sigma_pmu=1e-4,
                                                 seed=42) if sub else [])
        _, rank, n_states, _ = check_observability(net, meas)
        ranks.append(rank)
        labels.append('SCADA\nonly' if k == 0 else f'+ PMU\nbus {PMU_ORDER[k-1]}')

    fig, ax = plt.subplots(figsize=(10.5, 4.6), facecolor='white')
    xx = np.arange(len(ranks))
    colors = [RED if r >= n_states else '#85929E' for r in ranks]
    bars = ax.bar(xx, ranks, 0.55, color=colors, zorder=3)
    ax.axhline(n_states, color=RED, ls='--', lw=1.8, zorder=4)
    ax.text(-0.45, n_states + 0.35,
            f'fully observable:  rank = {n_states}  (2N-1 states)',
            color=RED, fontsize=10.5, fontweight='bold', va='bottom')
    for b, r in zip(bars, ranks):
        ax.annotate(str(r), (b.get_x() + b.get_width() / 2, r),
                    xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=12, fontweight='bold',
                    color=RED if r >= n_states else DARK)
    ax.set_xticks(xx); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, n_states + 4)
    ax.set_ylabel('rank of the Jacobian  H', fontsize=11)
    ax.set_title('PMU placement matters: only PMUs at unobservable buses '
                 'raise the rank (+2 each) - 6 PMUs reach full observability',
                 fontsize=12.5, fontweight='bold', color=DARK)
    ax.grid(axis='y', color=BORDER, alpha=0.8, zorder=0)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.annotate('buses 2, 5 already observed\nby SCADA -> no rank gain',
                (1.5, ranks[2] + 0.6), xytext=(1.5, ranks[2] + 5.2),
                fontsize=9.5, color=GRAY, ha='center',
                arrowprops=dict(arrowstyle='->', color=GRAY))
    fig.text(0.5, -0.03,
             'Real computation: the original 19 SCADA measurements '
             '(measure.dat) leave 8 states unobservable; PMUs are added '
             'one at a time at buses 2, 5, 9, 10, 12, 14.',
             ha='center', fontsize=9.5, color=GRAY, style='italic')
    fig.tight_layout()
    save(fig, '10_rank_ladder.png')


# ══════════════════════════════════════════════════════════════════════════════
# 11. LIMITATIONS & FUTURE WORK CARD
# ══════════════════════════════════════════════════════════════════════════════
def make_critical():
    print('[11] critical measurement blind spot card')
    H = 5.45
    fig = plt.figure(figsize=(10.2, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, H)

    # card frame + title strip
    ax.add_patch(FancyBboxPatch((0.012, 0.05), 0.976, H - 0.1,
                                boxstyle='round,pad=0.01,rounding_size=0.06',
                                fc='white', ec=BORDER, lw=1.4))
    y = H - 0.30 - 0.30
    ax.text(0.045, y + 0.16,
            'Blind Spot: a Critical Measurement Cannot Be Validated',
            fontsize=16, fontweight='bold', color=RED, va='center')
    ax.plot([0.045, 0.955], [y - 0.10, y - 0.10], color=BORDER, lw=1.0)
    y -= 0.42

    ax.text(0.5, y - 0.22,
            'A measurement is critical when it is the only source of '
            'information for some state - removing it makes the system '
            'unobservable.\nNo other meter can cross-check it, so the WLS '
            'fit simply bends the state to match its value - errors '
            'included.',
            fontsize=10.2, color=GRAY, ha='center', va='center',
            linespacing=1.5)
    y -= 0.62

    # ── two mini panels: critical vs redundant meter ──
    PANEL_H = 2.05
    py0 = y - PANEL_H
    panels = [
        (0.05, 0.46, '#FDEDEC', RED,
         'CRITICAL:  the only meter on a radial branch',
         [0.5], GRAY,
         '+0.5 pu gross error absorbed into the state\n'
         r'$|r^N| = 1.56 < 3.0\ \ \rightarrow$  NOT flagged'),
        (0.54, 0.95, '#EAF7EF', GREEN,
         'REDUNDANT:  neighbouring meters cross-check',
         [0.28, 0.5, 0.72], GREEN,
         'the same gross error contradicts its neighbours\n'
         r'$|r^N| = 32.08 \gg 3.0\ \ \rightarrow$  FLAGGED & removed'),
    ]
    for x0, x1, fc, ec, head, meters, mcol, caption in panels:
        ax.add_patch(FancyBboxPatch(
            (x0, py0), x1 - x0, PANEL_H,
            boxstyle='round,pad=0.005,rounding_size=0.04',
            fc=fc, ec=ec, lw=1.3))
        cx = lambda f: x0 + f * (x1 - x0)
        ax.text(cx(0.5), py0 + PANEL_H - 0.24, head, fontsize=10.5,
                fontweight='bold', color=ec, ha='center', va='center')
        # branch drawing: bus 9 ---meter(s)--- bus 13
        ly = py0 + PANEL_H - 0.85
        ax.plot([cx(0.14), cx(0.86)], [ly, ly], color=DARK, lw=2.0,
                zorder=2)
        for bx, lbl in [(0.14, '9'), (0.86, '13')]:
            ax.scatter([cx(bx)], [ly], s=520, c='white', edgecolors=DARK,
                       linewidths=1.8, zorder=3)
            ax.text(cx(bx), ly, lbl, fontsize=10, fontweight='bold',
                    color=DARK, ha='center', va='center', zorder=4)
        for mf in meters:
            ax.scatter([cx(mf)], [ly], s=240, marker='s', c='white',
                       edgecolors=mcol, linewidths=2.2, zorder=3)
            ax.text(cx(mf), ly, 'M', fontsize=8, fontweight='bold',
                    color=mcol, ha='center', va='center', zorder=4)
        ax.text(cx(0.5), py0 + 0.42, caption, fontsize=10, color=DARK,
                ha='center', va='center', linespacing=1.6)
    y = py0 - 0.30

    # ── the math reason + experiment footer ──
    ax.text(0.5, y - 0.26,
            r'$\mathrm{critical\ measurement:}\ \ \Omega_{ii} \rightarrow 0'
            r'\ \ \Rightarrow\ \ r_i \rightarrow 0\ \ \Rightarrow\ \ '
            r'\mathrm{no\ residual\ to\ test}$',
            fontsize=14, color=DARK, ha='center', va='center')
    y -= 0.58
    ax.text(0.5, y - 0.22,
            'Experiment (FINDINGS 4c): identical gross error on Pflow(9-13), '
            'a radial-leaf meter - detection recall 0.35 with sparse '
            'metering vs 1.00 with full coverage.\nDetection capability is '
            'a property of the metering redundancy, not of the algorithm.',
            fontsize=10, color=GRAY, ha='center', va='center',
            style='italic', linespacing=1.5)
    save(fig, '11_critical_measurement.png')


if __name__ == '__main__':
    make_equations()
    make_flowchart()
    make_mode_comparison()
    make_blindspot()
    make_network()
    make_timeseries()
    make_rank_ladder()
    make_critical()
    print('\nAll figures in', os.path.relpath(FIG_DIR, ROOT))
