"""
EE574 - Time-Series State Estimator (demo viewer for generated streams)
=======================================================================
Loads a stream from gen_measurements.py / inject_bad_data.py, toggles SCADA/PMU,
solves AC-WLS at every consistent instant with iterative bad-data removal, and
presents it for a live demo:

  * headline scorecards (observability / RMSE before vs after / precision-recall)
  * Play/Pause animation that scrubs through time
  * Clean <-> Bad toggle (solves both twins; switch instantly)
  * network one-line diagram (voltage-coloured buses, PMU marks, bad data flashing)
  * per-instant state + bad-data tables and residual bars

It does NOT generate or inject data.   Run:  python gui_solver.py
"""
import os
import sys
import ast
import csv
import threading
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_timeseries import (solve_stream, load_manifest,
                              parse_measurement_file, meas_key, _lbl)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def default_stream_dir():
    base = os.path.join(DATA_DIR, 'results', 'timeseries')
    cands = [os.path.join(base, 'case_IEEE14')]
    if os.path.isdir(base):
        cands += [os.path.join(base, n) for n in sorted(os.listdir(base))
                  if not n.endswith('_bad')]
    for c in cands:
        if os.path.isfile(os.path.join(c, 'index.csv')):
            return c
    return base


# ── palette: white / red ──────────────────────────────────────────────────────
BG       = '#FFFFFF'
BG_MID   = '#F4F6F8'
BG_LIGHT = '#E8EDF2'
BG_HDR   = '#C0392B'       # red header bar
FG       = '#1C2833'
RED      = '#C0392B'
RED_LT   = '#FADBD8'       # light red tint for highlights
GREEN    = '#1E8449'
ORANGE   = '#D35400'
GRAY     = '#717D7E'
GRID_C   = '#D5D8DC'

SPEEDS = {'Slow': 800, 'Medium': 380, 'Fast': 150}


def spring_layout(n, edges, seed=1, iters=300):
    if n <= 1:
        return np.array([[0.5, 0.5]] * max(n, 1))
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


class _Stdout:
    def __init__(self, cb): self._cb = cb
    def write(self, t): self._cb(t)
    def flush(self): pass


class SolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('EE574  ·  Time-Series State Estimator')
        self.minsize(1240, 860)
        self.datasets = {}
        self.out = None
        self.which = 'clean'
        self.pos = None; self.edges_idx = []
        self._playing = False; self._after_id = None
        self._has_bad = False; self._clean_dir = None; self._bad_dir = None
        self._badrecs = {}; self._cmp_cache = {}
        self._vb = set(); self._bs = set()
        self._build_styles()
        self._build_ui()

    # ── styling ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        self.configure(bg=BG)

        s.configure('TFrame',        background=BG)
        s.configure('Card.TFrame',   background=BG_MID)
        s.configure('Tile.TFrame',   background=BG_MID,  relief='flat',
                    borderwidth=1)

        s.configure('TLabel',        background=BG,     foreground=FG,
                    font=('Segoe UI', 10))
        s.configure('Hdr.TLabel',    background=BG_HDR, foreground='#FFFFFF',
                    font=('Segoe UI', 14, 'bold'))
        s.configure('Sub.TLabel',    background=BG,     foreground=GRAY,
                    font=('Segoe UI', 9))
        s.configure('Tile.TLabel',   background=BG_MID, foreground=FG)
        s.configure('TileCap.TLabel',background=BG_MID, foreground=GRAY,
                    font=('Segoe UI', 8))
        s.configure('Sec.TLabel',    background=BG,     foreground=RED,
                    font=('Segoe UI', 10, 'bold'))

        s.configure('TCheckbutton',  background=BG, foreground=FG,
                    font=('Segoe UI', 10))
        s.map('TCheckbutton', background=[('active', BG)])
        s.configure('TRadiobutton',  background=BG, foreground=FG,
                    font=('Segoe UI', 10))
        s.map('TRadiobutton', background=[('active', BG)])

        s.configure('TButton',       background=BG_LIGHT, foreground=FG,
                    font=('Segoe UI', 9), relief='flat')
        s.map('TButton',
              background=[('active', BG_LIGHT), ('disabled', BG_LIGHT)],
              foreground=[('disabled', GRAY)])
        s.configure('Run.TButton',   background=RED, foreground='#FFFFFF',
                    font=('Segoe UI', 11, 'bold'), relief='flat')
        s.map('Run.TButton',
              background=[('active', '#922B21'), ('disabled', BG_LIGHT)],
              foreground=[('disabled', GRAY)])
        s.configure('Play.TButton',  background=GREEN, foreground='#FFFFFF',
                    font=('Segoe UI', 10, 'bold'), relief='flat')
        s.map('Play.TButton',
              background=[('active', '#155235')])

        s.configure('TNotebook',     background=BG, tabmargins=[0, 0, 0, 0])
        s.configure('TNotebook.Tab', background=BG_MID, foreground=GRAY,
                    padding=[16, 7], font=('Segoe UI', 10))
        s.map('TNotebook.Tab',
              background=[('selected', BG_HDR)],
              foreground=[('selected', '#FFFFFF')])

        s.configure('Treeview',          background='#FFFFFF', foreground=FG,
                    fieldbackground='#FFFFFF', font=('Consolas', 9), rowheight=22)
        s.configure('Treeview.Heading',  background=BG_LIGHT, foreground=RED,
                    font=('Segoe UI', 9, 'bold'), relief='flat')
        s.map('Treeview', background=[('selected', RED_LT)],
              foreground=[('selected', FG)])

        s.configure('TScrollbar', background=BG_MID, troughcolor=BG_LIGHT,
                    arrowcolor=GRAY)
        s.configure('TProgressbar', troughcolor=BG_LIGHT, background=RED)
        s.configure('TEntry',        fieldbackground='#FFFFFF', foreground=FG,
                    insertcolor=FG, bordercolor=BG_LIGHT, relief='flat')

    # ── top-level layout ──────────────────────────────────────────────────────
    def _build_ui(self):
        # header bar
        hdr = tk.Frame(self, bg=BG_HDR)
        hdr.pack(fill='x')
        tk.Label(hdr, text='Time-Series State Estimator',
                 bg=BG_HDR, fg='#FFFFFF',
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=16, pady=10)
        tk.Label(hdr,
                 text='load a stream · fuse SCADA+PMU · estimate · detect & remove bad data',
                 bg=BG_HDR, fg='#F8C0B8', font=('Segoe UI', 9, 'italic')).pack(side='left')

        # control row
        ctl = ttk.Frame(self)
        ctl.pack(fill='x', padx=16, pady=(10, 4))

        ttk.Label(ctl, text='Data folder:').grid(row=0, column=0, sticky='w')
        self._dir = tk.StringVar(value=default_stream_dir())
        ttk.Entry(ctl, textvariable=self._dir,
                  font=('Consolas', 9)).grid(row=0, column=1, columnspan=5,
                                              sticky='ew', padx=8)
        ttk.Button(ctl, text='Browse',
                   command=self._browse).grid(row=0, column=6, padx=2)

        ttk.Label(ctl, text='CDF file:').grid(row=1, column=0, sticky='w', pady=4)
        self._cdf = tk.StringVar(value='')
        ttk.Entry(ctl, textvariable=self._cdf,
                  font=('Consolas', 9)).grid(row=1, column=1, columnspan=3,
                                              sticky='ew', padx=8)
        ttk.Button(ctl, text='Browse',
                   command=self._browse_cdf).grid(row=1, column=4, padx=2, sticky='w')
        self._loads_pu = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text='loads in pu',
                        variable=self._loads_pu).grid(row=1, column=5, sticky='w')
        ttk.Label(ctl, text='(blank = use folder manifest)',
                  style='Sub.TLabel').grid(row=1, column=6, sticky='w')

        self._use_scada = tk.BooleanVar(value=True)
        self._use_pmu   = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text='SCADA',
                        variable=self._use_scada).grid(row=2, column=0, sticky='w', pady=6)
        ttk.Checkbutton(ctl, text='PMU',
                        variable=self._use_pmu).grid(row=2, column=1, sticky='w')
        ttk.Label(ctl, text='Threshold:').grid(row=2, column=2, sticky='e')
        self._thr = tk.StringVar(value='3.0')
        ttk.Entry(ctl, textvariable=self._thr, width=6).grid(
            row=2, column=3, sticky='w', padx=4)
        self._run_btn = ttk.Button(ctl, text='▶  Solve',
                                   style='Run.TButton', command=self._start)
        self._run_btn.grid(row=2, column=4, padx=10)
        self._which = tk.StringVar(value='clean')
        self._rb_clean = ttk.Radiobutton(ctl, text='Clean', value='clean',
                                         variable=self._which,
                                         command=self._switch_dataset, state='disabled')
        self._rb_bad   = ttk.Radiobutton(ctl, text='Bad', value='bad',
                                         variable=self._which,
                                         command=self._switch_dataset, state='disabled')
        self._rb_clean.grid(row=2, column=5, sticky='e')
        self._rb_bad.grid(row=2, column=6, sticky='w')
        ctl.columnconfigure(1, weight=1)

        # status + progress
        if os.path.isfile(os.path.join(self._dir.get(), 'index.csv')):
            hint = 'Select a stream folder and solve.'
        else:
            hint = ('No generated stream found - run  "python '
                    'gen_measurements.py --cdf <case.dat>"  first, then '
                    'Browse to its output folder (contains index.csv).')
        self._status = ttk.Label(self, text=hint, style='Sub.TLabel')
        self._status.pack(fill='x', padx=16)
        self._prog = ttk.Progressbar(self, mode='determinate', length=300)
        self._prog.pack(fill='x', padx=16, pady=2)

        # scorecards
        self._build_scorecards()

        # instant label
        self._estlabel = ttk.Label(self, text='', style='Sub.TLabel',
                                   font=('Segoe UI', 10, 'bold'))
        self._estlabel.pack(fill='x', padx=16, pady=(2, 0))

        # play / scrub bar
        self._build_timebar()

        # ── notebook: only Network + State & Residuals ────────────────────────
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=16, pady=(4, 12))
        self._nb = nb

        # Network tab
        self._net_tab = ttk.Frame(nb)
        nb.add(self._net_tab, text='  Network  ')

        # State & Residuals tab
        self._build_state_tab(nb)

    def _build_scorecards(self):
        bar = ttk.Frame(self)
        bar.pack(fill='x', padx=16, pady=6)
        self._tiles = {}
        specs = [
            ('Case',                  140),
            ('Observable',            110),
            ('RMSE before removal',   150),
            ('RMSE after removal',    150),
            ('Bad now',                90),
            ('Unverifiable',          110),
            ('Detection (run)',        180),
        ]
        for name, w in specs:
            card = ttk.Frame(bar, style='Tile.TFrame')
            card.pack(side='left', padx=4)
            ttk.Label(card, text=name, style='TileCap.TLabel').pack(
                anchor='w', padx=10, pady=(6, 0))
            v = ttk.Label(card, text='—', style='Tile.TLabel',
                          font=('Segoe UI', 13, 'bold'),
                          width=max(8, w // 11))
            v.pack(anchor='w', padx=10, pady=(0, 7))
            self._tiles[name] = v

    def _build_timebar(self):
        tb = ttk.Frame(self)
        tb.pack(fill='x', padx=16, pady=4)
        self._play_btn = ttk.Button(tb, text='▶  Play', style='Play.TButton',
                                    command=self._toggle_play, state='disabled')
        self._play_btn.pack(side='left')
        self._speed = tk.StringVar(value='Medium')
        ttk.Combobox(tb, textvariable=self._speed, values=list(SPEEDS),
                     width=8, state='readonly').pack(side='left', padx=6)
        self._slider = ttk.Scale(tb, from_=0, to=0, orient='horizontal',
                                 command=self._on_slide)
        self._slider.pack(side='left', fill='x', expand=True, padx=8)
        self._tlabel = ttk.Label(tb, text='t = —', style='Sub.TLabel', width=22)
        self._tlabel.pack(side='left')

    def _build_state_tab(self, nb):
        tab = ttk.Frame(nb)
        nb.add(tab, text='  State & Bad Data  ')

        # state table (left) + bad data table (right)
        top = ttk.Frame(tab)
        top.pack(fill='both', expand=True, padx=6, pady=(6, 0))
        top.columnconfigure(0, weight=1, uniform='cols')
        top.columnconfigure(1, weight=1, uniform='cols')
        top.rowconfigure(1, weight=1)

        # ── state table (left column) ──
        ttk.Label(top, text='Estimated state  (after bad-data removal)',
                  style='Sec.TLabel').grid(row=0, column=0, sticky='w',
                                           padx=(0, 6), pady=(0, 4))
        lf = ttk.Frame(top)
        lf.grid(row=1, column=0, sticky='nsew', padx=(0, 6))
        lf.rowconfigure(0, weight=1)
        lf.columnconfigure(0, weight=1)
        cols_s = ('Bus', 'V_est (pu)', 'θ_est (°)', 'V_true (pu)', 'ΔV (pu)')
        self._state = ttk.Treeview(lf, columns=cols_s, show='headings', height=10)
        for c, w in zip(cols_s, (55, 100, 95, 100, 95)):
            self._state.heading(c, text=c)
            self._state.column(c, width=w, anchor='center')
        vsb_s = ttk.Scrollbar(lf, orient='vertical', command=self._state.yview)
        self._state.configure(yscrollcommand=vsb_s.set)
        self._state.grid(row=0, column=0, sticky='nsew')
        vsb_s.grid(row=0, column=1, sticky='ns')
        self._state.tag_configure('warn', foreground=ORANGE)
        self._state.tag_configure('good', foreground=GREEN)

        # ── bad data table (right column) ──
        ttk.Label(top, text='Bad data  (removed / missed / unverifiable)',
                  style='Sec.TLabel').grid(row=0, column=1, sticky='w',
                                           pady=(0, 4))
        rf = ttk.Frame(top)
        rf.grid(row=1, column=1, sticky='nsew')
        rf.rowconfigure(0, weight=1)
        rf.columnconfigure(0, weight=1)
        cols_b = ('Measurement', 'r_n', 'value', 'status')
        self._bad = ttk.Treeview(rf, columns=cols_b, show='headings', height=10)
        for c, w in zip(cols_b, (165, 70, 95, 145)):
            self._bad.heading(c, text=c)
            self._bad.column(c, width=w,
                             anchor='w' if c == 'Measurement' else 'center')
        vsb_b = ttk.Scrollbar(rf, orient='vertical', command=self._bad.yview)
        self._bad.configure(yscrollcommand=vsb_b.set)
        self._bad.grid(row=0, column=0, sticky='nsew')
        vsb_b.grid(row=0, column=1, sticky='ns')
        self._bad.tag_configure('tp',   foreground=GREEN)
        self._bad.tag_configure('fp',   foreground=ORANGE)
        self._bad.tag_configure('fn',   foreground=RED)
        self._bad.tag_configure('crit', foreground=GRAY)

        # ── residuals chart in its own tab ──
        res_tab = ttk.Frame(nb)
        nb.add(res_tab, text='  Normalized Residuals  ')
        self._rframe = ttk.Frame(res_tab)
        self._rframe.pack(fill='both', expand=True, padx=6, pady=6)

    # ── browse / solve ────────────────────────────────────────────────────────
    def _browse(self):
        init = self._dir.get()
        while init and not os.path.isdir(init):
            init = os.path.dirname(init.rstrip('/\\'))
        p = filedialog.askdirectory(title='Select a measurement data folder '
                                          '(the one containing index.csv)',
                                    initialdir=init or DATA_DIR)
        if p:
            self._dir.set(p)
            self._prefill_from_manifest(p)

    def _browse_cdf(self):
        p = filedialog.askopenfilename(
            title='Select CDF network file', initialdir=DATA_DIR,
            filetypes=[('CDF/DAT files', '*.dat'), ('All files', '*.*')])
        if p:
            self._cdf.set(p)

    def _prefill_from_manifest(self, d):
        clean_dir, _ = self._twins(d)
        for cand in (d, clean_dir):
            try:
                m = load_manifest(cand)
            except Exception:
                m = {}
            if m:
                if m.get('cdf_file'):
                    self._cdf.set(m['cdf_file'])
                self._loads_pu.set(bool(m.get('loads_in_pu')))
                return

    def _twins(self, d):
        d = d.rstrip('/\\')
        if d.endswith('_bad'):
            return d[:-4], d
        return d, d + '_bad'

    def _start(self):
        if not (self._use_scada.get() or self._use_pmu.get()):
            self._set_status('Select at least one of SCADA / PMU.', RED)
            return
        self._pause()
        self._run_btn.configure(state='disabled')
        self._set_status('Solving...', ORANGE)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        old = sys.stdout
        sys.stdout = _Stdout(lambda t: None)   # suppress console (no console tab)
        try:
            chosen = self._dir.get().strip()
            clean_dir, bad_dir = self._twins(chosen)
            opts = dict(
                use_scada=self._use_scada.get(),
                use_pmu=self._use_pmu.get(),
                threshold=float(self._thr.get()),
                progress=self._progress,
                cdf_path=(self._cdf.get().strip() or None),
                loads_in_pu=(True if self._loads_pu.get() else None),
            )
            ds = {}
            if os.path.isfile(os.path.join(clean_dir, 'index.csv')):
                ds['clean'] = solve_stream(clean_dir, **opts)
            if os.path.isfile(os.path.join(bad_dir, 'index.csv')):
                ds['bad'] = solve_stream(bad_dir, **opts)
            if not ds:
                raise FileNotFoundError(
                    'no index.csv found in the chosen folder or its twin')
            self.datasets = ds
            self._clean_dir, self._bad_dir = clean_dir, bad_dir
            self._has_bad = 'bad' in ds
            self._badrecs = (self._load_bad_records(bad_dir)
                             if self._has_bad else {})
            self._cmp_cache = {}
            self._default_which = (
                'bad' if chosen.rstrip('/\\').endswith('_bad') and 'bad' in ds
                else ('clean' if 'clean' in ds else 'bad'))
            self.after(0, self._on_solved)
        except Exception as exc:
            import traceback; traceback.print_exc()
            msg = str(exc)
            self.after(0, lambda: (
                self._set_status(f'Error: {msg}', RED),
                self._run_btn.configure(state='normal')))
        finally:
            sys.stdout = old

    def _progress(self, i, total, t):
        self.after(0, lambda: (
            self._prog.configure(maximum=total, value=i),
            self._set_status(
                f'Solving instant {i}/{total}  (t={t:.0f} s)...', ORANGE)))

    def _on_solved(self):
        self._run_btn.configure(state='normal')
        self._rb_clean.configure(
            state='normal' if 'clean' in self.datasets else 'disabled')
        self._rb_bad.configure(
            state='normal' if 'bad' in self.datasets else 'disabled')
        any_out = next(iter(self.datasets.values()))
        net = any_out['net']; nums = any_out['nums']
        bidx = {n: i for i, n in enumerate(nums)}
        self.edges_idx = [(bidx[br['from_bus']], bidx[br['to_bus']])
                          for br in net.branches]
        self.pos = spring_layout(net.n, self.edges_idx)
        self._bidx = bidx
        self._vb = {b['num'] for b in net.buses}
        self._bs = ({(br['from_bus'], br['to_bus']) for br in net.branches}
                    | {(br['to_bus'], br['from_bus']) for br in net.branches})
        self._which.set(self._default_which)
        self._switch_dataset()
        self._play_btn.configure(state='normal')
        self._set_status('Done.  Use Play or the slider; toggle Clean / Bad.', GREEN)

    def _switch_dataset(self):
        self.which = self._which.get()
        self.out = self.datasets.get(self.which)
        if not self.out or not self.out['results']:
            self._set_status('That dataset has no solved instants.', RED)
            return
        n = len(self.out['results'])
        self._slider.configure(from_=0, to=n - 1)
        i = min(int(self._slider.get()), n - 1)
        self._show_instant(i)

    # ── per-instant rendering ─────────────────────────────────────────────────
    def _on_slide(self, _v):
        if self.out and self.out['results']:
            self._show_instant(int(float(self._slider.get())))

    def _show_instant(self, i):
        res = self.out['results']
        i = max(0, min(i, len(res) - 1))
        r = res[i]
        self._tlabel.configure(text=f"t = {r['t']:.1f} s   ({i+1}/{len(res)})")

        crit  = r.get('critical', [])
        names = ', '.join(c['label'] for c in crit[:3]) + \
                (f' +{len(crit)-3} more' if len(crit) > 3 else '')
        warn  = (f"      ⚠ unverifiable: {names}" if crit else '')

        if r['n_removed']:
            jb, ja = r.get('J_before'), r['J']
            jtxt = (f"    J: {jb:.0f} → {ja:.0f}"
                    if (jb is not None and ja is not None) else '')
            self._estlabel.configure(
                text=f"⟳  {r['n_removed']} bad measurement(s) removed → "
                     f"state RE-ESTIMATED{jtxt}{warn}",
                foreground=ORANGE)
        else:
            self._estlabel.configure(
                text=f"✓  no bad data detected — single estimate{warn}",
                foreground=ORANGE if crit else GREEN)

        self._update_scorecards(r)
        self._fill_state(r)
        self._fill_bad(r)
        self._draw_network(r)
        self._draw_residuals(r)

    def _update_scorecards(self, r):
        m = self.out['meta']
        self._tiles['Case'].configure(text=f"{m['case']}  [{self.which}]",
                                      foreground=FG)
        obs = r['observable']
        self._tiles['Observable'].configure(
            text='YES' if obs else 'NO',
            foreground=GREEN if obs else RED)
        rb = r.get('rmse_V_before')
        improved = (rb is not None and r['rmse_V'] is not None
                    and rb > 1.5 * r['rmse_V'])
        self._tiles['RMSE before removal'].configure(
            text=_e(rb), foreground=ORANGE if improved else GRAY)
        self._tiles['RMSE after removal'].configure(
            text=_e(r['rmse_V']), foreground=GREEN if improved else FG)
        self._tiles['Bad now'].configure(
            text=str(r['n_removed']),
            foreground=RED if r['n_removed'] else GREEN)
        ncrit = len(crit := r.get('critical', []))
        self._tiles['Unverifiable'].configure(
            text=str(ncrit), foreground=ORANGE if ncrit else GREEN)
        results = self.out['results']
        if m['has_badlog']:
            TP = sum(x['tp'] for x in results)
            FP = sum(x['fp'] for x in results)
            FN = sum(x['fn'] for x in results)
            rec  = TP / (TP + FN) if (TP + FN) else 1.0
            prec = TP / (TP + FP) if (TP + FP) else 1.0
            self._tiles['Detection (run)'].configure(
                text=f"P={prec:.2f}  R={rec:.2f}", foreground=FG)
        else:
            self._tiles['Detection (run)'].configure(
                text='clean stream', foreground=GRAY)

    def _fill_state(self, r):
        for it in self._state.get_children():
            self._state.delete(it)
        if r['V'] is None:
            return
        nums  = self.out['nums']
        truth = self.out['truth']
        Vt = None
        if truth is not None and round(r['t'], 6) in truth:
            Vt, _ = truth[round(r['t'], 6)]
        for k, n in enumerate(nums):
            dv  = (r['V'][k] - Vt[k]) if Vt is not None else 0.0
            tag = 'warn' if (Vt is not None and abs(dv) > 0.01) else 'good'
            self._state.insert('', 'end', tags=(tag,), values=(
                n,
                f"{r['V'][k]:.5f}",
                f"{np.rad2deg(r['theta'][k]):.3f}",
                f"{Vt[k]:.5f}" if Vt is not None else '—',
                f"{dv:+.5f}"   if Vt is not None else '—',
            ))

    def _fill_bad(self, r):
        for it in self._bad.get_children():
            self._bad.delete(it)
        actual = set(r['actual_bad'])
        for rm in r['removed']:
            if r['tp'] is None:
                status, tag = 'removed', 'tp'
            elif str(rm['key']) in actual:
                status, tag = 'correct (TP)', 'tp'
            else:
                status, tag = 'false alarm (FP)', 'fp'
            self._bad.insert('', 'end', tags=(tag,), values=(
                rm['label'], f"{rm['r_n']:+.2f}",
                f"{rm['value']:.5f}", status))
        if r['tp'] is not None:
            removed = {str(rm['key']) for rm in r['removed']}
            for k in sorted(actual - removed):
                self._bad.insert('', 'end', tags=('fn',),
                                 values=(_key_label(k), '—', '—', 'missed (FN)'))
        for c in r.get('critical', []):
            self._bad.insert('', 'end', tags=('crit',), values=(
                c['label'], '—', f"{c['value']:.5f}",
                'unverifiable (blind spot)'))

    # ── network diagram ───────────────────────────────────────────────────────
    def _bad_nodes_edges(self, r):
        nodes, edges = set(), set()
        for rm in r['removed']:
            if 'bus' in rm:
                if rm['bus'] in self._bidx:
                    nodes.add(self._bidx[rm['bus']])
            elif (rm.get('from_bus') in self._bidx
                  and rm.get('to_bus') in self._bidx):
                edges.add(frozenset((self._bidx[rm['from_bus']],
                                     self._bidx[rm['to_bus']])))
        return nodes, edges

    def _crit_nodes_edges(self, r):
        nodes, edges = set(), set()
        for c in r.get('critical', []):
            _, l1, l2 = c['key']
            if l2 is None:
                if l1 in self._bidx:
                    nodes.add(self._bidx[l1])
            elif l1 in self._bidx and l2 in self._bidx:
                edges.add(frozenset((self._bidx[l1], self._bidx[l2])))
        return nodes, edges

    def _draw_network(self, r):
        for w in self._net_tab.winfo_children():
            w.destroy()
        net = self.out['net']; nums = self.out['nums']; pos = self.pos
        bad_nodes,  bad_edges  = self._bad_nodes_edges(r)
        crit_nodes, crit_edges = self._crit_nodes_edges(r)
        pmu = set(self.out['meta']['pmu_buses'])

        fig = Figure(figsize=(9, 6.4), facecolor=BG)
        ax  = fig.add_subplot(111, facecolor=BG_MID)
        ax.axis('off')
        ax.set_facecolor(BG_MID)
        ax.set_title(
            f"{self.out['meta']['case']} [{self.which}]  ·  t={r['t']:.1f} s  ·  "
            f"{r['n_removed']} bad removed",
            color=FG, fontsize=11, fontweight='bold', pad=10)

        for (i, j) in self.edges_idx:
            e   = frozenset((i, j))
            bad = e in bad_edges
            crit_e = e in crit_edges and not bad
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=RED  if bad  else (GRAY if crit_e else '#B0BEC5'),
                    lw=3.0     if bad  else (2.2  if crit_e else 1.4),
                    ls=(0, (4, 3)) if crit_e else '-',
                    zorder=2 if (bad or crit_e) else 1)

        V = (r['V'] if r['V'] is not None else np.full(len(nums), np.nan))
        if r['V'] is not None:
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=V,
                            cmap='RdYlGn', vmin=0.94, vmax=1.06,
                            s=340, zorder=3,
                            edgecolors=BG_LIGHT, linewidths=1.2)
            cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cb.set_label('|V| (pu)', color=FG, fontsize=9)
            cb.ax.yaxis.set_tick_params(color=GRAY)
            plt_set_cb_colors(cb, GRAY)
        else:
            ax.scatter(pos[:, 0], pos[:, 1], c=GRAY, s=340, zorder=3)

        for k, n in enumerate(nums):
            if n in pmu:
                ax.scatter(pos[k, 0], pos[k, 1], s=580,
                           facecolors='none', edgecolors=GREEN,
                           linewidths=2.0, zorder=4)
            if k in bad_nodes:
                ax.scatter(pos[k, 0], pos[k, 1], s=740,
                           facecolors='none', edgecolors=RED,
                           linewidths=2.6, zorder=5)
            if k in crit_nodes and k not in bad_nodes:
                ax.scatter(pos[k, 0], pos[k, 1], s=740,
                           facecolors='none', edgecolors=GRAY,
                           linewidths=2.0, linestyle=(0, (4, 3)), zorder=5)
            ax.annotate(str(n), (pos[k, 0], pos[k, 1]),
                        color=FG, fontsize=7,
                        ha='center', va='center', zorder=6,
                        fontweight='bold')

        ax.scatter([], [], s=120, facecolors='none', edgecolors=GREEN,
                   label='PMU bus')
        ax.scatter([], [], s=120, facecolors='none', edgecolors=RED,
                   label='bad data removed')
        if crit_nodes or crit_edges:
            ax.plot([], [], color=GRAY, ls=(0, (4, 3)), lw=2.0,
                    label='unverifiable channel')
        ax.legend(facecolor=BG_MID, edgecolor=BG_LIGHT,
                  labelcolor=FG, fontsize=9, loc='upper right')
        fig.tight_layout()

        c = FigureCanvasTkAgg(fig, master=self._net_tab)
        c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

    # ── residuals chart ───────────────────────────────────────────────────────
    def _draw_residuals(self, r):
        for w in self._rframe.winfo_children():
            w.destroy()
        if r['r_n_first'] is None:
            return

        rn      = np.abs(r['r_n_first'])
        labels  = r['labels']
        removed = {rm['label'] for rm in r['removed']}
        crit    = {c['label'] for c in r.get('critical', [])}
        thr     = float(self._thr.get() or 3.0)

        fig = Figure(figsize=(10, 2.8), facecolor=BG)
        ax  = fig.add_subplot(111, facecolor=BG_MID)
        for sp in ax.spines.values():
            sp.set_color(GRID_C)
        ax.tick_params(colors=GRAY)

        bar_colors = [
            RED   if labels[i] in removed else
            GRAY  if labels[i] in crit    else
            '#5DADE2'
            for i in range(len(rn))
        ]
        ax.bar(np.arange(len(rn)), rn, color=bar_colors, zorder=3,
               edgecolor='none')

        if crit:
            top = max(float(np.max(rn)), thr)
            for i in range(len(rn)):
                if labels[i] in crit:
                    ax.annotate(labels[i], (i, rn[i] + 0.02 * top),
                                color=GRAY, fontsize=8,
                                ha='center', va='bottom', rotation=90)

        ax.axhline(thr, color=RED, ls='--', lw=1.4,
                   label=f'threshold {thr}')
        ax.set_title(
            'Normalized residuals  (red = removed, blue = clean, gray = unverifiable)',
            color=FG, fontsize=10)
        ax.set_ylabel('|r_n|', color=GRAY, fontsize=9)
        ax.grid(axis='y', color=GRID_C, alpha=0.8, zorder=0)
        ax.set_ylim(bottom=0)

        from matplotlib.patches import Patch
        handles, _ = ax.get_legend_handles_labels()
        handles += [
            Patch(facecolor='#5DADE2', label='clean measurement'),
            Patch(facecolor=RED,       label='removed (bad)'),
        ]
        if crit:
            handles.append(Patch(facecolor=GRAY, label='unverifiable'))
        ax.legend(handles=handles, facecolor=BG_MID, edgecolor=GRID_C,
                  labelcolor=FG, fontsize=8, loc='upper right')
        fig.tight_layout()

        c = FigureCanvasTkAgg(fig, master=self._rframe)
        c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

    # ── clean-vs-bad helpers (still used for _badrecs) ────────────────────────
    def _load_bad_records(self, bad_dir):
        out = {}
        p = os.path.join(bad_dir, 'bad_log.csv')
        if not os.path.isfile(p):
            return out
        try:
            with open(p, newline='') as f:
                for row in csv.DictReader(f):
                    loc2 = row['loc2']
                    l1 = int(row['loc1'])
                    l2 = None if loc2 in ('', None) else int(loc2)
                    label = (f"{row['type']}(bus {l1})"
                             if l2 is None
                             else f"{row['type']}({l1}-{l2})")
                    out.setdefault(row['file'], []).append({
                        'key':    (row['type'], l1, l2),
                        'label':  label,
                        'type':   row['type'],
                        'clean':  float(row['clean_value']),
                        'bad':    float(row['bad_value']),
                        'error':  float(row['error']),
                        'sigmas': float(row['error_sigmas']),
                        'mode':   row.get('mode', ''),
                    })
        except Exception:
            return {}
        return out

    # ── animation ─────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._playing:
            self._pause()
        elif self.out and self.out['results']:
            self._playing = True
            self._play_btn.configure(text='⏸  Pause')
            self._advance()

    def _advance(self):
        if not self._playing or not self.out:
            return
        n = len(self.out['results'])
        i = (int(float(self._slider.get())) + 1) % n
        self._slider.set(i)
        self._after_id = self.after(
            SPEEDS.get(self._speed.get(), 380), self._advance)

    def _pause(self):
        self._playing = False
        self._play_btn.configure(text='▶  Play')
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None

    # ── helpers ───────────────────────────────────────────────────────────────
    def _set_status(self, text, color=GRAY):
        self._status.configure(text=text, foreground=color)


def plt_set_cb_colors(cb, color):
    """Set colorbar tick label colors (matplotlib helper)."""
    try:
        for lbl in cb.ax.get_yticklabels():
            lbl.set_color(color)
    except Exception:
        pass


def _e(v):
    return f"{v:.2e}" if v is not None else '—'


def _key_label(ks):
    try:
        typ, l1, l2 = ast.literal_eval(ks)
        return (f"{typ}(bus {l1})" if l2 is None
                else f"{typ}({l1}-{l2})")
    except Exception:
        return ks


if __name__ == '__main__':
    SolverApp().mainloop()
