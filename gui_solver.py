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
import threading
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solve_timeseries import solve_stream, load_manifest

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BG, BG_MID, BG_LIGHT = '#0A1628', '#0D2B5E', '#1A3A6B'
FG, ORANGE, GREEN, RED, GRAY = '#FFFFFF', '#E87722', '#44CC88', '#FF4444', '#8899AA'
BLUE, CONSOLE = '#4A90D9', '#060E1C'
SPEEDS = {'Slow': 800, 'Medium': 380, 'Fast': 150}


# ── network layout (Fruchterman-Reingold spring layout, no extra deps) ─────────
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
        self.datasets = {}          # 'clean'/'bad' -> solve_stream output
        self.out = None
        self.which = 'clean'
        self.pos = None; self.edges_idx = []
        self._playing = False; self._after_id = None
        self._build_styles()
        self._build_ui()

    # ── styling ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self); s.theme_use('clam'); self.configure(bg=BG)
        s.configure('TFrame', background=BG)
        s.configure('Card.TFrame', background=BG_MID)
        s.configure('Tile.TFrame', background=BG_LIGHT)
        s.configure('TLabel', background=BG, foreground=FG, font=('Segoe UI', 10))
        s.configure('Title.TLabel', background=BG_MID, foreground=FG, font=('Segoe UI', 14, 'bold'))
        s.configure('Sub.TLabel', background=BG, foreground=GRAY, font=('Segoe UI', 9))
        s.configure('Tile.TLabel', background=BG_LIGHT, foreground=FG)
        s.configure('TileCap.TLabel', background=BG_LIGHT, foreground=GRAY, font=('Segoe UI', 8))
        s.configure('TCheckbutton', background=BG, foreground=FG, font=('Segoe UI', 10))
        s.map('TCheckbutton', background=[('active', BG)])
        s.configure('TRadiobutton', background=BG, foreground=FG, font=('Segoe UI', 10))
        s.map('TRadiobutton', background=[('active', BG)])
        s.configure('TButton', background=BG_LIGHT, foreground=FG, font=('Segoe UI', 9))
        s.configure('Run.TButton', background=ORANGE, foreground=FG, font=('Segoe UI', 11, 'bold'))
        s.map('Run.TButton', background=[('active', '#C96010'), ('disabled', '#333')])
        s.configure('Play.TButton', background=GREEN, foreground='#06210F', font=('Segoe UI', 10, 'bold'))
        s.configure('TNotebook', background=BG)
        s.configure('TNotebook.Tab', background=BG_MID, foreground=GRAY, padding=[14, 6])
        s.map('TNotebook.Tab', background=[('selected', BG_LIGHT)], foreground=[('selected', FG)])
        s.configure('Treeview', background=BG_MID, foreground=FG, fieldbackground=BG_MID,
                    font=('Consolas', 9), rowheight=21)
        s.configure('Treeview.Heading', background=BG_LIGHT, foreground=ORANGE, font=('Segoe UI', 9, 'bold'))
        s.configure('TProgressbar', troughcolor=BG, background=ORANGE)

    # ── layout ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = ttk.Frame(self, style='Card.TFrame'); hdr.pack(fill='x')
        ttk.Label(hdr, text='Time-Series State Estimator', style='Title.TLabel',
                  background=BG_MID).pack(side='left', padx=16, pady=10)
        ttk.Label(hdr, text='load a stream · fuse SCADA+PMU · estimate · detect & remove bad data',
                  background=BG_MID, foreground=GRAY, font=('Segoe UI', 9, 'italic')).pack(side='left')

        ctl = ttk.Frame(self); ctl.pack(fill='x', padx=16, pady=(8, 4))
        ttk.Label(ctl, text='Data folder:').grid(row=0, column=0, sticky='w')
        self._dir = tk.StringVar(value=os.path.join(DATA_DIR, 'results', 'timeseries'))
        ttk.Entry(ctl, textvariable=self._dir, font=('Consolas', 9)).grid(
            row=0, column=1, columnspan=5, sticky='ew', padx=8)
        ttk.Button(ctl, text='Browse', command=self._browse).grid(row=0, column=6, padx=2)

        ttk.Label(ctl, text='CDF file:').grid(row=1, column=0, sticky='w', pady=4)
        self._cdf = tk.StringVar(value='')
        ttk.Entry(ctl, textvariable=self._cdf, font=('Consolas', 9)).grid(
            row=1, column=1, columnspan=3, sticky='ew', padx=8)
        ttk.Button(ctl, text='Browse', command=self._browse_cdf).grid(row=1, column=4, padx=2, sticky='w')
        self._loads_pu = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctl, text='loads in pu', variable=self._loads_pu).grid(row=1, column=5, sticky='w')
        ttk.Label(ctl, text='(blank = use folder manifest)', style='Sub.TLabel').grid(
            row=1, column=6, sticky='w')

        self._use_scada = tk.BooleanVar(value=True)
        self._use_pmu = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text='SCADA', variable=self._use_scada).grid(row=2, column=0, sticky='w', pady=6)
        ttk.Checkbutton(ctl, text='PMU', variable=self._use_pmu).grid(row=2, column=1, sticky='w')
        ttk.Label(ctl, text='Threshold:').grid(row=2, column=2, sticky='e')
        self._thr = tk.StringVar(value='3.0')
        ttk.Entry(ctl, textvariable=self._thr, width=6).grid(row=2, column=3, sticky='w', padx=4)
        self._run_btn = ttk.Button(ctl, text='▶  Solve', style='Run.TButton', command=self._start)
        self._run_btn.grid(row=2, column=4, padx=10)
        # clean/bad toggle
        self._which = tk.StringVar(value='clean')
        self._rb_clean = ttk.Radiobutton(ctl, text='Clean', value='clean',
                                         variable=self._which, command=self._switch_dataset, state='disabled')
        self._rb_bad = ttk.Radiobutton(ctl, text='Bad', value='bad',
                                       variable=self._which, command=self._switch_dataset, state='disabled')
        self._rb_clean.grid(row=2, column=5, sticky='e'); self._rb_bad.grid(row=2, column=6, sticky='w')
        ctl.columnconfigure(1, weight=1)

        self._status = ttk.Label(self, text='Select a stream folder and solve.', style='Sub.TLabel')
        self._status.pack(fill='x', padx=16)
        self._prog = ttk.Progressbar(self, mode='determinate', length=300)
        self._prog.pack(fill='x', padx=16, pady=2)

        self._build_scorecards()
        self._build_timebar()

        nb = ttk.Notebook(self); nb.pack(fill='both', expand=True, padx=16, pady=(4, 10))
        self._nb = nb
        self._net_tab = ttk.Frame(nb); nb.add(self._net_tab, text=' Network ')
        self._build_state_tab(nb)
        self._track_tab = ttk.Frame(nb); nb.add(self._track_tab, text=' Tracking / Detection ')
        cf = ttk.Frame(nb); nb.add(cf, text=' Console ')
        self._console = scrolledtext.ScrolledText(cf, state='disabled', bg=CONSOLE, fg='#B8D0F0',
                                                   font=('Consolas', 9), relief='flat')
        self._console.pack(fill='both', expand=True, padx=4, pady=4)

    def _build_scorecards(self):
        bar = ttk.Frame(self); bar.pack(fill='x', padx=16, pady=4)
        self._tiles = {}
        specs = [('Case', 130), ('Observable', 120), ('RMSE |V| after', 130),
                 ('RMSE |V| before', 130), ('Bad now', 90), ('Detection (run)', 170)]
        for name, w in specs:
            t = ttk.Frame(bar, style='Tile.TFrame'); t.pack(side='left', padx=4)
            ttk.Label(t, text=name, style='TileCap.TLabel').pack(anchor='w', padx=8, pady=(4, 0))
            v = ttk.Label(t, text='—', style='Tile.TLabel', font=('Segoe UI', 13, 'bold'), width=max(8, w // 11))
            v.pack(anchor='w', padx=8, pady=(0, 5))
            self._tiles[name] = v

    def _build_timebar(self):
        tb = ttk.Frame(self); tb.pack(fill='x', padx=16, pady=2)
        self._play_btn = ttk.Button(tb, text='▶  Play', style='Play.TButton',
                                    command=self._toggle_play, state='disabled')
        self._play_btn.pack(side='left')
        self._speed = tk.StringVar(value='Medium')
        ttk.Combobox(tb, textvariable=self._speed, values=list(SPEEDS), width=8,
                     state='readonly').pack(side='left', padx=6)
        self._slider = ttk.Scale(tb, from_=0, to=0, orient='horizontal', command=self._on_slide)
        self._slider.pack(side='left', fill='x', expand=True, padx=8)
        self._tlabel = ttk.Label(tb, text='t = —', style='Sub.TLabel', width=22); self._tlabel.pack(side='left')

    def _build_state_tab(self, nb):
        tab = ttk.Frame(nb); nb.add(tab, text=' State & Residuals ')
        body = ttk.Frame(tab); body.pack(fill='both', expand=True)
        lf = ttk.Frame(body); lf.pack(side='left', fill='both', expand=True, padx=6, pady=4)
        ttk.Label(lf, text='Estimated state', foreground=ORANGE, font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        cols = ('Bus', 'V_est', 'θ_est(°)', 'V_true', 'ΔV')
        self._state = ttk.Treeview(lf, columns=cols, show='headings', height=12)
        for c, w in zip(cols, (50, 90, 90, 90, 90)):
            self._state.heading(c, text=c); self._state.column(c, width=w, anchor='center')
        self._state.pack(fill='both', expand=True)
        self._state.tag_configure('warn', foreground='#FFB830')
        rf = ttk.Frame(body); rf.pack(side='left', fill='both', expand=True, padx=6, pady=4)
        ttk.Label(rf, text='Bad data (removed / missed)', foreground=ORANGE,
                  font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        bcols = ('Measurement', 'r_n', 'value', 'status')
        self._bad = ttk.Treeview(rf, columns=bcols, show='headings', height=12)
        for c, w in zip(bcols, (150, 70, 90, 120)):
            self._bad.heading(c, text=c); self._bad.column(c, width=w,
                              anchor='w' if c == 'Measurement' else 'center')
        self._bad.pack(fill='both', expand=True)
        self._bad.tag_configure('tp', foreground=GREEN)
        self._bad.tag_configure('fp', foreground=ORANGE)
        self._bad.tag_configure('fn', foreground=RED)
        self._rframe = ttk.Frame(tab); self._rframe.pack(fill='both', expand=True, padx=6)

    # ── solve ────────────────────────────────────────────────────────────────────
    def _browse(self):
        p = filedialog.askdirectory(title='Select a measurement data folder', initialdir=self._dir.get())
        if p:
            self._dir.set(p)
            self._prefill_from_manifest(p)

    def _browse_cdf(self):
        p = filedialog.askopenfilename(title='Select CDF network file', initialdir=DATA_DIR,
                                       filetypes=[('CDF/DAT files', '*.dat'), ('All files', '*.*')])
        if p:
            self._cdf.set(p)

    def _prefill_from_manifest(self, d):
        """If the folder (or its clean twin) has a manifest, pre-fill CDF + loads-in-pu."""
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
            self._set_status('Select at least one of SCADA / PMU.', RED); return
        self._pause()
        self._run_btn.configure(state='disabled')
        self._set_status('Solving...', ORANGE); self._clear_console()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        old = sys.stdout; sys.stdout = _Stdout(self._append)
        try:
            chosen = self._dir.get().strip()
            clean_dir, bad_dir = self._twins(chosen)
            opts = dict(use_scada=self._use_scada.get(), use_pmu=self._use_pmu.get(),
                        threshold=float(self._thr.get()), progress=self._progress,
                        cdf_path=(self._cdf.get().strip() or None),
                        loads_in_pu=(True if self._loads_pu.get() else None))
            ds = {}
            if os.path.isfile(os.path.join(clean_dir, 'index.csv')):
                print(f'Solving CLEAN: {clean_dir}'); ds['clean'] = solve_stream(clean_dir, **opts)
            if os.path.isfile(os.path.join(bad_dir, 'index.csv')):
                print(f'Solving BAD:   {bad_dir}'); ds['bad'] = solve_stream(bad_dir, **opts)
            if not ds:
                raise FileNotFoundError('no index.csv found in the chosen folder or its twin')
            self.datasets = ds
            self._default_which = 'bad' if chosen.rstrip('/\\').endswith('_bad') and 'bad' in ds else \
                                  ('clean' if 'clean' in ds else 'bad')
            for k in ds:
                self._summary(ds[k], k)
            self.after(0, self._on_solved)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.after(0, lambda: (self._set_status(f'Error: {exc}', RED),
                                   self._run_btn.configure(state='normal')))
        finally:
            sys.stdout = old

    def _progress(self, i, total, t):
        self.after(0, lambda: (self._prog.configure(maximum=total, value=i),
                               self._set_status(f'Solving instant {i}/{total} (t={t:.0f}s)...', ORANGE)))

    def _summary(self, out, tag):
        m, res = out['meta'], out['results']
        line = f"[{tag}] {m['case']}: {len(res)} instants, cadence={m['instants']}, removed={sum(r['n_removed'] for r in res)}"
        if m['has_badlog']:
            TP = sum(r['tp'] for r in res); FP = sum(r['fp'] for r in res); FN = sum(r['fn'] for r in res)
            rec = TP / (TP + FN) if (TP + FN) else 1.0; prec = TP / (TP + FP) if (TP + FP) else 1.0
            line += f"  TP={TP} FP={FP} FN={FN} recall={rec:.2f} precision={prec:.2f}"
        print(line)

    def _on_solved(self):
        self._run_btn.configure(state='normal')
        self._rb_clean.configure(state='normal' if 'clean' in self.datasets else 'disabled')
        self._rb_bad.configure(state='normal' if 'bad' in self.datasets else 'disabled')
        # build network layout once (topology shared across twins)
        any_out = next(iter(self.datasets.values()))
        net = any_out['net']; nums = any_out['nums']
        bidx = {n: i for i, n in enumerate(nums)}
        self.edges_idx = [(bidx[br['from_bus']], bidx[br['to_bus']]) for br in net.branches]
        self.pos = spring_layout(net.n, self.edges_idx)
        self._bidx = bidx
        self._which.set(self._default_which)
        self._switch_dataset()
        self._play_btn.configure(state='normal')
        self._set_status('Done. Use Play or the slider; toggle Clean/Bad.', GREEN)

    def _switch_dataset(self):
        self.which = self._which.get()
        self.out = self.datasets.get(self.which)
        if not self.out or not self.out['results']:
            self._set_status('That dataset has no solved instants.', RED); return
        n = len(self.out['results'])
        self._slider.configure(from_=0, to=n - 1)
        self._draw_tracking()
        i = min(int(self._slider.get()), n - 1)
        self._show_instant(i)

    # ── per-instant rendering ────────────────────────────────────────────────────
    def _on_slide(self, _v):
        if self.out and self.out['results']:
            self._show_instant(int(float(self._slider.get())))

    def _show_instant(self, i):
        res = self.out['results']; i = max(0, min(i, len(res) - 1)); r = res[i]
        self._tlabel.configure(text=f"t = {r['t']:.1f} s   ({i+1}/{len(res)})")
        self._update_scorecards(r)
        self._fill_state(r); self._fill_bad(r)
        self._draw_network(r); self._draw_residuals(r)

    def _update_scorecards(self, r):
        m = self.out['meta']
        self._tiles['Case'].configure(text=f"{m['case']}  [{self.which}]")
        obs = r['observable']
        self._tiles['Observable'].configure(text='YES' if obs else 'NO',
                                            foreground=GREEN if obs else RED)
        self._tiles['RMSE |V| after'].configure(text=_e(r['rmse_V']), foreground=FG)
        rb = r.get('rmse_V_before')
        improved = (rb is not None and r['rmse_V'] is not None and rb > 1.5 * r['rmse_V'])
        self._tiles['RMSE |V| before'].configure(text=_e(rb),
                                                 foreground=ORANGE if improved else GRAY)
        self._tiles['Bad now'].configure(text=str(r['n_removed']),
                                         foreground=RED if r['n_removed'] else GREEN)
        res = self.out['results']
        if m['has_badlog']:
            TP = sum(x['tp'] for x in res); FP = sum(x['fp'] for x in res); FN = sum(x['fn'] for x in res)
            rec = TP / (TP + FN) if (TP + FN) else 1.0; prec = TP / (TP + FP) if (TP + FP) else 1.0
            self._tiles['Detection (run)'].configure(text=f"P={prec:.2f} R={rec:.2f}", foreground=FG)
        else:
            self._tiles['Detection (run)'].configure(text='clean stream', foreground=GRAY)

    def _fill_state(self, r):
        for it in self._state.get_children():
            self._state.delete(it)
        if r['V'] is None:
            return
        nums = self.out['nums']; truth = self.out['truth']
        Vt = THt = None
        if truth is not None and round(r['t'], 6) in truth:
            Vt, THt = truth[round(r['t'], 6)]
        for k, n in enumerate(nums):
            dv = (r['V'][k] - Vt[k]) if Vt is not None else 0.0
            self._state.insert('', 'end', tags=('warn' if (Vt is not None and abs(dv) > 0.01) else '',),
                               values=(n, f"{r['V'][k]:.5f}", f"{np.rad2deg(r['theta'][k]):.3f}",
                                       f"{Vt[k]:.5f}" if Vt is not None else '—',
                                       f"{dv:+.5f}" if Vt is not None else '—'))

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
            self._bad.insert('', 'end', tags=(tag,),
                             values=(rm['label'], f"{rm['r_n']:+.2f}", f"{rm['value']:.5f}", status))
        if r['tp'] is not None:
            removed = {str(rm['key']) for rm in r['removed']}
            for k in sorted(actual - removed):
                self._bad.insert('', 'end', tags=('fn',), values=(k, '—', '—', 'missed (FN)'))

    def _bad_nodes_edges(self, r):
        """Bus indices and edge index-pairs flagged bad at this instant."""
        nodes, edges = set(), set()
        for rm in r['removed']:
            if 'bus' in rm:
                if rm['bus'] in self._bidx:
                    nodes.add(self._bidx[rm['bus']])
            elif rm.get('from_bus') in self._bidx and rm.get('to_bus') in self._bidx:
                edges.add(frozenset((self._bidx[rm['from_bus']], self._bidx[rm['to_bus']])))
        return nodes, edges

    def _draw_network(self, r):
        for w in self._net_tab.winfo_children():
            w.destroy()
        net = self.out['net']; nums = self.out['nums']; pos = self.pos
        bad_nodes, bad_edges = self._bad_nodes_edges(r)
        pmu = set(self.out['meta']['pmu_buses'])
        fig = Figure(figsize=(9, 6.4), facecolor=BG)
        ax = fig.add_subplot(111, facecolor=BG_MID); ax.axis('off')
        ax.set_title(f"{self.out['meta']['case']} [{self.which}]  ·  t={r['t']:.1f}s  ·  "
                     f"{r['n_removed']} bad removed", color=FG, fontsize=11, fontweight='bold')
        # edges
        for (i, j) in self.edges_idx:
            bad = frozenset((i, j)) in bad_edges
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=RED if bad else '#33486B', lw=3.0 if bad else 1.3,
                    zorder=2 if bad else 1)
        # nodes coloured by |V|
        V = r['V'] if r['V'] is not None else np.full(len(nums), np.nan)
        if r['V'] is not None:
            sc = ax.scatter(pos[:, 0], pos[:, 1], c=V, cmap='coolwarm', vmin=0.94, vmax=1.10,
                            s=320, zorder=3, edgecolors='#0A1628', linewidths=1.0)
            fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02, label='|V| (pu)')
        else:
            ax.scatter(pos[:, 0], pos[:, 1], c=GRAY, s=320, zorder=3)
        # PMU rings + bad rings + labels
        for k, n in enumerate(nums):
            if n in pmu:
                ax.scatter(pos[k, 0], pos[k, 1], s=560, facecolors='none',
                           edgecolors=GREEN, linewidths=2.0, zorder=4)
            if k in bad_nodes:
                ax.scatter(pos[k, 0], pos[k, 1], s=720, facecolors='none',
                           edgecolors=RED, linewidths=2.6, zorder=5)
            ax.annotate(str(n), (pos[k, 0], pos[k, 1]), color=FG, fontsize=7,
                        ha='center', va='center', zorder=6)
        ax.scatter([], [], s=120, facecolors='none', edgecolors=GREEN, label='PMU bus')
        ax.scatter([], [], s=120, facecolors='none', edgecolors=RED, label='bad data')
        ax.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=8, loc='upper right')
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=self._net_tab); c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

    def _draw_residuals(self, r):
        for w in self._rframe.winfo_children():
            w.destroy()
        if r['r_n_first'] is None:
            return
        rn = np.abs(r['r_n_first']); labels = r['labels']
        removed = {rm['label'] for rm in r['removed']}
        fig = Figure(figsize=(10, 2.7), facecolor=BG)
        ax = fig.add_subplot(111, facecolor=BG_MID)
        for sp in ax.spines.values():
            sp.set_color(BG_LIGHT)
        ax.tick_params(colors=GRAY)
        colors = [RED if labels[i] in removed else BLUE for i in range(len(rn))]
        ax.bar(np.arange(len(rn)), rn, color=colors, zorder=3)
        ax.axhline(float(self._thr.get() or 3.0), color=ORANGE, ls='--', lw=1.3, label='threshold')
        ax.set_title('Normalized residuals at this instant (red = removed)', color=FG, fontsize=10)
        ax.set_ylabel('|r_n|', color=GRAY, fontsize=9); ax.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=8)
        ax.grid(axis='y', color=BG_LIGHT, alpha=.5)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=self._rframe); c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

    def _draw_tracking(self):
        for w in self._track_tab.winfo_children():
            w.destroy()
        out = self.out; res = out['results']; m = out['meta']
        t = np.array([r['t'] for r in res]); nums = out['nums']; idx = {n: i for i, n in enumerate(nums)}
        buses = (m['pmu_buses'] or nums)[:3]
        fig = Figure(figsize=(10, 6.6), facecolor=BG)
        a0 = fig.add_subplot(211, facecolor=BG_MID); a1 = fig.add_subplot(212, facecolor=BG_MID, sharex=a0)
        for ax in (a0, a1):
            for sp in ax.spines.values():
                sp.set_color(BG_LIGHT)
            ax.tick_params(colors=GRAY); ax.grid(alpha=.3, color=BG_LIGHT)
        Vest = np.array([r['V'] if r['V'] is not None else [np.nan]*len(nums) for r in res])
        for bn in buses:
            a0.plot(t, Vest[:, idx[bn]], '--', lw=1.1, label=f'bus {bn} est')
        if out['truth'] is not None:
            for bn in buses:
                vt = np.array([out['truth'][round(tt, 6)][0][idx[bn]]
                               if round(tt, 6) in out['truth'] else np.nan for tt in t])
                a0.plot(t, vt, '-', lw=1.0, alpha=.55, label=f'bus {bn} true')
        a0.set_ylabel('|V| (pu)', color=GRAY)
        a0.set_title(f"{m['case']} [{self.which}]: estimate vs truth", color=FG)
        a0.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=7, ncol=3)
        a1.step(t, [r['n_removed'] for r in res], where='mid', color=ORANGE, label='removed (detected)')
        if m['has_badlog']:
            a1.step(t, [len(r['actual_bad']) for r in res], where='mid', color=RED, alpha=.6,
                    label='actual bad (in SE set)')
        a1.set_xlabel('time (s)', color=GRAY); a1.set_ylabel('# bad data', color=GRAY)
        a1.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=8)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=self._track_tab); c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

    # ── animation ─────────────────────────────────────────────────────────────────
    def _toggle_play(self):
        if self._playing:
            self._pause()
        elif self.out and self.out['results']:
            self._playing = True; self._play_btn.configure(text='⏸  Pause')
            self._advance()

    def _advance(self):
        if not self._playing or not self.out:
            return
        n = len(self.out['results'])
        i = (int(float(self._slider.get())) + 1) % n
        self._slider.set(i)            # triggers _on_slide -> _show_instant
        self._after_id = self.after(SPEEDS.get(self._speed.get(), 380), self._advance)

    def _pause(self):
        self._playing = False
        self._play_btn.configure(text='▶  Play')
        if self._after_id:
            self.after_cancel(self._after_id); self._after_id = None

    # ── console helpers ──────────────────────────────────────────────────────────
    def _append(self, text):
        self.after(0, self._do_append, text)

    def _do_append(self, text):
        self._console.configure(state='normal'); self._console.insert('end', text)
        self._console.see('end'); self._console.configure(state='disabled')

    def _clear_console(self):
        self._console.configure(state='normal'); self._console.delete('1.0', 'end')
        self._console.configure(state='disabled')

    def _set_status(self, text, color=GRAY):
        self._status.configure(text=text, foreground=color)


def _e(v):
    return f"{v:.2e}" if v is not None else '—'


if __name__ == '__main__':
    SolverApp().mainloop()
