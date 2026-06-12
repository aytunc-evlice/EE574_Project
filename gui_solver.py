"""
EE574 - Time-Series State Estimator (viewer for GENERATED measurement streams)
==============================================================================
Loads a stream produced by gen_measurements.py / inject_bad_data.py, lets you
toggle SCADA / PMU, solves AC-WLS at every (consistent) time instant with
iterative bad-data removal, and lets you scrub through time to see the estimate
and exactly which measurements were flagged as bad (checked against bad_log.csv
when present).  It does NOT generate or inject data.

Run:  python gui_solver.py
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
from solve_timeseries import solve_stream

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

BG, BG_MID, BG_LIGHT = '#0A1628', '#0D2B5E', '#1A3A6B'
FG, ORANGE, GREEN, RED, GRAY = '#FFFFFF', '#E87722', '#44CC88', '#FF4444', '#8899AA'
CONSOLE = '#060E1C'


class _Stdout:
    def __init__(self, cb): self._cb = cb
    def write(self, t): self._cb(t)
    def flush(self): pass


class SolverApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('EE574  ·  Time-Series State Estimator')
        self.minsize(1180, 800)
        self.out = None
        self._build_styles()
        self._build_ui()

    # ── styling ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self); s.theme_use('clam')
        self.configure(bg=BG)
        s.configure('TFrame', background=BG)
        s.configure('Card.TFrame', background=BG_MID)
        s.configure('TLabel', background=BG, foreground=FG, font=('Segoe UI', 10))
        s.configure('Title.TLabel', background=BG_MID, foreground=FG, font=('Segoe UI', 14, 'bold'))
        s.configure('Sub.TLabel', background=BG, foreground=GRAY, font=('Segoe UI', 9))
        s.configure('TCheckbutton', background=BG, foreground=FG, font=('Segoe UI', 10))
        s.map('TCheckbutton', background=[('active', BG)])
        s.configure('TButton', background=BG_LIGHT, foreground=FG, font=('Segoe UI', 9))
        s.configure('Run.TButton', background=ORANGE, foreground=FG, font=('Segoe UI', 11, 'bold'))
        s.map('Run.TButton', background=[('active', '#C96010'), ('disabled', '#333')])
        s.configure('TNotebook', background=BG)
        s.configure('TNotebook.Tab', background=BG_MID, foreground=GRAY, padding=[14, 6])
        s.map('TNotebook.Tab', background=[('selected', BG_LIGHT)], foreground=[('selected', FG)])
        s.configure('Treeview', background=BG_MID, foreground=FG, fieldbackground=BG_MID,
                    font=('Consolas', 9), rowheight=21)
        s.configure('Treeview.Heading', background=BG_LIGHT, foreground=ORANGE, font=('Segoe UI', 9, 'bold'))
        s.configure('TProgressbar', troughcolor=BG, background=ORANGE)
        s.configure('Horizontal.TScale', background=BG)

    # ── layout ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        hdr = ttk.Frame(self, style='Card.TFrame'); hdr.pack(fill='x')
        ttk.Label(hdr, text='Time-Series State Estimator', style='Title.TLabel',
                  background=BG_MID).pack(side='left', padx=16, pady=10)
        ttk.Label(hdr, text='load a generated stream · solve · detect & remove bad data',
                  background=BG_MID, foreground=GRAY, font=('Segoe UI', 9, 'italic')).pack(side='left')

        # controls
        ctl = ttk.Frame(self); ctl.pack(fill='x', padx=16, pady=8)
        ttk.Label(ctl, text='Stream folder:').grid(row=0, column=0, sticky='w')
        self._dir = tk.StringVar(value=os.path.join(DATA_DIR, 'results', 'timeseries'))
        ttk.Entry(ctl, textvariable=self._dir, font=('Consolas', 9), width=70).grid(
            row=0, column=1, columnspan=4, sticky='ew', padx=8)
        ttk.Button(ctl, text='Browse', command=self._browse).grid(row=0, column=5, padx=2)

        self._use_scada = tk.BooleanVar(value=True)
        self._use_pmu = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text='SCADA', variable=self._use_scada).grid(row=1, column=0, sticky='w', pady=8)
        ttk.Checkbutton(ctl, text='PMU', variable=self._use_pmu).grid(row=1, column=1, sticky='w')
        ttk.Label(ctl, text='Threshold:').grid(row=1, column=2, sticky='e')
        self._thr = tk.StringVar(value='3.0')
        ttk.Entry(ctl, textvariable=self._thr, width=6).grid(row=1, column=3, sticky='w', padx=4)
        ttk.Label(ctl, text='Max removals/instant:').grid(row=1, column=4, sticky='e')
        self._maxr = tk.StringVar(value='6')
        ttk.Entry(ctl, textvariable=self._maxr, width=6).grid(row=1, column=5, sticky='w', padx=4)
        ctl.columnconfigure(1, weight=1)

        run = ttk.Frame(self); run.pack(fill='x', padx=16)
        self._run_btn = ttk.Button(run, text='▶  Solve stream', style='Run.TButton', command=self._start)
        self._run_btn.pack(side='left')
        self._status = ttk.Label(run, text='Select a stream folder and solve.', style='Sub.TLabel')
        self._status.pack(side='left', padx=14)
        self._prog = ttk.Progressbar(run, mode='determinate', length=220)
        self._prog.pack(side='left')

        nb = ttk.Notebook(self); nb.pack(fill='both', expand=True, padx=16, pady=(8, 12))
        self._nb = nb
        self._build_instant_tab(nb)
        self._build_tracking_tab(nb)
        cf = ttk.Frame(nb); nb.add(cf, text=' Console ')
        self._console = scrolledtext.ScrolledText(cf, state='disabled', bg=CONSOLE, fg='#B8D0F0',
                                                   font=('Consolas', 9), relief='flat')
        self._console.pack(fill='both', expand=True, padx=4, pady=4)

    def _build_instant_tab(self, nb):
        tab = ttk.Frame(nb); nb.add(tab, text=' Per-Instant ')
        top = ttk.Frame(tab); top.pack(fill='x', pady=6)
        ttk.Label(top, text='Time instant:').pack(side='left', padx=(8, 4))
        self._tvar = tk.IntVar(value=0)
        self._slider = ttk.Scale(top, from_=0, to=0, orient='horizontal',
                                 command=self._on_slide, length=460)
        self._slider.pack(side='left', padx=4)
        self._tlabel = ttk.Label(top, text='—', style='Sub.TLabel'); self._tlabel.pack(side='left', padx=10)
        self._ilabel = ttk.Label(top, text='', font=('Segoe UI', 10, 'bold')); self._ilabel.pack(side='left', padx=10)

        body = ttk.Frame(tab); body.pack(fill='both', expand=True)
        # state table
        lf = ttk.Frame(body); lf.pack(side='left', fill='both', expand=True, padx=6)
        ttk.Label(lf, text='Estimated state', foreground=ORANGE,
                  font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        cols = ('Bus', 'V_est', 'θ_est(°)', 'V_true', 'ΔV')
        self._state = ttk.Treeview(lf, columns=cols, show='headings', height=14)
        for c, w in zip(cols, (50, 90, 90, 90, 90)):
            self._state.heading(c, text=c); self._state.column(c, width=w, anchor='center')
        self._state.pack(fill='both', expand=True)
        self._state.tag_configure('warn', foreground='#FFB830')
        # bad-data table
        rf = ttk.Frame(body); rf.pack(side='left', fill='both', expand=True, padx=6)
        ttk.Label(rf, text='Bad data (removed / missed)', foreground=ORANGE,
                  font=('Segoe UI', 10, 'bold')).pack(anchor='w')
        bcols = ('Measurement', 'r_n', 'value', 'status')
        self._bad = ttk.Treeview(rf, columns=bcols, show='headings', height=14)
        for c, w in zip(bcols, (160, 70, 90, 110)):
            self._bad.heading(c, text=c); self._bad.column(c, width=w,
                              anchor='w' if c == 'Measurement' else 'center')
        self._bad.pack(fill='both', expand=True)
        self._bad.tag_configure('tp', foreground=GREEN)
        self._bad.tag_configure('fp', foreground=ORANGE)
        self._bad.tag_configure('fn', foreground=RED)
        # residual plot
        self._rframe = ttk.Frame(tab); self._rframe.pack(fill='both', expand=True, padx=6, pady=4)

    def _build_tracking_tab(self, nb):
        self._tracking = ttk.Frame(nb); nb.add(self._tracking, text=' Tracking / Detection ')

    # ── actions ─────────────────────────────────────────────────────────────────
    def _browse(self):
        p = filedialog.askdirectory(title='Select a timeseries stream folder',
                                    initialdir=self._dir.get())
        if p:
            self._dir.set(p)

    def _start(self):
        if not (self._use_scada.get() or self._use_pmu.get()):
            self._set_status('Select at least one of SCADA / PMU.', RED); return
        self._run_btn.configure(state='disabled')
        self._set_status('Solving...', ORANGE)
        self._clear_console()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        old = sys.stdout; sys.stdout = _Stdout(self._append)
        try:
            d = self._dir.get().strip()
            out = solve_stream(
                d, use_scada=self._use_scada.get(), use_pmu=self._use_pmu.get(),
                threshold=float(self._thr.get()), max_removals=int(self._maxr.get()),
                progress=self._progress)
            self.out = out
            self._summary(out)
            self.after(0, self._on_solved)
        except Exception as exc:
            import traceback; traceback.print_exc()
            self.after(0, lambda: self._set_status(f'Error: {exc}', RED))
            self.after(0, lambda: self._run_btn.configure(state='normal'))
        finally:
            sys.stdout = old

    def _progress(self, i, total, t):
        self.after(0, lambda: (self._prog.configure(maximum=total, value=i),
                               self._set_status(f'Solving instant {i}/{total} (t={t:.0f}s)...', ORANGE)))

    def _summary(self, out):
        m, res = out['meta'], out['results']
        tot = sum(r['n_removed'] for r in res)
        print(f"\n{m['case']}: {len(res)} instants (SCADA={m['use_scada']}, PMU={m['use_pmu']}, "
              f"cadence={m['instants']})")
        print(f"  bad data removed: {tot}")
        if m['has_badlog']:
            TP = sum(r['tp'] for r in res); FP = sum(r['fp'] for r in res); FN = sum(r['fn'] for r in res)
            rec = TP / (TP + FN) if (TP + FN) else 1.0
            prec = TP / (TP + FP) if (TP + FP) else 1.0
            print(f"  detection vs bad_log: TP={TP} FP={FP} FN={FN}  recall={rec:.2f} precision={prec:.2f}")

    def _on_solved(self):
        res = self.out['results']
        self._run_btn.configure(state='normal')
        if not res:
            self._set_status('No instants solved (check stream / selection).', RED); return
        self._set_status(f"Done: {len(res)} instants, {self.out['meta']['instants']} cadence.", GREEN)
        self._slider.configure(from_=0, to=len(res) - 1)
        self._slider.set(0)
        self._draw_tracking()
        self._show_instant(0)
        self._nb.select(0)

    def _on_slide(self, _v):
        if self.out and self.out['results']:
            self._show_instant(int(float(self._slider.get())))

    # ── per-instant rendering ────────────────────────────────────────────────────
    def _show_instant(self, i):
        res = self.out['results']
        i = max(0, min(i, len(res) - 1))
        r = res[i]
        net = self.out['net']; nums = self.out['nums']
        truth = self.out['truth']
        self._tlabel.configure(text=f"t = {r['t']:.1f} s   (instant {i+1}/{len(res)})")
        obs = 'OBSERVABLE' if r['observable'] else 'NOT OBSERVABLE'
        info = f"{obs}   removed {r['n_removed']}   J={_fmt(r['J'])}"
        if r['tp'] is not None:
            info += f"   TP={r['tp']} FP={r['fp']} FN={r['fn']}"
        self._ilabel.configure(text=info, foreground=GREEN if r['observable'] else RED)

        # state table
        for it in self._state.get_children():
            self._state.delete(it)
        if r['V'] is not None:
            Vt = THt = None
            if truth is not None and round(r['t'], 6) in truth:
                Vt, THt = truth[round(r['t'], 6)]
            for k, n in enumerate(nums):
                vest = r['V'][k]; th = np.rad2deg(r['theta'][k])
                vtru = f"{Vt[k]:.5f}" if Vt is not None else '—'
                dv = (vest - Vt[k]) if Vt is not None else 0.0
                tag = 'warn' if (Vt is not None and abs(dv) > 0.01) else ''
                self._state.insert('', 'end', tags=(tag,), values=(
                    n, f"{vest:.5f}", f"{th:.3f}", vtru,
                    f"{dv:+.5f}" if Vt is not None else '—'))

        # bad-data table: removed (TP/FP) + missed (FN)
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
            removed_keys = {str(rm['key']) for rm in r['removed']}
            for k in sorted(actual - removed_keys):
                self._bad.insert('', 'end', tags=('fn',),
                                 values=(k, '—', '—', 'missed (FN)'))

        self._draw_residuals(r)

    def _draw_residuals(self, r):
        for w in self._rframe.winfo_children():
            w.destroy()
        if r['r_n_first'] is None:
            return
        rn = np.abs(r['r_n_first']); labels = r['labels']
        removed_lbls = {rm['label'] for rm in r['removed']}
        fig = Figure(figsize=(10, 2.8), facecolor=BG)
        ax = fig.add_subplot(111, facecolor=BG_MID)
        for sp in ax.spines.values():
            sp.set_color(BG_LIGHT)
        ax.tick_params(colors=GRAY)
        colors = [RED if labels[i] in removed_lbls else '#4A90D9' for i in range(len(rn))]
        ax.bar(np.arange(len(rn)), rn, color=colors, zorder=3)
        ax.axhline(float(self._thr.get() or 3.0), color=ORANGE, ls='--', lw=1.3, label='threshold')
        ax.set_title('Normalized residuals at this instant (red = removed)', color=FG, fontsize=10)
        ax.set_xlabel('measurement index', color=GRAY, fontsize=9)
        ax.set_ylabel('|r_n|', color=GRAY, fontsize=9)
        ax.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=8); ax.grid(axis='y', color=BG_LIGHT, alpha=.5)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=self._rframe); c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

    def _draw_tracking(self):
        for w in self._tracking.winfo_children():
            w.destroy()
        out = self.out; res = out['results']; m = out['meta']
        t = np.array([r['t'] for r in res])
        nums = out['nums']; idx = {n: i for i, n in enumerate(nums)}
        buses = (m['pmu_buses'] or nums)[:3]
        fig = Figure(figsize=(10, 7), facecolor=BG)
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
        a0.set_ylabel('|V| (pu)', color=GRAY); a0.set_title(f"{m['case']}: estimate vs truth", color=FG)
        a0.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=7, ncol=3)
        a1.step(t, [r['n_removed'] for r in res], where='mid', color=ORANGE, label='removed (detected)')
        if m['has_badlog']:
            a1.step(t, [len(r['actual_bad']) for r in res], where='mid', color=RED, alpha=.6,
                    label='actual bad (in SE set)')
        a1.set_xlabel('time (s)', color=GRAY); a1.set_ylabel('# bad data', color=GRAY)
        a1.legend(facecolor=BG_LIGHT, labelcolor=FG, fontsize=8)
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=self._tracking); c.draw()
        c.get_tk_widget().pack(fill='both', expand=True)

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


def _fmt(v):
    return f"{v:.1f}" if v is not None else '—'


if __name__ == '__main__':
    SolverApp().mainloop()
