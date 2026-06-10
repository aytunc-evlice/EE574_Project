"""
EE574 AC-WLS State Estimator — Graphical Interface
Run:  python gui.py
"""
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import threading
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.parser       import parse_ieee_cdf, parse_measurements
from src.network      import Network
from src.estimator    import wls_estimate
from src.observability import observability_report
from src.bad_data     import compute_normalized_residuals, bad_data_report
from src.scenarios    import scenario_3_bad_data
from src.simulation   import generate_scada_measurements, generate_pmu_measurements
from src.results      import compare_with_loadflow

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── palette (matches poster) ──────────────────────────────────────────────────
BG       = '#0A1628'
BG_MID   = '#0D2B5E'
BG_LIGHT = '#1A3A6B'
FG       = '#FFFFFF'
ORANGE   = '#E87722'
GREEN    = '#44CC88'
RED      = '#FF4444'
GRAY     = '#8899AA'
CONSOLE  = '#060E1C'


class _StdoutRedirect:
    def __init__(self, callback):
        self._cb = callback

    def write(self, text):
        self._cb(text)

    def flush(self):
        pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('EE574  ·  AC-WLS State Estimator')
        self.minsize(1150, 760)
        self._results = {}
        self._build_styles()
        self._build_ui()

    # ── styles ────────────────────────────────────────────────────────────────
    def _build_styles(self):
        s = ttk.Style(self)
        s.theme_use('clam')
        self.configure(bg=BG)
        s.configure('TFrame',      background=BG)
        s.configure('Card.TFrame', background=BG_MID)
        s.configure('TLabel',      background=BG,     foreground=FG,     font=('Segoe UI', 10))
        s.configure('Title.TLabel',background=BG_MID, foreground=FG,     font=('Segoe UI', 14, 'bold'))
        s.configure('Sub.TLabel',  background=BG,     foreground=GRAY,   font=('Segoe UI', 9))
        s.configure('TEntry',      fieldbackground=BG_MID, foreground=FG,
                    insertcolor=FG, bordercolor=BG_LIGHT)
        s.configure('TButton',     background=BG_LIGHT, foreground=FG,   font=('Segoe UI', 9))
        s.map('TButton', background=[('active', BG_LIGHT)])
        s.configure('Run.TButton', background=ORANGE, foreground=FG, font=('Segoe UI', 11, 'bold'))
        s.map('Run.TButton',
              background=[('active', '#C96010'), ('disabled', '#333')],
              foreground=[('disabled', GRAY)])
        s.configure('TNotebook',     background=BG, tabmargins=[0, 0, 0, 0])
        s.configure('TNotebook.Tab', background=BG_MID, foreground=GRAY,
                    padding=[14, 6], font=('Segoe UI', 10))
        s.map('TNotebook.Tab',
              background=[('selected', BG_LIGHT)],
              foreground=[('selected', FG)])
        s.configure('Treeview',         background=BG_MID, foreground=FG,
                    fieldbackground=BG_MID, font=('Consolas', 9), rowheight=22)
        s.configure('Treeview.Heading', background=BG_LIGHT, foreground=ORANGE,
                    font=('Segoe UI', 9, 'bold'))
        s.map('Treeview', background=[('selected', BG_LIGHT)])
        s.configure('TScrollbar', background=BG_MID, troughcolor=BG, arrowcolor=GRAY)
        s.configure('TProgressbar', troughcolor=BG, background=ORANGE)
        s.configure('TLabelframe',       background=BG, bordercolor=BG_LIGHT)
        s.configure('TLabelframe.Label', background=BG, foreground=ORANGE,
                    font=('Segoe UI', 10, 'bold'))

    # ── top-level layout ──────────────────────────────────────────────────────
    def _build_ui(self):
        self._build_header()
        self._build_file_selectors()
        self._build_run_row()
        self._build_notebook()

    def _build_header(self):
        fr = ttk.Frame(self, style='Card.TFrame')
        fr.pack(fill='x')
        ttk.Label(fr, text='AC-WLS State Estimator', style='Title.TLabel',
                  background=BG_MID).pack(side='left', padx=16, pady=10)
        ttk.Label(fr, text='EE574 Real-Time Power System Monitoring',
                  background=BG_MID, foreground=GRAY,
                  font=('Segoe UI', 9, 'italic')).pack(side='left', padx=4)

    def _build_file_selectors(self):
        fr = ttk.Frame(self)
        fr.pack(fill='x', padx=16, pady=(12, 4))

        def row(label, var, browse_cmd, clear_cmd=None, r=0):
            ttk.Label(fr, text=label).grid(row=r, column=0, sticky='w', pady=3)
            e = ttk.Entry(fr, textvariable=var, font=('Consolas', 9))
            e.grid(row=r, column=1, sticky='ew', padx=8)
            ttk.Button(fr, text='Browse', command=browse_cmd).grid(row=r, column=2, padx=2)
            if clear_cmd:
                ttk.Button(fr, text='Clear', command=clear_cmd).grid(row=r, column=3, padx=2)

        self._cdf_var  = tk.StringVar(value=os.path.join(DATA_DIR, 'ieee_cdf_sample.dat'))
        self._meas_var = tk.StringVar(value='')
        self._out_var  = tk.StringVar(value=os.path.join(DATA_DIR, 'results'))

        row('CDF Network File:',            self._cdf_var,  self._browse_cdf,  r=0)
        row('Measurement File (optional):', self._meas_var, self._browse_meas,
            clear_cmd=lambda: self._meas_var.set(''), r=1)
        row('Output Directory:',            self._out_var,  self._browse_out,  r=2)
        fr.columnconfigure(1, weight=1)

    def _build_run_row(self):
        fr = ttk.Frame(self)
        fr.pack(fill='x', padx=16, pady=6)
        self._run_btn = ttk.Button(fr, text='▶  Run Pipeline',
                                   style='Run.TButton', command=self._start_run)
        self._run_btn.pack(side='left')
        self._status_lbl = ttk.Label(fr, text='Ready.', style='Sub.TLabel')
        self._status_lbl.pack(side='left', padx=14)
        self._progress = ttk.Progressbar(fr, mode='indeterminate', length=180)
        self._progress.pack(side='left')

    def _build_notebook(self):
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=16, pady=(0, 12))
        self._nb = nb

        # Console
        cf = ttk.Frame(nb); nb.add(cf, text=' Console ')
        self._console = scrolledtext.ScrolledText(
            cf, state='disabled', bg=CONSOLE, fg='#B8D0F0',
            font=('Consolas', 9), insertbackground=FG, relief='flat', bd=0)
        self._console.pack(fill='both', expand=True, padx=4, pady=4)

        # Summary
        sf = ttk.Frame(nb); nb.add(sf, text=' Summary ')
        self._build_summary_tab(sf)

        # Voltage plot
        vf = ttk.Frame(nb); nb.add(vf, text=' Voltage Profile ')
        self._v_frame = vf

        # Residuals plot
        rf = ttk.Frame(nb); nb.add(rf, text=' Normalized Residuals ')
        self._r_frame = rf

    # ── summary tab ───────────────────────────────────────────────────────────
    def _build_summary_tab(self, parent):
        # Network info card
        info = ttk.LabelFrame(parent, text='Network Info', padding=8)
        info.pack(fill='x', padx=8, pady=8)
        self._info = {}
        fields = [
            ('Buses', 0, 0), ('Branches', 0, 2), ('States', 0, 4),
            ('Slack Bus', 1, 0), ('PMU Buses', 1, 2), ('Observable (Sc2)', 1, 4),
        ]
        for name, row, col in fields:
            ttk.Label(info, text=name + ':', foreground=GRAY,
                      font=('Segoe UI', 9)).grid(row=row, column=col,
                                                  sticky='w', padx=(10, 4), pady=3)
            lbl = ttk.Label(info, text='—', font=('Segoe UI', 10, 'bold'))
            lbl.grid(row=row, column=col+1, sticky='w', padx=(0, 24), pady=3)
            self._info[name] = lbl

        # Scenario comparison table
        sc_lf = ttk.LabelFrame(parent, text='Scenario Comparison', padding=8)
        sc_lf.pack(fill='x', padx=8, pady=4)
        sc_cols = ('Scenario', 'Meas', 'Rank', 'Conv', 'Iters', 'J_WLS', 'RMSE_V (pu)', 'BD')
        self._sc_tree = ttk.Treeview(sc_lf, columns=sc_cols, show='headings', height=4)
        widths = {'Scenario': 300, 'Meas': 55, 'Rank': 75, 'Conv': 55,
                  'Iters': 50, 'J_WLS': 85, 'RMSE_V (pu)': 100, 'BD': 85}
        for c in sc_cols:
            self._sc_tree.heading(c, text=c)
            self._sc_tree.column(c, width=widths[c],
                                 anchor='w' if c == 'Scenario' else 'center')
        self._sc_tree.pack(fill='x')

        # State vector table
        sv_lf = ttk.LabelFrame(parent, text='Estimated State  —  Scenario 2 (clean)', padding=8)
        sv_lf.pack(fill='both', expand=True, padx=8, pady=4)
        sv_cols = ('Bus', 'Name', 'V_true (pu)', 'V_est (pu)', 'θ_est (°)', 'ΔV (pu)')
        self._sv_tree = ttk.Treeview(sv_lf, columns=sv_cols, show='headings')
        sv_widths = {'Bus': 50, 'Name': 130, 'V_true (pu)': 110,
                     'V_est (pu)': 110, 'θ_est (°)': 100, 'ΔV (pu)': 100}
        for c in sv_cols:
            self._sv_tree.heading(c, text=c)
            self._sv_tree.column(c, width=sv_widths[c],
                                 anchor='w' if c == 'Name' else 'center')
        vsb = ttk.Scrollbar(sv_lf, orient='vertical', command=self._sv_tree.yview)
        self._sv_tree.configure(yscrollcommand=vsb.set)
        self._sv_tree.pack(side='left', fill='both', expand=True)
        vsb.pack(side='right', fill='y')
        self._sv_tree.tag_configure('warn', foreground='#FFB830')

    # ── file dialogs ──────────────────────────────────────────────────────────
    def _browse_cdf(self):
        p = filedialog.askopenfilename(
            title='Select CDF Network File',
            filetypes=[('CDF/DAT files', '*.dat'), ('All files', '*.*')],
            initialdir=DATA_DIR)
        if p:
            self._cdf_var.set(p)

    def _browse_meas(self):
        p = filedialog.askopenfilename(
            title='Select Measurement File',
            filetypes=[('DAT files', '*.dat'), ('All files', '*.*')],
            initialdir=DATA_DIR)
        if p:
            self._meas_var.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title='Select Output Directory', initialdir=DATA_DIR)
        if p:
            self._out_var.set(p)

    # ── pipeline runner ───────────────────────────────────────────────────────
    def _start_run(self):
        self._run_btn.configure(state='disabled')
        self._progress.start(12)
        self._set_status('Running...', ORANGE)
        self._clear_console()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        old = sys.stdout
        sys.stdout = _StdoutRedirect(self._append_console)
        try:
            self._run_pipeline()
        except Exception as exc:
            import traceback
            print(f'\n[ERROR] {exc}\n')
            traceback.print_exc()
        finally:
            sys.stdout = old
            self.after(0, self._finish_run)

    def _finish_run(self):
        self._progress.stop()
        self._run_btn.configure(state='normal')
        if self._results.get('ok'):
            self._set_status('Complete.', GREEN)
            self._update_gui()
            self._nb.select(1)
        else:
            self._set_status('Error — see Console tab.', RED)

    # ── core pipeline (runs in worker thread) ─────────────────────────────────
    def _run_pipeline(self):
        np.random.seed(42)
        res = self._results = {}

        cdf_path  = self._cdf_var.get().strip()
        meas_path = self._meas_var.get().strip()
        out_dir   = self._out_var.get().strip()

        if not os.path.isfile(cdf_path):
            raise FileNotFoundError(f'CDF file not found:\n  {cdf_path}')
        os.makedirs(out_dir, exist_ok=True)

        # 1. Parse ─────────────────────────────────────────────────────────────
        print('\n[1] Parsing network ...')
        buses, branches, mva_base = parse_ieee_cdf(cdf_path)
        net = Network(buses, branches)
        slack = net.buses[net.slack_idx]
        print(f'    {net.n} buses,  {len(branches)} branches,  MVA base = {mva_base:.0f} MVA')
        print(f'    Slack bus : {slack["num"]}  ({slack["name"].strip()})')
        print(f'    States    : {net.n_states}  ({net.n-1} angles + {net.n} voltages)')

        _DEFAULT = os.path.join(DATA_DIR, 'ieee_cdf_sample.dat')
        if os.path.abspath(cdf_path) == os.path.abspath(_DEFAULT):
            pmu_buses = [2, 5, 9, 10, 12, 14]
        else:
            _ns = [b['num'] for b in net.buses if b['type'] != 3]
            pmu_buses = _ns[::3] if _ns else []
        print(f'    PMU buses : {pmu_buses}')

        valid_buses = {b['num'] for b in buses}
        valid_br    = {(br['from_bus'], br['to_bus']) for br in branches}
        for br in branches:
            valid_br.add((br['to_bus'], br['from_bus']))
        slack_num = slack['num']

        # Original measurements (optional)
        has_meas = bool(meas_path) and os.path.isfile(meas_path)
        if has_meas:
            orig_meas = parse_measurements(meas_path, valid_buses, valid_br, max_sections=6)
            orig_meas = [m for m in orig_meas
                         if not (m['type'] == 'Vang' and m.get('bus') == slack_num)]
            print(f'\n    Original SCADA ({os.path.basename(meas_path)}): {len(orig_meas)} measurements')
        else:
            orig_meas = []
            if meas_path:
                print(f'\n    Measurement file not found: {meas_path}')
            print('\n    No measurement file — Scenario 1 will be skipped.')

        # Synthetic measurements
        synth_scada = generate_scada_measurements(net, sigma_vmag=0.005, sigma_pq=0.01, seed=42)
        synth_pmu   = generate_pmu_measurements(net, pmu_buses, sigma_pmu=0.0001, seed=42)
        synth_full  = synth_scada + synth_pmu
        print(f'\n    Synthetic SCADA : {len(synth_scada)} measurements')
        print(f'    Synthetic PMU   : {len(synth_pmu)} measurements  →  total {len(synth_full)}')

        # 2. Observability ─────────────────────────────────────────────────────
        print('\n[2] Observability analysis ...')
        if orig_meas:
            obs1, rank1 = observability_report(net, orig_meas)
        else:
            obs1, rank1 = False, 0
        obs2, rank2 = observability_report(net, synth_full)

        # 3. Scenario 1 ────────────────────────────────────────────────────────
        if orig_meas:
            print(f'\n[3] Scenario 1 — Original SCADA  '
                  f'(rank {rank1}/{net.n_states})')
            x1, conv1, iters1, _, J1 = wls_estimate(
                net, orig_meas, verbose=True, use_loadflow_start=True)
            rmse1_V, _ = compare_with_loadflow(net, x1, label='Scenario 1')
        else:
            print('\n[3] Scenario 1: SKIPPED (no measurement file)')
            x1, conv1, iters1, J1, rmse1_V = None, False, 0, 0.0, None

        # 4. Scenario 2 ────────────────────────────────────────────────────────
        print(f'\n[4] Scenario 2 — Synthetic SCADA + PMU  '
              f'(rank {rank2}/{net.n_states})')
        x2, conv2, iters2, _, J2 = wls_estimate(
            net, synth_full, verbose=True, use_loadflow_start=True)
        rmse2_V, rmse2_th = (None, None)
        r_n2 = None
        if conv2:
            rmse2_V, rmse2_th = compare_with_loadflow(net, x2, label='Scenario 2')
            _, r_n2, _ = compute_normalized_residuals(net, synth_full, x2)
            flagged = sum(1 for r in r_n2 if abs(r) > 3.0)
            print(f'    Measurements exceeding threshold 3.0: {flagged}')

        # 5. Scenario 3 ────────────────────────────────────────────────────────
        print('\n[5] Scenario 3 — Injected gross error + bad data detection')
        target = next(
            (i, m) for i, m in enumerate(synth_full)
            if m['type'] == 'Pflow'
            and m.get('from_bus') not in pmu_buses
            and m.get('to_bus')   not in pmu_buses
        )
        bad_idx, bad_m = target
        s3_meas, _ = scenario_3_bad_data(synth_full, bad_idx, gross_error_pu=0.5)
        print(f'    +0.5 pu injected into measurement [{bad_idx}]  '
              f'Pflow({bad_m["from_bus"]}→{bad_m["to_bus"]})')

        x3, conv3, iters3, _, J3 = wls_estimate(
            net, s3_meas, verbose=True, use_loadflow_start=True)
        suspects = bad_data_report(net, s3_meas, x3)
        bd_detected = bool(suspects)
        bd_correct  = bool(suspects and suspects[0][0] == bad_idx)
        r_n3 = None
        if conv3:
            _, r_n3, _ = compute_normalized_residuals(net, s3_meas, x3)

        # Store everything for GUI display
        res.update(
            ok=True,
            net=net, pmu_buses=pmu_buses,
            obs1=obs1, rank1=rank1,
            obs2=obs2, rank2=rank2,
            orig_meas=orig_meas,
            x1=x1, conv1=conv1, iters1=iters1, J1=J1,
            x2=x2, conv2=conv2, iters2=iters2, J2=J2,
            rmse2_V=rmse2_V, rmse2_th=rmse2_th,
            x3=x3, conv3=conv3, iters3=iters3, J3=J3,
            synth_full=synth_full, s3_meas=s3_meas,
            r_n2=r_n2, r_n3=r_n3,
            bd_detected=bd_detected, bd_correct=bd_correct, bad_m=bad_m,
        )
        print('\nPipeline complete.')

    # ── GUI update (main thread) ───────────────────────────────────────────────
    def _update_gui(self):
        res = self._results
        net = res['net']

        # Network info
        self._info['Buses'].config(text=str(net.n))
        self._info['Branches'].config(text=str(len(net.branches)))
        self._info['States'].config(text=str(net.n_states))
        self._info['Slack Bus'].config(
            text=f'{net.buses[net.slack_idx]["num"]}  '
                 f'({net.buses[net.slack_idx]["name"].strip()})')
        self._info['PMU Buses'].config(text=str(res['pmu_buses']))
        obs_ok = res['obs2']
        self._info['Observable (Sc2)'].config(
            text='YES  ✓' if obs_ok else 'NO',
            foreground=GREEN if obs_ok else RED)

        # Scenario table
        for row in self._sc_tree.get_children():
            self._sc_tree.delete(row)

        def fmt(v, fmt_str='.1f'):
            return format(v, fmt_str) if v is not None else '—'

        if res['orig_meas']:
            self._sc_tree.insert('', 'end', values=(
                'Sc1: Original SCADA',
                len(res['orig_meas']),
                f'{res["rank1"]}/{net.n_states}',
                'YES' if res['conv1'] else 'NO',
                res['iters1'], fmt(res['J1']), '—', 'N/A',
            ))

        self._sc_tree.insert('', 'end', values=(
            'Sc2: Synthetic SCADA + PMU (clean)',
            len(res['synth_full']),
            f'{res["rank2"]}/{net.n_states}',
            'YES ✓' if res['conv2'] else 'NO',
            res['iters2'],
            fmt(res['J2']) if res['conv2'] else '—',
            fmt(res['rmse2_V'], '.5f') if res['rmse2_V'] else '—',
            '—',
        ))
        self._sc_tree.insert('', 'end', values=(
            'Sc3: Bad data injected',
            len(res['s3_meas']),
            f'{res["rank2"]}/{net.n_states}',
            'YES ✓' if res['conv3'] else 'NO',
            res['iters3'],
            fmt(res['J3']) if res['conv3'] else '—',
            '—',
            ('YES ✓' if res['bd_correct'] else 'DETECTED') if res['bd_detected'] else 'NO',
        ))

        # State vector (Scenario 2)
        for row in self._sv_tree.get_children():
            self._sv_tree.delete(row)
        if res['conv2'] and res['x2'] is not None:
            x2 = res['x2']
            n  = net.n
            angles = np.concatenate([[0.0], x2[:n-1]])
            volts  = x2[n-1:]
            for i, bus in enumerate(net.buses):
                lf_V  = bus.get('Vm', 1.0)
                est_V = volts[i]
                dV    = est_V - lf_V
                tag   = 'warn' if abs(dV) > 0.01 else ''
                self._sv_tree.insert('', 'end', values=(
                    bus['num'],
                    bus.get('name', '').strip(),
                    f'{lf_V:.5f}',
                    f'{est_V:.5f}',
                    f'{np.degrees(angles[i]):.4f}',
                    f'{dV:+.5f}',
                ), tags=(tag,))

        self._draw_voltage_plot()
        self._draw_residuals_plot()

    # ── embedded plots ─────────────────────────────────────────────────────────
    def _draw_voltage_plot(self):
        for w in self._v_frame.winfo_children():
            w.destroy()

        res = self._results
        net = res['net']
        n   = net.n
        bus_nums = [b['num'] for b in net.buses]
        lf_V = np.array([b.get('Vm', 1.0) for b in net.buses])

        fig = Figure(figsize=(10, 4.8), facecolor=BG)
        ax  = fig.add_subplot(111, facecolor=BG_MID)
        for sp in ax.spines.values():
            sp.set_color(BG_LIGHT)
        ax.tick_params(colors=GRAY, which='both')
        ax.set_xlabel('Bus Number', color=GRAY, fontsize=10)
        ax.set_ylabel('Voltage Magnitude (pu)', color=GRAY, fontsize=10)
        ax.set_title('Voltage Profile  —  Estimated vs. Load-Flow Truth',
                     color=FG, fontsize=11, fontweight='bold')

        x  = np.arange(n)
        w  = 0.22
        ax.bar(x - w, lf_V, width=w, label='Load-flow (truth)',
               color='#3A7BD5', alpha=0.9, zorder=3)
        if res.get('conv2') and res['x2'] is not None:
            ax.bar(x, res['x2'][n-1:], width=w, label='Sc2: Clean SCADA+PMU',
                   color=GREEN, alpha=0.9, zorder=3)
        if res.get('conv3') and res['x3'] is not None:
            ax.bar(x + w, res['x3'][n-1:], width=w, label='Sc3: After BD removal',
                   color=ORANGE, alpha=0.9, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(bus_nums, color=GRAY, fontsize=8)
        ax.grid(True, axis='y', color=BG_LIGHT, alpha=0.7, zorder=0)
        ax.legend(facecolor=BG_LIGHT, edgecolor=BG_LIGHT, labelcolor=FG, fontsize=9)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._v_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(canvas, self._v_frame).update()

    def _draw_residuals_plot(self):
        for w in self._r_frame.winfo_children():
            w.destroy()

        res = self._results
        plots = []
        if res.get('r_n2') is not None:
            plots.append(('Sc2: Clean SCADA + PMU',  res['r_n2'], GREEN))
        if res.get('r_n3') is not None:
            plots.append(('Sc3: Bad data injected',   res['r_n3'], ORANGE))

        if not plots:
            ttk.Label(self._r_frame, text='No residuals available.').pack(pady=20)
            return

        fig = Figure(figsize=(10, 4.8), facecolor=BG)
        for idx, (title, r_n, color) in enumerate(plots):
            ax = fig.add_subplot(1, len(plots), idx + 1, facecolor=BG_MID)
            for sp in ax.spines.values():
                sp.set_color(BG_LIGHT)
            ax.tick_params(colors=GRAY)

            abs_r = np.abs(r_n)
            bar_colors = [RED if v > 3.0 else color for v in abs_r]
            ax.bar(np.arange(len(r_n)), abs_r, color=bar_colors, alpha=0.85, zorder=3)
            ax.axhline(3.0, color=RED, linestyle='--', linewidth=1.5,
                       label='Threshold 3.0', zorder=4)
            ax.set_xlabel('Measurement Index', color=GRAY, fontsize=10)
            ax.set_ylabel('|Normalized Residual|', color=GRAY, fontsize=10)
            ax.set_title(title, color=FG, fontsize=11, fontweight='bold')
            ax.legend(facecolor=BG_LIGHT, edgecolor=BG_LIGHT, labelcolor=FG, fontsize=9)
            ax.grid(True, axis='y', color=BG_LIGHT, alpha=0.7, zorder=0)
            ax.set_ylim(bottom=0)
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self._r_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)
        NavigationToolbar2Tk(canvas, self._r_frame).update()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _append_console(self, text):
        self.after(0, self._do_append, text)

    def _do_append(self, text):
        self._console.configure(state='normal')
        self._console.insert('end', text)
        self._console.see('end')
        self._console.configure(state='disabled')

    def _clear_console(self):
        self._console.configure(state='normal')
        self._console.delete('1.0', 'end')
        self._console.configure(state='disabled')

    def _set_status(self, text, color=GRAY):
        self._status_lbl.configure(text=text, foreground=color)


if __name__ == '__main__':
    app = App()
    app.mainloop()
