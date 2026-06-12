"""
EE574 - Solve a GENERATED measurement stream with iterative bad-data removal.
=============================================================================
Reads a stream produced by gen_measurements.py / inject_bad_data.py (clean or
*_bad), and at every time instant fuses the selected SCADA / PMU measurements,
runs AC-WLS state estimation, and if bad data is present it reports, removes,
and re-estimates in a loop (Largest Normalized Residual).  No data is generated
or injected here.

SCADA / PMU are independently selectable: use both (fused), or only one.

    python solve_timeseries.py --dir results/timeseries/case_RAND20_bad
    python solve_timeseries.py --dir results/timeseries/case_5BUS --no-pmu
    python solve_timeseries.py --dir results/timeseries/case_IEEE14_bad --no-scada

Outputs in --dir: se_results.csv, se_removed.csv, and tracking/detection plots.
"""
import os
import sys
import csv
import json
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser    import parse_ieee_cdf, parse_measurement_file, fuse
from src.network   import Network
from src.se_solver import estimate_with_bad_data, meas_key

# measurement types the WLS estimator actually uses (others can't be detected)
SE_TYPES = {'Vmag', 'Vang', 'Pinj', 'Qinj', 'Pflow', 'Qflow'}


# ── configuration / network ───────────────────────────────────────────────────
def load_manifest(d):
    """Best-effort manifest for a stream dir (clean has manifest.json; a *_bad
    dir has manifest_bad.json pointing at its clean source).  {} if none."""
    mpath = os.path.join(d, 'manifest.json')
    if os.path.exists(mpath):
        return json.load(open(mpath))
    mbpath = os.path.join(d, 'manifest_bad.json')
    if os.path.exists(mbpath):
        clean = json.load(open(mbpath)).get('source_dir', '')
        cm = os.path.join(clean, 'manifest.json')
        if os.path.exists(cm):
            return json.load(open(cm))
    return {}


def resolve_config(d, cdf_path=None, loads_in_pu=None, pmu_buses=None):
    """
    Resolve CDF / loads_in_pu / pmu_buses for a stream dir.  Explicit arguments
    win over the manifest; if there is no manifest, a CDF must be supplied.
    """
    cfg = dict(load_manifest(d))
    if cdf_path:
        cfg['cdf_file'] = cdf_path
    if loads_in_pu is not None:
        cfg['loads_in_pu'] = loads_in_pu
    if pmu_buses is not None:
        cfg['pmu_buses'] = pmu_buses
    if not cfg.get('cdf_file'):
        raise ValueError("no manifest in this folder - please select a CDF file")
    if not os.path.exists(cfg['cdf_file']):
        raise FileNotFoundError(f"CDF file not found: {cfg['cdf_file']}")
    cfg.setdefault('loads_in_pu', False)
    cfg.setdefault('case', os.path.basename(d.rstrip('/\\')))
    cfg.setdefault('pmu_buses', None)
    return cfg


def _time_from_name(fname):
    """'pmu_t00012_00.dat' -> 12.0"""
    try:
        tag = fname.split('_t', 1)[1].rsplit('.', 1)[0]   # '00012_00'
        return float(tag.replace('_', '.'))
    except Exception:
        return 0.0


def resolve_index(d):
    """index.csv rows if present, else auto-discover pmu/ and scada/ *.dat."""
    p = os.path.join(d, 'index.csv')
    if os.path.exists(p):
        return list(csv.DictReader(open(p)))
    rows = []
    for kind in ('pmu', 'scada'):
        for f in sorted(glob.glob(os.path.join(d, kind, '*.dat'))):
            base = os.path.basename(f)
            rows.append({'t_sec': str(_time_from_name(base)), 'kind': kind,
                         'file': f"{kind}/{base}", 'lambda': ''})
    if not rows:
        raise ValueError("no index.csv and no pmu/ or scada/ *.dat files found")
    return rows


def build_network(cfg):
    buses, branches, mva = parse_ieee_cdf(cfg['cdf_file'])
    if cfg.get('loads_in_pu'):
        for b in buses:
            b['P_load'] *= mva; b['Q_load'] *= mva
            b['P_gen'] *= mva; b['Q_gen'] *= mva
    return Network(buses, branches)


# ── ground-truth helpers ──────────────────────────────────────────────────────
def load_truth(d, nums):
    p = os.path.join(d, 'truth.csv')
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        t = round(float(r['t_sec']), 6)
        out[t] = (np.array([float(r[f'V{n}']) for n in nums]),
                  np.array([float(r[f'Vang_rad{n}']) for n in nums]))
    return out


def load_badlog(d):
    """{relative_file: set of SE-relevant bad keys}."""
    p = os.path.join(d, 'bad_log.csv')
    if not os.path.exists(p):
        return None
    out = {}
    for r in csv.DictReader(open(p)):
        if r['type'] not in SE_TYPES:
            continue
        loc2 = r['loc2']
        key = (r['type'], int(r['loc1']), None if loc2 in ('', None) else int(loc2))
        out.setdefault(r['file'], set()).add(key)
    return out


# ── core: solve the whole stream ───────────────────────────────────────────────
def solve_stream(d, use_scada=True, use_pmu=True, threshold=3.0, max_removals=6,
                 instants=None, progress=None, cdf_path=None, loads_in_pu=None,
                 pmu_buses=None):
    if not (use_scada or use_pmu):
        raise ValueError("select at least one of SCADA / PMU")

    cfg = resolve_config(d, cdf_path, loads_in_pu, pmu_buses)
    net = build_network(cfg)
    nums = [b['num'] for b in net.buses]
    slack_num = net.buses[net.slack_idx]['num']        # derive from net, not manifest
    vb = {b['num'] for b in net.buses}
    bs = ({(br['from_bus'], br['to_bus']) for br in net.branches}
          | {(br['to_bus'], br['from_bus']) for br in net.branches})

    rows = resolve_index(d)
    pmu = {round(float(r['t_sec']), 6): r['file'] for r in rows if r['kind'] == 'pmu'}
    scada = {round(float(r['t_sec']), 6): r['file'] for r in rows if r['kind'] == 'scada'}
    lam = {round(float(r['t_sec']), 6): (float(r['lambda']) if r.get('lambda') else None)
           for r in rows}

    # PMU buses: explicit/manifest, else infer from a PMU snapshot's Vmag/Vang
    if not cfg.get('pmu_buses') and pmu:
        first = next(iter(pmu.values()))
        pm = parse_measurement_file(os.path.join(d, *first.split('/')), vb, bs)
        cfg['pmu_buses'] = sorted({m['bus'] for m in pm
                                   if m['type'] in ('Vmag', 'Vang') and 'bus' in m})
    cfg.setdefault('pmu_buses', [])

    truth = load_truth(d, nums)
    badlog = load_badlog(d)

    # Estimation instants.  Default to the SLOWEST selected stream's cadence so
    # every estimate uses time-consistent data: fusing fresh PMU (1 s) with a
    # stale SCADA scan (up to 5 s old) injects spurious inconsistencies that the
    # bad-data test would flag as false positives.  At SCADA instants both
    # streams are fresh.  ('pmu'/'all' remain available but carry SCADA staleness.)
    if instants is None:
        instants = 'scada' if use_scada else 'pmu'
    if instants == 'pmu':
        times = sorted(pmu)
    elif instants == 'scada':
        times = sorted(scada)
    else:
        times = sorted(set(pmu) | set(scada))

    scada_times = sorted(scada)

    def read(rel):
        return parse_measurement_file(os.path.join(d, *rel.split('/')), vb, bs)

    results = []
    x_prev = None
    for i, t in enumerate(times):
        sets, used = [], {}
        if use_scada and scada_times:
            sfile = scada.get(t) or scada[max(s for s in scada_times if s <= t)]
            sets.append(read(sfile)); used['scada'] = sfile
        if use_pmu and t in pmu:
            sets.append(read(pmu[t])); used['pmu'] = pmu[t]

        meas = fuse(*sets)
        meas = [m for m in meas if not (m['type'] == 'Vang' and m.get('bus') == slack_num)]
        if not meas:
            continue

        sol = estimate_with_bad_data(net, meas, threshold=threshold,
                                     max_removals=max_removals, x0=x_prev)
        if sol['converged'] and sol['observable']:
            x_prev = sol['x']

        # accuracy vs truth (after removal) and before removal (raw fit)
        rmse_V = rmse_ang = rmse_V_before = None
        if truth is not None and round(t, 6) in truth and sol['V'] is not None:
            Vt, THt = truth[round(t, 6)]
            rmse_V = float(np.sqrt(np.mean((sol['V'] - Vt) ** 2)))
            rmse_ang = float(np.rad2deg(np.sqrt(np.mean((sol['theta'] - THt) ** 2))))
            if sol['V_first'] is not None:
                rmse_V_before = float(np.sqrt(np.mean((sol['V_first'] - Vt) ** 2)))

        # detection scoring vs ground truth (only over measurements actually fed in)
        tp = fp = fn = None
        actual = set()
        if badlog is not None:
            present = {meas_key(m) for m in meas}
            for stream, f in used.items():
                actual |= (badlog.get(f, set()) & present)
            removed_keys = {r['key'] for r in sol['removed']}
            tp = len(removed_keys & actual)
            fp = len(removed_keys - actual)
            fn = len(actual - removed_keys)

        results.append({
            't': t, 'lam': lam.get(t),
            'n_meas': len(meas), 'n_removed': len(sol['removed']),
            'n_iter': sol['n_iter'], 'converged': sol['converged'],
            'observable': sol['observable'], 'J': sol['J'],
            'V': sol['V'], 'theta': sol['theta'],
            'removed': sol['removed'],
            'r_n_first': sol['r_n_first'],
            'labels': [_lbl(m) for m in meas],
            'rmse_V': rmse_V, 'rmse_ang': rmse_ang, 'rmse_V_before': rmse_V_before,
            'actual_bad': sorted(str(k) for k in actual),
            'tp': tp, 'fp': fp, 'fn': fn,
            'used': used,
        })
        if progress:
            progress(i + 1, len(times), t)

    meta = {
        'case': cfg['case'], 'dir': d, 'pmu_buses': cfg['pmu_buses'],
        'n_buses': net.n, 'n_states': net.n_states, 'slack_bus': slack_num,
        'use_scada': use_scada, 'use_pmu': use_pmu, 'threshold': threshold,
        'instants': instants, 'has_truth': truth is not None,
        'has_badlog': badlog is not None,
    }
    return {'net': net, 'nums': nums, 'meta': meta, 'truth': truth,
            'results': results}


def _lbl(m):
    if 'bus' in m:
        return f"{m['type']}({m['bus']})"
    return f"{m['type']}({m['from_bus']}-{m['to_bus']})"


# ── CSV + plot writers (CLI) ───────────────────────────────────────────────────
def write_outputs(d, out):
    res = out['results']
    has_bl = out['meta']['has_badlog']

    with open(os.path.join(d, 'se_results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        hdr = ['t_sec', 'lambda', 'n_meas', 'n_removed', 'n_iter', 'converged',
               'observable', 'J', 'rmse_V', 'rmse_ang_deg']
        if has_bl:
            hdr += ['n_actual_bad', 'TP', 'FP', 'FN']
        w.writerow(hdr)
        for r in res:
            row = [f"{r['t']:.3f}", _f(r['lam'], '.5f'), r['n_meas'], r['n_removed'],
                   r['n_iter'], int(r['converged']), int(r['observable']),
                   _f(r['J'], '.4f'), _f(r['rmse_V'], '.3e'), _f(r['rmse_ang'], '.3e')]
            if has_bl:
                row += [len(r['actual_bad']), r['tp'], r['fp'], r['fn']]
            w.writerow(row)

    with open(os.path.join(d, 'se_removed.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_sec', 'type', 'label', 'r_n', 'value',
                    'was_actually_bad' if has_bl else 'r_n_note'])
        for r in res:
            actual = set(r['actual_bad'])
            for rm in r['removed']:
                flag = (str(rm['key']) in actual) if has_bl else ''
                w.writerow([f"{r['t']:.3f}", rm['type'], rm['label'],
                            f"{rm['r_n']:.3f}", f"{rm['value']:.5f}", flag])

    _plots(d, out)


def _f(v, fmt):
    return format(v, fmt) if v is not None else ''


def _plots(d, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable - CSVs written, plots skipped)")
        return
    res = out['results']
    if not res:
        return
    t = np.array([r['t'] for r in res])
    nums = out['nums']
    idx = {n: i for i, n in enumerate(nums)}
    plot_buses = out['meta']['pmu_buses'][:3] or nums[:3]

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # voltages
    Vest = np.array([r['V'] if r['V'] is not None else [np.nan] * len(nums) for r in res])
    for bn in plot_buses:
        ax[0].plot(t, Vest[:, idx[bn]], '--', lw=1.0, label=f'bus {bn} est')
    if out['truth'] is not None:
        for bn in plot_buses:
            vt = np.array([out['truth'][round(tt, 6)][0][idx[bn]]
                           if round(tt, 6) in out['truth'] else np.nan for tt in t])
            ax[0].plot(t, vt, '-', lw=1.3, alpha=.6, label=f'bus {bn} true')
    ax[0].set_ylabel('|V| (pu)'); ax[0].legend(ncol=3, fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title(f"{out['meta']['case']}: estimate vs truth "
                    f"(SCADA={out['meta']['use_scada']}, PMU={out['meta']['use_pmu']})")
    # bad data over time
    ax[1].step(t, [r['n_removed'] for r in res], where='mid', label='removed (detected)')
    if out['meta']['has_badlog']:
        ax[1].step(t, [len(r['actual_bad']) for r in res], where='mid',
                   label='actual bad (in SE set)', alpha=.7)
    ax[1].set_ylabel('# bad data'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    # rmse / detection
    if out['meta']['has_truth']:
        ax[2].plot(t, [r['rmse_V'] or np.nan for r in res], label='RMSE |V| (pu)')
        ax[2].plot(t, [(r['rmse_ang'] or np.nan) / 100 for r in res],
                   label='RMSE angle (deg/100)')
    ax[2].set_xlabel('time (s)'); ax[2].set_ylabel('RMSE'); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(d, 'se_tracking.png'), dpi=120)
    print(f"  plot -> {os.path.join(d, 'se_tracking.png')}")


def main():
    ap = argparse.ArgumentParser(description='Solve a measurement stream with bad-data removal')
    ap.add_argument('--dir', required=True, help='timeseries dir (clean or *_bad)')
    ap.add_argument('--no-scada', action='store_true', help='exclude SCADA measurements')
    ap.add_argument('--no-pmu', action='store_true', help='exclude PMU measurements')
    ap.add_argument('--threshold', type=float, default=3.0)
    ap.add_argument('--max-removals', type=int, default=6)
    ap.add_argument('--instants', choices=['pmu', 'scada', 'all'], default=None)
    ap.add_argument('--cdf', default=None, help='override/supply the CDF network file')
    ap.add_argument('--loads-in-pu', action='store_true',
                    help='CDF Load/Gen columns are per-unit (rescale by MVA base)')
    ap.add_argument('--pmu-buses', default=None, help='comma-separated PMU bus numbers')
    args = ap.parse_args()

    out = solve_stream(args.dir, use_scada=not args.no_scada, use_pmu=not args.no_pmu,
                       threshold=args.threshold, max_removals=args.max_removals,
                       instants=args.instants, cdf_path=args.cdf,
                       loads_in_pu=(True if args.loads_in_pu else None),
                       pmu_buses=([int(x) for x in args.pmu_buses.split(',')]
                                  if args.pmu_buses else None))
    res = out['results']
    m = out['meta']
    n_inst = len(res)
    tot_removed = sum(r['n_removed'] for r in res)
    n_unobs = sum(1 for r in res if not r['observable'])
    print(f"{m['case']}: {n_inst} instants  "
          f"(SCADA={m['use_scada']}, PMU={m['use_pmu']})  threshold={m['threshold']}")
    print(f"  bad data removed: {tot_removed} across {sum(1 for r in res if r['n_removed'])} instants")
    if n_unobs:
        print(f"  unobservable instants: {n_unobs}/{n_inst}")
    if m['has_truth']:
        rv = [r['rmse_V'] for r in res if r['rmse_V'] is not None]
        if rv:
            print(f"  mean RMSE_V = {np.mean(rv):.2e} pu")
    if m['has_badlog']:
        TP = sum(r['tp'] for r in res); FP = sum(r['fp'] for r in res); FN = sum(r['fn'] for r in res)
        rec = TP / (TP + FN) if (TP + FN) else 1.0
        prec = TP / (TP + FP) if (TP + FP) else 1.0
        print(f"  detection vs bad_log: TP={TP} FP={FP} FN={FN}  "
              f"recall={rec:.2f} precision={prec:.2f}")
    write_outputs(args.dir, out)


if __name__ == '__main__':
    main()
