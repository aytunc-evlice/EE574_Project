"""
EE574 - Time-varying SCADA / PMU measurement-stream generator
=============================================================
Generate a realistic stream of measurements from an IEEE CDF case.  The true
network state evolves over time: loads are scaled by a load profile and the AC
power flow is re-solved at every time instant, then noisy measurements are
sampled from that true state.

Cadence (matches the project scenario): PMU phasors every 1 s, SCADA every 5 s.

Each snapshot is written as a standalone file in the read_me_meas.txt 8-section
format, so it is directly consumable by the existing pipeline
(`python run_pipeline.py --meas <snapshot.dat>`).  Sigmas are heterogeneous
across measurements but constant in time (each meter has a fixed accuracy
class drawn once from a per-type range).

Examples
--------
    python gen_measurements.py --cdf testing/cases/case_IEEE14.dat
    python gen_measurements.py --cdf testing/cases/case_5BUS.dat --horizon 120
    python gen_measurements.py --cdf ieee_cdf_sample.dat --pmu-buses 2,5,9,10,12,14

Outputs (under --out, default results/timeseries/<case>/):
    pmu/pmu_t*.dat       PMU snapshots   (Vmag, Vang [, Imag, Iang])
    scada/scada_t*.dat   SCADA snapshots (Vmag, Pinj, Qinj, Pflow, Qflow)
    index.csv            t_sec, kind, n_meas, lambda, file
    truth.csv            true V/theta per timestamp (ground truth for the SE)
    sigmas.csv           the frozen per-channel sigma assignment
    manifest.json        full run configuration
"""
import os
import sys
import json
import csv
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser     import parse_ieee_cdf
from src.network    import Network
from src.powerflow  import solve_power_flow
from src.profiles   import LoadProfile
from src.simulation import (build_measurement_catalog, sample_snapshot,
                            DEFAULT_SIGMA_RANGES)
from src.meas_writer import write_measurement_file

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='EE574 time-varying measurement generator')
    p.add_argument('--cdf', default=os.path.join(DATA_DIR, 'ieee_cdf_sample.dat'),
                   help='CDF network file')
    p.add_argument('--all', default=None, metavar='DIR',
                   help='generate a stream for EVERY .dat file in DIR '
                        '(one output subfolder per case; --out becomes the base)')
    p.add_argument('--out', default=None, help='output directory')
    p.add_argument('--horizon', type=float, default=60.0, help='duration in seconds')
    p.add_argument('--pmu-dt',   type=float, default=1.0, help='PMU sampling period (s)')
    p.add_argument('--scada-dt', type=float, default=5.0, help='SCADA sampling period (s)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--pmu-buses', default=None,
                   help='comma-separated PMU bus numbers (default: heuristic)')
    p.add_argument('--inj-buses', choices=['gen', 'all'], default='gen',
                   help='SCADA P/Q injection placement')
    p.add_argument('--loads-in-pu', action='store_true',
                   help='the CDF Load/Gen columns are already per-unit (as in '
                        'testing/cases/*.dat); multiply them back by the MVA '
                        'base so magnitudes match read_me_meas.txt')
    p.add_argument('--no-pmu-current', action='store_true',
                   help='do not generate PMU current phasors (Imag/Iang)')
    p.add_argument('--drop-pmu-prob', type=float, default=0.0,
                   help='per-channel PMU dropout probability (missing data)')
    # load-profile shape
    p.add_argument('--diurnal-amp', type=float, default=0.12)
    p.add_argument('--period', type=float, default=None,
                   help='diurnal period (s); default = horizon')
    return p.parse_args()


def default_pmu_buses(net, cdf_file):
    default_14 = os.path.join(DATA_DIR, 'ieee_cdf_sample.dat')
    if os.path.abspath(cdf_file) == os.path.abspath(default_14):
        return [2, 5, 9, 10, 12, 14]
    non_slack = [b['num'] for b in net.buses if b['type'] != 3]
    return non_slack[::3]


def time_grid(horizon, dt):
    n = int(round(horizon / dt))
    return [round(k * dt, 6) for k in range(n + 1)]


def file_tag(t):
    return f"t{t:08.2f}".replace('.', '_')      # e.g. t00012_00


def generate_case(cdf_file, out_dir, args):
    """Generate one measurement stream for `cdf_file` into `out_dir`."""
    stem = os.path.splitext(os.path.basename(cdf_file))[0]
    os.makedirs(os.path.join(out_dir, 'pmu'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'scada'), exist_ok=True)

    # 1. network
    buses, branches, mva = parse_ieee_cdf(cdf_file)
    if args.loads_in_pu:
        # CDF Load/Gen columns were already per-unit but the parser divided
        # them by the MVA base -- undo that so flows/injections are realistic.
        for b in buses:
            b['P_load'] *= mva; b['Q_load'] *= mva
            b['P_gen']  *= mva; b['Q_gen']  *= mva
    net = Network(buses, branches)
    pmu_buses = ([int(x) for x in args.pmu_buses.split(',')]
                 if args.pmu_buses else default_pmu_buses(net, cdf_file))
    slack_num = net.buses[net.slack_idx]['num']
    print(f"Network : {stem}  ({net.n} buses, {len(branches)} branches, "
          f"slack {slack_num})")
    print(f"PMU buses: {pmu_buses}")
    print(f"Cadence : PMU every {args.pmu_dt}s, SCADA every {args.scada_dt}s, "
          f"horizon {args.horizon}s")

    # 2. time grids
    pmu_times   = time_grid(args.horizon, args.pmu_dt)
    scada_times = time_grid(args.horizon, args.scada_dt)
    all_times   = sorted(set(pmu_times) | set(scada_times))

    # 3. load profile + frozen sigma catalog
    prof = LoadProfile(net, all_times, seed=args.seed,
                       diurnal_amp=args.diurnal_amp, period=args.period)
    catalog = build_measurement_catalog(
        net, pmu_buses, sigma_ranges=DEFAULT_SIGMA_RANGES, seed=args.seed,
        inj_buses=args.inj_buses, pmu_current=not args.no_pmu_current)
    print(f"Channels: {len(catalog['scada'])} SCADA, {len(catalog['pmu'])} PMU "
          f"(each with its own fixed sigma)")

    # 4. solve PF once per unique timestamp (warm-started from previous step)
    state = {}     # t -> dict(V, theta, lam, conv, mis)
    V0 = theta0 = None
    n_fail = 0
    for t in all_times:
        P_spec, Q_spec = prof.injections(t)
        V, theta, conv, iters, mis = solve_power_flow(
            net, P_spec, Q_spec, V0=V0, theta0=theta0)
        if not conv:
            n_fail += 1
        V0, theta0 = V, theta              # warm start next instant
        state[t] = dict(V=V.copy(), theta=theta.copy(),
                        lam=prof.lam(t), conv=conv, mis=mis)
    print(f"Power flow solved at {len(all_times)} instants"
          + (f"  ({n_fail} non-converged!)" if n_fail else "  (all converged)"))

    # 5. sample + write snapshots
    index_rows = []

    for t in pmu_times:
        st = state[t]
        meas = sample_snapshot(net, catalog['pmu'], st['V'], st['theta'],
                               base_seed=args.seed, time_index=int(round(t * 1000)),
                               group_tag=1, drop_prob=args.drop_pmu_prob)
        rel = os.path.join('pmu', f"pmu_{file_tag(t)}.dat")
        write_measurement_file(os.path.join(out_dir, rel), meas)
        index_rows.append((t, 'pmu', len(meas), st['lam'], rel.replace('\\', '/')))

    for t in scada_times:
        st = state[t]
        meas = sample_snapshot(net, catalog['scada'], st['V'], st['theta'],
                               base_seed=args.seed, time_index=int(round(t * 1000)),
                               group_tag=0)
        rel = os.path.join('scada', f"scada_{file_tag(t)}.dat")
        write_measurement_file(os.path.join(out_dir, rel), meas)
        index_rows.append((t, 'scada', len(meas), st['lam'], rel.replace('\\', '/')))

    index_rows.sort(key=lambda r: (r[0], r[1]))

    # 6. index.csv
    with open(os.path.join(out_dir, 'index.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_sec', 'kind', 'n_meas', 'lambda', 'file'])
        for t, kind, n, lam, rel in index_rows:
            w.writerow([f"{t:.3f}", kind, n, f"{lam:.5f}", rel])

    # 7. truth.csv (ground-truth state per timestamp)
    with open(os.path.join(out_dir, 'truth.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        nums = [b['num'] for b in net.buses]
        w.writerow(['t_sec', 'lambda', 'pf_converged', 'pf_max_mismatch']
                   + [f"V{n}" for n in nums]
                   + [f"Vang_rad{n}" for n in nums])
        for t in all_times:
            st = state[t]
            w.writerow([f"{t:.3f}", f"{st['lam']:.5f}", int(st['conv']),
                        f"{st['mis']:.3e}"]
                       + [f"{v:.6f}" for v in st['V']]
                       + [f"{a:.6f}" for a in st['theta']])

    # 8. sigmas.csv (frozen per-channel accuracy)
    with open(os.path.join(out_dir, 'sigmas.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['channel_id', 'group', 'type', 'loc1', 'loc2', 'sigma'])
        cid = 0
        for grp in ('scada', 'pmu'):
            for ch in catalog[grp]:
                if 'bus' in ch:
                    loc1, loc2 = ch['bus'], ''
                else:
                    loc1, loc2 = ch['from_bus'], ch['to_bus']
                w.writerow([cid, grp, ch['type'], loc1, loc2, f"{ch['sigma']:.6e}"])
                cid += 1

    # 9. manifest.json
    manifest = {
        'case': stem,
        'cdf_file': os.path.abspath(cdf_file),
        'mva_base': mva,
        'n_buses': net.n,
        'n_branches': len(branches),
        'slack_bus': slack_num,
        'pmu_buses': pmu_buses,
        'horizon_s': args.horizon,
        'pmu_dt_s': args.pmu_dt,
        'scada_dt_s': args.scada_dt,
        'seed': args.seed,
        'sigma_ranges': DEFAULT_SIGMA_RANGES,
        'inj_buses': args.inj_buses,
        'loads_in_pu': args.loads_in_pu,
        'pmu_current': not args.no_pmu_current,
        'drop_pmu_prob': args.drop_pmu_prob,
        'profile': {'diurnal_amp': args.diurnal_amp,
                    'period_s': args.period or args.horizon},
        'n_pmu_snapshots': len(pmu_times),
        'n_scada_snapshots': len(scada_times),
        'n_scada_channels': len(catalog['scada']),
        'n_pmu_channels': len(catalog['pmu']),
        'pf_non_converged': n_fail,
    }
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(pmu_times)} PMU + {len(scada_times)} SCADA snapshots")
    print(f"Output : {out_dir}")
    print(f"         index.csv, truth.csv, sigmas.csv, manifest.json")
    return {'stem': stem, 'n_pmu': len(pmu_times),
            'n_scada': len(scada_times), 'n_fail': n_fail, 'out': out_dir}


def main():
    args = parse_args()

    # ── single case ──────────────────────────────────────────────────────────
    if not args.all:
        stem = os.path.splitext(os.path.basename(args.cdf))[0]
        out_dir = args.out or os.path.join(DATA_DIR, 'results', 'timeseries', stem)
        generate_case(args.cdf, out_dir, args)
        return

    # ── batch: every .dat in a folder ────────────────────────────────────────
    dat_files = sorted(glob.glob(os.path.join(args.all, '*.dat')))
    if not dat_files:
        print(f"No .dat files found in {args.all}")
        sys.exit(1)
    base = args.out or os.path.join(DATA_DIR, 'results', 'timeseries')
    print(f"Generating measurement streams for {len(dat_files)} case(s) "
          f"in {args.all}\n")

    summary = []
    for dat in dat_files:
        stem = os.path.splitext(os.path.basename(dat))[0]
        out_dir = os.path.join(base, stem)
        print(f"{'='*64}\n  CASE: {stem}\n{'='*64}")
        try:
            summary.append(generate_case(dat, out_dir, args))
        except Exception as e:                       # one bad case must not stop the run
            print(f"  [ERROR] {stem}: {e}")
            summary.append({'stem': stem, 'error': str(e)})
        print()

    print(f"{'='*64}\n  SUMMARY ({len(summary)} cases)\n{'='*64}")
    for s in summary:
        if 'error' in s:
            print(f"  {s['stem']:30s}  ERROR: {s['error']}")
        else:
            warn = f"   <-- {s['n_fail']} PF NON-CONVERGED" if s['n_fail'] else ""
            print(f"  {s['stem']:30s}  {s['n_pmu']:4d} PMU + "
                  f"{s['n_scada']:3d} SCADA snapshots{warn}")
    print(f"\nAll outputs under: {base}")


if __name__ == '__main__':
    main()
