"""
EE574 - Bad-data injector for the time-varying measurement streams
==================================================================
Takes a CLEAN stream produced by gen_measurements.py and writes a corrupted
TWIN next to it (the clean data is never modified).  Each file gets up to
`--max-bad` gross errors (default 2 -> at most 4 across a SCADA+PMU pair):

  * persistent broken meters  - a fixed channel per stream, same fault every file
  * transient spikes          - extra random channels, one file at a time

Every gross error is UNMISTAKABLY wrong relative to the truth: either a large
multiple of the real value (bad = factor * clean, |factor| in [--scale LO HI],
default >= 10x) or its negation (bad = -clean, with prob --negate-prob).  A
sigma floor (--min-sigma) guarantees the change never reads as small noise.
Exactly WHICH measurements were corrupted is reported in `bad_log.csv` (one row
per injected error) and summarised in `manifest_bad.json`.

This is data generation only - no state estimation is performed.

Examples
--------
    python inject_bad_data.py --dir results/timeseries/case_RAND20
    python inject_bad_data.py --all results/timeseries           # every clean case
"""
import os
import sys
import csv
import json
import glob
import shutil
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser       import parse_measurement_file
from src.meas_writer  import write_measurement_file
from src.bad_injection import (load_channels, choose_persistent, inject_file,
                               file_rng)
import numpy as np

STREAM_TAG = {'scada': 0, 'pmu': 1}


def parse_args():
    p = argparse.ArgumentParser(description='EE574 bad-data injector')
    p.add_argument('--dir', default=None, help='a clean timeseries dir (with index.csv)')
    p.add_argument('--all', default=None, metavar='BASE',
                   help='process every clean case under BASE (skips *_bad dirs)')
    p.add_argument('--out', default=None,
                   help='output dir (single mode); default <dir>_bad')
    p.add_argument('--max-bad', type=int, default=2,
                   help='max bad measurements per file (default 2)')
    p.add_argument('--n-persistent', type=int, default=1,
                   help='persistent broken meters per stream (default 1)')
    p.add_argument('--scale', type=float, nargs=2, default=[10.0, 30.0],
                   metavar=('LO', 'HI'),
                   help='gross error as a magnitude-multiple of the true value '
                        '(default 10 30 -> at least 10x)')
    p.add_argument('--negate-prob', type=float, default=0.5,
                   help='probability a gross error is a polarity flip (bad=-clean) '
                        'instead of a scale-up (default 0.5)')
    p.add_argument('--min-sigma', type=float, default=10.0,
                   help='floor on |error|/sigma so near-zero values still get a '
                        'detectable gross error (default 10)')
    p.add_argument('--scope', choices=['both', 'scada', 'pmu'], default='both',
                   help='which streams are eligible for corruption')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def process_case(clean_dir, out_dir, args):
    scale_range = tuple(args.scale)
    negate_prob = float(args.negate_prob)
    min_sigma = float(args.min_sigma)
    streams_in_scope = (['scada', 'pmu'] if args.scope == 'both' else [args.scope])

    rows = list(csv.DictReader(open(os.path.join(clean_dir, 'index.csv'))))
    channels = load_channels(os.path.join(clean_dir, 'sigmas.csv'))

    # ── choose persistent broken meters (seeded, deterministic) ───────────────
    setup_rng = np.random.RandomState(args.seed)
    persistent = {}
    for stream in ('scada', 'pmu'):                     # fixed order for reproducibility
        if stream in streams_in_scope:
            persistent[stream] = choose_persistent(
                channels.get(stream, []), args.n_persistent,
                scale_range, negate_prob, setup_rng)
        else:
            persistent[stream] = {}

    # ── corrupt each file ─────────────────────────────────────────────────────
    bad_log, index_out = [], []
    n_bad_total = 0
    for r in rows:
        kind = r['kind']
        rel = r['file']
        src = os.path.join(clean_dir, *rel.split('/'))
        dst = os.path.join(out_dir, *rel.split('/'))
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        if kind in streams_in_scope:
            t = float(r['t_sec'])
            meas = parse_measurement_file(src, include_current=True)
            rng = file_rng(args.seed, round(t * 1000), STREAM_TAG[kind])
            corrupted, records = inject_file(meas, persistent[kind], args.max_bad,
                                             scale_range, negate_prob, min_sigma, rng)
            write_measurement_file(dst, corrupted)
            for rec in records:
                bad_log.append({'t_sec': r['t_sec'], 'kind': kind, 'file': rel, **rec})
            n_bad = len(records)
            n_bad_total += n_bad
        else:
            shutil.copy(src, dst)                       # out-of-scope stream kept clean
            n_bad = 0

        index_out.append({**r, 'n_bad': n_bad})

    os.makedirs(out_dir, exist_ok=True)

    # ── index.csv (mirrors clean + n_bad column) ──────────────────────────────
    with open(os.path.join(out_dir, 'index.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_sec', 'kind', 'n_meas', 'lambda', 'file', 'n_bad'])
        for r in index_out:
            w.writerow([r['t_sec'], r['kind'], r['n_meas'], r['lambda'],
                        r['file'], r['n_bad']])

    # ── bad_log.csv: the report of WHICH measurements are bad ──────────────────
    cols = ['t_sec', 'kind', 'file', 'mode', 'style', 'factor', 'type',
            'loc1', 'loc2', 'sigma', 'clean_value', 'bad_value', 'error',
            'error_sigmas', 'pos']
    with open(os.path.join(out_dir, 'bad_log.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rec in bad_log:
            w.writerow({c: rec.get(c, '') for c in cols})

    # ── copy ground truth + sigma assignment (unchanged) ──────────────────────
    for fn in ('truth.csv', 'sigmas.csv'):
        sp = os.path.join(clean_dir, fn)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(out_dir, fn))

    # ── manifest_bad.json (config + persistent meters + totals) ───────────────
    def pers_report(stream):
        out = []
        for key, f in persistent.get(stream, {}).items():
            out.append({'type': key[0], 'loc1': key[1],
                        'loc2': ('' if key[2] is None else key[2]),
                        'sigma': f.get('sigma'), 'style': f['style'],
                        'factor': f['factor']})
        return out

    n_files_bad = sum(1 for r in index_out if r['n_bad'] > 0)
    by_type = {}
    for rec in bad_log:
        by_type[rec['type']] = by_type.get(rec['type'], 0) + 1
    manifest = {
        'source_dir': os.path.abspath(clean_dir),
        'max_bad_per_file': args.max_bad,
        'n_persistent_per_stream': args.n_persistent,
        'scale_range': list(scale_range),
        'negate_prob': negate_prob,
        'min_sigma': min_sigma,
        'scope': args.scope,
        'seed': args.seed,
        'persistent_meters': {'scada': pers_report('scada'),
                              'pmu': pers_report('pmu')},
        'n_bad_total': n_bad_total,
        'n_files_with_bad': n_files_bad,
        'n_files_total': len(index_out),
        'bad_by_type': by_type,
    }
    with open(os.path.join(out_dir, 'manifest_bad.json'), 'w') as f:
        json.dump(manifest, f, indent=2)

    # ── console report ────────────────────────────────────────────────────────
    case = os.path.basename(clean_dir.rstrip('/\\'))
    print(f"  {case}: {n_bad_total} gross errors across {n_files_bad}/{len(index_out)} files")
    for stream in streams_in_scope:
        for p in pers_report(stream):
            loc = f"{p['loc1']}" + (f"->{p['loc2']}" if p['loc2'] != '' else '')
            fault = ('negate (bad=-clean)' if p['style'] == 'negate'
                     else f"scale x{p['factor']:+.1f}")
            print(f"     persistent[{stream}]: {p['type']}({loc})  fault={fault}")
    return {'case': case, 'n_bad': n_bad_total, 'n_files_bad': n_files_bad,
            'n_files': len(index_out), 'out': out_dir}


def main():
    args = parse_args()

    if args.dir:
        clean = args.dir.rstrip('/\\')
        out = args.out or (clean + '_bad')
        print(f"Injecting bad data: {clean} -> {out}")
        process_case(clean, out, args)
        print(f"\nReport: {os.path.join(out, 'bad_log.csv')}")
        return

    if args.all:
        cases = sorted(d for d in glob.glob(os.path.join(args.all, '*'))
                       if os.path.isfile(os.path.join(d, 'index.csv'))
                       and not d.rstrip('/\\').endswith('_bad'))
        if not cases:
            print(f"No clean cases found under {args.all}")
            sys.exit(1)
        print(f"Injecting bad data into {len(cases)} case(s)\n")
        summary = []
        for clean in cases:
            out = clean.rstrip('/\\') + '_bad'
            summary.append(process_case(clean, out, args))
        print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
        for s in summary:
            print(f"  {s['case']:22s}  {s['n_bad']:4d} errors / "
                  f"{s['n_files_bad']:3d} of {s['n_files']} files -> {s['out']}")
        return

    print("Specify --dir <clean_case> or --all <base>")
    sys.exit(1)


if __name__ == '__main__':
    main()
