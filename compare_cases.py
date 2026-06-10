"""
EE574 - Multi-Case Comparison
==============================
Runs the AC-WLS pipeline on every CDF file in a folder and produces:
  - A side-by-side console table of key metrics
  - results/comparison.png  -- grouped bar charts for visual comparison

Usage
-----
    # Compare all clean test cases (default)
    python compare_cases.py

    # Compare a custom folder
    python compare_cases.py testing/cases/

    # Compare the corrupted-model cases (expect degraded metrics)
    python compare_cases.py testing/cases_bad/

    # Compare a hand-picked list of files
    python compare_cases.py testing/cases/case_3BUS.dat testing/cases/case_RAND40.dat

Metrics collected per case
--------------------------
    Buses / Branches    Network size
    States              2N-1
    Rank / States       Observability (synthetic SCADA+PMU)
    Observable          YES / NO
    SC2 Iters           WLS iterations (Scenario 2, clean)
    SC2 J               WLS objective value (should be ~ degrees of freedom)
    SC2 RMSE_V          Voltage RMSE vs load-flow reference  (pu)
    SC2 RMSE_th         Angle RMSE vs load-flow reference    (deg)
    BD Detected         Bad data correctly flagged  YES / NO
    BD Correct ID       Largest residual = injected measurement  YES / NO
    SC3 RMSE_V          Voltage RMSE after bad-data removal  (pu)
"""
import os
import sys
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.parser        import parse_ieee_cdf
from src.network       import Network
from src.estimator     import wls_estimate
from src.observability import check_observability
from src.bad_data      import compute_normalized_residuals, identify_bad_data
from src.simulation    import generate_scada_measurements, generate_pmu_measurements
from src.scenarios     import scenario_3_bad_data

RESULTS_DIR = os.path.join(ROOT, 'results')

# ── metric keys and display names ────────────────────────────────────────────
METRIC_LABELS = {
    'n_buses':      'Buses',
    'n_branches':   'Branches',
    'n_states':     'States',
    'rank':         'Rank',
    'observable':   'Observable',
    'sc2_iters':    'SC2 Iters',
    'sc2_J':        'SC2 J',
    'sc2_rmse_V':   'SC2 RMSE_V (pu)',
    'sc2_rmse_th':  'SC2 RMSE_th (deg)',
    'bd_detected':  'BD Detected',
    'bd_correct':   'BD Correct ID',
    'sc3_rmse_V':   'SC3 RMSE_V (pu)',
}


# ─────────────────────────────────────────────────────────────────────────────
def _pmu_buses(net):
    """Every 3rd non-slack bus number."""
    non_slack = [b['num'] for b in net.buses if b['type'] != 3]
    return non_slack[::3] if non_slack else []


def run_case(path):
    """
    Run the estimator on one CDF file and return a metrics dict.
    Returns {'name': ..., 'error': ...} on failure.
    """
    name = os.path.splitext(os.path.basename(path))[0]
    metrics = {'name': name}

    try:
        buses, branches, _ = parse_ieee_cdf(path)
        net = Network(buses, branches)

        metrics['n_buses']    = net.n
        metrics['n_branches'] = len(branches)
        metrics['n_states']   = net.n_states

        pmu = _pmu_buses(net)
        scada = generate_scada_measurements(net, 0.005, 0.01, seed=7)
        pmu_m = generate_pmu_measurements(net, pmu, 0.0001, seed=7)
        meas  = scada + pmu_m

        # ── observability ───────────────────────────────────────────────────
        obs, rank, n_states, _ = check_observability(net, meas)
        metrics['rank']       = rank
        metrics['observable'] = 'YES' if obs else 'NO'

        V_true  = np.array([b['V']     for b in net.buses])
        th_true = np.array([b['theta'] for b in net.buses])

        # ── Scenario 2: clean ───────────────────────────────────────────────
        x2, conv2, iters2, _, J2 = wls_estimate(
            net, meas, verbose=False, use_loadflow_start=True)
        metrics['sc2_conv']  = conv2
        metrics['sc2_iters'] = iters2
        metrics['sc2_J']     = round(J2, 2)

        V2, th2 = net.state_to_VT(x2)
        metrics['sc2_rmse_V']  = float(np.sqrt(np.mean((V2  - V_true) ** 2)))
        metrics['sc2_rmse_th'] = float(np.degrees(
                                    np.sqrt(np.mean((th2 - th_true) ** 2))))

        # ── Scenario 3: bad data injection ──────────────────────────────────
        pmu_set = set(pmu)
        target = next(
            ((i, m) for i, m in enumerate(meas)
             if m['type'] == 'Pflow'
             and m.get('from_bus') not in pmu_set
             and m.get('to_bus')   not in pmu_set),
            None
        )
        if target is None:
            # Fallback: any Pflow
            target = next(((i, m) for i, m in enumerate(meas)
                           if m['type'] == 'Pflow'), None)

        if target is None:
            metrics['bd_detected'] = 'N/A'
            metrics['bd_correct']  = 'N/A'
            metrics['sc3_rmse_V']  = float('nan')
        else:
            bad_idx, _ = target
            s3_meas, _ = scenario_3_bad_data(meas, bad_idx, gross_error_pu=0.5)

            x3, _, _, _, _ = wls_estimate(
                net, s3_meas, verbose=False, use_loadflow_start=True)
            _, r_n, _ = compute_normalized_residuals(net, s3_meas, x3)

            detected = abs(r_n[bad_idx]) > 3.0
            metrics['bd_detected'] = 'YES' if detected else 'NO'

            top = identify_bad_data(r_n, s3_meas)
            correct_id = (top is not None and top[0] == bad_idx)
            metrics['bd_correct'] = 'YES' if correct_id else 'NO'

            # Re-estimate after removing flagged measurement
            if top is not None:
                clean = [m for k, m in enumerate(s3_meas) if k != top[0]]
                x3b, _, _, _, _ = wls_estimate(
                    net, clean, verbose=False, use_loadflow_start=True)
                V3b, _ = net.state_to_VT(x3b)
                metrics['sc3_rmse_V'] = float(
                    np.sqrt(np.mean((V3b - V_true) ** 2)))
            else:
                metrics['sc3_rmse_V'] = float('nan')

    except Exception as exc:
        metrics['error'] = f"{type(exc).__name__}: {exc}"

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
def print_comparison_table(results):
    """Print a formatted side-by-side comparison table."""
    keys = list(METRIC_LABELS.keys())
    col_w = max(18, max(len(r['name']) for r in results) + 2)
    lbl_w = 20

    sep = '-' * (lbl_w + col_w * len(results) + 2)

    print('\n' + '=' * len(sep))
    print('MULTI-CASE COMPARISON')
    print('=' * len(sep))

    # Header row
    header = f"{'Metric':<{lbl_w}}"
    for r in results:
        header += f"  {r['name']:>{col_w - 2}}"
    print(header)
    print(sep)

    def _fmt(r, key):
        if 'error' in r:
            return 'ERROR'
        v = r.get(key, '-')
        if isinstance(v, float):
            if np.isnan(v):
                return 'N/A'
            if key in ('sc2_rmse_V', 'sc3_rmse_V'):
                return f"{v:.5f}"
            if key == 'sc2_rmse_th':
                return f"{v:.4f}"
            if key == 'sc2_J':
                return f"{v:.1f}"
            return f"{v:.4f}"
        return str(v)

    # Group rows visually
    groups = [
        ('-- Network --',  ['n_buses', 'n_branches', 'n_states']),
        ('-- Observability --', ['rank', 'observable']),
        ('-- Scenario 2: Clean --', ['sc2_iters', 'sc2_J', 'sc2_rmse_V', 'sc2_rmse_th']),
        ('-- Bad Data --', ['bd_detected', 'bd_correct', 'sc3_rmse_V']),
    ]

    for group_title, group_keys in groups:
        print(f"\n  {group_title}")
        for key in group_keys:
            label = METRIC_LABELS.get(key, key)
            row = f"  {label:<{lbl_w - 2}}"
            for r in results:
                row += f"  {_fmt(r, key):>{col_w - 2}}"
            print(row)

    print('\n' + '=' * len(sep))

    # Summary line
    n_pass = sum(1 for r in results
                 if 'error' not in r
                 and r.get('observable') == 'YES'
                 and r.get('sc2_conv', False)
                 and r.get('sc2_rmse_V', 1.0) < 0.01
                 and r.get('bd_detected') in ('YES', 'N/A'))
    print(f"  {n_pass}/{len(results)} cases fully passed "
          f"(observable + converged + RMSE<0.01 + BD detected)")


# ─────────────────────────────────────────────────────────────────────────────
def save_comparison_plot(results, out_dir):
    """
    Three-panel figure:
      Panel 1 - RMSE voltage (Sc2 clean vs Sc3 after removal)
      Panel 2 - WLS objective J (Sc2, proportional to degrees of freedom)
      Panel 3 - Observability rank vs n_states
    """
    os.makedirs(out_dir, exist_ok=True)

    # Filter out error cases
    ok = [r for r in results if 'error' not in r]
    if not ok:
        print("  No valid cases to plot.")
        return

    names     = [r['name'].replace('case_', '') for r in ok]
    x         = np.arange(len(ok))
    bar_w     = 0.35

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('AC-WLS State Estimator — Multi-Case Comparison', fontsize=13)

    # ── Panel 1: RMSE voltage ─────────────────────────────────────────────
    ax = axes[0]
    rmse2 = [r.get('sc2_rmse_V', float('nan')) for r in ok]
    rmse3 = [r.get('sc3_rmse_V', float('nan')) for r in ok]
    b1 = ax.bar(x - bar_w / 2, rmse2, bar_w, label='Sc2: Clean',
                color='tab:blue', alpha=0.85)
    b2 = ax.bar(x + bar_w / 2, rmse3, bar_w, label='Sc3: After BD removal',
                color='tab:orange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('RMSE Voltage (pu)')
    ax.set_title('Voltage RMSE vs Load-Flow Reference')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)
    # Annotate bars with values
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        if not np.isnan(h) and h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=6, rotation=90)

    # ── Panel 2: WLS objective J ──────────────────────────────────────────
    ax = axes[1]
    J_vals = [r.get('sc2_J', 0) for r in ok]
    bars = ax.bar(x, J_vals, color='tab:green', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('WLS Objective J')
    ax.set_title('WLS Objective J (Sc2 Clean)\n~= degrees of freedom when no bad data')
    ax.grid(axis='y', alpha=0.4)
    for bar, val in zip(bars, J_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                f'{val:.0f}', ha='center', va='bottom', fontsize=7)

    # ── Panel 3: observability rank ───────────────────────────────────────
    ax = axes[2]
    ranks   = [r.get('rank',     0) for r in ok]
    n_states = [r.get('n_states', 1) for r in ok]
    pct     = [100.0 * rk / ns for rk, ns in zip(ranks, n_states)]
    colors  = ['tab:green' if p >= 100 else 'tab:red' for p in pct]
    bars = ax.bar(x, pct, color=colors, alpha=0.85)
    ax.axhline(100, color='black', linestyle='--', linewidth=1, label='Full rank (100%)')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Rank / n_states  (%)')
    ax.set_ylim(0, 115)
    ax.set_title('Observability\n(green = fully observable)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.4)
    for bar, rk, ns in zip(bars, ranks, n_states):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{rk}/{ns}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(out_dir, 'comparison.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Comparison plot saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
def save_comparison_csv(results, out_dir):
    """Save the metrics table as a CSV file."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'comparison.csv')
    keys = ['name'] + list(METRIC_LABELS.keys())
    with open(path, 'w') as f:
        f.write(','.join(keys) + '\n')
        for r in results:
            row = []
            for k in keys:
                v = r.get(k, '')
                if isinstance(v, float) and np.isnan(v):
                    v = 'N/A'
                row.append(str(v))
            f.write(','.join(row) + '\n')
    print(f"  CSV saved            : {path}")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Compare AC-WLS estimator metrics across multiple CDF cases')
    parser.add_argument('inputs', nargs='*',
                        help='CDF files or a single directory '
                             '(default: testing/cases/)')
    parser.add_argument('--out', default=RESULTS_DIR,
                        help='Output directory for comparison.png / .csv')
    args = parser.parse_args()

    # Resolve input file list
    if not args.inputs:
        folder = os.path.join(ROOT, 'testing', 'cases')
        files  = sorted(glob.glob(os.path.join(folder, '*.dat')))
    elif len(args.inputs) == 1 and os.path.isdir(args.inputs[0]):
        files  = sorted(glob.glob(os.path.join(args.inputs[0], '*.dat')))
    else:
        files  = [f for f in args.inputs if f.endswith('.dat')]

    if not files:
        print("No .dat files found. Run: python testing/cdf_gen.py  first.")
        sys.exit(1)

    print(f"Comparing {len(files)} case(s) ...\n")

    results = []
    for i, path in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"  [{i}/{len(files)}] {name} ...", end=' ', flush=True)
        r = run_case(path)
        results.append(r)
        if 'error' in r:
            print(f"ERROR: {r['error']}")
        else:
            obs  = r.get('observable', '?')
            rmse = r.get('sc2_rmse_V', float('nan'))
            bd   = r.get('bd_detected', '?')
            print(f"obs={obs}  RMSE_V={rmse:.5f}  BD={bd}")

    print_comparison_table(results)
    save_comparison_plot(results, args.out)
    save_comparison_csv(results, args.out)


if __name__ == '__main__':
    main()
