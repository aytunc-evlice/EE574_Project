"""
EE574 - Run the AC-WLS state estimator across a generated measurement stream.
=============================================================================
Demonstrates time-varying state estimation: at every PMU instant the latest
SCADA scan is fused with the current PMU phasors and the WLS estimator (warm
started from the previous estimate, as a real EMS does) tracks the moving true
state produced by gen_measurements.py.

    python gen_measurements.py --cdf ieee_cdf_sample.dat --out results/timeseries/demo
    python run_timeseries.py --dir results/timeseries/demo

Outputs (in --dir): estimates.csv and tracking.png (true vs estimated voltage
magnitude / angle for a few buses, plus RMSE over time).
"""
import os
import sys
import json
import csv
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.parser    import parse_ieee_cdf, parse_measurement_file, fuse
from src.network   import Network
from src.estimator import wls_estimate


def load_truth(path, nums):
    ts, lam, V, TH = [], [], [], []
    for r in csv.DictReader(open(path)):
        ts.append(float(r['t_sec'])); lam.append(float(r['lambda']))
        V.append([float(r[f'V{n}']) for n in nums])
        TH.append([float(r[f'Vang_rad{n}']) for n in nums])
    return (np.array(ts), np.array(lam), np.array(V), np.array(TH))


def main():
    ap = argparse.ArgumentParser(description='Run WLS SE over a measurement stream')
    ap.add_argument('--dir', required=True, help='timeseries directory (with index.csv)')
    ap.add_argument('--plot-buses', default=None,
                    help='comma-separated bus numbers to plot (default: first 3 PMU buses)')
    args = ap.parse_args()

    man = json.load(open(os.path.join(args.dir, 'manifest.json')))
    buses, branches, mva = parse_ieee_cdf(man['cdf_file'])
    net = Network(buses, branches)
    nums = [b['num'] for b in net.buses]
    slack_num = man['slack_bus']
    vb = {b['num'] for b in buses}
    bs = ({(br['from_bus'], br['to_bus']) for br in branches}
          | {(br['to_bus'], br['from_bus']) for br in branches})

    # index of snapshots by stream
    rows = list(csv.DictReader(open(os.path.join(args.dir, 'index.csv'))))
    pmu_files   = {float(r['t_sec']): r['file'] for r in rows if r['kind'] == 'pmu'}
    scada_files = {float(r['t_sec']): r['file'] for r in rows if r['kind'] == 'scada'}
    pmu_times   = sorted(pmu_files)
    scada_times = sorted(scada_files)

    ts_tru, lam_tru, V_tru, TH_tru = load_truth(
        os.path.join(args.dir, 'truth.csv'), nums)
    truth_at = {round(t, 6): k for k, t in enumerate(ts_tru)}

    def read(rel):
        return parse_measurement_file(os.path.join(args.dir, rel), vb, bs)

    est_t, est_V, est_TH, rmse_V, rmse_TH = [], [], [], [], []
    x_prev = None
    print(f"Tracking {man['case']} over {len(pmu_times)} PMU instants "
          f"(SCADA every {man['scada_dt_s']}s) ...")
    for t in pmu_times:
        scada_t = max((s for s in scada_times if s <= t), default=scada_times[0])
        meas = fuse(read(scada_files[scada_t]), read(pmu_files[t]))
        meas = [m for m in meas
                if not (m['type'] == 'Vang' and m.get('bus') == slack_num)]

        if x_prev is None:
            x, conv, it, _, J = wls_estimate(net, meas, verbose=False,
                                             use_loadflow_start=True)
        else:
            x, conv, it, _, J = wls_estimate(net, meas, verbose=False, x0=x_prev)
        x_prev = x
        Vest, THest = net.state_to_VT(x)

        k = truth_at[round(t, 6)]
        est_t.append(t); est_V.append(Vest.copy()); est_TH.append(THest.copy())
        rmse_V.append(np.sqrt(np.mean((Vest - V_tru[k]) ** 2)))
        rmse_TH.append(np.rad2deg(np.sqrt(np.mean((THest - TH_tru[k]) ** 2))))

    est_t = np.array(est_t); est_V = np.array(est_V); est_TH = np.array(est_TH)
    rmse_V = np.array(rmse_V); rmse_TH = np.array(rmse_TH)

    # estimates.csv
    with open(os.path.join(args.dir, 'estimates.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t_sec'] + [f'V{n}' for n in nums] + [f'Vang_rad{n}' for n in nums]
                   + ['rmse_V', 'rmse_ang_deg'])
        for i, t in enumerate(est_t):
            w.writerow([f'{t:.3f}'] + [f'{v:.6f}' for v in est_V[i]]
                       + [f'{a:.6f}' for a in est_TH[i]]
                       + [f'{rmse_V[i]:.3e}', f'{rmse_TH[i]:.3e}'])

    print(f"  mean RMSE_V   = {rmse_V.mean():.2e} pu")
    print(f"  mean RMSE_ang = {rmse_TH.mean():.2e} deg")

    # tracking plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("  (matplotlib unavailable - skipped plot; estimates.csv written)")
        return

    plot_buses = ([int(x) for x in args.plot_buses.split(',')]
                  if args.plot_buses else man['pmu_buses'][:3])
    idx = {n: i for i, n in enumerate(nums)}
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for bn in plot_buses:
        i = idx[bn]
        ax[0].plot(ts_tru, V_tru[:, i], '-',  lw=1.4, label=f'bus {bn} true')
        ax[0].plot(est_t, est_V[:, i], '--', lw=1.0, label=f'bus {bn} est')
        ax[1].plot(ts_tru, np.rad2deg(TH_tru[:, i]), '-',  lw=1.4, label=f'bus {bn} true')
        ax[1].plot(est_t, np.rad2deg(est_TH[:, i]), '--', lw=1.0, label=f'bus {bn} est')
    ax[0].set_ylabel('|V| (pu)');  ax[0].legend(ncol=3, fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title(f"{man['case']}: WLS estimate tracking the time-varying state")
    ax[1].set_ylabel('angle (deg)'); ax[1].legend(ncol=3, fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(est_t, rmse_V, label='RMSE |V| (pu)')
    ax[2].plot(est_t, rmse_TH / 100.0, label='RMSE angle (deg/100)')
    ax[2].set_xlabel('time (s)'); ax[2].set_ylabel('RMSE'); ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    png = os.path.join(args.dir, 'tracking.png')
    fig.savefig(png, dpi=120)
    print(f"  plot   -> {png}")
    print(f"  csv    -> {os.path.join(args.dir, 'estimates.csv')}")


if __name__ == '__main__':
    main()
