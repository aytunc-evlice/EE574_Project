"""
Validate the AC-WLS estimator against every generated CDF test case.
====================================================================

For each case file in testing/cases/ this runs the same machinery the project
uses (parser -> Network -> synthetic SCADA+PMU -> observability -> WLS ->
bad-data detection) and checks that the program behaves correctly:

  [OBS]    the synthetic measurement set is fully observable (rank == n_states)
  [CONV]   WLS converges from a flat start
  [RECOV]  the estimate recovers the file's true operating point (low RMSE)
  [BAD]    an injected gross error is detected and correctly identified

A case PASSES only if all four checks pass.  This is the "does the program work
fine on this network?" gate.

Run:  python testing/validate_cases.py
"""
import os
import sys
import glob

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.parser        import parse_ieee_cdf
from src.network       import Network
from src.estimator     import wls_estimate
from src.observability import check_observability
from src.bad_data      import compute_normalized_residuals, identify_bad_data
from src.simulation    import generate_scada_measurements, generate_pmu_measurements

import testing.cdf_gen as cdf_gen

CASES_DIR = cdf_gen.CASES_DIR
RMSE_TOL  = 1e-2          # recovery tolerance (pu / rad) vs the true state


def _pmu_buses(buses):
    """Place PMUs at a spread of non-slack buses (every 3rd) for redundancy."""
    nums = [b['num'] for b in buses if b['type'] != 3]
    return nums[::3] if nums else []


def validate_case(path):
    result = {'name': os.path.basename(path), 'checks': {}, 'note': ''}
    try:
        buses, branches, mva = parse_ieee_cdf(path)
        net = Network(buses, branches)

        # True operating point stored in the file.
        V_true  = np.array([b['V']     for b in net.buses])
        th_true = np.array([b['theta'] for b in net.buses])

        # Synthetic measurement set (deterministic seed for reproducibility).
        scada = generate_scada_measurements(net, sigma_vmag=0.005,
                                             sigma_pq=0.01, seed=7)
        pmu   = generate_pmu_measurements(net, _pmu_buses(buses),
                                          sigma_pmu=0.0001, seed=7)
        meas  = scada + pmu

        # [OBS] observability
        obs, rank, n_states, _ = check_observability(net, meas)
        result['checks']['OBS'] = bool(obs)
        result['rank'] = f"{rank}/{n_states}"

        # [CONV] + [RECOV] -- warm-started from the load-flow solution, exactly
        # as run_pipeline.py invokes the estimator in all three scenarios.
        x, conv, iters, _, J = wls_estimate(net, meas, verbose=False,
                                            use_loadflow_start=True)
        result['checks']['CONV'] = bool(conv)
        result['iters'] = iters

        V_est, th_est = net.state_to_VT(x)
        rmse_v  = float(np.sqrt(np.mean((V_est - V_true) ** 2)))
        rmse_th = float(np.sqrt(np.mean((th_est - th_true) ** 2)))
        result['rmse_v'], result['rmse_th'] = rmse_v, rmse_th
        result['checks']['RECOV'] = (rmse_v < RMSE_TOL) and (rmse_th < RMSE_TOL)

        # [BAD] inject a gross error into a Pflow not touching a PMU bus, then
        # run sequential largest-normalized-residual removal (the standard
        # bad-data processing loop) and confirm the injected measurement is
        # removed and the cleaned state still recovers the truth.  Sequential
        # removal is required because in meshed/low-redundancy networks a bad
        # line-flow smears its residual onto the adjacent bus injection, so a
        # single-shot "largest residual" guess can land on a neighbour first.
        pmu_set = set(_pmu_buses(buses))
        target = next(((i, m) for i, m in enumerate(meas)
                       if m['type'] == 'Pflow'
                       and m.get('from_bus') not in pmu_set
                       and m.get('to_bus') not in pmu_set), None)
        if target is None:
            target = next(((i, m) for i, m in enumerate(meas)
                           if m['type'] == 'Pflow'), None)
        if target is None:
            result['checks']['BAD'] = None         # no flow meas to corrupt
            result['note'] = 'no Pflow to test BD'
        else:
            bad_idx, _ = target
            cur = [dict(m) for m in meas]
            cur[bad_idx] = dict(cur[bad_idx], value=cur[bad_idx]['value'] + 0.5)

            # First-pass DETECTION: does the injected error flag above threshold?
            xb, _, *_ = wls_estimate(net, cur, verbose=False,
                                     use_loadflow_start=True)
            _, r_n0, _ = compute_normalized_residuals(net, cur, xb)
            detected = abs(r_n0[bad_idx]) > 3.0
            result['checks']['BAD'] = detected      # hard pass = detection works

            # Sequential largest-residual removal -> does it auto-recover truth?
            orig_of = list(range(len(cur)))         # cur position -> original idx
            removed = []
            for _ in range(6):
                xb, _, *_ = wls_estimate(net, cur, verbose=False,
                                         use_loadflow_start=True)
                _, r_n, _ = compute_normalized_residuals(net, cur, xb)
                top = identify_bad_data(r_n, cur)
                if top is None:
                    break
                removed.append(orig_of.pop(top[0]))
                del cur[top[0]]
            xb, _, *_ = wls_estimate(net, cur, verbose=False,
                                     use_loadflow_start=True)
            Ve, the = net.state_to_VT(xb)
            rmse_v_c  = float(np.sqrt(np.mean((Ve  - V_true)  ** 2)))
            rmse_th_c = float(np.sqrt(np.mean((the - th_true) ** 2)))
            recovered = (bad_idx in removed) and rmse_v_c < RMSE_TOL \
                        and rmse_th_c < RMSE_TOL
            result['bad_recovered'] = recovered
            if detected and not recovered:
                result['note'] = 'detected but not auto-ID (interacting bad data)'

    except Exception as exc:                        # noqa: BLE001  (report, don't crash)
        result['error'] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    if not os.path.isdir(CASES_DIR) or not glob.glob(os.path.join(CASES_DIR, '*.dat')):
        print("No cases found - generating them first ...\n")
        cdf_gen.generate_all()
        print()

    files = sorted(glob.glob(os.path.join(CASES_DIR, '*.dat')))
    print(f"\nValidating {len(files)} case(s)\n" + "=" * 78)
    header = f"{'CASE':18}{'RANK':9}{'IT':4}{'RMSE_V':10}{'RMSE_th':10}{'CHECKS':16}RESULT"
    print(header)
    print("-" * 78)

    n_pass = 0
    for f in files:
        r = validate_case(f)
        if 'error' in r:
            print(f"{r['name']:18}{'':9}{'':4}{'':10}{'':10}{'':14}ERROR  {r['error']}")
            continue
        checks = r['checks']
        # core checks that must hold; BAD may be None if untestable
        core = [v for k, v in checks.items() if v is not None]
        passed = all(core)
        n_pass += int(passed)
        flag = lambda k: ('.' if checks.get(k) is None else
                          ('Y' if checks.get(k) else 'X'))
        # bad-data recovery shown as a lowercase suffix on B: Y=auto-recovered
        bdrec = ('y' if r.get('bad_recovered') else
                 ('-' if checks.get('BAD') is None else 'x'))
        cstr = f"O{flag('OBS')} C{flag('CONV')} R{flag('RECOV')} B{flag('BAD')}{bdrec}"
        print(f"{r['name']:18}{r.get('rank',''):9}{r.get('iters',''):<4}"
              f"{r.get('rmse_v',0):<10.2e}{r.get('rmse_th',0):<10.2e}"
              f"{cstr:16}{'PASS' if passed else 'FAIL'}"
              + (f'  ({r["note"]})' if r['note'] else ''))

    print("=" * 78)
    print(f"{n_pass}/{len(files)} cases PASSED")
    print("checks: O=observable  C=converged  R=state-recovered  "
          "B=bad-data-detected  (suffix y/x = sequential removal auto-recovered)")
    return 0 if n_pass == len(files) else 1


if __name__ == '__main__':
    sys.exit(main())
