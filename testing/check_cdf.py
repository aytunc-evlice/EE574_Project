"""
CDF data-integrity checker for the EE574 state-estimator test cases.
====================================================================

This validates the *input data*, not the estimator.  Before trusting any
algorithm result on a generated network you want to know the network itself is
well-formed, physically plausible, internally consistent, and that the
synthetic measurements contain no unintended gross errors.

Three layers are checked per CDF file:

  A. STRUCTURE / PARAMETERS  - one slack bus, unique bus ids, valid & connected
     branches, X != 0, positive taps, plausible voltages, no NaN/Inf,
     non-singular Ybus, full observability with the standard measurement set.

  B. OPERATING-POINT CONSISTENCY - is the stored (V, theta) a real power-flow
     solution?  Power-balance mismatch  P_inj_calc - (P_gen - P_load)  ~ 0 at
     every bus, and total active losses  sum(P_inj) >= 0  (mandatory for R>=0).

  C. MEASUREMENT CLEANLINESS - at the TRUE state the residual of every generated
     measurement is just its noise draw, so |z - h(x*)| / sigma ~ N(0,1).  Any
     value far in the tail (>4-5 sigma) would signal accidental bad data.

Severities:  ERROR = invalid/unusable data, WARN = unusual but tolerable, OK.

Run:  python testing/check_cdf.py            # check every file in testing/cases/
      python testing/check_cdf.py <file.dat> # check one file
"""
import os
import sys
import glob

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.parser     import parse_ieee_cdf
from src.network    import Network, build_ybus
from src.simulation import generate_scada_measurements, generate_pmu_measurements

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases')


# ── plausibility thresholds ─────────────────────────────────────────────────
V_LO, V_HI       = 0.90, 1.10      # acceptable voltage band (WARN outside)
TAP_LO, TAP_HI   = 0.80, 1.20      # acceptable tap band (WARN outside)
ANG_MAX_DEG      = 60.0            # |angle| sanity ceiling (WARN above)
# The CDF format stores V to 3 decimals and angle to 0.01 deg, so an operating
# point read back from a file carries ~1e-2 pu inherent power-balance error.
# Below this it is "consistent within storage precision"; well above it the
# load/gen columns are simply not a power-flow solution (e.g. a truncated
# reference case).  This never gates the estimator, which uses V,theta only.
MISMATCH_ROUND   = 5e-2            # <= this: explained by CDF rounding -> OK
MISMATCH_LOOSE   = 5e-1            # above this: load/gen clearly not a PF soln
COND_WARN        = 1e10            # gain-matrix condition number
NOISE_SIGMA_WARN = 5.0            # |z - h(x*)| / sigma tail (clean ~ N(0,1))


class Report:
    def __init__(self, name):
        self.name = name
        self.items = []           # (severity, message)
        self.metrics = {}

    def add(self, sev, msg):
        self.items.append((sev, msg))

    def ok(self, msg):   self.add('OK',    msg)
    def warn(self, msg): self.add('WARN',  msg)
    def err(self, msg):  self.add('ERROR', msg)

    @property
    def status(self):
        if any(s == 'ERROR' for s, _ in self.items): return 'ERROR'
        if any(s == 'WARN'  for s, _ in self.items): return 'WARN'
        return 'OK'


def _connected(n, branches, bus_idx):
    """BFS over the branch graph; return True if all buses are reachable."""
    adj = {i: [] for i in range(n)}
    for br in branches:
        i, j = bus_idx[br['from_bus']], bus_idx[br['to_bus']]
        adj[i].append(j)
        adj[j].append(i)
    seen, stack = {0}, [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n


def check_file(path):
    rep = Report(os.path.basename(path))

    # ── parse ───────────────────────────────────────────────────────────────
    try:
        buses, branches, mva = parse_ieee_cdf(path)
    except Exception as exc:
        rep.err(f"parse failed: {type(exc).__name__}: {exc}")
        return rep
    if not buses:
        rep.err("no buses parsed (column misalignment?)")
        return rep
    n = len(buses)
    bus_idx = {b['num']: i for i, b in enumerate(buses)}
    rep.metrics['size'] = f"{n}b/{len(branches)}br"

    # ── A. structure ──────────────────────────────────────────────────────
    nums = [b['num'] for b in buses]
    if len(set(nums)) != n:
        rep.err("duplicate bus numbers")
    n_slack = sum(1 for b in buses if b['type'] == 3)
    if n_slack != 1:
        rep.err(f"slack bus count = {n_slack} (need exactly 1)")

    seen_edges = set()
    for br in branches:
        fb, tb = br['from_bus'], br['to_bus']
        if fb not in bus_idx or tb not in bus_idx:
            rep.err(f"branch {fb}-{tb} references missing bus")
            continue
        if fb == tb:
            rep.err(f"self-loop branch at bus {fb}")
        key = (min(fb, tb), max(fb, tb))
        if key in seen_edges:
            rep.warn(f"duplicate/parallel branch {fb}-{tb}")
        seen_edges.add(key)
        if br['X'] == 0.0:
            rep.err(f"branch {fb}-{tb} has X=0 (singular series admittance)")
        if br['R'] < 0:
            rep.warn(f"branch {fb}-{tb} has negative R")
        if br['B'] < 0:
            rep.warn(f"branch {fb}-{tb} has negative charging B")
        if br['tap'] <= 0:
            rep.err(f"branch {fb}-{tb} has non-positive tap {br['tap']}")
        elif not (TAP_LO <= br['tap'] <= TAP_HI):
            rep.warn(f"branch {fb}-{tb} tap {br['tap']:.3f} outside [{TAP_LO},{TAP_HI}]")

    # voltages / angles / NaN
    arr = np.array([[b['V'], b['theta'], b['P_load'], b['Q_load'],
                     b['P_gen'], b['Q_gen'], b['G_sh'], b['B_sh']] for b in buses])
    if not np.all(np.isfinite(arr)):
        rep.err("non-finite value in bus data (NaN/Inf)")
    for b in buses:
        if not (V_LO <= b['V'] <= V_HI):
            rep.warn(f"bus {b['num']} |V|={b['V']:.3f} outside [{V_LO},{V_HI}]")
        if abs(np.rad2deg(b['theta'])) > ANG_MAX_DEG:
            rep.warn(f"bus {b['num']} angle {np.rad2deg(b['theta']):.1f} deg is large")

    if rep.status == 'ERROR':         # don't build a model on broken structure
        return rep

    # ── Ybus / observability ────────────────────────────────────────────────
    net = Network(buses, branches)
    if not np.all(np.isfinite(net.Y)):
        rep.err("Ybus contains non-finite entries")
        return rep
    V_true  = np.array([b['V']     for b in net.buses])
    th_true = np.array([b['theta'] for b in net.buses])

    if not _connected(n, branches, bus_idx):
        rep.err("network graph is NOT connected")

    pmu = [b['num'] for b in buses if b['type'] != 3][::3]
    meas = (generate_scada_measurements(net, 0.005, 0.01, seed=7)
            + generate_pmu_measurements(net, pmu, 0.0001, seed=7))
    x_true = net.VT_to_state(V_true, th_true)
    H = net.jacobian(x_true, meas)
    rank = int(np.linalg.matrix_rank(H, tol=1e-6))
    rep.metrics['rank'] = f"{rank}/{net.n_states}"
    if rank < net.n_states:
        rep.err(f"NOT observable: rank {rank} < {net.n_states} states")

    # gain-matrix conditioning
    sig = np.array([m['sigma'] for m in meas])
    W = np.diag(1.0 / sig ** 2)
    G = H.T @ W @ H
    cond = float(np.linalg.cond(G))
    rep.metrics['cond'] = f"{cond:.1e}"
    if not np.isfinite(cond):
        rep.err("gain matrix H^T W H is singular")
    elif cond > COND_WARN:
        rep.warn(f"gain matrix ill-conditioned (cond={cond:.1e})")

    # ── B. operating-point consistency ──────────────────────────────────────
    p_mis = q_mis = 0.0
    for i, b in enumerate(net.buses):
        p_calc = net.calc_pinj(V_true, th_true, i)
        q_calc = net.calc_qinj(V_true, th_true, i)
        p_spec = b['P_gen'] - b['P_load']        # already pu
        q_spec = b['Q_gen'] - b['Q_load']
        p_mis = max(p_mis, abs(p_calc - p_spec))
        q_mis = max(q_mis, abs(q_calc - q_spec))
    mismatch = max(p_mis, q_mis)
    rep.metrics['mismatch'] = f"{mismatch:.1e}"
    # Not an estimator error: the estimator never reads the load/gen columns.
    if mismatch > MISMATCH_LOOSE:
        rep.warn(f"power-balance mismatch {mismatch:.2e} pu: load/gen columns are "
                 f"NOT a power-flow solution (decorative; e.g. truncated case). "
                 f"Harmless for estimator tests.")
    elif mismatch > MISMATCH_ROUND:
        rep.warn(f"power-balance mismatch {mismatch:.2e} pu: load/gen loosely "
                 f"consistent with voltages.")

    losses = float(sum(net.calc_pinj(V_true, th_true, i) for i in range(n)))
    rep.metrics['loss_pu'] = f"{losses:+.4f}"
    if losses < -1e-6:
        rep.err(f"total active losses {losses:.2e} pu < 0 (non-physical)")

    # ── C. measurement cleanliness (no unintended bad data) ─────────────────
    h_true = net.h_vector(x_true, meas)
    z      = np.array([m['value'] for m in meas])
    std_err = np.abs(z - h_true) / sig          # ~ N(0,1) if only noise added
    max_se = float(np.max(std_err))
    n_gross = int(np.sum(std_err > NOISE_SIGMA_WARN))
    rep.metrics['max|err|/sig'] = f"{max_se:.2f}"
    if n_gross:
        rep.warn(f"{n_gross} measurement(s) with |noise|>{NOISE_SIGMA_WARN} sigma "
                 f"(max {max_se:.1f}) - check generator noise")

    if rep.status == 'OK':
        rep.ok("data valid, consistent, and measurements clean")
    return rep


def main(argv):
    if len(argv) > 1:
        files = argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(CASES_DIR, '*.dat')))
        if not files:
            print("No case files found. Run: python testing/cdf_gen.py")
            return 1

    print(f"Checking {len(files)} CDF file(s)\n" + "=" * 92)
    hdr = (f"{'CASE':16}{'SIZE':9}{'RANK':9}{'COND':9}{'MISMATCH':10}"
           f"{'LOSS_pu':10}{'maxErr/s':9}STATUS")
    print(hdr)
    print("-" * 92)

    worst = 'OK'
    for f in files:
        r = check_file(f)
        m = r.metrics
        print(f"{r.name:16}{m.get('size',''):9}{m.get('rank',''):9}"
              f"{m.get('cond',''):9}{m.get('mismatch',''):10}"
              f"{m.get('loss_pu',''):10}{m.get('max|err|/sig',''):9}{r.status}")
        # detail lines for anything not OK
        for sev, msg in r.items:
            if sev != 'OK':
                print(f"      [{sev}] {msg}")
        if r.status == 'ERROR' or (r.status == 'WARN' and worst != 'ERROR'):
            worst = r.status

    print("=" * 92)
    print("Columns: COND=gain-matrix condition number, MISMATCH=max power-balance "
          "residual (pu),\n         LOSS_pu=total active losses (must be >=0), "
          "maxErr/s=largest measurement noise in sigmas.")
    print(f"Overall: {worst}")
    return 1 if worst == 'ERROR' else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
