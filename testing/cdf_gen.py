"""
CDF / measurement test-case generator for the EE574 AC-WLS state estimator.
============================================================================

Produces IEEE-CDF network files that are *column-exact* for src/parser.py and,
optionally, matching measure.dat files.  Two families of cases are generated:

  * STANDARD cases  - small, fixed networks (3-, 5-bus) plus a re-emit of the
                      repo's real 14-bus file.  Used to validate correctness.
  * RANDOM cases    - connected random topologies of varying size, with some
                      off-nominal transformers, for robustness / stress testing.

A test "case" is fully defined by a connected network plus a chosen operating
point (V, theta).  Because the synthetic measurement generator in
src/simulation.py derives all measurements from (V, theta) via Ybus, ANY
reasonable operating point yields a self-consistent, fully-recoverable case.
Bus load/generation columns are back-solved from the chosen state so the files
are also valid as classic "solved load flow" CDF files.

Run:  python testing/cdf_gen.py        # writes files into testing/cases/
"""
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.parser  import parse_ieee_cdf
from src.network import Network

CASES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases')


# ────────────────────────────────────────────────────────────────────────────
# Column-exact field placement.  Every (start, width) below mirrors a slice in
# src/parser.py.  _put writes into a mutable character buffer; _num formats a
# number to an exact width and raises if it would overflow (which would shift
# every following column).
# ────────────────────────────────────────────────────────────────────────────
def _buf(width):
    return [' '] * width


def _put(buf, col, text):
    for k, ch in enumerate(text):
        buf[col + k] = ch


def _num(val, width, dec):
    s = f"{val:.{dec}f}"
    if len(s) > width:
        raise ValueError(f"value {val} needs >{width} cols (got '{s}')")
    return s.rjust(width)


def _int(val, width):
    s = f"{val:d}"
    if len(s) > width:
        raise ValueError(f"int {val} needs >{width} cols")
    return s.rjust(width)


def _bus_line(bus):
    """Format one bus card.  Mirrors parser slices:
    num[0:4] name[5:17] type[24:26] V[27:32] ang_deg[33:39]
    Pload[40:48] Qload[49:57] Pgen[59:66] Qgen[67:74] Gsh[106:114] Bsh[114:122]
    A trailing field at col 124 keeps len(line) > 122 so the shunt columns and
    g_sh are actually read by the parser (it guards on line length).
    """
    b = _buf(128)
    _put(b, 0,   _int(bus['num'], 4))
    _put(b, 5,   f"{bus['name'][:12]:<12}")
    _put(b, 24,  _int(bus['type'], 2))
    _put(b, 27,  _num(bus['V'], 5, 3))
    _put(b, 33,  _num(np.rad2deg(bus['theta']), 6, 2))
    _put(b, 40,  _num(bus['P_load'], 8, 2))
    _put(b, 49,  _num(bus['Q_load'], 8, 2))
    _put(b, 59,  _num(bus['P_gen'], 7, 2))
    _put(b, 67,  _num(bus['Q_gen'], 7, 2))
    _put(b, 106, _num(bus['G_sh'], 8, 4))
    _put(b, 114, _num(bus['B_sh'], 8, 4))
    _put(b, 124, '   0')           # remote-controlled-bus field -> keeps length
    return ''.join(b).rstrip()


def _branch_line(br):
    """Format one branch card.  Mirrors parser slices:
    from[0:4] to[5:9] R[19:29] X[29:40] B[40:50] tap[76:82]
    A trailing field past col 82 keeps len(line) > 82 so the tap column is read.
    """
    b = _buf(90)
    _put(b, 0,  _int(br['from_bus'], 4))
    _put(b, 5,  _int(br['to_bus'], 4))
    _put(b, 17, '1 0')                       # circuit / type (cosmetic)
    _put(b, 19, _num(br['R'], 10, 5))
    _put(b, 29, _num(br['X'], 11, 5))
    _put(b, 40, _num(br['B'], 10, 5))
    _put(b, 76, _num(br['tap'], 6, 3))
    _put(b, 83, '0.0')                        # keeps length > 82
    return ''.join(b).rstrip()


def write_cdf(buses, branches, path, mva_base=100.0, case_id="GEN CASE"):
    """Write a column-exact IEEE-CDF file."""
    lines = []
    title = _buf(60)
    _put(title, 0,  '01/01/26')
    _put(title, 10, 'EE574 GEN')
    _put(title, 30, _num(mva_base, 6, 1))
    _put(title, 38, '2026')
    _put(title, 42, 'W')
    _put(title, 45, case_id[:14])
    lines.append(''.join(title).rstrip())

    lines.append(f"BUS DATA FOLLOWS{'':25}{len(buses):4d} ITEMS")
    lines.extend(_bus_line(b) for b in buses)
    lines.append('-999')

    lines.append(f"BRANCH DATA FOLLOWS{'':22}{len(branches):4d} ITEMS")
    lines.extend(_branch_line(br) for br in branches)
    lines.append('-999')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ────────────────────────────────────────────────────────────────────────────
# Case construction
# ────────────────────────────────────────────────────────────────────────────
def back_solve_injections(buses, branches):
    """Given each bus's (V, theta) and type, compute the bus injections at that
    operating point and fill P_load/Q_load (PQ buses) or P_gen/Q_gen (PV/slack)
    so the CDF is a valid solved-load-flow file.  Injection = gen - load (pu)."""
    net   = Network(buses, branches)
    V     = np.array([b['V']     for b in buses])
    theta = np.array([b['theta'] for b in buses])
    for i, b in enumerate(buses):
        p = net.calc_pinj(V, theta, i) * 100.0   # back to MW (mva_base=100)
        q = net.calc_qinj(V, theta, i) * 100.0
        if b['type'] in (2, 3):                   # PV / slack -> generator bus
            b['P_gen'], b['Q_gen'] = p, q
            b['P_load'] = b['Q_load'] = 0.0
        else:                                     # PQ -> load bus
            b['P_load'], b['Q_load'] = -p, -q
            b['P_gen'] = b['Q_gen'] = 0.0
    return buses


def make_bus(num, name, btype, V, theta_deg, G_sh=0.0, B_sh=0.0):
    return {'num': num, 'name': name, 'type': btype,
            'V': float(V), 'theta': np.deg2rad(theta_deg),
            'P_load': 0.0, 'Q_load': 0.0, 'P_gen': 0.0, 'Q_gen': 0.0,
            'G_sh': float(G_sh), 'B_sh': float(B_sh)}


def make_branch(fb, tb, R, X, B=0.0, tap=1.0):
    return {'from_bus': fb, 'to_bus': tb,
            'R': float(R), 'X': float(X), 'B': float(B), 'tap': float(tap)}


# ── Standard fixed cases ────────────────────────────────────────────────────
def case_3bus():
    buses = [
        make_bus(1, 'Slack', 3, 1.060,  0.0),
        make_bus(2, 'Gen2',  2, 1.040, -2.5),
        make_bus(3, 'Load3', 0, 1.015, -5.0, B_sh=0.02),
    ]
    branches = [
        make_branch(1, 2, 0.02, 0.06, 0.03),
        make_branch(1, 3, 0.05, 0.16, 0.02),
        make_branch(2, 3, 0.04, 0.12, 0.02),
    ]
    return buses, branches, "3BUS"


def case_5bus():
    buses = [
        make_bus(1, 'Slack', 3, 1.060,  0.0),
        make_bus(2, 'Gen2',  2, 1.045, -3.0),
        make_bus(3, 'Load3', 0, 1.020, -6.0),
        make_bus(4, 'Load4', 0, 1.010, -7.5, B_sh=0.05),
        make_bus(5, 'Gen5',  2, 1.030, -5.0),
    ]
    branches = [
        make_branch(1, 2, 0.0192, 0.0575, 0.0528),
        make_branch(1, 3, 0.0452, 0.1652, 0.0408),
        make_branch(2, 3, 0.0570, 0.1737, 0.0368),
        make_branch(2, 4, 0.0,    0.2560, 0.0,    tap=0.95),   # transformer
        make_branch(3, 4, 0.0132, 0.0379, 0.0084),
        make_branch(4, 5, 0.0472, 0.1983, 0.0418),
    ]
    return buses, branches, "5BUS"


def case_from_repo_14bus():
    """Re-emit the repo's real IEEE 14-bus file: parse -> write.  Doubles as a
    round-trip check against genuine CDF data."""
    cdf = os.path.join(PROJECT_ROOT, 'ieee_cdf_sample.dat')
    buses, branches, mva = parse_ieee_cdf(cdf)
    return buses, branches, "IEEE14"


# ── Random connected networks ──────────────────────────────────────────────
def case_random(n, seed):
    """Random connected network: spanning tree (guarantees connectivity) plus a
    few extra meshing branches; ~1 in 4 branches is an off-nominal transformer.
    Operating point: slack at 1.06/0 deg, others near 1.0 pu with small angles.
    """
    rng = np.random.default_rng(seed)

    # Bus types: bus 1 slack, ~25% of the rest PV, remainder PQ.
    buses = []
    for k in range(1, n + 1):
        if k == 1:
            btype, V, ang = 3, 1.060, 0.0
        elif rng.random() < 0.25:
            btype = 2
            V, ang = float(rng.uniform(1.02, 1.06)), float(rng.uniform(-8, 0))
        else:
            btype = 0
            V, ang = float(rng.uniform(0.96, 1.04)), float(rng.uniform(-12, 0))
        bsh = 0.02 if rng.random() < 0.2 else 0.0
        buses.append(make_bus(k, f"Bus{k}", btype, V, ang, B_sh=bsh))

    def rand_branch(fb, tb):
        if rng.random() < 0.25:                       # transformer
            return make_branch(fb, tb, 0.0,
                               float(rng.uniform(0.05, 0.25)), 0.0,
                               tap=float(rng.uniform(0.90, 1.05)))
        return make_branch(fb, tb,
                           float(rng.uniform(0.005, 0.06)),
                           float(rng.uniform(0.02, 0.20)),
                           float(rng.uniform(0.0, 0.05)))

    # Spanning tree: connect each new bus to a random earlier one.
    edges = set()
    branches = []
    for tb in range(2, n + 1):
        fb = int(rng.integers(1, tb))
        edges.add((fb, tb))
        branches.append(rand_branch(fb, tb))

    # Extra meshing branches (about n/3), avoiding duplicates / self-loops.
    extra = max(1, n // 3)
    tries = 0
    while extra > 0 and tries < 50 * n:
        tries += 1
        a = int(rng.integers(1, n + 1))
        b = int(rng.integers(1, n + 1))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key in edges:
            continue
        edges.add(key)
        branches.append(rand_branch(key[0], key[1]))
        extra -= 1

    return buses, branches, f"RAND{n}"


# ────────────────────────────────────────────────────────────────────────────
def _verify_roundtrip(buses, branches, path):
    """Re-parse a written file and confirm it matches what we intended.  This is
    the guarantee that the fixed-column formatting lines up with the parser."""
    pb, pbr, _ = parse_ieee_cdf(path)
    assert len(pb) == len(buses),     f"{path}: bus count {len(pb)} != {len(buses)}"
    assert len(pbr) == len(branches), f"{path}: branch count {len(pbr)} != {len(branches)}"
    for a, c in zip(buses, pb):
        assert a['num'] == c['num'] and a['type'] == c['type'], f"{path}: bus id/type"
        assert abs(a['V'] - c['V']) < 5e-4,                     f"{path}: V mismatch bus {a['num']}"
        assert abs(a['theta'] - c['theta']) < 1e-3,             f"{path}: theta mismatch bus {a['num']}"
    for a, c in zip(branches, pbr):
        assert (a['from_bus'], a['to_bus']) == (c['from_bus'], c['to_bus']), f"{path}: branch ends"
        assert abs(a['tap'] - c['tap']) < 5e-4,                 f"{path}: tap mismatch {a['from_bus']}-{a['to_bus']}"


def generate_all(out_dir=CASES_DIR):
    os.makedirs(out_dir, exist_ok=True)
    specs = [case_3bus(), case_5bus(), case_from_repo_14bus()]
    specs += [case_random(n, seed=100 + n) for n in (4, 7, 10, 20, 40)]

    written = []
    for buses, branches, cid in specs:
        # repo 14-bus already carries a real solution; others get back-solved.
        if cid != "IEEE14":
            back_solve_injections(buses, branches)
        path = os.path.join(out_dir, f"case_{cid}.dat")
        write_cdf(buses, branches, path, case_id=cid)
        _verify_roundtrip(buses, branches, path)
        written.append((cid, path, len(buses), len(branches)))
        print(f"  wrote {os.path.relpath(path, PROJECT_ROOT):40}  "
              f"{len(buses):3d} buses, {len(branches):3d} branches  [round-trip OK]")
    return written


if __name__ == '__main__':
    print("Generating CDF test cases ...")
    generate_all()
    print("Done.")
