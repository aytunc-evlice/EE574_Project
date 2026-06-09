"""
Bad-DATA CDF generator for the EE574 test cases.
================================================

Generates CDF files containing deliberate errors in the *network model itself*
(wrong tap ratio, wrong impedance, wrong topology, wrong bus data) - distinct
from bad *measurement* data.  Each corrupted file is produced from a clean base
case by applying exactly ONE labeled fault, and is accompanied by a JSON
manifest recording the ground truth (what was changed, from what, to what).

This module only GENERATES data - it never runs the estimator.

Fault catalog (one fault per file):
    A1_tap_subtle   transformer tap off by one ~2% step   (0.96 -> 0.94)  subtle
    A1_tap_gross    transformer tap ~20% off              (0.96 -> 0.80)  gross
    A2_Z_swap       branch R and X swapped                                gross
    A2_Z_scale      branch R,X scaled x10                                 gross
    B1_endpoint     branch rewired to a wrong bus                         gross
    B2_orient_xfmr  from/to flipped on a transformer                      subtle
    B3_missing      a branch deleted                                      gross
    B4_parallel     a branch duplicated                                   moderate
    C1_type_flip    a PV bus turned PQ                                    moderate
    C2_voltage      a bus voltage driven out of band                     gross
    C3_load_units   a load scaled x100 (MW/pu confusion)                 gross
    C3_loadgen_swap load and generation swapped at a bus                 moderate
    C4_shunt        wrong/spurious shunt susceptance                     subtle
    C5_dup_busnum   a bus number duplicated                              invalid

Run:  python testing/cdf_corrupt.py        # writes files into testing/cases_bad/
"""
import os
import sys
import copy
import json

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from testing.cdf_gen import (case_3bus, case_5bus, case_from_repo_14bus,
                             case_random, back_solve_injections, write_cdf)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cases_bad')

TAP_STEP = 0.02          # realistic ~2% tap-changer step


# ── small helpers for selecting targets deterministically ──────────────────
def _transformers(branches):
    return [i for i, br in enumerate(branches) if abs(br['tap'] - 1.0) > 1e-9]


def _lines_with_R(branches):
    return [i for i, br in enumerate(branches) if br['R'] > 1e-9]


def _pv_buses(buses):
    return [i for i, b in enumerate(buses) if b['type'] == 2]


def _nonslack_buses(buses):
    return [i for i, b in enumerate(buses) if b['type'] != 3]


def _load_buses(buses):
    return [i for i, b in enumerate(buses)
            if abs(b['P_load']) > 1e-6 or abs(b['Q_load']) > 1e-6]


def _br_tag(br):
    return f"{br['from_bus']}-{br['to_bus']}"


# ── fault functions: each mutates (buses, branches) IN PLACE on a deep copy,
#    and returns a partial record (target/field/original/corrupted) or None
#    if the fault is not applicable to this network. ─────────────────────────
def f_A1_tap_subtle(buses, branches, rng):
    idx = _transformers(branches)
    if not idx:
        return None
    k = int(rng.choice(idx))
    br = branches[k]
    old = br['tap']
    new = round(old - TAP_STEP, 4)              # one ~2% step off
    br['tap'] = new
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': br['to_bus']},
            'field': 'tap', 'original': old, 'corrupted': new}


def f_A1_tap_gross(buses, branches, rng):
    idx = _transformers(branches)
    if not idx:
        return None
    k = int(rng.choice(idx))
    br = branches[k]
    old = br['tap']
    new = round(old * 0.80, 4)                  # ~20% off
    br['tap'] = new
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': br['to_bus']},
            'field': 'tap', 'original': old, 'corrupted': new}


def f_A2_Z_swap(buses, branches, rng):
    idx = _lines_with_R(branches)               # need R>0 for swap to matter
    if not idx:
        return None
    k = int(rng.choice(idx))
    br = branches[k]
    oldR, oldX = br['R'], br['X']
    br['R'], br['X'] = oldX, oldR
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': br['to_bus']},
            'field': 'R,X', 'original': {'R': oldR, 'X': oldX},
            'corrupted': {'R': br['R'], 'X': br['X']}}


def f_A2_Z_scale(buses, branches, rng):
    k = int(rng.integers(len(branches)))
    br = branches[k]
    oldR, oldX = br['R'], br['X']
    br['R'], br['X'] = round(oldR * 10, 6), round(oldX * 10, 6)
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': br['to_bus']},
            'field': 'R,X', 'original': {'R': oldR, 'X': oldX},
            'corrupted': {'R': br['R'], 'X': br['X']}, 'factor': 10}


def f_B1_endpoint(buses, branches, rng):
    k = int(rng.integers(len(branches)))
    br = branches[k]
    nums = [b['num'] for b in buses]
    existing = {(min(b['from_bus'], b['to_bus']), max(b['from_bus'], b['to_bus']))
                for b in branches}
    cand = [n for n in nums if n != br['from_bus'] and n != br['to_bus']
            and (min(br['from_bus'], n), max(br['from_bus'], n)) not in existing]
    if not cand:
        return None
    old_to = br['to_bus']
    new_to = int(rng.choice(cand))
    br['to_bus'] = new_to
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': old_to},
            'field': 'to_bus', 'original': old_to, 'corrupted': new_to}


def f_B2_orient_xfmr(buses, branches, rng):
    idx = _transformers(branches)
    if not idx:
        return None
    k = int(rng.choice(idx))
    br = branches[k]
    fb, tb = br['from_bus'], br['to_bus']
    br['from_bus'], br['to_bus'] = tb, fb        # flip tap side
    return {'target': {'type': 'branch', 'from': fb, 'to': tb},
            'field': 'orientation', 'original': f"{fb}->{tb}",
            'corrupted': f"{tb}->{fb}"}


def f_B3_missing(buses, branches, rng):
    k = int(rng.integers(len(branches)))
    br = branches.pop(k)
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': br['to_bus']},
            'field': 'branch', 'original': 'present', 'corrupted': 'deleted'}


def f_B4_parallel(buses, branches, rng):
    k = int(rng.integers(len(branches)))
    br = branches[k]
    branches.append(copy.deepcopy(br))
    return {'target': {'type': 'branch', 'from': br['from_bus'], 'to': br['to_bus']},
            'field': 'branch', 'original': '1 circuit', 'corrupted': '2 (duplicated)'}


def f_C1_type_flip(buses, branches, rng):
    idx = _pv_buses(buses)
    if not idx:
        return None
    k = int(rng.choice(idx))
    b = buses[k]
    b['type'] = 0                                # PV -> PQ
    return {'target': {'type': 'bus', 'num': b['num']},
            'field': 'type', 'original': 2, 'corrupted': 0}


def f_C2_voltage(buses, branches, rng):
    idx = _nonslack_buses(buses)
    k = int(rng.choice(idx))
    b = buses[k]
    old = b['V']
    b['V'] = 1.45                                # clearly out of band
    return {'target': {'type': 'bus', 'num': b['num']},
            'field': 'V', 'original': old, 'corrupted': 1.45}


def f_C3_load_units(buses, branches, rng):
    # MW/pu confusion: the per-unit load value is entered in the MW field, so
    # the load reads 100x too small (e.g. 94.2 MW typed as 0.942).
    idx = _load_buses(buses) or _nonslack_buses(buses)
    k = int(rng.choice(idx))
    b = buses[k]
    oldP, oldQ = b['P_load'], b['Q_load']
    b['P_load'], b['Q_load'] = round(oldP * 0.01, 4), round(oldQ * 0.01, 4)
    return {'target': {'type': 'bus', 'num': b['num']},
            'field': 'P_load,Q_load', 'original': {'P': oldP, 'Q': oldQ},
            'corrupted': {'P': b['P_load'], 'Q': b['Q_load']}, 'factor': 0.01}


def f_C3_loadgen_swap(buses, branches, rng):
    k = int(rng.choice(_nonslack_buses(buses)))
    b = buses[k]
    pl, ql, pg, qg = b['P_load'], b['Q_load'], b['P_gen'], b['Q_gen']
    b['P_load'], b['Q_load'], b['P_gen'], b['Q_gen'] = pg, qg, pl, ql
    return {'target': {'type': 'bus', 'num': b['num']},
            'field': 'load<->gen',
            'original': {'P_load': pl, 'Q_load': ql, 'P_gen': pg, 'Q_gen': qg},
            'corrupted': {'P_load': pg, 'Q_load': qg, 'P_gen': pl, 'Q_gen': ql}}


def f_C4_shunt(buses, branches, rng):
    k = int(rng.choice(_nonslack_buses(buses)))
    b = buses[k]
    old = b['B_sh']
    b['B_sh'] = round(old + 0.30, 4)             # spurious/oversized cap bank
    return {'target': {'type': 'bus', 'num': b['num']},
            'field': 'B_sh', 'original': old, 'corrupted': b['B_sh']}


def f_C5_dup_busnum(buses, branches, rng):
    if len(buses) < 2:
        return None
    a, c = (int(x) for x in rng.choice(len(buses), size=2, replace=False))
    old = buses[c]['num']
    buses[c]['num'] = buses[a]['num']            # duplicate a's number onto c
    return {'target': {'type': 'bus', 'num': old},
            'field': 'num', 'original': old, 'corrupted': buses[a]['num']}


# ── registry: code -> (category, severity, expected_manifestation, func) ────
FAULTS = [
    ('A1_tap_subtle',  'A-branch-parameter', 'subtle',
     'parses OK; Ybus slightly altered; near-undetectable parameter error', f_A1_tap_subtle),
    ('A1_tap_gross',   'A-branch-parameter', 'gross',
     'parses OK; Ybus altered; large reactive/voltage error', f_A1_tap_gross),
    ('A2_Z_swap',      'A-branch-parameter', 'gross',
     'parses OK; series R/X transposed; wrong flows', f_A2_Z_swap),
    ('A2_Z_scale',     'A-branch-parameter', 'gross',
     'parses OK; impedance 10x; branch looks far weaker/stronger', f_A2_Z_scale),
    ('B1_endpoint',    'B-topology', 'gross',
     'parses OK; branch connects wrong bus; Ybus topology wrong', f_B1_endpoint),
    ('B2_orient_xfmr', 'B-topology', 'subtle',
     'parses OK; transformer tap on wrong side; subtle asymmetry', f_B2_orient_xfmr),
    ('B3_missing',     'B-topology', 'gross',
     'parses OK with fewer branches; flows rerouted / possible island', f_B3_missing),
    ('B4_parallel',    'B-topology', 'moderate',
     'parses OK; duplicated branch halves effective impedance', f_B4_parallel),
    ('C1_type_flip',   'C-bus-data', 'moderate',
     'parses OK; PV bus modeled as PQ; voltage support lost', f_C1_type_flip),
    ('C2_voltage',     'C-bus-data', 'gross',
     'parses OK; implausible bus voltage (out of band)', f_C2_voltage),
    ('C3_load_units',  'C-bus-data', 'gross',
     'parses OK; load 100x too small (pu typed in MW field); power balance broken', f_C3_load_units),
    ('C3_loadgen_swap','C-bus-data', 'moderate',
     'parses OK; load and generation swapped at a bus', f_C3_loadgen_swap),
    ('C4_shunt',       'C-bus-data', 'subtle',
     'parses OK; wrong shunt susceptance; reactive error', f_C4_shunt),
    ('C5_dup_busnum',  'C-bus-data', 'invalid',
     'duplicate bus number; bus index collision (parser keeps last)', f_C5_dup_busnum),
]


def base_cases():
    """Clean base networks (same set as cdf_gen.generate_all), as dict models."""
    specs = [case_3bus(), case_5bus(), case_from_repo_14bus()]
    specs += [case_random(n, seed=100 + n) for n in (4, 7, 10, 20, 40)]
    bases = []
    for buses, branches, cid in specs:
        if cid != 'IEEE14':
            back_solve_injections(buses, branches)
        bases.append((cid, buses, branches))
    return bases


def generate_bad_suite(out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    bases = base_cases()
    index = []
    n_written = n_skipped = 0

    print(f"{'FILE':40}{'TARGET':14}{'CHANGE':32}{'SEV'}")
    print('-' * 96)
    for ci, (cid, buses, branches) in enumerate(bases):
        for fi, (code, category, severity, expected, func) in enumerate(FAULTS):
            rng = np.random.default_rng(1000 * ci + fi + 7)   # reproducible
            b2 = copy.deepcopy(buses)
            br2 = copy.deepcopy(branches)
            rec = func(b2, br2, rng)
            if rec is None:
                n_skipped += 1
                continue

            stem = f"case_{cid}_{code}"
            dat_path = os.path.join(out_dir, stem + '.dat')
            json_path = os.path.join(out_dir, stem + '.json')
            write_cdf(b2, br2, dat_path, case_id=cid[:14])

            manifest = {
                'file': stem + '.dat',
                'source_case': f"case_{cid}",
                'fault_code': code,
                'category': category,
                'severity': severity,
                'expected_manifestation': expected,
                'seed': int(1000 * ci + fi + 7),
                **rec,
            }
            with open(json_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            index.append(manifest)
            n_written += 1

            tgt = rec['target']
            tgt_s = (f"bus {tgt['num']}" if tgt['type'] == 'bus'
                     else f"br {tgt['from']}-{tgt['to']}")
            chg = f"{_short(rec.get('original'))} -> {_short(rec.get('corrupted'))}"
            print(f"{stem + '.dat':40}{tgt_s:14}{chg[:31]:32}{severity}")

    with open(os.path.join(out_dir, 'INDEX.json'), 'w') as f:
        json.dump({'count': n_written, 'faults': index}, f, indent=2)

    print('-' * 96)
    print(f"Wrote {n_written} corrupted case(s) + INDEX.json to "
          f"{os.path.relpath(out_dir, PROJECT_ROOT)}  ({n_skipped} N/A skipped)")
    return index


def _short(v):
    """Compact repr for the summary table."""
    if isinstance(v, dict):
        return '{' + ','.join(f"{k}:{_short(x)}" for k, x in v.items()) + '}'
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


if __name__ == '__main__':
    print("Generating bad-data CDF cases ...\n")
    generate_bad_suite()
    print("Done.")
