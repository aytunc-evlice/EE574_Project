"""
Inject gross errors ("bad data") into an already-generated clean measurement
stream, producing a corrupted twin plus a log of exactly what was changed.

Gross-error model (per stream, SCADA and PMU handled independently).  A bad
measurement is made UNMISTAKABLY wrong relative to the true reading - never a
small perturbation that could pass for noise:

  * scale  :  bad = factor * clean,   |factor| in [scale_lo, scale_hi]  (>= 10)
  * negate :  bad = -clean            (polarity reversal)

Voltage magnitude (Vmag) is special-cased to a large POSITIVE multiple only - a
meter physically cannot report a negative |V|, and a negated PMU voltage is an
extreme self-masking leverage point that wrecks the estimate without ever being
flagged.  A sigma floor then guarantees the change is at least `min_sigma` *
sigma even when the true value is near zero, so "10x of ~0" can never sneak in
as a small error.

Two fault populations:
  * persistent broken meters - a fixed set of channels carries the SAME fault
    (gain or polarity) in every file (a miscalibrated / mis-wired meter);
  * transient spikes         - extra random channels corrupted in one file only.
Per file the total number of bad measurements is a random 1..max_bad.

This module only manipulates measurement values; it performs no estimation.
"""
import csv
import numpy as np


def meas_key(m):
    """Stable identity of a measurement: (type, loc1, loc2-or-None)."""
    if 'bus' in m:
        return (m['type'], int(m['bus']), None)
    return (m['type'], int(m['from_bus']), int(m['to_bus']))


def load_channels(sigmas_csv):
    """Read sigmas.csv -> {group: [(key, sigma), ...]} for persistent picking."""
    streams = {}
    with open(sigmas_csv, newline='') as f:
        for r in csv.DictReader(f):
            loc2 = r['loc2']
            key = (r['type'], int(r['loc1']),
                   None if loc2 in ('', None) else int(loc2))
            streams.setdefault(r['group'], []).append((key, float(r['sigma'])))
    return streams


# a voltage-magnitude meter can never read a negative |V|, so Vmag is corrupted
# by a large POSITIVE multiple only (never negated / sign-flipped)
POSITIVE_ONLY = {'Vmag'}


def draw_fault(rng, scale_range, negate_prob, mtype=None):
    """Random gross-fault spec: {'style': 'scale'|'negate', 'factor': float}.

    With probability `negate_prob` the meter flips polarity (bad = -clean);
    otherwise it reads a large multiple of the truth (|factor| >= scale_range[0]).
    Sign-flipping is suppressed for `mtype` in POSITIVE_ONLY (e.g. Vmag), where a
    negative reading is physically impossible and would just be an extreme,
    self-masking leverage point.
    """
    positive_only = mtype in POSITIVE_ONLY
    if not positive_only and rng.uniform() < negate_prob:
        return {'style': 'negate', 'factor': -1.0}
    sign = 1.0 if positive_only else rng.choice([-1.0, 1.0])
    return {'style': 'scale', 'factor': float(rng.uniform(*scale_range) * sign)}


def apply_fault(clean, sigma, fault, min_sigma, rng):
    """
    Corrupted value for a clean reading under `fault`.

    bad = -clean (negate) or factor*clean (scale), then floored so that
    |bad - clean| >= min_sigma * sigma.  The floor matters only when the true
    value is so small that a multiple of it would still be a tiny (undetectable)
    error - it forces a genuinely gross reading instead.
    """
    bad = -clean if fault['style'] == 'negate' else fault['factor'] * clean
    floor = min_sigma * sigma
    if abs(bad - clean) < floor:
        if bad > clean:
            s = 1.0
        elif bad < clean:
            s = -1.0
        else:
            s = float(rng.choice([-1.0, 1.0]))   # clean == 0 and exactly hit
        bad = clean + s * floor
    return float(bad)


def choose_persistent(channels, n_persistent, scale_range, negate_prob, rng):
    """
    Pick `n_persistent` channels and assign each a FIXED gross fault (a gain or
    polarity fault that recurs in every file).
    channels: list of (key, sigma).  Returns {key: {'style','factor','sigma'}}.
    """
    if not channels or n_persistent <= 0:
        return {}
    n = min(n_persistent, len(channels))
    idx = rng.choice(len(channels), size=n, replace=False)
    faults = {}
    for j in np.atleast_1d(idx):
        key, sigma = channels[int(j)]
        faults[key] = {**draw_fault(rng, scale_range, negate_prob, key[0]), 'sigma': sigma}
    return faults


def inject_file(measurements, persistent_faults, max_bad, scale_range,
                negate_prob, min_sigma, rng):
    """
    Corrupt one file's measurement list.

    measurements      : list of measurement dicts (type/value/sigma + location)
    persistent_faults : {key: {'style','factor','sigma'}} for this stream
    max_bad           : max bad measurements in this file (e.g. 2)
    scale_range       : (lo, hi) magnitude-multiple range for gross errors
    negate_prob       : probability a fault is a polarity flip instead of a scale
    min_sigma         : floor on |error|/sigma (never inject a small error)
    rng               : np.random.RandomState (seeded per file)

    Returns (corrupted_measurements, records).
    """
    out = [dict(m) for m in measurements]      # work on copies
    n = len(out)
    records = []
    if n == 0:
        return out, records

    keys = [meas_key(m) for m in out]
    pers_idx = [i for i, k in enumerate(keys) if k in persistent_faults]

    target = min(int(rng.randint(1, max_bad + 1)), n)   # 1..max_bad, capped to n
    chosen = list(pers_idx[:max_bad])                   # persistent always bad
    need = target - len(chosen)
    if need > 0:
        pool = [i for i in range(n) if i not in set(chosen)]
        if pool:
            pick = rng.choice(pool, size=min(need, len(pool)), replace=False)
            chosen += [int(i) for i in np.atleast_1d(pick)]

    for i in chosen:
        m = out[i]
        clean = measurements[i]['value']
        sigma = m['sigma']
        if keys[i] in persistent_faults:
            fault = persistent_faults[keys[i]]
            mode = 'persistent'
        else:
            fault = draw_fault(rng, scale_range, negate_prob, m['type'])
            mode = 'transient'
        bad = apply_fault(clean, sigma, fault, min_sigma, rng)
        m['value'] = bad
        err = bad - clean
        k = keys[i]
        records.append({
            'pos': i, 'type': m['type'],
            'loc1': k[1], 'loc2': ('' if k[2] is None else k[2]),
            'sigma': sigma, 'clean_value': clean, 'bad_value': bad,
            'error': err, 'error_sigmas': (err / sigma if sigma else 0.0),
            'mode': mode, 'style': fault['style'], 'factor': fault['factor'],
        })
    return out, records


def file_rng(seed, time_index, stream_tag):
    """Deterministic per-file RNG (reproducible, independent across files)."""
    s = (int(seed) * 1_000_003 + int(time_index) * 131 + int(stream_tag)) % (2 ** 32)
    return np.random.RandomState(s)
