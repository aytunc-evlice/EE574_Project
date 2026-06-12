"""
Inject gross errors ("bad data") into an already-generated clean measurement
stream, producing a corrupted twin plus a log of exactly what was changed.

Model (per stream, SCADA and PMU handled independently):
  * persistent broken meters - a fixed set of channels gets a constant bias for
    the whole run (a miscalibrated / stuck meter);
  * transient spikes         - additional random channels are corrupted in a
    given file only.
Each gross error is  sign * m * sigma_channel,  m ~ U(sig_mult_lo, sig_mult_hi),
so it scales with that meter's own accuracy. Per file the total number of bad
measurements is a random 1..max_bad.

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


def choose_persistent(channels, n_persistent, sig_mult, rng):
    """
    Pick `n_persistent` channels and assign each a fixed bias.
    channels: list of (key, sigma).  Returns {key: bias}.
    """
    if not channels or n_persistent <= 0:
        return {}
    n = min(n_persistent, len(channels))
    idx = rng.choice(len(channels), size=n, replace=False)
    biases = {}
    for j in np.atleast_1d(idx):
        key, sigma = channels[int(j)]
        biases[key] = float(rng.choice([-1.0, 1.0]) * rng.uniform(*sig_mult) * sigma)
    return biases


def inject_file(measurements, persistent_biases, max_bad, sig_mult, rng):
    """
    Corrupt one file's measurement list.

    measurements      : list of measurement dicts (type/value/sigma + location)
    persistent_biases : {key: fixed bias} for this stream's broken meters
    max_bad           : max bad measurements in this file (e.g. 2)
    sig_mult          : (lo, hi) range for the sigma-multiple of transient errors
    rng               : np.random.RandomState (seeded per file)

    Returns (corrupted_measurements, records).
    """
    out = [dict(m) for m in measurements]      # work on copies
    n = len(out)
    records = []
    if n == 0:
        return out, records

    keys = [meas_key(m) for m in out]
    pers_idx = [i for i, k in enumerate(keys) if k in persistent_biases]

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
        if keys[i] in persistent_biases:
            err = persistent_biases[keys[i]]
            mode = 'persistent'
        else:
            err = float(rng.choice([-1.0, 1.0]) * rng.uniform(*sig_mult) * sigma)
            mode = 'transient'
        m['value'] = clean + err
        k = keys[i]
        records.append({
            'pos': i, 'type': m['type'],
            'loc1': k[1], 'loc2': ('' if k[2] is None else k[2]),
            'sigma': sigma, 'clean_value': clean, 'bad_value': m['value'],
            'error': err, 'error_sigmas': err / sigma, 'mode': mode,
        })
    return out, records


def file_rng(seed, time_index, stream_tag):
    """Deterministic per-file RNG (reproducible, independent across files)."""
    s = (int(seed) * 1_000_003 + int(time_index) * 131 + int(stream_tag)) % (2 ** 32)
    return np.random.RandomState(s)
