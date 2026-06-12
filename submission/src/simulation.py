"""
Generate synthetic SCADA and PMU measurements from the load-flow solution
(with additive Gaussian noise) for realistic state estimation experiments.

Two layers:
  * generate_scada_measurements / generate_pmu_measurements
        snapshot helpers used by the static pipeline (single operating point,
        one scalar sigma per measurement type).
  * build_measurement_catalog / sample_snapshot
        the time-varying layer.  A catalog assigns every measurement channel
        its OWN sigma once (drawn from a per-type range and then frozen for the
        whole run), while sample_snapshot draws fresh Gaussian noise at each
        time instant from the true state (V, theta) of that instant.
"""
import numpy as np

from .meas_writer import branch_current


def generate_scada_measurements(network, sigma_vmag=0.005, sigma_pq=0.01,
                                 seed=None):
    """
    Generate a typical SCADA set from the load-flow solution stored in the
    network bus data:
      - Vmag at all buses
      - P/Q injections at generator buses (type 2 or 3)
      - P/Q flows on every branch

    All values are corrupted with Gaussian noise of the given sigmas.
    """
    if seed is not None:
        np.random.seed(seed)

    net = network
    V     = np.array([b['V']     for b in net.buses])
    theta = np.array([b['theta'] for b in net.buses])
    x_lf  = net.VT_to_state(V, theta)

    meas = []

    # Voltage magnitude at all buses
    for b in net.buses:
        i = net.bus_idx[b['num']]
        true_val = V[i]
        meas.append({'type': 'Vmag', 'bus': b['num'],
                     'value': true_val + np.random.normal(0, sigma_vmag),
                     'sigma': sigma_vmag})

    # P/Q injections at generator and slack buses
    for b in net.buses:
        if b['type'] in (2, 3):
            i   = net.bus_idx[b['num']]
            p   = net.calc_pinj(V, theta, i)
            q   = net.calc_qinj(V, theta, i)
            meas.append({'type': 'Pinj', 'bus': b['num'],
                         'value': p + np.random.normal(0, sigma_pq),
                         'sigma': sigma_pq})
            meas.append({'type': 'Qinj', 'bus': b['num'],
                         'value': q + np.random.normal(0, sigma_pq),
                         'sigma': sigma_pq})

    # P/Q flow on every branch
    for br in net.branches:
        fb = br['from_bus']
        tb = br['to_bus']
        p  = net.calc_pflow(V, theta, fb, tb)
        q  = net.calc_qflow(V, theta, fb, tb)
        meas.append({'type': 'Pflow', 'from_bus': fb, 'to_bus': tb,
                     'value': p + np.random.normal(0, sigma_pq),
                     'sigma': sigma_pq})
        meas.append({'type': 'Qflow', 'from_bus': fb, 'to_bus': tb,
                     'value': q + np.random.normal(0, sigma_pq),
                     'sigma': sigma_pq})

    return meas


def generate_pmu_measurements(network, pmu_buses, sigma_pmu=0.0001, seed=None):
    """
    Generate PMU Vmag + Vang at the listed bus numbers.
    PMU measurements are much more accurate than SCADA (smaller sigma).
    """
    if seed is not None:
        np.random.seed(seed)

    net   = network
    V     = np.array([b['V']     for b in net.buses])
    theta = np.array([b['theta'] for b in net.buses])

    meas = []
    for bus_num in pmu_buses:
        i = net.bus_idx[bus_num]
        meas.append({'type': 'Vmag', 'bus': bus_num,
                     'value': V[i]     + np.random.normal(0, sigma_pmu),
                     'sigma': sigma_pmu})
        meas.append({'type': 'Vang', 'bus': bus_num,
                     'value': theta[i] + np.random.normal(0, sigma_pmu),
                     'sigma': sigma_pmu})
    return meas


# ══════════════════════════════════════════════════════════════════════════════
# Time-varying layer: fixed per-measurement sigma catalog + per-instant sampling
# ══════════════════════════════════════════════════════════════════════════════

# Each measurement channel draws its sigma ONCE from the range for its type and
# then keeps it constant for the entire time series (a meter's accuracy class
# does not drift).  Different channels of the same type get different sigmas.
DEFAULT_SIGMA_RANGES = {
    'scada_vmag': (0.008, 0.012),   # SCADA voltage magnitude
    'scada_pq':   (0.015, 0.025),   # SCADA Pinj/Qinj/Pflow/Qflow
    'pmu_v':      (5e-5, 2e-4),     # PMU voltage phasor (Vmag, Vang)
    'pmu_i':      (1e-4, 5e-4),     # PMU current phasor (Imag, Iang)
}


def _draw_sigma(rng, ranges, key):
    lo, hi = ranges[key]
    return float(rng.uniform(lo, hi))


def build_measurement_catalog(net, pmu_buses, sigma_ranges=None, seed=0,
                              inj_buses='gen', pmu_current=True):
    """
    Build the fixed instrumentation layout: every channel with its own frozen
    sigma.  Returns a dict with two lists, 'scada' and 'pmu', of channel dicts.

    A channel is {type, sigma, group, ...location...} where location is 'bus'
    for single-bus quantities or 'from_bus'/'to_bus' for branch quantities.

    Parameters
    ----------
    pmu_buses   : list of bus numbers carrying a PMU
    sigma_ranges: per-type sigma ranges (defaults to DEFAULT_SIGMA_RANGES)
    seed        : RNG seed for the (one-time) sigma assignment
    inj_buses   : 'gen'  -> SCADA P/Q injections only at generator+slack buses
                  'all'  -> SCADA P/Q injections at every bus
    pmu_current : also place PMU current phasors on branches incident to a PMU
    """
    ranges = sigma_ranges or DEFAULT_SIGMA_RANGES
    rng = np.random.RandomState(seed)
    pmu_set = set(pmu_buses)

    scada, pmu = [], []

    # ── SCADA: Vmag at every bus ────────────────────────────────────────────
    for b in net.buses:
        scada.append({'type': 'Vmag', 'bus': b['num'], 'group': 'scada',
                      'sigma': _draw_sigma(rng, ranges, 'scada_vmag')})

    # ── SCADA: P/Q injections ───────────────────────────────────────────────
    inj_targets = (net.buses if inj_buses == 'all'
                   else [b for b in net.buses if b['type'] in (2, 3)])
    for b in inj_targets:
        scada.append({'type': 'Pinj', 'bus': b['num'], 'group': 'scada',
                      'sigma': _draw_sigma(rng, ranges, 'scada_pq')})
        scada.append({'type': 'Qinj', 'bus': b['num'], 'group': 'scada',
                      'sigma': _draw_sigma(rng, ranges, 'scada_pq')})

    # ── SCADA: P/Q flows on every branch ────────────────────────────────────
    for br in net.branches:
        for t in ('Pflow', 'Qflow'):
            scada.append({'type': t, 'from_bus': br['from_bus'],
                          'to_bus': br['to_bus'], 'group': 'scada',
                          'sigma': _draw_sigma(rng, ranges, 'scada_pq')})

    # ── PMU: voltage phasor at PMU buses ────────────────────────────────────
    for bus_num in pmu_buses:
        for t in ('Vmag', 'Vang'):
            pmu.append({'type': t, 'bus': bus_num, 'group': 'pmu',
                        'sigma': _draw_sigma(rng, ranges, 'pmu_v')})

    # ── PMU: current phasor on incident branches ────────────────────────────
    if pmu_current:
        for br in net.branches:
            if br['from_bus'] in pmu_set:
                fb, tb = br['from_bus'], br['to_bus']
            elif br['to_bus'] in pmu_set:
                fb, tb = br['to_bus'], br['from_bus']   # measure from PMU side
            else:
                continue
            for t in ('Imag', 'Iang'):
                pmu.append({'type': t, 'from_bus': fb, 'to_bus': tb,
                            'group': 'pmu',
                            'sigma': _draw_sigma(rng, ranges, 'pmu_i')})

    return {'scada': scada, 'pmu': pmu}


def _true_value(net, V, theta, ch):
    """Noise-free value of a catalog channel at state (V, theta)."""
    t = ch['type']
    if t == 'Vmag':
        return V[net.bus_idx[ch['bus']]]
    if t == 'Vang':
        return theta[net.bus_idx[ch['bus']]]
    if t == 'Pinj':
        return net.calc_pinj(V, theta, net.bus_idx[ch['bus']])
    if t == 'Qinj':
        return net.calc_qinj(V, theta, net.bus_idx[ch['bus']])
    if t == 'Pflow':
        return net.calc_pflow(V, theta, ch['from_bus'], ch['to_bus'])
    if t == 'Qflow':
        return net.calc_qflow(V, theta, ch['from_bus'], ch['to_bus'])
    if t in ('Imag', 'Iang'):
        Imag, Iang = branch_current(net, V, theta, ch['from_bus'], ch['to_bus'])
        return Imag if t == 'Imag' else Iang
    raise ValueError(f"unknown channel type {t}")


def sample_snapshot(net, channels, V, theta, base_seed=0, time_index=0,
                    group_tag=0, drop_prob=0.0):
    """
    Sample one measurement snapshot from the true state (V, theta).

    Each channel keeps its frozen sigma; the additive Gaussian noise is redrawn
    with an RNG seeded by (base_seed, time_index, group_tag), so a given time
    instant is reproducible yet differs from every other instant.

    drop_prob : probability that a channel is missing this snapshot (e.g. lost
                PMU frame).  Default 0 (no dropouts).

    Returns a list of measurement dicts {type, value, sigma, location...}.
    """
    rng = np.random.RandomState((int(base_seed) * 1_000_003
                                 + int(time_index) * 131
                                 + int(group_tag)) % (2**32))
    out = []
    for ch in channels:
        if drop_prob > 0.0 and rng.random_sample() < drop_prob:
            continue
        true_val = _true_value(net, V, theta, ch)
        sigma = ch['sigma']
        m = {'type': ch['type'], 'value': float(true_val + rng.normal(0, sigma)),
             'sigma': sigma}
        if 'bus' in ch:
            m['bus'] = ch['bus']
        else:
            m['from_bus'] = ch['from_bus']
            m['to_bus'] = ch['to_bus']
        out.append(m)
    return out
