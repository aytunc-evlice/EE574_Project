"""
Generate synthetic SCADA and PMU measurements from the load-flow solution
(with additive Gaussian noise) for realistic state estimation experiments.
"""
import numpy as np


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
