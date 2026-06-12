"""Numerical observability analysis."""
import numpy as np


def check_observability(network, measurements, tol=1e-6):
    """
    Numerical observability check.
    Builds Jacobian H at flat start and checks rank vs number of states.

    Returns:
        observable: bool
        rank: actual rank of H
        n_states: required rank for full observability
        unobservable_states: indices of likely unobservable states
    """
    x0 = network.flat_start()
    H = network.jacobian(x0, measurements)
    n_states = network.n_states
    m = len(measurements)

    rank = np.linalg.matrix_rank(H, tol=tol)
    observable = (rank >= n_states)

    unobservable_states = []
    if not observable:
        # SVD to find near-null directions (unobservable states)
        _, s, Vt = np.linalg.svd(H)
        # Singular values below tol correspond to unobservable modes
        for i, sv in enumerate(s):
            if sv < tol:
                # The corresponding right singular vector shows which states are unobservable
                unobservable_states.append(i)

    return observable, rank, n_states, unobservable_states


def observability_report(network, measurements):
    """Print a human-readable observability report."""
    n = network.n
    obs, rank, n_states, unobs = check_observability(network, measurements)

    print("=" * 50)
    print("OBSERVABILITY ANALYSIS")
    print("=" * 50)
    print(f"Number of buses      : {n}")
    print(f"Number of states     : {n_states}  ({n-1} angles + {n} voltages)")
    print(f"Number of measurements: {len(measurements)}")
    print(f"Jacobian rank        : {rank}")
    print(f"Observable           : {'YES' if obs else 'NO'}")

    if not obs:
        deficit = n_states - rank
        print(f"Observability deficit: {deficit} state(s)")
        print("Action: Add measurements to cover unobserved buses/branches.")

    # Check which buses have no measurements at all
    measured_buses = set()
    for m in measurements:
        if 'bus' in m:
            measured_buses.add(m['bus'])
        if 'from_bus' in m:
            measured_buses.add(m['from_bus'])
            measured_buses.add(m['to_bus'])

    all_buses = {b['num'] for b in network.buses}
    unmeasured = all_buses - measured_buses
    if unmeasured:
        print(f"Buses with no direct measurements: {sorted(unmeasured)}")

    print("=" * 50)
    return obs, rank
