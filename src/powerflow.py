"""
Newton-Raphson AC power flow (polar form).

Re-solves the network operating point (V, theta) for a given set of specified
bus injections.  Used by the time-varying measurement generator to obtain a
self-consistent "true state" at every time instant after the loads/generation
are scaled by the load profile.

This is a from-scratch implementation (no black-box solver), consistent with
the project's academic-integrity rule.  It reuses Network.calc_pinj / calc_qinj
for the power-balance equations and builds the standard textbook Jacobian
directly (NOT Network.jacobian, whose injection angle off-diagonal terms use a
different sign convention that the WLS line search tolerates but a bare Newton
iteration does not).

Bus type convention (IEEE CDF):
    3 -> slack (V, theta fixed)
    2 -> PV    (V fixed, P specified, Q free)
    others (0, 1) -> PQ (P, Q specified, V/theta free)
"""
import numpy as np


def classify_buses(net):
    """Return (slack_idx, pv_idx, pq_idx) as lists of bus *indices*."""
    slack_idx = net.slack_idx
    pv_idx, pq_idx = [], []
    for i, b in enumerate(net.buses):
        if i == slack_idx:
            continue
        if b['type'] == 2:
            pv_idx.append(i)
        else:
            pq_idx.append(i)
    return slack_idx, pv_idx, pq_idx


def _pf_jacobian(net, V, theta, P, Q):
    """
    Full n x n textbook Jacobian blocks of (P_calc, Q_calc) w.r.t. (theta, V).

    P, Q are the injections already evaluated at (V, theta).
    Returns (J11, J12, J21, J22) where
        J11 = dP/dtheta,  J12 = dP/dV,
        J21 = dQ/dtheta,  J22 = dQ/dV.
    """
    n = net.n
    G, B = net.G, net.B
    J11 = np.zeros((n, n))
    J12 = np.zeros((n, n))
    J21 = np.zeros((n, n))
    J22 = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                J11[i, i] = -Q[i] - B[i, i] * V[i] ** 2
                J12[i, i] = P[i] / V[i] + G[i, i] * V[i]
                J21[i, i] = P[i] - G[i, i] * V[i] ** 2
                J22[i, i] = Q[i] / V[i] - B[i, i] * V[i]
            else:
                d = theta[i] - theta[j]
                cos_d, sin_d = np.cos(d), np.sin(d)
                J11[i, j] = V[i] * V[j] * (G[i, j] * sin_d - B[i, j] * cos_d)
                J12[i, j] = V[i] * (G[i, j] * cos_d + B[i, j] * sin_d)
                J21[i, j] = -V[i] * V[j] * (G[i, j] * cos_d + B[i, j] * sin_d)
                J22[i, j] = V[i] * (G[i, j] * sin_d - B[i, j] * cos_d)
    return J11, J12, J21, J22


def solve_power_flow(net, P_spec, Q_spec, V0=None, theta0=None,
                     max_iter=30, tol=1e-8):
    """
    Solve the AC power flow.

    Parameters
    ----------
    net      : Network
    P_spec   : length-n array of specified real injections  (pu, gen - load)
               (slack entry ignored; PV/PQ entries enforced)
    Q_spec   : length-n array of specified reactive injections (pu, gen - load)
               (slack and PV entries ignored)
    V0, theta0 : optional warm start (length n).  Defaults: PV/slack voltages
               from bus data, 1.0 elsewhere; slack angle from bus data, 0 else.

    Returns
    -------
    V, theta   : solved state (length n)
    converged  : bool
    iters      : int
    max_mismatch : float (final max |P/Q mismatch| over enforced equations)
    """
    n = net.n
    slack_idx, pv_idx, pq_idx = classify_buses(net)

    # ── warm start ────────────────────────────────────────────────────────────
    if V0 is None:
        V = np.ones(n)
        for i, b in enumerate(net.buses):
            if b['type'] in (2, 3):
                V[i] = b['V']
    else:
        V = np.array(V0, dtype=float).copy()

    if theta0 is None:
        theta = np.zeros(n)
        theta[slack_idx] = net.buses[slack_idx]['theta']
    else:
        theta = np.array(theta0, dtype=float).copy()

    # Hold the fixed quantities exactly at their setpoints.
    V[slack_idx]     = net.buses[slack_idx]['V']
    theta[slack_idx] = net.buses[slack_idx]['theta']
    for i in pv_idx:
        V[i] = net.buses[i]['V']

    ang_unknown = pv_idx + pq_idx            # buses whose angle is solved
    vol_unknown = pq_idx                     # buses whose voltage is solved
    n_ang, n_vol = len(ang_unknown), len(vol_unknown)

    converged = False
    max_mis = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        P = np.array([net.calc_pinj(V, theta, i) for i in range(n)])
        Q = np.array([net.calc_qinj(V, theta, i) for i in range(n)])

        dP = np.array([P_spec[i] - P[i] for i in ang_unknown])
        dQ = np.array([Q_spec[i] - Q[i] for i in vol_unknown])
        mismatch = np.concatenate([dP, dQ]) if (n_ang + n_vol) else np.array([])

        max_mis = float(np.max(np.abs(mismatch))) if mismatch.size else 0.0
        if max_mis < tol:
            converged = True
            break
        if mismatch.size == 0:
            converged = True
            break

        J11, J12, J21, J22 = _pf_jacobian(net, V, theta, P, Q)

        # Reduced Jacobian: rows [dP(ang_unknown); dQ(vol_unknown)],
        #                   cols [dtheta(ang_unknown); dV(vol_unknown)]
        Jpp = J11[np.ix_(ang_unknown, ang_unknown)]
        Jpv = J12[np.ix_(ang_unknown, vol_unknown)]
        Jqp = J21[np.ix_(vol_unknown, ang_unknown)]
        Jqv = J22[np.ix_(vol_unknown, vol_unknown)]
        J = np.block([[Jpp, Jpv], [Jqp, Jqv]])

        try:
            delta = np.linalg.solve(J, mismatch)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(J, mismatch, rcond=None)

        for k, i in enumerate(ang_unknown):
            theta[i] += delta[k]
        for k, i in enumerate(vol_unknown):
            V[i] += delta[n_ang + k]

    return V, theta, converged, it, max_mis


def base_injections(net):
    """
    Specified injections (pu) from the parsed CDF bus data:
        P_spec = P_gen - P_load,  Q_spec = Q_gen - Q_load.
    Returns (P_spec, Q_spec, P_load, Q_load, P_gen, Q_gen) as length-n arrays.
    """
    n = net.n
    P_load = np.array([b['P_load'] for b in net.buses])
    Q_load = np.array([b['Q_load'] for b in net.buses])
    P_gen  = np.array([b['P_gen']  for b in net.buses])
    Q_gen  = np.array([b['Q_gen']  for b in net.buses])
    return P_gen - P_load, Q_gen - Q_load, P_load, Q_load, P_gen, Q_gen
