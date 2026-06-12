"""
State estimation with iterative bad-data removal (Largest Normalized Residual).

Unlike the static pipeline (which injects an error and removes it once), this
solver takes a GIVEN measurement set, estimates the state, and if any
normalized residual exceeds the threshold it removes the single worst
measurement, re-estimates, and repeats until the set is clean (or a guard
stops it).  It is the per-instant engine behind solve_timeseries.py / gui_solver.py.
"""
import numpy as np

from .estimator import wls_estimate
from .bad_data  import compute_normalized_residuals, _meas_label


def chi2_threshold(dof, alpha=0.99):
    """
    Upper alpha-quantile of the chi-square distribution with `dof` degrees of
    freedom (the bad-data detection threshold for the WLS objective J).
    Uses scipy if available, else the Wilson-Hilferty approximation.
    """
    if dof <= 0:
        return np.inf
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(alpha, dof))
    except Exception:
        z = 2.326348 if alpha >= 0.99 else 1.644854   # ~99% / ~95% normal quantile
        return dof * (1.0 - 2.0 / (9.0 * dof) + z * np.sqrt(2.0 / (9.0 * dof))) ** 3


def meas_key(m):
    """Stable identity (type, loc1, loc2-or-None) for matching/scoring."""
    if 'bus' in m:
        return (m['type'], int(m['bus']), None)
    return (m['type'], int(m['from_bus']), int(m['to_bus']))


def _loc_fields(m):
    if 'bus' in m:
        return {'bus': int(m['bus'])}
    return {'from_bus': int(m['from_bus']), 'to_bus': int(m['to_bus'])}


def estimate_with_bad_data(net, measurements, threshold=3.0, max_removals=6,
                           x0=None, use_loadflow_start=True, alpha=0.99):
    """
    Estimate the state and iteratively remove bad data.

    Detection uses the chi-square test on the WLS objective J (bad data is
    present only if J exceeds its chi-square threshold for the current degrees
    of freedom); identification then removes the single largest |normalized
    residual| (> `threshold`), re-estimates, and repeats until J is back under
    the threshold (clean) or a guard stops it.  On clean data J stays below the
    threshold, so nothing is removed.

    Returns a dict:
        x, V, theta        : final state
        converged          : WLS converged on the final (cleaned) set
        observable         : enough measurements remained (>= n_states)
        n_iter             : total WLS iterations across all passes
        J, J_threshold     : final objective and its chi-square threshold
        bad_present        : J exceeded the chi-square threshold on the 1st pass
        removed            : list of removed-measurement dicts (in removal order),
                             each {type, r_n, value, sigma, label, key, <loc>}
        kept               : the surviving measurements (list of dicts)
        r_n_kept           : final normalized residuals of the kept set
        r_n_first          : normalized residuals of the FULL input set (1st pass)
        flagged_remaining  : True if J still exceeds threshold at the end
    """
    meas = list(measurements)                 # copy the list; dicts are not mutated
    removed = []
    x = x0
    x_first = None
    n_iter = 0
    converged = False
    J = J_thr = J_first = None
    r_n = r_n_first = None
    bad_present = False
    observable = len(meas) >= net.n_states

    for step in range(max_removals + 1):
        if len(meas) < net.n_states:
            observable = False
            break

        if x is None:
            x, converged, it, _, J = wls_estimate(
                net, meas, verbose=False, use_loadflow_start=use_loadflow_start)
        else:
            x, converged, it, _, J = wls_estimate(net, meas, verbose=False, x0=x)
        n_iter += it

        _, r_n, _ = compute_normalized_residuals(net, meas, x)
        if r_n_first is None:
            r_n_first = r_n.copy()
            x_first = x.copy()
            J_first = J

        dof = len(meas) - net.n_states
        J_thr = chi2_threshold(dof, alpha)
        if step == 0:
            bad_present = J > J_thr

        k = int(np.argmax(np.abs(r_n)))
        worst = abs(r_n[k])

        # chi-square gate: stop when the residuals are statistically consistent
        consistent = (J <= J_thr) or (worst <= threshold)
        cap_hit = step == max_removals
        would_break = (len(meas) - 1) < net.n_states

        if consistent or cap_hit or would_break:
            break

        bad = meas[k]
        removed.append({
            'type': bad['type'], 'key': meas_key(bad), 'label': _meas_label(bad),
            'r_n': float(r_n[k]), 'value': float(bad['value']),
            'sigma': float(bad['sigma']), **_loc_fields(bad),
        })
        meas = meas[:k] + meas[k + 1:]

    flagged_remaining = bool(J is not None and J_thr is not None and J > J_thr)
    V, theta = net.state_to_VT(x) if x is not None else (None, None)
    V_first, theta_first = net.state_to_VT(x_first) if x_first is not None else (None, None)

    return {
        'x': x, 'V': V, 'theta': theta,
        'V_first': V_first, 'theta_first': theta_first, 'J_first': J_first,
        'converged': converged, 'observable': observable,
        'n_iter': n_iter, 'J': J, 'J_threshold': J_thr, 'bad_present': bad_present,
        'removed': removed, 'kept': meas,
        'r_n_kept': r_n, 'r_n_first': r_n_first,
        'flagged_remaining': flagged_remaining,
    }
