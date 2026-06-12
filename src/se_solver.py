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
                           x0=None, use_loadflow_start=True, alpha=0.99,
                           prescreen_gate=None):
    """
    Estimate the state and iteratively remove bad data.

    Detection uses the chi-square test on the WLS objective J (bad data is
    present only if J exceeds its chi-square threshold for the current degrees
    of freedom); identification then removes the single largest |normalized
    residual| (> `threshold`), re-estimates, and repeats until J is back under
    the threshold (clean) or a guard stops it.  On clean data J stays below the
    threshold, so nothing is removed.

    Gross-error pre-screen (optional): when a prior state `x0` is given and
    `prescreen_gate` is set, any measurement whose innovation |z - h(x0)| / sigma
    exceeds `prescreen_gate` is removed up front, before the first WLS solve.
    This sidesteps the leverage trap where a huge error on a high-weight (tiny-
    sigma) channel drags the solution to fit itself and smears the inconsistency
    onto good neighbours.  The gate is set well above the clean innovation level
    (a stale prior plus the very small PMU sigma make even good PMU innovations
    reach tens of sigma) so only solve-destroying outliers are caught; the normal
    normalized-residual loop then cleans up the moderate errors at proper
    sensitivity.

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
        critical           : channels whose bad data is UNDETECTABLE here, each
                             {label, key, redundancy, min_detectable, value, sigma}.
                             A gross error e on channel i yields an expected
                             normalized residual |e|*sqrt(red_i)/sigma_i, so the
                             smallest detectable error is
                             e_min = threshold*sigma/sqrt(red);  a channel is
                             flagged when e_min exceeds both a polarity flip
                             (2|z|) and a 10-sigma bias - i.e. no realistic
                             gross error would trip the residual test.
    """
    meas = list(measurements)                 # copy the list; dicts are not mutated
    removed = []
    x = x0
    x_first = None
    n_iter = 0
    converged = False
    J = J_thr = J_first = None
    r_n = r_n_first = None
    first_labels = None
    bad_present = False
    observable = len(meas) >= net.n_states

    # ── gross-error pre-screen against the prior state ────────────────────────
    if (x0 is not None and prescreen_gate is not None
            and np.isfinite(prescreen_gate) and len(meas) > net.n_states):
        h0 = net.h_vector(x0, meas)
        z = np.array([m['value'] for m in meas])
        sig = np.array([m['sigma'] for m in meas])
        innov = np.abs(z - h0) / np.where(sig > 0, sig, 1.0)
        order = sorted(range(len(meas)), key=lambda i: -innov[i])
        drop = []
        for i in order:
            if innov[i] <= prescreen_gate:
                break
            if (len(meas) - len(drop) - 1) < net.n_states:
                break          # stop dropping before the set becomes unobservable
            drop.append(i)
        for i in sorted(drop, key=lambda j: -innov[j]):
            bad = meas[i]
            removed.append({
                'type': bad['type'], 'key': meas_key(bad), 'label': _meas_label(bad),
                'r_n': float(innov[i]), 'value': float(bad['value']),
                'sigma': float(bad['sigma']), 'by': 'prescreen', **_loc_fields(bad),
            })
        if drop:
            dropset = set(drop)
            meas = [m for i, m in enumerate(meas) if i not in dropset]
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

        _, r_n, Om_diag = compute_normalized_residuals(net, meas, x)
        if r_n_first is None:
            r_n_first = r_n.copy()
            x_first = x.copy()
            J_first = J
            first_labels = [_meas_label(m) for m in meas]
            meas_first = list(meas)
            Om_first = Om_diag.copy()

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
            'sigma': float(bad['sigma']), 'by': 'lnr', **_loc_fields(bad),
        })
        meas = meas[:k] + meas[k + 1:]

    # unverifiable channels: redundancy so low that no realistic gross error
    # (polarity flip or 10-sigma bias) would produce a normalized residual
    # above the threshold - bad data there is invisible to the LNR test
    critical = []
    if r_n_first is not None:
        sig_f = np.array([m['sigma'] for m in meas_first])
        z_f = np.array([m['value'] for m in meas_first])
        red = np.maximum(Om_first / np.maximum(sig_f ** 2, 1e-30), 1e-12)
        e_min = threshold * sig_f / np.sqrt(red)
        for i in np.flatnonzero(e_min > np.maximum(2 * np.abs(z_f), 10 * sig_f)):
            m = meas_first[i]
            critical.append({
                'label': _meas_label(m), 'key': meas_key(m),
                'redundancy': float(red[i]), 'min_detectable': float(e_min[i]),
                'value': float(z_f[i]), 'sigma': float(sig_f[i]),
            })

    flagged_remaining = bool(J is not None and J_thr is not None and J > J_thr)
    V, theta = net.state_to_VT(x) if x is not None else (None, None)
    V_first, theta_first = net.state_to_VT(x_first) if x_first is not None else (None, None)

    return {
        'x': x, 'V': V, 'theta': theta,
        'V_first': V_first, 'theta_first': theta_first, 'J_first': J_first,
        'converged': converged, 'observable': observable,
        'n_iter': n_iter, 'J': J, 'J_threshold': J_thr, 'bad_present': bad_present,
        'removed': removed, 'kept': meas,
        'r_n_kept': r_n, 'r_n_first': r_n_first, 'first_labels': first_labels,
        'flagged_remaining': flagged_remaining, 'critical': critical,
    }
