"""Bad data detection and identification using normalized residual test."""
import numpy as np
from numpy.linalg import solve, lstsq


THRESHOLD = 3.0   # Chi-squared normalized residual threshold


def compute_normalized_residuals(network, measurements, x):
    """
    Compute normalized residuals after state estimation.

    r_n[i] = r[i] / sqrt(Omega[i,i])
    where Omega = R - H (H^T R^-1 H)^-1 H^T  is the residual covariance.

    Returns:
        r: raw residuals (z - h(x))
        r_n: normalized residuals
        Omega_diag: diagonal of residual covariance
    """
    m = len(measurements)
    z = np.array([meas['value'] for meas in measurements])
    sigma = np.array([meas['sigma'] for meas in measurements])
    R = np.diag(sigma**2)
    W = np.diag(1.0 / sigma**2)

    h = network.h_vector(x, measurements)
    H = network.jacobian(x, measurements)
    r = z - h

    # Information matrix G = H^T W H
    HtW = H.T @ W
    G_mat = HtW @ H
    try:
        G_inv = np.linalg.inv(G_mat)
    except np.linalg.LinAlgError:
        G_inv = np.linalg.pinv(G_mat)

    # Residual covariance Omega = R - H G^-1 H^T
    Omega = R - H @ G_inv @ H.T

    # Protect against negative diagonal due to numerical issues
    Omega_diag = np.maximum(np.diag(Omega), 1e-12)
    r_n = r / np.sqrt(Omega_diag)

    return r, r_n, Omega_diag


def detect_bad_data(r_n, measurements, threshold=THRESHOLD):
    """
    Detect if any measurement is bad data (|r_n| > threshold).
    Returns list of (index, meas_label, |r_n|) sorted by |r_n| descending.
    """
    suspects = []
    for i, (rn, m) in enumerate(zip(r_n, measurements)):
        if abs(rn) > threshold:
            suspects.append((i, _meas_label(m), abs(rn)))
    suspects.sort(key=lambda t: t[2], reverse=True)
    return suspects


def identify_bad_data(r_n, measurements, threshold=THRESHOLD):
    """
    Identify single bad measurement: return index of largest |r_n|.
    Returns (index, label, |r_n|) or None if none detected.
    """
    suspects = detect_bad_data(r_n, measurements, threshold)
    return suspects[0] if suspects else None


def bad_data_report(network, measurements, x, threshold=THRESHOLD):
    """Print bad data detection report and return suspects."""
    r, r_n, Omega_diag = compute_normalized_residuals(network, measurements, x)

    print("=" * 60)
    print("BAD DATA DETECTION (Normalized Residual Test)")
    print(f"Threshold: {threshold:.1f}")
    print("=" * 60)
    print(f"{'#':>3}  {'Measurement':<22}  {'z':>10}  {'r':>10}  {'r_n':>10}  {'Flag'}")
    print("-" * 60)
    for i, m in enumerate(measurements):
        label = _meas_label(m)
        flag = "*** BAD ***" if abs(r_n[i]) > threshold else ""
        print(f"{i:>3}  {label:<22}  {m['value']:>10.5f}  {r[i]:>10.5f}  {r_n[i]:>10.3f}  {flag}")
    print("=" * 60)

    suspects = detect_bad_data(r_n, measurements, threshold)
    if suspects:
        print(f"Bad data detected in {len(suspects)} measurement(s):")
        for idx, label, val in suspects:
            print(f"  -> [{idx}] {label}  |r_n| = {val:.3f}")
    else:
        print("No bad data detected.")
    print("=" * 60)
    return suspects


def _meas_label(m):
    if m['type'] in ('Vmag', 'Vang', 'Pinj', 'Qinj'):
        return f"{m['type']}(bus {m['bus']})"
    else:
        return f"{m['type']}({m['from_bus']}-{m['to_bus']})"
