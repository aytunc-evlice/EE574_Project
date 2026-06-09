"""AC Weighted Least Squares State Estimator with backtracking line search."""
import numpy as np
from numpy.linalg import solve, lstsq


def wls_estimate(network, measurements, max_iter=50, tol=1e-4,
                 x0=None, verbose=True, use_loadflow_start=False):
    """
    Run iterative AC-WLS state estimation.

    Uses backtracking line search to improve convergence robustness.

    Parameters
    ----------
    use_loadflow_start : bool
        If True (and x0 is None), initialize from the load-flow solution
        stored in the network bus data rather than flat start.

    Returns
    -------
    x           : converged state vector
    converged   : bool
    iterations  : int
    residuals   : z - h(x) at final iterate
    J_obj       : WLS objective value r^T W r
    """
    net    = network
    m      = len(measurements)
    z      = np.array([meas['value'] for meas in measurements])
    sigma  = np.array([meas['sigma'] for meas in measurements])
    W      = np.diag(1.0 / sigma**2)

    # Starting point
    if x0 is not None:
        x = x0.copy()
    elif use_loadflow_start:
        V     = np.array([b['V']     for b in net.buses])
        theta = np.array([b['theta'] for b in net.buses])
        x     = net.VT_to_state(V, theta)
    else:
        x = net.flat_start()

    converged = False

    for it in range(max_iter):
        h    = net.h_vector(x, measurements)
        H    = net.jacobian(x, measurements)
        r    = z - h
        HtW  = H.T @ W
        G_mat = HtW @ H
        g_vec = HtW @ r

        try:
            dx = solve(G_mat, g_vec)
        except np.linalg.LinAlgError:
            dx, _, _, _ = lstsq(G_mat, g_vec, rcond=None)

        # Backtracking line search: ensure WLS objective decreases
        J_cur = float(r @ W @ r)
        alpha = 1.0
        for _ in range(20):
            x_new  = x + alpha * dx
            h_new  = net.h_vector(x_new, measurements)
            r_new  = z - h_new
            J_new  = float(r_new @ W @ r_new)
            if J_new < J_cur + 1e-10 or alpha < 1e-8:
                break
            alpha *= 0.5

        x = x + alpha * dx

        if verbose:
            print(f"  Iter {it+1:2d}: max|dx|={np.max(np.abs(alpha*dx)):.2e}  "
                  f"alpha={alpha:.3f}  J={J_new:.4e}")

        if np.max(np.abs(alpha * dx)) < tol:
            converged = True
            break

    h_final   = net.h_vector(x, measurements)
    residuals = z - h_final
    J_obj     = float(residuals @ W @ residuals)

    return x, converged, it + 1, residuals, J_obj
