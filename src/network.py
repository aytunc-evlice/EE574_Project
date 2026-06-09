"""Network model: Ybus construction and power flow equations."""
import numpy as np


def build_ybus(buses, branches):
    """Build complex admittance matrix Y_bus."""
    n = len(buses)
    bus_idx = {b['num']: i for i, b in enumerate(buses)}
    Y = np.zeros((n, n), dtype=complex)

    for br in branches:
        i = bus_idx[br['from_bus']]
        j = bus_idx[br['to_bus']]
        r, x, b_ch, tau = br['R'], br['X'], br['B'], br['tap']
        y = 1.0 / complex(r, x)   # series admittance
        b_sh = b_ch / 2.0          # half line charging per side

        # tap-side (i) self  /  impedance-side (j) self  /  mutual
        Y[i, i] += y / tau**2 + 1j * b_sh
        Y[j, j] += y          + 1j * b_sh
        Y[i, j] -= y / tau
        Y[j, i] -= y / tau

    for k, b in enumerate(buses):
        Y[k, k] += complex(b['G_sh'], b['B_sh'])

    return Y


class Network:
    """Holds network data and provides power flow calculations."""

    def __init__(self, buses, branches):
        self.buses    = buses
        self.branches = branches
        self.n        = len(buses)
        self.bus_idx  = {b['num']: i for i, b in enumerate(buses)}
        self.Y        = build_ybus(buses, branches)
        self.G        = self.Y.real
        self.B        = self.Y.imag
        self.slack_idx = next(i for i, b in enumerate(buses) if b['type'] == 3)
        self.n_states  = 2 * self.n - 1

    # ── state vector helpers ──────────────────────────────────────────────────

    def _non_slack(self):
        return [i for i in range(self.n) if i != self.slack_idx]

    def state_to_VT(self, x):
        """Unpack state vector -> (V, theta) arrays, both length n."""
        n = self.n
        theta = np.zeros(n)
        for k, i in enumerate(self._non_slack()):
            theta[i] = x[k]
        V = x[n - 1:].copy()
        return V, theta

    def VT_to_state(self, V, theta):
        n = self.n
        x = np.zeros(2 * n - 1)
        for k, i in enumerate(self._non_slack()):
            x[k] = theta[i]
        x[n - 1:] = V
        return x

    def flat_start(self):
        V     = np.ones(self.n)
        theta = np.zeros(self.n)
        for i, b in enumerate(self.buses):
            if b['type'] in (2, 3):    # PV or slack: use known voltage
                V[i]     = b['V']
                theta[i] = b['theta']
        return self.VT_to_state(V, theta)

    # ── bus injections ────────────────────────────────────────────────────────

    def calc_pinj(self, V, theta, i):
        P = 0.0
        for j in range(self.n):
            d = theta[i] - theta[j]
            P += V[j] * (self.G[i, j] * np.cos(d) + self.B[i, j] * np.sin(d))
        return V[i] * P

    def calc_qinj(self, V, theta, i):
        Q = 0.0
        for j in range(self.n):
            d = theta[i] - theta[j]
            Q += V[j] * (self.G[i, j] * np.sin(d) - self.B[i, j] * np.cos(d))
        return V[i] * Q

    # ── branch flows ──────────────────────────────────────────────────────────

    def _branch_coefs(self, from_bus_num, to_bus_num):
        """
        Return (g, b, b_sh, tau, g_ii, b_ii) for the branch.

        g_ii = g/tau^2  if from_bus_num is the tap side (branch.from_bus)
             = g        if from_bus_num is the impedance side
        b_ii = b/tau^2  (same rule) — used for the V_from^2 Q coefficient
        b_sh = half line charging (same on both sides)
        """
        for br in self.branches:
            if br['from_bus'] == from_bus_num and br['to_bus'] == to_bus_num:
                # Normal direction: from_bus is tap side
                r, x, b_ch, tau = br['R'], br['X'], br['B'], br['tap']
                y = 1.0 / complex(r, x)
                g, b = y.real, y.imag
                b_sh = b_ch / 2.0
                return g, b, b_sh, tau, g / tau**2, b / tau**2
            if br['from_bus'] == to_bus_num and br['to_bus'] == from_bus_num:
                # Reversed: from_bus_num is the impedance side
                r, x, b_ch, tau = br['R'], br['X'], br['B'], br['tap']
                y = 1.0 / complex(r, x)
                g, b = y.real, y.imag
                b_sh = b_ch / 2.0
                return g, b, b_sh, tau, g, b
        raise ValueError(f"Branch {from_bus_num}-{to_bus_num} not found")

    def calc_pflow(self, V, theta, from_bus, to_bus):
        """Active power flow FROM from_bus TO to_bus (bus numbers)."""
        i = self.bus_idx[from_bus]
        j = self.bus_idx[to_bus]
        g, b, b_sh, tau, g_ii, _ = self._branch_coefs(from_bus, to_bus)
        d = theta[i] - theta[j]
        return g_ii * V[i]**2 - (V[i] * V[j] / tau) * (g * np.cos(d) + b * np.sin(d))

    def calc_qflow(self, V, theta, from_bus, to_bus):
        """Reactive power flow FROM from_bus TO to_bus (bus numbers)."""
        i = self.bus_idx[from_bus]
        j = self.bus_idx[to_bus]
        g, b, b_sh, tau, _, b_ii = self._branch_coefs(from_bus, to_bus)
        d = theta[i] - theta[j]
        return -(b_ii + b_sh) * V[i]**2 - (V[i] * V[j] / tau) * (g * np.sin(d) - b * np.cos(d))

    # ── measurement vector and Jacobian ──────────────────────────────────────

    def h_vector(self, x, measurements):
        """Compute h(x) for all measurements."""
        V, theta = self.state_to_VT(x)
        h = np.zeros(len(measurements))
        for k, m in enumerate(measurements):
            t = m['type']
            if t == 'Vmag':
                h[k] = V[self.bus_idx[m['bus']]]
            elif t == 'Vang':
                h[k] = theta[self.bus_idx[m['bus']]]
            elif t == 'Pinj':
                h[k] = self.calc_pinj(V, theta, self.bus_idx[m['bus']])
            elif t == 'Qinj':
                h[k] = self.calc_qinj(V, theta, self.bus_idx[m['bus']])
            elif t == 'Pflow':
                h[k] = self.calc_pflow(V, theta, m['from_bus'], m['to_bus'])
            elif t == 'Qflow':
                h[k] = self.calc_qflow(V, theta, m['from_bus'], m['to_bus'])
        return h

    def jacobian(self, x, measurements):
        """Build Jacobian H = dh/dx analytically."""
        n          = self.n
        n_meas     = len(measurements)
        H          = np.zeros((n_meas, self.n_states))
        V, theta   = self.state_to_VT(x)
        non_slack  = self._non_slack()
        G, B       = self.G, self.B

        def ang_col(i):
            """State column index for angle of bus i (-1 if slack)."""
            if i == self.slack_idx:
                return -1
            return non_slack.index(i)

        def vol_col(i):
            return (n - 1) + i

        for row, m in enumerate(measurements):
            t = m['type']

            if t == 'Vmag':
                i = self.bus_idx[m['bus']]
                H[row, vol_col(i)] = 1.0

            elif t == 'Vang':
                i = self.bus_idx[m['bus']]
                c = ang_col(i)
                if c >= 0:
                    H[row, c] = 1.0
                # slack bus angle is fixed; row stays zero -> OK

            elif t in ('Pinj', 'Qinj'):
                i   = self.bus_idx[m['bus']]
                Pi  = self.calc_pinj(V, theta, i)
                Qi  = self.calc_qinj(V, theta, i)
                Vi  = V[i]
                Gii = G[i, i]
                Bii = B[i, i]

                if t == 'Pinj':
                    for j in range(n):
                        if j == i:
                            continue
                        d   = theta[i] - theta[j]
                        c   = ang_col(j)
                        if c >= 0:
                            H[row, c] += Vi * V[j] * (-G[i, j] * np.sin(d) + B[i, j] * np.cos(d))
                        H[row, vol_col(j)] += Vi * (G[i, j] * np.cos(d) + B[i, j] * np.sin(d))
                    c = ang_col(i)
                    if c >= 0:
                        H[row, c] += -(Qi + Vi**2 * Bii)
                    H[row, vol_col(i)] += Pi / Vi + Vi * Gii

                else:  # Qinj
                    for j in range(n):
                        if j == i:
                            continue
                        d   = theta[i] - theta[j]
                        c   = ang_col(j)
                        if c >= 0:
                            H[row, c] += -Vi * V[j] * (G[i, j] * np.cos(d) + B[i, j] * np.sin(d))
                        H[row, vol_col(j)] += Vi * (G[i, j] * np.sin(d) - B[i, j] * np.cos(d))
                    c = ang_col(i)
                    if c >= 0:
                        H[row, c] += Pi - Vi**2 * Gii
                    H[row, vol_col(i)] += Qi / Vi - Vi * Bii

            elif t in ('Pflow', 'Qflow'):
                fb = m['from_bus']
                tb = m['to_bus']
                i  = self.bus_idx[fb]
                j  = self.bus_idx[tb]
                g_s, b_s, b_sh, tau, g_ii, b_ii = self._branch_coefs(fb, tb)
                Vi, Vj = V[i], V[j]
                d      = theta[i] - theta[j]
                cos_d  = np.cos(d)
                sin_d  = np.sin(d)

                if t == 'Pflow':
                    # dP/d_theta_i
                    val = (Vi * Vj / tau) * (g_s * sin_d - b_s * cos_d)
                    ci  = ang_col(i)
                    if ci >= 0:
                        H[row, ci] += val
                    # dP/d_theta_j
                    cj = ang_col(j)
                    if cj >= 0:
                        H[row, cj] -= val
                    # dP/d_V_i
                    H[row, vol_col(i)] += 2 * g_ii * Vi - (Vj / tau) * (g_s * cos_d + b_s * sin_d)
                    # dP/d_V_j
                    H[row, vol_col(j)] -= (Vi / tau) * (g_s * cos_d + b_s * sin_d)

                else:  # Qflow
                    # dQ/d_theta_i
                    val = -(Vi * Vj / tau) * (g_s * cos_d + b_s * sin_d)
                    ci  = ang_col(i)
                    if ci >= 0:
                        H[row, ci] += val
                    # dQ/d_theta_j
                    cj = ang_col(j)
                    if cj >= 0:
                        H[row, cj] -= val
                    # dQ/d_V_i
                    H[row, vol_col(i)] += -2 * (b_ii + b_sh) * Vi - (Vj / tau) * (g_s * sin_d - b_s * cos_d)
                    # dQ/d_V_j
                    H[row, vol_col(j)] -= (Vi / tau) * (g_s * sin_d - b_s * cos_d)

        return H
