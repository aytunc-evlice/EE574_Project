"""
Time-varying load / generation profile for the measurement-stream generator.

The true network operating point changes over time because the loads vary.
This module produces, for every time instant, a per-bus load multiplier that is
applied to the base loads parsed from the CDF file.  The shape is:

    bus_factor(t, i) = clip( diurnal(t) + ar_global(t) + div * ar_bus(t, i),
                             lam_min, lam_max )

  * diurnal(t)   - a slow sinusoidal "daily" trend compressed into the horizon
  * ar_global(t) - a small system-wide AR(1) fluctuation (correlated load drift)
  * ar_bus(t, i) - an independent small AR(1) fluctuation per bus (load diversity)

Generation follows the load: each generator's real output is scaled so that
total generation tracks total load (the slack bus absorbs losses and the
residual).  Reactive load scales with active load at constant power factor.

Everything is driven by a single seeded RNG, so a given (seed, timestamps,
params) always yields the same trajectory.
"""
import numpy as np

from .powerflow import base_injections


def _ar1(rng, n_steps, rho, sigma):
    """One AR(1) sample path of length n_steps, mean 0, started at 0."""
    e = np.zeros(n_steps)
    for k in range(1, n_steps):
        e[k] = rho * e[k - 1] + sigma * rng.standard_normal()
    return e


class LoadProfile:
    """Pre-computes per-bus load multipliers and bus injections per timestamp."""

    def __init__(self, net, timestamps, seed=0,
                 diurnal_amp=0.12, period=None, phase=0.0,
                 ar_rho=0.9, ar_sigma=0.004,
                 div_amp=0.03, div_rho=0.85, div_sigma=0.006,
                 lam_min=0.6, lam_max=1.4, scale_gen=True):
        """
        Parameters
        ----------
        net        : Network
        timestamps : sorted list/array of unique time instants (seconds)
        seed       : RNG seed (reproducibility)
        diurnal_amp: amplitude of the slow sinusoidal trend
        period     : period of the diurnal sinusoid in seconds
                     (default: the full horizon, i.e. one cycle over the run)
        ar_*       : AR(1) params for the system-wide fluctuation
        div_*      : AR(1) params + amplitude for per-bus load diversity
        lam_min/max: clip range for the multiplier (keeps the PF solvable)
        scale_gen  : if True, scale generator real output to track total load
        """
        self.net = net
        self.timestamps = np.asarray(timestamps, dtype=float)
        self.scale_gen = scale_gen
        rng = np.random.RandomState(seed)

        n = net.n
        K = len(self.timestamps)
        horizon = float(self.timestamps[-1] - self.timestamps[0]) or 1.0
        if period is None:
            period = horizon

        # slow diurnal trend (1.0-centred)
        diurnal = 1.0 + diurnal_amp * np.sin(
            2 * np.pi * (self.timestamps - self.timestamps[0]) / period + phase)

        # system-wide AR(1) drift
        ar_global = _ar1(rng, K, ar_rho, ar_sigma)

        # per-bus AR(1) diversity (independent paths)
        ar_bus = np.zeros((K, n))
        for i in range(n):
            ar_bus[:, i] = _ar1(rng, K, div_rho, div_sigma)

        factor = (diurnal[:, None] + ar_global[:, None] + div_amp * ar_bus)
        self.bus_factor = np.clip(factor, lam_min, lam_max)      # shape (K, n)
        self.lam_global = np.clip(diurnal + ar_global, lam_min, lam_max)

        # base data
        (self.P_spec0, self.Q_spec0,
         self.P_load0, self.Q_load0,
         self.P_gen0, self.Q_gen0) = base_injections(net)
        self.is_gen = np.array([b['type'] in (2, 3) for b in net.buses])
        self.total_pload0 = float(np.sum(self.P_load0))
        self.total_pgen0  = float(np.sum(self.P_gen0[self.is_gen]))

        self._index = {round(float(t), 9): k for k, t in enumerate(self.timestamps)}

    def k_of(self, t):
        return self._index[round(float(t), 9)]

    def injections(self, t):
        """
        Return (P_spec, Q_spec) length-n arrays of specified injections (pu)
        for the operating point at time t.
        """
        k = self.k_of(t)
        f = self.bus_factor[k]                       # per-bus multiplier
        P_load = f * self.P_load0
        Q_load = f * self.Q_load0

        P_gen = self.P_gen0.copy()
        if self.scale_gen and self.total_pgen0 > 1e-9:
            # scale generator real output so total gen tracks total load
            gen_scale = float(np.sum(P_load)) / self.total_pload0 \
                if self.total_pload0 > 1e-9 else 1.0
            P_gen[self.is_gen] = self.P_gen0[self.is_gen] * gen_scale

        P_spec = P_gen - P_load
        Q_spec = self.Q_gen0 - Q_load
        return P_spec, Q_spec

    def lam(self, t):
        return float(self.lam_global[self.k_of(t)])
