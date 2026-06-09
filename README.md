# EE574 Semester Project – AC-WLS Power System State Estimator

## Overview

This project implements a realistic **AC Weighted Least Squares (WLS) State Estimator**
for an Energy Management System (EMS), applied to the IEEE 14-bus test network extracted
from the IEEE 30-bus Common Data Format file.

### Features implemented

| Requirement | Module |
|---|---|
| AC-WLS state estimation (Newton-based) | `src/estimator.py` |
| PMU phasor measurement integration | `src/simulation.py`, `run_pipeline.py` |
| Observability analysis (numerical rank) | `src/observability.py` |
| Bad data detection & identification (normalized residual test) | `src/bad_data.py` |
| Three-scenario comparison | `run_pipeline.py` |

---

## Quick Start

```bash
# Install dependencies (Python 3.9+)
pip install numpy scipy matplotlib

# Run the complete pipeline
python run_pipeline.py
```

Output is printed to the console and two plots are saved in `results/`:
- `results/voltage_profiles.png` – Estimated voltage magnitudes and angles vs load-flow
- `results/normalized_residuals.png` – Normalized residuals for Scenarios 2 and 3

---

## Project Structure

```
EE574_Project/
├── run_pipeline.py          # Single-command entry point
├── ieee_cdf_sample.dat      # IEEE 14-bus network (from 30-bus CDF)
├── measure.dat              # Original measurement file (for Scenario 1)
├── src/
│   ├── parser.py            # IEEE CDF + measurement file parser
│   ├── network.py           # Ybus construction + AC power flow equations
│   ├── estimator.py         # AC-WLS Newton iterations + line search
│   ├── observability.py     # Numerical rank-based observability check
│   ├── bad_data.py          # Normalized residual test
│   ├── simulation.py        # Synthetic measurement generator
│   ├── scenarios.py         # Scenario helpers (PMU augmentation, bad data injection)
│   └── results.py           # Printing + matplotlib plots
└── results/                 # Output plots (created on first run)
```

---

## Methodology

### AC-WLS State Estimation

The state vector is **x = [θ₂ … θₙ, V₁ … Vₙ]** (2N−1 states; θ₁ = 0 is the slack reference).

The WLS objective minimises **J(x) = (z − h(x))ᵀ W (z − h(x))** where
- **z** = measurement vector
- **h(x)** = nonlinear measurement functions (V, P/Q injections, P/Q flows)
- **W = R⁻¹** = diagonal weight matrix (inverse of measurement noise covariance)

Each iteration solves the normal equations:

```
(HᵀWH) Δx = Hᵀ W (z − h(x))
```

Convergence is accelerated with **backtracking line search**: the step size α is halved
until the WLS objective decreases, preventing divergence from bad initial conditions.

The transformer π-model is correctly implemented for branches 6→9, 6→10, and 4→12
(off-nominal tap ratios 0.978, 0.969, 0.932).

### Observability Analysis

Observability is checked numerically by computing the rank of the Jacobian matrix H at
flat start. The system is observable if and only if rank(H) equals the number of states.

### Bad Data Detection

After convergence the **normalized residual** for each measurement is

```
r̄ᵢ = rᵢ / √Ωᵢᵢ ,   where Ω = R − H(HᵀWH)⁻¹Hᵀ
```

If |r̄ᵢ| exceeds **threshold 3.0**, the measurement is flagged as bad data. The
measurement with the largest |r̄ᵢ| is identified and removed; the estimator then re-runs
without it.

---

## Scenarios

### Scenario 1 – Original SCADA (Unobservable)

Uses the 19 measurements from `measure.dat` (sections 1–6, excluding the angle reference
at the slack bus). The Jacobian has **rank 19 < 27** (deficit of 8 states). Buses 9, 10,
12, and 14 have no SCADA coverage, so their states cannot be uniquely determined.

The estimator is warm-started from the load-flow solution. The large normalized residuals
(|r̄| > 10 000 for some channels) reflect a model–measurement mismatch: `measure.dat` was
created for the full 30-bus network; applied to the 14-bus sub-model the measurements are
inconsistent.

### Scenario 2 – Synthetic SCADA + PMU (Fully Observable, Clean)

Synthetic SCADA measurements are generated from the 14-bus load-flow solution by adding
Gaussian noise (σ_V = 0.005 pu, σ_PQ = 0.01 pu). PMU phasors (σ_PMU = 0.0001 pu) are
added at buses **2, 5, 9, 10, 12, 14**, yielding **72 measurements** for 27 states
(45 redundant).

Results:
- Converges in **2 iterations**
- **RMSE voltage = 0.00055 pu**, RMSE angle = 0.016°
- **0 measurements exceed the bad-data threshold** ✓

### Scenario 3 – Bad Data Detection and Identification

A gross error of **+0.5 pu** is injected into the Pflow(1→3) measurement.

Results:
- Largest normalised residual: **Pflow(1→3), |r̄| = 44.9** → **correctly identified** ✓
- After removing the bad measurement and re-estimating: RMSE voltage = 0.00054 pu,
  RMSE angle = 0.018° (essentially identical to the clean Scenario 2)

### Summary Table

| Scenario | Measurements | Rank | Converged | WLS J | Correct BD ID |
|---|---|---|---|---|---|
| 1 – Original SCADA | 19 | 19/27 | YES* | 25101 | N/A |
| 2 – Synthetic SCADA + PMU | 72 | 27/27 | YES | 42.4 | — |
| 3 – Bad data injected | 72 | 27/27 | YES | 2236 | YES |

\* Warm-started from load-flow; unobservable buses retain load-flow values.

---

## Dependencies

| Package | Version tested | Purpose |
|---|---|---|
| numpy | ≥ 1.24 | Linear algebra, array operations |
| scipy | ≥ 1.10 | (imported; lstsq fallback) |
| matplotlib | ≥ 3.7 | Result plots |

Install with: `pip install numpy scipy matplotlib`

---

## Academic Integrity

This project was developed with assistance from **Claude (Anthropic)** as an AI coding
assistant. All theoretical formulations (AC power flow, WLS normal equations, normalized
residual test) are based on standard power system analysis references:

- A. Monticelli, *State Estimation in Electric Power Systems*, Kluwer, 1999.
- A.J. Wood & B.F. Wollenberg, *Power Generation, Operation and Control*, Wiley, 2013.
- IEEE CDF format: "Common Data Format for Exchange of Solved Load Flow Data",
  *IEEE Trans. PAS*, Vol. PAS-92, No. 6, 1973.

No black-box state estimation solver was used. All equations are implemented from scratch
in `src/network.py` and `src/estimator.py`.
