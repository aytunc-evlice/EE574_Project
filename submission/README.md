# EE574 — AC-WLS Power System State Estimator

**Course:** EE574 Power System Real-Time Monitoring & Control — Semester Project
**Authors:** Ogün Altun (2165785) · Aytunç Evlice (2516177)

A complete AC weighted-least-squares (WLS) state estimator implemented from
scratch in Python: IEEE CDF network parser, Newton solver with analytical
Jacobian, transformer π-model with off-nominal taps, numerical observability
analysis, PMU integration, and normalized-residual bad data detection &
removal. Works both as a single-shot pipeline and as a time-series estimator
tracking a moving system state.

---

## 1. Requirements

- Python 3.10+ (tested on 3.12/3.14)
- numpy, matplotlib (`tkinter` ships with standard Python installers)

```
pip install -r requirements.txt
```

No other setup is needed — every command below is run **from this folder**.

---

## 2. Folder contents

| Path | Description |
|------|-------------|
| `run_pipeline.py` | Main entry point: 3-scenario state estimation pipeline |
| `gui.py` | Graphical interface for the pipeline (select CDF + measurements, view results) |
| `gen_measurements.py` | Time-varying SCADA (5 s) / PMU (1 s) measurement-stream generator |
| `inject_bad_data.py` | Injects gross errors into a generated stream (for detection studies) |
| `solve_timeseries.py` | Runs the estimator across a measurement stream (command line) |
| `run_timeseries.py` | Alternative stream runner with tracking plots |
| `gui_solver.py` | Graphical interface for time-series streams (network view, states, residuals) |
| `src/` | Estimator library (parser, network model, power flow, WLS solver, observability, bad data) |
| `ieee_cdf_sample.dat` | IEEE 14-bus network in IEEE Common Data Format |
| `measure.dat` | Original 19-measurement SCADA set for the IEEE 14-bus network |
| `cases/case_IEEE14.dat` | IEEE 14-bus CDF test case |
| `cases/case_RAND20.dat` | Randomized 20-bus CDF test case |
| `cases/case_RAND40.dat` | Randomized 40-bus CDF test case |
| `read_me_ieee_cdf.txt` | IEEE CDF file-format reference |
| `read_me_meas.txt` | Measurement file-format reference |

All outputs (plots, CSV files) are written to a `results/` folder created on
first run.

---

## 3. Quick start

### A. Run the full pipeline on the IEEE 14-bus system

```
python run_pipeline.py
```

This runs three scenarios and prints a summary table:

1. **Scenario 1 — Original SCADA** (`measure.dat`, 19 measurements):
   observability analysis shows rank 19/27 → buses 9, 10, 12, 14 unobservable.
2. **Scenario 2 — Synthetic SCADA + PMU** (fully observable, clean):
   converges in 2 iterations, RMSE ≈ 10⁻⁴ pu.
3. **Scenario 3 — Gross error injected**: the normalized-residual test
   (|rᴺ| > 3.0) identifies and removes the bad measurement, state recovers.

Voltage-profile and residual plots are saved to `results/`.

### B. Run on the 20-bus / 40-bus test cases

```
python run_pipeline.py --cdf cases/case_RAND20.dat
python run_pipeline.py --cdf cases/case_RAND40.dat
```

(no measurement file exists for these → Scenario 1 is skipped automatically;
measurements are generated synthetically from a power-flow solution).

### C. Run with instructor-provided files

```
python run_pipeline.py --cdf <network.dat> --meas <measurements.dat> --out results/mycase
```

- `--cdf`  any network in IEEE Common Data Format (see `read_me_ieee_cdf.txt`)
- `--meas` any measurement file in the 8-section format (see `read_me_meas.txt`)

Observability and bad data are always checked on the provided measurement set.

### D. Graphical interface

```
python gui.py
```

Browse for a CDF file and (optionally) a measurement file, press **Run** —
observability report, estimated states, residuals and bad data flags are
shown in the tabs.

---

## 4. Time-series operation (moving system state)

The estimator can track a time-varying state, fusing PMU phasors (1 s) with
SCADA scans (5 s) exactly like an EMS.

```
# 1. generate a 60 s measurement stream (IEEE-14, PMUs at 2,5,9,10,12,14)
python gen_measurements.py --cdf cases/case_IEEE14.dat --out results/timeseries/case_IEEE14

# 2. (optional) inject gross errors into a copy of the stream
python inject_bad_data.py --dir results/timeseries/case_IEEE14

# 3. solve across the stream — prints RMSE and bad-data report, saves tracking plot
python solve_timeseries.py --dir results/timeseries/case_IEEE14
python solve_timeseries.py --dir results/timeseries/case_IEEE14_bad

# 4. or explore the stream interactively
python gui_solver.py
```

`gui_solver.py` opens on the most recent generated stream (use **Browse** to
pick another). It shows the network one-line diagram, the estimated state and
bad data table, and the normalized-residual chart with a time slider.

The same works for the 20- and 40-bus cases, e.g.
`python gen_measurements.py --cdf cases/case_RAND40.dat --out results/timeseries/case_RAND40`.

---

## 5. Interpreting the output

- **Observability:** `rank(H) = 2N−1` means observable; otherwise the
  unobservable buses are listed and their estimates are not trustworthy.
- **WLS objective J:** for clean data J ≈ degrees of freedom
  (m − n_states). A much larger J signals bad data or model mismatch.
- **Normalized residuals:** |rᴺ| ≤ 3.0 is consistent with noise; the largest
  |rᴺ| > 3.0 is flagged, removed, and the state is re-estimated iteratively.
- **Caveat:** a *critical* measurement (the only source of information for a
  state) produces a near-zero residual — its errors cannot be detected by any
  residual test. Detection capability is a property of metering redundancy.

---

## 6. References

- A. Monticelli, *State Estimation in Electric Power Systems*, Kluwer, 1999
- A.J. Wood & B.F. Wollenberg, *Power Generation, Operation and Control*, Wiley, 2013
- IEEE Common Data Format, *IEEE Trans. PAS*-92(6), 1973
