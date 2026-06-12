# GUI Solver — Test Findings Report

**Date:** 2026-06-12
**Scope:** `gui_solver.py` + `solve_timeseries.py` pipeline, tested with freshly generated data.
**Status:** findings only — no fixes applied.

## Test setup

Fresh stream generated from `ieee_cdf_sample.dat` (14 buses, 17 branches, slack 1):

```
python gen_measurements.py --cdf ieee_cdf_sample.dat --out results/_guitest/case_TEST \
       --horizon 60 --pmu-dt 1 --scada-dt 5 --seed 42
python inject_bad_data.py --dir results/_guitest/case_TEST --seed 7
```

- 61 PMU snapshots (1 s) at buses {2, 5, 9, 10, 12, 14}; 13 SCADA snapshots (5 s)
- Corrupted twin: 102 gross errors across 74/74 files (24 SE-type), incl. 2 persistent meters
- GUI exercised headlessly: solve, render of every instant (clean + bad), dataset
  toggles, play/pause, error paths, edge-case inputs

The happy path works: both twins solve, all 26 instant renders complete without
errors, twins auto-discovery / manifest pre-fill / clean-bad toggle all behave.

---

## 1. Critical bugs

### 1.1 Any solve error permanently bricks the GUI (`gui_solver.py:319-322`) — **FIXED 2026-06-12**

*(Fixed by binding `msg = str(exc)` before the deferred lambda. Verified: all
error paths now show the message and re-enable Solve; recovery solve works.)*

The worker's `except` block defers the error display:

```python
except Exception as exc:
    self.after(0, lambda: (self._set_status(f'Error: {exc}', RED),
                           self._run_btn.configure(state='normal')))
```

Python deletes `exc` when the `except` block exits, and the lambda runs later on
the Tk event loop → `NameError: cannot access free variable 'exc'`. **Confirmed
live:** the status bar never shows the error and the Solve button stays disabled
forever (the `configure` in the same lambda never runs). App restart required.

### 1.2 Default data folder errors on first click (`gui_solver.py:128`) — **FIXED 2026-06-12**

*(Fixed with `default_stream_dir()`: picks the first solvable case under
results/timeseries, preferring case_IEEE14; falls back to the base folder.
Verified: out-of-the-box first Solve now loads clean+bad IEEE14 twins.)*

`_dir` defaults to `results/timeseries` — a *parent* of case folders with no
`index.csv`. Opening the app and pressing Solve immediately hits the
"no index.csv" error, which via 1.1 bricks the button. The out-of-the-box first
click fails.

## 2. Bugs

### 2.1 Invalid threshold after solving crashes the render path (`gui_solver.py:517`)

`_draw_residuals` calls `float(self._thr.get() or 3.0)` on every redraw. Editing
the threshold field to a non-numeric value and scrubbing the slider raises
`ValueError` inside a Tk callback (confirmed) — the residual panel dies.

### 2.2 Duplicate labels mis-color the residual chart (`gui_solver.py:509-515`)

SCADA and PMU both measure Vmag at PMU buses, so fused instants contain two
measurements with the identical label (confirmed: `Vmag(bus 2)` ×2 at every
instant). `_draw_residuals` marks removed bars by *label* (a set), so removing
the PMU copy colors **both** bars red.

### 2.3 Slider desyncs when dataset length shrinks (`gui_solver.py:359-369`)

After a PMU-only solve (61 instants) with the slider at 55, re-solving
SCADA-only (13 instants) reconfigures the range to 0–12 but leaves the slider
*value* at 55 (confirmed). Display is clamped, but the thumb sticks at the end
and the next Play tick jumps to `(55+1) % 13 = 4`.

## 3. Lower-priority issues

- **Play "Fast" can't keep up:** full `_show_instant` redraw (4 matplotlib
  figures rebuilt) averaged **206 ms** on 14 buses vs the 150 ms Fast tick;
  worse on larger cases (RAND40).
- **Threading hygiene** (`gui_solver.py:292-328`): worker swaps global
  `sys.stdout` and calls `self.after()` from a background thread. Works in
  practice; technically Tk is not thread-safe.
- **`_load_bad_records` swallows parse errors** (`gui_solver.py:631-632`): a
  malformed `bad_log.csv` silently shows "0 corrupted" in the compare tab.
- **Dead work in `_compare_baseline`** (`gui_solver.py:642-670`): parses clean
  and bad files per instant and computes `dsig`/`is_bad`, but only
  `len(baseline)` is ever used (caption count).
- **Label format inconsistency:** se_solver labels read `Vmag(bus 2)`; the
  compare tab's `_lbl` produces `Vmag(2)` — same channel named two ways.
- **Progressbar never resets** between runs.
- **GUI doesn't expose `--instants`:** fused mode always solves at SCADA
  cadence (see 4.1); the CLI's `pmu`/`all` options are unreachable from the GUI.
- **Unobservable instants still show plausible RMSE/state values** (see 4.3):
  the "Observable: NO" tile turns red, but RMSE tiles and the state table still
  display numbers from the non-converged last iterate.

## 4. Estimator behavior — SCADA / PMU evaluation

### 4.1 Cadence: fused mode discards 48 of 61 PMU snapshots (by design)

With SCADA enabled, `solve_stream` solves only at SCADA instants (5 s)
(`solve_timeseries.py:174-181`). Rationale (in code comment): solving at 1 s
would fuse fresh PMU with a stale carried-forward SCADA scan → time-inconsistent
sets → bad-data false positives. Trade-off: zero staleness false alarms, but the
1 s PMU resolution is unused in fused mode. (A future improvement: inflate
sigmas on stale SCADA instead of skipping instants.)

### 4.2 Mode comparison on the test case

| Mode | Instants | Observable | RMSE V (pu) | RMSE ang (°) | Detection (bad twin) |
|---|---|---|---|---|---|
| SCADA+PMU | 13 (5 s) | 13/13 | **7.8e-4** | **0.046** | TP=20 FP=0 FN=0, F1=1.00 |
| SCADA only | 13 (5 s) | 13/13 | 3.5e-3 | 0.175–0.305 | TP=17 FP=0 FN=2, F1=0.94 |
| PMU only | 61 (1 s) | **0/61** | (7.6e-4)* | (0.63)* | TP=0 FN=5, **recall=0.00** |

\* misleading — values come from non-converged, unobservable solves (see 4.3).

- **SCADA+PMU** — headline mode: best accuracy, perfect detection.
- **SCADA only** — legitimate degraded mode: ~4.5× worse voltage accuracy,
  4–7× worse angles, recall 0.89.
- **PMU only** — structurally broken on this case (see 4.3).

### 4.3 PMU-only is structurally unobservable — current phasors are discarded

The generator writes PMU branch-current phasors (verified in the raw files:
12 Imag + 12 Iang entries), but `parse_measurement_file` drops Imag/Iang by
default (`src/parser.py:203-231`, "The WLS state estimator does not use current
phasors") and `se_solver.py` has no current-phasor measurement model. What
remains is 6 Vmag + 6 Vang = 12 measurements for 27 states → unobservable at
every instant, non-converged, and all bad data missed.

PMU-only can never work on this network unless (a) the estimator gains
current-phasor measurement functions, or (b) PMUs cover nearly all buses.
The GUI allows selecting PMU-only with no warning.

## 4b. Additional CDF cases — RAND20 / RAND40 (added later)

Repo offers 8 CDFs in `testing/cases/` (3–40 buses). IEEE14 duplicates
`ieee_cdf_sample.dat`; the 3/4/5/7-bus cases are too small to stress anything.
**RAND40 is the most useful test case** — it produced the first imperfect
detection (F1=0.95 with a real FP and FN) and the worst render times; RAND20 is
a good mid-size demo case (still F1=1.00).

### Generator bugs found while creating these streams

1. **`--loads-in-pu` help text is a trap.** The help says testing/cases/*.dat
   are per-unit "(as in testing/cases/*.dat)", but the working
   `results/timeseries/case_RAND40` manifest says `loads_in_pu: false`. Using
   the flag on these files multiplies loads ×100 → **power flow diverges at all
   61 instants** (RAND20 and RAND40 both). The GUI's "loads in pu" checkbox has
   the same trap.
2. **`gen_measurements.py` writes a full stream even when every power flow
   diverges.** Only a one-line console note and a manifest field
   (`pf_non_converged: 61`) record it; `truth.csv` and all snapshots contain
   garbage, and the exit code is 0.
3. **Fixed-width writer overflow corrupts files.** Diverged values like
   `-24960.29` overflow the column width and run into the adjacent field
   (`'8-24960.29112'`), making the files unparseable — `inject_bad_data.py`
   crashes inside `parse_measurement_file`. The meas-writer/parser round-trip
   is broken for large magnitudes.

### Mode comparison (correctly generated, loads_in_pu=false)

| Case | Mode | Inst. | Obs. | RMSE V (pu) | RMSE ang (°) | Detection (bad) |
|---|---|---|---|---|---|---|
| RAND20 | SCADA+PMU | 13 | 13/13 | 1.3e-3 | 0.064 | F1=1.00 (TP=20) |
| RAND20 | SCADA only | 13 | 13/13 | 3.0e-3 | 0.143 | F1=1.00 (TP=19) |
| RAND20 | PMU only | 61 | 0/61 | (5.9e-3)* | (0.53)* | R=0.00 (FN=8) |
| RAND40 | SCADA+PMU | 13 | 13/13 | 2.1e-3 | 0.313 | **F1=0.95** (TP=18 FP=1 FN=1) |
| RAND40 | SCADA only | 13 | 13/13 | 3.1e-3 | 0.380 | F1=0.95 (TP=18 FP=1 FN=1) |
| RAND40 | PMU only | 61 | 0/61 | (5.8e-3)* | (0.73)* | R=0.00 (FN=5) |

\* non-converged/unobservable solves (same Imag/Iang root cause as 4.3 —
13 PMU buses × 2 = 26 measurements vs 79 states on RAND40).

- PMU-only unobservability is **systemic**, not IEEE14-specific: every case
  fails the same way because current phasors are parsed away.
- GUI renders all RAND40 instants without errors; avg `_show_instant` =
  **260 ms** on 40 buses (vs 206 ms on 14) — confirms Play "Fast" (150 ms)
  and approaches "Medium" (380 ms) budget.

## 4c. Second RAND20 measurement set — critical-measurement blind spot

A second measurement set on the same 20-bus network (seeds 99/13, vs 42/7)
collapsed detection: **recall 0.35 vs 1.00**, with angle RMSE degrading 16×
(1.019° vs 0.064°) because the undetected error biases the state. A 5% PMU
dropout variant gave identical results (dropout itself is handled fine).

| RAND20 set | SCADA+PMU detection | RMSE ang after removal |
|---|---|---|
| set 1 (42/7) | F1=1.00 (TP=20) | 0.064° |
| set 2 (99/13) | **F1=0.52 (TP=7 FN=13)** | **1.019°** |
| set 2 + inj at all buses | F1=1.00 (TP=20 FN=0) | 0.059° |

**Root cause — not a solver bug.** All 13 FN are one persistent meter:
`Pflow(9-13)`, a 46.6σ polarity flip. Bus 13 is a radial leaf, its only branch
is 9-13, and with `--inj-buses gen` (default) neither bus 9 nor 13 has
injection measurements. `Pflow(9-13)` is therefore (near-)critical: the WLS
solution absorbs the flipped flow into θ13, its normalized residual stays at
1.56 (< 3.0), and J=46.2 stays under the chi-square gate. Textbook result: bad
data on a critical measurement is undetectable by LNR.

**Controlled proof** (same instant, same flipped measurement, only redundancy
differs):

| Configuration | n_meas | \|r_n\| of Pflow(9-13) | Outcome |
|---|---|---|---|
| inj at gen buses only | 98 | 1.56 | **not detected**, absorbed into θ13 |
| inj at all buses | 124 | 32.08 | correctly identified & removed |

**Implications**
- Detection scores are seed/placement-dependent: set 1 scored F1=1.00 only
  because its persistent faults happened to land on redundant channels.
- Meter placement matters more than the detector: `--inj-buses all` (or a PMU
  at leaf buses) closes the blind spot entirely.
- The demo/report should not present single-seed detection scores as the
  detector's quality; run multiple seeds or report criticality.

## 5. Suggested improvements (not applied)

1. Fix the `exc` capture in the worker except block (bind it:
   `except Exception as exc: msg = str(exc)` then use `msg` in the lambda) —
   restores both error display and the Solve button. *(critical)*
2. Point the default folder at an existing case (or validate before solving). *(critical)*
3. Parse/validate the threshold once per solve; reuse the value in renders. 
4. Key the residual chart's removed-set by measurement key, not label.
5. Reset the slider to 0 (or scale proportionally) in `_switch_dataset`.
6. Throttle Play: skip render if the previous frame is still drawing, or update
   artists in-place instead of rebuilding figures.
7. Grey out / suppress RMSE and state values when `observable` is false.
8. Warn (or disable PMU-only) when the PMU voltage-only set cannot observe the
   network; or implement Imag/Iang in the estimator to use the generated data.
9. Expose the `--instants` cadence choice in the GUI.
10. Make `gen_measurements.py` fail loudly (non-zero exit, no/flagged output)
    when power flows diverge; fix the `--loads-in-pu` help text to match the
    working configuration for `testing/cases/*.dat`.
11. Widen or delimit the measurement-writer columns so large values can't run
    into adjacent fields (round-trip safety).
12. ~~Flag critical / low-redundancy measurements in the solver output and
    GUI~~ — **IMPLEMENTED 2026-06-12.** `estimate_with_bad_data` now returns
    `critical`: channels where the minimum detectable gross error
    e_min = threshold·σ/√(Ω_ii/σ²) exceeds both a polarity flip (2|z|) and a
    10σ bias — i.e. no realistic gross error would trip the LNR test. The GUI
    shows an "Unverifiable" scorecard tile, an orange banner warning
    ("bad data there is undetectable"), gray rows in the bad-data table, and
    gray bars in the residual chart. Works without truth/bad_log (the
    CDF+measurements-only demo scenario). Validated: flags exactly
    `Pflow(9-13)` on RAND20 v2 (13/13 instants, the known miss), nothing on
    the inj=all variant, and found a real blind spot in the stock IEEE14
    stream (`Pflow(9-10)`, near-zero flow — confirmed undetectable by a
    controlled flip, |r_n|=0.49).

## Test artifacts

- `results/_guitest/case_TEST/` — clean stream (seed 42)
- `results/_guitest/case_TEST_bad/` — corrupted twin (seed 7, `bad_log.csv`)
- `results/_guitest/case_NOMAN/` — manifest-less copy (error-path test)
- `results/_guitest/case_R20[_bad]/`, `case_R40[_bad]/` — RAND20/RAND40 streams
  (seeds 42/7, loads_in_pu=false)
- `results/_guitest/case_R20v2[_bad]/` — RAND20 set 2 (seeds 99/13; the
  critical-measurement blind-spot case)
- `results/_guitest/case_R20drop[_bad]/` — set 2 with 5% PMU dropout
- `results/_guitest/case_R20v2inj[_bad]/` — set 2 with `--inj-buses all`
  (blind spot closed)

All throwaway; safe to delete.
