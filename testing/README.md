# Test-case generation & validation

Tooling to confirm the AC-WLS state estimator works on networks beyond the
bundled IEEE 14-bus case.

## TL;DR

```bash
python testing/cdf_gen.py        # generate CLEAN CDF test cases -> testing/cases/
python testing/cdf_corrupt.py    # generate BAD-DATA CDF cases   -> testing/cases_bad/
python testing/check_cdf.py      # verify the INPUT DATA is valid & clean
python testing/validate_cases.py # run the estimator on every clean case, PASS/FAIL
```

`validate_cases.py` auto-generates the cases if `testing/cases/` is empty, so
running it alone is enough.

Run `check_cdf.py` **before** trusting any estimator result: it confirms the
network data is well-formed and the generated measurements carry no unintended
bad data, so that an estimator failure points at the algorithm, not the input.

## What a "test case" is

You only need a **CDF network file**. You do **not** need to write `measure.dat`:
the SCADA + PMU measurements are generated from the network's operating point
(V, θ) by `src/simulation.py`. So a valid case = a connected network + a chosen
operating point, and the estimator should recover exactly that point.

`measure.dat` is only used by Scenario 1 in `run_pipeline.py` (the original
real-SCADA / unobservable demo) and is not required for new networks.

## `cdf_gen.py` — the generator

- Writes **column-exact** IEEE-CDF files (the parser is fixed-column, not
  whitespace-delimited). Every field offset mirrors a slice in `src/parser.py`,
  and each written file is **re-parsed and checked** (`_verify_roundtrip`) so a
  misaligned column fails loudly instead of silently corrupting data.
- Two families:
  - **Standard** — `case_3BUS`, `case_5BUS` (hand-built, with a transformer),
    and `case_IEEE14` (re-emitted from the repo's real file: a round-trip test
    against genuine CDF data).
  - **Random** — `case_RAND{4,7,10,20,40}`: connected random topologies
    (spanning tree + mesh), ~1-in-4 branches an off-nominal transformer.
- Operating points are chosen, then bus load/generation are **back-solved** from
  the network equations so the files are also valid solved-load-flow CDFs.

To add your own case, build `buses`/`branches` with `make_bus` / `make_branch`,
then `write_cdf(...)`; or call `case_random(n, seed)`.

## `cdf_corrupt.py` — bad-DATA generator

Generates CDF files with deliberate errors in the **network model itself**
(wrong tap, wrong impedance, wrong topology, wrong bus data) — *not* bad
measurement data. Each corrupted file is a clean base case with exactly **one
labeled fault**, plus a JSON manifest recording the ground truth.

Fault catalog (one fault per file):

| Code | Fault | Severity |
|---|---|---|
| `A1_tap_subtle` | transformer tap off one ~2% step (0.96→0.94) | subtle |
| `A1_tap_gross` | transformer tap ~20% off (0.96→0.80) | gross |
| `A2_Z_swap` | branch R and X swapped | gross |
| `A2_Z_scale` | branch R,X scaled ×10 | gross |
| `B1_endpoint` | branch rewired to a wrong bus | gross |
| `B2_orient_xfmr` | from/to flipped on a transformer (tap on wrong side) | subtle |
| `B3_missing` | a branch deleted | gross |
| `B4_parallel` | a branch duplicated | moderate |
| `C1_type_flip` | a PV bus turned PQ | moderate |
| `C2_voltage` | a bus voltage driven out of band (1.45) | gross |
| `C3_load_units` | load 100× too small (pu typed in MW field) | gross |
| `C3_loadgen_swap` | load and generation swapped at a bus | moderate |
| `C4_shunt` | wrong/spurious shunt susceptance | subtle |
| `C5_dup_busnum` | a bus number duplicated | invalid |

Output: `testing/cases_bad/case_<base>_<code>.dat` + matching `.json` manifest,
plus a top-level `INDEX.json` cataloguing all faults. Faults that don't apply to
a base network (tap/orientation faults need a transformer; `C1` needs a PV bus)
are skipped with a count. Each manifest looks like:

```json
{
  "file": "case_5BUS_A1_tap_subtle.dat",
  "source_case": "case_5BUS", "fault_code": "A1_tap_subtle",
  "category": "A-branch-parameter", "severity": "subtle",
  "expected_manifestation": "parses OK; Ybus slightly altered; near-undetectable",
  "target": {"type": "branch", "from": 2, "to": 4},
  "field": "tap", "original": 0.95, "corrupted": 0.93, "seed": 1007
}
```

The `severity: subtle` faults (small tap error, wrong transformer side, shunt
error) are the valuable ones — they don't announce themselves. Filter the
manifests by `severity` to build a detectability gradient.

Clean baselines stay in `testing/cases/`, so every corrupted file has a
before/after pair.

## `check_cdf.py` — input data-integrity checker

Validates the *data*, not the estimator. Three layers per file:

| Layer | Checks |
|---|---|
| **A. Structure / parameters** | one slack bus, unique bus ids, connected graph, no self-loops, X≠0, positive taps, plausible V/angles, no NaN/Inf, non-singular Ybus, full observability, gain-matrix conditioning |
| **B. Operating-point consistency** | power-balance mismatch `P_inj − (P_gen−P_load)` per bus; total active losses `Σ P_inj ≥ 0` (mandatory for R≥0) |
| **C. Measurement cleanliness** | at the true state, `|z − h(x*)|/σ` ~ N(0,1); any value ≫4σ would mean an accidental gross error |

Severities: **ERROR** = invalid/unusable (breaks the estimator), **WARN** =
unusual but tolerable, **OK**.

Notes on reading the output:
- **MISMATCH ~1e-2 pu is normal** — the CDF format stores V to 3 decimals and
  angle to 0.01°, so a round-tripped operating point has that much inherent
  power-balance error. Only a large mismatch (≫5e-2) means the load/gen columns
  aren't a power-flow solution. This never gates the estimator (it reads V/θ,
  not load/gen), so it is at most a WARN.
- `case_IEEE14` warns with a ~2.7 pu mismatch: the bundled file is the IEEE
  30-bus case truncated to 14 buses, so its load/gen columns are decorative.
  Harmless for the estimator; flagged for transparency.
- **maxErr/s** is the largest generated-measurement noise in σ units. Clean
  generation sits around 2σ; a value ≫4–5σ would flag unintended bad data.

## `validate_cases.py` — the gate

For each case it runs the real pipeline machinery and checks:

| Check | Meaning |
|---|---|
| **O** observable | synthetic measurement set has full Jacobian rank |
| **C** converged | WLS converges (warm-started from load flow, as `run_pipeline.py` does) |
| **R** state-recovered | estimate matches the file's true (V, θ) within RMSE < 1e-2 |
| **B** bad-data-detected | an injected +0.5 pu gross error flags above the 3.0 threshold |

The `B` flag carries a lowercase suffix for **sequential-removal auto-recovery**:
`y` = removing the largest-residual measurement(s) cleans the error and recovers
the state; `x` = detected but the wrong measurement is removed first.

`case_RAND10` shows `BYx`: it is a textbook **interacting bad-data** topology
(two correlated flows out of one bus) where the largest-normalized-residual rule
removes the wrong one and masks the error. This is a known limitation of the
method, not a bug — included deliberately as a stress case.

A case PASSES on O+C+R+B (detection); auto-recovery is reported, not required.
Exit code is non-zero if any case fails, so this works as a CI gate.
