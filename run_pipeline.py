"""
EE574 Semester Project - AC-WLS State Estimator Pipeline
=========================================================
Run:   python run_pipeline.py
Output: results/ directory with plots and printed reports.

Components
----------
1.  Parse IEEE CDF network and original measurement file (measure.dat).
2.  Observability analysis with the original SCADA set.
3.  Three AC-WLS scenarios:
      Scenario 1 - Original SCADA measurements (unobservable, realistic data)
      Scenario 2 - Synthetic SCADA + PMU (fully observable, consistent data)
      Scenario 3 - Scenario 2 with injected gross error -> detection & removal
4.  Normalized-residual bad data detection in Scenario 3.
5.  Voltage-profile and residual bar plots saved to results/.

Note on Scenario 1:
    The original measure.dat was created for the full 30-bus system; the
    measurements reference states from all 30 buses. Applied to the 14-bus
    network sub-model they produce large residuals, illustrating that partial
    network observability leads to poor estimation quality.

Note on Scenarios 2-3:
    Synthetic SCADA measurements (Gaussian noise added to the load-flow values)
    are used so that the state estimation and bad data detection can be validated
    against a known ground truth.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.parser      import parse_ieee_cdf, parse_measurements
from src.network     import Network
from src.estimator   import wls_estimate
from src.observability import observability_report
from src.bad_data    import (compute_normalized_residuals, bad_data_report,
                              identify_bad_data)
from src.scenarios   import (scenario_3_bad_data, print_scenario_comparison)
from src.simulation  import generate_scada_measurements, generate_pmu_measurements
from src.results     import (print_state_results, compare_with_loadflow,
                              save_voltage_plot, save_residual_plot)

np.random.seed(42)

DATA_DIR    = os.path.dirname(os.path.abspath(__file__))
CDF_FILE    = os.path.join(DATA_DIR, 'ieee_cdf_sample.dat')
MEAS_FILE   = os.path.join(DATA_DIR, 'measure.dat')
RESULTS_DIR = os.path.join(DATA_DIR, 'results')

# PMU placement: covers all previously unobservable buses (9,10,12,14)
#                plus redundancy at generator buses 2,5 for bad data detection.
PMU_BUSES = [2, 5, 9, 10, 12, 14]

# ──────────────────────────────────────────────────────────────────────────────
# 1. PARSE NETWORK
# ──────────────────────────────────────────────────────────────────────────────
print("\n[1] Parsing network ...")
buses, branches, mva_base = parse_ieee_cdf(CDF_FILE)
net = Network(buses, branches)
print(f"    {net.n} buses, {len(branches)} branches, MVA base = {mva_base:.0f} MVA")
print(f"    Slack bus: {net.buses[net.slack_idx]['num']} "
      f"({net.buses[net.slack_idx]['name']})")
print(f"    States: {net.n_states}  ({net.n-1} angles + {net.n} voltages)")

valid_buses       = {b['num'] for b in buses}
valid_branch_set  = {(br['from_bus'], br['to_bus']) for br in branches}
for br in branches:
    valid_branch_set.add((br['to_bus'], br['from_bus']))
slack_bus_num = net.buses[net.slack_idx]['num']

# ── Original measurements from measure.dat ────────────────────────────────────
orig_meas = parse_measurements(MEAS_FILE, valid_buses, valid_branch_set,
                                max_sections=6)
orig_meas = [m for m in orig_meas
             if not (m['type'] == 'Vang' and m.get('bus') == slack_bus_num)]
print(f"\n    Original SCADA measurements (measure.dat, secs 1-6): {len(orig_meas)}")

# ── Synthetic measurements generated from load-flow ───────────────────────────
synth_scada = generate_scada_measurements(net, sigma_vmag=0.005, sigma_pq=0.01,
                                           seed=42)
print(f"    Synthetic SCADA measurements generated: {len(synth_scada)}")
synth_pmu   = generate_pmu_measurements(net, PMU_BUSES, sigma_pmu=0.0001, seed=42)
print(f"    Synthetic PMU measurements (buses {PMU_BUSES}): {len(synth_pmu)}")
synth_full  = synth_scada + synth_pmu

# ──────────────────────────────────────────────────────────────────────────────
# 2. OBSERVABILITY ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[2] Observability analysis ...")
print("\n  -- Original SCADA (measure.dat, 6 sections) --")
obs1, rank1 = observability_report(net, orig_meas)

print("\n  -- Synthetic SCADA + PMU --")
obs2, rank2 = observability_report(net, synth_full)

# ──────────────────────────────────────────────────────────────────────────────
# 3. SCENARIO 1 - ORIGINAL SCADA (Unobservable)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[3] Scenario 1: Original SCADA measurements (unobservable)")
print(f"    {len(orig_meas)} measurements, Jacobian rank {rank1}/27 (deficit {27-rank1})")
x1, conv1, iters1, res1, J1 = wls_estimate(
    net, orig_meas, verbose=True, use_loadflow_start=True)
print_state_results(net, x1, "SCENARIO 1 - Original SCADA (rank-deficient)")
rmse1_V, rmse1_th = compare_with_loadflow(net, x1, label="Scenario 1")

# Show that normalized residuals reveal large model-measurement mismatch
print("\n  Top-5 normalized residuals (Scenario 1):")
_, r_n1_raw, _ = compute_normalized_residuals(net, orig_meas, x1)
top5 = sorted(enumerate(r_n1_raw), key=lambda t: abs(t[1]), reverse=True)[:5]
for idx, rn in top5:
    m = orig_meas[idx]
    if m['type'] in ('Vmag','Vang','Pinj','Qinj'):
        lbl = f"{m['type']}(bus {m['bus']})"
    else:
        lbl = f"{m['type']}({m['from_bus']}-{m['to_bus']})"
    print(f"    [{idx:2d}] {lbl:<25}  r_n = {rn:+.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. SCENARIO 2 - SYNTHETIC SCADA + PMU (Fully observable, clean)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[4] Scenario 2: Synthetic SCADA + PMU (fully observable, consistent data)")
print(f"    {len(synth_full)} measurements, rank {rank2}/27")
x2, conv2, iters2, res2, J2 = wls_estimate(
    net, synth_full, verbose=True, use_loadflow_start=True)
if conv2:
    print_state_results(net, x2, "SCENARIO 2 - Synthetic SCADA + PMU (clean)")
    rmse2_V, rmse2_th = compare_with_loadflow(net, x2, label="Scenario 2")

    # Verify all normalized residuals below threshold
    _, r_n2, _ = compute_normalized_residuals(net, synth_full, x2)
    flagged = [(i, m, r_n2[i]) for i, m in enumerate(synth_full)
               if abs(r_n2[i]) > 3.0]
    print(f"\n    Measurements exceeding threshold 3.0: {len(flagged)}")
    for idx, m, rn in sorted(flagged, key=lambda t: abs(t[2]), reverse=True)[:5]:
        if m['type'] in ('Vmag','Vang','Pinj','Qinj'):
            lbl = f"{m['type']}(bus {m['bus']})"
        else:
            lbl = f"{m['type']}({m['from_bus']}-{m['to_bus']})"
        print(f"      [{idx:2d}] {lbl:<25}  r_n = {rn:+.3f}")
else:
    print("  WARNING: Scenario 2 did not converge.")

# ──────────────────────────────────────────────────────────────────────────────
# 5. SCENARIO 3 - BAD DATA (Inject gross error + detect + remove)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[5] Scenario 3: Scenario 2 + injected gross error in one flow measurement")
# Pick a Pflow measurement that is not connected to a PMU bus for a clean test
target = next(
    (i, m) for i, m in enumerate(synth_full)
    if m['type'] == 'Pflow'
    and m.get('from_bus') not in PMU_BUSES
    and m.get('to_bus')   not in PMU_BUSES
)
bad_idx, bad_m = target
gross_err = 0.5
s3_meas, orig_val = scenario_3_bad_data(synth_full, bad_idx,
                                         gross_error_pu=gross_err)
print(f"    Injected +{gross_err} pu into measurement [{bad_idx}]: "
      f"Pflow({bad_m['from_bus']}->{bad_m['to_bus']})  "
      f"{orig_val:.5f} -> {s3_meas[bad_idx]['value']:.5f}")

x3, conv3, iters3, res3, J3 = wls_estimate(
    net, s3_meas, verbose=True, use_loadflow_start=True)

print("\n  Bad Data Detection:")
suspects = bad_data_report(net, s3_meas, x3)

if suspects:
    idx_found, label_found, rn_found = suspects[0]
    correct = (idx_found == bad_idx)
    print(f"\n  Largest normalized residual: [{idx_found}] {label_found}  "
          f"|r_n| = {rn_found:.3f}")
    print(f"  Injected bad measurement was: [{bad_idx}] "
          f"Pflow({bad_m['from_bus']}->{bad_m['to_bus']})")
    print(f"  Identification correct: {'YES' if correct else 'NO'}")

    # Re-estimate after removing the flagged measurement
    print("\n  Re-estimating without identified bad measurement ...")
    clean_meas = [m for k, m in enumerate(s3_meas) if k != idx_found]
    x3b, conv3b, _, res3b, J3b = wls_estimate(
        net, clean_meas, verbose=False, use_loadflow_start=True)
    print_state_results(net, x3b, "SCENARIO 3 - After Bad Data Removal")
    compare_with_loadflow(net, x3b, label="Scenario 3 (cleaned)")

# ──────────────────────────────────────────────────────────────────────────────
# 6. COMPARISON SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
sc_results = [
    {'name': 'Sc1: Original SCADA (unobservable)',
     'converged': conv1, 'iterations': iters1,
     'n_meas': len(orig_meas), 'J_obj': J1, 'x': x1},
    {'name': 'Sc2: Synthetic SCADA + PMU (clean)',
     'converged': conv2, 'iterations': iters2,
     'n_meas': len(synth_full), 'J_obj': J2,
     'x': x2 if conv2 else None},
    {'name': 'Sc3: Bad data injected (detected)',
     'converged': conv3, 'iterations': iters3,
     'n_meas': len(s3_meas), 'J_obj': J3, 'x': x3},
]
print_scenario_comparison(sc_results)

# ──────────────────────────────────────────────────────────────────────────────
# 7. PLOTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n[7] Saving result plots ...")
save_voltage_plot(net, sc_results, RESULTS_DIR)

r_n_list, labels_rn = [], []
if conv2:
    _, r_n2p, _ = compute_normalized_residuals(net, synth_full, x2)
    r_n_list.append(r_n2p);  labels_rn.append('Sc2: Clean')
if conv3:
    _, r_n3p, _ = compute_normalized_residuals(net, s3_meas, x3)
    r_n_list.append(r_n3p);  labels_rn.append('Sc3: Bad data')
if r_n_list:
    save_residual_plot(synth_full, r_n_list, labels_rn, RESULTS_DIR)

print("\nPipeline complete.  Results in:", RESULTS_DIR)
