"""
Three scenario definitions for comparing state estimation performance.

Scenario 1: Base SCADA measurements (from measure.dat) – may be unobservable
Scenario 2: SCADA + PMU phasor measurements at selected buses (full observability)
Scenario 3: Scenario 2 with one injected bad data measurement
"""
import numpy as np
import copy


def scenario_1_base(measurements):
    """
    Scenario 1: Use base measurements from measure.dat as-is.
    This scenario may reveal observability issues.
    """
    return copy.deepcopy(measurements)


def scenario_2_with_pmu(measurements, pmu_buses):
    """
    Scenario 2: Augment base measurements with PMU Vmag + Vang at pmu_buses.
    PMU buses should be provided with their true state values (from loadflow).
    pmu_buses: list of dicts {bus, V, theta} where theta is in radians.
    """
    meas = copy.deepcopy(measurements)
    pmu_sigma = 0.0001   # PMU measurements are very accurate
    for p in pmu_buses:
        meas.append({
            'type': 'Vmag', 'bus': p['bus'],
            'value': p['V'] + np.random.normal(0, pmu_sigma),
            'sigma': pmu_sigma,
        })
        meas.append({
            'type': 'Vang', 'bus': p['bus'],
            'value': p['theta'] + np.random.normal(0, pmu_sigma),
            'sigma': pmu_sigma,
        })
    return meas


def scenario_3_bad_data(measurements, bad_idx, gross_error_pu=0.5):
    """
    Scenario 3: Inject a gross error into measurement at bad_idx.
    gross_error_pu: magnitude of the gross error in pu.
    """
    meas = copy.deepcopy(measurements)
    original_value = meas[bad_idx]['value']
    meas[bad_idx]['value'] = original_value + gross_error_pu
    meas[bad_idx]['_injected_error'] = gross_error_pu
    return meas, original_value


def print_scenario_comparison(results):
    """
    Print side-by-side comparison of scenario results.
    results: list of dicts with keys: name, converged, iterations, J_obj, n_meas
    """
    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Scenario':<35}  {'Conv':>5}  {'Iters':>5}  {'Meas':>5}  {'J_obj':>12}")
    print("-" * 70)
    for r in results:
        conv_str = "YES" if r['converged'] else "NO"
        print(f"{r['name']:<35}  {conv_str:>5}  {r['iterations']:>5}  "
              f"{r['n_meas']:>5}  {r['J_obj']:>12.4f}")
    print("=" * 70)
