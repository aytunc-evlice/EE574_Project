"""Output and visualization utilities."""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_state_results(network, x, title="STATE ESTIMATION RESULTS"):
    """Print voltage magnitudes and angles for all buses."""
    V, theta = network.state_to_VT(x)
    print("\n" + "=" * 55)
    print(title)
    print("=" * 55)
    print(f"{'Bus':>4}  {'Name':<14}  {'V (pu)':>8}  {'theta (deg)':>11}")
    print("-" * 55)
    for i, b in enumerate(network.buses):
        print(f"{b['num']:>4}  {b['name']:<14}  {V[i]:>8.5f}  {np.degrees(theta[i]):>11.4f}")
    print("=" * 55)


def compare_with_loadflow(network, x, label=""):
    """
    Compare estimated states with the load flow solution stored in the bus data.
    Returns RMSE for voltage and angle.
    """
    V_est, theta_est = network.state_to_VT(x)
    V_lf = np.array([b['V'] for b in network.buses])
    theta_lf = np.array([b['theta'] for b in network.buses])  # radians

    dV = V_est - V_lf
    dth = np.degrees(theta_est - theta_lf)

    rmse_V = np.sqrt(np.mean(dV**2))
    rmse_th = np.sqrt(np.mean(dth**2))

    print(f"\n{'Estimation vs Load Flow':}")
    if label:
        print(f"  [{label}]")
    print(f"  RMSE Voltage (pu)   : {rmse_V:.6f}")
    print(f"  RMSE Angle (deg)    : {rmse_th:.6f}")
    print(f"  Max |dV| (pu)       : {np.max(np.abs(dV)):.6f}  at bus {network.buses[np.argmax(np.abs(dV))]['num']}")
    print(f"  Max |dTheta| (deg)  : {np.max(np.abs(dth)):.6f}  at bus {network.buses[np.argmax(np.abs(dth))]['num']}")
    return rmse_V, rmse_th


def save_voltage_plot(network, scenario_results, output_dir):
    """Plot and save voltage profiles for all scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    buses = [b['num'] for b in network.buses]
    # Load flow reference
    V_lf = np.array([b['V'] for b in network.buses])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1, ax2 = axes
    ax1.plot(buses, V_lf, 'k--', linewidth=2, label='Load Flow (reference)')
    ax2.plot(buses, np.degrees([b['theta'] for b in network.buses]),
             'k--', linewidth=2, label='Load Flow (reference)')

    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for sc, col in zip(scenario_results, colors):
        if sc.get('x') is None:
            continue
        V, theta = network.state_to_VT(sc['x'])
        ax1.plot(buses, V, 'o-', color=col, label=sc['name'], markersize=4)
        ax2.plot(buses, np.degrees(theta), 's-', color=col, label=sc['name'], markersize=4)

    ax1.set_xlabel('Bus number')
    ax1.set_ylabel('Voltage magnitude (pu)')
    ax1.set_title('Voltage Magnitudes')
    ax1.legend(fontsize=8)
    ax1.grid(True)

    ax2.set_xlabel('Bus number')
    ax2.set_ylabel('Voltage angle (deg)')
    ax2.set_title('Voltage Angles')
    ax2.legend(fontsize=8)
    ax2.grid(True)

    plt.tight_layout()
    path = os.path.join(output_dir, 'voltage_profiles.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Voltage profile plot saved: {path}")


def save_residual_plot(measurements, r_n_list, labels, output_dir):
    """Bar chart of normalized residuals across scenarios."""
    os.makedirs(output_dir, exist_ok=True)
    m = len(measurements)
    x_pos = np.arange(m)

    fig, ax = plt.subplots(figsize=(max(10, m * 0.5), 5))
    width = 0.8 / max(len(r_n_list), 1)
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    for k, (r_n, label) in enumerate(zip(r_n_list, labels)):
        offset = (k - len(r_n_list) / 2 + 0.5) * width
        ax.bar(x_pos + offset, np.abs(r_n), width=width * 0.9,
               label=label, color=colors[k % 3], alpha=0.8)

    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=1.5, label='Threshold (3.0)')
    meas_labels = []
    for m_obj in measurements:
        if m_obj['type'] in ('Vmag', 'Vang', 'Pinj', 'Qinj'):
            meas_labels.append(f"{m_obj['type']}\nbus {m_obj['bus']}")
        else:
            meas_labels.append(f"{m_obj['type']}\n{m_obj['from_bus']}-{m_obj['to_bus']}")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(meas_labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('|Normalized residual|')
    ax.set_title('Normalized Residuals by Scenario')
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(output_dir, 'normalized_residuals.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Residual plot saved: {path}")
