"""Parse IEEE Common Data Format and measurement files."""
import numpy as np


def parse_ieee_cdf(filename):
    """
    Parse IEEE CDF file. Returns (buses, branches, mva_base).
    buses: list of dicts with keys: num, name, type, V, theta, P_load, Q_load,
           P_gen, Q_gen, G_sh, B_sh
    branches: list of dicts with keys: from_bus, to_bus, R, X, B, tap
    """
    buses = []
    branches = []
    mva_base = 100.0

    with open(filename, 'r') as f:
        lines = f.readlines()

    section = 'title'
    for line in lines:
        stripped = line.rstrip()
        if not stripped:
            continue

        if stripped.startswith('BUS DATA'):
            section = 'bus'
            continue
        elif stripped.startswith('BRANCH DATA'):
            section = 'branch'
            continue
        elif stripped.strip().startswith('-999'):
            section = None
            continue

        if section == 'title':
            # MVA base in columns 31-36 (1-indexed) = index 30:36
            try:
                mva_base = float(stripped[30:36].strip())
            except (ValueError, IndexError):
                pass
            section = None
            continue

        if section == 'bus':
            try:
                bus_num  = int(stripped[0:4])
                name     = stripped[5:17].strip()
                bus_type = int(stripped[24:26])
                v_mag    = float(stripped[27:32])
                v_ang    = float(stripped[33:39])    # degrees
                p_load   = float(stripped[40:48]) / mva_base
                q_load   = float(stripped[49:57]) / mva_base
                p_gen    = float(stripped[59:66]) / mva_base
                q_gen    = float(stripped[67:74]) / mva_base
                g_sh     = float(stripped[106:114]) if len(stripped) > 114 else 0.0
                b_sh     = float(stripped[114:122]) if len(stripped) > 122 else 0.0
                buses.append({
                    'num': bus_num, 'name': name, 'type': bus_type,
                    'V': v_mag, 'theta': np.deg2rad(v_ang),
                    'P_load': p_load, 'Q_load': q_load,
                    'P_gen': p_gen, 'Q_gen': q_gen,
                    'G_sh': g_sh, 'B_sh': b_sh,
                })
            except (ValueError, IndexError):
                pass

        elif section == 'branch':
            try:
                from_bus = int(stripped[0:4])
                to_bus   = int(stripped[5:9])
                r        = float(stripped[19:29])
                x        = float(stripped[29:40])
                b        = float(stripped[40:50])
                tap_str  = stripped[76:82].strip() if len(stripped) > 82 else ''
                tap      = float(tap_str) if tap_str else 0.0
                if tap == 0.0:
                    tap = 1.0
                branches.append({
                    'from_bus': from_bus, 'to_bus': to_bus,
                    'R': r, 'X': x, 'B': b, 'tap': tap,
                })
            except (ValueError, IndexError):
                pass

    return buses, branches, mva_base


def parse_measurements(filename, valid_buses, valid_branch_set, max_sections=None):
    """
    Parse the measurement file. Returns list of measurement dicts.
    Each dict: {type, bus/from_bus/to_bus, value, sigma, residual}
    Types: 'Vmag', 'Vang', 'Pinj', 'Qinj', 'Pflow', 'Qflow'

    valid_buses: set of bus numbers present in the network
    valid_branch_set: set of (from, to) tuples (both orderings) present in network
    """
    measurements = []
    meas_types_single = ['Vmag', 'Vang', 'Pinj', 'Qinj']
    meas_types_branch = ['Pflow', 'Qflow', 'Pflow_aux', 'Qflow_aux']

    with open(filename, 'r') as f:
        lines = [l.strip() for l in f.readlines()]

    idx = 0
    section_count = 0  # which measurement section we're in

    while idx < len(lines):
        line = lines[idx]
        if not line:
            idx += 1
            continue

        parts = line.split()
        if not parts:
            idx += 1
            continue

        # Try to read count line (single integer)
        if len(parts) == 1:
            try:
                count = int(parts[0])
            except ValueError:
                idx += 1
                continue

            if count == 0:
                break  # end marker

            idx += 1
            mtype = None
            if section_count < 4:
                mtype = meas_types_single[section_count]
            elif section_count < 8:
                mtype = meas_types_branch[section_count - 4]

            for _ in range(count):
                if idx >= len(lines):
                    break
                dline = lines[idx].split()
                idx += 1
                if not dline:
                    continue
                try:
                    if mtype in ('Vmag', 'Vang', 'Pinj', 'Qinj'):
                        bus  = int(dline[0])
                        val  = float(dline[1])
                        sig  = float(dline[2])
                        if bus in valid_buses:
                            mt = mtype if mtype in ('Vmag', 'Vang') else mtype
                            measurements.append({
                                'type': mtype, 'bus': bus,
                                'value': val, 'sigma': sig,
                            })
                    else:
                        fb   = int(dline[0])
                        tb   = int(dline[1])
                        val  = float(dline[2])
                        sig  = float(dline[3])
                        if (fb, tb) in valid_branch_set or (tb, fb) in valid_branch_set:
                            # Map aux types back to Pflow/Qflow
                            mt = 'Pflow' if mtype == 'Pflow_aux' else (
                                 'Qflow' if mtype == 'Qflow_aux' else mtype)
                            measurements.append({
                                'type': mt, 'from_bus': fb, 'to_bus': tb,
                                'value': val, 'sigma': sig,
                            })
                except (ValueError, IndexError):
                    pass

            section_count += 1
            if max_sections is not None and section_count >= max_sections:
                break
        else:
            idx += 1

    return measurements
