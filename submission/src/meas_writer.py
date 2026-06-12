"""
Write a measurement set to disk in the IEEE-style 8-section format used by
read_me_meas.txt / measure.dat, and compute PMU branch current phasors.

Section order (each: a count line, then that many data rows; terminated by 0):
    1 Vmag    bus  value sigma resid
    2 Vang    bus  value sigma resid
    3 Pinj    bus  value sigma resid
    4 Qinj    bus  value sigma resid
    5 Pflow   from to value sigma resid
    6 Qflow   from to value sigma resid
    7 Imag    from to value sigma resid   (PMU current magnitude)
    8 Iang    from to value sigma resid   (PMU current phase)

The residual column is always 0 (it is a state-estimation output, not an input).
Files are written with CRLF line endings to match the original data files, and
round-trip exactly through src.parser.parse_measurements.
"""
import numpy as np

# (type, kind) in the exact order the parser expects.
SECTIONS = [
    ('Vmag',  'single'),
    ('Vang',  'single'),
    ('Pinj',  'single'),
    ('Qinj',  'single'),
    ('Pflow', 'branch'),
    ('Qflow', 'branch'),
    ('Imag',  'branch'),
    ('Iang',  'branch'),
]

NL = '\r\n'


def branch_current(net, V, theta, from_bus, to_bus):
    """
    Branch current phasor seen from `from_bus`, derived from the complex power
    flow:  S = V_i * conj(I)  =>  |I| = |S| / V_i,  angle(I) = theta_i - angle(S).

    Returns (I_magnitude_pu, I_angle_rad).
    """
    i = net.bus_idx[from_bus]
    P = net.calc_pflow(V, theta, from_bus, to_bus)
    Q = net.calc_qflow(V, theta, from_bus, to_bus)
    Imag = np.hypot(P, Q) / V[i]
    Iang = theta[i] - np.arctan2(Q, P)
    return Imag, Iang


def _fmt_single(m):
    return f"{int(m['bus']):3d}{float(m['value']):12.5f} {float(m['sigma']):.6f} 0"


def _fmt_branch(m):
    return (f"{int(m['from_bus']):4d}{int(m['to_bus']):4d}"
            f"{float(m['value']):12.5f} {float(m['sigma']):.6f} 0")


def write_measurement_file(path, measurements):
    """
    Write `measurements` (list of dicts with keys type/value/sigma and either
    'bus' or 'from_bus'+'to_bus') to `path` in the 8-section format.
    Measurement order within each section is preserved.
    """
    by_type = {t: [] for t, _ in SECTIONS}
    for m in measurements:
        if m['type'] in by_type:
            by_type[m['type']].append(m)

    lines = []
    for mtype, kind in SECTIONS:
        rows = by_type[mtype]
        lines.append(f"{len(rows):4d}")
        fmt = _fmt_single if kind == 'single' else _fmt_branch
        for m in rows:
            lines.append(fmt(m))
    lines.append(f"{0:4d}")          # end marker
    lines.append('')                 # trailing newline

    with open(path, 'w', newline='') as f:
        f.write(NL.join(lines))
