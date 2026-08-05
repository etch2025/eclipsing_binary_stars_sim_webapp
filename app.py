"""
Eclipsing Binary Stars Light Curve Simulator -- Streamlit App

Ported from the standalone script (LC_v5.py). All physics is unchanged; this file
wraps it in an interactive Streamlit UI so every input parameter is a widget instead
of a hardcoded constant.

Major Assumptions (unchanged from the original script):
- Uniform Luminosity Distribution (no limb darkening; luminosity/projected area = const)
- Both stars are perfectly spherical, with projected areas being perfect circles
- Orbit is a Keplerian ellipse defined by (P or a, e, i, omega); the longitude of the
  ascending node (Omega) is omitted since it only rotates the sky-projected geometry
  and has no effect on a single-band light curve.
- P and a are mutually dependent via Kepler's third law, given (m1 + m2): whichever one
  you choose as the "independent input" below, the other is derived automatically.
"""

import io
import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import imageio.v2 as imageio
import streamlit as st
import imageio_ffmpeg

# matplotlib's ffmpeg writer normally relies on an `ffmpeg` binary already being
# installed on the host system (e.g. via apt/packages.txt), which most Streamlit
# deployments don't have out of the box. imageio-ffmpeg ships a static ffmpeg
# binary as a pip package, so pointing matplotlib at it makes mp4 export work
# everywhere, with no system-level dependency.
plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

# --------------------------------------------------
# General Scientific Constants
AU = 1.5e11        # Astronomical Unit, m
G = 6.67e-11       # Gravitational Constant
M_Sol = 1.99e30    # Solar Mass, kg
L_Sol = 3.9e26     # Solar Luminosity, W
R_Sol = 6.96e8     # Solar Radius, m
YR = 365 * 24 * 60 * 60  # Julian year, s

st.set_page_config(page_title="Eclipsing Binary Simulator", layout="wide")


# ====================================================
# Physics (pure functions -- no globals, everything passed in explicitly so this
# is safe to call repeatedly from Streamlit with different widget values, and safe
# to cache)
# ====================================================

def solve_kepler(M, e, tol=1e-12, max_iter=200):
    """Newton-Raphson solve of M = E - e*sin(E) for eccentric anomaly E."""
    M = np.atleast_1d(np.asarray(M, dtype=float))
    E = M.copy() if e < 0.8 else np.full_like(M, np.pi)
    for _ in range(max_iter):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        dE = f / fp
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def true_anomaly(E, e):
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))


def orbit_state(t, w, e, a_Rsol, om, inc):
    """
    Relative position of star 2 w.r.t. star 1, in Solar Radii.
    Returns: x, y (sky-plane), z (line-of-sight, +z = star 2 in front of star 1),
             d (projected separation), r (true 3D separation), nu (true anomaly)
    """
    M = np.mod(w * np.asarray(t, dtype=float), 2 * np.pi)
    E = solve_kepler(M, e)
    nu = true_anomaly(E, e)
    r = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu))
    x = r * np.cos(om + nu)
    y = r * np.sin(om + nu) * np.cos(inc)
    z = r * np.sin(om + nu) * np.sin(inc)
    d = np.sqrt(x**2 + y**2)
    return x, y, z, d, r, nu


def A_c(d, r1, r2):
    """Projected area eclipsed, R_Sol^2."""
    d = np.atleast_1d(np.asarray(d, dtype=float))
    area = np.zeros_like(d)
    outside = d >= (r1 + r2)
    inside = d <= abs(r1 - r2)
    overlap = (~outside) & (~inside)
    area[inside] = np.pi * min(r1, r2)**2
    if np.any(overlap):
        dv = d[overlap]
        arg1 = np.clip((dv**2 + r1**2 - r2**2) / (2 * dv * r1), -1.0, 1.0)
        arg2 = np.clip((dv**2 + r2**2 - r1**2) / (2 * dv * r2), -1.0, 1.0)
        term1 = r1**2 * np.arccos(arg1)
        term2 = r2**2 * np.arccos(arg2)
        term3 = 0.5 * np.sqrt(np.clip((dv**2 - (r2 - r1)**2) * ((r1 + r2)**2 - dv**2), 0.0, None))
        area[overlap] = term1 + term2 - term3
    return area


def flux(d, z, r1, r2, L1, L2, A1, A2, L_total):
    """Total system luminosity given projected separation d and line-of-sight offset z."""
    d = np.atleast_1d(np.asarray(d, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))
    L = np.full_like(d, L_total)
    eclipsing = d < (r1 + r2)
    if np.any(eclipsing):
        dc = d[eclipsing]
        zc = z[eclipsing]
        Ac = A_c(dc, r1, r2)
        Lc = np.empty_like(dc)
        total = dc <= abs(r1 - r2)
        front2 = zc > 0   # star 2 in front -> occults star 1
        front1 = ~front2  # star 1 in front -> occults star 2

        m = front2 & total
        Lc[m] = L2 if r2 >= r1 else L2 + (A1 - A2) / A1 * L1
        m = front2 & ~total
        Lc[m] = L2 + (A1 - Ac[m]) / A1 * L1

        m = front1 & total
        Lc[m] = L1 if r1 >= r2 else L1 + (A2 - A1) / A2 * L2
        m = front1 & ~total
        Lc[m] = L1 + (A2 - Ac[m]) / A2 * L2

        L[eclipsing] = Lc
    return L


def get_segments(mask):
    """Indices of contiguous True runs in a boolean array."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    splits = np.where(np.diff(idx) > 1)[0]
    return np.split(idx, splits + 1)


def get_segments_circular(mask):
    """Like get_segments, but stitches a run that straddles the array boundary."""
    segs = [list(s) for s in get_segments(mask)]
    if len(segs) > 1 and mask[0] and mask[-1]:
        first, last = segs[0], segs[-1]
        segs = segs[1:-1] + [last + first]
    return segs


# ====================================================
# Full simulation -- returns everything needed for plotting + diagnostics.
# Cached on all physical inputs so dragging an unrelated widget (e.g. a color picker)
# doesn't re-run the (potentially expensive) Kepler solve.
# ====================================================

@st.cache_data(show_spinner="Solving Kepler's equation and scanning the orbit...")
def simulate(m1, r1, L1, m2, r2, L2, orbit_input, P_days_in, a_AU_in, i_deg, e, omega_deg, n_samples):
    A1 = np.pi * r1**2
    A2 = np.pi * r2**2
    L_total = L1 + L2

    # P and a are mutually dependent via Kepler's third law, given (m1+m2) -- whichever
    # one is NOT the chosen input is derived, so they can never drift out of consistency.
    if orbit_input == "P":
        P = P_days_in * 24 * 3600                        # seconds
        sma = ((P / YR)**2 * (m1 + m2))**(1 / 3) * AU     # meters, derived (Kepler III)
    else:
        sma = a_AU_in * AU                                # meters, direct input
        P = np.sqrt((sma / AU)**3 / (m1 + m2)) * YR       # seconds, derived (Kepler III)

    a_Rsol = sma / R_Sol
    w = (2 * np.pi) / P
    om = np.radians(omega_deg)
    inc = np.radians(i_deg)

    # --- Numerical scan over one full period ---
    t_arr = np.linspace(0, P, n_samples)
    x_arr, y_arr, z_arr, d_arr, r_arr, nu_arr = orbit_state(t_arr, w, e, a_Rsol, om, inc)
    L_arr = flux(d_arr, z_arr, r1, r2, L1, L2, A1, A2, L_total)

    eclipse_mask = d_arr < (r1 + r2)
    primary_mask = eclipse_mask & (z_arr > 0)
    secondary_mask = eclipse_mask & (z_arr < 0)
    primary_segs = get_segments_circular(primary_mask)
    secondary_segs = get_segments_circular(secondary_mask)
    eclipses_occur = len(primary_segs) > 0 or len(secondary_segs) > 0

    # --- Re-anchor t=0 to first contact of whichever eclipse is DEEPER ---
    def _deepest_L(segs):
        if not segs:
            return None
        return L_arr[max(segs, key=len)].min()

    L_min_primary_raw = _deepest_L(primary_segs)
    L_min_secondary_raw = _deepest_L(secondary_segs)

    if primary_segs and secondary_segs:
        idx0 = max(primary_segs, key=len)[0] if L_min_primary_raw <= L_min_secondary_raw \
            else max(secondary_segs, key=len)[0]
    elif primary_segs:
        idx0 = max(primary_segs, key=len)[0]
    elif secondary_segs:
        idx0 = max(secondary_segs, key=len)[0]
    else:
        idx0 = 0

    x_arr = np.roll(x_arr, -idx0)
    y_arr = np.roll(y_arr, -idx0)
    z_arr = np.roll(z_arr, -idx0)
    d_arr = np.roll(d_arr, -idx0)
    r_arr = np.roll(r_arr, -idx0)
    nu_arr = np.roll(nu_arr, -idx0)
    L_arr = np.roll(L_arr, -idx0)

    eclipse_mask = d_arr < (r1 + r2)
    primary_mask = eclipse_mask & (z_arr > 0)
    secondary_mask = eclipse_mask & (z_arr < 0)
    primary_segs = get_segments_circular(primary_mask)
    secondary_segs = get_segments_circular(secondary_mask)

    r_peri = a_Rsol * (1 - e)

    def eclipse_info(segs):
        if not segs:
            return None
        seg = max(segs, key=len)
        mid_idx = seg[np.argmin(L_arr[seg])]
        return {
            't_start': t_arr[seg[0]], 't_end': t_arr[seg[-1]],
            't_mid': t_arr[mid_idx], 'L_min': L_arr[mid_idx],
            'd_min': d_arr[mid_idx], 'duration': t_arr[seg[-1]] - t_arr[seg[0]],
        }

    # True anomaly / separation at each conjunction -- needed for analytic
    # eclipse-geometry diagnostics even when no eclipse occurs (matches LC_v5.py).
    nu_c_zpos = np.radians(90.0 - omega_deg)
    nu_c_zneg = np.radians(270.0 - omega_deg)
    r_c_zpos = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_c_zpos))
    r_c_zneg = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_c_zneg))

    pe = se = None
    pe_segs, se_segs = [], []

    if not eclipses_occur:
        # Keep geometric conjunction labeling so i_min / i_grazing / P_min / a_min still print
        nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
        nu_c_se, r_c_se = nu_c_zneg, r_c_zneg
    else:
        pe_z_pos = eclipse_info(primary_segs)
        se_z_neg = eclipse_info(secondary_segs)

        # "Primary" = deeper eclipse. Always keep both conjunction geometries so
        # diagnostics print for both even if only one eclipse is detected.
        if pe_z_pos and se_z_neg:
            if pe_z_pos['L_min'] <= se_z_neg['L_min']:
                pe, pe_segs = pe_z_pos, primary_segs
                se, se_segs = se_z_neg, secondary_segs
                nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
                nu_c_se, r_c_se = nu_c_zneg, r_c_zneg
            else:
                pe, pe_segs = se_z_neg, secondary_segs
                se, se_segs = pe_z_pos, primary_segs
                nu_c_pe, r_c_pe = nu_c_zneg, r_c_zneg
                nu_c_se, r_c_se = nu_c_zpos, r_c_zpos
        elif pe_z_pos:
            pe, pe_segs = pe_z_pos, primary_segs
            nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
            nu_c_se, r_c_se = nu_c_zneg, r_c_zneg
        elif se_z_neg:
            pe, pe_segs = se_z_neg, secondary_segs
            nu_c_pe, r_c_pe = nu_c_zneg, r_c_zneg
            nu_c_se, r_c_se = nu_c_zpos, r_c_zpos
        else:
            nu_c_pe, r_c_pe = nu_c_zpos, r_c_zpos
            nu_c_se, r_c_se = nu_c_zneg, r_c_zneg

    # --- Analytic eclipse-geometry metrics (generalized to e > 0) ---
    Rsum = r1 + r2
    Rdiff = r1 - r2  # signed, not abs()

    def eclipse_geometry(r_c, nu_c):
        if r_c is None:
            return None
        b_c = r_c * np.cos(inc)
        sin_i = np.sin(inc)
        denom_c = 1 + e * np.cos(nu_c)
        if sin_i > 0:
            arg = np.clip(np.sqrt(max(Rsum**2 - b_c**2, 0.0)) / (r_c * sin_i), -1.0, 1.0)
            half_angle = np.arcsin(arg)
        else:
            half_angle = 0.0
        duration = half_angle * P * (1 - e**2)**1.5 / (np.pi * denom_c**2)
        i_min = np.degrees(np.arccos(np.clip(Rsum / r_c, -1.0, 1.0))) if r_c > 0 else np.nan
        i_grazing = np.degrees(np.arccos(np.clip(Rdiff / r_c, -1.0, 1.0))) if r_c > 0 else np.nan
        return {'b': b_c, 'duration': duration, 'i_min': i_min, 'i_grazing': i_grazing}

    geo_pe = eclipse_geometry(r_c_pe, nu_c_pe)
    geo_se = eclipse_geometry(r_c_se, nu_c_se)

    # Minimum possible orbital separation / period: smallest a (at this e) keeping
    # periastron >= r1+r2, i.e. a_min = (r1+r2)/(1-e) -- matches LC_v5.py.
    a_min_Rsol = Rsum / (1 - e)
    a_min = a_min_Rsol * R_Sol  # meters
    P_min = np.sqrt((a_min / AU)**3 / (m1 + m2)) * YR

    # Maximum possible sma for an eclipse at this inclination: |b| = r_c |cos i| < r1+r2
    # ⇒ r_c < (r1+r2)/|cos i|. Convert each conjunction's r_c limit back to a via
    # r_c = a (1-e²)/(1+e cos ν_c). Take the larger of the two (at least one eclipse).
    cos_i_abs = abs(np.cos(inc))
    if cos_i_abs > 1e-15:
        r_c_max = Rsum / cos_i_abs
        a_max_pe = r_c_max * (1 + e * np.cos(nu_c_pe)) / (1 - e**2)
        a_max_se = r_c_max * (1 + e * np.cos(nu_c_se)) / (1 - e**2)
        a_max = max(a_max_pe, a_max_se) * R_Sol  # meters
        P_max = np.sqrt((a_max / AU)**3 / (m1 + m2)) * YR
    else:
        a_max = np.inf
        P_max = np.inf

    return {
        'P': P, 'sma': sma, 'a_Rsol': a_Rsol, 'w': w, 'om': om, 'inc': inc,
        't_arr': t_arr, 'x_arr': x_arr, 'y_arr': y_arr, 'z_arr': z_arr,
        'd_arr': d_arr, 'r_arr': r_arr, 'L_arr': L_arr, 'L_total': L_total,
        'eclipses_occur': eclipses_occur, 'pe': pe, 'se': se,
        'pe_segs': pe_segs, 'se_segs': se_segs,
        'r_peri': r_peri, 'geo_pe': geo_pe, 'geo_se': geo_se,
        'P_min': P_min, 'P_max': P_max, 'a_min': a_min, 'a_max': a_max,
        'A1': A1, 'A2': A2,
    }


# ====================================================
# Plotting
# ====================================================

def build_figure(res, r1, r2, i_deg, e, omega_deg, n_periods, target,
                  primary_color, secondary_color, m1, L1, m2, L2):
    P = res['P']
    a_Rsol = res['a_Rsol']
    t_arr, L_arr = res['t_arr'], res['L_arr']
    x_arr, y_arr, z_arr, d_arr = res['x_arr'], res['y_arr'], res['z_arr'], res['d_arr']
    n_samples = len(t_arr)
    pe, se, pe_segs, se_segs = res['pe'], res['se'], res['pe_segs'], res['se_segs']
    L_total = res['L_total']

    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0, wspace=0.3)
    ax_top = fig.add_subplot(gs[0, :])
    ax_p1 = fig.add_subplot(gs[1, 0], aspect='equal')
    ax_p2 = fig.add_subplot(gs[1, 1], aspect='equal')
    ax_p3 = fig.add_subplot(gs[1, 2], aspect='equal')
    ax_p4 = fig.add_subplot(gs[1, 3], aspect='equal')
    for ax in (ax_top, ax_p1, ax_p2, ax_p3, ax_p4):
        ax.grid(True, alpha=0.3 if ax is not ax_top else 1.0)

    if not res['eclipses_occur']:
        for k in range(n_periods):
            ax_top.plot(t_arr / P + k, L_arr, 'black', label='Full Flux' if k == 0 else None)
        ax_top.set_ylim([0, 1.05 * L_total])
        panel_axes = [ax_p1, ax_p2, ax_p3, ax_p4]
        panel_times = [0, P / 4, P / 2, 3 * P / 4]
        panel_titles = ["Full System"] * 4
    else:
        for k in range(n_periods):
            offset = k
            ax_top.plot(t_arr / P + offset, L_arr, color='black', lw=1.2)
            for seg in pe_segs:
                ax_top.plot(t_arr[seg] / P + offset, L_arr[seg], color='red', lw=1.5)
            for seg in se_segs:
                ax_top.plot(t_arr[seg] / P + offset, L_arr[seg], color='blue', lw=1.5)
        legend_lines = [Line2D([0], [0], color='black', label='Full Flux'),
                         Line2D([0], [0], color='red', label='Primary Eclipse'),
                         Line2D([0], [0], color='blue', label='Secondary Eclipse')]
        ax_top.legend(handles=legend_lines, loc='lower right')

        y_candidates = [L_total]
        if pe: y_candidates.append(pe['L_min'])
        if se: y_candidates.append(se['L_min'])
        y_min = min(y_candidates)
        ax_top.set_ylim([y_min - 0.05 * y_min, 1.05 * L_total])

        events = []
        if pe: events.append(('Primary Eclipse', pe['t_mid'], pe))
        if se: events.append(('Secondary Eclipse', se['t_mid'], se))
        events.sort(key=lambda ev: ev[1])

        panel_axes = [ax_p1, ax_p2, ax_p3, ax_p4]
        if len(events) == 2:
            (name_a, t_a, info_a), (name_b, t_b, info_b) = events
            gap1_mid = 0.5 * (info_a['t_end'] + info_b['t_start'])
            gap2_start = info_b['t_end']
            gap2_end = info_a['t_start'] + P
            gap2_mid = 0.5 * (gap2_start + gap2_end)
            if gap2_mid > P:
                gap2_mid -= P
            panel_times = [t_a, gap1_mid, t_b, gap2_mid]
            panel_titles = [name_a, "Full Flux", name_b, "Full Flux"]
        else:
            name_a, t_a, info_a = events[0]
            gap_mid = 0.5 * (info_a['t_end'] + (info_a['t_start'] + P))
            if gap_mid > P:
                gap_mid -= P
            panel_times = [t_a, gap_mid, gap_mid, gap_mid]
            panel_titles = [name_a, "Full Flux", "Full Flux", "Full Flux"]

        ax_top.set_title(
            f"{target}\n"
            rf"m₁ = {m1} $M_\odot$, r₁ = {r1} $R_\odot$, L₁ = {L1} $L_\odot$,    "
            rf"m₂ = {m2} $M_\odot$, r₂ = {r2} $R_\odot$, L₂ = {L2} $L_\odot$" + "\n"
            f"P = {P/(24*60**2):.4f} d, a = {res['sma']/AU:.4f} AU, e = {e:.4f}, ω = {omega_deg:.4f}°, i = {i_deg:.4f}°\n"
            + (f"Eclipse Duration: {pe['duration']/60:.4f} min, b = {pe['d_min']/r1:.4f}   " if pe or se else "No Eclipse Occurs   ")
        )

    ax_top.set_xlabel("Phase")
    ax_top.set_ylabel("Solar Luminosities")
    ax_top.set_xlim([0, n_periods])

    lim = 1.5 * a_Rsol * (1 + e)
    nu_full = np.linspace(0, 2 * np.pi, 500)
    r_full = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_full))
    orbit_x = r_full * np.cos(res['om'] + nu_full)
    orbit_y = r_full * np.sin(res['om'] + nu_full) * np.cos(res['inc'])

    dt = t_arr[1] - t_arr[0]
    for ax, t_val, title in zip(panel_axes, panel_times, panel_titles):
        idx_t = int(round((t_val % P) / dt)) % n_samples
        x_t, y_t, z_t = float(x_arr[idx_t]), float(y_arr[idx_t]), float(z_arr[idx_t])
        L_t = float(L_arr[idx_t])

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_xlabel(r"Solar Radii $R_\odot$")
        ax.set_ylabel(r"Solar Radii $R_\odot$")
        ax.set_title(rf"{title}" + "\n" + rf"t = {t_val/P:.4f}" + "\n" + rf"L = {L_t:.4f} $L_\odot$")

        ax.plot(orbit_x, orbit_y, color='black', lw=1, zorder=1)
        star1 = Circle((0, 0), r1, color=primary_color, label='m1', zorder=(2 if z_t >= 0 else 3))
        star2 = Circle((x_t, y_t), r2, color=secondary_color, label='m2', zorder=(3 if z_t >= 0 else 2))
        ax.add_patch(star1)
        ax.add_patch(star2)

    panel_axes[-1].legend(
        handles=[Line2D([0], [0], color='black', lw=1, label='Relative Orbit'),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor=primary_color, markersize=10, label='m1'),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor=secondary_color, markersize=10, label='m2')],
        loc="lower center", bbox_to_anchor=(-1.5, -0.35), ncol=3
    )
    return fig


def build_animation(res, r1, r2, i_deg, e, omega_deg, n_periods, target,
                     primary_color, secondary_color, m1, L1, m2, L2,
                     n_frames_per_period=200, fps=30, progress=None):
    """Port of LC_v5_animation.py: side-by-side orbit + light-curve animation,
    saved as mp4 (and a gif for inline preview). Uses the same absolute Keplerian
    phase convention as that script (not the eclipse-reanchored phase of the
    static figure). Defaults match LC_v5_animation: 200 frames/period @ 30 fps.

    `progress` is an optional st.progress handle updated during frame render / encode.
    Caching is handled at the call site via st.session_state (so the bar can update)."""
    P, a_Rsol, om, inc, w = res['P'], res['a_Rsol'], res['om'], res['inc'], res['w']
    A1, A2, L_total = res['A1'], res['A2'], res['L_total']
    pe, se = res['pe'], res['se']

    if progress is not None:
        progress.progress(0.0, text="Computing animation orbit states...")

    # Background light-curve trace (absolute Keplerian phase, like LC_v5_animation).
    # Cap samples so Streamlit stays responsive; the static figure already has the
    # high-resolution scan from simulate().
    n_bg = min(max(len(res['t_arr']), 10_000), 200_000)
    t_bg = np.linspace(0, n_periods * P, n_bg * n_periods)
    _x, _y, z_bg, d_bg, _r, _nu = orbit_state(t_bg, w, e, a_Rsol, om, inc)
    L_bg = flux(d_bg, z_bg, r1, r2, L1, L2, A1, A2, L_total)

    n_frames = int(n_frames_per_period * n_periods)
    t_frames = np.linspace(0, n_periods * P, n_frames, endpoint=False)
    x_f, y_f, z_f, d_f, r_f, nu_f = orbit_state(t_frames, w, e, a_Rsol, om, inc)
    L_f = flux(d_f, z_f, r1, r2, L1, L2, A1, A2, L_total)

    if progress is not None:
        progress.progress(0.05, text="Setting up animation figure...")

    fig, (ax_orbit, ax_lc) = plt.subplots(1, 2, figsize=(13, 6))

    title_str = (
        f"{target}\n"
        rf"m₁ = {m1} $M_\odot$, r₁ = {r1} $R_\odot$, L₁ = {L1} $L_\odot$,    "
        rf"m₂ = {m2} $M_\odot$, r₂ = {r2} $R_\odot$, L₂ = {L2} $L_\odot$" + "\n"
        f"P = {P/(24*60**2):.4f} d, a = {res['sma']/AU:.4f} AU, e = {e:.4f}, "
        f"ω = {omega_deg:.4f}°, i = {i_deg:.4f}°\n"
    )
    if pe is not None or se is not None:
        parts = []
        if pe is not None:
            parts.append(rf"Primary Eclipse L$_{{min}}$ = {pe['L_min']:.4f} $L_\odot$")
        if se is not None:
            parts.append(rf"Secondary Eclipse L$_{{min}}$ = {se['L_min']:.4f} $L_\odot$")
        title_str += "    ".join(parts) + "\n"
        if pe is not None:
            title_str += rf"Eclipse Duration = {pe['duration']/60:.4f} min, b = {pe['d_min']/r1:.4f}"
    else:
        title_str += "No Eclipse Occurs"

    fig.suptitle(title_str)
    fig.subplots_adjust(top=0.75, wspace=0.3)

    lim = 1.5 * a_Rsol * (1 + e)
    ax_orbit.set_xlim(-lim, lim)
    ax_orbit.set_ylim(-lim, lim)
    ax_orbit.set_aspect('equal')
    ax_orbit.set_xlabel(r"Solar Radii $R_\odot$")
    ax_orbit.set_ylabel(r"Solar Radii $R_\odot$")
    ax_orbit.grid(True, alpha=0.3)

    nu_full = np.linspace(0, 2 * np.pi, 500)
    r_full = a_Rsol * (1 - e**2) / (1 + e * np.cos(nu_full))
    orbit_x = r_full * np.cos(om + nu_full)
    orbit_y = r_full * np.sin(om + nu_full) * np.cos(inc)
    ax_orbit.plot(orbit_x, orbit_y, color='black', lw=1, zorder=1)

    star1_patch = Circle((0, 0), r1, color=primary_color, zorder=2)
    star2_patch = Circle((x_f[0], y_f[0]), r2, color=secondary_color, zorder=3)
    ax_orbit.add_patch(star1_patch)
    ax_orbit.add_patch(star2_patch)
    ax_orbit.legend(
        handles=[Line2D([0], [0], marker='o', color='w', markerfacecolor=primary_color, markersize=10, label='m1'),
                 Line2D([0], [0], marker='o', color='w', markerfacecolor=secondary_color, markersize=10, label='m2')],
        loc='upper right'
    )
    time_text = ax_orbit.text(0.02, 0.02, '', transform=ax_orbit.transAxes)

    ax_lc.plot(t_bg / P, L_bg, color='black', lw=1)
    ax_lc.set_xlim(0, n_periods)
    y_pad = 0.05 * L_total
    ax_lc.set_ylim(min(L_bg) - y_pad, L_total + y_pad)
    ax_lc.set_xlabel("Phase")
    ax_lc.set_ylabel("Solar Luminosities")
    ax_lc.grid(True, alpha=0.3)

    marker, = ax_lc.plot([], [], 'o', color='red', ms=8, zorder=3)
    vline = ax_lc.axvline(0, color='gray', lw=1, ls='--')

    def update(frame):
        x_t, y_t, z_t, L_t = x_f[frame], y_f[frame], z_f[frame], L_f[frame]
        phase = t_frames[frame] / P
        star2_patch.center = (x_t, y_t)
        # whichever star is nearer the observer (z >= 0) draws on top -- same as LC_v5_animation
        star1_patch.set_zorder(2 if z_t >= 0 else 4)
        star2_patch.set_zorder(4 if z_t >= 0 else 2)
        marker.set_data([phase], [L_t])
        vline.set_xdata([phase, phase])
        time_text.set_text(f"phase = {phase:.3f}\nL = {L_t:.3f} $L_\\odot$")

    # Render frame-by-frame once, then encode two outputs from the same frames:
    # an .mp4 (via imageio's bundled ffmpeg binary -- no system dependency) for
    # the download button, and a .gif for inline display (st.video doesn't
    # reliably autoplay/loop everywhere the way an inline gif does).
    #
    # fig.canvas isn't guaranteed to be an Agg canvas (it depends on whatever
    # backend Matplotlib picked for this environment), and buffer_rgba() is an
    # Agg-only method. savefig() sidesteps this by spinning up its own Agg
    # renderer internally regardless of the active backend -- do the same here
    # by attaching an explicit FigureCanvasAgg instead of using fig.canvas directly.
    agg_canvas = FigureCanvasAgg(fig)

    tmp_dir = tempfile.mkdtemp()
    mp4_path = os.path.join(tmp_dir, "anim.mp4")
    try:
        frames = []
        writer = imageio.get_writer(mp4_path, fps=fps, codec='libx264', quality=8)
        # Throttle UI updates — st.progress on every frame is much slower than rendering.
        progress_step = max(1, n_frames // 50)
        for frame in range(n_frames):
            update(frame)
            agg_canvas.draw()
            rgb = np.array(agg_canvas.buffer_rgba())[..., :3]  # np.array() copies; np.asarray() would alias the renderer's buffer
            frames.append(rgb)
            writer.append_data(rgb)
            if progress is not None and (frame % progress_step == 0 or frame + 1 == n_frames):
                # Reserve the last ~10% of the bar for GIF encoding
                frac = 0.05 + 0.85 * (frame + 1) / n_frames
                progress.progress(
                    frac,
                    text=f"Rendering animation: frame {frame + 1}/{n_frames}",
                )
        writer.close()
        with open(mp4_path, "rb") as fh:
            video_bytes = fh.read()

        if progress is not None:
            progress.progress(0.95, text="Encoding GIF preview...")
        gif_buf = io.BytesIO()
        imageio.mimsave(gif_buf, frames, format="GIF", fps=fps, loop=0)
        gif_bytes = gif_buf.getvalue()
        if progress is not None:
            progress.progress(1.0, text="Animation ready")
    finally:
        plt.close(fig)
        for fname in os.listdir(tmp_dir):
            os.remove(os.path.join(tmp_dir, fname))
        os.rmdir(tmp_dir)

    return video_bytes, "video/mp4", "mp4", gif_bytes, n_frames_per_period, fps


def build_diagnostics_text(res, m1, m2, r1, e, omega_deg):
    """Build the diagnostics block as a list of (label, latex) pairs, each rendered
    with st.latex so the numbers/units show up as typeset math rather than plain text."""
    P, sma, a_Rsol, r_peri = res['P'], res['sma'], res['a_Rsol'], res['r_peri']
    pe, se = res['pe'], res['se']
    geo_pe, geo_se = res['geo_pe'], res['geo_se']
    P_min, P_max = res['P_min'], res['P_max']
    a_min, a_max = res['a_min'], res['a_max']

    lines = []
    lines.append(("Orbital Period", rf"P = {P/(24*60*60):.4f}\ \text{{days}}"))
    lines.append(("Semi-major Axis", rf"a = {sma/AU:.4f}\ \text{{AU}} \quad "
                                      rf"(\text{{periastron: }} {r_peri:.4f}\ R_\odot,\ "
                                      rf"\text{{apastron: }} {a_Rsol*(1+e):.4f}\ R_\odot)"))
    lines.append(("Eccentricity & Argument of Periastron",
                   rf"e = {e:.4f} \qquad \omega = {omega_deg:.4f}^\circ"))

    if res['eclipses_occur']:
        if pe:
            lines.append(("Primary Eclipse",
                           rf"\text{{duration}} = {pe['duration']/60:.4f}\ \text{{min}}, \quad "
                           rf"d_{{\min}} = {pe['d_min']:.4f}\ R_\odot, \quad "
                           rf"L_{{\min}} = {pe['L_min']:.4f}\ L_\odot"))
        if se:
            lines.append(("Secondary Eclipse",
                           rf"\text{{duration}} = {se['duration']/60:.4f}\ \text{{min}}, \quad "
                           rf"d_{{\min}} = {se['d_min']:.4f}\ R_\odot, \quad "
                           rf"L_{{\min}} = {se['L_min']:.4f}\ L_\odot"))
    else:
        lines.append(("Eclipses", r"\text{No eclipses occur for this geometry.}"))

    # Geometric bounds always print (even with no eclipse), matching LC_v5.py
    if geo_pe:
        lines.append(("Primary Transit Duration", rf"{geo_pe['duration']/60:.4f}\ \text{{minutes}}"))
        lines.append(("Primary Impact Parameter",
                       rf"b = {geo_pe['b']:.4f}\ R_\odot \quad b/r_1 = {geo_pe['b']/r1:.4f}"))
        lines.append(("Primary Minimum Inclination for Eclipse",
                       rf"[{geo_pe['i_min']:.4f}^\circ, {180-geo_pe['i_min']:.4f}^\circ]"))
        lines.append(("Primary Minimum Grazing Eclipse Inclination",
                       rf"[{geo_pe['i_grazing']:.4f}^\circ, {180-geo_pe['i_grazing']:.4f}^\circ]"))
    if geo_se:
        lines.append(("Secondary Transit Duration", rf"{geo_se['duration']/60:.4f}\ \text{{minutes}}"))
        lines.append(("Secondary Impact Parameter",
                       rf"b = {geo_se['b']:.4f}\ R_\odot \quad b/r_1 = {geo_se['b']/r1:.4f}"))
        lines.append(("Secondary Minimum Inclination for Eclipse",
                       rf"[{geo_se['i_min']:.4f}^\circ, {180-geo_se['i_min']:.4f}^\circ]"))
        lines.append(("Secondary Minimum Grazing Eclipse Inclination",
                       rf"[{geo_se['i_grazing']:.4f}^\circ, {180-geo_se['i_grazing']:.4f}^\circ]"))

    # [a_min, a_max): min orbital separation → max sma that still allows an eclipse at this i
    if np.isfinite(a_max):
        lines.append(("Possible Semi-Major Axis for Eclipse",
                       rf"{a_min/AU:.4f}\ \text{{AU}} \le a < {a_max/AU:.4f}\ \text{{AU}}"))
        lines.append(("Possible Orbital Period for Eclipse",
                       rf"{P_min/(24*60*60):.4f}\ \text{{days}} \le P < {P_max/(24*60*60):.4f}\ \text{{days}}"))
    else:
        lines.append(("Possible Semi-Major Axis for Eclipse",
                       rf"{a_min/AU:.4f}\ \text{{AU}} \le a < \infty\ \text{{AU}} \quad (i \approx 90^\circ)"))
        lines.append(("Possible Orbital Period for Eclipse",
                       rf"{P_min/(24*60*60):.4f}\ \text{{days}} \le P < \infty\ \text{{days}} "
                       rf"\quad (i \approx 90^\circ)"))

    return lines


# ====================================================
# Sidebar UI -- wrapped in a form so changing widgets does NOT rerun the
# script (and therefore does not interrupt / restart generation). Submit with
# the button or by pressing Enter in a text/number field.
# ====================================================

with st.sidebar:
    st.header("Input Parameters")
    with st.form("generate_form", clear_on_submit=False):
        target = st.text_input("Target Name", "Algol AB (Beta Persei)")

        st.subheader("Primary Star (Star 1)")
        m1 = st.number_input("Mass m₁ (M☉)", 0.0001, 300.0000, 3.17, 0.0001, format="%.4f")
        r1 = st.number_input("Radius r₁ (R☉)", 0.0001, 300.0000, 2.73, 0.0001, format="%.4f")
        L1 = st.number_input("Luminosity L₁ (L☉)", 0.0001, 1_000_000.0, 182.0, 0.0001, format="%.4f")
        primary_color = st.color_picker("Color 1", "#00BFFF")

        st.subheader("Secondary Star (Star 2)")
        m2 = st.number_input("Mass m₂ (M☉)", 0.0001, 200.0000, 0.7, 0.0001, format="%.4f")
        r2 = st.number_input("Radius r₂ (R☉)", 0.0001, 200.0000, 3.48, 0.0001, format="%.4f")
        L2 = st.number_input("Luminosity L₂ (L☉)", 0.0001, 1_000_000.0, 6.92, 0.0001, format="%.4f")
        secondary_color = st.color_picker("Color 2", "#FFA500")

        st.subheader("Orbital Elements")
        orbit_choice = st.radio(
            "Independent input (the other is derived via Kepler's third law)",
            ["Period (P)", "Semi-Major Axis (a)"],
        )
        orbit_input = "P" if orbit_choice == "Period (P)" else "a"
        # Both shown inside the form so switching the radio doesn't require a
        # pre-submit rerun to reveal the other field.
        P_days_in = st.number_input("Period P (days)", 0.0001, 100_000.0, 2.8673, 0.0001, format="%.4f")
        a_AU_in = st.number_input("Semi-Major Axis a (AU)", 0.0001, 1000.0, 0.0620, 0.0001, format="%.4f")
        st.caption("Only the selected independent input above is used; the other is derived.")

        i_deg = st.number_input("Inclination i (deg)", 0.0, 180.0, 98.7, 0.0001, format="%.4f")
        e = st.number_input("Eccentricity e", 0.0, 0.99, 0.0, 0.0001, format="%.4f")
        omega_deg = st.number_input("Argument of Periastron ω (deg)", 0.0, 360.0, 0.0, 0.0001, format="%.4f")

        st.subheader("Simulation Settings")
        n_samples = st.select_slider(
            "Resolution (samples/period)",
            options=[10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000],
            value=200_000,
        )
        n_periods = st.number_input("Periods to display", 1, 10, 1)

        st.subheader("Animation Settings")
        n_frames_per_period = st.number_input("Frames per period", 10, 1000, 200, 10)
        fps = st.number_input("FPS", 1, 60, 30, 1)

        submitted = st.form_submit_button(
            "Generate light curve", type="primary", use_container_width=True
        )


# ====================================================
# Run + Render (only on explicit Generate / Enter)
# ====================================================

st.title("Eclipsing Binary Star Light Curve Simulator")
st.caption("Powered by Matplotlib and NumPy.")
st.markdown("[Locally-Run Version](https://github.com/etch2025/eclipsing_binary_stars_sim)")
st.subheader("Created by [Ethan Chen](https://www.chenastronomy.com/)", divider="gray")

if submitted:
    res = simulate(
        m1, r1, L1, m2, r2, L2, orbit_input, P_days_in, a_AU_in, i_deg, e, omega_deg, n_samples
    )

    img_progress = st.progress(0, text="Generating light curve figure...")
    fig = build_figure(
        res, r1, r2, i_deg, e, omega_deg, n_periods, target,
        primary_color, secondary_color, m1, L1, m2, L2,
    )
    img_progress.progress(0.55, text="Exporting PNG (dpi=500)...")
    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=500, bbox_inches="tight")
    png_buf.seek(0)
    png_bytes = png_buf.getvalue()
    plt.close(fig)
    img_progress.progress(1.0, text="Figure ready")
    img_progress.empty()

    download_name = f"{target}_{(res['P']/86400):.3f}d_{res['sma']/AU:.3f}AU_{e:.3f}.png"

    anim_progress = st.progress(0, text="Rendering animation...")
    video_bytes, video_mime, video_ext, gif_bytes, n_frames_anim, fps_anim = build_animation(
        res, r1, r2, i_deg, e, omega_deg, n_periods, target,
        primary_color, secondary_color, m1, L1, m2, L2,
        n_frames_per_period=int(n_frames_per_period), fps=int(fps),
        progress=anim_progress,
    )
    anim_progress.empty()

    # Filename matches LC_v5_animation.py's OUTPUT_FILE convention.
    anim_download_name = (
        f"{target}_{(res['P']/86400):.3f}d_{res['sma']/AU:.3f}AU_"
        f"{n_frames_anim}_{e}_{n_frames_anim}frames_{fps_anim}.{video_ext}"
    )

    st.session_state["run"] = {
        "eclipses_occur": res["eclipses_occur"],
        "png_bytes": png_bytes,
        "download_name": download_name,
        "gif_bytes": gif_bytes,
        "video_bytes": video_bytes,
        "video_mime": video_mime,
        "video_ext": video_ext,
        "anim_download_name": anim_download_name,
        "diagnostics": build_diagnostics_text(res, m1, m2, r1, e, omega_deg),
    }

if "run" in st.session_state:
    run = st.session_state["run"]

    if not run["eclipses_occur"]:
        st.warning(
            "No eclipse occurs for this geometry — try increasing inclination toward 90°, "
            "reducing the semi-major axis, or increasing the stellar radii."
        )

    st.image(run["png_bytes"])

    st.download_button(
        label="Download light curve (PNG)",
        data=run["png_bytes"],
        file_name=run["download_name"],
        mime="image/png",
    )

    st.image(run["gif_bytes"])

    st.download_button(
        label=f"Download orbit animation ({run['video_ext'].upper()})",
        data=run["video_bytes"],
        file_name=run["anim_download_name"],
        mime=run["video_mime"],
    )

    st.subheader("Diagnostics")
    for label, tex in run["diagnostics"]:
        st.markdown(f"**{label}**")
        st.latex(tex)
else:
    st.info(
        "Adjust parameters in the sidebar, then click **Generate light curve** "
        "(or press Enter in a text/number field)."
    )

with st.expander("About this simulator"):
    st.markdown(
        """
        - **Uniform surface brightness** is assumed (no limb darkening) -- flux is proportional to
          uncovered projected area.
        - Both stars are treated as perfect spheres with circular projected disks.
        - The orbit is a full Keplerian ellipse `(P or a, e, i, ω)`; Kepler's equation is solved
          numerically at every sampled timestep (exact for any eccentricity, not a small-angle
          approximation).
        - **Primary Eclipse** is defined by *depth* (whichever conjunction is dimmer), not by which
          star happens to be in front -- these can disagree if the star in front is the brighter one.
        - `P` and `a` (semi-major axis) are linked via Kepler's third law using the current masses:
          pick which one is your independent input in the sidebar, and the other updates automatically.
        - Generation only runs when you click **Generate light curve** (or press Enter) -- changing
          sidebar settings alone will not restart a render.
        """
    )