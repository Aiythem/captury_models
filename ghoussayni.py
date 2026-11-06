"""Ghoussayni sagittal heel/toe speed events, fixed 0.5 m/s variant.

Implementation follows the M4 definition and preprocessing reported by Bonci
et al. (2022), doi:10.3389/fbioe.2022.868928, Table 1: Ghoussayni (2004)
with the threshold increased to 0.5 m/s following Bruening & Ridge (2014).
This is not claimed to reproduce the original 2004 threshold/protocol.
Measured lab velocities are projected onto pelvis AP and lab vertical axes;
pelvis velocity is NOT subtracted. Missing data are never interpolated.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from mocap_labels import display_marker_name

METHOD = "ghoussayni_sagittal_velocity_fixed_threshold"


def _spans(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return zip(edges[::2], edges[1::2])


def _filtered(time, xyz, sos):
    """Filter each contiguous observed span; invalidate derivative endpoints."""
    result = np.full_like(xyz, np.nan, dtype=float)
    valid = np.isfinite(xyz).all(axis=0)
    # A timestamp hole splits the signal even when both endpoint samples exist.
    holes = np.flatnonzero(np.diff(time) > 1.5 * np.median(np.diff(time))) + 1
    for first, stop in _spans(valid):
        cuts = [first, *holes[(holes > first) & (holes < stop)], stop]
        for start, end in zip(cuts[:-1], cuts[1:]):
            if end - start <= 15:  # scipy's padding length for order 4
                continue
            filtered = sosfiltfilt(sos, xyz[:, start:end], axis=1)
            result[:, start + 1 : end - 1] = filtered[:, 1:-1]
    return result


def detect_ghoussayni(time, labels, points_mm, *, threshold_m_s=0.5, cutoff_hz=7.0):
    """Return an auditable event report and per-frame signals on the C3D clock.

    Marker roles: FCC=calcaneus/heel, FM2=second metatarsal/forefoot; explicit
    HEE/TOE aliases are also accepted. Duplicate candidates are rejected.
    Pelvis AP is the horizontal posterior-to-anterior midpoint direction from
    LIAS/RIAS/LIPS/RIPS. This implementation assumes the Motive Z-up C3D frame.
    Events use the first observed sample across a threshold (no subframe fit).
    """
    time = np.asarray(time, dtype=float)
    points_mm = np.asarray(points_mm, dtype=float)
    if (
        time.ndim != 1
        or len(time) < 2
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("Ghoussayni requires finite, strictly increasing times.")
    if points_mm.shape != (3, len(labels), len(time)):
        raise ValueError("Expected marker positions with shape (3, markers, frames).")
    rate = 1.0 / np.median(np.diff(time))
    if not np.isfinite(threshold_m_s) or threshold_m_s <= 0:
        raise ValueError("Ghoussayni speed threshold must be positive and finite.")
    if not np.isfinite(cutoff_hz) or not 0 < cutoff_hz < rate / 2:
        raise ValueError("Ghoussayni filter cutoff must be between zero and Nyquist.")
    dt = np.diff(time)
    if np.any((dt <= 1.5 / rate) & ~np.isclose(dt, 1 / rate, rtol=1e-4, atol=1e-9)):
        raise ValueError(
            "Ghoussayni filtering requires uniformly sampled observations."
        )
    names = [display_marker_name(str(label)).upper() for label in labels]
    sos = butter(4, cutoff_hz, fs=rate, output="sos")
    chosen = {}
    missing = np.full((3, len(time)), np.nan)

    def marker(role, aliases):
        indices = [i for i, name in enumerate(names) if name in aliases]
        chosen[role] = {
            "status": "ok" if len(indices) == 1 else "missing_or_ambiguous_marker",
            "labels": [str(labels[i]) for i in indices],
            "accepted_aliases": list(aliases),
        }
        return (
            _filtered(time, points_mm[:, indices[0]], sos)
            if len(indices) == 1
            else missing.copy()
        )

    pelvis = {
        name: marker(name, (name, alias))
        for name, alias in (
            ("LIAS", "LASI"),
            ("RIAS", "RASI"),
            ("LIPS", "LPSI"),
            ("RIPS", "RPSI"),
        )
    }
    ap = (pelvis["LIAS"] + pelvis["RIAS"] - pelvis["LIPS"] - pelvis["RIPS"]) / 2
    ap[2] = 0
    lengths = np.linalg.norm(ap, axis=0)
    ap = np.divide(ap, lengths, out=np.full_like(ap, np.nan), where=lengths > 1e-6)
    report = {
        "method": METHOD,
        "variant": (
            "fixed_threshold_0.5_m_s_literature_variant"
            if threshold_m_s == 0.5
            else "custom_fixed_threshold"
        ),
        "references": [
            "https://doi.org/10.1016/j.gaitpost.2003.10.001",
            "https://doi.org/10.3389/fbioe.2022.868928",
        ],
        "velocity_threshold_m_s": threshold_m_s,
        "filter": {
            "family": "Butterworth",
            "design_order": 4,
            "application": "forward_backward_zero_phase",
            "cutoff_hz": cutoff_hz,
        },
        "input_unit": "mm",
        "velocity_unit": "m/s",
        "vertical_axis": "+Z",
        "sagittal_plane": "per_frame_horizontal_pelvis_AP_and_lab_Z",
        "frame_adaptation": "Motive pelvis landmark AP horizontalized with laboratory Z; not the full Bonci pelvis-cluster frame",
        "scope": "overground heel-first gait; not validated for treadmill, turns or forefoot-first contact",
        "derivative": "numpy.gradient_on_filtered_lab_positions",
        "event_timing": "first_sample_across_threshold",
        "gap_policy": "filter_contiguous_spans_separately_no_interpolation_no_events_at_recovery",
        "time_step_s": float(1 / rate),
        "force_plate_validated": False,
        "marker_roles": chosen,
        "sides": {},
        "initial_contacts_s": {},
        "toe_offs_s": {},
    }
    rows = [{"time": float(t), "event_method": METHOD} for t in time]
    for side, prefix in (("left", "L"), ("right", "R")):
        heel = marker(side + "_heel", (prefix + "FCC", prefix + "HEE", prefix + "HEEL"))
        toe = marker(side + "_toe", (prefix + "FM2", prefix + "TOE"))

        def speed(xyz):
            velocity = np.gradient(xyz / 1000.0, time, axis=1)
            return np.sqrt(np.sum(velocity * ap, axis=0) ** 2 + velocity[2] ** 2)

        heel_speed, toe_speed = speed(heel), speed(toe)
        valid = np.isfinite(heel_speed) & np.isfinite(toe_speed)
        continuous = np.r_[False, np.diff(time) < 1.5 / rate]
        adjacent = valid & np.r_[False, valid[:-1]] & continuous
        below_heel, below_toe = heel_speed < threshold_m_s, toe_speed < threshold_m_s
        hc = adjacent & below_heel & ~np.r_[False, below_heel[:-1]]
        to = adjacent & ~below_toe & np.r_[False, below_toe[:-1]]
        hr = adjacent & ~below_heel & np.r_[False, below_heel[:-1]]
        tc = adjacent & below_toe & ~np.r_[False, below_toe[:-1]]
        report["initial_contacts_s"][side] = time[hc].tolist()
        report["toe_offs_s"][side] = time[to].tolist()
        report["sides"][side] = {
            "available": bool(valid.any()),
            "valid_frames": int(valid.sum()),
            "heel_contact_s": time[hc].tolist(),
            "heel_rise_s": time[hr].tolist(),
            "toe_contact_s": time[tc].tolist(),
            "toe_off_s": time[to].tolist(),
        }
        contact = False
        for i, row in enumerate(rows):
            if not valid[i] or to[i]:
                contact = False
            if hc[i]:
                contact = True
            row.update(
                {
                    f"{side}_heel_speed_m_s": float(heel_speed[i]),
                    f"{side}_toe_speed_m_s": float(toe_speed[i]),
                    f"{side}_foot_speed_mm_s": float(heel_speed[i] * 1000.0),
                    f"{side}_foot_z_mm": float(heel[2, i]),
                    f"{side}_contact": bool(contact),
                    f"{side}_contact_available": bool(valid[i]),
                    f"{side}_initial_contact": bool(hc[i]),
                    f"{side}_toe_off": bool(to[i]),
                    f"{side}_heel_rise": bool(hr[i]),
                    f"{side}_toe_contact": bool(tc[i]),
                }
            )
    report["status"] = (
        "diagnostic_only"
        if any(v["available"] for v in report["sides"].values())
        else "unavailable"
    )
    return report, rows
