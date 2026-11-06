"""Reviewable squat/jump candidates on the Motive clock, not force-plate events."""

import csv
import numpy as np
from gait_cycles import _time_array, _runs


def excursion_cycles(
    time,
    signal,
    *,
    task,
    threshold,
    peak_threshold=None,
    min_duration_s=0.1,
    max_duration_s=10,
    max_gap_s=0.05,
    start_s=None,
    end_s=None,
):
    time = _time_array(time)
    signal = np.asarray(signal, dtype=float)
    if signal.shape != time.shape:
        raise ValueError("Event signal and timestamps must have the same shape.")
    if not (
        0 < min_duration_s < max_duration_s and max_gap_s > 0 and np.isfinite(threshold)
    ):
        raise ValueError("Invalid event thresholds, duration or gap limits.")
    cycles, rejected = [], []
    finite = np.isfinite(signal)
    for index, (first, stop) in enumerate(_runs(finite & (signal > threshold)), 1):
        reason = None
        if first == 0 or stop == len(time) or not (finite[first - 1] and finite[stop]):
            rejected.append(
                {"cycle_id": f"{task}_{index:03d}", "reason": "incomplete_boundary"}
            )
            continue
        start, end = first - 1, stop
        peak = first + int(np.argmax(signal[first:stop]))
        duration = time[end] - time[start]
        if np.any(np.diff(time[start : end + 1]) > max_gap_s):
            reason = "missing_event_data"
        elif not min_duration_s <= duration <= max_duration_s:
            reason = "cycle_duration_outside_limits"
        elif peak_threshold is not None and signal[peak] < peak_threshold:
            reason = "insufficient_knee_flexion"
        elif (start_s is not None and time[start] < start_s) or (
            end_s is not None and time[end] > end_s
        ):
            reason = "outside_selected_window"
        row = dict(
            cycle_id=f"{task}_{index:03d}",
            cycle_index=index,
            side="both",
            start_s=float(time[start]),
            end_s=float(time[end]),
            duration_s=float(duration),
            peak_s=float(time[peak]),
            definition=(
                "standing_deepest_knee_flexion_standing"
                if task == "squat"
                else "marker_estimated_takeoff_to_landing"
            ),
        )
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            cycles.append(row)
    return dict(
        status="diagnostic_only" if cycles else "no_complete_cycle",
        method=(
            "bilateral_knee_geometry"
            if task == "squat"
            else "bilateral_foot_marker_height"
        ),
        task=task,
        force_plate_validated=False,
        cycles=cycles,
        rejected_cycles=rejected,
        threshold=float(threshold),
        peak_threshold=peak_threshold,
        minimum_cycle_duration_s=min_duration_s,
        maximum_cycle_duration_s=max_duration_s,
        maximum_event_gap_s=max_gap_s,
    )


def squat_cycles(runs, *, standing_deg=15, peak_deg=40, **kwargs):
    if not 0 <= standing_deg < peak_deg <= 180:
        raise ValueError(
            "Squat thresholds require 0 <= standing < peak <= 180 degrees."
        )
    run = next((r for r in runs if r is not None and r.system == "motive"), None)
    if run is None:
        raise ValueError("Squat detection requires Motive joint-centre trajectories.")
    from kinematic_conventions import load_kinematic_conventions, segment_source_names

    names = segment_source_names(
        load_kinematic_conventions(), "motive_model", getattr(run, "source_kind", "bvh")
    )
    angles = []
    for side in ("left", "right"):
        segments = [names.get(f"{side}_{part}") for part in ("thigh", "shank", "foot")]
        if any(name is None or name not in run.centres_native for name in segments):
            raise ValueError(
                f"Squat detection requires observed {side} hip, knee and ankle centres."
            )
        hip, knee, ankle = [run.centres_native[name] for name in segments]
        u, v = hip - knee, ankle - knee
        denom = np.linalg.norm(u, axis=0) * np.linalg.norm(v, axis=0)
        cos = np.divide(
            np.sum(u * v, axis=0),
            denom,
            out=np.full_like(denom, np.nan),
            where=denom > 1e-12,
        )
        angles.append(180 - np.degrees(np.arccos(np.clip(cos, -1, 1))))
    # Both knees must be observed and flexed; native Euler axes are not relabeled.
    signal = np.min(angles, axis=0)
    report = excursion_cycles(
        run.time,
        signal,
        task="squat",
        threshold=standing_deg,
        peak_threshold=peak_deg,
        **kwargs,
    )
    report["signal"] = "minimum_bilateral_geometric_knee_flexion_deg"
    return report


def jump_cycles(labels, points_mm, time, *, height_mm=30, **kwargs):
    from mocap_labels import marker_indices_by_display_label

    if not np.isfinite(height_mm) or height_mm <= 0:
        raise ValueError("Jump height threshold must be positive.")
    lookup = marker_indices_by_display_label(labels)
    names = ("LFCC", "LFM2", "RFCC", "RFM2")
    if any(len(lookup.get(name, [])) != 1 for name in names):
        raise ValueError("Jump detection requires unique LFCC/LFM2/RFCC/RFM2 markers.")
    z = np.asarray(points_mm)[2, [lookup[n][0] for n in names], :]
    if np.any(np.sum(np.isfinite(z), axis=1) < 3):
        raise ValueError("Insufficient observed foot markers for jump detection.")
    baseline = np.nanpercentile(z, 10, axis=1)
    signal = np.min(z - baseline[:, None], axis=0)
    report = excursion_cycles(time, signal, task="jump", threshold=height_mm, **kwargs)
    report.update(
        marker_names=names,
        baseline_z_mm=baseline.tolist(),
        limitation="Height crossings approximate take-off/landing; review boundaries. Requires observed ground contact.",
        signal="minimum_height_above_individual_marker_baselines_mm",
    )
    return report


def reviewed_task_cycles(
    path,
    trial,
    task,
    *,
    min_duration_s,
    max_duration_s,
    start_s=None,
    end_s=None,
    **_kwargs,
):
    cycles = []
    with open(path, newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"trial", "start_s", "end_s"} <= set(reader.fieldnames or []):
            raise ValueError(
                "Reviewed squat/jump CSV needs trial,start_s,end_s columns."
            )
        for row in reader:
            if row["trial"] != trial:
                continue
            start, end = float(row["start_s"]), float(row["end_s"])
            if (
                not np.isfinite([start, end]).all()
                or start < 0
                or not min_duration_s <= end - start <= max_duration_s
            ):
                raise ValueError("Invalid reviewed cycle boundaries or duration.")
            if (start_s is not None and start < start_s) or (
                end_s is not None and end > end_s
            ):
                continue
            if cycles and start < cycles[-1]["end_s"]:
                raise ValueError(
                    "Reviewed intervals must be ordered and non-overlapping."
                )
            i = len(cycles) + 1
            cycles.append(
                dict(
                    cycle_id=f"{task}_{i:03d}",
                    cycle_index=i,
                    side="both",
                    start_s=start,
                    end_s=end,
                    duration_s=end - start,
                )
            )
    return dict(
        task=task,
        method="reviewed_intervals_csv",
        status="diagnostic_only" if cycles else "no_complete_cycle",
        cycles=cycles,
        rejected_cycles=[],
        force_plate_validated=False,
        events_csv=str(path),
    )
