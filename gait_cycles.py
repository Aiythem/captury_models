"""Full same-foot gait cycles and source-specific lower-limb CSV exports.

Automatic contacts are kinematic estimates, not force-plate events. Native DoF
axes are preserved; naming a segment does not establish anatomical equivalence.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from kinematic_conventions import load_kinematic_conventions, segment_source_names

BASE_FIELDS = (
    "trial",
    "participant",
    "source",
    "side",
    "joint",
    "q_name",
    "axis",
    "coordinate_kind",
    "unit",
    "coordinate_semantics",
)
RAW_FIELDS = (*BASE_FIELDS, "time_s", "value")
CYCLE_FIELDS = (
    *BASE_FIELDS,
    "cycle_id",
    "cycle_index",
    "start_s",
    "end_s",
    "duration_s",
    "toe_off_s",
    "toe_off_percent",
    "peak_s",
    "peak_percent",
    "phase_percent",
    "time_s",
    "value",
    "normalization_status",
    "event_method",
)
METRIC_FIELDS = (
    *BASE_FIELDS,
    "cycle_id",
    "cycle_index",
    "start_s",
    "end_s",
    "duration_s",
    "mean_value",
    "rom_value",
    "normalization_status",
    "event_method",
)


def _time_array(values):
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Times must be finite, strictly increasing, and unique.")
    return times


def _runs(mask):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(edges[::2], edges[1::2]))


def _cycle_report(
    events,
    *,
    method,
    min_duration_s,
    max_duration_s,
    start_s,
    end_s,
    valid_interval=None,
):
    if not (
        np.isfinite(min_duration_s)
        and np.isfinite(max_duration_s)
        and 0 < min_duration_s < max_duration_s
    ):
        raise ValueError("Gait duration limits must satisfy 0 < minimum < maximum.")
    cycles, rejected = [], []
    for side in ("left", "right"):
        onsets = events.get(side, [])
        for index, (start, end) in enumerate(zip(onsets[:-1], onsets[1:]), 1):
            duration = end - start
            row = {
                "side": side,
                "cycle_id": f"{side}_{index:03d}",
                "cycle_index": index,
                "start_s": float(start),
                "end_s": float(end),
                "duration_s": float(duration),
                "definition": "same_foot_initial_contact_to_next_initial_contact",
            }
            reason = None
            if (
                start_s is not None
                and start < start_s
                or end_s is not None
                and end > end_s
            ):
                reason = "outside_selected_window"
            elif not min_duration_s <= duration <= max_duration_s:
                reason = "cycle_duration_outside_limits"
            elif valid_interval is not None and not valid_interval(side, start, end):
                reason = "missing_contact_data"
            if reason:
                rejected.append({**row, "reason": reason})
            else:
                cycles.append(row)
    return {
        "status": "diagnostic_only" if cycles else "no_complete_cycle",
        "method": method,
        "force_plate_validated": False,
        "cycles": cycles,
        "rejected_cycles": rejected,
        "initial_contacts_s": events,
        "minimum_cycle_duration_s": min_duration_s,
        "maximum_cycle_duration_s": max_duration_s,
    }


def cycles_from_ghoussayni_rows(
    rows, *, min_duration_s=0.4, max_duration_s=2.5, start_s=None, end_s=None
):
    """Keep complete observed heel-contact cycles with one intervening toe-off."""
    time = _time_array([r["time"] for r in rows])
    events = {
        side: [r["time"] for r in rows if r.get(f"{side}_initial_contact")]
        for side in ("left", "right")
    }
    toe_offs = {
        side: [r["time"] for r in rows if r.get(f"{side}_toe_off")]
        for side in ("left", "right")
    }

    def valid_interval(side, first, last):
        indices = np.flatnonzero((time >= first) & (time <= last))
        return all(
            rows[i].get(f"{side}_contact_available", False) for i in indices
        ) and (
            len(indices) < 2
            or np.all(np.diff(time[indices]) <= 1.5 * np.median(np.diff(time)))
        )

    report = _cycle_report(
        events,
        method=rows[0]["event_method"],
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        start_s=start_s,
        end_s=end_s,
        valid_interval=valid_interval,
    )
    accepted = []
    for cycle in report["cycles"]:
        offs = [
            t for t in toe_offs[cycle["side"]] if cycle["start_s"] < t < cycle["end_s"]
        ]
        if len(offs) != 1:
            report["rejected_cycles"].append(
                {**cycle, "reason": "missing_or_multiple_toe_offs"}
            )
            continue
        accepted.append(
            {
                **cycle,
                "toe_off_s": offs[0],
                "toe_off_percent": 100
                * (offs[0] - cycle["start_s"])
                / cycle["duration_s"],
            }
        )
    report["cycles"] = accepted
    report["status"] = "diagnostic_only" if accepted else "no_complete_cycle"
    report["toe_offs_s"] = toe_offs
    report["event_sequence_policy"] = (
        "one_observed_toe_off_between_successive_heel_contacts"
    )
    return report


def cycles_from_contacts(
    rows,
    *,
    min_duration_s=0.4,
    max_duration_s=2.5,
    start_s=None,
    end_s=None,
    minimum_contact_s=0.08,
    merge_gap_s=0.1,
):
    """Use Ghoussayni event flags, or debounce legacy contact-only rows."""
    if rows and rows[0].get("event_method", "").startswith("ghoussayni"):
        return cycles_from_ghoussayni_rows(
            rows,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            start_s=start_s,
            end_s=end_s,
        )
    if not rows:
        return _cycle_report(
            {},
            method="estimated_foot_marker_contacts",
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
            start_s=start_s,
            end_s=end_s,
        )
    times = _time_array([row["time"] for row in rows])
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    events, validity = {}, {}
    for side in ("left", "right"):
        valid = np.array(
            [
                bool(row.get(f"{side}_contact_available", False))
                and np.isfinite(float(row.get(f"{side}_foot_z_mm", np.nan)))
                and np.isfinite(float(row.get(f"{side}_foot_speed_mm_s", np.nan)))
                for row in rows
            ]
        )
        validity[side] = valid
        contacts = (
            np.array([bool(row.get(f"{side}_contact", False)) for row in rows]) & valid
        )
        # Bridge brief detector flicker only where the marker observations are valid.
        for first, stop in _runs(~contacts):
            if (
                first > 0
                and stop < len(times)
                and valid[first:stop].all()
                and times[stop] - times[first] <= merge_gap_s
            ):
                contacts[first:stop] = True
        for first, stop in _runs(contacts):
            if times[stop - 1] - times[first] + dt < minimum_contact_s:
                contacts[first:stop] = False
        events[side] = [
            float(times[first])
            for first, stop in _runs(contacts)
            if first > 0
            and valid[first - 1]
            and times[first] - times[first - 1] <= 2.5 * dt
        ]

    def interval_valid(side, start, end):
        mask = (times >= start) & (times <= end)
        return validity[side][mask].all() and np.all(np.diff(times[mask]) <= 2.5 * dt)

    report = _cycle_report(
        events,
        method="estimated_foot_marker_contacts",
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        start_s=start_s,
        end_s=end_s,
        valid_interval=interval_valid,
    )
    report.update({"minimum_contact_s": minimum_contact_s, "merge_gap_s": merge_gap_s})
    return report


def cycles_from_event_csv(
    path, trial, *, min_duration_s=0.4, max_duration_s=2.5, start_s=None, end_s=None
):
    """Read initial contacts in Motive elapsed seconds: [trial,]side,time_s."""
    events = {"left": [], "right": []}
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"side", "time_s"}.issubset(reader.fieldnames or []):
            raise ValueError(
                "Contact CSV requires side,time_s columns; times are Motive elapsed seconds."
            )
        for row in reader:
            side = row["side"].strip().lower()
            time = float(row["time_s"])
            trial_name = row.get("trial", trial).strip()
            if side not in events or not np.isfinite(time) or time < 0:
                raise ValueError(
                    "Contacts require left/right side and finite nonnegative time_s."
                )
            key = (trial_name, side, time)
            if key in seen:
                raise ValueError(f"Duplicate initial contact: {key}")
            seen.add(key)
            if trial_name == trial:
                events[side].append(time)
    events = {side: sorted(values) for side, values in events.items()}
    report = _cycle_report(
        events,
        method="supplied_initial_contacts",
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
        start_s=start_s,
        end_s=end_s,
    )
    report["events_csv"] = str(path)
    return report


def lower_limb_channels(runs):
    """Extract every available model coordinate for hip, knee and ankle segments."""
    registry = load_kinematic_conventions()
    channels = []
    for run in runs:
        source_id = {
            "motive": "motive_model",
            "captury": "captury_model",
            "biobuddy": "biobuddy_motive57",
        }[run.system]
        names = segment_source_names(registry, source_id, run.source_kind)
        for side in ("left", "right"):
            for joint, segment in (
                ("hip", "thigh"),
                ("knee", "shank"),
                ("ankle", "foot"),
            ):
                segment_name = names.get(f"{side}_{segment}")
                if not segment_name:
                    continue
                for index, name in enumerate(run.q_names):
                    if not name.startswith(segment_name + "_"):
                        continue
                    suffix = name[len(segment_name) + 1 :]
                    unit = run.q_units[index]
                    rotation = unit == "rad"
                    translation = any(
                        word in suffix.lower() for word in ("trans", "position")
                    )
                    if not rotation and not translation:
                        raise ValueError(
                            f"Unknown coordinate units: {run.system}/{name}: {unit}"
                        )
                    axis_match = re.search(r"([XYZ])$|^([XYZ])", suffix, re.I)
                    axis = (
                        next((g for g in axis_match.groups() if g), "")
                        if axis_match
                        else ""
                    )
                    values = np.asarray(run.q[index], dtype=float)
                    values = (
                        np.degrees(values)
                        if rotation
                        else values * run.unit_scale_to_m * 1000.0
                    )
                    channels.append(
                        {
                            "source": run.system,
                            "side": side,
                            "joint": joint,
                            "q_name": name,
                            "axis": axis.upper(),
                            "coordinate_kind": (
                                "rotation" if rotation else "translation"
                            ),
                            "unit": "deg" if rotation else "mm",
                            "coordinate_semantics": "native_model_dof",
                            "time": np.asarray(run.time, dtype=float),
                            "values": values,
                        }
                    )
    return channels


def normalize_cycle(
    time, values, cycle, *, n_points=101, max_gap_s=0.05, angular=False
):
    """Resample one whole cycle, with no extrapolation or interpolation over long gaps."""
    if n_points < 2 or not np.isfinite(max_gap_s) or max_gap_s <= 0:
        raise ValueError(
            "Normalization needs at least two points and a positive gap limit."
        )
    times = _time_array(time)
    values = np.asarray(values, dtype=float)
    if values.shape != times.shape:
        raise ValueError("Time and coordinate arrays must have matching shapes.")
    missing = np.full(n_points, np.nan)
    start, end = cycle["start_s"], cycle["end_s"]
    finite = np.isfinite(values)
    ft = times[finite]
    if len(ft) < 2 or start < ft[0] - 1e-9 or end > ft[-1] + 1e-9:
        return missing, "outside_source_coverage"
    gaps = np.diff(ft)
    if np.any((gaps > max_gap_s + 1e-9) & (ft[:-1] < end) & (ft[1:] > start)):
        return missing, "gap_exceeds_limit"
    fy = values[finite].copy()
    if angular:
        # Preserve a continuous branch within each observed span; do not join long gaps.
        boundaries = np.r_[0, np.flatnonzero(gaps > max_gap_s) + 1, len(ft)]
        for first, stop in zip(boundaries[:-1], boundaries[1:]):
            fy[first:stop] = np.degrees(np.unwrap(np.radians(fy[first:stop])))
    target = np.linspace(start, end, n_points)
    return np.interp(target, ft, fy), "ok"


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_gait_results(
    trial_dir,
    trial,
    runs,
    contact_rows,
    *,
    participant="",
    enabled=False,
    is_static=False,
    events_csv=None,
    event_detection=None,
    start_s=None,
    end_s=None,
    n_points=101,
    min_duration_s=0.4,
    max_duration_s=2.5,
    max_gap_s=0.05,
    write_npz,
    task="gait",
    cycle_report=None,
    export_csv=False,
):
    """Write raw CSVs plus complete normalized cycles; unavailable curves retain reasons."""
    trial_dir = Path(trial_dir)
    channels = lower_limb_channels(runs)
    raw_rows, normalized, metrics = [], [], []
    for channel in channels:
        meta = {k: v for k, v in channel.items() if k not in {"time", "values"}}
        meta.update({"trial": trial, "participant": participant or ""})
        raw_rows.extend(
            {**meta, "time_s": float(t), "value": float(v)}
            for t, v in zip(channel["time"], channel["values"])
        )
    if enabled and not is_static:
        kwargs = dict(
            start_s=start_s,
            end_s=end_s,
            min_duration_s=min_duration_s,
            max_duration_s=max_duration_s,
        )
        report = (
            cycle_report
            if cycle_report is not None
            else (
                cycles_from_event_csv(events_csv, trial, **kwargs)
                if events_csv
                else cycles_from_contacts(contact_rows, **kwargs)
            )
        )
        if event_detection and not events_csv:
            report["event_detection"] = event_detection
    else:
        report = {
            "status": "static_trial" if is_static else "disabled",
            "cycles": [],
            "rejected_cycles": [],
            "method": "none",
        }
    for channel in channels:
        meta = {k: v for k, v in channel.items() if k not in {"time", "values"}}
        meta.update({"trial": trial, "participant": participant or ""})
        for cycle in report["cycles"]:
            if cycle["side"] not in {channel["side"], "both"}:
                continue
            values, status = normalize_cycle(
                channel["time"],
                channel["values"],
                cycle,
                n_points=n_points,
                max_gap_s=max_gap_s,
                angular=channel["unit"] == "deg",
            )
            identity = {
                **cycle,
                **meta,
                "normalization_status": status,
                "event_method": report["method"],
                "peak_percent": (
                    100 * (cycle["peak_s"] - cycle["start_s"]) / cycle["duration_s"]
                    if "peak_s" in cycle
                    else np.nan
                ),
            }
            metrics.append(
                {
                    **identity,
                    "mean_value": float(np.mean(values)) if status == "ok" else np.nan,
                    "rom_value": float(np.ptp(values)) if status == "ok" else np.nan,
                }
            )
            normalized.extend(
                {
                    **identity,
                    "phase_percent": float(p),
                    "time_s": float(t),
                    "value": float(v),
                }
                for p, t, v in zip(
                    np.linspace(0.0, 100.0, n_points),
                    np.linspace(cycle["start_s"], cycle["end_s"], n_points),
                    values,
                )
            )
    outputs = {}
    tables = [
        ("lower_limb_kinematics", raw_rows, RAW_FIELDS),
        ("lower_limb_gait_cycles", normalized, CYCLE_FIELDS),
        ("gait_cycle_metrics", metrics, METRIC_FIELDS),
    ]
    for joint in ("hip", "knee", "ankle"):
        tables.extend(
            [
                (
                    f"{joint}_kinematics",
                    [r for r in raw_rows if r["joint"] == joint],
                    RAW_FIELDS,
                ),
                (
                    f"{joint}_gait_cycles",
                    [r for r in normalized if r["joint"] == joint],
                    CYCLE_FIELDS,
                ),
            ]
        )
    for name, rows, fields in tables:
        if export_csv:
            path = trial_dir / f"{name}.csv"
            _write_csv(path, rows, fields)
            outputs[name] = str(path)
    for name, rows in (
        ("lower_limb_kinematics", raw_rows),
        ("gait_cycle_metrics", metrics),
    ):
        path = trial_dir / f"{name}.npz"
        write_npz(path, rows)
        outputs[name + "_cache"] = str(path)
    npz = trial_dir / "gait_cycle_timeseries.npz"
    write_npz(npz, normalized)
    outputs["gait_cycle_timeseries"] = str(npz)
    report.update(
        {
            "trial": trial,
            "phase": "full_gait_cycle" if task == "gait" else task + "_cycle",
            "task": task,
            "csv_export": "explicit_only" if not export_csv else "requested",
            "points_per_cycle": n_points,
            "maximum_interpolation_gap_s": max_gap_s,
            "angular_interpolation": "unwrapped_degrees_within_observed_spans",
            "coordinates": "native_model_dofs_not_anatomically_harmonized",
            "available_channels": [
                {k: v for k, v in channel.items() if k not in {"time", "values"}}
                for channel in channels
            ],
            "valid_curve_cycles": sum(
                r["normalization_status"] == "ok" for r in metrics
            ),
            "unavailable_curve_cycles": sum(
                r["normalization_status"] != "ok" for r in metrics
            ),
        }
    )
    report_path = trial_dir / "gait_cycles.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    outputs["gait_cycles"] = str(report_path)
    return report, outputs
