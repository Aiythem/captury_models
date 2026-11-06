"""Available-source result adapter for the shared comparison workflow.

Native q curves stay independent. Cross-capture spatial/time registration is
handled by the paired adapter; no registration is inferred from axis names.
"""

from __future__ import annotations
import json
import os
import numpy as np


def native_q_rows(trial, runs):
    summaries, curves = [], []
    for run in runs:
        for name, unit, values in zip(run.q_names, run.q_units, run.q):
            finite = values[np.isfinite(values)]
            suffix = "rad" if unit == "rad" else "native"
            summaries.append(
                {
                    "trial": trial,
                    "source": run.system,
                    "q_name": name,
                    "unit": unit,
                    f"mean_{suffix}": float(np.mean(finite)) if finite.size else np.nan,
                    f"rom_{suffix}": float(np.ptp(finite)) if finite.size else np.nan,
                    "comparison_status": "native_source_curve_not_anatomical_agreement",
                }
            )
            curves.extend(
                {
                    "trial": trial,
                    "source": run.system,
                    "q_name": name,
                    "time": float(time),
                    run.system: float(value),
                }
                for time, value in zip(run.time, values)
            )
    return summaries, curves


def prepare_capture(bundle, trial_dir, args, source, calibration, c):
    coordinates = c.resolve_system_coordinate_frames(
        source, c.requested_model_to_own_c3d_axis(args, source)
    )
    root_mode = (
        calibration["root_offset_mode"]
        if calibration
        else c.requested_root_offset_mode(args, source, None)
    )
    run, audit = c.build_model_run_with_rotation_audit(
        bundle, source, args, trial_dir, root_mode, coordinates.model_to_own_c3d
    )
    centres = c.centres_to_c3d_mm(
        run.centres_native, run.unit_scale_to_m, coordinates.model_to_own_c3d
    )
    if calibration is None:
        rotation, translation = np.eye(3), np.zeros(3)
        fit = {"status": "disabled", "method": "identity"}
        if source == "motive" and not args.disable_motive_marker_alignment:
            rotation, translation, fit = c.model_to_motive_marker_alignment(
                centres, run.time, bundle.motive_c3d
            )
            if fit.get("status") != "ok":
                raise RuntimeError(
                    f"Cannot align Motive static model to C3D markers: {fit}. Check the static recording and source units/axes."
                )
        calibration = {
            "static_trial": bundle.name,
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
            "fit": fit,
            "root_offset_mode": c.selected_root_offset_mode(run),
            "status": (
                "diagnostic_static_marker_proxy_alignment"
                if source == "motive"
                else "native_source_frame_no_inter_system_registration"
            ),
        }
    rotation, translation = np.asarray(calibration["rotation"]), np.asarray(
        calibration["translation_mm"]
    )
    centres = c.apply_alignment(centres, rotation, translation)
    centres = {
        name: coordinates.own_c3d_to_common @ points for name, points in centres.items()
    }
    rotations = c.rotations_to_c3d(
        run.rotations_native,
        coordinates.own_c3d_to_common @ rotation.T @ coordinates.model_to_own_c3d,
    )
    return run, centres, rotations, audit, coordinates.to_report(), calibration


def compare_available_trial(bundle, out_root, args, *, api, calibration=None):
    """Produce the same result schemas for selected available sources."""
    c = api
    selected = tuple(getattr(args, "sources", ("motive", "biobuddy")))
    capture_sources = [
        s
        for s in ("captury", "motive")
        if s in selected and getattr(bundle, f"{s}_c3d", None)
    ]
    if len(capture_sources) > 1:
        raise ValueError(
            "Paired capture recordings require the registered comparison adapter."
        )
    if selected and (args.rotate_body_segments_180_x or args.reexpress_rotations_zxy):
        raise ValueError(
            "Available-source results currently display native q and segment frames. Disable the 180-degree frame rotation and ZXY reexpression options."
        )
    trial_dir = out_root / c.safe_name(bundle.name)
    trial_dir.mkdir(parents=True, exist_ok=True)
    full_runs, source_centres, source_rotations, audits, coordinates = (
        [],
        {},
        {},
        {},
        {},
    )
    for source in capture_sources:
        run, centres, rotations, audit, frame, calibration = prepare_capture(
            bundle, trial_dir, args, source, calibration, c
        )
        full_runs.append(run)
        source_centres[source], source_rotations[source] = centres, rotations
        audits[source], coordinates[source] = audit, frame
    biobuddy = None
    ik_report = {"status": "not_selected"}
    if "biobuddy" in selected:
        biobuddy, ik_report = c.run_biobuddy_ik_for_trial(
            bundle,
            trial_dir,
            args.biobuddy_biomod,
            args.biobuddy_unit_scale_to_m,
            args.ik_max_frames,
            args.angle_label_regex,
            args.out_dir / "biobuddy_ik_cache",
            force=args.no_cache,
            cache_only=not args.run_ik_batch,
        )
        if biobuddy is not None:
            full_runs.append(biobuddy)
            source_centres["biobuddy"] = c.centres_to_c3d_mm(
                biobuddy.centres_native, biobuddy.unit_scale_to_m, "identity"
            )
            source_rotations["biobuddy"] = biobuddy.rotations_native
    # The marker-based events use the Motive clock. Never apply them to an
    # unregistered Captury clock just because both recordings start at zero.
    motive_clock = (
        bool(full_runs)
        and "captury" not in capture_sources
        and bundle.motive_c3d is not None
    )
    event_report, full_contact_rows = {
        "status": "unavailable",
        "reason": "no_registered_motive_clock",
    }, []
    if motive_clock:
        event_report, full_contact_rows = c.detect_trial_events_and_contacts(
            bundle.motive_c3d,
            trial_dir,
            bundle.name,
        )
    if (
        full_runs
        and args.cut_mode == "movement"
        and not motive_clock
        and bundle.name != args.static_trial
    ):
        raise ValueError(
            "Movement-based cropping requires a registered Motive marker clock; select full or manual cropping for these sources."
        )
    start, end, cut_mode = c.resolve_cut_window(
        args.cut_mode if full_runs and bundle.name != args.static_trial else "full",
        args.time_start,
        args.time_end,
        event_report,
    )
    if bundle.name == args.static_trial:
        start, end, cut_mode = None, None, "full"
    if motive_clock and (start is not None or end is not None):
        event_report, _ = c.detect_trial_events_and_contacts(
            bundle.motive_c3d,
            trial_dir,
            bundle.name,
            time_start_s=start,
            time_end_s=end,
        )
    runs, centres, rotations, times = [], {}, {}, {}
    registry = c.load_kinematic_conventions()
    for full in full_runs:
        run = c.trim_model_run(full, start, end)
        if not run.time.size:
            if full.system != "biobuddy":
                raise ValueError(
                    f"No {full.system.title()} frames inside the selected time window for {bundle.name}."
                )
            ik_report = {
                "status": "missing",
                "reason": "no_ik_frames_in_selected_window",
            }
            continue
        runs.append(run)
        source = run.system
        source_id = "biobuddy_motive57" if source == "biobuddy" else f"{source}_model"
        names = c.segment_source_names(registry, source_id, run.source_kind)
        mask = c.time_window_mask(full.time, start, end)
        raw_centres = c.trim_centres(source_centres[source], mask)
        raw_rotations = c.trim_rotations(source_rotations[source], mask)
        centres[source] = {
            key: raw_centres[name] for key, name in names.items() if name in raw_centres
        }
        rotations[source] = {
            key: raw_rotations[name]
            for key, name in names.items()
            if name in raw_rotations
        }
        times[source] = run.time
    limitations = [
        "Native q curves are independent; matching names do not establish anatomical equivalence.",
        "Segment-origin distances and native-frame rotation differences are diagnostic only.",
    ]
    if "captury" in centres and "biobuddy" in centres:
        # Keep q and dimensions, but do not fabricate an overlay or a paired
        # metric without independent cross-system space/time calibration.
        centres.pop("biobuddy")
        rotations.pop("biobuddy")
        limitations.append(
            "Captury/BioBuddy spatial overlay and paired metrics are unavailable without paired static Captury/Motive registration; native q and dimensions remain available."
        )
    centre_rows, centre_curves = [], []
    for source, joints in centres.items():
        if source == "biobuddy" and "motive" in centres:
            continue
        for joint, points in joints.items():
            for index, time in enumerate(times[source]):
                centre_curves.append(
                    {
                        "trial": bundle.name,
                        "joint": joint,
                        "time": float(time),
                        **{
                            f"{source}_{axis}_mm": float(points[i, index])
                            for i, axis in enumerate("xyz")
                        },
                    }
                )
    if "biobuddy" in centres and "motive" in centres:
        c.add_biobuddy_centre_rows(
            bundle.name,
            centre_rows,
            centre_curves,
            centres["motive"],
            times["motive"],
            centres["biobuddy"],
            times["biobuddy"],
            args.joint_filter,
        )
    q_rows, q_curves = native_q_rows(bundle.name, runs)
    segment_rows, segment_curves, segment_report = c.segment_rotation_metric_rows(
        bundle.name, rotations, times, args.segment_reference
    )
    for row in centre_rows + segment_rows:
        row["comparison_status"] = "diagnostic_native_segment_origins_and_frames"
    dimensions = c.model_dimension_rows(
        bundle.name, [r for r in runs if r.system != "biobuddy"]
    )
    dimension_report = {"status": "not_selected"}
    if "biobuddy" in selected:
        bio_dimensions, dimension_report = c.biobuddy_dimension_rows(
            bundle.name, args.biobuddy_biomod, args.biobuddy_unit_scale_to_m
        )
        dimensions.extend(bio_dimensions)
    if bundle.motive_c3d and ("motive" in selected or "biobuddy" in selected):
        c.analyze_motive_occlusions(
            bundle.motive_c3d, trial_dir, bundle.name, generate_figure=False
        )
    else:
        c.write_rows(trial_dir / "motive_marker_occlusions.csv", [])
    outputs = {}
    for stem, rows in (
        ("joint_centre_metrics", centre_rows),
        ("kinematics_q_metrics", q_rows),
        ("segment_rotation_metrics", segment_rows),
        ("model_dimensions", dimensions),
        ("skin_marker_correspondence_metrics", []),
        ("captury_c3d_angle_metrics", []),
    ):
        for row in rows:
            row["participant"] = bundle.participant
        path = trial_dir / f"{stem}.csv"
        c.write_rows(path, rows)
        outputs[stem] = str(path)
    for stem, rows in (
        ("joint_centre_timeseries", centre_curves),
        ("kinematics_q_timeseries", q_curves),
        ("segment_rotation_timeseries", segment_curves),
        ("skin_marker_correspondence_timeseries", []),
        ("captury_c3d_angle_timeseries", []),
    ):
        path = trial_dir / f"{stem}.npz"
        c.write_table_npz(path, rows)
        outputs[stem] = str(path)
    if "motive" in capture_sources:
        path = trial_dir / "motive_static_alignment.json"
        path.write_text(json.dumps(calibration, indent=2))
        outputs["motive_static_alignment"] = str(path)
    if not motive_clock:
        (trial_dir / "trial_events.json").write_text(json.dumps(event_report, indent=2))
        c.write_rows(trial_dir / "trial_events_contacts.csv", [])
    for filename in (
        "spatial_calibration.json",
        "captury_c3d_angle_decode.json",
        "joint_kinematics_d4_d6.json",
        "joint_kinematics_d4_d6.npz",
        "temporal_synchronization.json",
        "metric_quality.json",
        "contact_cycles.json",
        "temporal_synchronization_timeseries.npz",
        "temporal_phase_normalized.npz",
        "phase_normalized_scientific_timeseries.npz",
    ):
        (trial_dir / filename).unlink(missing_ok=True)
    if "motive" not in capture_sources:
        (trial_dir / "motive_static_alignment.json").unlink(missing_ok=True)
    report = {
        "trial": bundle.name,
        "participant": bundle.participant,
        "workflow": "source_comparison",
        "available_sources": [r.system for r in runs],
        "files": {
            f"{s}_c3d": str(getattr(bundle, f"{s}_c3d"))
            for s in ("captury", "motive")
            if getattr(bundle, f"{s}_c3d", None)
        },
        "models": {
            r.system: {
                "biomod": str(r.biomod_path),
                "source_kind": r.source_kind,
                "unit_scale_to_m": r.unit_scale_to_m,
                "root_offset_policy": r.root_offset_policy,
            }
            for r in runs
            if r.system != "biobuddy"
        },
        "axis_conversion": coordinates,
        "time_window": {"start_s": start, "end_s": end, "effective_cut_mode": cut_mode},
        "alignment": {
            "status": (
                "frozen_motive_static_marker_alignment"
                if "motive" in capture_sources
                else "native_source_frame"
            ),
            "motive_model_to_c3d_markers": (
                calibration if "motive" in capture_sources else None
            ),
        },
        "temporal_synchronization": {
            "status": (
                "not_applicable_single_motive_clock"
                if motive_clock
                else "unavailable_inter_system_registration"
            )
        },
        "biobuddy_ik_batch": ik_report,
        "segment_rotations": segment_report,
        "bvh_fbx_rotation_audit": audits,
        "trial_events": event_report,
        "outputs": outputs,
        "skin_marker_correspondence": {
            "status": "unavailable",
            "reason": "requires_both_capture_recordings",
        },
        "limitations": limitations,
    }
    report["models"]["biobuddy"] = dimension_report
    (trial_dir / "run_report.json").write_text(json.dumps(report, indent=2))
    if (
        args.visualize
        and runs
        and (args.visualize_trial is None or args.visualize_trial == bundle.name)
    ):
        if args.headless:
            os.environ["PYORERUN_HEADLESS"] = "1"
        _, PhaseRerun, PyoMarkers = c.require_pyorerun()
        display_time = runs[0].time
        phase = PhaseRerun(display_time)
        for source, source_centres in centres.items():
            sampled = c.interpolate_centres_to_time(
                source_centres, times[source], display_time
            )
            labels = sorted(sampled)
            if labels:
                phase.add_xp_markers(
                    source,
                    PyoMarkers(
                        data=np.stack([sampled[label] for label in labels], axis=1),
                        channels=labels,
                    ),
                )
        if not args.headless:
            phase.rerun("source_comparison_joint_centres", notebook=False)
    return report, calibration
