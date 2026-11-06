#!/usr/bin/env python3
"""Batch comparison for Captury/Motive kinematic datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from bvh_c3d_biobuddy_pyorerun_compare import (
    append_joint_centre_markers_to_biomod,
    as_str_list,
    build_biomod_from_bvh_with_biobuddy,
    build_biomod_from_fbx_with_biobuddy,
    clone_c3d_dict,
    collect_fbx_joint_names_depth_first,
    compute_model_joint_centres_native,
    convert_biobuddy_ply_meshes_to_vtp,
    extract_q_from_biobuddy_bvh_parser,
    extract_q_from_fbx_parser,
    get_c3d_param,
    interpolate_array,
    require_biorbd,
    require_ezc3d,
    require_pyorerun,
    save_model_joint_centres,
    save_q_outputs,
    split_c3d_points,
)
from compare_capture_systems import (
    DEFAULT_LANDMARK_MAP,
    detect_angle_indices,
    load_landmark_map,
    unit_scale_to_mm,
)
from mocap_alignment import (
    apply_row_alignment,
    compose_row_alignment,
    kabsch_rows,
)
from mocap_labels import (
    DEFAULT_MARKER_PREFIXES_TO_STRIP,
    display_marker_name,
    is_joint_centre_marker_label,
    marker_display_labels,
    marker_indices_by_display_label,
)
from model_comparison_metrics import (
    aggregate_trial_metrics_by_participant,
    bland_altman,
    joint_center_error_xyz,
    waveform_metrics,
)
from run_biobuddy_c3d_ik import (
    cached_ik_is_valid,
    ik_cache_fingerprint,
    run_direct_biobuddy_ik,
)
from spatial_calibration import (
    RowRigidTransform,
    SpatialCalibration,
    fit_held_out_centre_alignment,
)
from kinematic_conventions import (
    build_provenance_manifest,
    load_kinematic_conventions,
    segment_source_names,
)
from isb_segment_audit import (
    build_isb_d1_d3_audit,
    load_biobuddy_audit_sidecars,
    write_isb_d1_d3_audit,
)
from isb_compliance_report import (
    build_isb_d1_d6_report,
    build_reproducibility_manifest,
    write_isb_d1_d6_report,
    write_reproducibility_manifest,
)
from kinematic_rotations import (
    assess_rotation_source_equivalence,
    canonicalize_rotation_series,
    canonicalize_segment_rotation_mapping,
    change_lab_basis,
    compare_segment_rotation_mappings,
    rotation_vector as canonical_rotation_vector,
    slerp_rotation_series,
)
from joint_kinematics import (
    audit_joint_kinematics_source,
    frame_corrections_from_static_audit,
    write_joint_kinematics_audit,
)
from captury_c3d_angles import analyze_captury_angle_channels
from captury_frame_calibration import (
    apply_segment_frame_corrections,
    apply_simple_captury_axis_rename,
    fit_captury_segment_frame_corrections,
    is_static_frame_calibration_trial,
    load_captury_q_landmark_config,
)
from mocap_coordinate_frames import (
    resolve_system_coordinate_frames,
    transform_point_trajectories,
)
from temporal_synchronization import (
    LAG_CONVENTION,
    apply_time_offset,
    composite_joint_centre_speed,
    interpolate_finite_signal,
    interpolate_finite_array,
    normalize_selected_phase,
    normalize_timeseries_groups,
    resolve_temporal_synchronization,
)

DEFAULT_DATA_ROOT = Path("local_trials/2026-06-30_P6_flat")
DEFAULT_OUTPUT_ROOT = Path("out_p6_motive_captury_comparison")
ANGLE_LABEL_REGEX = r"(?i)(^.*angles?$|^.*_angle[s]?$|angle)"
FOOT_MARKER_PATTERN = r"(LFCC|RFCC|LFM|RFM|LDP|RDP|Foot|Toe|Heel)"
CACHE_VERSION = 13
ROTATION_SEQUENCE_ZXY = "ZXY"
DEFAULT_ALIGNMENT_CALIBRATION_CENTRES = (
    "Hips",
    "Head",
    "LeftShoulder",
    "RightShoulder",
)
SCIENTIFIC_IMPLEMENTATION_FILES = {
    "comparison_batch": Path(__file__),
    "available_source_results": Path(__file__).with_name("available_source_results.py"),
    "comparison_workflow": Path(__file__).with_name("comparison_workflow.py"),
    "kinematic_conventions_registry": Path(__file__).with_name(
        "kinematic_conventions.json"
    ),
    "kinematic_conventions_code": Path(__file__).with_name("kinematic_conventions.py"),
    "kinematic_rotations_code": Path(__file__).with_name("kinematic_rotations.py"),
    "joint_kinematics_registry": Path(__file__).with_name("isb_joint_kinematics.json"),
    "joint_kinematics_code": Path(__file__).with_name("joint_kinematics.py"),
    "isb_segment_audit_code": Path(__file__).with_name("isb_segment_audit.py"),
    "isb_compliance_report_code": Path(__file__).with_name("isb_compliance_report.py"),
    "spatial_calibration_code": Path(__file__).with_name("spatial_calibration.py"),
    "mocap_alignment_code": Path(__file__).with_name("mocap_alignment.py"),
    "captury_c3d_angle_decoder": Path(__file__).with_name("captury_c3d_angles.py"),
    "captury_c3d_angle_registry": Path(__file__).with_name("captury_c3d_angles.json"),
    "captury_frame_calibration_code": Path(__file__).with_name(
        "captury_frame_calibration.py"
    ),
    "captury_q_landmarks": Path(__file__).with_name("captury_q_landmarks.json"),
    "mocap_coordinate_frames_code": Path(__file__).with_name(
        "mocap_coordinate_frames.py"
    ),
    "biobuddy_ik_code": Path(__file__).with_name("run_biobuddy_c3d_ik.py"),
    "temporal_synchronization_code": Path(__file__).with_name(
        "temporal_synchronization.py"
    ),
    "comparison_metrics_code": Path(__file__).with_name("model_comparison_metrics.py"),
}


def file_sha256(path: Path | str) -> str:
    """Return a streaming SHA-256 digest for one reproducibility artifact."""

    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


MODEL_JOINT_MARKER_PROXIES = {
    "Hips": ("LIAS", "RIAS", "LIPS", "RIPS"),
    "LeftUpLeg": ("LIAS", "LIPS", "LFTC"),
    "RightUpLeg": ("RIAS", "RIPS", "RFTC"),
    "LeftLeg": ("LFLE", "LFME"),
    "RightLeg": ("RFLE", "RFME"),
    "LeftFoot": ("LFAL", "LTAM", "LFAX"),
    "RightFoot": ("RFAL", "RTAM", "RFAX"),
}


@dataclass
class TrialBundle:
    name: str
    captury_c3d: Path | None
    captury_bvh: Path | None
    captury_fbx: Path | None
    motive_c3d: Path | None
    motive_bvh: Path | None
    motive_fbx: Path | None
    participant: str | None = None


@dataclass
class ModelRun:
    system: str
    source_kind: str
    biomod_path: Path
    q: np.ndarray
    q_names: list[str]
    q_units: list[str]
    time: np.ndarray
    joint_names: list[str]
    centres_native: dict[str, np.ndarray]
    rotations_native: dict[str, np.ndarray]
    unit_scale_to_m: float
    mesh_report: dict[str, Any]
    root_offset_policy: dict[str, Any]


def file_fingerprint(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _static_alignment_cache_payload(
    transform: SpatialCalibration | tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any] | None:
    if transform is None:
        return None
    if isinstance(transform, SpatialCalibration):
        return transform.to_dict()
    rotation, translation = transform
    return {
        "rotation": np.asarray(rotation, dtype=float).round(12).tolist(),
        "translation_mm": np.asarray(translation, dtype=float).round(12).tolist(),
    }


def trial_cache_fingerprint(
    bundle: TrialBundle,
    args: argparse.Namespace,
    static_alignment_transform: (
        SpatialCalibration | tuple[np.ndarray, np.ndarray] | None
    ) = None,
) -> dict[str, Any]:
    cached_ik_inputs = {}
    if (
        args.biobuddy_biomod
        and args.biobuddy_biomod.is_file()
        and not args.run_ik_batch
    ):
        fingerprint = biobuddy_ik_cache_fingerprint(
            args.biobuddy_biomod,
            bundle.motive_c3d,
            args.biobuddy_unit_scale_to_m,
            args.ik_max_frames,
            args.angle_label_regex,
        )
        directory = args.out_dir / "biobuddy_ik_cache" / fingerprint["digest"]
        cached_ik_inputs = {
            name: file_fingerprint(directory / name)
            for name in (
                "inverse_kinematics_trf.npz",
                "inverse_kinematics_trf_summary.json",
            )
        }
    payload = {
        "cache_version": CACHE_VERSION,
        "implementation": {
            name: file_fingerprint(path)
            for name, path in SCIENTIFIC_IMPLEMENTATION_FILES.items()
        },
        "inputs": {
            "cached_biobuddy_ik": cached_ik_inputs,
            "captury_c3d": file_fingerprint(bundle.captury_c3d),
            "captury_bvh": file_fingerprint(bundle.captury_bvh),
            "captury_fbx": file_fingerprint(bundle.captury_fbx),
            "motive_c3d": file_fingerprint(bundle.motive_c3d),
            "motive_bvh": file_fingerprint(bundle.motive_bvh),
            "motive_fbx": file_fingerprint(bundle.motive_fbx),
            "biobuddy_biomod": (
                file_fingerprint(args.biobuddy_biomod) if args.biobuddy_biomod else None
            ),
        },
        "options": {
            "sources": list(
                getattr(args, "sources", ("captury", "motive", "biobuddy"))
            ),
            "model_source": args.model_source,
            "audit_bvh_fbx_rotations": bool(
                getattr(args, "audit_bvh_fbx_rotations", False)
            ),
            "bvh_fbx_max_p95_geodesic_deg": float(
                getattr(args, "bvh_fbx_max_p95_geodesic_deg", 5.0)
            ),
            "model_to_c3d_axis": args.model_to_c3d_axis,
            "captury_model_to_c3d_axis": getattr(
                args, "captury_model_to_c3d_axis", None
            ),
            "motive_model_to_c3d_axis": getattr(args, "motive_model_to_c3d_axis", None),
            "captury_q_landmarks_json": (
                file_fingerprint(args.captury_q_landmarks_json)
                if getattr(args, "captury_q_landmarks_json", None)
                else file_fingerprint(
                    Path(__file__).with_name("captury_q_landmarks.json")
                )
            ),
            "captury_unit_scale_to_m": args.captury_unit_scale_to_m,
            "motive_unit_scale_to_m": args.motive_unit_scale_to_m,
            "biobuddy_unit_scale_to_m": args.biobuddy_unit_scale_to_m,
            "root_offset_mode": args.root_offset_mode,
            "captury_root_offset_mode": getattr(args, "captury_root_offset_mode", None),
            "motive_root_offset_mode": getattr(args, "motive_root_offset_mode", None),
            "angle_label_regex": args.angle_label_regex,
            "c3d_angle_unit": args.c3d_angle_unit,
            "landmark_map": (
                file_fingerprint(args.landmark_map) if args.landmark_map else None
            ),
            "segment_reference": args.segment_reference,
            "captury_reorient_thigh_y_from_cor": bool(
                args.captury_reorient_thigh_y_from_cor
            ),
            "captury_rename_axes": bool(getattr(args, "captury_rename_axes", False)),
            "captury_calibrate_segment_frames": bool(
                getattr(args, "captury_calibrate_segment_frames", False)
            ),
            "rotate_body_segments_180_x": bool(args.rotate_body_segments_180_x),
            "reexpress_rotations_zxy": bool(args.reexpress_rotations_zxy),
            "disable_static_model_alignment": bool(args.disable_static_model_alignment),
            "disable_motive_marker_alignment": bool(
                args.disable_motive_marker_alignment
            ),
            "spatial_alignment_mode": getattr(
                args, "spatial_alignment_mode", "held_out_centres"
            ),
            "alignment_calibration_centres": list(
                getattr(args, "alignment_calibration_centre", [])
                or DEFAULT_ALIGNMENT_CALIBRATION_CENTRES
            ),
            "joint_filter": list(args.joint_filter),
            "no_mesh": bool(args.no_mesh),
            "max_mesh_points": int(args.max_mesh_points),
            "run_ik_batch": bool(args.run_ik_batch),
            "ik_max_frames": int(args.ik_max_frames),
            "cut_mode": args.cut_mode,
            "time_start": args.time_start,
            "time_end": args.time_end,
            "temporal_sync_mode": getattr(args, "temporal_sync_mode", "auto"),
            "manual_lag_s": getattr(args, "manual_lag_s", None),
            "max_lag_s": float(getattr(args, "max_lag_s", 0.5)),
            "phase_normalization_points": int(
                getattr(args, "phase_normalization_points", 101)
            ),
            "no_figures": bool(args.no_figures),
        },
        "static_alignment": _static_alignment_cache_payload(static_alignment_transform),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "version": CACHE_VERSION,
        "digest": hashlib.sha256(encoded).hexdigest(),
        "payload": payload,
    }


def required_trial_outputs(
    trial_dir: Path, trial_name: str, *, include_rotation_audit: bool = False
) -> list[Path]:
    safe_trial = safe_name(trial_name)
    outputs = [
        trial_dir / f"{safe_trial}_motive_with_capjc_motjc.c3d",
        trial_dir / "joint_centre_metrics.csv",
        trial_dir / "joint_centre_timeseries.npz",
        trial_dir / "alignment_calibration_centre_metrics.csv",
        trial_dir / "alignment_calibration_centre_timeseries.npz",
        trial_dir / "spatial_calibration.json",
        trial_dir / "kinematics_q_metrics.csv",
        trial_dir / "kinematics_q_timeseries.npz",
        trial_dir / "captury_c3d_angle_metrics.csv",
        trial_dir / "captury_c3d_angle_timeseries.npz",
        trial_dir / "captury_c3d_angle_decode.json",
        trial_dir / "segment_rotation_metrics.csv",
        trial_dir / "segment_rotation_timeseries.npz",
        trial_dir / "joint_kinematics_d4_d6.json",
        trial_dir / "joint_kinematics_d4_d6.npz",
        trial_dir / "model_dimensions.csv",
        trial_dir / "motive_marker_occlusions.csv",
        trial_dir / "skin_marker_correspondence_proposal.json",
        trial_dir / "skin_marker_correspondence_metrics.csv",
        trial_dir / "skin_marker_correspondence_timeseries.npz",
        trial_dir / "trial_events_contacts.csv",
        trial_dir / "temporal_synchronization.json",
        trial_dir / "temporal_synchronization_timeseries.npz",
        trial_dir / "temporal_phase_normalized.npz",
        trial_dir / "phase_normalized_scientific_timeseries.npz",
        trial_dir / "contact_cycles.json",
        trial_dir / "metric_quality.json",
        trial_dir / "metric_sensitivity.json",
        trial_dir / "run_report.json",
    ]
    if include_rotation_audit:
        for system in ("captury", "motive"):
            outputs.extend(
                (
                    trial_dir / system / "bvh_fbx_rotation_audit.json",
                    trial_dir / system / "bvh_fbx_rotation_audit.npz",
                )
            )
    return outputs


def required_output_may_be_empty(path: Path) -> bool:
    """Return whether an empty file is a valid, complete scientific output."""

    return path.name == "skin_marker_correspondence_metrics.csv"


def biobuddy_ik_outputs_complete(report: Mapping[str, Any]) -> bool:
    """Return whether a cached report references both complete IK artifacts."""

    outputs = report.get("biobuddy_ik_batch", {}).get("outputs", {})
    required = (outputs.get("npz"), outputs.get("summary"))
    return all(
        value and Path(value).is_file() and Path(value).stat().st_size > 0
        for value in required
    )


def cached_trial_report(
    trial_dir: Path,
    bundle: TrialBundle,
    args: argparse.Namespace,
    static_alignment_transform: (
        SpatialCalibration | tuple[np.ndarray, np.ndarray] | None
    ) = None,
) -> dict[str, Any] | None:
    if args.no_cache:
        return None
    report_path = trial_dir / "run_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    expected = trial_cache_fingerprint(bundle, args, static_alignment_transform)
    if report.get("cache", {}).get("fingerprint") != expected:
        return None
    include_rotation_audit = bool(
        getattr(args, "audit_bvh_fbx_rotations", False)
        or getattr(args, "model_source", None) == "auto"
    )
    for output_path in required_trial_outputs(
        trial_dir,
        bundle.name,
        include_rotation_audit=include_rotation_audit,
    ):
        if not output_path.exists():
            return None
        if output_path.stat().st_size == 0 and not required_output_may_be_empty(
            output_path
        ):
            return None
    if args.run_ik_batch and not biobuddy_ik_outputs_complete(report):
        return None
    return report


def static_transform_from_report(
    report: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray] | None:
    alignment = report.get("alignment", {})
    rotation = alignment.get("rotation")
    translation = alignment.get("translation_mm")
    if rotation is None or translation is None:
        return None
    try:
        return np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    except (TypeError, ValueError):
        return None


def spatial_calibration_from_report(
    report: Mapping[str, Any],
) -> SpatialCalibration | None:
    """Load the versioned static calibration embedded in a trial report."""

    values = report.get("spatial_calibration")
    if not isinstance(values, Mapping):
        return None
    try:
        return SpatialCalibration.from_dict(values)
    except (KeyError, TypeError, ValueError):
        return None


def selected_root_offset_mode(run: ModelRun) -> str:
    """Return the CLI policy equivalent to a model run's selected convention."""

    selected = run.root_offset_policy.get("selected_mode")
    if selected == "subtract_static_offset_from_root_q":
        return "subtract"
    if selected == "keep_root_q_as_file":
        return "keep"
    raise ValueError(f"Unsupported selected root-offset policy: {selected!r}")


def requested_root_offset_mode(
    args: argparse.Namespace,
    system: str,
    calibration: SpatialCalibration | None,
) -> str:
    """Resolve a system-specific policy, forcing the static choice on dynamics."""

    if calibration is not None:
        frozen = (
            calibration.captury_root_offset_mode
            if system == "captury"
            else calibration.motive_root_offset_mode
        )
        if frozen in {"keep", "subtract"}:
            return frozen
    specific = getattr(args, f"{system}_root_offset_mode", None)
    return specific or args.root_offset_mode


def requested_model_to_own_c3d_axis(
    args: argparse.Namespace,
    system: str,
) -> str:
    """Resolve an explicit system-specific model-to-own-C3D basis mode.

    A single manual axis choice cannot be physically valid for both current
    exports: Captury C3D is ``+Y`` up and Motive C3D is ``+Z`` up. The legacy
    shared option is therefore accepted only in ``auto`` mode; explicit
    diagnostics must name the system they affect.
    """

    specific = getattr(args, f"{system}_model_to_c3d_axis", None)
    if specific:
        return str(specific)
    legacy = str(getattr(args, "model_to_c3d_axis", "auto"))
    if legacy != "auto":
        raise ValueError(
            "--model-to-c3d-axis is only valid with auto for Captury/Motive "
            "comparisons. Use --captury-model-to-c3d-axis and "
            "--motive-model-to-c3d-axis explicitly."
        )
    return "auto"


def safe_name(value: str) -> str:
    import re

    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_") or "trial"


def infer_participant_identifier(*paths: Path) -> str | None:
    """Infer a participant token only from explicit P-prefixed file/path tokens."""

    patterns = (
        re.compile(r"(?i)(?:^|[_\-/])(P\d+)(?:[_\-/.]|$)"),
        re.compile(r"(?i)(?:^|_)(P\d+)$"),
    )
    candidates: list[str] = []
    for path in paths:
        text = str(path)
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                candidates.append(match.group(1).upper())
                break
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def normalize_summary_source(rows: list[dict[str, Any]], default: str) -> None:
    """Replace absent/NaN source labels before CSV and population grouping."""

    for row in rows:
        value = row.get("source")
        if value is None or (
            isinstance(value, (float, np.floating)) and not np.isfinite(value)
        ):
            row["source"] = default
        elif not str(value).strip() or str(value).strip().lower() == "nan":
            row["source"] = default


def metric_quality_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize waveform eligibility without converting frames into subjects."""

    waveform_rows = [row for row in rows if "waveform_status" in row]
    status_counts: dict[str, int] = {}
    eligible = 0
    for row in waveform_rows:
        status = str(row["waveform_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        eligible += int(bool(row.get("shape_metrics_eligible", False)))
    return {
        "schema_version": 1,
        "status": "ok" if waveform_rows else "no_waveform_metrics",
        "statistical_unit": "trial_waveform",
        "waveforms": len(waveform_rows),
        "eligible_waveforms": eligible,
        "ineligible_waveforms": len(waveform_rows) - eligible,
        "descriptive_rows_excluded": len(rows) - len(waveform_rows),
        "status_counts": status_counts,
        "guarded_metrics": [
            "nrmse_range",
            "pearson_r_waveform",
            "lin_ccc_waveform",
        ],
        "limits_of_agreement_scope": (
            "descriptive_within_trial_frames_not_population_ci"
        ),
    }


def metric_sensitivity_report(
    *,
    root_policies: dict[str, dict[str, Any]],
    temporal: dict[str, Any],
    evaluation_centre_rows: list[dict[str, Any]],
    calibration_centre_rows: list[dict[str, Any]],
    rotation_audits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Report already-computed counterfactuals without inventing new variants."""

    root_report: dict[str, Any] = {}
    for system in ("captury", "motive"):
        policy = root_policies.get(system, {})
        subtract = policy.get("score_mm_subtract_static_offset")
        keep = policy.get("score_mm_keep_file_translation")
        if isinstance(subtract, (float, int)) and isinstance(keep, (float, int)):
            root_report[system] = {
                "status": "computed",
                "selected_mode": policy.get("selected_mode"),
                "score_mm_subtract_static_offset": float(subtract),
                "score_mm_keep_file_translation": float(keep),
                "score_difference_mm": float(abs(float(keep) - float(subtract))),
            }
        else:
            root_report[system] = {"status": "not_computed"}
    before = temporal.get("normalized_rmse_before")
    after = temporal.get("normalized_rmse_after")
    temporal_report = {"status": "not_computed"}
    if isinstance(before, (float, int)) and isinstance(after, (float, int)):
        temporal_report = {
            "status": "computed",
            "estimated_lag_s": temporal.get("estimated_lag_s"),
            "applied_lag_s": temporal.get("lag_s"),
            "normalized_rmse_before": float(before),
            "normalized_rmse_after": float(after),
            "normalized_rmse_improvement": float(before) - float(after),
        }

    def median_metric(rows: list[dict[str, Any]], metric: str) -> float | None:
        values = np.asarray(
            [float(row.get(metric, np.nan)) for row in rows], dtype=float
        )
        values = values[np.isfinite(values)]
        return float(np.median(values)) if values.size else None

    evaluation = median_metric(evaluation_centre_rows, "median_error_mm")
    calibration = median_metric(calibration_centre_rows, "median_error_mm")
    centre_report: dict[str, Any] = {"status": "not_computed"}
    if evaluation is not None and calibration is not None:
        centre_report = {
            "status": "computed",
            "calibration_centres_median_error_mm": calibration,
            "held_out_centres_median_error_mm": evaluation,
            "held_out_minus_calibration_mm": evaluation - calibration,
            "interpretation": "calibration_and_evaluation_sets_are_disjoint",
        }
    source_report: dict[str, Any] = {"status": "not_computed"}
    computed_sources: dict[str, Any] = {}
    for system, audit in rotation_audits.items():
        summary = audit.get("summary", audit)
        segments = summary.get("segments", {}) if isinstance(summary, dict) else {}
        values = [
            float(item["p95_geodesic_deg"])
            for item in segments.values()
            if isinstance(item, dict)
            and isinstance(item.get("p95_geodesic_deg"), (float, int))
        ]
        if values:
            computed_sources[system] = {
                "segments": len(values),
                "median_p95_geodesic_deg": float(np.median(values)),
                "max_p95_geodesic_deg": float(np.max(values)),
            }
    if computed_sources:
        source_report = {"status": "computed", "systems": computed_sources}
    return {
        "schema_version": 1,
        "status": "diagnostic_counterfactuals",
        "root_translation": root_report,
        "temporal_lag": temporal_report,
        "centre_definition": centre_report,
        "bvh_fbx_source": source_report,
        "axis_sequence_and_anatomical_frame": {
            "status": "not_computed",
            "reason": "requires_anatomically_valid_counterfactual_models",
        },
    }


def discover_trials(data_root: Path) -> list[TrialBundle]:
    if (data_root / "Motive").is_dir() or (data_root / "Captury").is_dir():
        return discover_flat_trials(data_root)

    trials: list[TrialBundle] = []
    for trial_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        captury_dir = trial_dir / "captury"
        motive_dir = trial_dir / "squelettes"
        if not captury_dir.is_dir() and not motive_dir.is_dir():
            continue
        captury_c3d = captury_dir / "P6.c3d"
        motive_c3ds = sorted(motive_dir.glob("*.c3d"))
        motive_bvhs = sorted(motive_dir.glob("*Skeleton 001.bvh"))
        motive_fbxs = sorted(motive_dir.glob("*.fbx"))
        if not captury_c3d.exists() and not motive_c3ds:
            continue
        trials.append(
            TrialBundle(
                name=trial_dir.name,
                captury_c3d=first_existing(captury_c3d),
                captury_bvh=first_existing(captury_dir / "P6.bvh"),
                captury_fbx=first_existing(captury_dir / "P6.fbx"),
                motive_c3d=motive_c3ds[0] if motive_c3ds else None,
                motive_bvh=motive_bvhs[0] if motive_bvhs else None,
                motive_fbx=motive_fbxs[0] if motive_fbxs else None,
                participant=infer_participant_identifier(
                    *([captury_c3d] if captury_c3d.exists() else []) + motive_c3ds[:1]
                ),
            )
        )
    return trials


def captury_flat_trial_name(path: Path) -> str:
    name = path.stem
    return name[: -len("_P6")] if name.endswith("_P6") else name


def motive_flat_trial_name(path: Path) -> str:
    name = path.stem
    if name.startswith("P6_"):
        name = name[3:]
    if name.endswith("_Skeleton 001"):
        name = name[: -len("_Skeleton 001")]
    return name


def discover_flat_trials(data_root: Path) -> list[TrialBundle]:
    captury_dir = data_root / "Captury"
    motive_dir = data_root / "Motive"
    captury: dict[str, dict[str, Path]] = {}
    motive: dict[str, dict[str, Path]] = {}

    for path in sorted(captury_dir.glob("*")):
        if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
            continue
        captury.setdefault(captury_flat_trial_name(path), {})[
            path.suffix.lower().lstrip(".")
        ] = path
    for path in sorted(motive_dir.glob("*")):
        if path.suffix.lower() not in {".bvh", ".fbx", ".c3d"}:
            continue
        motive.setdefault(motive_flat_trial_name(path), {})[
            path.suffix.lower().lstrip(".")
        ] = path

    trials: list[TrialBundle] = []
    trial_names = set(captury).union(motive)
    for trial in sorted(trial_names):
        cap_files = captury.get(trial, {})
        mot_files = motive.get(trial, {})
        if "c3d" not in cap_files and "c3d" not in mot_files:
            continue
        trials.append(
            TrialBundle(
                name=trial,
                captury_c3d=cap_files.get("c3d"),
                captury_bvh=cap_files.get("bvh"),
                captury_fbx=cap_files.get("fbx"),
                motive_c3d=mot_files.get("c3d"),
                motive_bvh=mot_files.get("bvh"),
                motive_fbx=mot_files.get("fbx"),
                participant=infer_participant_identifier(
                    *[
                        path
                        for path in (cap_files.get("c3d"), mot_files.get("c3d"))
                        if path
                    ]
                ),
            )
        )
    return trials


def first_existing(path: Path) -> Path | None:
    return path if path.exists() else None


def select_model_file(
    bundle: TrialBundle, system: str, model_source: str
) -> tuple[str, Path]:
    bvh = bundle.captury_bvh if system == "captury" else bundle.motive_bvh
    fbx = bundle.captury_fbx if system == "captury" else bundle.motive_fbx
    if model_source == "bvh":
        if bvh is None:
            raise FileNotFoundError(f"No {system} BVH for {bundle.name}.")
        return "bvh", bvh
    if model_source == "fbx":
        if fbx is None:
            raise FileNotFoundError(f"No {system} FBX for {bundle.name}.")
        return "fbx", fbx
    if bvh is not None:
        return "bvh", bvh
    if fbx is not None:
        return "fbx", fbx
    raise FileNotFoundError(f"No {system} BVH/FBX for {bundle.name}.")


def comparison_input_files(
    trials: list[TrialBundle], args: argparse.Namespace
) -> dict[str, Path]:
    """Return every input that materially contributes to this batch run."""

    from comparison_workflow import requested_sources

    selected = requested_sources(args)
    inputs: dict[str, Path] = {}
    for bundle in trials:
        prefix = bundle.name
        if "captury" in selected and bundle.captury_c3d is not None:
            inputs[f"{prefix}/captury/c3d"] = bundle.captury_c3d
        if bundle.motive_c3d is not None and (
            args.occlusions_only or "motive" in selected or "biobuddy" in selected
        ):
            inputs[f"{prefix}/motive/c3d"] = bundle.motive_c3d
        if not args.occlusions_only:
            audit_both = bool(
                getattr(args, "audit_bvh_fbx_rotations", False)
                or args.model_source == "auto"
            )
            for system in ("captury", "motive"):
                if system not in selected or getattr(bundle, f"{system}_c3d") is None:
                    continue
                bvh = bundle.captury_bvh if system == "captury" else bundle.motive_bvh
                fbx = bundle.captury_fbx if system == "captury" else bundle.motive_fbx
                if audit_both and bvh is not None and fbx is not None:
                    inputs[f"{prefix}/{system}/bvh"] = bvh
                    inputs[f"{prefix}/{system}/fbx"] = fbx
                    continue
                try:
                    source_kind, source_path = select_model_file(
                        bundle, system, args.model_source
                    )
                except FileNotFoundError:
                    continue
                inputs[f"{prefix}/{system}/{source_kind}"] = source_path
    if "biobuddy" in selected and getattr(args, "biobuddy_biomod", None) is not None:
        inputs["biobuddy/biomod"] = args.biobuddy_biomod
    if getattr(args, "run_ik_batch", False):
        inputs["implementation/biobuddy_ik_code"] = SCIENTIFIC_IMPLEMENTATION_FILES[
            "biobuddy_ik_code"
        ]
    if getattr(args, "landmark_map", None) is not None:
        inputs["markers/landmark_map"] = args.landmark_map
    return inputs


def provenance_trials_with_static(
    selected_trials: list[TrialBundle],
    discovered_trials: list[TrialBundle],
    static_trial_name: str,
) -> list[TrialBundle]:
    """Include a hidden static calibration trial exactly once in provenance."""

    result = list(selected_trials)
    if any(bundle.name == static_trial_name for bundle in result):
        return result
    static_bundle = next(
        (bundle for bundle in discovered_trials if bundle.name == static_trial_name),
        None,
    )
    if static_bundle is not None:
        result.append(static_bundle)
    return result


def split_static_calibration_trial(
    trials: list[TrialBundle], static_trial_name: str
) -> tuple[TrialBundle | None, list[TrialBundle]]:
    """Separate the declared static calibration from evaluation trials."""

    static_bundle = next(
        (bundle for bundle in trials if bundle.name == static_trial_name), None
    )
    dynamic_trials = [bundle for bundle in trials if bundle.name != static_trial_name]
    return static_bundle, dynamic_trials


def comparison_derived_artifacts(
    reports: list[dict[str, Any]],
    batch_artifacts: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Collect generated models and maps consumed later in the same run."""

    artifacts: dict[str, Path] = dict(batch_artifacts or {})
    for report in reports:
        trial = str(report["trial"])
        models = report.get("models", {})
        for system in ("captury", "motive"):
            biomod = models.get(system, {}).get("biomod")
            if biomod:
                artifacts[f"{trial}/{system}/generated_biomod"] = Path(biomod)
            rotation_audit = report.get("bvh_fbx_rotation_audit", {}).get(system, {})
            for source_kind, path in rotation_audit.get(
                "generated_biomods", {}
            ).items():
                artifacts[f"{trial}/{system}/{source_kind}/audit_generated_biomod"] = (
                    Path(path)
                )
            for artifact_kind, path in rotation_audit.get("artifacts", {}).items():
                artifacts[f"{trial}/{system}/rotation_audit_{artifact_kind}"] = Path(
                    path
                )
        marker_comparison = report.get("skin_marker_correspondence", {})
        if marker_comparison.get("map_source") == "automatic_proposal":
            proposal = report.get("outputs", {}).get(
                "skin_marker_correspondence_proposal"
            )
            if proposal:
                artifacts[f"{trial}/markers/automatic_proposal"] = Path(proposal)
        spatial_calibration = report.get("outputs", {}).get("spatial_calibration")
        if spatial_calibration:
            artifacts[f"{trial}/alignment/spatial_calibration"] = Path(
                spatial_calibration
            )
        motive_calibration = report.get("outputs", {}).get("motive_static_alignment")
        if motive_calibration:
            artifacts[f"{trial}/alignment/motive_static_alignment"] = Path(
                motive_calibration
            )
        joint_kinematics = report.get("outputs", {}).get("joint_kinematics_d4_d6")
        if joint_kinematics:
            artifacts[f"{trial}/kinematics/d4_d6_json"] = Path(joint_kinematics)
        joint_kinematics_timeseries = report.get("outputs", {}).get(
            "joint_kinematics_d4_d6_timeseries"
        )
        if joint_kinematics_timeseries:
            artifacts[f"{trial}/kinematics/d4_d6_timeseries"] = Path(
                joint_kinematics_timeseries
            )
        for output_name in (
            "captury_c3d_angle_decode",
            "captury_c3d_angle_metrics",
            "captury_c3d_angle_timeseries",
            "metric_quality",
            "metric_sensitivity",
        ):
            output_path = report.get("outputs", {}).get(output_name)
            if output_path:
                artifacts[f"{trial}/kinematics/{output_name}"] = Path(output_path)
        for output_name in (
            "temporal_synchronization",
            "temporal_synchronization_timeseries",
            "temporal_phase_normalized",
            "phase_normalized_scientific_timeseries",
            "contact_cycles",
        ):
            output_path = report.get("outputs", {}).get(output_name)
            if output_path:
                artifacts[f"{trial}/time/{output_name}"] = Path(output_path)
        biobuddy_ik_outputs = report.get("biobuddy_ik_batch", {}).get("outputs", {})
        for output_name in ("npz", "summary"):
            output_path = biobuddy_ik_outputs.get(output_name)
            if output_path:
                artifacts[f"{trial}/biobuddy/ik_{output_name}"] = Path(output_path)
    return artifacts


def write_comparison_provenance_manifest(
    trials: list[TrialBundle],
    args: argparse.Namespace,
    reports: list[dict[str, Any]] | None = None,
    batch_artifacts: dict[str, Path] | None = None,
) -> Path:
    """Write the reproducibility manifest for a selected comparison batch."""

    manifest = build_provenance_manifest(
        input_files=comparison_input_files(trials, args),
        derived_artifacts=comparison_derived_artifacts(
            reports or [], batch_artifacts=batch_artifacts
        ),
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    output_path = args.out_dir / "provenance_manifest.json"
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def native_unit_scale_to_m(
    system: str, source_kind: str, user_scale: float | None
) -> float:
    if user_scale is not None:
        return user_scale
    if system == "motive":
        return 0.01
    return 0.001


def biorbd_segment_names(model: Any) -> list[str]:
    return [
        model.segment(index).name().to_string() for index in range(model.nbSegment())
    ]


def compute_model_segment_rotations_native(
    biomod_path: Path,
    q: np.ndarray,
    keep_segment_names: set[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract segment global rotation matrices from a biorbd model.

    ``biorbd.Model.globalJCS(q, i)`` returns the homogeneous transform of each
    segment. This helper stores the rotational ``3x3`` block for every requested
    non-root segment as an array shaped ``(3, 3, n_frames)``.
    """

    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    if q.shape[0] != model.nbQ():
        raise RuntimeError(f"{biomod_path} expects {model.nbQ()} q, got {q.shape[0]}.")
    names = biorbd_segment_names(model)
    rotations: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        if name == "root":
            continue
        if keep_segment_names is not None and name not in keep_segment_names:
            continue
        rotations[name] = np.zeros((3, 3, q.shape[1]), dtype=float)
    for frame in range(q.shape[1]):
        q_frame = np.ascontiguousarray(q[:, frame], dtype=float)
        for index, name in enumerate(names):
            if name not in rotations:
                continue
            rt = np.asarray(model.globalJCS(q_frame, index).to_array(), dtype=float)
            rotations[name][:, :, frame] = rt[:3, :3]
    return rotations


def build_model_run(
    bundle: TrialBundle,
    system: str,
    model_source: str,
    out_dir: Path,
    include_mesh: bool,
    max_mesh_points: int,
    unit_scale_override: float | None,
    root_offset_mode: str,
    model_to_c3d_axis: str | np.ndarray,
    angle_label_regex: str,
) -> ModelRun:
    source_kind, source_path = select_model_file(bundle, system, model_source)
    source_dir = out_dir / system / source_kind
    source_dir.mkdir(parents=True, exist_ok=True)
    biomod_path = source_dir / f"{system}_{source_kind}_biobuddy.bioMod"
    unit_scale_to_m = native_unit_scale_to_m(system, source_kind, unit_scale_override)
    c3d_path = bundle.captury_c3d if system == "captury" else bundle.motive_c3d
    _labels, c3d_points_mm, _residuals, c3d_time = read_c3d_points_mm(
        c3d_path, angle_label_regex
    )
    mesh_report: dict[str, Any] = {
        "mesh_file_count": 0,
        "mesh_vertices": 0,
        "mesh_faces": 0,
    }
    if source_kind == "bvh":
        _, parser = build_biomod_from_bvh_with_biobuddy(
            source_path, biomod_path, add_joint_centre_markers=True
        )
        corrected_runtime = extract_q_from_biobuddy_bvh_parser(
            parser, apply_root_offset_correction=True
        )
        uncorrected_runtime = extract_q_from_biobuddy_bvh_parser(
            parser, apply_root_offset_correction=False
        )
        joint_names = corrected_runtime.joint_names
    else:
        _, parser, mesh_report = build_biomod_from_fbx_with_biobuddy(
            source_path,
            biomod_path,
            add_joint_centre_markers=True,
            include_mesh=include_mesh,
            max_mesh_points=max_mesh_points,
        )
        corrected_runtime = extract_q_from_fbx_parser(
            parser, source_path, apply_root_offset_correction=True
        )
        uncorrected_runtime = extract_q_from_fbx_parser(
            parser, source_path, apply_root_offset_correction=False
        )
        joint_names = collect_fbx_joint_names_depth_first(parser)
        mesh_dir = biomod_path.parent / "meshes"
        if include_mesh and mesh_dir.exists():
            try:
                mesh_report = convert_biobuddy_ply_meshes_to_vtp(mesh_dir)
            except Exception:
                pass
    use_correction, root_offset_policy, centres_native = (
        choose_root_offset_policy_in_c3d(
            source_name=f"{system}_{source_kind}",
            biomod_path=biomod_path,
            corrected_q=corrected_runtime.q,
            uncorrected_q=uncorrected_runtime.q,
            q_names=corrected_runtime.q_names,
            time=corrected_runtime.time,
            joint_names=joint_names,
            unit_scale_to_m=unit_scale_to_m,
            model_to_c3d_axis=model_to_c3d_axis,
            c3d_markers_mm=c3d_points_mm,
            c3d_time=c3d_time,
            requested_mode=root_offset_mode,
            out_dir=source_dir,
        )
    )
    runtime = corrected_runtime if use_correction else uncorrected_runtime
    rotations_native = compute_model_segment_rotations_native(
        biomod_path, runtime.q, set(joint_names)
    )
    root_offset_policy.update(
        {
            "source_file": str(source_path),
            "c3d_file": str(c3d_path),
            "root_offset_native": (
                runtime.root_offset_native.tolist()
                if runtime.root_offset_native is not None
                else None
            ),
            "root_offset_correction_applied": bool(
                runtime.root_offset_correction_applied
            ),
            "source_unit_scale_to_m": unit_scale_to_m,
        }
    )
    (source_dir / f"{system}_{source_kind}_root_translation_policy.json").write_text(
        json.dumps(root_offset_policy, indent=2), encoding="utf-8"
    )
    if source_kind == "fbx":
        append_joint_centre_markers_to_biomod(
            biomod_path, joint_names, marker_prefix=f"{system.upper()}JC_"
        )
    save_q_outputs(
        runtime.q,
        runtime.q_names,
        runtime.time,
        source_dir,
        source_name=system,
        q_units=runtime.q_units,
    )
    save_model_joint_centres(centres_native, runtime.time, source_dir, system)
    return ModelRun(
        system=system,
        source_kind=source_kind,
        biomod_path=biomod_path,
        q=runtime.q,
        q_names=runtime.q_names,
        q_units=runtime.q_units,
        time=runtime.time,
        joint_names=joint_names,
        centres_native=centres_native,
        rotations_native=rotations_native,
        unit_scale_to_m=unit_scale_to_m,
        mesh_report=mesh_report,
        root_offset_policy=root_offset_policy,
    )


def _rotation_audit_timeseries_rows(
    system: str, comparison: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment, values in comparison.get("timeseries", {}).items():
        time = np.asarray(values["time"], dtype=float)
        vectors = np.asarray(values["rotation_vector_rad"], dtype=float)
        geodesic = np.asarray(values["geodesic_deg"], dtype=float)
        for frame, time_value in enumerate(time):
            rows.append(
                {
                    "system": system,
                    "segment": segment,
                    "time_s": float(time_value),
                    "deviation_x_deg": float(np.rad2deg(vectors[0, frame])),
                    "deviation_y_deg": float(np.rad2deg(vectors[1, frame])),
                    "deviation_z_deg": float(np.rad2deg(vectors[2, frame])),
                    "geodesic_deg": float(geodesic[frame]),
                }
            )
    return rows


def write_bvh_fbx_rotation_audit(
    out_dir: Path,
    system: str,
    audit: Mapping[str, Any],
    comparison: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Write a compact audit summary and compressed rotation time series."""

    system_dir = out_dir / system
    system_dir.mkdir(parents=True, exist_ok=True)
    summary_path = system_dir / "bvh_fbx_rotation_audit.json"
    timeseries_path = system_dir / "bvh_fbx_rotation_audit.npz"
    summary_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_table_npz(
        timeseries_path,
        _rotation_audit_timeseries_rows(system, comparison or {}),
    )
    return {"summary": str(summary_path), "timeseries": str(timeseries_path)}


def build_model_run_with_rotation_audit(
    bundle: TrialBundle,
    system: str,
    args: argparse.Namespace,
    trial_dir: Path,
    root_offset_mode: str | None = None,
    model_to_own_c3d_axis: str | np.ndarray | None = None,
) -> tuple[ModelRun, dict[str, Any]]:
    """Build the selected export and optionally audit BVH/FBX equivalence.

    Explicit ``bvh`` or ``fbx`` selection remains authoritative. In ``auto``
    mode, both available exports must pass the declared SO(3) gate before BVH is
    selected; a failed gate requires the user to choose a source explicitly.
    """

    audit_requested = bool(
        getattr(args, "audit_bvh_fbx_rotations", False) or args.model_source == "auto"
    )
    available = {
        "bvh": bundle.captury_bvh if system == "captury" else bundle.motive_bvh,
        "fbx": bundle.captury_fbx if system == "captury" else bundle.motive_fbx,
    }

    def build(source_kind: str) -> ModelRun:
        return build_model_run(
            bundle,
            system,
            source_kind,
            trial_dir,
            include_mesh=not args.no_mesh,
            max_mesh_points=args.max_mesh_points,
            unit_scale_override=(
                args.captury_unit_scale_to_m
                if system == "captury"
                else args.motive_unit_scale_to_m
            ),
            root_offset_mode=root_offset_mode or args.root_offset_mode,
            model_to_c3d_axis=(
                args.model_to_c3d_axis
                if model_to_own_c3d_axis is None
                else model_to_own_c3d_axis
            ),
            angle_label_regex=args.angle_label_regex,
        )

    if not audit_requested or not all(available.values()):
        selected = build(args.model_source)
        audit = {
            "status": "not_requested" if not audit_requested else "single_source_only",
            "system": system,
            "available_sources": sorted(
                source_kind
                for source_kind, path in available.items()
                if path is not None
            ),
            "selected_source": selected.source_kind,
            "blocks_automatic_source_selection": False,
            "meaning": (
                "BVH/FBX equivalence was not evaluated."
                if not audit_requested
                else "Only one model export is available; no cross-format equivalence can be tested."
            ),
        }
        if audit_requested:
            audit["artifacts"] = write_bvh_fbx_rotation_audit(trial_dir, system, audit)
        return selected, audit

    runs = {source_kind: build(source_kind) for source_kind in ("bvh", "fbx")}
    registry = load_kinematic_conventions()
    source_id = "captury_model" if system == "captury" else "motive_model"
    mapped: dict[str, dict[str, np.ndarray]] = {}
    mapping_reports: dict[str, dict[str, list[str]]] = {}
    for source_kind, run in runs.items():
        mapped[source_kind], mapping_reports[source_kind] = (
            canonicalize_segment_rotation_mapping(
                run.rotations_native,
                segment_source_names(registry, source_id, source_kind),
            )
        )
    comparison = compare_segment_rotation_mappings(
        mapped["bvh"],
        runs["bvh"].time,
        mapped["fbx"],
        runs["fbx"].time,
    )
    comparison["missing_in_reference"] = sorted(
        set(comparison["missing_in_reference"]).union(
            mapping_reports["bvh"]["missing_canonical_segments"]
        )
    )
    comparison["missing_in_test"] = sorted(
        set(comparison["missing_in_test"]).union(
            mapping_reports["fbx"]["missing_canonical_segments"]
        )
    )
    verdict = assess_rotation_source_equivalence(
        comparison,
        max_p95_geodesic_deg=float(args.bvh_fbx_max_p95_geodesic_deg),
    )
    audit = {
        "system": system,
        "reference_source": "bvh",
        "test_source": "fbx",
        "matrix_convention": "R_model_segment; columns are local axes in model coordinates",
        "mapping": mapping_reports,
        "time_alignment": comparison["time_alignment"],
        "summary": comparison["summary"],
        "verdict": verdict,
        "status": verdict["status"],
        "blocks_automatic_source_selection": verdict[
            "blocks_automatic_source_selection"
        ],
        "generated_biomods": {
            source_kind: str(run.biomod_path) for source_kind, run in runs.items()
        },
    }
    audit["artifacts"] = write_bvh_fbx_rotation_audit(
        trial_dir, system, audit, comparison
    )
    if args.model_source == "auto" and verdict["blocks_automatic_source_selection"]:
        raise RuntimeError(
            f"Automatic {system} model-source selection is blocked: BVH and FBX "
            f"are not equivalent within {args.bvh_fbx_max_p95_geodesic_deg:g} deg "
            f"p95. Select --model-source bvh or --model-source fbx explicitly; "
            f"see {audit['artifacts']['summary']}."
        )
    selected_kind = "bvh" if args.model_source == "auto" else args.model_source
    return runs[selected_kind], audit


def model_to_c3d_matrix(axis_mode: str | np.ndarray) -> np.ndarray:
    """Return one validated model-to-C3D basis matrix.

    String modes are kept for backwards-compatible single-source commands.
    The multi-system workflow resolves source-specific matrices upstream and
    passes the concrete 3x3 matrix here, avoiding a hidden shared ``auto``
    assumption for Captury and Motive.
    """

    if not isinstance(axis_mode, str):
        matrix = np.asarray(axis_mode, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError(f"Axis matrix must be 3x3, got {matrix.shape}")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-8):
            raise ValueError("Axis matrix must be orthonormal")
        if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-8):
            raise ValueError("Axis matrix must have determinant +1")
        return matrix
    if axis_mode == "auto":
        axis_mode = "y_up_to_z_up"
    if axis_mode == "identity":
        return np.eye(3)
    if axis_mode == "y_up_to_z_up":
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
    raise ValueError(f"Unsupported axis conversion: {axis_mode}")


def centres_to_c3d_mm(
    centres_native: dict[str, np.ndarray],
    unit_scale_to_m: float,
    axis_mode: str | np.ndarray,
) -> dict[str, np.ndarray]:
    matrix = model_to_c3d_matrix(axis_mode)
    factor = unit_scale_to_m * 1000.0
    return {name: matrix @ (values * factor) for name, values in centres_native.items()}


def rotations_to_c3d(
    rotations_native: dict[str, np.ndarray],
    axis_mode: str | np.ndarray,
    row_global_rotation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    matrix = model_to_c3d_matrix(axis_mode)
    if row_global_rotation is not None:
        matrix = np.asarray(row_global_rotation, dtype=float).T @ matrix
    return {
        name: change_lab_basis(values, matrix)
        for name, values in rotations_native.items()
    }


def trim_rotations(
    rotations: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[:, :, mask] for name, values in rotations.items()}


ROTATION_180_AROUND_LOCAL_X_THEN_Y = np.diag([-1.0, -1.0, 1.0])


def rotate_segment_frames_180_x(
    rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Rotate every segment frame by local ``R(x, pi) @ R(y, pi)``.

    Segment rotation matrices are stored as columns expressing local axes in the
    C3D/global frame. Right multiplication therefore changes the segment-local
    basis while preserving the global trajectory of the segment origin.
    """

    return {
        name: np.einsum("ijf,jk->ikf", values, ROTATION_180_AROUND_LOCAL_X_THEN_Y)
        for name, values in rotations.items()
    }


def _safe_unit_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    fallback_norm = float(np.linalg.norm(fallback))
    if fallback_norm > 1e-12:
        return fallback / fallback_norm
    return np.asarray([0.0, 1.0, 0.0], dtype=float)


def orient_segment_y_from_cor(
    rotations: dict[str, np.ndarray],
    centres_mm: dict[str, np.ndarray],
    segment: str,
    proximal_joint: str,
    distal_joint: str,
) -> dict[str, np.ndarray]:
    """Return rotations where ``segment`` Y axis follows proximal -> distal CoR.

    The X axis keeps the original segment orientation as much as possible by
    projecting the previous X axis onto the plane orthogonal to the corrected Y.
    This yields a right-handed orthonormal frame per frame.
    """

    if (
        segment not in rotations
        or proximal_joint not in centres_mm
        or distal_joint not in centres_mm
    ):
        return rotations

    corrected = dict(rotations)
    original = np.asarray(rotations[segment], dtype=float)
    proximal = np.asarray(centres_mm[proximal_joint], dtype=float)
    distal = np.asarray(centres_mm[distal_joint], dtype=float)
    n_frames = min(original.shape[2], proximal.shape[1], distal.shape[1])
    if n_frames <= 0:
        return rotations

    frames = original.copy()
    for frame in range(n_frames):
        y_axis = _safe_unit_vector(
            distal[:, frame] - proximal[:, frame], original[:, 1, frame]
        )
        x_axis = original[:, 0, frame]
        x_axis = x_axis - np.dot(x_axis, y_axis) * y_axis
        if np.linalg.norm(x_axis) <= 1e-12:
            x_axis = np.cross(original[:, 2, frame], y_axis)
        if np.linalg.norm(x_axis) <= 1e-12:
            helper = (
                np.asarray([1.0, 0.0, 0.0])
                if abs(y_axis[0]) < 0.9
                else np.asarray([0.0, 0.0, 1.0])
            )
            x_axis = np.cross(helper, y_axis)
        x_axis = _safe_unit_vector(x_axis, original[:, 0, frame])
        z_axis = _safe_unit_vector(np.cross(x_axis, y_axis), original[:, 2, frame])
        x_axis = _safe_unit_vector(np.cross(y_axis, z_axis), x_axis)
        frames[:, :, frame] = np.column_stack((x_axis, y_axis, z_axis))

    corrected[segment] = frames
    return corrected


def correct_captury_thigh_y_from_cor(
    rotations: dict[str, np.ndarray], centres_mm: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Orient Captury thigh Y axes from hip CoR toward knee CoR."""

    corrected = rotations
    for segment, proximal, distal in (
        ("LeftUpLeg", "LeftUpLeg", "LeftLeg"),
        ("RightUpLeg", "RightUpLeg", "RightLeg"),
    ):
        corrected = orient_segment_y_from_cor(
            corrected, centres_mm, segment, proximal, distal
        )
    return corrected


SEGMENT_RELATIVE_ROTATION_PAIRS = (
    ("SegRel_HipsSpine", "Hips", "Spine"),
    ("SegRel_LeftHip", "Hips", "LeftUpLeg"),
    ("SegRel_RightHip", "Hips", "RightUpLeg"),
    ("SegRel_LeftKnee", "LeftUpLeg", "LeftLeg"),
    ("SegRel_RightKnee", "RightUpLeg", "RightLeg"),
    ("SegRel_LeftAnkle", "LeftLeg", "LeftFoot"),
    ("SegRel_RightAnkle", "RightLeg", "RightFoot"),
    ("SegRel_LeftShoulder", "Spine3", "LeftArm"),
    ("SegRel_RightShoulder", "Spine3", "RightArm"),
    ("SegRel_LeftElbow", "LeftArm", "LeftForeArm"),
    ("SegRel_RightElbow", "RightArm", "RightForeArm"),
)


def segment_relative_rotation_curves(
    rotations: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Extract segment-relative rotation-vector curves in radians."""

    curves: dict[str, np.ndarray] = {}
    for joint, proximal, distal in SEGMENT_RELATIVE_ROTATION_PAIRS:
        if proximal not in rotations or distal not in rotations:
            continue
        n_frames = min(rotations[proximal].shape[2], rotations[distal].shape[2])
        if n_frames <= 0:
            continue
        values = np.zeros((3, n_frames), dtype=float)
        for frame in range(n_frames):
            values[:, frame] = rotation_deviation_vector(
                rotations[proximal][:, :, frame],
                rotations[distal][:, :, frame],
            )
        curves[joint] = values
    return curves


def segment_relative_q_metric_rows(
    trial: str,
    captury_rotations: dict[str, np.ndarray],
    motive_rotations: dict[str, np.ndarray],
    captury_time: np.ndarray,
    motive_time: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compare derived joint angles from corrected segment coordinate systems."""

    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    motive_curves = segment_relative_rotation_curves(motive_rotations)
    component_names = ("x", "y", "z")
    captury_curves: dict[str, np.ndarray] = {}
    for joint, proximal, distal in SEGMENT_RELATIVE_ROTATION_PAIRS:
        if (
            proximal not in captury_rotations
            or distal not in captury_rotations
            or joint not in motive_curves
        ):
            continue
        overlap = (motive_time >= captury_time[0]) & (motive_time <= captury_time[-1])
        target_time = motive_time[overlap]
        values = np.full((3, motive_time.size), np.nan, dtype=float)
        if target_time.size:
            proximal_on_motive = slerp_rotation_series(
                captury_rotations[proximal], captury_time, target_time
            )
            distal_on_motive = slerp_rotation_series(
                captury_rotations[distal], captury_time, target_time
            )
            for target_index in range(target_time.size):
                values[:, np.flatnonzero(overlap)[target_index]] = (
                    rotation_deviation_vector(
                        proximal_on_motive[:, :, target_index],
                        distal_on_motive[:, :, target_index],
                    )
                )
        captury_curves[joint] = values
    for joint in sorted(set(captury_curves).intersection(motive_curves)):
        cap_curve = captury_curves[joint]
        mot_curve = motive_curves[joint]
        n_frames = min(cap_curve.shape[1], mot_curve.shape[1], motive_time.shape[0])
        if n_frames <= 0:
            continue
        cap_curve = cap_curve[:, :n_frames]
        mot_curve = mot_curve[:, :n_frames]
        for component_index, component in enumerate(component_names):
            q_name = f"{joint}_{component}"
            reference = mot_curve[component_index]
            test = cap_curve[component_index]
            summary_rows.append(
                {
                    "trial": trial,
                    "q_name": q_name,
                    "unit": "rad",
                    "source": "segment_relative_rotation",
                    **waveform_metrics(
                        reference,
                        test,
                        "rad",
                        time=motive_time[:n_frames],
                    ),
                }
            )
            for frame, time_value in enumerate(motive_time[:n_frames]):
                timeseries_rows.append(
                    {
                        "trial": trial,
                        "time": float(time_value),
                        "q_name": q_name,
                        "source": "segment_relative_rotation",
                        "motive": float(reference[frame]),
                        "captury": float(test[frame]),
                        "difference": float(test[frame] - reference[frame]),
                    }
                )
    return summary_rows, timeseries_rows


def _source_joint_articulations(
    source_id: str,
    source_format: str,
) -> dict[str, dict[str, str]]:
    """Map registry articulations to names present in one exporter format."""

    registry = load_kinematic_conventions()
    source = registry["sources"][source_id]
    segment_names = segment_source_names(registry, source_id, source_format)
    result: dict[str, dict[str, str]] = {}
    for articulation_id, articulation in source["articulations"].items():
        proximal = articulation["proximal"]
        distal = articulation["distal"]
        if proximal == "unknown" or distal == "unknown":
            continue
        result[articulation_id] = {
            "proximal": segment_names[proximal],
            "distal": segment_names[distal],
            "proximal_segment_id": proximal,
            "distal_segment_id": distal,
        }
    return result


def nearest_time_indices(
    source_time: np.ndarray, target_time: np.ndarray
) -> np.ndarray:
    if source_time.size == 0 or target_time.size == 0:
        return np.zeros(0, dtype=int)
    indices = np.searchsorted(source_time, target_time, side="left")
    indices = np.clip(indices, 0, source_time.size - 1)
    previous = np.clip(indices - 1, 0, source_time.size - 1)
    use_previous = np.abs(target_time - source_time[previous]) < np.abs(
        target_time - source_time[indices]
    )
    indices[use_previous] = previous[use_previous]
    return indices


def rotation_deviation_vector(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Return the rotation-vector deviation that maps ``R1`` to ``R2``.

    ``R = R1.T @ R2`` and the returned vector components are expressed in
    radians around the local X/Y/Z axes of the reference orientation. The
    canonical SO(3) logarithm remains stable at rotations close to 180 degrees.
    """

    return canonical_rotation_vector(R1, R2)


def axis_rotation_matrix(axis: str, angle: float) -> np.ndarray:
    """Return a right-handed elementary rotation matrix for one axis."""

    c = float(np.cos(angle))
    s = float(np.sin(angle))
    axis = str(axis).upper()
    if axis == "X":
        return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "Y":
        return np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "Z":
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unsupported rotation axis: {axis!r}")


def euler_matrix_from_sequence(angles: np.ndarray, sequence: str) -> np.ndarray:
    """Compose an Euler rotation matrix in the explicit listed-axis order."""

    sequence = str(sequence).upper()
    if len(sequence) != 3 or set(sequence) != {"X", "Y", "Z"}:
        raise ValueError(f"Unsupported Euler sequence: {sequence!r}")
    matrix = np.eye(3)
    for axis, angle in zip(sequence, np.asarray(angles, dtype=float), strict=True):
        matrix = matrix @ axis_rotation_matrix(axis, float(angle))
    return matrix


def euler_zxy_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Extract ``Z, X, Y`` Euler angles from ``Rz @ Rx @ Ry`` matrices."""

    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {matrix.shape}.")
    x_angle = float(np.arcsin(np.clip(matrix[2, 1], -1.0, 1.0)))
    cos_x = float(np.cos(x_angle))
    if abs(cos_x) < 1e-8:
        z_angle = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        y_angle = 0.0
    else:
        z_angle = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
        y_angle = float(np.arctan2(-matrix[2, 0], matrix[2, 2]))
    return np.asarray([z_angle, x_angle, y_angle], dtype=float)


def _rotational_q_indices_by_segment(
    q_names: list[str],
) -> dict[str, dict[str, int]]:
    indices: dict[str, dict[str, int]] = {}
    for index, q_name in enumerate(q_names):
        match = re.match(r"(.+)_rot([XYZ])$", str(q_name))
        if match is None:
            continue
        segment, axis = match.groups()
        indices.setdefault(segment, {})[axis] = index
    return indices


def reexpress_rotational_q_from_segment_rotations(
    q: np.ndarray,
    q_names: list[str],
    rotations: dict[str, np.ndarray],
    target_sequence: str = ROTATION_SEQUENCE_ZXY,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Re-express rotational q channels from segment rotation matrices.

    This is intentionally applied only to Captury/Motive comparison copies. The
    BioBuddy reference q is left as produced by the IK/model pipeline.  The
    supplied ``rotations`` can already include display/convention corrections
    such as ``R(x,180 deg)``; this function only extracts the requested Euler
    sequence and writes those angles back into matching ``*_rotX/Y/Z`` rows.
    """

    target_sequence = str(target_sequence).upper()
    if target_sequence != ROTATION_SEQUENCE_ZXY:
        raise ValueError(f"Unsupported target Euler sequence: {target_sequence!r}")
    updated = np.asarray(q, dtype=float).copy()
    indices_by_segment = _rotational_q_indices_by_segment(q_names)
    changed_segments: list[str] = []
    skipped_segments: list[str] = []
    target_extractors = {ROTATION_SEQUENCE_ZXY: euler_zxy_from_matrix}
    extractor = target_extractors[target_sequence]
    for segment, axis_indices in sorted(indices_by_segment.items()):
        if set(axis_indices) != {"X", "Y", "Z"} or segment not in rotations:
            skipped_segments.append(segment)
            continue
        segment_rotations = np.asarray(rotations[segment], dtype=float)
        n_frames = min(updated.shape[1], segment_rotations.shape[2])
        if n_frames <= 0:
            skipped_segments.append(segment)
            continue
        for frame in range(n_frames):
            angles = extractor(segment_rotations[:, :, frame])
            for angle, axis in zip(angles, target_sequence, strict=True):
                updated[axis_indices[axis], frame] = float(angle)
        changed_segments.append(segment)
    return updated, {
        "target_sequence": target_sequence,
        "changed_segments": changed_segments,
        "skipped_segments": skipped_segments,
    }


def reexpress_model_run_rotational_q(
    run: ModelRun,
    rotations: dict[str, np.ndarray],
    target_sequence: str = ROTATION_SEQUENCE_ZXY,
) -> tuple[ModelRun, dict[str, Any]]:
    q, report = reexpress_rotational_q_from_segment_rotations(
        run.q, run.q_names, rotations, target_sequence
    )
    return replace(run, q=q), report


def segment_rotation_metric_rows(
    trial: str,
    rotations_by_source: dict[str, dict[str, np.ndarray]],
    times_by_source: dict[str, np.ndarray],
    reference_source: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    reference_source = reference_source.lower()
    available = sorted(
        source
        for source, rotations in rotations_by_source.items()
        if rotations and times_by_source.get(source, np.asarray([])).size
    )
    report: dict[str, Any] = {
        "requested_reference": reference_source,
        "available_sources": available,
    }
    if reference_source not in rotations_by_source or not rotations_by_source.get(
        reference_source
    ):
        report["status"] = "missing_reference"
        return [], [], report
    else:
        report["effective_reference"] = reference_source
    reference_rotations = rotations_by_source[reference_source]
    reference_time = times_by_source[reference_source]
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for source, source_rotations in sorted(rotations_by_source.items()):
        if source == reference_source or not source_rotations:
            continue
        source_time = times_by_source[source]
        overlap_mask = (reference_time >= source_time[0]) & (
            reference_time <= source_time[-1]
        )
        reference_indices = np.flatnonzero(overlap_mask)
        overlap_time = reference_time[reference_indices]
        common_segments = sorted(
            set(reference_rotations).intersection(source_rotations)
        )
        for segment in common_segments:
            reference_series, reference_quality = canonicalize_rotation_series(
                reference_rotations[segment],
                context=f"segment metric {reference_source}/{segment}",
            )
            source_series, source_quality = canonicalize_rotation_series(
                source_rotations[segment],
                context=f"segment metric {source}/{segment}",
            )
            source_on_reference = slerp_rotation_series(
                source_series, source_time, overlap_time
            )
            values: list[dict[str, Any]] = []
            for overlap_frame, reference_frame in enumerate(reference_indices):
                vector_rad = rotation_deviation_vector(
                    reference_series[:, :, reference_frame],
                    source_on_reference[:, :, overlap_frame],
                )
                vector_deg = np.degrees(vector_rad)
                global_deg = float(np.linalg.norm(vector_deg))
                row = {
                    "trial": trial,
                    "reference": reference_source,
                    "source": source,
                    "segment": segment,
                    "time": float(reference_time[reference_frame]),
                    "global_deg": global_deg,
                    "geodesic_deg": global_deg,
                    "x_deg": float(vector_deg[0]),
                    "y_deg": float(vector_deg[1]),
                    "z_deg": float(vector_deg[2]),
                    "abs_x_deg": float(abs(vector_deg[0])),
                    "abs_y_deg": float(abs(vector_deg[1])),
                    "abs_z_deg": float(abs(vector_deg[2])),
                }
                values.append(row)
                timeseries_rows.append(row)
            if not values:
                continue
            global_values = np.asarray([row["global_deg"] for row in values])
            signed_x = np.asarray([row["x_deg"] for row in values])
            signed_y = np.asarray([row["y_deg"] for row in values])
            signed_z = np.asarray([row["z_deg"] for row in values])
            abs_x = np.asarray([row["abs_x_deg"] for row in values])
            abs_y = np.asarray([row["abs_y_deg"] for row in values])
            abs_z = np.asarray([row["abs_z_deg"] for row in values])
            component_agreement = {
                axis: bland_altman(np.zeros(component.shape), component)
                for axis, component in (
                    ("x", signed_x),
                    ("y", signed_y),
                    ("z", signed_z),
                )
            }
            summary_rows.append(
                {
                    "trial": trial,
                    "reference": reference_source,
                    "source": source,
                    "segment": segment,
                    "median_global_deg": float(np.nanmedian(global_values)),
                    "p95_global_deg": float(np.nanpercentile(global_values, 95)),
                    "max_global_deg": float(np.nanmax(global_values)),
                    "median_geodesic_deg": float(np.nanmedian(global_values)),
                    "p95_geodesic_deg": float(np.nanpercentile(global_values, 95)),
                    "max_geodesic_deg": float(np.nanmax(global_values)),
                    "rmse_global_deg": float(np.sqrt(np.nanmean(global_values**2))),
                    "rmse_geodesic_deg": float(np.sqrt(np.nanmean(global_values**2))),
                    "median_abs_x_deg": float(np.nanmedian(abs_x)),
                    "median_abs_y_deg": float(np.nanmedian(abs_y)),
                    "median_abs_z_deg": float(np.nanmedian(abs_z)),
                    "p95_abs_x_deg": float(np.nanpercentile(abs_x, 95)),
                    "p95_abs_y_deg": float(np.nanpercentile(abs_y, 95)),
                    "p95_abs_z_deg": float(np.nanpercentile(abs_z, 95)),
                    **{
                        f"bias_{axis}_deg": agreement["bias"]
                        for axis, agreement in component_agreement.items()
                    },
                    **{
                        f"loa_lower_{axis}_deg": agreement["loa_lower"]
                        for axis, agreement in component_agreement.items()
                    },
                    **{
                        f"loa_upper_{axis}_deg": agreement["loa_upper"]
                        for axis, agreement in component_agreement.items()
                    },
                    "paired_frames": len(values),
                    "limits_of_agreement_scope": (
                        "descriptive_within_trial_frames_not_population_ci"
                    ),
                    "reference_max_projection_frobenius": reference_quality[
                        "max_projection_frobenius"
                    ],
                    "source_max_projection_frobenius": source_quality[
                        "max_projection_frobenius"
                    ],
                }
            )
    report["status"] = "ok" if summary_rows else "no_common_segments"
    return summary_rows, timeseries_rows, report


def root_alignment_score_mm(
    centres_c3d_mm: dict[str, np.ndarray],
    source_time: np.ndarray,
    c3d_markers_mm: np.ndarray,
    c3d_time: np.ndarray,
    max_frames: int = 120,
) -> float:
    if not centres_c3d_mm or c3d_markers_mm.size == 0:
        return float("inf")
    stacked = np.stack(list(centres_c3d_mm.values()), axis=1)
    centres_on_c3d = interpolate_finite_array(stacked, source_time, c3d_time)
    n_frames = centres_on_c3d.shape[2]
    frame_indices = np.linspace(0, n_frames - 1, min(max_frames, n_frames), dtype=int)
    frame_scores: list[float] = []
    for frame in frame_indices:
        centres = centres_on_c3d[:, :, frame].T
        markers = c3d_markers_mm[:, :, frame].T
        finite_centres = np.all(np.isfinite(centres), axis=1)
        finite_markers = np.all(np.isfinite(markers), axis=1)
        centres = centres[finite_centres]
        markers = markers[finite_markers]
        if centres.size == 0 or markers.size == 0:
            continue
        distances = np.linalg.norm(centres[:, None, :] - markers[None, :, :], axis=2)
        frame_scores.append(float(np.nanmedian(np.nanmin(distances, axis=1))))
    return float(np.nanmedian(frame_scores)) if frame_scores else float("inf")


def choose_root_offset_policy_in_c3d(
    source_name: str,
    biomod_path: Path,
    corrected_q: np.ndarray,
    uncorrected_q: np.ndarray,
    q_names: list[str],
    time: np.ndarray,
    joint_names: list[str],
    unit_scale_to_m: float,
    model_to_c3d_axis: str | np.ndarray,
    c3d_markers_mm: np.ndarray,
    c3d_time: np.ndarray,
    requested_mode: str,
    out_dir: Path,
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    corrected_centres = compute_model_joint_centres_native(
        biomod_path, corrected_q, set(joint_names)
    )
    uncorrected_centres = compute_model_joint_centres_native(
        biomod_path, uncorrected_q, set(joint_names)
    )
    corrected_score = root_alignment_score_mm(
        centres_to_c3d_mm(corrected_centres, unit_scale_to_m, model_to_c3d_axis),
        time,
        c3d_markers_mm,
        c3d_time,
    )
    uncorrected_score = root_alignment_score_mm(
        centres_to_c3d_mm(uncorrected_centres, unit_scale_to_m, model_to_c3d_axis),
        time,
        c3d_markers_mm,
        c3d_time,
    )
    if requested_mode == "subtract":
        use_correction = True
    elif requested_mode == "keep":
        use_correction = False
    else:
        use_correction = corrected_score <= uncorrected_score
    report = {
        "source": source_name,
        "requested_mode": requested_mode,
        "selected_mode": (
            "subtract_static_offset_from_root_q"
            if use_correction
            else "keep_root_q_as_file"
        ),
        "score_mm_subtract_static_offset": corrected_score,
        "score_mm_keep_file_translation": uncorrected_score,
        "score_frame": "c3d_mm_after_model_to_c3d_axis",
        "model_to_c3d_axis": (
            model_to_c3d_axis
            if isinstance(model_to_c3d_axis, str)
            else np.asarray(model_to_c3d_axis, dtype=float).tolist()
        ),
        "q_names": q_names,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{source_name}_root_translation_policy.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    selected_centres = corrected_centres if use_correction else uncorrected_centres
    return use_correction, report, selected_centres


def static_alignment(
    captury_centres_mm: dict[str, np.ndarray],
    motive_centres_mm: dict[str, np.ndarray],
    min_points: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    common = sorted(set(captury_centres_mm).intersection(motive_centres_mm))
    moving_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    used: list[str] = []
    for name in common:
        cap = np.nanmean(captury_centres_mm[name], axis=1)
        mot = np.nanmean(motive_centres_mm[name], axis=1)
        if np.all(np.isfinite(cap)) and np.all(np.isfinite(mot)):
            moving_rows.append(cap)
            reference_rows.append(mot)
            used.append(name)
    if len(used) < min_points:
        return (
            np.eye(3),
            np.zeros(3),
            {"status": "not_enough_common_centres", "used_centres": used},
        )
    reference = np.vstack(reference_rows)
    moving = np.vstack(moving_rows)
    rotation, translation = kabsch_rows(reference, moving)
    aligned = moving @ rotation + translation
    residuals = np.linalg.norm(aligned - reference, axis=1)
    return (
        rotation,
        translation,
        {
            "status": "ok",
            "used_centres": used,
            "median_static_residual_mm": float(np.nanmedian(residuals)),
            "p95_static_residual_mm": float(np.nanpercentile(residuals, 95)),
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    )


def apply_alignment(
    centres_mm: dict[str, np.ndarray], rotation: np.ndarray, translation: np.ndarray
) -> dict[str, np.ndarray]:
    aligned: dict[str, np.ndarray] = {}
    for name, values in centres_mm.items():
        rows = apply_row_alignment(values.T, rotation, translation)
        aligned[name] = rows.T
    return aligned


def yaw_rotation_rows(angle_rad: float) -> np.ndarray:
    cosine = float(np.cos(angle_rad))
    sine = float(np.sin(angle_rad))
    return np.asarray(
        [
            [cosine, sine, 0.0],
            [-sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def yaw_alignment_rows(
    reference_rows: np.ndarray,
    moving_rows: np.ndarray,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = np.asarray(reference_rows, dtype=float)
    moving = np.asarray(moving_rows, dtype=float)
    valid = np.all(np.isfinite(reference), axis=1) & np.all(np.isfinite(moving), axis=1)
    reference = reference[valid]
    moving = moving[valid]
    if reference.shape[0] < min_points:
        return (
            np.eye(3),
            np.zeros(3),
            {
                "status": "not_enough_points",
                "n_points": int(reference.shape[0]),
            },
        )

    reference_center = np.nanmedian(reference, axis=0)
    moving_center = np.nanmedian(moving, axis=0)
    reference_xy = reference[:, :2] - reference_center[:2]
    moving_xy = moving[:, :2] - moving_center[:2]
    numerator = float(
        np.nansum(moving_xy[:, 0] * reference_xy[:, 1])
        - np.nansum(moving_xy[:, 1] * reference_xy[:, 0])
    )
    denominator = float(
        np.nansum(moving_xy[:, 0] * reference_xy[:, 0])
        + np.nansum(moving_xy[:, 1] * reference_xy[:, 1])
    )
    angle_rad = float(np.arctan2(numerator, denominator))
    rotation = yaw_rotation_rows(angle_rad)
    rotated = moving @ rotation
    translation = np.nanmedian(reference - rotated, axis=0)
    residuals = np.linalg.norm(rotated + translation - reference, axis=1)
    return (
        rotation,
        translation,
        {
            "status": "ok",
            "n_points": int(reference.shape[0]),
            "yaw_deg": float(np.degrees(angle_rad)),
            "median_residual_mm": float(np.nanmedian(residuals)),
            "p95_residual_mm": float(np.nanpercentile(residuals, 95)),
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    )


def horizontal_principal_axis_rows(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=float)
    values = values[np.all(np.isfinite(values), axis=1)]
    if values.shape[0] < 3:
        return np.asarray((1.0, 0.0), dtype=float)
    centered = values[:, :2] - np.nanmean(values[:, :2], axis=0)
    covariance = centered.T @ centered / max(1, centered.shape[0] - 1)
    _values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.asarray((1.0, 0.0), dtype=float)
    return axis / norm


def nearest_distance_score(
    reference_rows: np.ndarray, moving_rows: np.ndarray
) -> float:
    reference = np.asarray(reference_rows, dtype=float)
    moving = np.asarray(moving_rows, dtype=float)
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    moving = moving[np.all(np.isfinite(moving), axis=1)]
    if reference.size == 0 or moving.size == 0:
        return float("inf")
    distances = np.linalg.norm(moving[:, None, :] - reference[None, :, :], axis=2)
    return float(np.nanmean(np.nanmin(distances, axis=1)))


def pca_yaw_alignment_rows(
    reference_rows: np.ndarray,
    moving_rows: np.ndarray,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    reference = np.asarray(reference_rows, dtype=float)
    moving = np.asarray(moving_rows, dtype=float)
    reference = reference[np.all(np.isfinite(reference), axis=1)]
    moving = moving[np.all(np.isfinite(moving), axis=1)]
    if reference.shape[0] < min_points or moving.shape[0] < min_points:
        return (
            np.eye(3),
            np.zeros(3),
            {
                "status": "not_enough_points",
                "reference_points": int(reference.shape[0]),
                "moving_points": int(moving.shape[0]),
            },
        )
    reference_axis = horizontal_principal_axis_rows(reference)
    moving_axis = horizontal_principal_axis_rows(moving)
    base_angle = float(
        np.arctan2(
            moving_axis[0] * reference_axis[1] - moving_axis[1] * reference_axis[0],
            np.dot(moving_axis, reference_axis),
        )
    )
    candidates: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    for angle_rad in (base_angle, base_angle + np.pi):
        rotation = yaw_rotation_rows(angle_rad)
        rotated = moving @ rotation
        translation = np.nanmedian(reference, axis=0) - np.nanmedian(rotated, axis=0)
        score = nearest_distance_score(reference, rotated + translation)
        candidates.append((score, rotation, translation, angle_rad))
    score, rotation, translation, angle_rad = min(candidates, key=lambda item: item[0])
    return (
        rotation,
        translation,
        {
            "status": "ok_pca_fallback",
            "reference_points": int(reference.shape[0]),
            "moving_points": int(moving.shape[0]),
            "yaw_deg": float(np.degrees(angle_rad)),
            "nearest_distance_mm": score,
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        },
    )


def interpolate_centres_to_time(
    centres_mm: dict[str, np.ndarray], source_time: np.ndarray, target_time: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: interpolate_finite_array(values, source_time, target_time)
        for name, values in centres_mm.items()
    }


def time_window_mask(
    time: np.ndarray, start_s: float | None, end_s: float | None
) -> np.ndarray:
    mask = np.ones(time.shape[0], dtype=bool)
    if start_s is not None:
        mask &= time >= float(start_s)
    if end_s is not None:
        mask &= time <= float(end_s)
    if not np.any(mask):
        raise ValueError(
            f"Empty time window start={start_s!r}, end={end_s!r} for "
            f"signal spanning {float(time[0]) if time.size else np.nan:.3f} to "
            f"{float(time[-1]) if time.size else np.nan:.3f} s."
        )
    return mask


def trim_centres(
    centres: dict[str, np.ndarray], mask: np.ndarray
) -> dict[str, np.ndarray]:
    return {name: values[:, mask] for name, values in centres.items()}


def trim_model_run(
    run: ModelRun, start_s: float | None, end_s: float | None
) -> ModelRun:
    if start_s is None and end_s is None:
        return run
    mask = time_window_mask(run.time, start_s, end_s)
    return replace(
        run,
        q=run.q[:, mask],
        time=run.time[mask],
        centres_native=trim_centres(run.centres_native, mask),
        rotations_native=trim_rotations(run.rotations_native, mask),
    )


def resolve_cut_window(
    cut_mode: str,
    manual_start_s: float | None,
    manual_end_s: float | None,
    event_report: dict[str, Any],
) -> tuple[float | None, float | None, str]:
    if cut_mode == "full":
        return None, None, "full"
    if cut_mode == "movement":
        return (
            float(event_report["movement_start_time"]),
            float(event_report["movement_end_time"]),
            "movement",
        )
    if manual_start_s is not None and manual_end_s is not None:
        if manual_start_s > manual_end_s:
            raise ValueError(
                f"Manual time window start ({manual_start_s}) is after end ({manual_end_s})."
            )
    if manual_start_s is None and manual_end_s is None:
        return None, None, "full"
    return manual_start_s, manual_end_s, "manual"


def append_centres_to_motive_c3d(
    motive_c3d_path: Path,
    output_path: Path,
    captury_centres_mm: dict[str, np.ndarray],
    motive_centres_mm: dict[str, np.ndarray],
    captury_time: np.ndarray,
    motive_time: np.ndarray,
    angle_label_regex: str,
) -> Path:
    split = split_c3d_points(
        motive_c3d_path, bvh_unit_scale_to_m=0.01, angle_label_regex=angle_label_regex
    )
    c3d_copy = clone_c3d_dict(split.c3d)
    cap_on_motive = interpolate_centres_to_time(
        captury_centres_mm, captury_time, split.time
    )
    mot_on_motive = interpolate_centres_to_time(
        motive_centres_mm, motive_time, split.time
    )
    old_points = np.asarray(c3d_copy["data"]["points"], dtype=float)
    old_labels = as_str_list(get_c3d_param(c3d_copy, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d_copy, "POINT", "UNITS", [""]))[0]
    )

    labels: list[str] = []
    blocks: list[np.ndarray] = []
    for prefix, centres in (("CAPJC_", cap_on_motive), ("MOTJC_", mot_on_motive)):
        for name in sorted(centres):
            labels.append(f"{prefix}{name}")
            blocks.append(centres[name] / unit_mm)
    if blocks:
        xyz = np.stack(blocks, axis=1)
        residuals = np.zeros((1, xyz.shape[1], xyz.shape[2]), dtype=float)
        old_points = np.concatenate(
            (old_points, np.concatenate((xyz, residuals), axis=0)), axis=1
        )
    c3d_copy["data"]["points"] = old_points
    c3d_copy["parameters"]["POINT"]["LABELS"]["value"] = old_labels + labels
    descriptions = as_str_list(get_c3d_param(c3d_copy, "POINT", "DESCRIPTIONS", []))
    if len(descriptions) < len(old_labels):
        descriptions += [""] * (len(old_labels) - len(descriptions))
    descriptions += [
        "Joint centre generated from BioBuddy model and transformed to Motive C3D axes"
    ] * len(labels)
    c3d_copy["parameters"]["POINT"]["DESCRIPTIONS"]["value"] = descriptions
    c3d_copy["parameters"]["POINT"]["USED"]["value"] = [len(old_labels) + len(labels)]
    if "meta_points" in c3d_copy.get("data", {}):
        del c3d_copy["data"]["meta_points"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c3d_copy.write(str(output_path))
    return output_path


def centre_metric_rows(
    trial: str,
    captury_centres_mm: dict[str, np.ndarray],
    motive_centres_mm: dict[str, np.ndarray],
    captury_time: np.ndarray,
    motive_time: np.ndarray,
    joint_filters: list[str] | None = None,
    excluded_joints: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    cap_on_motive = interpolate_centres_to_time(
        captury_centres_mm, captury_time, motive_time
    )
    excluded = excluded_joints or set()
    common = sorted(
        set(cap_on_motive).intersection(motive_centres_mm).difference(excluded)
    )
    metric_joints = common
    if joint_filters:
        import re

        regexes = [re.compile(pattern) for pattern in joint_filters]
        metric_joints = [
            joint for joint in common if any(regex.search(joint) for regex in regexes)
        ]
    metric_joint_set = set(metric_joints)
    for joint in common:
        cap = cap_on_motive[joint].T
        mot = motive_centres_mm[joint].T
        valid = np.all(np.isfinite(cap), axis=1) & np.all(np.isfinite(mot), axis=1)
        errors = np.linalg.norm(cap - mot, axis=1)
        if joint in metric_joint_set:
            summary_rows.append(
                {
                    "trial": trial,
                    "joint": joint,
                    "n_frames": int(valid.sum()),
                    "median_error_mm": (
                        float(np.nanmedian(errors[valid])) if valid.any() else np.nan
                    ),
                    "p95_error_mm": (
                        float(np.nanpercentile(errors[valid], 95))
                        if valid.any()
                        else np.nan
                    ),
                    "max_error_mm": (
                        float(np.nanmax(errors[valid])) if valid.any() else np.nan
                    ),
                    **joint_center_error_xyz(mot, cap),
                }
            )
        for i, time_value in enumerate(motive_time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "joint": joint,
                    "captury_x_mm": cap[i, 0],
                    "captury_y_mm": cap[i, 1],
                    "captury_z_mm": cap[i, 2],
                    "motive_x_mm": mot[i, 0],
                    "motive_y_mm": mot[i, 1],
                    "motive_z_mm": mot[i, 2],
                    "distance_mm": errors[i],
                }
            )
    return summary_rows, timeseries_rows


def add_biobuddy_centre_rows(
    trial: str,
    summary_rows: list[dict[str, Any]],
    timeseries_rows: list[dict[str, Any]],
    motive_centres_mm: dict[str, np.ndarray],
    motive_time: np.ndarray,
    biobuddy_centres_mm: dict[str, np.ndarray],
    biobuddy_time: np.ndarray,
    joint_filters: list[str] | None = None,
) -> None:
    """Add BioBuddy IK centre columns and summary rows in place."""

    bio_on_motive = interpolate_centres_to_time(
        biobuddy_centres_mm, biobuddy_time, motive_time
    )
    common = sorted(set(bio_on_motive).intersection(motive_centres_mm))
    metric_joints = common
    if joint_filters:
        regexes = [re.compile(pattern) for pattern in joint_filters]
        metric_joints = [
            joint for joint in common if any(regex.search(joint) for regex in regexes)
        ]
    metric_joint_set = set(metric_joints)
    row_by_joint_frame: dict[tuple[str, int], dict[str, Any]] = {}
    counts_by_joint: dict[str, int] = {}
    for row in timeseries_rows:
        joint = str(row["joint"])
        frame_index = counts_by_joint.get(joint, 0)
        row_by_joint_frame[(joint, frame_index)] = row
        counts_by_joint[joint] = frame_index + 1
    for joint in common:
        bio = bio_on_motive[joint].T
        mot = motive_centres_mm[joint].T
        valid = np.all(np.isfinite(bio), axis=1) & np.all(np.isfinite(mot), axis=1)
        errors = np.linalg.norm(bio - mot, axis=1)
        if joint in metric_joint_set:
            summary_rows.append(
                {
                    "trial": trial,
                    "joint": joint,
                    "source": "biobuddy",
                    "n_frames": int(valid.sum()),
                    "median_error_mm": (
                        float(np.nanmedian(errors[valid])) if valid.any() else np.nan
                    ),
                    "p95_error_mm": (
                        float(np.nanpercentile(errors[valid], 95))
                        if valid.any()
                        else np.nan
                    ),
                    "max_error_mm": (
                        float(np.nanmax(errors[valid])) if valid.any() else np.nan
                    ),
                    **joint_center_error_xyz(mot, bio),
                }
            )
        for i, time_value in enumerate(motive_time):
            row = row_by_joint_frame.get((joint, i))
            if row is None:
                row = {"trial": trial, "time": float(time_value), "joint": joint}
                timeseries_rows.append(row)
            row["biobuddy_x_mm"] = bio[i, 0]
            row["biobuddy_y_mm"] = bio[i, 1]
            row["biobuddy_z_mm"] = bio[i, 2]
            row["biobuddy_distance_mm"] = errors[i]


def q_metric_rows(
    trial: str, captury: ModelRun, motive: ModelRun
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    cap_q = {name: captury.q[i] for i, name in enumerate(captury.q_names)}
    mot_q = {name: motive.q[i] for i, name in enumerate(motive.q_names)}
    for q_name in sorted(set(cap_q).intersection(mot_q)):
        cap_curve = interpolate_finite_array(
            cap_q[q_name][None, :], captury.time, motive.time
        )[0]
        mot_curve = mot_q[q_name]
        unit = "rad" if "rot" in q_name.lower() else "native"
        summary_rows.append(
            {
                "trial": trial,
                "q_name": q_name,
                "unit": unit,
                "source": "captury",
                **waveform_metrics(mot_curve, cap_curve, unit, time=motive.time),
            }
        )
        for i, time_value in enumerate(motive.time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "q_name": q_name,
                    "source": "captury",
                    "motive": mot_curve[i],
                    "captury": cap_curve[i],
                    "difference": cap_curve[i] - mot_curve[i],
                }
            )
    return summary_rows, timeseries_rows


def q_metric_rows_with_optional_biobuddy(
    trial: str, captury: ModelRun, motive: ModelRun, biobuddy: ModelRun | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows, timeseries_rows = q_metric_rows(trial, captury, motive)
    if biobuddy is None:
        return summary_rows, timeseries_rows
    bio_q = {name: biobuddy.q[i] for i, name in enumerate(biobuddy.q_names)}
    mot_q = {name: motive.q[i] for i, name in enumerate(motive.q_names)}
    for q_name in sorted(set(bio_q).intersection(mot_q)):
        bio_curve = interpolate_finite_array(
            bio_q[q_name][None, :], biobuddy.time, motive.time
        )[0]
        mot_curve = mot_q[q_name]
        unit = "rad" if "rot" in q_name.lower() else "native"
        summary_rows.append(
            {
                "trial": trial,
                "q_name": q_name,
                "unit": unit,
                "source": "biobuddy_ik",
                **waveform_metrics(mot_curve, bio_curve, unit, time=motive.time),
            }
        )
        for i, time_value in enumerate(motive.time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "q_name": q_name,
                    "source": "biobuddy_ik",
                    "motive": mot_curve[i],
                    "captury": np.nan,
                    "biobuddy": bio_curve[i],
                    "difference": bio_curve[i] - mot_curve[i],
                }
            )
    return summary_rows, timeseries_rows


def c3d_angle_scale_to_deg(unit: str) -> float:
    normalized = str(unit).strip().lower()
    if normalized in {"rad", "radian", "radians"}:
        return 180.0 / np.pi
    return 1.0


def sanitize_channel_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"\W+", "_", str(name).strip()).strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def captury_c3d_angle_rows(
    trial: str,
    captury_c3d: Path,
    angle_label_regex: str,
    c3d_angle_unit: str,
    cut_start_s: float | None,
    cut_end_s: float | None,
    temporal_lag_s: float = 0.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    split = split_c3d_points(
        captury_c3d,
        bvh_unit_scale_to_m=0.001,
        angle_label_regex=angle_label_regex,
    )
    decode_report = analyze_captury_angle_channels(
        split.c3d, requested_unit=c3d_angle_unit
    )
    if not split.angle_labels or split.angle_data.size == 0:
        return [], [], decode_report
    corrected_time = apply_time_offset(split.time, temporal_lag_s)
    mask = time_window_mask(corrected_time, cut_start_s, cut_end_s)
    time = corrected_time[mask]
    angle_deg = split.angle_data[:, :, mask] * float(
        decode_report["value_scale_to_deg"]
    )
    summary_rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    axis_names = ("X", "Y", "Z")
    channel_by_point_index = {
        int(channel["point_index"]): channel
        for channel in decode_report["channels"]
        if channel["point_index"] is not None
    }
    for angle_index, angle_label in enumerate(split.angle_labels):
        source_point_index = split.angle_indices[angle_index]
        decoded_channel = channel_by_point_index.get(source_point_index, {})
        safe_label = sanitize_channel_name(angle_label, f"angle_{angle_index}")
        for axis_index, axis_name in enumerate(axis_names):
            values = angle_deg[axis_index, angle_index, :]
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                continue
            q_name = f"CapturyC3D_{safe_label}_{axis_name}"
            summary_rows.append(
                {
                    "trial": trial,
                    "q_name": q_name,
                    "unit": "deg",
                    "source": "captury_c3d",
                    "c3d_angle_label": angle_label,
                    "c3d_angle_axis": axis_name,
                    "articulation": decoded_channel.get("articulation"),
                    "identity_decoded": bool(
                        decoded_channel.get("identity_decoded", False)
                    ),
                    "component_semantics_decoded": False,
                    "eligible_for_anatomical_agreement": False,
                    "angle_unit_source": decode_report["unit"].get("source"),
                    "c3d_mean_deg": float(np.mean(finite)),
                    "c3d_sd_deg": float(np.std(finite)),
                    "c3d_min_deg": float(np.min(finite)),
                    "c3d_max_deg": float(np.max(finite)),
                }
            )
            for frame_index, time_value in enumerate(time):
                timeseries_rows.append(
                    {
                        "trial": trial,
                        "time": float(time_value),
                        "q_name": q_name,
                        "source": "captury_c3d",
                        "captury_c3d": float(values[frame_index]),
                        "articulation": decoded_channel.get("articulation"),
                        "source_component": axis_name,
                        "component_semantics_decoded": False,
                        "eligible_for_anatomical_agreement": False,
                    }
                )
    return summary_rows, timeseries_rows, decode_report


def c3d_angle_inventory(path: Path, angle_label_regex: str) -> dict[str, Any]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    angle_indices = detect_angle_indices(c3d, labels, angle_label_regex)
    return {
        "path": str(path),
        "angle_count": len(angle_indices),
        "angles": {name: labels[index] for name, index in angle_indices.items()},
    }


def duplicate_label_inventory(path: Path) -> dict[str, Any]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    duplicates = {label: count for label, count in counts.items() if count > 1}
    return {
        "path": str(path),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
    }


def read_c3d_points_mm(
    path: Path,
    angle_label_regex: str = ANGLE_LABEL_REGEX,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(path))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    angle_indices = set(detect_angle_indices(c3d, labels, angle_label_regex).values())
    marker_indices = [
        index for index in range(len(labels)) if index not in angle_indices
    ]
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points = np.asarray(c3d["data"]["points"], dtype=float)
    xyz_mm = points[:3, marker_indices, :] * unit_mm
    residuals = (
        points[3, marker_indices, :]
        if points.shape[0] > 3
        else np.zeros((len(marker_indices), points.shape[2]))
    )
    xyz_mm[:, residuals < 0] = np.nan
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [120])
    rate = float(
        rate_value[0]
        if isinstance(rate_value, (list, tuple, np.ndarray))
        else rate_value
    )
    time = np.arange(xyz_mm.shape[2], dtype=float) / rate
    return [labels[index] for index in marker_indices], xyz_mm, residuals, time


def clean_marker_label(label: str) -> str:
    return display_marker_name(label)


def marker_indices_by_clean_label(labels: list[str]) -> dict[str, list[int]]:
    return marker_indices_by_display_label(labels)


def average_marker_group(
    points_mm: np.ndarray, indices: list[int]
) -> np.ndarray | None:
    if not indices:
        return None
    values = points_mm[:, indices, :]
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=1)


def marker_proxy_centres_from_c3d(
    labels: list[str], points_mm: np.ndarray
) -> dict[str, np.ndarray]:
    lookup = marker_indices_by_clean_label(labels)
    proxies: dict[str, np.ndarray] = {}
    for joint, marker_labels in MODEL_JOINT_MARKER_PROXIES.items():
        indices = [
            index
            for marker_label in marker_labels
            for index in lookup.get(marker_label, [])
        ]
        signal = average_marker_group(points_mm, indices)
        if signal is not None:
            proxies[joint] = signal
    return proxies


def paired_model_marker_rows(
    model_centres_mm: dict[str, np.ndarray],
    model_time: np.ndarray,
    marker_proxy_centres_mm: dict[str, np.ndarray],
    marker_time: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    moving_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    used_joints: list[str] = []
    for joint in sorted(set(model_centres_mm).intersection(marker_proxy_centres_mm)):
        model_signal = interpolate_finite_array(
            model_centres_mm[joint], model_time, marker_time
        ).T
        marker_signal = marker_proxy_centres_mm[joint].T
        valid = np.all(np.isfinite(model_signal), axis=1) & np.all(
            np.isfinite(marker_signal), axis=1
        )
        if not np.any(valid):
            continue
        moving_rows.append(model_signal[valid])
        reference_rows.append(marker_signal[valid])
        used_joints.append(joint)
    if not moving_rows:
        return np.empty((0, 3)), np.empty((0, 3)), []
    return np.vstack(reference_rows), np.vstack(moving_rows), used_joints


def stacked_finite_rows_from_centres(
    centres_mm: dict[str, np.ndarray],
    time: np.ndarray,
    reference_time: np.ndarray,
    max_rows: int = 2000,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for values in centres_mm.values():
        interpolated = interpolate_finite_array(values, time, reference_time).T
        interpolated = interpolated[np.all(np.isfinite(interpolated), axis=1)]
        if interpolated.size:
            rows.append(interpolated)
    if not rows:
        return np.empty((0, 3))
    stacked = np.vstack(rows)
    if stacked.shape[0] > max_rows:
        step = int(np.ceil(stacked.shape[0] / max_rows))
        stacked = stacked[::step]
    return stacked


def stacked_finite_rows_from_marker_points(
    points_mm: np.ndarray, max_rows: int = 4000
) -> np.ndarray:
    rows = np.asarray(points_mm, dtype=float).transpose(2, 1, 0).reshape(-1, 3)
    rows = rows[np.all(np.isfinite(rows), axis=1)]
    if rows.shape[0] > max_rows:
        step = int(np.ceil(rows.shape[0] / max_rows))
        rows = rows[::step]
    return rows


def model_to_motive_marker_alignment(
    model_centres_mm: dict[str, np.ndarray],
    model_time: np.ndarray,
    motive_c3d: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    labels, marker_points_mm, residuals, marker_time = read_c3d_points_mm(motive_c3d)
    if residuals.shape == marker_points_mm.shape[1:]:
        marker_points_mm = marker_points_mm.copy()
        marker_points_mm[:, residuals < 0] = np.nan
    marker_proxies = marker_proxy_centres_from_c3d(labels, marker_points_mm)
    reference_rows, moving_rows, used_joints = paired_model_marker_rows(
        model_centres_mm, model_time, marker_proxies, marker_time
    )
    rotation, translation, report = yaw_alignment_rows(reference_rows, moving_rows)
    report["used_proxy_joints"] = used_joints
    report["method"] = "motive_57_marker_proxies"
    if report.get("status") == "ok":
        return rotation, translation, report

    marker_rows = stacked_finite_rows_from_marker_points(marker_points_mm)
    model_rows = stacked_finite_rows_from_centres(
        model_centres_mm, model_time, marker_time
    )
    rotation, translation, fallback_report = pca_yaw_alignment_rows(
        marker_rows, model_rows
    )
    fallback_report["used_proxy_joints"] = used_joints
    fallback_report["method"] = "horizontal_pca_fallback"
    fallback_report["proxy_status"] = report
    return rotation, translation, fallback_report


def occlusion_rows_from_points(
    trial: str, labels: list[str], points_mm: np.ndarray, residuals: np.ndarray
) -> list[dict[str, Any]]:
    finite_xyz = np.all(np.isfinite(points_mm), axis=0)
    missing = ~finite_xyz
    if residuals.shape == missing.shape:
        missing = missing | (residuals < 0)
    rows: list[dict[str, Any]] = []
    for i, label in enumerate(labels):
        marker_missing = missing[i]
        rows.append(
            {
                "trial": trial,
                "marker_order": i,
                "marker": clean_marker_label(label),
                "raw_marker": label,
                "missing_frames": int(np.sum(marker_missing)),
                "total_frames": int(marker_missing.shape[0]),
                "missing_percent": float(100.0 * np.mean(marker_missing)),
            }
        )
    return rows


def analyze_motive_occlusions(
    motive_c3d: Path,
    trial_dir: Path,
    trial: str,
    generate_figure: bool = True,
) -> tuple[list[dict[str, Any]], Path | None]:
    labels, points_mm, residuals, _ = read_c3d_points_mm(motive_c3d)
    rows = occlusion_rows_from_points(trial, labels, points_mm, residuals)
    csv_path = trial_dir / "motive_marker_occlusions.csv"
    write_rows(csv_path, rows)
    if not generate_figure:
        return rows, None
    fig_path = plot_metric_barh(
        pd.DataFrame(rows),
        category="marker",
        metric="missing_percent",
        output_path=trial_dir / "figures" / "occlusions" / "motive_missing_percent.png",
        title="Motive marker occlusions",
        xlabel="missing_percent",
    )
    return rows, fig_path


def foot_contact_from_markers(
    points_mm: np.ndarray, indices: list[int], dt: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Detect kinematic contact, or report unavailable when no foot exists."""

    foot = average_marker_group(points_mm, indices)
    n_frames = points_mm.shape[2]
    if foot is None:
        nan = np.full(n_frames, np.nan)
        return np.zeros(n_frames, dtype=bool), nan, nan, False
    z = foot[2]
    foot_speed = np.linalg.norm(np.gradient(foot, dt, axis=1), axis=0)
    finite = np.isfinite(z) & np.isfinite(foot_speed)
    if np.count_nonzero(finite) < 3:
        return np.zeros(n_frames, dtype=bool), z, foot_speed, False
    z_limit = float(np.nanpercentile(z, 35))
    speed_limit = float(np.nanpercentile(foot_speed, 35))
    contact = finite & (z <= z_limit) & (foot_speed <= speed_limit)
    return contact, z, foot_speed, True


def detect_trial_events_and_contacts(
    motive_c3d: Path,
    trial_dir: Path,
    trial: str,
    foot_marker_regex: str = FOOT_MARKER_PATTERN,
    time_start_s: float | None = None,
    time_end_s: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import re

    labels, points_mm, _residuals, time = read_c3d_points_mm(motive_c3d)
    finite = np.all(np.isfinite(points_mm), axis=0)
    dt = float(np.nanmedian(np.diff(time))) if time.shape[0] > 1 else 1.0 / 120.0
    velocity = np.gradient(points_mm, dt, axis=2)
    speed = np.linalg.norm(velocity, axis=0)
    speed[~finite] = np.nan
    median_speed = np.nanmedian(speed, axis=0)
    baseline = float(np.nanpercentile(median_speed, 10))
    high = float(np.nanpercentile(median_speed, 95))
    threshold = baseline + 0.15 * (high - baseline)
    moving = median_speed > threshold
    moving_indices = np.flatnonzero(moving)
    start_index = int(moving_indices[0]) if moving_indices.size else 0
    end_index = (
        int(moving_indices[-1]) if moving_indices.size else int(time.shape[0] - 1)
    )

    regex = re.compile(foot_marker_regex, re.IGNORECASE)
    left_indices = [
        i
        for i, label in enumerate(labels)
        if regex.search(clean_marker_label(label))
        and clean_marker_label(label).startswith("L")
    ]
    right_indices = [
        i
        for i, label in enumerate(labels)
        if regex.search(clean_marker_label(label))
        and clean_marker_label(label).startswith("R")
    ]

    left_contact, left_z, left_speed, left_available = foot_contact_from_markers(
        points_mm, left_indices, dt
    )
    right_contact, right_z, right_speed, right_available = foot_contact_from_markers(
        points_mm, right_indices, dt
    )
    rows: list[dict[str, Any]] = []
    contact_mask = time_window_mask(time, time_start_s, time_end_s)
    for i, time_value in enumerate(time):
        if not contact_mask[i]:
            continue
        rows.append(
            {
                "trial": trial,
                "time": float(time_value),
                "movement_speed_mm_s": float(median_speed[i]),
                "left_foot_z_mm": (
                    float(left_z[i]) if np.isfinite(left_z[i]) else np.nan
                ),
                "right_foot_z_mm": (
                    float(right_z[i]) if np.isfinite(right_z[i]) else np.nan
                ),
                "left_foot_speed_mm_s": (
                    float(left_speed[i]) if np.isfinite(left_speed[i]) else np.nan
                ),
                "right_foot_speed_mm_s": (
                    float(right_speed[i]) if np.isfinite(right_speed[i]) else np.nan
                ),
                "left_contact": bool(left_contact[i]),
                "right_contact": bool(right_contact[i]),
                "left_contact_available": left_available,
                "right_contact_available": right_available,
            }
        )
    write_rows(trial_dir / "trial_events_contacts.csv", rows)
    report = {
        "trial": trial,
        "movement_start_index": start_index,
        "movement_end_index": end_index,
        "movement_start_time": float(time[start_index]),
        "movement_end_time": float(time[end_index]),
        "movement_speed_threshold_mm_s": threshold,
        "manual_time_start_s": time_start_s,
        "manual_time_end_s": time_end_s,
        "used_start_time": (
            float(time[contact_mask][0]) if np.any(contact_mask) else np.nan
        ),
        "used_end_time": (
            float(time[contact_mask][-1]) if np.any(contact_mask) else np.nan
        ),
        "left_foot_markers": [labels[i] for i in left_indices],
        "right_foot_markers": [labels[i] for i in right_indices],
        "left_contact_available": left_available,
        "right_contact_available": right_available,
    }
    (trial_dir / "trial_events.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, rows


def contact_cycle_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return diagnostic foot-contact cycles from Motive kinematic contacts."""

    cycles: list[dict[str, Any]] = []
    if not rows:
        return {"status": "unavailable", "cycles": cycles}
    times = np.asarray([float(row["time"]) for row in rows], dtype=float)
    available_sides: list[str] = []
    minimum_cycle_duration_s = 0.2
    for side in ("left", "right"):
        if not any(bool(row.get(f"{side}_contact_available", True)) for row in rows):
            continue
        available_sides.append(side)
        contacts = np.asarray([bool(row[f"{side}_contact"]) for row in rows])
        starts = np.flatnonzero(contacts & np.r_[True, ~contacts[:-1]])
        for cycle_index, (first, second) in enumerate(zip(starts[:-1], starts[1:])):
            duration_s = float(times[second] - times[first])
            if duration_s < minimum_cycle_duration_s:
                continue
            cycles.append(
                {
                    "side": side,
                    "cycle_index": cycle_index,
                    "start_s": float(times[first]),
                    "end_s": float(times[second]),
                    "duration_s": duration_s,
                    "definition": "successive_kinematic_contact_onsets",
                }
            )
    return {
        "status": (
            "unavailable"
            if not available_sides
            else "diagnostic_only" if cycles else "no_complete_cycle"
        ),
        "method": "foot_marker_height_and_speed_contact_onsets",
        "force_plate_validated": False,
        "available_sides": available_sides,
        "minimum_cycle_duration_s": minimum_cycle_duration_s,
        "cycles": cycles,
    }


SEGMENT_LENGTH_PAIR_CANDIDATES = [
    (
        "pelvis_to_spine",
        [("Hips", "Spine"), ("pelvis", "spine_01"), ("Pelvis", "Thorax")],
    ),
    (
        "left_thigh",
        [("LeftUpLeg", "LeftLeg"), ("thigh_l", "calf_l"), ("LThigh", "LShank")],
    ),
    (
        "left_shank",
        [("LeftLeg", "LeftFoot"), ("calf_l", "foot_l"), ("LShank", "LFoot")],
    ),
    ("left_foot", [("LeftFoot", "LeftToeBase"), ("foot_l", "ball_l")]),
    (
        "right_thigh",
        [("RightUpLeg", "RightLeg"), ("thigh_r", "calf_r"), ("RThigh", "RShank")],
    ),
    (
        "right_shank",
        [("RightLeg", "RightFoot"), ("calf_r", "foot_r"), ("RShank", "RFoot")],
    ),
    ("right_foot", [("RightFoot", "RightToeBase"), ("foot_r", "ball_r")]),
    (
        "left_upper_arm",
        [
            ("LeftArm", "LeftForeArm"),
            ("upperarm_l", "lowerarm_l"),
            ("LUpperArm", "LForearm"),
        ],
    ),
    (
        "left_forearm",
        [("LeftForeArm", "LeftHand"), ("lowerarm_l", "hand_l"), ("LForearm", "LHand")],
    ),
    (
        "right_upper_arm",
        [
            ("RightArm", "RightForeArm"),
            ("upperarm_r", "lowerarm_r"),
            ("RUpperArm", "RForearm"),
        ],
    ),
    (
        "right_forearm",
        [
            ("RightForeArm", "RightHand"),
            ("lowerarm_r", "hand_r"),
            ("RForearm", "RHand"),
        ],
    ),
]
SEGMENT_LENGTH_PAIRS = [
    (name, candidates[0][0], candidates[0][1])
    for name, candidates in SEGMENT_LENGTH_PAIR_CANDIDATES
]


def segment_length_pair_for_centres(
    centres_mm: Mapping[str, np.ndarray], candidates: list[tuple[str, str]]
) -> tuple[str, str] | None:
    """Choose the first segment-name pair available in a centre dictionary."""

    for proximal, distal in candidates:
        if proximal in centres_mm and distal in centres_mm:
            return proximal, distal
    return None


def dimension_rows_from_centres(
    trial: str,
    system: str,
    source_kind: str,
    centres_mm: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Summarize segment lengths from joint-centre positions in millimetres.

    The comparison GUI presents model dimensions as one row per anatomical
    segment and source. Captury/Motive dimensions come from animated model
    centres over time, while the BioBuddy template currently contributes a
    neutral-pose model. Both cases share this helper: arrays can contain one
    frame or many frames, and the median/standard deviation are computed over
    the available samples.
    """

    rows: list[dict[str, Any]] = []
    for name, candidates in SEGMENT_LENGTH_PAIR_CANDIDATES:
        pair = segment_length_pair_for_centres(centres_mm, candidates)
        if pair is None:
            continue
        proximal, distal = pair
        length = np.linalg.norm(centres_mm[distal] - centres_mm[proximal], axis=0)
        rows.append(
            {
                "trial": trial,
                "system": system,
                "source_kind": source_kind,
                "dimension": name,
                "median_length_mm": float(np.nanmedian(length)),
                "sd_length_mm": float(np.nanstd(length)),
            }
        )
    return rows


def model_dimension_rows(trial: str, runs: list[ModelRun]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        centres_mm = centres_to_c3d_mm(
            run.centres_native, run.unit_scale_to_m, "identity"
        )
        rows.extend(
            dimension_rows_from_centres(trial, run.system, run.source_kind, centres_mm)
        )
    return rows


def biomod_neutral_centres_mm(
    biomod_path: Path, unit_scale_to_m: float
) -> dict[str, np.ndarray]:
    """Return neutral-pose segment origins from a biorbd/BioBuddy model.

    BioBuddy models generated from the Motive 57 template do not yet provide an
    analysed q(t) in this pipeline. For model dimensions, the neutral segment
    origins are enough because the template lengths are static. Values are
    returned in millimetres with shape ``(3, 1)`` to match animated centre
    arrays used by Captury and Motive.
    """

    biorbd = require_biorbd()
    model = biorbd.Model(str(biomod_path))
    q = np.zeros(model.nbQ())
    centres_mm: dict[str, np.ndarray] = {}
    for index, name in enumerate(biorbd_segment_names(model)):
        rt = np.asarray(model.globalJCS(q, index).to_array(), dtype=float)
        centres_mm[name] = (rt[:3, 3] * unit_scale_to_m * 1000.0).reshape(3, 1)
    return centres_mm


def biobuddy_dimension_rows(
    trial: str,
    biomod_path: Path | None,
    unit_scale_to_m: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load an optional BioBuddy model and expose it as a dimensions source."""

    if biomod_path is None:
        return [], {
            "status": "missing",
            "reason": "no_biobuddy_biomod_argument",
        }
    if not biomod_path.exists():
        return [], {
            "status": "missing",
            "path": str(biomod_path),
            "reason": "biobuddy_biomod_not_found",
        }
    centres_mm = biomod_neutral_centres_mm(biomod_path, unit_scale_to_m)
    rows = dimension_rows_from_centres(trial, "biobuddy", "motive_57", centres_mm)
    return rows, {
        "status": "ok",
        "path": str(biomod_path),
        "unit_scale_to_m": unit_scale_to_m,
        "segments": len(centres_mm),
        "dimensions": len(rows),
    }


def model_run_from_biobuddy_ik_npz(
    biomod_path: Path,
    ik_npz_path: Path,
    *,
    system: str = "biobuddy",
    source_kind: str = "motive_57_ik",
    unit_scale_to_m: float = 1.0,
) -> ModelRun:
    """Load a BioBuddy IK result and expose it as a third comparison source."""

    data = np.load(ik_npz_path, allow_pickle=True)
    q = np.asarray(data["q"], dtype=float)
    time = np.asarray(data["time"], dtype=float)
    q_names = [str(name) for name in data["q_names"].tolist()]
    q_units = ["rad" if "rot" in name.lower() else "native" for name in q_names]
    joint_names = biorbd_segment_names(require_biorbd().Model(str(biomod_path)))
    centres_native = compute_model_joint_centres_native(
        biomod_path, q, set(joint_names)
    )
    rotations_native = compute_model_segment_rotations_native(
        biomod_path, q, set(joint_names)
    )
    return ModelRun(
        system=system,
        source_kind=source_kind,
        biomod_path=biomod_path,
        q=q,
        q_names=q_names,
        q_units=q_units,
        time=time,
        joint_names=joint_names,
        centres_native=centres_native,
        rotations_native=rotations_native,
        unit_scale_to_m=unit_scale_to_m,
        mesh_report={},
        root_offset_policy={"source": "biobuddy_c3d_ik", "ik_npz": str(ik_npz_path)},
    )


def biobuddy_ik_cache_fingerprint(
    biomod_path: Path,
    c3d_path: Path,
    unit_scale_to_m: float,
    max_frames: int,
    angle_label_regex: str,
) -> dict[str, Any]:
    """Identify saved IK using the same defaults as the batch/static solver."""
    return ik_cache_fingerprint(
        biomod_path,
        c3d_path,
        method="trf",
        max_frames=max_frames,
        biomod_unit_scale_to_m=unit_scale_to_m,
        marker_prefixes_to_strip=DEFAULT_MARKER_PREFIXES_TO_STRIP,
        angle_label_regex=angle_label_regex,
        xtol=1e-6,
    )


def run_biobuddy_ik_for_trial(
    bundle: TrialBundle,
    trial_dir: Path,
    biomod_path: Path | None,
    unit_scale_to_m: float,
    max_frames: int,
    angle_label_regex: str,
    cache_dir: Path,
    force: bool = False,
    *,
    cache_only: bool = False,
) -> tuple[ModelRun | None, dict[str, Any]]:
    """Run or reuse BioBuddy nonlinear TRF IK for one trial."""

    if biomod_path is None:
        return None, {"status": "missing", "reason": "no_biobuddy_biomod_argument"}
    if not biomod_path.exists():
        return None, {
            "status": "missing",
            "path": str(biomod_path),
            "reason": "biobuddy_biomod_not_found",
        }
    ik_dir = trial_dir / "biobuddy_ik"
    source_name = f"biobuddy_{safe_name(bundle.name)}"
    try:
        if cache_only:
            fingerprint = biobuddy_ik_cache_fingerprint(
                biomod_path,
                bundle.motive_c3d,
                unit_scale_to_m,
                max_frames,
                angle_label_regex,
            )
            directory = cache_dir / fingerprint["digest"]
            ik_npz = directory / "inverse_kinematics_trf.npz"
            summary = directory / "inverse_kinematics_trf_summary.json"
            if not summary.is_file() or not ik_npz.is_file():
                return None, {"status": "missing", "reason": "no_matching_cached_ik"}
            report = json.loads(summary.read_text(encoding="utf-8"))
            if report.get("cache", {}).get("digest") != fingerprint[
                "digest"
            ] or not cached_ik_is_valid(report, ik_npz):
                return None, {"status": "missing", "reason": "invalid_cached_ik"}
            report["cache"]["hit"] = True
            report["outputs"]["npz"] = str(ik_npz)
            print(
                f"[BioBuddy IK] résultat existant réutilisé: {bundle.name}", flush=True
            )
        else:
            report = run_direct_biobuddy_ik(
                biomod_path,
                bundle.motive_c3d,
                ik_dir,
                source_name=source_name,
                biomod_unit_scale_to_m=unit_scale_to_m,
                angle_label_regex=angle_label_regex,
                max_frames=max_frames,
                cache_dir=cache_dir,
                force=force,
            )
            ik_npz = Path(report["outputs"]["npz"])
        run = model_run_from_biobuddy_ik_npz(
            biomod_path,
            ik_npz,
            unit_scale_to_m=unit_scale_to_m,
        )
        return run, report
    except Exception as exc:
        return None, {"status": "error", "error": str(exc), "path": str(biomod_path)}


def is_synthetic_joint_centre_label(label: str) -> bool:
    return is_joint_centre_marker_label(label)


def unique_marker_labels(labels: list[str]) -> list[str]:
    return marker_display_labels(labels)


def proposal_candidate_indices(labels: list[str]) -> list[int]:
    return [
        index
        for index, label in enumerate(labels)
        if clean_marker_label(label) and not is_synthetic_joint_centre_label(label)
    ]


def propose_marker_correspondences_from_points(
    motive_labels: list[str],
    motive_points: np.ndarray,
    motive_time: np.ndarray,
    captury_labels: list[str],
    captury_points: np.ndarray,
    captury_time: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    captury_lag_s: float = 0.0,
    max_median_error_mm: float = 250.0,
    max_pairs: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Propose one-to-one skin marker pairs after Captury -> Motive alignment."""

    motive_unique = unique_marker_labels(motive_labels)
    captury_unique = unique_marker_labels(captury_labels)
    captury_time = apply_time_offset(captury_time, captury_lag_s)
    motive_indices = proposal_candidate_indices(motive_labels)
    captury_indices = proposal_candidate_indices(captury_labels)
    candidates: list[dict[str, Any]] = []
    for motive_index in motive_indices:
        motive_signal = motive_points[:, motive_index, :].T
        motive_valid = np.all(np.isfinite(motive_signal), axis=1)
        if not motive_valid.any():
            continue
        for captury_index in captury_indices:
            captury_signal = captury_points[:, captury_index, :]
            captury_on_motive = (
                interpolate_finite_array(captury_signal, captury_time, motive_time).T
                @ rotation
                + translation
            )
            valid = motive_valid & np.all(np.isfinite(captury_on_motive), axis=1)
            if not valid.any():
                continue
            distance = np.linalg.norm(
                captury_on_motive[valid] - motive_signal[valid], axis=1
            )
            if distance.size == 0:
                continue
            median_error = float(np.nanmedian(distance))
            candidates.append(
                {
                    "name": (
                        f"{motive_unique[motive_index]}_to_"
                        f"{captury_unique[captury_index]}"
                    ),
                    "reference": [motive_unique[motive_index]],
                    "test": [captury_unique[captury_index]],
                    "median_error_mm": median_error,
                    "p95_error_mm": float(np.nanpercentile(distance, 95)),
                    "n_frames": int(valid.sum()),
                }
            )
    selected: list[dict[str, Any]] = []
    used_motive: set[str] = set()
    used_captury: set[str] = set()
    for candidate in sorted(candidates, key=lambda row: row["median_error_mm"]):
        motive_label = str(candidate["reference"][0])
        captury_label = str(candidate["test"][0])
        if motive_label in used_motive or captury_label in used_captury:
            continue
        if float(candidate["median_error_mm"]) > max_median_error_mm:
            continue
        selected.append(candidate)
        used_motive.add(motive_label)
        used_captury.add(captury_label)
        if len(selected) >= max_pairs:
            break
    report = {
        "method": "greedy_min_median_distance_after_joint_centre_alignment",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "max_median_error_mm": max_median_error_mm,
        "max_pairs": max_pairs,
    }
    return selected, report


def propose_marker_correspondences(
    motive_c3d: Path,
    captury_c3d: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    angle_label_regex: str,
    captury_lag_s: float = 0.0,
    captury_c3d_to_common: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    motive_labels, motive_points, _motive_residuals, motive_time = read_c3d_points_mm(
        motive_c3d, angle_label_regex
    )
    captury_labels, captury_points, _captury_residuals, captury_time = (
        read_c3d_points_mm(captury_c3d, angle_label_regex)
    )
    if captury_c3d_to_common is not None:
        captury_points = transform_point_trajectories(
            captury_points, captury_c3d_to_common
        )
    return propose_marker_correspondences_from_points(
        motive_labels,
        motive_points,
        motive_time,
        captury_labels,
        captury_points,
        captury_time,
        rotation,
        translation,
        captury_lag_s=captury_lag_s,
    )


def marker_correspondence_rows(
    trial: str,
    motive_c3d: Path,
    captury_c3d: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    landmark_map: list[dict[str, Any]],
    captury_lag_s: float = 0.0,
    captury_c3d_to_common: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    motive_labels, motive_points, _motive_residuals, motive_time = read_c3d_points_mm(
        motive_c3d
    )
    captury_labels, captury_points, _captury_residuals, captury_time = (
        read_c3d_points_mm(captury_c3d)
    )
    if captury_c3d_to_common is not None:
        captury_points = transform_point_trajectories(
            captury_points, captury_c3d_to_common
        )
    captury_time = apply_time_offset(captury_time, captury_lag_s)
    motive_lookup = marker_indices_by_clean_label(motive_labels)
    captury_lookup = marker_indices_by_clean_label(captury_labels)
    rows: list[dict[str, Any]] = []
    timeseries_rows: list[dict[str, Any]] = []
    for item in landmark_map:
        name = str(item["name"])
        motive_indices = [
            index
            for label in item["reference"]
            for index in motive_lookup.get(label, [])
        ]
        captury_indices = [
            index for label in item["test"] for index in captury_lookup.get(label, [])
        ]
        motive_signal = average_marker_group(motive_points, motive_indices)
        captury_signal = average_marker_group(captury_points, captury_indices)
        if motive_signal is None or captury_signal is None:
            continue
        captury_on_motive = (
            interpolate_finite_array(captury_signal, captury_time, motive_time).T
            @ rotation
            + translation
        )
        motive_rows = motive_signal.T
        valid = np.all(np.isfinite(captury_on_motive), axis=1) & np.all(
            np.isfinite(motive_rows), axis=1
        )
        distance = np.linalg.norm(captury_on_motive - motive_rows, axis=1)
        rows.append(
            {
                "trial": trial,
                "landmark": name,
                "motive_labels": ";".join(item["reference"]),
                "captury_labels": ";".join(item["test"]),
                "n_frames": int(np.sum(valid)),
                "median_error_mm": (
                    float(np.nanmedian(distance[valid])) if valid.any() else np.nan
                ),
                "p95_error_mm": (
                    float(np.nanpercentile(distance[valid], 95))
                    if valid.any()
                    else np.nan
                ),
                "rmse_error_mm": (
                    float(np.sqrt(np.nanmean(distance[valid] ** 2)))
                    if valid.any()
                    else np.nan
                ),
            }
        )
        difference = captury_on_motive - motive_rows
        for frame, time_value in enumerate(motive_time):
            timeseries_rows.append(
                {
                    "trial": trial,
                    "time": float(time_value),
                    "landmark": name,
                    "motive_labels": ";".join(item["reference"]),
                    "captury_labels": ";".join(item["test"]),
                    "error_x_mm": float(difference[frame, 0]),
                    "error_y_mm": float(difference[frame, 1]),
                    "error_z_mm": float(difference[frame, 2]),
                    "distance_mm": float(distance[frame]),
                }
            )
    return rows, timeseries_rows


def temporal_synchronization_artifacts(
    motive_centres_mm: Mapping[str, np.ndarray],
    motive_time: np.ndarray,
    captury_centres_mm: Mapping[str, np.ndarray],
    captury_original_time: np.ndarray,
    synchronization: dict[str, Any],
    *,
    start_s: float | None,
    end_s: float | None,
    phase_points: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Build compact synchronization and normalized-phase traces.

    The traces use the same translation-invariant composite joint-centre speed
    as the lag estimator. They are diagnostic signals, not anatomical angles.
    """

    reported_centres = synchronization.get("used_centres", [])
    common = sorted(
        set(reported_centres) or set(motive_centres_mm).intersection(captury_centres_mm)
    )
    motive_signal, motive_used = composite_joint_centre_speed(
        {name: motive_centres_mm[name] for name in common}, motive_time
    )
    captury_signal, captury_used = composite_joint_centre_speed(
        {name: captury_centres_mm[name] for name in common}, captury_original_time
    )
    used = sorted(set(motive_used).intersection(captury_used))
    if used and (set(motive_used) != set(used) or set(captury_used) != set(used)):
        motive_signal, _ = composite_joint_centre_speed(
            {name: motive_centres_mm[name] for name in used}, motive_time
        )
        captury_signal, _ = composite_joint_centre_speed(
            {name: captury_centres_mm[name] for name in used},
            captury_original_time,
        )
    lag_s = float(synchronization.get("lag_s", 0.0))
    captury_corrected_time = apply_time_offset(captury_original_time, lag_s)
    captury_on_motive = interpolate_finite_signal(
        captury_signal, captury_corrected_time, motive_time
    )
    valid = np.isfinite(motive_signal) & np.isfinite(captury_on_motive)
    selected = valid.copy()
    if start_s is not None:
        selected &= motive_time >= float(start_s)
    if end_s is not None:
        selected &= motive_time <= float(end_s)
    timeseries_rows = [
        {
            "time_s": float(time_value),
            "motive_composite_speed": float(motive_signal[index]),
            "captury_composite_speed": float(captury_on_motive[index]),
            "difference": float(captury_on_motive[index] - motive_signal[index]),
            "in_selected_window": bool(selected[index]),
        }
        for index, time_value in enumerate(motive_time)
    ]
    phase_rows, phase_report = normalize_selected_phase(
        motive_time,
        motive_signal,
        captury_original_time,
        captury_signal,
        lag_s=lag_s,
        start_s=start_s,
        end_s=end_s,
        n_points=phase_points,
    )
    if (
        np.count_nonzero(selected) >= 3
        and np.std(motive_signal[selected]) > 1e-9
        and np.std(captury_on_motive[selected]) > 1e-9
    ):
        reference = motive_signal[selected]
        moving = captury_on_motive[selected]
        reference_z = (reference - np.mean(reference)) / np.std(reference)
        moving_z = (moving - np.mean(moving)) / np.std(moving)
        residual_rmse = float(np.sqrt(np.mean((moving_z - reference_z) ** 2)))
        correlation = float(np.corrcoef(reference, moving)[0, 1])
    else:
        residual_rmse = None
        correlation = None
    evaluation = {
        "signal": "dimensionless_composite_joint_centre_speed",
        "used_centres": used,
        "selected_samples": int(np.count_nonzero(selected)),
        "selected_start_s": (
            float(motive_time[selected][0]) if np.any(selected) else None
        ),
        "selected_end_s": (
            float(motive_time[selected][-1]) if np.any(selected) else None
        ),
        "correlation": correlation,
        "normalized_residual_rmse": residual_rmse,
        "phase_normalization": phase_report,
    }
    return timeseries_rows, phase_rows, evaluation


def finite_range(values: np.ndarray, axis: int) -> np.ndarray:
    """Return max-min along ``axis`` while preserving all-missing slices as NaN."""

    array = np.asarray(values, dtype=float)
    finite = np.isfinite(array)
    maximum = np.max(np.where(finite, array, -np.inf), axis=axis)
    minimum = np.min(np.where(finite, array, np.inf), axis=axis)
    result = maximum - minimum
    return np.where(np.any(finite, axis=axis), result, np.nan)


def vertical_amplitude_report(enriched_c3d: Path) -> dict[str, Any]:
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(enriched_c3d))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points_mm = np.asarray(c3d["data"]["points"][:3], dtype=float) * unit_mm
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(labels):
        if not (label.startswith("CAPJC_") or label.startswith("MOTJC_")):
            continue
        values = points_mm[:, index, :]
        ranges = finite_range(values, axis=1)
        rows.append(
            {
                "label": label,
                "x_range_mm": float(ranges[0]),
                "y_range_mm": float(ranges[1]),
                "z_range_mm": float(ranges[2]),
            }
        )
    joint_indices = [labels.index(row["label"]) for row in rows]
    if joint_indices:
        joint_points = points_mm[:, joint_indices, :]
        spatial_ranges = finite_range(joint_points, axis=1)
        median_spatial_range = np.nanmedian(spatial_ranges, axis=1)
        max_spatial_range = np.nanmax(spatial_ranges, axis=1)
    else:
        median_spatial_range = np.full(3, np.nan)
        max_spatial_range = np.full(3, np.nan)
    return {
        "labels": len(rows),
        "median_z_range_mm": (
            float(np.nanmedian([row["z_range_mm"] for row in rows])) if rows else np.nan
        ),
        "median_xy_range_mm": (
            float(
                np.nanmedian(
                    [max(row["x_range_mm"], row["y_range_mm"]) for row in rows]
                )
            )
            if rows
            else np.nan
        ),
        "median_spatial_extent_x_mm": float(median_spatial_range[0]),
        "median_spatial_extent_y_mm": float(median_spatial_range[1]),
        "median_spatial_extent_z_mm": float(median_spatial_range[2]),
        "max_spatial_extent_x_mm": float(max_spatial_range[0]),
        "max_spatial_extent_y_mm": float(max_spatial_range[1]),
        "max_spatial_extent_z_mm": float(max_spatial_range[2]),
        "rows": rows,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def write_table_npz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    columns = np.asarray(list(dataframe.columns), dtype=str)
    payload: dict[str, np.ndarray] = {"columns": columns}
    for index, column in enumerate(columns):
        series = dataframe[str(column)]
        if pd.api.types.is_numeric_dtype(series):
            payload[f"col_{index}"] = series.to_numpy()
        else:
            payload[f"col_{index}"] = series.fillna("").astype(str).to_numpy(dtype=str)
    np.savez_compressed(path, **payload)


def metric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    columns: list[str] = []
    for column in df.columns:
        if column in exclude:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


METRIC_FIGURE_METADATA_COLUMNS = {
    "paired_samples",
    "paired_frames",
    "paired_coverage",
    "minimum_paired_samples",
    "minimum_paired_coverage",
    "minimum_amplitude_rad",
    "minimum_amplitude_native",
    "shape_metrics_eligible",
}


def plot_metric_barh(
    df: pd.DataFrame,
    category: str,
    metric: str,
    output_path: Path,
    title: str,
    xlabel: str,
) -> Path | None:
    if (
        df.empty
        or category not in df.columns
        or "trial" not in df.columns
        or metric not in df.columns
    ):
        return None
    values = df[[category, "trial", metric]].copy()
    values[metric] = pd.to_numeric(values[metric], errors="coerce")
    values = values.dropna(subset=[metric])
    if values.empty:
        return None
    pivot = values.pivot_table(
        index=category, columns="trial", values=metric, aggfunc="mean"
    )
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=True).index]
    height = min(24.0, max(5.0, 0.28 * max(1, len(pivot.index)) + 1.5))
    width = min(18.0, max(8.0, 1.8 * max(1, len(pivot.columns)) + 6.0))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(width, height), constrained_layout=True)
    pivot.plot(kind="barh", ax=axis)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(category)
    axis.grid(axis="x", alpha=0.3)
    axis.legend(title="trial", loc="best")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_metric_figures(
    centre_rows: list[dict[str, Any]], q_rows: list[dict[str, Any]], out_dir: Path
) -> dict[str, list[str]]:
    figure_paths: dict[str, list[str]] = {"joint_centres": [], "kinematics_q": []}
    figures_dir = out_dir / "figures"
    if centre_rows:
        centre_df = pd.DataFrame(centre_rows)
        for metric in metric_columns(
            centre_df,
            {"trial", "joint", "n_frames", *METRIC_FIGURE_METADATA_COLUMNS},
        ):
            path = plot_metric_barh(
                centre_df,
                category="joint",
                metric=metric,
                output_path=figures_dir / "joint_centres" / f"{metric}.png",
                title=f"Joint centres - {metric}",
                xlabel=metric,
            )
            if path is not None:
                figure_paths["joint_centres"].append(str(path))
    if q_rows:
        q_df = pd.DataFrame(q_rows)
        for metric in metric_columns(
            q_df,
            {
                "trial",
                "q_name",
                "unit",
                "source",
                "participant",
                *METRIC_FIGURE_METADATA_COLUMNS,
            },
        ):
            path = plot_metric_barh(
                q_df,
                category="q_name",
                metric=metric,
                output_path=figures_dir / "kinematics_q" / f"{metric}.png",
                title=f"Kinematics q - {metric}",
                xlabel=metric,
            )
            if path is not None:
                figure_paths["kinematics_q"].append(str(path))
    return figure_paths


def visualize_enriched_c3d(
    enriched_c3d: Path, wait_seconds: float, headless: bool
) -> None:
    if headless:
        os.environ["PYORERUN_HEADLESS"] = "1"
    _, PhaseRerun, PyoMarkers = require_pyorerun()
    ezc3d = require_ezc3d()
    c3d = ezc3d.c3d(str(enriched_c3d))
    labels = as_str_list(get_c3d_param(c3d, "POINT", "LABELS", []))
    unit_mm = unit_scale_to_mm(
        as_str_list(get_c3d_param(c3d, "POINT", "UNITS", [""]))[0]
    )
    points = np.asarray(c3d["data"]["points"][:3], dtype=float) * unit_mm
    rate_value = get_c3d_param(c3d, "POINT", "RATE", [120])
    rate = float(
        rate_value[0]
        if isinstance(rate_value, (list, tuple, np.ndarray))
        else rate_value
    )
    time = np.arange(points.shape[2], dtype=float) / rate
    keep = [
        i for i, label in enumerate(labels) if label.startswith(("CAPJC_", "MOTJC_"))
    ]
    if not keep:
        raise RuntimeError(f"No CAPJC_/MOTJC_ labels in {enriched_c3d}.")
    phase = PhaseRerun(time)
    phase.add_xp_markers(
        "p6_joint_centres",
        PyoMarkers(data=points[:, keep, :], channels=[labels[i] for i in keep]),
    )
    if headless:
        return
    phase.rerun("p6_motive_captury_joint_centres", notebook=False)
    if wait_seconds > 0:
        import time as time_module

        time_module.sleep(wait_seconds)


def compare_trial(
    bundle: TrialBundle,
    out_root: Path,
    args: argparse.Namespace,
    static_alignment_transform: SpatialCalibration | None = None,
) -> tuple[dict[str, Any], SpatialCalibration]:
    trial_dir = out_root / safe_name(bundle.name)
    trial_dir.mkdir(parents=True, exist_ok=True)
    cache_fingerprint = trial_cache_fingerprint(
        bundle, args, static_alignment_transform
    )
    cached_report = cached_trial_report(
        trial_dir, bundle, args, static_alignment_transform
    )
    if cached_report is not None:
        print(f"Using cached trial outputs: {bundle.name}")
        cached_report.setdefault("cache", {})["hit"] = True
        enriched_c3d = cached_report.get("outputs", {}).get("enriched_c3d")
        if (
            args.visualize
            and (args.visualize_trial is None or args.visualize_trial == bundle.name)
            and enriched_c3d
        ):
            visualize_enriched_c3d(
                Path(enriched_c3d), args.rerun_wait_seconds, args.headless
            )
        cached_calibration = spatial_calibration_from_report(cached_report)
        if cached_calibration is None:
            raise RuntimeError(
                f"Cached report for {bundle.name} has no spatial calibration. "
                "Re-run with --no-cache."
            )
        return cached_report, static_alignment_transform or cached_calibration
    captury_root_mode = requested_root_offset_mode(
        args, "captury", static_alignment_transform
    )
    motive_root_mode = requested_root_offset_mode(
        args, "motive", static_alignment_transform
    )
    captury_axis_mode = requested_model_to_own_c3d_axis(args, "captury")
    motive_axis_mode = requested_model_to_own_c3d_axis(args, "motive")
    captury_coordinates = resolve_system_coordinate_frames("captury", captury_axis_mode)
    motive_coordinates = resolve_system_coordinate_frames("motive", motive_axis_mode)
    captury, captury_source_audit = build_model_run_with_rotation_audit(
        bundle,
        "captury",
        args,
        trial_dir,
        captury_root_mode,
        captury_coordinates.model_to_own_c3d,
    )
    motive, motive_source_audit = build_model_run_with_rotation_audit(
        bundle,
        "motive",
        args,
        trial_dir,
        motive_root_mode,
        motive_coordinates.model_to_own_c3d,
    )
    cap_own_c3d_mm = centres_to_c3d_mm(
        captury.centres_native,
        captury.unit_scale_to_m,
        captury_coordinates.model_to_own_c3d,
    )
    mot_own_c3d_mm = centres_to_c3d_mm(
        motive.centres_native,
        motive.unit_scale_to_m,
        motive_coordinates.model_to_own_c3d,
    )
    cap_c3d_mm = {
        name: captury_coordinates.own_c3d_to_common @ values
        for name, values in cap_own_c3d_mm.items()
    }
    mot_c3d_mm = {
        name: motive_coordinates.own_c3d_to_common @ values
        for name, values in mot_own_c3d_mm.items()
    }
    alignment_report: dict[str, Any]
    spatial_calibration = static_alignment_transform
    if args.disable_static_model_alignment:
        captury_to_motive = RowRigidTransform.identity()
        rotation = captury_to_motive.rotation
        translation = captury_to_motive.translation
        alignment_report = {
            "status": "disabled_static_model_alignment",
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
            "note": "Captury centres are kept in their converted C3D frame without Captury -> Motive model alignment.",
        }
    elif spatial_calibration is None:
        alignment_mode = getattr(args, "spatial_alignment_mode", "held_out_centres")
        if alignment_mode == "legacy_all_centres":
            rotation, translation, alignment_report = static_alignment(
                cap_c3d_mm, mot_c3d_mm
            )
            captury_to_motive = RowRigidTransform(rotation, translation)
            alignment_report.update(
                {
                    "method": "legacy_all_common_centres_kabsch_rows",
                    "protocol_status": "circular_diagnostic_only",
                    "calibration_centres": alignment_report.get("used_centres", []),
                    "evaluation_centres": [],
                }
            )
        else:
            requested_centres = tuple(
                getattr(args, "alignment_calibration_centre", [])
                or DEFAULT_ALIGNMENT_CALIBRATION_CENTRES
            )
            captury_to_motive, alignment_report = fit_held_out_centre_alignment(
                cap_c3d_mm, mot_c3d_mm, requested_centres
            )
            if alignment_report["status"] != "ok":
                raise RuntimeError(
                    "Static held-out centre alignment failed: "
                    f"{alignment_report['status']}. Calibration centres: "
                    f"{alignment_report.get('calibration_centres', [])}."
                )
            rotation = captury_to_motive.rotation
            translation = captury_to_motive.translation
            alignment_report["protocol_status"] = "non_circular_held_out"
    else:
        captury_to_motive = spatial_calibration.captury_to_motive
        rotation = captury_to_motive.rotation
        translation = captury_to_motive.translation
        alignment_report = {
            "status": "reused_frozen_static_alignment",
            "method": "frozen_static_calibration",
            "protocol_status": spatial_calibration.status,
            "calibration_centres": list(spatial_calibration.calibration_centres),
            "evaluation_centres": list(spatial_calibration.evaluation_centres),
            "rotation": rotation.tolist(),
            "translation_mm": translation.tolist(),
        }
    cap_aligned_mm = apply_alignment(cap_c3d_mm, rotation, translation)
    if spatial_calibration is not None:
        motive_to_c3d = spatial_calibration.motive_to_c3d
        model_marker_rotation = motive_to_c3d.rotation
        model_marker_translation = motive_to_c3d.translation
        model_marker_report = {
            "status": "reused_frozen_static_alignment",
            "method": "frozen_static_calibration",
            "rotation": model_marker_rotation.tolist(),
            "translation_mm": model_marker_translation.tolist(),
        }
    elif args.disable_motive_marker_alignment:
        motive_to_c3d = RowRigidTransform.identity()
        model_marker_rotation = motive_to_c3d.rotation
        model_marker_translation = motive_to_c3d.translation
        model_marker_report = {
            "status": "disabled_motive_marker_alignment",
            "method": "identity",
            "rotation": model_marker_rotation.tolist(),
            "translation_mm": model_marker_translation.tolist(),
            "note": "Motive model centres are not yaw/translation-aligned to Motive C3D marker proxies.",
        }
    else:
        model_marker_rotation, model_marker_translation, model_marker_report = (
            model_to_motive_marker_alignment(mot_c3d_mm, motive.time, bundle.motive_c3d)
        )
        motive_to_c3d = RowRigidTransform(
            model_marker_rotation, model_marker_translation
        )
    captury_frame_calibration: dict[str, Any] | None = None
    if getattr(args, "captury_calibrate_segment_frames", False):
        if spatial_calibration is not None:
            stored = spatial_calibration.captury_segment_frame_calibration
            if not stored:
                captury_frame_calibration = {
                    "status": "missing_frozen_static_calibration",
                    "source_kind": captury.source_kind,
                }
            elif stored.get("source_kind") != captury.source_kind:
                captury_frame_calibration = {
                    "status": "source_kind_mismatch",
                    "static_source_kind": stored.get("source_kind"),
                    "dynamic_source_kind": captury.source_kind,
                    "note": "The static correction is not applied across BVH and FBX exports.",
                }
            else:
                captury_frame_calibration = dict(stored)
                captury_frame_calibration["status"] = "reused_frozen_static_calibration"
        elif not is_static_frame_calibration_trial(bundle.name, args.static_trial):
            captury_frame_calibration = {
                "status": "requires_static_trial_calibration",
                "trial": bundle.name,
                "static_trial": args.static_trial,
                "note": (
                    "A dynamic trial cannot fit a Captury segment-frame "
                    "calibration; run the named static trial first."
                ),
            }
        else:
            q_landmarks_config = load_captury_q_landmark_config(
                args.captury_q_landmarks_json
                or Path(__file__).with_name("captury_q_landmarks.json")
            )
            captury_labels, captury_points_mm, _residuals, _time = read_c3d_points_mm(
                bundle.captury_c3d, args.angle_label_regex
            )
            captury_rotations_own_c3d = rotations_to_c3d(
                captury.rotations_native, captury_coordinates.model_to_own_c3d
            )
            corrections, calibration_report = fit_captury_segment_frame_corrections(
                captury_rotations_own_c3d,
                cap_own_c3d_mm,
                captury_labels,
                captury_points_mm,
                q_landmarks_config,
            )
            captury_frame_calibration = {
                "status": calibration_report["status"],
                "source_kind": captury.source_kind,
                "q_landmarks_json": str(
                    args.captury_q_landmarks_json
                    or Path(__file__).with_name("captury_q_landmarks.json")
                ),
                "matrices": {
                    segment: matrix.tolist() for segment, matrix in corrections.items()
                },
                "report": calibration_report,
            }
    if spatial_calibration is None:
        calibration_centres = tuple(alignment_report.get("calibration_centres", []))
        evaluation_centres = tuple(alignment_report.get("evaluation_centres", []))
        status = str(
            alignment_report.get("protocol_status", alignment_report["status"])
        )
        if status == "circular_diagnostic_only":
            evaluation_centres = ()
        spatial_calibration = SpatialCalibration(
            static_trial=bundle.name,
            calibration_centres=calibration_centres,
            evaluation_centres=evaluation_centres,
            captury_to_motive=captury_to_motive,
            motive_to_c3d=motive_to_c3d,
            status=status,
            captury_root_offset_mode=selected_root_offset_mode(captury),
            motive_root_offset_mode=selected_root_offset_mode(motive),
            captury_segment_frame_calibration=captury_frame_calibration,
        )
    marker_rotation, marker_translation = compose_row_alignment(
        rotation, translation, model_marker_rotation, model_marker_translation
    )
    cap_aligned_mm = apply_alignment(
        cap_aligned_mm, model_marker_rotation, model_marker_translation
    )
    mot_c3d_mm = apply_alignment(
        mot_c3d_mm, model_marker_rotation, model_marker_translation
    )
    cap_rotations_c3d = rotations_to_c3d(
        captury.rotations_native,
        captury_coordinates.model_to_common_c3d,
        rotation @ model_marker_rotation,
    )
    mot_rotations_c3d = rotations_to_c3d(
        motive.rotations_native,
        motive_coordinates.model_to_common_c3d,
        model_marker_rotation,
    )
    segment_orientation_report: dict[str, Any] = {
        "captury_reorient_thigh_y_from_cor": bool(
            args.captury_reorient_thigh_y_from_cor
        ),
        "rotate_body_segments_180_x": bool(args.rotate_body_segments_180_x),
        "reexpress_rotations_zxy": bool(args.reexpress_rotations_zxy),
        "captury_rename_axes": bool(getattr(args, "captury_rename_axes", False)),
        "captury_calibrate_segment_frames": bool(
            getattr(args, "captury_calibrate_segment_frames", False)
        ),
        "applied": [],
    }
    if getattr(args, "captury_rename_axes", False):
        cap_rotations_c3d = apply_simple_captury_axis_rename(cap_rotations_c3d)
        segment_orientation_report["applied"].append(
            "captury_simple_global_axis_rename_Ry_90_deg"
        )
    elif getattr(args, "captury_calibrate_segment_frames", False):
        calibration_payload = captury_frame_calibration or {}
        matrices = {
            segment: np.asarray(matrix, dtype=float)
            for segment, matrix in calibration_payload.get("matrices", {}).items()
        }
        if matrices:
            cap_rotations_c3d = apply_segment_frame_corrections(
                cap_rotations_c3d, matrices
            )
            segment_orientation_report["applied"].append(
                "captury_static_q_landmarks_and_own_model_centres"
            )
        segment_orientation_report["captury_static_frame_calibration"] = (
            calibration_payload
        )
    if args.captury_reorient_thigh_y_from_cor:
        cap_rotations_c3d = correct_captury_thigh_y_from_cor(
            cap_rotations_c3d, cap_aligned_mm
        )
        segment_orientation_report["applied"].append(
            "captury_thigh_y_axis_from_hip_to_knee_cor"
        )
    if args.rotate_body_segments_180_x:
        cap_rotations_c3d = rotate_segment_frames_180_x(cap_rotations_c3d)
        mot_rotations_c3d = rotate_segment_frames_180_x(mot_rotations_c3d)
        segment_orientation_report["applied"].append(
            "captury_and_motive_segment_frames_rotated_180_deg_about_local_x_then_local_y"
        )
    if args.reexpress_rotations_zxy:
        segment_orientation_report["applied"].append(
            "captury_and_motive_q_reexpressed_from_corrected_segment_matrices_as_ZXY"
        )
    alignment_report["motive_model_to_c3d_markers"] = model_marker_report
    captury_original_time = captury.time.copy()
    temporal_sync_mode = getattr(args, "temporal_sync_mode", "auto")
    if temporal_sync_mode == "auto" and bundle.name == args.static_trial:
        temporal_synchronization = {
            "status": "static_reference_no_lag",
            "method": "auto_skipped_for_static_reference",
            "applied": False,
            "lag_s": 0.0,
            "lag_convention": LAG_CONVENTION,
            "reference_clock": "motive",
            "moving_clock": "captury",
            "reason": "constant_lag_requires_a_dynamic_trial",
        }
    else:
        temporal_synchronization = resolve_temporal_synchronization(
            temporal_sync_mode,
            mot_c3d_mm,
            motive.time,
            cap_aligned_mm,
            captury_original_time,
            manual_lag_s=getattr(args, "manual_lag_s", None),
            max_lag_s=float(getattr(args, "max_lag_s", 0.5)),
        )
    temporal_lag_s = float(temporal_synchronization.get("lag_s", 0.0))
    captury = replace(
        captury,
        time=apply_time_offset(captury_original_time, temporal_lag_s),
    )
    enriched_c3d = append_centres_to_motive_c3d(
        bundle.motive_c3d,
        trial_dir / f"{safe_name(bundle.name)}_motive_with_capjc_motjc.c3d",
        cap_aligned_mm,
        mot_c3d_mm,
        captury.time,
        motive.time,
        args.angle_label_regex,
    )
    detected_event_report, _ = detect_trial_events_and_contacts(
        bundle.motive_c3d, trial_dir, bundle.name
    )
    cut_start_s, cut_end_s, effective_cut_mode = resolve_cut_window(
        args.cut_mode, args.time_start, args.time_end, detected_event_report
    )
    temporal_rows, phase_rows, temporal_evaluation = temporal_synchronization_artifacts(
        mot_c3d_mm,
        motive.time,
        cap_aligned_mm,
        captury_original_time,
        temporal_synchronization,
        start_s=cut_start_s,
        end_s=cut_end_s,
        phase_points=int(getattr(args, "phase_normalization_points", 101)),
    )
    temporal_synchronization["comparison_window"] = {
        "cut_mode": args.cut_mode,
        "effective_cut_mode": effective_cut_mode,
        "start_s": cut_start_s,
        "end_s": cut_end_s,
    }
    temporal_synchronization["evaluation"] = temporal_evaluation
    captury_metrics = trim_model_run(captury, cut_start_s, cut_end_s)
    motive_metrics = trim_model_run(motive, cut_start_s, cut_end_s)
    biobuddy_run, biobuddy_ik_report = run_biobuddy_ik_for_trial(
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
    if biobuddy_run is not None:
        biobuddy_run = trim_model_run(biobuddy_run, cut_start_s, cut_end_s)
    cap_aligned_metrics_mm = trim_centres(
        cap_aligned_mm, time_window_mask(captury.time, cut_start_s, cut_end_s)
    )
    mot_metrics_mm = trim_centres(
        mot_c3d_mm, time_window_mask(motive.time, cut_start_s, cut_end_s)
    )
    centre_rows, centre_ts_rows = centre_metric_rows(
        bundle.name,
        cap_aligned_metrics_mm,
        mot_metrics_mm,
        captury_metrics.time,
        motive_metrics.time,
        args.joint_filter,
        excluded_joints=(
            set(spatial_calibration.calibration_centres)
            if spatial_calibration.status == "non_circular_held_out"
            else set()
        ),
    )
    calibration_centre_names = set(spatial_calibration.calibration_centres)
    calibration_centre_rows, calibration_centre_ts_rows = centre_metric_rows(
        bundle.name,
        {
            name: values
            for name, values in cap_aligned_metrics_mm.items()
            if name in calibration_centre_names
        },
        {
            name: values
            for name, values in mot_metrics_mm.items()
            if name in calibration_centre_names
        },
        captury_metrics.time,
        motive_metrics.time,
    )
    if biobuddy_run is not None:
        bio_centres_mm = centres_to_c3d_mm(
            biobuddy_run.centres_native, biobuddy_run.unit_scale_to_m, "identity"
        )
        add_biobuddy_centre_rows(
            bundle.name,
            centre_rows,
            centre_ts_rows,
            mot_metrics_mm,
            motive_metrics.time,
            bio_centres_mm,
            biobuddy_run.time,
            args.joint_filter,
        )
    cap_rotation_metrics = trim_rotations(
        cap_rotations_c3d, time_window_mask(captury.time, cut_start_s, cut_end_s)
    )
    mot_rotation_metrics = trim_rotations(
        mot_rotations_c3d, time_window_mask(motive.time, cut_start_s, cut_end_s)
    )
    q_captury_metrics = captury_metrics
    q_motive_metrics = motive_metrics
    q_reexpression_report: dict[str, Any] = {
        "enabled": bool(args.reexpress_rotations_zxy),
        "target_sequence": (
            ROTATION_SEQUENCE_ZXY if args.reexpress_rotations_zxy else None
        ),
        "applied_sources": [],
        "biobuddy_modified": False,
    }
    if args.reexpress_rotations_zxy:
        q_captury_metrics, captury_q_report = reexpress_model_run_rotational_q(
            captury_metrics, cap_rotation_metrics, ROTATION_SEQUENCE_ZXY
        )
        q_motive_metrics, motive_q_report = reexpress_model_run_rotational_q(
            motive_metrics, mot_rotation_metrics, ROTATION_SEQUENCE_ZXY
        )
        q_reexpression_report.update(
            {
                "applied_sources": ["captury", "motive"],
                "captury": captury_q_report,
                "motive": motive_q_report,
            }
        )
    q_rows, q_ts_rows = q_metric_rows_with_optional_biobuddy(
        bundle.name, q_captury_metrics, q_motive_metrics, biobuddy_run
    )
    c3d_angle_rows, c3d_angle_ts_rows, c3d_angle_decode_report = captury_c3d_angle_rows(
        bundle.name,
        bundle.captury_c3d,
        args.angle_label_regex,
        args.c3d_angle_unit,
        cut_start_s,
        cut_end_s,
        temporal_lag_s,
    )
    q_rows.extend(c3d_angle_rows)
    q_ts_rows.extend(c3d_angle_ts_rows)
    segment_q_rows, segment_q_ts_rows = segment_relative_q_metric_rows(
        bundle.name,
        cap_rotation_metrics,
        mot_rotation_metrics,
        captury_metrics.time,
        motive_metrics.time,
    )
    q_rows.extend(segment_q_rows)
    q_ts_rows.extend(segment_q_ts_rows)
    segment_rows, segment_ts_rows, segment_report = segment_rotation_metric_rows(
        bundle.name,
        {
            "captury": cap_rotation_metrics,
            "motive": mot_rotation_metrics,
            "biobuddy": (
                biobuddy_run.rotations_native if biobuddy_run is not None else {}
            ),
        },
        {
            "captury": captury_metrics.time,
            "motive": motive_metrics.time,
            "biobuddy": (
                biobuddy_run.time
                if biobuddy_run is not None
                else np.asarray([], dtype=float)
            ),
        },
        args.segment_reference,
    )
    joint_kinematics_audits: dict[str, dict[str, Any]] = {
        "captury": audit_joint_kinematics_source(
            "captury_model",
            cap_rotation_metrics,
            captury_metrics.time,
            _source_joint_articulations("captury_model", captury.source_kind),
            {},
        ),
        "motive": audit_joint_kinematics_source(
            "motive_model",
            mot_rotation_metrics,
            motive_metrics.time,
            _source_joint_articulations("motive_model", motive.source_kind),
            {},
        ),
    }
    if biobuddy_run is not None:
        biobuddy_evidence = load_biobuddy_audit_sidecars(args.biobuddy_biomod)
        biobuddy_static_sidecar = Path(args.biobuddy_biomod).with_suffix(
            ".isb_static.json"
        )
        biobuddy_frame_corrections = frame_corrections_from_static_audit(
            biobuddy_evidence["static_evaluation"]
        )
        joint_kinematics_audits["biobuddy"] = audit_joint_kinematics_source(
            "biobuddy_motive57",
            biobuddy_run.rotations_native,
            biobuddy_run.time,
            _source_joint_articulations("biobuddy_motive57", "biomod"),
            biobuddy_frame_corrections,
            frame_correction_provenance={
                "path": str(biobuddy_static_sidecar),
                "sidecar_sha256": file_sha256(biobuddy_static_sidecar),
                "biomod_sha256": biobuddy_evidence["static_evaluation"].get(
                    "biomod_sha256"
                ),
                "status": biobuddy_evidence["static_evaluation"].get("status"),
            },
        )
    joint_kinematics_paths = write_joint_kinematics_audit(
        trial_dir, joint_kinematics_audits
    )
    occlusion_rows, occlusion_figure = analyze_motive_occlusions(
        bundle.motive_c3d,
        trial_dir,
        bundle.name,
        generate_figure=not args.no_figures,
    )
    event_report, contact_rows = detect_trial_events_and_contacts(
        bundle.motive_c3d,
        trial_dir,
        bundle.name,
        time_start_s=cut_start_s,
        time_end_s=cut_end_s,
    )
    dimension_rows = model_dimension_rows(bundle.name, [captury, motive])
    biobuddy_dimension_extra_rows, biobuddy_dimension_report = biobuddy_dimension_rows(
        bundle.name, args.biobuddy_biomod, args.biobuddy_unit_scale_to_m
    )
    dimension_rows.extend(biobuddy_dimension_extra_rows)
    marker_proposal, marker_proposal_report = propose_marker_correspondences(
        bundle.motive_c3d,
        bundle.captury_c3d,
        marker_rotation,
        marker_translation,
        args.angle_label_regex,
        temporal_lag_s,
        captury_coordinates.own_c3d_to_common,
    )
    marker_proposal_path = trial_dir / "skin_marker_correspondence_proposal.json"
    marker_proposal_path.write_text(
        json.dumps(marker_proposal, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.landmark_map is None:
        landmark_map = marker_proposal
        marker_map_source = "automatic_proposal"
    else:
        landmark_map = load_landmark_map(args.landmark_map)
        marker_map_source = str(args.landmark_map)
    marker_rows, marker_ts_rows = marker_correspondence_rows(
        bundle.name,
        bundle.motive_c3d,
        bundle.captury_c3d,
        marker_rotation,
        marker_translation,
        landmark_map,
        temporal_lag_s,
        captury_coordinates.own_c3d_to_common,
    )
    phase_points = int(getattr(args, "phase_normalization_points", 101))
    normalized_scientific_rows: list[dict[str, Any]] = []
    for family, rows, groups, values in (
        (
            "joint_centres",
            centre_ts_rows,
            ("joint",),
            (
                "distance_mm",
                "biobuddy_distance_mm",
                "captury_x_mm",
                "captury_y_mm",
                "captury_z_mm",
                "motive_x_mm",
                "motive_y_mm",
                "motive_z_mm",
            ),
        ),
        (
            "kinematics_q",
            q_ts_rows,
            ("q_name",),
            ("motive", "captury", "biobuddy", "captury_c3d", "difference"),
        ),
        (
            "segment_rotations",
            segment_ts_rows,
            ("reference", "source", "segment"),
            ("global_deg", "x_deg", "y_deg", "z_deg"),
        ),
        (
            "skin_markers",
            marker_ts_rows,
            ("landmark",),
            ("error_x_mm", "error_y_mm", "error_z_mm", "distance_mm"),
        ),
    ):
        existing_values = tuple(
            value for value in values if any(value in row for row in rows)
        )
        if not existing_values:
            continue
        normalized = normalize_timeseries_groups(
            rows,
            time_key="time",
            group_keys=groups,
            value_keys=existing_values,
            start_s=cut_start_s,
            end_s=cut_end_s,
            n_points=phase_points,
        )
        for row in normalized:
            row["family"] = family
        normalized_scientific_rows.extend(normalized)
    cycles = contact_cycle_report(contact_rows)
    summary_tables = (
        centre_rows,
        calibration_centre_rows,
        q_rows,
        c3d_angle_rows,
        segment_rows,
        dimension_rows,
        marker_rows,
    )
    for rows in summary_tables:
        for row in rows:
            row.setdefault("participant", bundle.participant or "")
    quality = metric_quality_report(q_rows)
    sensitivity = metric_sensitivity_report(
        root_policies={
            "captury": captury.root_offset_policy,
            "motive": motive.root_offset_policy,
        },
        temporal=temporal_synchronization,
        evaluation_centre_rows=centre_rows,
        calibration_centre_rows=calibration_centre_rows,
        rotation_audits={
            "captury": captury_source_audit,
            "motive": motive_source_audit,
        },
    )
    write_rows(trial_dir / "joint_centre_metrics.csv", centre_rows)
    write_table_npz(trial_dir / "joint_centre_timeseries.npz", centre_ts_rows)
    write_rows(
        trial_dir / "alignment_calibration_centre_metrics.csv",
        calibration_centre_rows,
    )
    write_table_npz(
        trial_dir / "alignment_calibration_centre_timeseries.npz",
        calibration_centre_ts_rows,
    )
    spatial_calibration_path = trial_dir / "spatial_calibration.json"
    spatial_calibration_path.write_text(
        json.dumps(spatial_calibration.to_dict(), indent=2), encoding="utf-8"
    )
    write_rows(trial_dir / "kinematics_q_metrics.csv", q_rows)
    write_table_npz(trial_dir / "kinematics_q_timeseries.npz", q_ts_rows)
    write_rows(trial_dir / "captury_c3d_angle_metrics.csv", c3d_angle_rows)
    write_table_npz(trial_dir / "captury_c3d_angle_timeseries.npz", c3d_angle_ts_rows)
    c3d_angle_decode_path = trial_dir / "captury_c3d_angle_decode.json"
    c3d_angle_decode_path.write_text(
        json.dumps(c3d_angle_decode_report, indent=2), encoding="utf-8"
    )
    write_rows(trial_dir / "segment_rotation_metrics.csv", segment_rows)
    write_table_npz(trial_dir / "segment_rotation_timeseries.npz", segment_ts_rows)
    write_rows(trial_dir / "model_dimensions.csv", dimension_rows)
    write_rows(trial_dir / "skin_marker_correspondence_metrics.csv", marker_rows)
    write_table_npz(
        trial_dir / "skin_marker_correspondence_timeseries.npz", marker_ts_rows
    )
    temporal_sync_path = trial_dir / "temporal_synchronization.json"
    temporal_sync_path.write_text(
        json.dumps(temporal_synchronization, indent=2), encoding="utf-8"
    )
    write_table_npz(
        trial_dir / "temporal_synchronization_timeseries.npz", temporal_rows
    )
    write_table_npz(trial_dir / "temporal_phase_normalized.npz", phase_rows)
    write_table_npz(
        trial_dir / "phase_normalized_scientific_timeseries.npz",
        normalized_scientific_rows,
    )
    contact_cycles_path = trial_dir / "contact_cycles.json"
    contact_cycles_path.write_text(json.dumps(cycles, indent=2), encoding="utf-8")
    metric_quality_path = trial_dir / "metric_quality.json"
    metric_quality_path.write_text(json.dumps(quality, indent=2), encoding="utf-8")
    metric_sensitivity_path = trial_dir / "metric_sensitivity.json"
    metric_sensitivity_path.write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )
    plot_metric_barh(
        pd.DataFrame(dimension_rows),
        category="dimension",
        metric="median_length_mm",
        output_path=trial_dir / "figures" / "model_dimensions" / "median_length_mm.png",
        title="Model dimensions - median_length_mm",
        xlabel="median_length_mm",
    )
    plot_metric_barh(
        pd.DataFrame(marker_rows),
        category="landmark",
        metric="median_error_mm",
        output_path=trial_dir / "figures" / "skin_markers" / "median_error_mm.png",
        title="Skin marker correspondences - median_error_mm",
        xlabel="median_error_mm",
    )
    vertical_report = vertical_amplitude_report(enriched_c3d)
    report: dict[str, Any] = {
        "trial": bundle.name,
        "participant": bundle.participant,
        "files": {
            "captury_c3d": str(bundle.captury_c3d),
            "captury_bvh": str(bundle.captury_bvh) if bundle.captury_bvh else None,
            "captury_fbx": str(bundle.captury_fbx) if bundle.captury_fbx else None,
            "motive_c3d": str(bundle.motive_c3d),
            "motive_bvh": str(bundle.motive_bvh) if bundle.motive_bvh else None,
            "motive_fbx": str(bundle.motive_fbx) if bundle.motive_fbx else None,
        },
        "models": {
            "captury": {
                "source_kind": captury.source_kind,
                "biomod": str(captury.biomod_path),
                "unit_scale_to_m": captury.unit_scale_to_m,
                "mesh": captury.mesh_report,
                "n_q": int(captury.q.shape[0]),
                "n_frames": int(captury.q.shape[1]),
                "root_offset_policy": captury.root_offset_policy,
            },
            "motive": {
                "source_kind": motive.source_kind,
                "biomod": str(motive.biomod_path),
                "unit_scale_to_m": motive.unit_scale_to_m,
                "mesh": motive.mesh_report,
                "n_q": int(motive.q.shape[0]),
                "n_frames": int(motive.q.shape[1]),
                "root_offset_policy": motive.root_offset_policy,
            },
            "biobuddy": biobuddy_dimension_report,
        },
        "axis_conversion": {
            "requested_model_to_own_c3d": {
                "legacy": args.model_to_c3d_axis,
                "captury": captury_axis_mode,
                "motive": motive_axis_mode,
            },
            "captury": captury_coordinates.to_report(),
            "motive": motive_coordinates.to_report(),
        },
        "time_window": {
            "cut_mode": args.cut_mode,
            "effective_cut_mode": effective_cut_mode,
            "manual_start_s": args.time_start,
            "manual_end_s": args.time_end,
            "used_start_s": cut_start_s,
            "used_end_s": cut_end_s,
            "captury_frames": int(captury_metrics.time.shape[0]),
            "motive_frames": int(motive_metrics.time.shape[0]),
        },
        "temporal_synchronization": temporal_synchronization,
        "alignment": alignment_report,
        "spatial_calibration": spatial_calibration.to_dict(),
        "outputs": {
            "enriched_c3d": str(enriched_c3d),
            "joint_centre_metrics": str(trial_dir / "joint_centre_metrics.csv"),
            "joint_centre_timeseries": str(trial_dir / "joint_centre_timeseries.npz"),
            "alignment_calibration_centre_metrics": str(
                trial_dir / "alignment_calibration_centre_metrics.csv"
            ),
            "alignment_calibration_centre_timeseries": str(
                trial_dir / "alignment_calibration_centre_timeseries.npz"
            ),
            "spatial_calibration": str(spatial_calibration_path),
            "kinematics_q_metrics": str(trial_dir / "kinematics_q_metrics.csv"),
            "kinematics_q_timeseries": str(trial_dir / "kinematics_q_timeseries.npz"),
            "captury_c3d_angle_metrics": str(
                trial_dir / "captury_c3d_angle_metrics.csv"
            ),
            "captury_c3d_angle_timeseries": str(
                trial_dir / "captury_c3d_angle_timeseries.npz"
            ),
            "captury_c3d_angle_decode": str(c3d_angle_decode_path),
            "segment_rotation_metrics": str(trial_dir / "segment_rotation_metrics.csv"),
            "segment_rotation_timeseries": str(
                trial_dir / "segment_rotation_timeseries.npz"
            ),
            "joint_kinematics_d4_d6": str(joint_kinematics_paths["json"]),
            "joint_kinematics_d4_d6_timeseries": str(
                joint_kinematics_paths["timeseries"]
            ),
            "motive_marker_occlusions": str(trial_dir / "motive_marker_occlusions.csv"),
            "trial_events_contacts": str(trial_dir / "trial_events_contacts.csv"),
            "temporal_synchronization": str(temporal_sync_path),
            "temporal_synchronization_timeseries": str(
                trial_dir / "temporal_synchronization_timeseries.npz"
            ),
            "temporal_phase_normalized": str(
                trial_dir / "temporal_phase_normalized.npz"
            ),
            "phase_normalized_scientific_timeseries": str(
                trial_dir / "phase_normalized_scientific_timeseries.npz"
            ),
            "contact_cycles": str(contact_cycles_path),
            "metric_quality": str(metric_quality_path),
            "metric_sensitivity": str(metric_sensitivity_path),
            "model_dimensions": str(trial_dir / "model_dimensions.csv"),
            "skin_marker_correspondence_metrics": str(
                trial_dir / "skin_marker_correspondence_metrics.csv"
            ),
            "skin_marker_correspondence_timeseries": str(
                trial_dir / "skin_marker_correspondence_timeseries.npz"
            ),
            "skin_marker_correspondence_proposal": str(marker_proposal_path),
        },
        "skin_marker_correspondence": {
            "map_source": marker_map_source,
            "proposal": marker_proposal_report,
            "alignment": {
                "method": "captury_to_motive_joint_centres_then_motive_model_to_c3d_markers",
                "rotation": marker_rotation.tolist(),
                "translation_mm": marker_translation.tolist(),
            },
        },
        "trial_events": event_report,
        "contact_cycles": cycles,
        "metric_quality": quality,
        "metric_sensitivity": sensitivity,
        "segment_rotations": segment_report,
        "joint_kinematics_d4_d6": {
            source: {
                "source_id": audit["source_id"],
                "articulations": audit["articulations"],
            }
            for source, audit in joint_kinematics_audits.items()
        },
        "bvh_fbx_rotation_audit": {
            "captury": captury_source_audit,
            "motive": motive_source_audit,
        },
        "segment_orientation_corrections": segment_orientation_report,
        "q_reexpression": q_reexpression_report,
        "occlusion_figure": str(occlusion_figure) if occlusion_figure else None,
        "occlusion_marker_count": len(occlusion_rows),
        "angle_inventory": {
            "captury": c3d_angle_inventory(bundle.captury_c3d, args.angle_label_regex),
            "motive": c3d_angle_inventory(bundle.motive_c3d, args.angle_label_regex),
        },
        "captury_c3d_angle_decode": c3d_angle_decode_report,
        "duplicate_label_inventory": {
            "captury": duplicate_label_inventory(bundle.captury_c3d),
            "motive": duplicate_label_inventory(bundle.motive_c3d),
        },
        "vertical_amplitude": vertical_report,
        "limitations": [
            "BVH/FBX generalized-coordinate comparisons use matching q names only.",
            "Euler angle values can differ because Captury and Motive may export different axis orders and local segment frames.",
            "Joint-centre distances are the primary spatial comparison after static rigid alignment.",
        ],
        "cache": {
            "version": CACHE_VERSION,
            "hit": False,
            "fingerprint": cache_fingerprint,
        },
    }
    report["biobuddy_ik_batch"] = biobuddy_ik_report
    if args.visualize and (
        args.visualize_trial is None or args.visualize_trial == bundle.name
    ):
        visualize_enriched_c3d(enriched_c3d, args.rerun_wait_seconds, args.headless)
    (trial_dir / "run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report, spatial_calibration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare P6 Captury and Motive BVH/FBX/C3D trials."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("captury", "motive", "biobuddy"),
        default=["captury", "motive", "biobuddy"],
        help="Sources to include when available, resolved independently for each trial.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--trial", action="append", default=[], help="Trial name filter. Repeatable."
    )
    parser.add_argument(
        "--joint-filter",
        action="append",
        default=[],
        help="Regex filter for compared joint centres. Repeatable.",
    )
    parser.add_argument(
        "--static-trial",
        default="Static",
        help="Trial used to compute Captury -> Motive alignment.",
    )
    parser.add_argument(
        "--time-start",
        type=float,
        default=None,
        help="Manual analysis window start in seconds.",
    )
    parser.add_argument(
        "--time-end",
        type=float,
        default=None,
        help="Manual analysis window end in seconds.",
    )
    parser.add_argument(
        "--cut-mode",
        choices=["manual", "movement", "full"],
        default="manual",
        help=(
            "Trial cutting mode: manual uses --time-start/--time-end when provided, "
            "movement uses detected movement bounds, full ignores both."
        ),
    )
    parser.add_argument(
        "--temporal-sync-mode",
        choices=["auto", "manual", "none"],
        default="auto",
        help=(
            "Captury-to-Motive clock synchronization. Auto estimates one constant "
            "lag from common joint-centre speeds; manual requires --manual-lag-s."
        ),
    )
    parser.add_argument(
        "--manual-lag-s",
        type=float,
        default=None,
        help=(
            "Manual Captury lag in seconds. Positive values place Captury samples "
            "later on the Motive clock."
        ),
    )
    parser.add_argument(
        "--max-lag-s",
        type=float,
        default=0.5,
        help="Maximum absolute lag searched in automatic mode (default: 0.5 s).",
    )
    parser.add_argument(
        "--phase-normalization-points",
        type=int,
        default=101,
        help="Number of samples used to normalize the selected phase to 0-100%%.",
    )
    parser.add_argument("--model-source", choices=["auto", "bvh", "fbx"], default="bvh")
    parser.add_argument(
        "--audit-bvh-fbx-rotations",
        action="store_true",
        help=(
            "Build both model exports when available and report their canonical "
            "segment-frame SO(3) deviations without changing an explicit source choice."
        ),
    )
    parser.add_argument(
        "--bvh-fbx-max-p95-geodesic-deg",
        type=float,
        default=5.0,
        help=(
            "Maximum per-segment p95 geodesic deviation accepted by automatic "
            "BVH/FBX source selection (default: 5 degrees)."
        ),
    )
    parser.add_argument(
        "--root-offset-mode",
        choices=["auto", "subtract", "keep"],
        default="auto",
        help=(
            "How to handle static BVH/FBX root offsets: auto scores both "
            "subtract and keep conventions against the matching C3D marker cloud."
        ),
    )
    parser.add_argument(
        "--captury-root-offset-mode",
        choices=["auto", "subtract", "keep"],
        default="keep",
        help=(
            "Captury root-translation policy (default: keep). Dynamic trials "
            "reuse the Static choice."
        ),
    )
    parser.add_argument(
        "--motive-root-offset-mode",
        choices=["auto", "subtract", "keep"],
        default=None,
        help=(
            "Motive root-translation policy. When omitted, inherits "
            "--root-offset-mode; dynamic trials reuse the Static choice."
        ),
    )
    parser.add_argument(
        "--spatial-alignment-mode",
        choices=["held_out_centres", "legacy_all_centres"],
        default="held_out_centres",
        help=(
            "Static Captury-to-Motive calibration protocol. The default reserves "
            "calibration centres and excludes them from primary evaluation metrics."
        ),
    )
    parser.add_argument(
        "--alignment-calibration-centre",
        action="append",
        default=[],
        help=(
            "Exact joint-centre name reserved for static alignment. Repeatable. "
            "Defaults to Hips, Head, LeftShoulder and RightShoulder."
        ),
    )
    parser.add_argument(
        "--model-to-c3d-axis",
        choices=["auto", "y_up_to_z_up", "identity"],
        default="auto",
        help=(
            "Legacy shared mode. Use auto for Captury/Motive comparisons; "
            "manual modes must use the system-specific options below."
        ),
    )
    parser.add_argument(
        "--captury-model-to-c3d-axis",
        choices=["auto", "y_up_to_z_up", "identity"],
        default=None,
        help=(
            "Optional Captury model -> Captury-own-C3D basis override. "
            "Default auto is +Y -> +Y."
        ),
    )
    parser.add_argument(
        "--motive-model-to-c3d-axis",
        choices=["auto", "y_up_to_z_up", "identity"],
        default=None,
        help=(
            "Optional Motive model -> Motive-own-C3D basis override. "
            "Default auto is +Y -> +Z."
        ),
    )
    parser.add_argument(
        "--captury-q-landmarks-json",
        type=Path,
        default=None,
        help=(
            "Optional Captury Q_* occurrence/role configuration. The bundled "
            "captury_q_landmarks.json is used when omitted."
        ),
    )
    captury_frame_mode = parser.add_mutually_exclusive_group()
    captury_frame_mode.add_argument(
        "--captury-rename-axes",
        action="store_true",
        help=(
            "Apply the simple global Captury local-axis relabelling R_y(90 deg). "
            "It is a diagnostic baseline, not an ISB calibration."
        ),
    )
    captury_frame_mode.add_argument(
        "--captury-calibrate-segment-frames",
        action="store_true",
        help=(
            "Fit optional Captury pelvis/thigh local-frame matrices from its own "
            "static Q_* C3D landmarks and Captury model joint centres."
        ),
    )
    parser.add_argument("--captury-unit-scale-to-m", type=float, default=None)
    parser.add_argument("--motive-unit-scale-to-m", type=float, default=None)
    parser.add_argument(
        "--biobuddy-biomod",
        type=Path,
        default=None,
        help=(
            "Optional BioBuddy/Biorbd model to include as a third source in "
            "model-dimension comparisons."
        ),
    )
    parser.add_argument(
        "--biobuddy-unit-scale-to-m",
        type=float,
        default=1.0,
        help=(
            "Scale applied to neutral BioBuddy model coordinates before "
            "dimension reporting. Default assumes the bioMod is in metres."
        ),
    )
    parser.add_argument("--angle-label-regex", default=ANGLE_LABEL_REGEX)
    parser.add_argument(
        "--landmark-map",
        type=Path,
        default=None,
        help="Optional JSON map for non-joint-centre Motive/Captury marker pairs.",
    )
    parser.add_argument(
        "--c3d-angle-unit",
        choices=["deg", "rad"],
        default="deg",
        help=(
            "Unit used by Captury C3D angle channels stored in POINT. Captury P6 "
            "does not expose dedicated angle-unit metadata, so the default 'deg' "
            "is an explicit assumption."
        ),
    )
    parser.add_argument(
        "--segment-reference",
        choices=["biobuddy", "motive", "captury"],
        default="biobuddy",
        help="Reference source for segment rotation deviation metrics.",
    )
    parser.add_argument(
        "--captury-reorient-thigh-y-from-cor",
        action="store_true",
        help=(
            "Correct Captury thigh segment frames by orienting the local Y axis "
            "from hip CoR to knee CoR before segment-angle metrics."
        ),
    )
    parser.add_argument(
        "--rotate-body-segments-180-x",
        action="store_true",
        help=(
            "Rotate Captury and Motive segment frames by local R(x, 180 deg) "
            "followed by local R(y, 180 deg) before segment-angle metrics."
        ),
    )
    parser.add_argument(
        "--reexpress-rotations-zxy",
        action="store_true",
        help=(
            "Re-express Captury and Motive rotational q channels from corrected "
            "segment rotation matrices in Z-X-Y order before q metrics. BioBuddy "
            "q remains unchanged."
        ),
    )
    parser.add_argument(
        "--no-mesh", action="store_true", help="Skip FBX visual mesh extraction."
    )
    parser.add_argument("--max-mesh-points", type=int, default=0)
    parser.add_argument(
        "--disable-static-model-alignment",
        action="store_true",
        help="Disable the static Captury model -> Motive model rigid alignment.",
    )
    parser.add_argument(
        "--disable-motive-marker-alignment",
        action="store_true",
        help="Disable the Motive model -> Motive C3D marker-proxy yaw/translation alignment.",
    )
    parser.add_argument(
        "--run-biobuddy-ik-batch",
        "--run-ik-batch",
        dest="run_ik_batch",
        action="store_true",
        help=(
            "Run cached nonlinear TRF IK only for the BioBuddy model. The old "
            "--run-ik-batch spelling remains as a compatibility alias."
        ),
    )
    parser.add_argument("--ik-max-frames", type=int, default=0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize-trial", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rerun-wait-seconds", type=float, default=1.0)
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip PNG metric figure generation."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force recomputation even when trial outputs match the cache.",
    )
    parser.add_argument(
        "--occlusions-only",
        action="store_true",
        help="Only compute Motive marker occlusion CSV outputs.",
    )
    parser.add_argument("--list-trials", action="store_true")
    return parser.parse_args()


def run_occlusions_only(
    trials: list[TrialBundle], args: argparse.Namespace
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for bundle in trials:
        trial_dir = args.out_dir / safe_name(bundle.name)
        trial_dir.mkdir(parents=True, exist_ok=True)
        if bundle.motive_c3d is None:
            write_rows(trial_dir / "motive_marker_occlusions.csv", [])
            reports.append(
                {
                    "trial": bundle.name,
                    "status": "unavailable",
                    "reason": "no_motive_c3d",
                }
            )
            continue
        rows, figure = analyze_motive_occlusions(
            bundle.motive_c3d,
            trial_dir,
            bundle.name,
            generate_figure=not args.no_figures,
        )
        all_rows.extend(rows)
        reports.append(
            {
                "trial": bundle.name,
                "motive_c3d": str(bundle.motive_c3d),
                "rows": len(rows),
                "output": str(trial_dir / "motive_marker_occlusions.csv"),
                "figure": str(figure) if figure else None,
            }
        )
    write_rows(args.out_dir / "all_motive_marker_occlusions.csv", all_rows)
    (args.out_dir / "run_report_occlusions.json").write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "out_dir": str(args.out_dir),
                "n_trials": len(trials),
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Computed occlusions for {len(trials)} trial(s).")
    print(f"Occlusions: {args.out_dir / 'all_motive_marker_occlusions.csv'}")
    return all_rows


def main() -> None:
    args = parse_args()
    if args.model_to_c3d_axis != "auto":
        raise ValueError(
            "--model-to-c3d-axis is a legacy shared option and must be auto "
            "for Captury/Motive comparisons. Select source-specific overrides "
            "with --captury-model-to-c3d-axis and "
            "--motive-model-to-c3d-axis."
        )
    discovered_trials = discover_trials(args.data_root)
    if not any(bundle.name == args.static_trial for bundle in discovered_trials):
        candidates = [
            bundle.name
            for bundle in discovered_trials
            if bundle.name.endswith("_" + args.static_trial)
        ]
        if len(candidates) == 1:
            args.static_trial = candidates[0]
    if args.list_trials:
        for bundle in discovered_trials:
            print(bundle.name)
        return
    trials = list(discovered_trials)
    if args.trial:
        requested = set(args.trial)
        available = {bundle.name for bundle in trials}
        for name in list(requested - available):
            matches = [trial for trial in available if trial.endswith("_" + name)]
            if len(matches) == 1:
                requested.remove(name)
                requested.add(matches[0])
        trials = [bundle for bundle in trials if bundle.name in requested]
    if not trials:
        raise RuntimeError(f"No trials found in {args.data_root}.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    provenance_trials = (
        trials
        if args.occlusions_only
        else provenance_trials_with_static(trials, discovered_trials, args.static_trial)
    )
    if args.occlusions_only:
        provenance_path = write_comparison_provenance_manifest(provenance_trials, args)
        run_occlusions_only(trials, args)
        print(f"Provenance: {provenance_path}")
        return

    biobuddy_audit_evidence = load_biobuddy_audit_sidecars(
        args.biobuddy_biomod if "biobuddy" in args.sources else None
    )
    convention_registry = load_kinematic_conventions()
    isb_audit = build_isb_d1_d3_audit(
        convention_registry,
        biomod_verification=biobuddy_audit_evidence["biomod_verification"],
        static_evaluation=biobuddy_audit_evidence["static_evaluation"],
    )
    isb_audit_paths = write_isb_d1_d3_audit(args.out_dir, isb_audit)
    scientific_artifacts = {
        "scientific/isb_d1_d3_json": isb_audit_paths["json"],
        "scientific/isb_d1_d3_table": isb_audit_paths["table"],
    }
    provenance_path = write_comparison_provenance_manifest(
        provenance_trials, args, batch_artifacts=scientific_artifacts
    )

    static_bundle, _ = split_static_calibration_trial(
        discovered_trials, args.static_trial
    )
    if static_bundle is None:
        raise RuntimeError(
            f"Static calibration trial {args.static_trial!r} was not found. "
            "Choose it with --static-trial before comparing dynamic trials."
        )
    static_requested = not args.trial or any(
        bundle.name == args.static_trial for bundle in trials
    )
    _, trials = split_static_calibration_trial(trials, args.static_trial)
    from comparison_workflow import ComparisonWorkflow

    workflow = ComparisonWorkflow(static_bundle, args, api=sys.modules[__name__])
    reports: list[dict[str, Any]] = []
    provenance_reports: list[dict[str, Any]] = []
    if static_requested:
        static_report = workflow.process(static_bundle, args.out_dir)
        reports.append(static_report)
        provenance_reports.append(static_report)
    for bundle in trials:
        report = workflow.process(bundle, args.out_dir)
        reports.append(report)
        provenance_reports.append(report)
    provenance_reports.extend(workflow.static_reports)

    all_centre_rows: list[dict[str, Any]] = []
    all_q_rows: list[dict[str, Any]] = []
    all_occlusion_rows: list[dict[str, Any]] = []
    all_dimension_rows: list[dict[str, Any]] = []
    all_marker_rows: list[dict[str, Any]] = []
    all_segment_rows: list[dict[str, Any]] = []
    for report in reports:
        trial_dir = args.out_dir / safe_name(report["trial"])
        centre_csv = trial_dir / "joint_centre_metrics.csv"
        q_csv = trial_dir / "kinematics_q_metrics.csv"
        occlusion_csv = trial_dir / "motive_marker_occlusions.csv"
        dimension_csv = trial_dir / "model_dimensions.csv"
        marker_csv = trial_dir / "skin_marker_correspondence_metrics.csv"
        segment_csv = trial_dir / "segment_rotation_metrics.csv"
        if centre_csv.exists() and centre_csv.stat().st_size:
            all_centre_rows.extend(pd.read_csv(centre_csv).to_dict("records"))
        if q_csv.exists() and q_csv.stat().st_size:
            all_q_rows.extend(pd.read_csv(q_csv).to_dict("records"))
        if occlusion_csv.exists() and occlusion_csv.stat().st_size:
            all_occlusion_rows.extend(pd.read_csv(occlusion_csv).to_dict("records"))
        if dimension_csv.exists() and dimension_csv.stat().st_size:
            all_dimension_rows.extend(pd.read_csv(dimension_csv).to_dict("records"))
        if marker_csv.exists() and marker_csv.stat().st_size:
            all_marker_rows.extend(pd.read_csv(marker_csv).to_dict("records"))
        if segment_csv.exists() and segment_csv.stat().st_size:
            all_segment_rows.extend(pd.read_csv(segment_csv).to_dict("records"))
    normalize_summary_source(all_q_rows, "captury")
    normalize_summary_source(all_centre_rows, "captury")
    write_rows(args.out_dir / "all_joint_centre_metrics.csv", all_centre_rows)
    write_rows(args.out_dir / "all_kinematics_q_metrics.csv", all_q_rows)
    write_rows(args.out_dir / "all_motive_marker_occlusions.csv", all_occlusion_rows)
    write_rows(args.out_dir / "all_model_dimensions.csv", all_dimension_rows)
    write_rows(args.out_dir / "all_segment_rotation_metrics.csv", all_segment_rows)
    write_rows(
        args.out_dir / "all_skin_marker_correspondence_metrics.csv", all_marker_rows
    )
    population_q_rows, population_q_report = aggregate_trial_metrics_by_participant(
        all_q_rows,
        metric_keys=(
            "bias_rad",
            "mae_rad",
            "rmse_rad",
            "bias_native",
            "mae_native",
            "rmse_native",
            "nrmse_range",
            "linear_gain",
            "linear_offset_rad",
            "reference_rom_rad",
            "test_rom_rad",
            "rom_difference_rad",
            "max_time_difference_s",
            "min_time_difference_s",
        ),
        group_keys=("trial", "source", "q_name"),
    )
    population_centre_rows, population_centre_report = (
        aggregate_trial_metrics_by_participant(
            all_centre_rows,
            metric_keys=(
                "median_error_mm",
                "p95_error_mm",
                "mae_euclidean",
                "rmse_euclidean",
            ),
            group_keys=("trial", "source", "joint"),
        )
    )
    population_segment_rows, population_segment_report = (
        aggregate_trial_metrics_by_participant(
            all_segment_rows,
            metric_keys=(
                "median_global_deg",
                "p95_global_deg",
                "rmse_global_deg",
                "median_abs_x_deg",
                "median_abs_y_deg",
                "median_abs_z_deg",
            ),
            group_keys=("trial", "reference", "source", "segment"),
        )
    )
    population_q_path = args.out_dir / "population_kinematics_summary.csv"
    population_centre_path = args.out_dir / "population_joint_centre_summary.csv"
    population_segment_path = args.out_dir / "population_segment_summary.csv"
    population_report_path = args.out_dir / "population_aggregation.json"
    write_rows(population_q_path, population_q_rows)
    write_rows(population_centre_path, population_centre_rows)
    write_rows(population_segment_path, population_segment_rows)
    population_report = {
        "schema_version": 1,
        "statistical_unit": "participant",
        "frames_used_as_population_observations": False,
        "excluded_from_population_aggregation": {
            "limits_of_agreement": (
                "frame-level descriptive bounds are not population LoA"
            ),
            "pearson_r_waveform": "requires Fisher-z aggregation",
            "lin_ccc_waveform": "requires a dedicated repeated-measures model",
        },
        "kinematics": population_q_report,
        "joint_centres": population_centre_report,
        "segments": population_segment_report,
        "outputs": {
            "kinematics": str(population_q_path),
            "joint_centres": str(population_centre_path),
            "segments": str(population_segment_path),
        },
    }
    population_report_path.write_text(
        json.dumps(population_report, indent=2), encoding="utf-8"
    )
    scientific_artifacts.update(
        {
            "scientific/population_kinematics": population_q_path,
            "scientific/population_joint_centres": population_centre_path,
            "scientific/population_segments": population_segment_path,
            "scientific/population_aggregation": population_report_path,
        }
    )
    isb_d1_d6_report = build_isb_d1_d6_report(
        isb_audit, provenance_reports, convention_registry
    )
    isb_d1_d6_paths = write_isb_d1_d6_report(args.out_dir, isb_d1_d6_report)
    reproducibility_manifest = build_reproducibility_manifest(
        args.out_dir, provenance_reports
    )
    reproducibility_manifest_path = write_reproducibility_manifest(
        args.out_dir, reproducibility_manifest
    )
    scientific_artifacts.update(
        {
            "scientific/isb_d1_d6_json": isb_d1_d6_paths["json"],
            "scientific/isb_d1_d6_table": isb_d1_d6_paths["table"],
            "scientific/reproducibility_manifest": reproducibility_manifest_path,
        }
    )
    figures = (
        {
            "joint_centres": [],
            "kinematics_q": [],
            "occlusions": [],
            "model_dimensions": [],
            "skin_markers": [],
        }
        if args.no_figures
        else generate_metric_figures(all_centre_rows, all_q_rows, args.out_dir)
    )
    if not args.no_figures:
        figures.setdefault("occlusions", [])
        figures.setdefault("model_dimensions", [])
        figures.setdefault("skin_markers", [])
        if all_occlusion_rows:
            path = plot_metric_barh(
                pd.DataFrame(all_occlusion_rows),
                category="marker",
                metric="missing_percent",
                output_path=args.out_dir
                / "figures"
                / "occlusions"
                / "missing_percent.png",
                title="Motive marker occlusions - missing_percent",
                xlabel="missing_percent",
            )
            if path:
                figures["occlusions"].append(str(path))
        if all_dimension_rows:
            path = plot_metric_barh(
                pd.DataFrame(all_dimension_rows),
                category="dimension",
                metric="median_length_mm",
                output_path=args.out_dir
                / "figures"
                / "model_dimensions"
                / "median_length_mm.png",
                title="Model dimensions - median_length_mm",
                xlabel="median_length_mm",
            )
            if path:
                figures["model_dimensions"].append(str(path))
        if all_marker_rows:
            for metric in ("median_error_mm", "p95_error_mm", "rmse_error_mm"):
                path = plot_metric_barh(
                    pd.DataFrame(all_marker_rows),
                    category="landmark",
                    metric=metric,
                    output_path=args.out_dir
                    / "figures"
                    / "skin_markers"
                    / f"{metric}.png",
                    title=f"Skin marker correspondences - {metric}",
                    xlabel=metric,
                )
                if path:
                    figures["skin_markers"].append(str(path))
    (args.out_dir / "run_report.json").write_text(
        json.dumps(
            {
                "data_root": str(args.data_root),
                "out_dir": str(args.out_dir),
                "n_trials": len(reports),
                "static_trial": args.static_trial,
                "isb_d1_d3_audit": {
                    "summary": str(isb_audit_paths["json"]),
                    "table": str(isb_audit_paths["table"]),
                },
                "isb_d1_d6_report": {
                    "summary": str(isb_d1_d6_paths["json"]),
                    "table": str(isb_d1_d6_paths["table"]),
                },
                "reproducibility_manifest": str(reproducibility_manifest_path),
                "figures": figures,
                "population_aggregation": population_report,
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    provenance_path = write_comparison_provenance_manifest(
        provenance_trials,
        args,
        provenance_reports,
        batch_artifacts=scientific_artifacts,
    )
    print(f"Compared {len(reports)} trial(s).")
    print(f"Joint-centre metrics: {args.out_dir / 'all_joint_centre_metrics.csv'}")
    print(f"Kinematics metrics: {args.out_dir / 'all_kinematics_q_metrics.csv'}")
    if not args.no_figures:
        print(f"Figures: {args.out_dir / 'figures'}")
    print(f"Report: {args.out_dir / 'run_report.json'}")
    print(f"ISB D1-D3 audit: {isb_audit_paths['json']}")
    print(f"ISB D1-D6 report: {isb_d1_d6_paths['json']}")
    print(f"Reproducibility manifest: {reproducibility_manifest_path}")
    print(f"Provenance: {provenance_path}")


if __name__ == "__main__":
    main()
