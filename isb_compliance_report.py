"""Consolidate ISB D1-D6 evidence and publish reproducible curve recipes.

The segment-frame audit (D1-D3) and joint-kinematics audit (D4-D6) deliberately
have different scopes.  This module normalizes their metadata for reporting
without changing the underlying scientific conclusions:

* D1-D3 rows have ``trial_id=None`` and target one segment;
* D4-D6 rows target one articulation in one trial;
* Captury model files, Captury C3D angle channels, Motive model files and the
  BioBuddy model remain distinct sources;
* missing or undecoded sources are explicit blockers and never borrow data
  from another source.

The reproducibility manifest is likewise descriptive.  It fingerprints JSON
and NPZ artifacts already written by the CLI and records enough information to
read table-like time series without importing GUI code.  It does not infer
units or anatomical semantics beyond conservative column-name conventions.
"""

from __future__ import annotations

import hashlib
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

SOURCE_ORDER = (
    "captury_model",
    "captury_c3d_angles",
    "motive_model",
    "biobuddy_motive57",
)

CURVE_FILTER_KEYS = (
    "trial",
    "participant",
    "family",
    "reference",
    "source",
    "system",
    "joint",
    "segment",
    "q_name",
    "landmark",
    "marker",
    "dimension",
)

SHORT_SOURCE_IDS = {
    "captury": "captury_model",
    "motive": "motive_model",
    "biobuddy": "biobuddy_motive57",
}

BLOCKING_STATUS_CLASSES = {
    "deviation",
    "unknown",
    "not_evaluated",
    "blocked",
    "unavailable",
}


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _status_class(raw_status: object) -> str:
    value = str(raw_status or "unknown").strip().lower()
    if value in {"conforme", "conforme_target"}:
        return "conformant"
    if value in {"jcs_target_evaluated", "euler_target_evaluated"}:
        return "conformant"
    if value == "deviation" or value.startswith("deviation_"):
        return "deviation"
    if value in {"inconnu", "unknown"}:
        return "unknown"
    if value in {"documente_non_evalue", "not_evaluated"}:
        return "not_evaluated"
    if value.startswith("non_applicable"):
        return "not_applicable"
    if value.startswith("blocked"):
        return "blocked"
    if value.startswith("unavailable") or value in {
        "missing",
        "missing_source",
        "not_run",
        "stale",
    }:
        return "unavailable"
    return "unknown"


def _confidence(
    *,
    source_id: str,
    status_class: str,
    numeric_status: str | None = None,
    runtime_template_status: str | None = None,
    euler_status: str | None = None,
) -> tuple[str, str]:
    if status_class in {"unknown", "unavailable"}:
        return "unknown", "No source-specific convention was demonstrated."
    if numeric_status == "available_from_static_landmarks":
        return "measured_static", "Compared numerically from static landmarks."
    if euler_status == "available":
        return (
            "computed_anatomical",
            "Computed after source-to-anatomical frame corrections.",
        )
    if source_id == "biobuddy_motive57" and runtime_template_status == "available":
        return "runtime_template", "Read from the active BioBuddy runtime template."
    return "declared", "Read from a versioned convention or target registry."


def _source_format(
    source_id: str, trial_report: Mapping[str, Any] | None = None
) -> str:
    if source_id == "captury_c3d_angles":
        return "c3d"
    if source_id == "biobuddy_motive57":
        return "biomod"
    models = (trial_report or {}).get("models", {})
    key = "captury" if source_id == "captury_model" else "motive"
    model = models.get(key, {}) if isinstance(models, Mapping) else {}
    if isinstance(model, Mapping) and model.get("source_kind"):
        return str(model["source_kind"])
    return "bvh_or_fbx"


def _blockers(raw_status: object, status_class: str) -> list[str]:
    value = str(raw_status or "").lower()
    if status_class == "not_applicable" or status_class == "conformant":
        return []
    if "missing_frame_correction" in value:
        return ["missing_frame_correction"]
    if "missing_segment" in value:
        return ["missing_segment"]
    if value == "missing_source":
        return ["missing_source"]
    if value == "missing_articulation":
        return ["missing_articulation"]
    if value == "unknown_euler_sequence":
        return ["unknown_euler_sequence"]
    if status_class == "deviation":
        return ["non_harmonized_isb_deviation"]
    if status_class == "not_evaluated":
        return ["convention_not_evaluated"]
    if status_class == "unknown":
        return ["unknown_convention"]
    if status_class == "unavailable":
        return ["unavailable_source_or_result"]
    return ["blocked_comparison"]


def _normalized_row(
    *,
    source_id: str,
    source_format: str,
    trial_id: str | None,
    scope_kind: str,
    entity_id: str,
    deviation_id: str,
    raw_status: object,
    evidence: Iterable[object] = (),
    numeric_status: str | None = None,
    numeric_value: object = None,
    numeric_unit: str = "",
    runtime_template_status: str | None = None,
    euler_status: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    status_class = _status_class(raw_status)
    confidence, confidence_reason = _confidence(
        source_id=source_id,
        status_class=status_class,
        numeric_status=numeric_status,
        runtime_template_status=runtime_template_status,
        euler_status=euler_status,
    )
    blocker_codes = _blockers(raw_status, status_class)
    return {
        "source_id": source_id,
        "source_format": source_format,
        "trial_id": trial_id,
        "scope_kind": scope_kind,
        "entity_id": entity_id,
        "deviation_id": deviation_id,
        "raw_status": str(raw_status),
        "status_class": status_class,
        "confidence": confidence,
        "confidence_reason": confidence_reason,
        "blocking": bool(status_class in BLOCKING_STATUS_CLASSES),
        "blocker_codes": list(blocker_codes),
        "evidence": [str(item) for item in evidence],
        "numeric_status": str(numeric_status or "not_available"),
        "numeric_value": numeric_value,
        "numeric_unit": numeric_unit,
        "details": dict(details or {}),
    }


def _d1_d3_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for segment in report.get("segments", []):
        source_id = str(segment["source_id"])
        for deviation_id in ("D1", "D2", "D3"):
            numeric_value: object = None
            numeric_unit = ""
            if (
                deviation_id == "D1"
                and segment.get("angular_deviation_deg") is not None
            ):
                numeric_value = segment.get("angular_deviation_deg")
                numeric_unit = "deg"
            elif (
                deviation_id == "D3" and segment.get("origin_deviation_mm") is not None
            ):
                numeric_value = segment.get("origin_deviation_mm")
                numeric_unit = "mm"
            rows.append(
                _normalized_row(
                    source_id=source_id,
                    source_format=_source_format(source_id),
                    trial_id=None,
                    scope_kind="segment",
                    entity_id=str(segment["segment_id"]),
                    deviation_id=deviation_id,
                    raw_status=segment.get(f"{deviation_id}_status", "inconnu"),
                    evidence=segment.get(f"{deviation_id}_evidence", []),
                    numeric_status=str(segment.get("numeric_status", "not_available")),
                    numeric_value=numeric_value,
                    numeric_unit=numeric_unit,
                    runtime_template_status=str(
                        segment.get("runtime_template_status", "not_applicable")
                    ),
                    details={
                        "source_name": segment.get("source_name"),
                        "parent": segment.get("parent"),
                        "rotation_vector_deg": segment.get("rotation_vector_deg"),
                    },
                )
            )
    return rows


def _registry_articulations(
    registry: Mapping[str, Any], source_id: str
) -> Mapping[str, Any]:
    source = registry.get("sources", {}).get(source_id, {})
    articulations = (
        source.get("articulations", {}) if isinstance(source, Mapping) else {}
    )
    return articulations if isinstance(articulations, Mapping) else {}


def _audit_by_source(trial_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    raw = trial_report.get("joint_kinematics_d4_d6", {})
    if not isinstance(raw, Mapping):
        return result
    for short_id, audit in raw.items():
        if not isinstance(audit, Mapping):
            continue
        source_id = str(
            audit.get("source_id") or SHORT_SOURCE_IDS.get(str(short_id), short_id)
        )
        result[source_id] = audit
    return result


def _model_d4_d6_rows(
    source_id: str,
    trial_report: Mapping[str, Any],
    registry: Mapping[str, Any],
    audit: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_id = str(trial_report.get("trial", ""))
    registry_articulations = _registry_articulations(registry, source_id)
    audit_articulations = (
        audit.get("articulations", {}) if isinstance(audit, Mapping) else {}
    )
    if not isinstance(audit_articulations, Mapping):
        audit_articulations = {}
    entity_ids = sorted(set(registry_articulations) | set(audit_articulations))
    for entity_id in entity_ids:
        declared = registry_articulations.get(entity_id, {})
        computed = audit_articulations.get(entity_id)
        if audit is None:
            statuses = {deviation: "missing_source" for deviation in ("D4", "D5", "D6")}
            computed = {}
        elif computed is None:
            statuses = {
                deviation: "missing_articulation" for deviation in ("D4", "D5", "D6")
            }
            computed = {}
        else:
            statuses = {
                deviation: computed.get(
                    f"{deviation}_status",
                    declared.get("isb_audit", {})
                    .get(deviation, {})
                    .get("status", "inconnu"),
                )
                for deviation in ("D4", "D5", "D6")
            }
        for deviation_id in ("D4", "D5", "D6"):
            declared_evidence = (
                declared.get("isb_audit", {}).get(deviation_id, {}).get("evidence", [])
                if isinstance(declared, Mapping)
                else []
            )
            rows.append(
                _normalized_row(
                    source_id=source_id,
                    source_format=_source_format(source_id, trial_report),
                    trial_id=trial_id,
                    scope_kind="articulation",
                    entity_id=str(entity_id),
                    deviation_id=deviation_id,
                    raw_status=statuses[deviation_id],
                    evidence=declared_evidence,
                    numeric_status=str(computed.get("matrix_status", "not_available")),
                    euler_status=str(computed.get("euler_status", "unavailable")),
                    details={
                        **{
                            key: computed.get(key, declared.get(key))
                            for key in (
                                "proximal",
                                "distal",
                                "sequence",
                                "type",
                                "matrix_status",
                                "euler_status",
                                "singular_frame_count",
                                "near_singular_frame_count",
                                "max_roundtrip_geodesic_deg",
                                "proximal_frame_correction",
                                "distal_frame_correction",
                            )
                            if computed.get(key, declared.get(key)) is not None
                        },
                        "proximal_segment_id": computed.get(
                            "proximal_segment_id", declared.get("proximal")
                        ),
                        "distal_segment_id": computed.get(
                            "distal_segment_id", declared.get("distal")
                        ),
                    },
                )
            )
    return rows


def _captury_c3d_d4_d6_rows(
    trial_report: Mapping[str, Any], registry: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    trial_id = str(trial_report.get("trial", ""))
    decode = trial_report.get("captury_c3d_angle_decode", {})
    decoded_by_articulation = (
        {
            str(channel.get("articulation")): channel
            for channel in decode.get("channels", [])
            if isinstance(channel, Mapping)
            and channel.get("identity_decoded")
            and channel.get("articulation")
        }
        if isinstance(decode, Mapping)
        else {}
    )
    for entity_id, declared in sorted(
        _registry_articulations(registry, "captury_c3d_angles").items()
    ):
        channel = decoded_by_articulation.get(str(entity_id))
        for deviation_id in ("D4", "D5", "D6"):
            if channel is None:
                raw_status = "missing_articulation"
            elif deviation_id == "D4" and not channel.get("rotation_sequence"):
                raw_status = "unknown_euler_sequence"
            else:
                raw_status = (
                    declared.get("isb_audit", {})
                    .get(deviation_id, {})
                    .get("status", "inconnu")
                )
            evidence = (
                declared.get("isb_audit", {}).get(deviation_id, {}).get("evidence", [])
            )
            rows.append(
                _normalized_row(
                    source_id="captury_c3d_angles",
                    source_format="c3d",
                    trial_id=trial_id,
                    scope_kind="articulation",
                    entity_id=str(entity_id),
                    deviation_id=deviation_id,
                    raw_status=raw_status,
                    evidence=evidence,
                    numeric_status=(
                        "identity_decoded_component_semantics_unknown"
                        if channel is not None
                        else "not_available"
                    ),
                    details={
                        "channel_label": declared.get("channel_label"),
                        "proximal_segment_id": declared.get("proximal"),
                        "distal_segment_id": declared.get("distal"),
                        "rotation_sequence": (
                            channel.get("rotation_sequence") if channel else None
                        ),
                        "component_semantics_status": (
                            decode.get("component_semantics_status")
                            if isinstance(decode, Mapping)
                            else None
                        ),
                    },
                )
            )
    return rows


def _segment_harmonization_statuses(
    d1_d3_rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], bool]:
    statuses: dict[tuple[str, str], list[bool]] = {}
    for row in d1_d3_rows:
        key = (str(row["source_id"]), str(row["entity_id"]))
        statuses.setdefault(key, []).append(
            str(row["status_class"]) in {"conformant", "not_applicable"}
        )
    return {key: bool(values and all(values)) for key, values in statuses.items()}


def _apply_upstream_segment_blockers(
    rows: list[dict[str, Any]], harmonized: Mapping[tuple[str, str], bool]
) -> None:
    for row in rows:
        if row["scope_kind"] != "articulation" or row["deviation_id"] != "D4":
            continue
        details = row.get("details", {})
        endpoints = [
            details.get("proximal_segment_id"),
            details.get("distal_segment_id"),
        ]
        if not all(endpoints):
            blocker = "missing_canonical_segment_endpoints"
            if blocker not in row["blocker_codes"]:
                row["blocker_codes"].append(blocker)
            row["blocking"] = True
            continue
        if all(
            harmonized.get((str(row["source_id"]), str(segment)), False)
            for segment in endpoints
        ):
            continue
        blocker = "upstream_segment_frames_not_harmonized"
        if blocker not in row["blocker_codes"]:
            row["blocker_codes"].append(blocker)
        row["blocking"] = True


def build_isb_d1_d6_report(
    d1_d3_report: Mapping[str, Any],
    trial_reports: Iterable[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one conservative, filterable D1-D6 report."""

    rows = _d1_d3_rows(d1_d3_report)
    d1_d3_rows = list(rows)
    trials = [dict(report) for report in trial_reports]
    for trial_report in trials:
        audits = _audit_by_source(trial_report)
        rows.extend(
            _model_d4_d6_rows(
                "captury_model", trial_report, registry, audits.get("captury_model")
            )
        )
        rows.extend(
            _model_d4_d6_rows(
                "motive_model", trial_report, registry, audits.get("motive_model")
            )
        )
        rows.extend(
            _model_d4_d6_rows(
                "biobuddy_motive57",
                trial_report,
                registry,
                audits.get("biobuddy_motive57"),
            )
        )
        rows.extend(_captury_c3d_d4_d6_rows(trial_report, registry))

    _apply_upstream_segment_blockers(rows, _segment_harmonization_statuses(d1_d3_rows))

    sources = {
        source_id: {
            "display_name": registry.get("sources", {})
            .get(source_id, {})
            .get("display_name", source_id),
            "representation": registry.get("sources", {})
            .get(source_id, {})
            .get("representation", "unknown"),
        }
        for source_id in SOURCE_ORDER
    }
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status_class"])
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "scope": "ISB deviations D1-D6",
        "source_order": list(SOURCE_ORDER),
        "sources": sources,
        "trials": sorted(
            {str(report.get("trial")) for report in trials if report.get("trial")}
        ),
        "status_semantics": dict(registry.get("status_semantics", {})),
        "interpretation": (
            "Unknown, unavailable, unevaluated or deviating conventions block "
            "anatomically harmonized agreement. Non-applicable criteria do not."
        ),
        "status_counts": status_counts,
        "rows": rows,
    }


def _jsonable_cell(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_cell(item) for item in value]
    return value


def _write_table_npz(path: Path, rows: list[Mapping[str, Any]]) -> None:
    columns = list(rows[0]) if rows else []
    payload: dict[str, np.ndarray] = {
        "columns": np.asarray(columns, dtype=str),
        "column_encoding": np.asarray(["json"] * len(columns), dtype=str),
    }
    for index, column in enumerate(columns):
        payload[f"col_{index}"] = np.asarray(
            [
                json.dumps(_jsonable_cell(row.get(column)), sort_keys=True)
                for row in rows
            ],
            dtype=str,
        )
    np.savez_compressed(path, **payload)


def read_isb_d1_d6_table(path: Path | str) -> list[dict[str, Any]]:
    """Read the compact table without pandas or pickle."""

    with np.load(Path(path), allow_pickle=False) as arrays:
        columns = [str(value) for value in arrays["columns"]]
        values = [arrays[f"col_{index}"] for index in range(len(columns))]
        encodings = (
            [str(value) for value in arrays["column_encoding"]]
            if "column_encoding" in arrays.files
            else ["plain"] * len(columns)
        )
    return [
        {
            column: (
                json.loads(str(values[index][row]))
                if encodings[index] == "json"
                else values[index][row].item()
            )
            for index, column in enumerate(columns)
        }
        for row in range(len(values[0]) if values else 0)
    ]


def write_isb_d1_d6_report(
    output_dir: Path | str, report: Mapping[str, Any]
) -> dict[str, Path]:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "isb_d1_d6_report.json"
    table_path = destination / "isb_d1_d6_table.npz"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_table_npz(table_path, list(report.get("rows", [])))
    return {"json": json_path, "table": table_path}


def _resolve_artifact_path(output_root: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    if path.exists():
        return path.resolve()
    return (output_root / path).resolve()


def _npz_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as arrays:
        keys = list(arrays.files)
        shapes = {key: list(arrays[key].shape) for key in keys}
        dtypes = {key: str(arrays[key].dtype) for key in keys}
        columns = (
            [str(value) for value in arrays["columns"]]
            if "columns" in arrays.files
            else []
        )
    return {"npz_keys": keys, "shapes": shapes, "dtypes": dtypes, "columns": columns}


def _column_unit(column: str) -> str:
    name = column.lower()
    if name.endswith("_mm_s"):
        return "mm/s"
    if name.endswith("_mm"):
        return "mm"
    if name.endswith("_deg"):
        return "deg"
    if name.endswith("_rad"):
        return "rad"
    if name.endswith("_s") or name == "time":
        return "s"
    if "percent" in name:
        return "%"
    return "unknown"


def _table_curve_recipes(
    artifact_id: str, trial_id: str, path: Path, metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    columns = list(metadata.get("columns", []))
    if not columns:
        return []
    with np.load(path, allow_pickle=False) as arrays:
        column_arrays = {
            column: arrays[f"col_{index}"] for index, column in enumerate(columns)
        }
    x_key = next(
        (
            name
            for name in ("phase_percent", "time_s", "time", "frame")
            if name in columns
        ),
        None,
    )
    if x_key is None:
        return []
    categorical = [
        name
        for name in CURVE_FILTER_KEYS
        if name in column_arrays
        and name != x_key
        and column_arrays[name].dtype.kind in {"U", "S", "b"}
    ]
    recipes: list[dict[str, Any]] = []
    for y_key, values in column_arrays.items():
        if y_key == x_key or values.dtype.kind not in "iuf":
            continue
        combinations = sorted(
            {
                tuple((key, str(column_arrays[key][row])) for key in categorical)
                for row in range(values.shape[0])
            }
        )
        recipes.append(
            {
                "curve_id": f"{trial_id}/{artifact_id.split('/', 1)[-1]}/{y_key}",
                "trial_id": trial_id,
                "artifact_id": artifact_id,
                "storage": "table_npz",
                "x_key": x_key,
                "y_key": y_key,
                "filter_keys": categorical,
                "filter_combinations": [dict(value) for value in combinations],
                "x_unit": _column_unit(x_key),
                "y_unit": _column_unit(y_key),
                "preprocessing_chain": [
                    "read_table_npz",
                    "filter_rows",
                    "plot_finite_pairs",
                ],
            }
        )
    return recipes


def _csv_metadata(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = [dict(row) for row in reader]
        columns = list(reader.fieldnames or [])
    numeric_columns: list[str] = []
    for column in columns:
        values = [row.get(column, "") for row in rows if row.get(column, "") != ""]
        if values:
            try:
                [float(value) for value in values]
            except ValueError:
                continue
            numeric_columns.append(column)
    return {
        "columns": columns,
        "numeric_columns": numeric_columns,
        "row_count": len(rows),
    }, rows


def _csv_curve_recipes(
    artifact_id: str,
    rows: list[dict[str, str]],
    metadata: Mapping[str, Any],
) -> list[dict[str, Any]]:
    columns = list(metadata.get("columns", []))
    numeric = list(metadata.get("numeric_columns", []))
    categorical = [column for column in CURVE_FILTER_KEYS if column in columns]
    trial_id = "Batch"
    recipes: list[dict[str, Any]] = []
    for y_key in numeric:
        x_key = next(
            (
                key
                for key in reversed(categorical)
                if key not in {"participant", "trial"}
            ),
            categorical[0] if categorical else "row_index",
        )
        combinations = sorted(
            {tuple((key, row.get(key, "")) for key in categorical) for row in rows}
        )
        base = {
            "curve_id": f"Batch/{artifact_id.split('/', 1)[-1]}/{y_key}",
            "trial_id": trial_id,
            "artifact_id": artifact_id,
            "storage": "csv_table",
            "x_key": x_key,
            "y_key": y_key,
            "filter_keys": categorical,
            "filter_combinations": [dict(value) for value in combinations],
            "x_unit": "category",
            "y_unit": _column_unit(y_key),
            "preprocessing_chain": ["read_csv", "filter_rows", "plot_values"],
        }
        recipes.append(base)
    return recipes


def _hierarchical_curve_recipes(
    artifact_id: str, trial_id: str, path: Path, metadata: Mapping[str, Any]
) -> list[dict[str, Any]]:
    keys = list(metadata.get("npz_keys", []))
    time_by_prefix = {
        key.rsplit("/", 1)[0]: key
        for key in keys
        if key.rsplit("/", 1)[-1] in {"time", "time_s", "phase_percent"}
    }
    recipes: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as arrays:
        for key in keys:
            prefix, _, field = key.rpartition("/")
            if prefix not in time_by_prefix or key == time_by_prefix[prefix]:
                continue
            values = arrays[key]
            time_values = arrays[time_by_prefix[prefix]]
            component_axis: int | None = None
            if values.ndim == 1 and values.shape[0] == time_values.shape[0]:
                indices = [None]
            elif values.ndim == 2 and values.shape[-1] == time_values.shape[0]:
                component_axis = 0
                indices = list(range(values.shape[0]))
            else:
                continue
            for component in indices:
                suffix = f"[{component}]" if component is not None else ""
                recipes.append(
                    {
                        "curve_id": (
                            f"{trial_id}/{artifact_id.split('/', 1)[-1]}/"
                            f"{prefix}/{field}{suffix}"
                        ),
                        "trial_id": trial_id,
                        "artifact_id": artifact_id,
                        "storage": "hierarchical_npz",
                        "x_key": time_by_prefix[prefix],
                        "y_key": key,
                        "component_axis": component_axis,
                        "component_index": component,
                        "filter_keys": [],
                        "x_unit": _column_unit(
                            time_by_prefix[prefix].rsplit("/", 1)[-1]
                        ),
                        "y_unit": _column_unit(field),
                        "preprocessing_chain": [
                            "load_npz_key",
                            "select_component",
                            "plot_finite_pairs",
                        ],
                    }
                )
    return recipes


def _transformation_refs(
    trial_id: str, outputs: Mapping[str, Any], artifacts: Mapping[str, Any]
) -> dict[str, str | None]:
    candidates = {
        "spatial": f"{trial_id}/"
        + (
            "motive_static_alignment"
            if "motive_static_alignment" in outputs
            else "spatial_calibration"
        ),
        "temporal": f"{trial_id}/temporal_synchronization",
        "quality": f"{trial_id}/metric_quality",
    }
    return {
        key: artifact_id if artifact_id in artifacts else None
        for key, artifact_id in candidates.items()
    }


def build_reproducibility_manifest(
    output_root: Path | str, trial_reports: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Fingerprint JSON/NPZ outputs and describe table-curve reconstruction."""

    root = Path(output_root).expanduser().resolve()
    artifacts: dict[str, Any] = {}
    omitted: list[dict[str, str]] = []
    recipes: list[dict[str, Any]] = []
    transformations: dict[str, dict[str, str | None]] = {}
    for report in trial_reports:
        trial_id = str(report.get("trial", ""))
        outputs = report.get("outputs", {})
        if not trial_id or not isinstance(outputs, Mapping):
            continue
        for output_name, raw_path in sorted(outputs.items()):
            if not raw_path:
                continue
            path = _resolve_artifact_path(root, raw_path)
            if path.suffix.lower() not in {".json", ".npz"}:
                continue
            artifact_id = f"{trial_id}/{output_name}"
            if not path.exists():
                omitted.append(
                    {"artifact_id": artifact_id, "path": str(path), "reason": "missing"}
                )
                continue
            try:
                relative_path = str(path.relative_to(root))
            except ValueError:
                relative_path = str(path)
            artifact: dict[str, Any] = {
                "relative_path": relative_path,
                "sha256": _sha256(path),
                "media_type": (
                    "application/x-npz"
                    if path.suffix.lower() == ".npz"
                    else "application/json"
                ),
                "size_bytes": path.stat().st_size,
            }
            if path.suffix.lower() == ".npz":
                artifact.update(_npz_metadata(path))
                if artifact.get("columns"):
                    recipes.extend(
                        _table_curve_recipes(artifact_id, trial_id, path, artifact)
                    )
                else:
                    recipes.extend(
                        _hierarchical_curve_recipes(
                            artifact_id, trial_id, path, artifact
                        )
                    )
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                artifact["schema_version"] = (
                    payload.get("schema_version")
                    if isinstance(payload, Mapping)
                    else None
                )
            artifacts[artifact_id] = artifact
        transformations[trial_id] = _transformation_refs(trial_id, outputs, artifacts)
    for path in sorted(root.glob("all_*.csv")):
        artifact_id = f"Batch/{path.stem}"
        metadata, csv_rows = _csv_metadata(path)
        artifacts[artifact_id] = {
            "relative_path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "media_type": "text/csv",
            "size_bytes": path.stat().st_size,
            **metadata,
        }
        recipes.extend(_csv_curve_recipes(artifact_id, csv_rows, metadata))
    for path in sorted(root.glob("*.json")) + sorted(root.glob("*.npz")):
        if path.name in {
            "comparison_reproducibility_manifest.json",
            "provenance_manifest.json",
            "run_report.json",
        }:
            continue
        artifact_id = f"Batch/{path.stem}"
        if artifact_id in artifacts:
            continue
        artifact = {
            "relative_path": str(path.relative_to(root)),
            "sha256": _sha256(path),
            "media_type": (
                "application/x-npz"
                if path.suffix.lower() == ".npz"
                else "application/json"
            ),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() == ".npz":
            artifact.update(_npz_metadata(path))
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            artifact["schema_version"] = (
                payload.get("schema_version") if isinstance(payload, Mapping) else None
            )
        artifacts[artifact_id] = artifact
    for recipe in recipes:
        trial_refs = transformations.get(str(recipe["trial_id"]), {})
        recipe["spatial_alignment_ref"] = trial_refs.get("spatial")
        recipe["temporal_alignment_ref"] = trial_refs.get("temporal")
        recipe["quality_report_ref"] = trial_refs.get("quality")
    manifest = {
        "schema_version": 1,
        "created_utc": _utc_now(),
        "output_root": str(root),
        "artifacts": artifacts,
        "omitted_artifacts": omitted,
        "transformations": transformations,
        "curve_recipes": recipes,
        "interpretation": (
            "Recipes reproduce stored curves only. Anatomical validity remains governed "
            "by the ISB D1-D6 report and trial quality metadata."
        ),
    }
    errors = validate_reproducibility_manifest(manifest, base_dir=root)
    manifest["validation"] = {
        "status": "valid" if not errors else "invalid",
        "errors": errors,
    }
    return manifest


def validate_reproducibility_manifest(
    manifest: Mapping[str, Any], *, base_dir: Path | str | None = None
) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("unsupported_schema_version")
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        return errors + ["artifacts_not_mapping"]
    base_value = base_dir if base_dir is not None else manifest.get("output_root")
    base_path = (
        Path(base_value).expanduser().resolve() if base_value is not None else None
    )
    if base_path is not None:
        for artifact_id, artifact in artifacts.items():
            artifact_path = Path(str(artifact.get("relative_path", "")))
            if not artifact_path.is_absolute():
                artifact_path = base_path / artifact_path
            if not artifact_path.exists():
                errors.append(f"missing_file:{artifact_id}")
            elif (
                artifact.get("sha256") and _sha256(artifact_path) != artifact["sha256"]
            ):
                errors.append(f"hash_mismatch:{artifact_id}")
    for refs in manifest.get("transformations", {}).values():
        if not isinstance(refs, Mapping):
            continue
        for artifact_id in refs.values():
            if artifact_id and artifact_id not in artifacts:
                errors.append(f"missing_transformation:{artifact_id}")
    for recipe in manifest.get("curve_recipes", []):
        artifact_id = recipe.get("artifact_id")
        if artifact_id not in artifacts:
            errors.append(f"missing_artifact:{artifact_id}")
            continue
        artifact = artifacts[artifact_id]
        trial_refs = manifest.get("transformations", {}).get(
            str(recipe.get("trial_id", "")), {}
        )
        recipe_ref_keys = {
            "spatial_alignment_ref": "spatial",
            "temporal_alignment_ref": "temporal",
            "quality_report_ref": "quality",
        }
        for recipe_key, transformation_key in recipe_ref_keys.items():
            reference = recipe.get(recipe_key)
            if reference and reference not in artifacts:
                errors.append(f"missing_recipe_transformation:{reference}")
            expected = (
                trial_refs.get(transformation_key)
                if isinstance(trial_refs, Mapping)
                else None
            )
            if reference != expected:
                errors.append(
                    f"recipe_transformation_mismatch:{recipe.get('curve_id')}:{recipe_key}"
                )
        columns = set(artifact.get("columns", []))
        available_keys = columns or set(artifact.get("npz_keys", []))
        for key_name in ("x_key", "y_key"):
            if recipe.get(key_name) not in available_keys:
                errors.append(f"missing_column:{artifact_id}:{recipe.get(key_name)}")
        for combination in recipe.get("filter_combinations", []):
            for filter_key in combination:
                if filter_key not in columns:
                    errors.append(f"missing_filter_column:{artifact_id}:{filter_key}")
    return errors


def write_reproducibility_manifest(
    output_dir: Path | str, manifest: Mapping[str, Any]
) -> Path:
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "comparison_reproducibility_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_curve_from_manifest(
    manifest_path: Path | str,
    curve_id: str,
    *,
    filters: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Reconstruct one table curve using only its versioned manifest."""

    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    recipe = next(
        (
            item
            for item in manifest.get("curve_recipes", [])
            if item.get("curve_id") == curve_id
        ),
        None,
    )
    if recipe is None:
        raise KeyError(f"Unknown curve_id: {curve_id}")
    requested_filters = {str(key): str(value) for key, value in (filters or {}).items()}
    allowed_filters = [
        {str(key): str(value) for key, value in combination.items()}
        for combination in recipe.get("filter_combinations", [])
    ]
    if requested_filters and requested_filters not in allowed_filters:
        raise ValueError(
            f"Unsupported filters for {curve_id}: {requested_filters}. "
            "Use one combination listed in filter_combinations."
        )
    artifact = manifest["artifacts"][recipe["artifact_id"]]
    artifact_path = Path(artifact["relative_path"])
    if not artifact_path.is_absolute():
        artifact_path = path.parent / artifact_path
    if _sha256(artifact_path) != artifact["sha256"]:
        raise ValueError(f"Artifact hash mismatch: {artifact_path}")
    if recipe["storage"] == "csv_table":
        with artifact_path.open("r", encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
        for key, expected in requested_filters.items():
            rows = [row for row in rows if row.get(key) == str(expected)]
        x = np.asarray(
            [row.get(recipe["x_key"], str(index)) for index, row in enumerate(rows)],
            dtype=str,
        )
        y = np.asarray([float(row[recipe["y_key"]]) for row in rows], dtype=float)
        loaded_filters = dict(requested_filters)
    else:
        with np.load(artifact_path, allow_pickle=False) as arrays:
            if recipe["storage"] == "table_npz":
                columns = [str(value) for value in arrays["columns"]]
                values = {
                    column: arrays[f"col_{index}"]
                    for index, column in enumerate(columns)
                }
                mask = np.ones(values[recipe["x_key"]].shape[0], dtype=bool)
                for key, expected in requested_filters.items():
                    mask &= values[key].astype(str) == str(expected)
                x = values[recipe["x_key"]][mask]
                y = values[recipe["y_key"]][mask]
                loaded_filters = {
                    key: values[key][mask] for key in recipe.get("filter_keys", [])
                }
            else:
                x = arrays[recipe["x_key"]]
                y = arrays[recipe["y_key"]]
                if recipe.get("component_axis") == 0:
                    y = y[int(recipe["component_index"])]
                loaded_filters = {}
    return {
        "curve_id": curve_id,
        "x": x,
        "y": y,
        "filters": loaded_filters,
        "x_unit": recipe["x_unit"],
        "y_unit": recipe["y_unit"],
    }
