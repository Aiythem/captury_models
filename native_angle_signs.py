"""Sign-only presentation of native lower-limb coordinates, never ISB harmonization.

Reference requested by the user:
Fukuchi CA, Fukuchi RK, Duarte M (2018). A public dataset of overground and
 treadmill walking kinematics and kinetics in healthy individuals.
PeerJ 6:e4640. https://doi.org/10.7717/peerj.4640

The supplied figure defines the desired positive directions, not the polarity
of Motive/BioBuddy/Captury exports. Polarity is inspected independently in the
exact selected model at q=0. No pose, position, frame, Euler sequence, timing or
zero offset is modified. The ankle transverse mapping is deliberately pending.
"""

from functools import lru_cache
from pathlib import Path
import hashlib
import json
import re
import numpy as np

SIGN_REFERENCE = {
    "authors": ["Claudiane A. Fukuchi", "Reginaldo K. Fukuchi", "Marcos Duarte"],
    "title": "A public dataset of overground and treadmill walking kinematics and kinetics in healthy individuals",
    "year": 2018,
    "journal": "PeerJ",
    "volume": 6,
    "article": "e4640",
    "doi": "10.7717/peerj.4640",
    "url": "https://doi.org/10.7717/peerj.4640",
    "use": "Requested reporting directions; not validation of native model conventions",
}
SIGN_FIELDS = (
    "raw_value",
    "raw_mean_value",
    "sign_multiplier",
    "sign_status",
    "positive_direction",
    "sign_plane",
    "sign_model_sha256",
    "sign_reference",
)
PLANE_AXES = {
    "Sagittal": {"motive": "X", "biobuddy": "Z", "captury": "X"},
    "Frontal": {"motive": "Z", "biobuddy": "X", "captury": "Z"},
    "Transverse": {"motive": "Y", "biobuddy": "Y", "captury": "Y"},
}
# These are candidate native-axis groupings, accepted only when the exact model
# axis is sufficiently parallel to the target direction. No axis is permuted.
DIRECTIONS = {
    "hip": ("flexion", "adduction", "internal_rotation"),
    "knee": ("flexion", "adduction", "internal_rotation"),
    "ankle": ("dorsiflexion", "inversion", ""),
}


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if not np.isfinite(v).all() or n < 1e-10:
        raise ValueError("Degenerate model reference geometry")
    return v / n


def reference_basis(positions):
    """Native-model neutral body basis: anterior, proximal, subject-right."""
    right = _unit(positions["right_thigh"] - positions["left_thigh"])
    up = (
        positions["right_thigh"]
        + positions["left_thigh"]
        - positions["right_foot"]
        - positions["left_foot"]
    ) / 2
    up = _unit(up - right * np.dot(up, right))
    anterior = _unit(np.cross(up, right))
    return np.column_stack((anterior, up, right))


def positive_axis(joint, side, plane, basis):
    anterior, up, right = np.asarray(basis).T
    mirror = 1 if side == "right" else -1
    if plane == "sagittal":
        return right * (-1 if joint == "knee" else 1)
    if plane == "frontal":
        return anterior * mirror
    if plane == "transverse" and joint != "ankle":
        return up * mirror
    return None


def channel_rule(joint, side, plane, basis, native_axis):
    rule = {
        "plane": plane,
        "sign_multiplier": 1,
        "positive_direction": "",
        "sign_status": "pending_ankle_transverse_mapping",
        "axis_projection": None,
    }
    target = positive_axis(joint, side, plane, basis)
    if target is None:
        return rule
    projection = float(np.dot(_unit(native_axis), target))
    rule["axis_projection"] = projection
    if abs(projection) < 0.9:
        rule["sign_status"] = "pending_oblique_axis"
        return rule
    rule.update(
        sign_multiplier=1 if projection > 0 else -1,
        sign_status="model_reference_polarity",
        positive_direction=DIRECTIONS[joint][
            ("sagittal", "frontal", "transverse").index(plane)
        ],
    )
    return rule


def _channel_identity(name):
    match = re.fullmatch(r"(.+)_(?:Rot([XYZ])|([XYZ])rotation)", str(name), re.I)
    return (match[1], (match[2] or match[3]).upper()) if match else None


@lru_cache(maxsize=64)
def _model_profile(path, sha256, source, source_kind):
    import biorbd
    from kinematic_conventions import load_kinematic_conventions, segment_source_names

    model = biorbd.Model(path)
    q = np.zeros(model.nbQ())
    segments = [s.name().to_string() for s in model.segments()]
    names = [s.to_string() for s in model.nameDof()]
    source_id = "biobuddy_motive57" if source == "biobuddy" else source + "_model"
    mapping = segment_source_names(load_kinematic_conventions(), source_id, source_kind)
    frames = {
        key: np.array(model.globalJCS(q, segments.index(name)).to_array(), copy=True)
        for key, name in mapping.items()
        if name in segments
    }
    for frame in frames.values():
        R = frame[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or np.linalg.det(R) < 0:
            raise ValueError("Invalid reference rotation")
    basis = reference_basis({key: T[:3, 3] for key, T in frames.items()})
    channels = {}
    for side in ("right", "left"):
        for joint, segment in [("hip", "thigh"), ("knee", "shank"), ("ankle", "foot")]:
            canonical = side + "_" + segment
            name = mapping.get(canonical)
            if name not in segments:
                continue
            R = frames[canonical][:3, :3]
            for index, qname in enumerate(names):
                identity = _channel_identity(qname)
                if identity is None or identity[0] != name:
                    continue
                plane = next(
                    (
                        p.lower()
                        for p, axes in PLANE_AXES.items()
                        if axes.get(source) == identity[1]
                    ),
                    None,
                )
                if plane is None:
                    continue
                plus = q.copy()
                minus = q.copy()
                plus[index] = 1e-5
                minus[index] = -1e-5
                Rp = np.array(
                    model.globalJCS(plus, segments.index(name)).to_array()[:3, :3],
                    copy=True,
                )
                Rm = np.array(
                    model.globalJCS(minus, segments.index(name)).to_array()[:3, :3],
                    copy=True,
                )
                D = ((Rp - Rm) / (2e-5)) @ R.T
                axis = (
                    np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
                    / 2
                )
                channels[qname] = {
                    **channel_rule(joint, side, plane, basis, axis),
                    "joint": joint,
                    "side": side,
                    "native_axis_at_q0": axis.tolist(),
                }
    return {
        "model_sha256": sha256,
        "model_path": path,
        "channels": channels,
        "reference_basis": basis.tolist(),
        "reference_pose": "model q=0",
        "interpretation": "native coordinate polarity only; anatomical equivalence unverified",
    }


def profiles_for_trial(trial_dir):
    """Read existing evidence only; never run IK or overwrite model/results."""
    if trial_dir is None:
        return {}
    directory = Path(trial_dir)
    try:
        report = json.loads((directory / "run_report.json").read_text())
    except (OSError, ValueError):
        return {}
    # Reexpressed q are not the native coordinates of the serialized model.
    if report.get("q_reexpression", {}).get("enabled", False):
        return {}
    result = {}
    for source in ("motive", "biobuddy", "captury"):
        info = report.get("models", {}).get(source, {})
        value = info.get("biomod") or info.get("path")
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            candidates = [
                Path(__file__).parent / path,
                Path.cwd() / path,
                directory / path,
            ]
            path = next((p for p in candidates if p.is_file()), path)
        try:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            # Existing q must belong to the selected model: BioBuddy cache records
            # its input hash. A modified model cannot justify old native signs.
            if source == "biobuddy":
                expected = (
                    report.get("biobuddy_ik_batch", {})
                    .get("cache", {})
                    .get("payload", {})
                    .get("biomod", {})
                    .get("sha256")
                )
                if not expected or expected != sha:
                    continue
            if source != "biobuddy":
                manifest_path = directory.parent / "provenance_manifest.json"
                try:
                    manifest = json.loads(manifest_path.read_text())
                    expected = (
                        manifest.get("derived_artifacts", {})
                        .get(
                            f"{report.get('trial', directory.name)}/{source}/generated_biomod",
                            {},
                        )
                        .get("sha256")
                    )
                except (OSError, ValueError):
                    expected = None
                if not expected or expected != sha:
                    continue
            result[source] = _model_profile(
                str(path.resolve()),
                sha,
                source,
                info.get(
                    "source_kind", "motive_57_ik" if source == "biobuddy" else "bvh"
                ),
            )
        except (OSError, ValueError, RuntimeError, KeyError, ImportError):
            continue
    return result


def correct_native_sign_table(table, trial_dir=None, *, profiles=None):
    """Copy native tables and multiply values once; ROM and all geometry untouched."""
    result = table.copy(deep=True)
    if result.empty:
        return result
    if (
        profiles is None
        and trial_dir is None
        and {"raw_value", "sign_multiplier", "sign_status", "sign_model_sha256"}
        <= set(result)
    ):
        return result
    profiles = profiles_for_trial(trial_dir) if profiles is None else profiles
    for value in ("value", "mean_value"):
        if value in result and "raw_" + value not in result:
            result["raw_" + value] = result[value]
    result["sign_multiplier"] = 1
    result["sign_status"] = "pending_model_evidence"
    result["positive_direction"] = ""
    result["sign_plane"] = ""
    result["sign_model_sha256"] = ""
    result["sign_reference"] = SIGN_REFERENCE["doi"]
    required = {"source", "side", "joint", "q_name", "axis", "unit"}
    if required <= set(result):
        for keys, group in result.groupby(
            ["source", "side", "joint", "q_name", "axis", "unit"], sort=False
        ):
            source, side, joint, name, axis, unit = keys
            if unit not in ("deg", "rad") or (
                "coordinate_kind" in group
                and not group.coordinate_kind.eq("rotation").all()
            ):
                result.loc[group.index, "sign_status"] = "not_applicable_nonrotation"
                continue
            if joint not in DIRECTIONS or side not in ("left", "right"):
                continue
            plane = next(
                (
                    p.lower()
                    for p, axes in PLANE_AXES.items()
                    if axes.get(source) == str(axis).upper()
                ),
                "",
            )
            result.loc[group.index, "sign_plane"] = plane
            if joint == "ankle" and plane == "transverse":
                result.loc[group.index, "sign_status"] = (
                    "pending_ankle_transverse_mapping"
                )
                continue
            profile = profiles.get(source, {})
            candidates = profile.get("channels", {})
            rule = candidates.get(name)
            if rule is None and _channel_identity(name) is not None:
                rule = next(
                    (
                        r
                        for k, r in candidates.items()
                        if _channel_identity(k) == _channel_identity(name)
                    ),
                    None,
                )
            if rule is None:
                continue
            if rule.get("joint", joint) != joint or rule.get("side", side) != side:
                continue
            for field in ("sign_multiplier", "sign_status", "positive_direction"):
                result.loc[group.index, field] = rule[field]
            result.loc[group.index, "sign_model_sha256"] = profile.get(
                "model_sha256", ""
            )
    for value in ("value", "mean_value"):
        if value in result:
            result[value] = result["raw_" + value] * result.sign_multiplier
    return result
