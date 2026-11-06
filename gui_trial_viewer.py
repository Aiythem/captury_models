"""Lightweight Tk C3D trial viewer components for the Captury/BioBuddy GUI."""

from __future__ import annotations

import csv
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from c3d_trial_viewer import (
    C3DMarkerData,
    camera_matrix_for_plane,
    camera_matrix_for_subject_view,
    default_camera_matrix,
    fit_center_and_scale,
    normalized_camera_matrix,
    project_points,
    rotation_matrix_from_drag,
)
from compare_capture_systems import DEFAULT_LANDMARK_MAP
from mocap_alignment import kabsch_rows
from mocap_labels import (
    display_marker_name,
    is_joint_centre_marker_label,
    marker_display_labels,
    marker_indices_by_display_label,
)

VIEWER_MARKER_COLOR = "#38bdf8"
VIEWER_SELECTED_OUTLINE_COLOR = "#111827"
VIEWER_AXIS_COLORS = {"X": "#ef4444", "Y": "#22c55e", "Z": "#3b82f6"}
COR_LAYER_LABELS = {
    "captury": "Captury",
    "motive": "Motive",
    "biobuddy": "BioBuddy",
}
DATA_SOURCE_COLORS = {
    "captury": "#f97316",
    "motive": "#0ea5e9",
    "biobuddy": "#22c55e",
}
DATA_SOURCE_MARKER_COLORS = {
    "captury": "#fb923c",
    "motive": "#38bdf8",
    "biobuddy": "#86efac",
}
COR_LAYER_COLORS = DATA_SOURCE_COLORS
SUBJECT_LEFT_RIGHT_MARKER_PAIRS = (
    ("LIAS", "RIAS"),
    ("LIPS", "RIPS"),
    ("LAH", "RAH"),
    ("LPH", "RPH"),
    ("LHLE", "RHLE"),
    ("LHME", "RHME"),
    ("LFLE", "RFLE"),
    ("LFME", "RFME"),
    ("LFAX", "RFAX"),
    ("LTAM", "RTAM"),
)
COR_CHAIN_CANDIDATES = (
    ("pelvis", "thorax"),
    ("thorax", "head"),
    ("pelvis", "left_thigh"),
    ("left_thigh", "left_shank"),
    ("left_shank", "left_foot"),
    ("pelvis", "right_thigh"),
    ("right_thigh", "right_shank"),
    ("right_shank", "right_foot"),
    ("thorax", "left_upper_arm"),
    ("left_upper_arm", "left_forearm"),
    ("left_forearm", "left_hand"),
    ("thorax", "right_upper_arm"),
    ("right_upper_arm", "right_forearm"),
    ("right_forearm", "right_hand"),
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Spine3"),
    ("Spine3", "Spine4"),
    ("Spine4", "Neck"),
    ("Neck", "Head"),
    ("Spine4", "LeftShoulder"),
    ("LeftShoulder", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine4", "RightShoulder"),
    ("RightShoulder", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
)


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    return np.asarray(vector, dtype=float) / norm


def local_chain_axes(
    joint: str,
    points: dict[str, np.ndarray],
    edges: Iterable[tuple[str, str]],
) -> dict[str, np.ndarray] | None:
    """Estimate a compact local XYZ triad from the visible CoR chain geometry.

    The viewer receives joint-centre positions, not full segment rotations.  For
    visualization we therefore orient the local Y axis along the first available
    parent-child segment connected to ``joint`` and derive an orthonormal X/Z
    basis from the laboratory vertical helper.  This is intentionally a display
    frame, not an exported biomechanical convention.
    """
    origin = points.get(joint)
    if origin is None:
        return None

    candidates: list[np.ndarray] = []
    for proximal, distal in edges:
        if proximal == joint and distal in points:
            candidates.append(np.asarray(points[distal], dtype=float) - origin)
        elif distal == joint and proximal in points:
            candidates.append(origin - np.asarray(points[proximal], dtype=float))

    y_axis = next(
        (axis for vector in candidates if (axis := _unit_vector(vector)) is not None),
        None,
    )
    if y_axis is None:
        return None

    helper = np.asarray((0.0, 0.0, 1.0))
    if abs(float(np.dot(helper, y_axis))) > 0.92:
        helper = np.asarray((1.0, 0.0, 0.0))
    x_axis = _unit_vector(np.cross(helper, y_axis))
    if x_axis is None:
        return None
    z_axis = _unit_vector(np.cross(x_axis, y_axis))
    if z_axis is None:
        return None
    return {"X": x_axis, "Y": y_axis, "Z": z_axis}


def data_source_color(source: str) -> str:
    key = str(source).strip().lower()
    if key in {"bio_buddy", "biorbd"}:
        key = "biobuddy"
    if key == "captury_c3d":
        key = "captury"
    return DATA_SOURCE_COLORS.get(key, "#64748b")


def data_source_marker_color(source: str) -> str:
    key = str(source).strip().lower()
    if key in {"bio_buddy", "biorbd"}:
        key = "biobuddy"
    if key == "captury_c3d":
        key = "captury"
    return DATA_SOURCE_MARKER_COLORS.get(key, VIEWER_MARKER_COLOR)


def transformed_marker_data(
    data: C3DMarkerData, rotation: np.ndarray, translation: np.ndarray
) -> C3DMarkerData:
    points = np.asarray(data.points, dtype=float)
    rows = points.reshape(3, -1).T @ rotation + translation
    transformed = rows.T.reshape(points.shape)
    return C3DMarkerData(
        labels=list(data.labels),
        points=transformed,
        rate=data.rate,
        unit=data.unit,
        time=None if data.time is None else np.asarray(data.time, dtype=float).copy(),
        display_labels=(
            None if data.display_labels is None else list(data.display_labels)
        ),
    )


def captury_marker_transform_from_report(
    report: dict[str, object],
) -> tuple[np.ndarray, np.ndarray] | None:
    alignment = report.get("alignment")
    if not isinstance(alignment, dict):
        return None
    rotation_values = alignment.get("rotation")
    translation_values = alignment.get("translation_mm")
    if rotation_values is None or translation_values is None:
        return None
    rotation = np.asarray(rotation_values, dtype=float)
    translation = np.asarray(translation_values, dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        return None

    model_marker = alignment.get("motive_model_to_c3d_markers")
    if isinstance(model_marker, dict):
        model_rotation_values = model_marker.get("rotation")
        model_translation_values = model_marker.get("translation_mm")
        if model_rotation_values is not None and model_translation_values is not None:
            model_rotation = np.asarray(model_rotation_values, dtype=float)
            model_translation = np.asarray(model_translation_values, dtype=float)
            if model_rotation.shape == (3, 3) and model_translation.shape == (3,):
                rotation, translation = (
                    rotation @ model_rotation,
                    translation @ model_rotation + model_translation,
                )
    return rotation, translation


def average_marker_group(points: np.ndarray, indices: list[int]) -> np.ndarray | None:
    if not indices:
        return None
    with np.errstate(invalid="ignore"):
        return np.nanmean(points[:, indices, :], axis=1)


def captury_marker_transform_from_c3d_layers(
    captury: C3DMarkerData,
    motive: C3DMarkerData,
    *,
    min_landmarks: int = 4,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate Captury-marker -> Motive-marker transform from anatomical labels.

    Captury C3D markerless landmarks usually use ``Q_*`` labels, while Motive uses
    skin-marker labels. The same editable landmark map used by the CLI
    comparison is reused here so the embedded GUI preview can orient the orange
    Captury marker cloud in the Motive marker frame.
    """

    captury_lookup = marker_indices_by_display_label(captury.labels)
    motive_lookup = marker_indices_by_display_label(motive.labels)
    moving_rows: list[np.ndarray] = []
    reference_rows: list[np.ndarray] = []
    for item in DEFAULT_LANDMARK_MAP:
        motive_indices = [
            index
            for label in item["reference"]
            for index in motive_lookup.get(label, [])
        ]
        captury_indices = [
            index for label in item["test"] for index in captury_lookup.get(label, [])
        ]
        motive_signal = average_marker_group(motive.points, motive_indices)
        captury_signal = average_marker_group(captury.points, captury_indices)
        if motive_signal is None or captury_signal is None:
            continue
        with np.errstate(invalid="ignore"):
            motive_point = np.nanmedian(motive_signal, axis=1)
            captury_point = np.nanmedian(captury_signal, axis=1)
        if np.all(np.isfinite(motive_point)) and np.all(np.isfinite(captury_point)):
            reference_rows.append(motive_point)
            moving_rows.append(captury_point)
    if len(moving_rows) < min_landmarks:
        return None
    return kabsch_rows(np.vstack(reference_rows), np.vstack(moving_rows))


def vertical_axis_label(kind: str, system: str | None = None) -> str:
    """Return the displayed native vertical-axis convention for one file."""

    normalized_kind = str(kind).strip().lower()
    if normalized_kind in {"bvh", "fbx"}:
        return "+Y modèle"
    if normalized_kind == "c3d":
        if str(system or "").strip().lower() == "captury":
            return "+Y labo"
        return "+Z labo"
    return "auto"


def joint_chain_edges(joints: Iterable[str]) -> list[tuple[str, str]]:
    available = set(joints)
    return [
        (proximal, distal)
        for proximal, distal in COR_CHAIN_CANDIDATES
        if proximal in available and distal in available
    ]


def available_cor_layers(fieldnames: Iterable[str]) -> list[str]:
    names = set(fieldnames)
    layers: list[str] = []
    for layer in COR_LAYER_LABELS:
        if all(f"{layer}_{axis}_mm" in names for axis in ("x", "y", "z")):
            layers.append(layer)
    return layers


@dataclass
class JointCentreChainData:
    layers: dict[str, dict[str, np.ndarray]]
    edges: list[tuple[str, str]]
    times: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        for joints in self.layers.values():
            for values in joints.values():
                return int(values.shape[0])
        return 0


def load_joint_centre_chain_data(path: Path) -> JointCentreChainData | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    if path.suffix.lower() == ".npz":
        return _load_joint_centre_chain_data_npz(path)
    return _load_joint_centre_chain_data_csv(path)


def _load_joint_centre_chain_data_npz(path: Path) -> JointCentreChainData | None:
    with np.load(path, allow_pickle=False) as data:
        fieldnames = [str(column) for column in data["columns"]]
        arrays = {
            column: data[f"col_{index}"] for index, column in enumerate(fieldnames)
        }
    layers = available_cor_layers(fieldnames)
    if not layers or "joint" not in arrays:
        return None
    joint_values = np.asarray([str(value) for value in arrays["joint"]])
    time_values = (
        np.asarray(arrays["time"], dtype=float)
        if "time" in arrays
        else np.arange(joint_values.shape[0], dtype=float)
    )
    layer_arrays: dict[str, dict[str, np.ndarray]] = {layer: {} for layer in layers}
    layer_times: dict[str, dict[str, np.ndarray]] = {layer: {} for layer in layers}
    for layer in layers:
        required_columns = [f"{layer}_{axis}_mm" for axis in ("x", "y", "z")]
        if any(column not in arrays for column in required_columns):
            continue
        for joint in sorted(set(joint_values)):
            if not joint:
                continue
            mask = joint_values == joint
            order = np.argsort(time_values[mask])
            points = np.column_stack(
                [
                    np.asarray(arrays[column], dtype=float)[mask]
                    for column in required_columns
                ]
            )
            points = points[order]
            if points.size:
                layer_arrays[layer][joint] = points
                layer_times[layer][joint] = time_values[mask][order]
    all_joints = {joint for joints in layer_arrays.values() for joint in joints}
    if not all_joints:
        return None
    return JointCentreChainData(
        layers=layer_arrays,
        edges=joint_chain_edges(all_joints),
        times=layer_times,
    )


def _load_joint_centre_chain_data_csv(path: Path) -> JointCentreChainData | None:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        layers = available_cor_layers(reader.fieldnames)
        if not layers:
            return None
        values: dict[str, dict[str, list[tuple[float, np.ndarray]]]] = {
            layer: {} for layer in layers
        }
        for row in reader:
            joint = str(row.get("joint", "")).strip()
            if not joint:
                continue
            try:
                time_value = float(row.get("time", 0.0))
            except (TypeError, ValueError):
                time_value = 0.0
            for layer in layers:
                try:
                    point = np.asarray(
                        [
                            float(row[f"{layer}_x_mm"]),
                            float(row[f"{layer}_y_mm"]),
                            float(row[f"{layer}_z_mm"]),
                        ],
                        dtype=float,
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                values[layer].setdefault(joint, []).append((time_value, point))
    layer_arrays: dict[str, dict[str, np.ndarray]] = {}
    layer_times: dict[str, dict[str, np.ndarray]] = {}
    for layer, joints in values.items():
        layer_arrays[layer] = {}
        layer_times[layer] = {}
        for joint, samples in joints.items():
            samples.sort(key=lambda item: item[0])
            layer_arrays[layer][joint] = np.vstack([point for _time, point in samples])
            layer_times[layer][joint] = np.asarray(
                [time for time, _point in samples], dtype=float
            )
    all_joints = {joint for joints in layer_arrays.values() for joint in joints}
    edges = joint_chain_edges(all_joints)
    return JointCentreChainData(layers=layer_arrays, edges=edges, times=layer_times)


class TkC3DTrialCanvas(tk.Canvas):
    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(
            parent,
            background="#ffffff",
            highlightthickness=1,
            highlightbackground="#d1d5db",
        )
        self.data: C3DMarkerData | None = None
        self.marker_layers: dict[str, C3DMarkerData] = {}
        self.marker_time_offsets_s: dict[str, float] = {}
        self.visible_marker_sources: set[str] = {"captury", "motive"}
        self.frame = 0
        self.camera = default_camera_matrix()
        self.zoom = 1.0
        self.pan_pixels = np.zeros(2)
        self.selected_label: str | None = None
        self.selected_markers: dict[str, set[str]] = {}
        self.marker_source = "motive"
        self.chain_data: JointCentreChainData | None = None
        self.visible_cor_layers: set[str] = {"captury", "motive"}
        self.show_chain_axes = False
        self.rotate_body_segments_180_x = False
        self._is_dragging = False
        self._last_mouse_xy: tuple[float, float] | None = None
        self._redraw_after_id: str | None = None
        self._show_labels_after_id: str | None = None
        self._show_labels = True

        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<ButtonPress-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Double-Button-1>", lambda _event: self.reset_camera())
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Button-4>", lambda _event: self._zoom(1.12))
        self.bind("<Button-5>", lambda _event: self._zoom(1 / 1.12))
        self.bind("<Button-2>", self._show_context_menu)
        self.bind("<Button-3>", self._show_context_menu)

    def set_data(self, data: C3DMarkerData | None) -> None:
        self.data = data
        self.marker_layers = {self.marker_source: data} if data is not None else {}
        self.frame = 0
        self.selected_label = None
        self.selected_markers = {}
        self.reset_camera()

    def set_marker_layers(self, layers: dict[str, C3DMarkerData]) -> None:
        normalized_layers = {
            str(source).strip().lower(): data for source, data in layers.items()
        }
        self.marker_layers = normalized_layers
        self.data = normalized_layers.get(
            "motive", next(iter(normalized_layers.values()), None)
        )
        self.frame = 0
        self.selected_label = None
        self.selected_markers = {}
        self.reset_camera()

    def set_marker_time_offsets(self, offsets_s: dict[str, float]) -> None:
        self.marker_time_offsets_s = {
            str(source).strip().lower(): float(offset)
            for source, offset in offsets_s.items()
        }
        self.redraw()

    def set_marker_source(self, source: str | None) -> None:
        self.marker_source = str(source or "motive").strip().lower()
        self.redraw()

    def set_joint_centre_chains(self, data: JointCentreChainData | None) -> None:
        self.chain_data = data
        self.redraw()

    def set_visible_cor_layers(self, layers: Iterable[str]) -> None:
        self.visible_cor_layers = set(layers)
        self.redraw()

    def set_show_chain_axes(self, show: bool) -> None:
        self.show_chain_axes = bool(show)
        self.redraw()

    def set_rotate_body_segments_180_x(self, enabled: bool) -> None:
        self.rotate_body_segments_180_x = bool(enabled)
        self.redraw()

    def set_visible_marker_sources(self, sources: Iterable[str]) -> None:
        self.visible_marker_sources = {
            str(source).strip().lower() for source in sources
        }
        self.redraw()

    def set_selected_markers(self, markers: dict[str, Iterable[str]]) -> None:
        self.selected_markers = {
            str(source).strip().lower(): {str(label) for label in labels}
            for source, labels in markers.items()
        }
        self.redraw()

    @property
    def n_frames(self) -> int:
        if self.marker_layers:
            reference = self.marker_layers.get("motive")
            if reference is not None:
                return reference.n_frames
            return max(data.n_frames for data in self.marker_layers.values())
        if self.data is not None:
            return self.data.n_frames
        if self.chain_data is not None:
            return self.chain_data.n_frames
        return 0

    def set_frame(self, frame: int) -> None:
        n_frames = self.n_frames
        if n_frames <= 0:
            self.frame = 0
        else:
            self.frame = max(0, min(int(frame), n_frames - 1))
        self.redraw()

    def reset_camera(self) -> None:
        self.camera = default_camera_matrix()
        self.zoom = 1.0
        self.pan_pixels = np.zeros(2)
        self._show_labels = True
        self.redraw()

    def set_camera_plane(self, plane: str) -> None:
        self.camera = camera_matrix_for_plane(plane)
        self.zoom = 1.0
        self.pan_pixels = np.zeros(2)
        self.redraw()

    def set_subject_view(self, view: str) -> None:
        points = self._fit_points()
        if points.size == 0:
            return
        self.camera = self._subject_camera_matrix(view, points)
        self.zoom = 1.0
        self.pan_pixels = np.zeros(2)
        self.redraw()

    @staticmethod
    def _normalized_vector(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return vector
        return vector / norm

    def _subject_camera_matrix(self, view: str, points: np.ndarray) -> np.ndarray:
        horizontal = self._anatomical_left_axis()
        if horizontal is None:
            return camera_matrix_for_subject_view(view, points)
        vertical = np.asarray((0.0, 0.0, 1.0), dtype=float)
        forward = self._normalized_vector(np.cross(vertical, horizontal))
        key = view.strip().lower()
        matrices = {
            "face": (horizontal, vertical, forward),
            "front": (horizontal, vertical, forward),
            "dos": (-horizontal, vertical, -forward),
            "back": (-horizontal, vertical, -forward),
            "cote": (forward, vertical, horizontal),
            "side": (forward, vertical, horizontal),
        }
        if key not in matrices:
            raise ValueError(f"Unknown subject view: {view}")
        return normalized_camera_matrix(np.asarray(matrices[key], dtype=float))

    def _anatomical_left_axis(self) -> np.ndarray | None:
        for source in ("motive", "captury"):
            if source not in self.visible_marker_sources:
                continue
            data = self.marker_layers.get(source)
            if data is None:
                continue
            axis = self._anatomical_left_axis_from_data(data, source)
            if axis is not None:
                return axis
        return None

    def _anatomical_left_axis_from_data(
        self, data: C3DMarkerData, source: str | None = None
    ) -> np.ndarray | None:
        points = self._marker_layer_points(data, source)
        if points.size == 0:
            return None
        lookup: dict[str, list[int]] = {}
        for index, label in enumerate(data.labels):
            lookup.setdefault(display_marker_name(label), []).append(index)
        vectors: list[np.ndarray] = []
        for left_label, right_label in SUBJECT_LEFT_RIGHT_MARKER_PAIRS:
            left_indices = lookup.get(left_label, [])
            right_indices = lookup.get(right_label, [])
            if not left_indices or not right_indices:
                continue
            left = np.nanmean(points[:, left_indices], axis=1)
            right = np.nanmean(points[:, right_indices], axis=1)
            vector = left - right
            vector[2] = 0.0
            if np.all(np.isfinite(vector)) and np.linalg.norm(vector) > 1e-9:
                vectors.append(vector)
        if not vectors:
            return None
        axis = np.nanmean(np.vstack(vectors), axis=0)
        axis[2] = 0.0
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 1e-9:
            return None
        return self._normalized_vector(axis)

    def _current_points(self) -> np.ndarray:
        layers: list[np.ndarray] = []
        for source in ("captury", "motive", "biobuddy"):
            if source not in self.visible_marker_sources:
                continue
            data = self.marker_layers.get(source)
            if data is None:
                continue
            layers.append(self._marker_layer_points(data, source))
        if not layers:
            return np.empty((3, 0))
        return np.column_stack(layers)

    def _reference_time_s(self) -> float:
        reference = self.marker_layers.get("motive") or getattr(self, "data", None)
        if reference is None or reference.n_frames <= 0:
            return 0.0
        if reference.time is not None and len(reference.time):
            index = max(0, min(self.frame, reference.n_frames - 1))
            return float(reference.time[index])
        return float(self.frame) / max(float(reference.rate), 1e-12)

    def _marker_layer_points(
        self, data: C3DMarkerData, source: str | None = None
    ) -> np.ndarray:
        if data.n_frames <= 0:
            return np.empty((3, 0))
        source_key = str(source or "motive").strip().lower()
        offsets = getattr(self, "marker_time_offsets_s", {})
        original_time = self._reference_time_s() - offsets.get(source_key, 0.0)
        data_time = (
            np.asarray(data.time, dtype=float)
            if data.time is not None
            else np.arange(data.n_frames, dtype=float) / max(float(data.rate), 1e-12)
        )
        if original_time < data_time[0] or original_time > data_time[-1]:
            return np.full((3, len(data.labels)), np.nan, dtype=float)
        frame = int(np.argmin(np.abs(data_time - original_time)))
        return data.points[:, :, frame]

    def _visible_chain_points_array(self) -> np.ndarray:
        if self.chain_data is None or not self.visible_cor_layers:
            return np.empty((3, 0))
        layers: list[np.ndarray] = []
        for layer in ("captury", "motive", "biobuddy"):
            if layer not in self.visible_cor_layers:
                continue
            joints = self.chain_data.layers.get(layer)
            if not joints:
                continue
            frame_points = self._chain_frame_points(joints, layer)
            if frame_points:
                layers.append(np.column_stack(list(frame_points.values())))
        if not layers:
            return np.empty((3, 0))
        return np.column_stack(layers)

    def _fit_points(self) -> np.ndarray:
        points = self._current_points()
        if points.size:
            return points
        return self._visible_chain_points_array()

    def _on_mouse_down(self, event: tk.Event) -> None:
        self.focus_set()
        self._drag_pan = bool(event.state & 0x0001)
        self._is_dragging = True
        self._show_labels = False
        self._last_mouse_xy = (float(event.x), float(event.y))
        self.configure(cursor="fleur")

    def _on_mouse_drag(self, event: tk.Event) -> None:
        if not self._is_dragging or self._last_mouse_xy is None:
            return
        last_x, last_y = self._last_mouse_xy
        delta_x = float(event.x) - last_x
        delta_y = float(event.y) - last_y
        if self._drag_pan:
            self.pan_pixels += (delta_x, delta_y)
        else:
            self.camera = normalized_camera_matrix(
                rotation_matrix_from_drag(delta_x, delta_y) @ self.camera
            )
        self._last_mouse_xy = (float(event.x), float(event.y))
        self._request_redraw()

    def _on_mouse_up(self, _event: tk.Event) -> None:
        self._is_dragging = False
        self._last_mouse_xy = None
        self.configure(cursor="")
        if self._show_labels_after_id:
            self.after_cancel(self._show_labels_after_id)
        self._show_labels_after_id = self.after(120, self._restore_labels)

    def _restore_labels(self) -> None:
        self._show_labels_after_id = None
        self._show_labels = True
        self.redraw()

    def _on_mouse_wheel(self, event: tk.Event) -> None:
        if event.delta:
            self._zoom(1.12 if event.delta > 0 else 1 / 1.12)
        return "break"

    def _zoom(self, factor: float) -> None:
        self.zoom = max(0.05, min(self.zoom * factor, 30.0))
        self.redraw()

    def _show_context_menu(self, event: tk.Event) -> None:
        menu = tk.Menu(self, tearoff=False)
        for plane in ("XY", "YZ", "XZ"):
            menu.add_command(
                label=plane, command=lambda plane=plane: self.set_camera_plane(plane)
            )
        menu.add_separator()
        for label, view in (("Face", "face"), ("Dos", "dos"), ("Côté", "cote")):
            menu.add_command(
                label=label, command=lambda view=view: self.set_subject_view(view)
            )
        menu.tk_popup(event.x_root, event.y_root)

    def _request_redraw(self) -> None:
        if self._redraw_after_id is not None:
            return
        self._redraw_after_id = self.after(33, self._flush_redraw)

    def _flush_redraw(self) -> None:
        self._redraw_after_id = None
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        if not self.marker_layers and self.chain_data is None:
            self._draw_empty_message(width, height, "Sélectionner un essai")
            return
        fit_points = self._fit_points()
        if fit_points.size == 0:
            self._draw_empty_message(width, height, "Aucune couche visible")
            return
        center, scale = fit_center_and_scale(fit_points, width, height, self.camera)
        center = center - self.camera.T @ np.array(
            [self.pan_pixels[0], -self.pan_pixels[1], 0.0]
        ) / max(scale * self.zoom, 1e-12)
        for source in ("captury", "motive", "biobuddy"):
            if source not in self.visible_marker_sources:
                continue
            data = self.marker_layers.get(source)
            if data is not None:
                self._draw_marker_layer(
                    data, source, center, scale * self.zoom, width, height
                )
        self._draw_joint_centre_chains(center, scale * self.zoom, width, height)
        self._draw_axes(width, height)
        self._draw_frame_badge(width, height)

    def _draw_marker_layer(
        self,
        data: C3DMarkerData,
        source: str,
        center: np.ndarray,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        points = self._marker_layer_points(data, source)
        if points.size == 0:
            return
        screen, depth = project_points(
            points, self.camera, center, scale, width, height
        )
        display_labels = (
            list(data.display_labels)
            if data.display_labels is not None
            and len(data.display_labels) == len(data.labels)
            else marker_display_labels(data.labels)
        )
        raw_display_labels = marker_display_labels(data.labels)
        for index in np.argsort(depth):
            point = points[:, index]
            if not np.all(np.isfinite(point)):
                continue
            label = data.labels[index]
            display_label = display_labels[index]
            selected = (
                label == self.selected_label
                or display_label == self.selected_label
                or raw_display_labels[index] == self.selected_label
                or label in self.selected_markers.get(source, set())
                or display_label in self.selected_markers.get(source, set())
                or raw_display_labels[index] in self.selected_markers.get(source, set())
            )
            x = float(screen[0, index])
            y = float(screen[1, index])
            radius = 7 if selected else 3
            color = data_source_marker_color(source)
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill=color,
                outline=VIEWER_SELECTED_OUTLINE_COLOR if selected else color,
                width=2 if selected else 1,
            )
            if self._show_labels and (selected or len(data.labels) <= 80):
                self.create_text(
                    x + 6,
                    y - 6,
                    text=display_label,
                    anchor="sw",
                    fill="#334155",
                    font=("TkDefaultFont", 8),
                )

    def _draw_joint_centre_chains(
        self, center: np.ndarray, scale: float, width: int, height: int
    ) -> None:
        if self.chain_data is None:
            return
        for layer in ("captury", "motive", "biobuddy"):
            if layer not in self.visible_cor_layers:
                continue
            joints = self.chain_data.layers.get(layer)
            if not joints:
                continue
            color = COR_LAYER_COLORS.get(layer, "#111827")
            frame_points = self._chain_frame_points(joints, layer)
            self._draw_chain_edges(frame_points, color, center, scale, width, height)
            self._draw_chain_points(frame_points, color, center, scale, width, height)
        if self.show_chain_axes:
            for layer in self._visible_chain_axis_layers():
                joints = self.chain_data.layers.get(layer) if self.chain_data else None
                if not joints:
                    continue
                frame_points = self._chain_frame_points(joints, layer)
                self._draw_chain_axes(layer, frame_points, center, scale, width, height)

    def _visible_chain_axis_layers(self) -> tuple[str, ...]:
        if self.chain_data is None:
            return ()
        available = tuple(
            layer
            for layer in ("captury", "motive", "biobuddy")
            if self.chain_data.layers.get(layer)
        )
        visible = tuple(
            layer for layer in available if layer in self.visible_cor_layers
        )
        return visible or available

    def _chain_frame_points(
        self, joints: dict[str, np.ndarray], layer: str | None = None
    ) -> dict[str, np.ndarray]:
        points: dict[str, np.ndarray] = {}
        for joint, values in joints.items():
            if values.size == 0:
                continue
            times = (
                self.chain_data.times.get(str(layer), {}).get(joint)
                if self.chain_data is not None and layer is not None
                else None
            )
            if times is not None and len(times) == values.shape[0]:
                reference_time = self._reference_time_s()
                if reference_time < times[0] or reference_time > times[-1]:
                    continue
                index = int(np.argmin(np.abs(times - reference_time)))
            elif self.n_frames <= 1 or values.shape[0] <= 1:
                index = min(self.frame, values.shape[0] - 1)
            else:
                ratio = self.frame / max(1, self.n_frames - 1)
                index = int(round(ratio * (values.shape[0] - 1)))
            point = values[index]
            if np.all(np.isfinite(point)):
                points[joint] = point
        return points

    def _draw_chain_edges(
        self,
        points: dict[str, np.ndarray],
        color: str,
        center: np.ndarray,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        edges = self.chain_data.edges if self.chain_data else ()
        for proximal, distal in edges:
            if proximal not in points or distal not in points:
                continue
            screen, _depth = project_points(
                np.column_stack((points[proximal], points[distal])),
                self.camera,
                center,
                scale,
                width,
                height,
            )
            self.create_line(
                float(screen[0, 0]),
                float(screen[1, 0]),
                float(screen[0, 1]),
                float(screen[1, 1]),
                fill=color,
                width=2,
            )

    def _draw_chain_points(
        self,
        points: dict[str, np.ndarray],
        color: str,
        center: np.ndarray,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        if not points:
            return
        names = list(points)
        screen, _depth = project_points(
            np.column_stack([points[name] for name in names]),
            self.camera,
            center,
            scale,
            width,
            height,
        )
        for index, name in enumerate(names):
            x = float(screen[0, index])
            y = float(screen[1, index])
            self.create_oval(
                x - 4,
                y - 4,
                x + 4,
                y + 4,
                fill="#ffffff",
                outline=color,
                width=2,
            )
            if self._show_labels and len(names) <= 28:
                self.create_text(
                    x + 6,
                    y + 5,
                    text=name,
                    anchor="nw",
                    fill=color,
                    font=("TkDefaultFont", 8),
                )

    def _draw_chain_axes(
        self,
        layer: str,
        points: dict[str, np.ndarray],
        center: np.ndarray,
        scale: float,
        width: int,
        height: int,
    ) -> None:
        if not points or self.chain_data is None:
            return
        axis_length = max(25.0 / max(scale, 1e-9), 25.0)
        for joint, origin in points.items():
            axes = self._chain_axes_for_layer(layer, joint, points)
            if axes is None:
                continue
            for label, direction in axes.items():
                screen, _depth = project_points(
                    np.column_stack((origin, origin + direction * axis_length)),
                    self.camera,
                    center,
                    scale,
                    width,
                    height,
                )
                self.create_line(
                    float(screen[0, 0]),
                    float(screen[1, 0]),
                    float(screen[0, 1]),
                    float(screen[1, 1]),
                    fill=VIEWER_AXIS_COLORS[label],
                    width=2,
                )

    def _chain_axes_for_layer(
        self, layer: str, joint: str, points: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray] | None:
        if self.chain_data is None:
            return None
        axes = local_chain_axes(joint, points, self.chain_data.edges)
        if axes is None:
            return None
        if self.rotate_body_segments_180_x and layer in {"captury", "motive"}:
            return {"X": -axes["X"], "Y": -axes["Y"], "Z": axes["Z"]}
        return axes

    def _draw_empty_message(self, width: int, height: int, message: str) -> None:
        self.create_text(
            width / 2,
            height / 2,
            text=message,
            fill="#64748b",
            font=("TkDefaultFont", 12),
        )

    def _draw_axes(self, width: int, height: int) -> None:
        origin_x = width - 72
        origin_y = height - 48
        self.create_rectangle(
            origin_x - 22,
            origin_y - 42,
            origin_x + 62,
            origin_y + 28,
            fill="#ffffff",
            outline="#e5e7eb",
        )
        axes = (
            ("X", np.asarray((1.0, 0.0, 0.0))),
            ("Y", np.asarray((0.0, 1.0, 0.0))),
            ("Z", np.asarray((0.0, 0.0, 1.0))),
        )
        for label, vector in axes:
            projected = self.camera @ vector
            end_x = origin_x + 30.0 * projected[0]
            end_y = origin_y - 30.0 * projected[1]
            color = VIEWER_AXIS_COLORS[label]
            self.create_line(origin_x, origin_y, end_x, end_y, fill=color, width=3)
            self.create_text(end_x + 5, end_y - 5, text=label, fill=color, anchor="w")

    def _draw_frame_badge(self, width: int, _height: int) -> None:
        if self.data is None:
            return
        text = f"{self.frame + 1}/{self.data.n_frames}"
        self.create_rectangle(8, 8, 86, 32, fill="#ffffff", outline="#e5e7eb")
        self.create_text(16, 20, text=text, anchor="w", fill="#334155")


def open_trial_view_window(parent, source, title="Visu 3D essai"):
    """Open an independent view of the currently loaded markers and CoR chains."""
    window = tk.Toplevel(parent)
    window.title(title)
    window.geometry("1100x800")
    toolbar = ttk.Frame(window)
    toolbar.pack(fill="x", padx=8, pady=6)
    viewer = TkC3DTrialCanvas(window)
    viewer.pack(fill="both", expand=True, padx=8)
    for name in ("data", "chain_data", "frame", "marker_source", "selected_label",
                 "show_chain_axes", "rotate_body_segments_180_x", "zoom"):
        setattr(viewer, name, getattr(source, name))
    viewer.marker_layers = dict(source.marker_layers)
    viewer.marker_time_offsets_s = dict(source.marker_time_offsets_s)
    viewer.selected_markers = {k: set(v) for k, v in source.selected_markers.items()}
    viewer.visible_marker_sources = set(source.visible_marker_sources)
    viewer.visible_cor_layers = set(source.visible_cor_layers)
    viewer.camera = source.camera.copy()
    viewer.pan_pixels = source.pan_pixels.copy()
    ttk.Label(toolbar, text="Glisser : rotation • Maj + glisser : déplacer • Molette : zoom").pack(side="left")
    for label, action in (("−", lambda: viewer._zoom(1 / 1.25)),
                          ("+", lambda: viewer._zoom(1.25)), ("Recentrer", viewer.reset_camera)):
        ttk.Button(toolbar, text=label, command=action).pack(side="right", padx=3)
    layers = ttk.Frame(window)
    layers.pack(fill="x", padx=8, pady=4)
    variables = []
    for label, available, visible in (
        ("Marqueurs", viewer.marker_layers, viewer.visible_marker_sources),
        ("CoR", viewer.chain_data.layers if viewer.chain_data else {}, viewer.visible_cor_layers),
    ):
        ttk.Label(layers, text=label).pack(side="left", padx=5)
        for key in available:
            var = tk.BooleanVar(window, value=key in visible)
            variables.append(var)
            def toggle(key=key, var=var, visible=visible):
                visible.add(key) if var.get() else visible.discard(key)
                viewer.redraw()
            ttk.Checkbutton(layers, text=COR_LAYER_LABELS.get(key, key), variable=var,
                            command=toggle).pack(side="left")
    controls = ttk.Frame(window)
    controls.pack(fill="x", padx=8, pady=8)
    frame = tk.DoubleVar(window, value=viewer.frame)
    playing = False
    timer = None
    def seek(value):
        viewer.set_frame(int(float(value)))
    slider = ttk.Scale(controls, from_=0, to=max(0, viewer.n_frames - 1),
                       variable=frame, command=seek)
    def tick():
        nonlocal timer
        timer = None
        if playing and viewer.n_frames:
            viewer.set_frame((viewer.frame + 1) % viewer.n_frames)
            frame.set(viewer.frame)
            rate = viewer.data.rate if viewer.data is not None else 120
            timer = window.after(max(1, round(1000 / max(rate, 1))), tick)
    def toggle_play():
        nonlocal playing, timer
        playing = not playing
        button.configure(text="Pause" if playing else "Lecture")
        if playing:
            tick()
        elif timer is not None:
            window.after_cancel(timer)
            timer = None
    button = ttk.Button(controls, text="Lecture", command=toggle_play)
    button.pack(side="left", padx=4)
    slider.pack(side="left", fill="x", expand=True)
    def close():
        nonlocal playing
        playing = False
        if timer is not None:
            window.after_cancel(timer)
        window.destroy()
    window.protocol("WM_DELETE_WINDOW", close)
    window.viewer = viewer
    window.layer_variables = variables
    viewer.redraw()
    return window
