"""Graph configuration and drawing helpers for the Captury/BioBuddy GUI."""

from __future__ import annotations

from typing import Iterable
from pathlib import Path

import numpy as np

from gui_trial_viewer import COR_LAYER_LABELS, data_source_color

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional plotting dependency
    pd = None

GRAPH_CONFIGS = {
    "occlusions": {
        "csv": "all_motive_marker_occlusions.csv",
        "groups": ("marker",),
        "metrics": ("missing_percent", "missing_frames"),
        "title": "Occlusions Motive",
    },
    "dimensions": {
        "csv": "all_model_dimensions.csv",
        "groups": ("system", "dimension"),
        "metrics": ("median_length_mm", "sd_length_mm"),
        "title": "Dimensions modèles",
    },
    "centres": {
        "csv": "all_joint_centre_metrics.csv",
        "groups": ("joint",),
        "metrics": (
            "median_error_mm",
            "p95_error_mm",
            "max_error_mm",
            "mae_x",
            "mae_y",
            "mae_z",
            "bias_x",
            "bias_y",
            "bias_z",
            "loa_lower_x",
            "loa_upper_x",
            "loa_lower_y",
            "loa_upper_y",
            "loa_lower_z",
            "loa_upper_z",
            "mae_euclidean",
            "rmse_euclidean",
        ),
        "title": "Centres articulaires",
    },
    "segments": {
        "csv": "all_segment_rotation_metrics.csv",
        "groups": ("reference", "source", "segment"),
        "metrics": (
            "median_global_deg",
            "p95_global_deg",
            "max_global_deg",
            "rmse_global_deg",
            "median_geodesic_deg",
            "p95_geodesic_deg",
            "max_geodesic_deg",
            "rmse_geodesic_deg",
            "median_abs_x_deg",
            "median_abs_y_deg",
            "median_abs_z_deg",
            "p95_abs_x_deg",
            "p95_abs_y_deg",
            "p95_abs_z_deg",
            "bias_x_deg",
            "bias_y_deg",
            "bias_z_deg",
            "loa_lower_x_deg",
            "loa_upper_x_deg",
            "loa_lower_y_deg",
            "loa_upper_y_deg",
            "loa_lower_z_deg",
            "loa_upper_z_deg",
        ),
        "title": "Segments",
    },
    "skin_markers": {
        "csv": "all_skin_marker_correspondence_metrics.csv",
        "groups": ("landmark",),
        "metrics": ("median_error_mm", "p95_error_mm", "rmse_error_mm"),
        "title": "Marqueurs cutanés",
    },
    "kinematics": {
        "csv": "all_kinematics_q_metrics.csv",
        "groups": ("source", "q_name"),
        "metrics": (
            "mean_rad",
            "rom_rad",
            "mean_native",
            "rom_native",
            "bias_rad",
            "mae_rad",
            "rmse_rad",
            "loa_lower_rad",
            "loa_upper_rad",
            "sd_difference_rad",
            "nrmse_range",
            "pearson_r_waveform",
            "lin_ccc_waveform",
            "linear_gain",
            "linear_offset_rad",
            "reference_rom_rad",
            "test_rom_rad",
            "rom_difference_rad",
            "max_time_difference_s",
            "min_time_difference_s",
            "paired_coverage",
            "bias_native",
            "mae_native",
            "rmse_native",
            "loa_lower_native",
            "loa_upper_native",
            "sd_difference_native",
            "linear_offset_native",
            "reference_rom_native",
            "test_rom_native",
            "rom_difference_native",
            "c3d_mean_deg",
            "c3d_sd_deg",
            "c3d_min_deg",
            "c3d_max_deg",
        ),
        "title": "Cinématiques",
    },
}


def read_table_npz(path: str | Path) -> "pd.DataFrame":
    if pd is None:
        raise RuntimeError("pandas is required to read GUI table npz files.")
    with np.load(Path(path), allow_pickle=False) as data:
        columns = [str(column) for column in data["columns"]]
        values = {column: data[f"col_{index}"] for index, column in enumerate(columns)}
    return pd.DataFrame(values)


EVENT_METRICS = (
    "movement_speed_mm_s",
    "left_foot_z_mm",
    "right_foot_z_mm",
    "left_foot_speed_mm_s",
    "right_foot_speed_mm_s",
    "left_contact",
    "right_contact",
    "motive_composite_speed",
    "captury_composite_speed",
    "difference",
    "reference",
    "moving",
)
KINEMATIC_RAD_METRICS = {
    "mean_rad",
    "rom_rad",
    "bias_rad",
    "mae_rad",
    "rmse_rad",
    "loa_lower_rad",
    "loa_upper_rad",
    "sd_difference_rad",
    "linear_offset_rad",
    "reference_rom_rad",
    "test_rom_rad",
    "rom_difference_rad",
}
KINEMATIC_TIMESERIES_COLUMNS = (
    "motive",
    "captury",
    "biobuddy",
    "captury_c3d",
    "difference",
)
SEGMENT_TIMESERIES_COLUMNS = (
    "global_deg",
    "x_deg",
    "y_deg",
    "z_deg",
)


def graph_metric_columns(
    dataframe: "pd.DataFrame", requested_metrics: Iterable[str]
) -> list[str]:
    if pd is None:
        return []
    return [
        metric
        for metric in requested_metrics
        if metric in dataframe.columns
        and pd.api.types.is_numeric_dtype(dataframe[metric])
    ]


def is_rotation_q_name(q_name: str) -> bool:
    return "_rot" in str(q_name).lower() or str(q_name).lower().endswith(
        ("_xrotation", "_yrotation", "_zrotation")
    )


def metric_display_name(metric: str, *, q_name: str | None = None) -> str:
    if metric in KINEMATIC_RAD_METRICS:
        return metric.replace("_rad", "_deg")
    if metric in KINEMATIC_TIMESERIES_COLUMNS and q_name is not None:
        return f"{metric} ({'deg' if is_rotation_q_name(q_name) else 'native'})"
    return metric


def values_for_display(
    values: "pd.Series | np.ndarray", metric: str, *, q_name: str | None = None
) -> "pd.Series | np.ndarray":
    if metric in KINEMATIC_RAD_METRICS or (
        metric in KINEMATIC_TIMESERIES_COLUMNS
        and q_name is not None
        and is_rotation_q_name(q_name)
    ):
        return values * (180.0 / np.pi)
    return values


def joint_centre_error_timeseries(
    dataframe: "pd.DataFrame", joint: str
) -> "pd.DataFrame":
    return joint_centre_error_timeseries_between(dataframe, joint, "motive", "captury")


def joint_centre_error_timeseries_between(
    dataframe: "pd.DataFrame", joint: str, reference: str, source: str
) -> "pd.DataFrame":
    reference = str(reference).strip().lower()
    source = str(source).strip().lower()
    required = {
        "time",
        "joint",
        f"{reference}_x_mm",
        f"{reference}_y_mm",
        f"{reference}_z_mm",
        f"{source}_x_mm",
        f"{source}_y_mm",
        f"{source}_z_mm",
    }
    if not required.issubset(dataframe.columns):
        return pd.DataFrame() if pd is not None else dataframe
    values = dataframe[dataframe["joint"].astype(str) == str(joint)].copy()
    if values.empty:
        return values
    for axis in ("x", "y", "z"):
        values[f"error_{axis}_mm"] = values[f"{source}_{axis}_mm"].astype(
            float
        ) - values[f"{reference}_{axis}_mm"].astype(float)
        values[f"abs_error_{axis}_mm"] = values[f"error_{axis}_mm"].abs()
    values["distance_mm"] = np.linalg.norm(
        values[["error_x_mm", "error_y_mm", "error_z_mm"]].to_numpy(dtype=float),
        axis=1,
    )
    values["time"] = values["time"].astype(float)
    return values.sort_values("time")


def draw_joint_centre_error_timeseries(
    axes: object, dataframe: "pd.DataFrame", trial: str, joint: str
) -> bool:
    return draw_joint_centre_error_timeseries_between(
        axes, dataframe, trial, joint, "motive", "captury"
    )


def draw_joint_centre_error_timeseries_between(
    axes: object,
    dataframe: "pd.DataFrame",
    trial: str,
    joint: str,
    reference: str,
    source: str,
) -> bool:
    values = joint_centre_error_timeseries_between(dataframe, joint, reference, source)
    if values.empty:
        axes.set_title(f"Aucune erreur temporelle: {source} vs {reference} / {joint}")
        return False
    curves = (
        ("distance_mm", "distance", "#111827"),
        ("abs_error_x_mm", "|x|", "#ef4444"),
        ("abs_error_y_mm", "|y|", "#22c55e"),
        ("abs_error_z_mm", "|z|", "#3b82f6"),
    )
    for column, label, color in curves:
        if column in values.columns:
            axes.plot(
                values["time"], values[column].astype(float), label=label, color=color
            )
    axes.set_title(f"{trial} - {source} vs {reference} - {joint}")
    axes.set_xlabel("Temps (s)")
    axes.set_ylabel("Erreur (mm)")
    axes.legend()
    axes.grid(alpha=0.3)
    return True


def draw_segment_rotation_timeseries(
    axes: object,
    dataframe: "pd.DataFrame",
    trial: str,
    source: str,
    segment: str,
) -> bool:
    required = {"time", "source", "segment", *SEGMENT_TIMESERIES_COLUMNS}
    if pd is None or not required.issubset(dataframe.columns):
        axes.set_title("Aucune rotation segmentaire temporelle")
        return False
    values = dataframe[
        (dataframe["source"].astype(str) == str(source))
        & (dataframe["segment"].astype(str) == str(segment))
    ].copy()
    if values.empty:
        axes.set_title(f"Aucune rotation segmentaire: {source} / {segment}")
        return False
    values["time"] = values["time"].astype(float)
    curves = (
        ("global_deg", "global", "#111827"),
        ("x_deg", "x", "#ef4444"),
        ("y_deg", "y", "#22c55e"),
        ("z_deg", "z", "#3b82f6"),
    )
    for column, label, color in curves:
        axes.plot(
            values["time"], values[column].astype(float), label=label, color=color
        )
    axes.set_title(f"{trial} - {source} / {segment} - déviation rotation")
    axes.set_xlabel("Temps (s)")
    axes.set_ylabel("Déviation (deg)")
    axes.legend()
    axes.grid(alpha=0.3)
    return True


def segment_rotation_boxplot_series(
    dataframe: "pd.DataFrame",
    metric: str,
    *,
    trial: str | None = None,
    source: str | None = None,
    segments: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    if pd is None or dataframe.empty:
        return []
    values = dataframe.copy()
    if trial and "trial" in values.columns:
        values = values[values["trial"].astype(str) == str(trial)]
    if source and "source" in values.columns:
        values = values[values["source"].astype(str) == str(source)]
    requested_segments = {str(segment) for segment in segments or () if str(segment)}
    if values.empty or "segment" not in values.columns:
        return []
    metric_column = {
        "median_global_deg": "global_deg",
        "p95_global_deg": "global_deg",
        "max_global_deg": "global_deg",
        "median_abs_x_deg": "abs_x_deg",
        "median_abs_y_deg": "abs_y_deg",
        "median_abs_z_deg": "abs_z_deg",
        "p95_abs_x_deg": "abs_x_deg",
        "p95_abs_y_deg": "abs_y_deg",
        "p95_abs_z_deg": "abs_z_deg",
    }.get(metric, "global_deg")
    series: list[dict[str, object]] = []
    group_columns = ["source", "segment"] if "source" in values.columns else ["segment"]
    for keys, rows in values.groupby(group_columns, sort=True):
        if isinstance(keys, tuple):
            source_label, segment = str(keys[0]), str(keys[1])
            label = f"{source_label} / {segment}"
        else:
            segment = str(keys)
            label = segment
        if requested_segments and segment not in requested_segments:
            continue
        if metric_column not in rows.columns:
            continue
        item_values = rows[metric_column].astype(float).dropna().to_numpy()
        if item_values.size:
            series.append(
                {
                    "metric": metric,
                    "label": label,
                    "values": np.abs(item_values),
                    "dataframe": rows,
                }
            )
    return series


def joint_centre_error_boxplot_series(
    dataframe: "pd.DataFrame",
    metric: str,
    *,
    trial: str | None = None,
    joints: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    return joint_centre_error_boxplot_series_between(
        dataframe,
        metric,
        trial=trial,
        joints=joints,
        reference="motive",
        source="captury",
    )


def joint_centre_error_boxplot_series_between(
    dataframe: "pd.DataFrame",
    metric: str,
    *,
    trial: str | None = None,
    joints: Iterable[str] | None = None,
    reference: str = "motive",
    source: str = "captury",
) -> list[dict[str, object]]:
    if pd is None or dataframe.empty:
        return []
    values = dataframe
    if trial and "trial" in values.columns:
        values = values[values["trial"].astype(str) == str(trial)]
    if values.empty or "joint" not in values.columns:
        return []
    requested_joints = {str(joint) for joint in joints or () if str(joint)}
    metric_column = {
        "mae_x": "abs_error_x_mm",
        "mae_y": "abs_error_y_mm",
        "mae_z": "abs_error_z_mm",
    }.get(metric, "distance_mm")
    series: list[dict[str, object]] = []
    for joint in sorted(values["joint"].astype(str).unique()):
        if requested_joints and joint not in requested_joints:
            continue
        joint_values = joint_centre_error_timeseries_between(
            values, joint, reference, source
        )
        if joint_values.empty or metric_column not in joint_values.columns:
            continue
        joint_errors = (
            joint_values[metric_column].astype(float).replace([np.inf, -np.inf], np.nan)
        )
        joint_errors = joint_errors.dropna().to_numpy()
        if joint_errors.size == 0:
            continue
        series.append(
            {
                "metric": metric,
                "label": joint,
                "source": source,
                "reference": reference,
                "dataframe": joint_values,
                "values": joint_errors,
            }
        )
    return series


def draw_metric_boxplot(
    axes: object, series: list[dict[str, object]], title: str
) -> None:
    labels = [str(item["label"]) for item in series]
    values = [item["values"] for item in series]
    positions = np.arange(1, len(values) + 1, dtype=float)
    boxes = axes.boxplot(
        values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "#111827",
            "markeredgecolor": "#111827",
            "markersize": 4,
        },
        medianprops={"color": "#111827", "linewidth": 1.6},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#64748b",
            "markersize": 3,
            "alpha": 0.7,
        },
    )
    for index, patch in enumerate(boxes["boxes"]):
        metric = str(series[index]["metric"])
        color = data_source_color(metric)
        if color == "#64748b":
            color = f"C{index % 10}"
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(0)
    for index, item in enumerate(series):
        item_values = np.asarray(item["values"], dtype=float)
        jitter = rng.uniform(-0.08, 0.08, size=item_values.shape[0])
        axes.scatter(
            np.full(item_values.shape[0], positions[index]) + jitter,
            item_values,
            s=16,
            alpha=0.65,
            color=boxes["boxes"][index].get_edgecolor(),
            edgecolors="none",
        )
    axes.set_title(title)
    axes.set_ylabel("valeur")
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=45, ha="right")
    axes.grid(axis="y", alpha=0.3)


def draw_dimension_metric_graph(
    axes: object, dataframe: "pd.DataFrame", metric: str
) -> None:
    if "dimension" not in dataframe.columns or "system" not in dataframe.columns:
        axes.set_title("Aucune donnée dimensionnelle")
        return
    values = dataframe.copy()
    values[metric] = values[metric].astype(float)
    dimensions = sorted(values["dimension"].dropna().astype(str).unique())
    systems = [
        system
        for system in ("captury", "motive", "biobuddy")
        if system in set(values["system"].dropna().astype(str).str.lower())
    ]
    for system in sorted(values["system"].dropna().astype(str).str.lower().unique()):
        if system not in systems:
            systems.append(system)
    if not dimensions or not systems:
        axes.set_title("Aucune donnée dimensionnelle")
        return
    x = np.arange(len(dimensions), dtype=float)
    width = min(0.8 / max(1, len(systems)), 0.28)
    offsets = (np.arange(len(systems)) - (len(systems) - 1) / 2.0) * width
    for index, system in enumerate(systems):
        system_rows = values[values["system"].astype(str).str.lower() == system]
        grouped = system_rows.groupby("dimension", sort=False)[metric].median()
        y = [float(grouped.get(dimension, np.nan)) for dimension in dimensions]
        axes.bar(
            x + offsets[index],
            y,
            width=width,
            label=COR_LAYER_LABELS.get(system, system),
            color=data_source_color(system),
        )
    axes.set_title(f"Dimensions modèles - {metric}")
    axes.set_ylabel(metric)
    axes.set_xticks(x)
    axes.set_xticklabels(dimensions, rotation=45, ha="right")
    axes.legend()
    axes.grid(axis="y", alpha=0.3)
