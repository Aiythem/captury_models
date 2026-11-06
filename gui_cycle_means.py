"""Means of checked cycles, retaining source-specific coordinate definitions."""

from pathlib import Path
import numpy as np
from gait_cycles import normalize_cycle
from gui_gait import cycle_options, selected_cycle_keys
from gui_graphs import read_table_npz
from native_angle_signs import correct_native_sign_table, PLANE_AXES, SIGN_REFERENCE


def selected_leg_means(raw, report, excluded=(), *, side="right", trial_dir=None):
    """Average checked coordinates with the same fixed signs used for CSV export."""
    selected = selected_cycle_keys(report, excluded)
    cycles = [
        c
        for c in cycle_options(report)
        if c["side"] == side and c["selection_key"] in selected
    ]
    if raw.empty or not cycles:
        return []
    raw = correct_native_sign_table(raw, trial_dir) if "sign_status" not in raw else raw
    data = raw[
        (raw.side == side)
        & raw.source.isin(["motive", "biobuddy", "captury"])
        & (raw.unit == "deg")
    ]
    channels = []
    for (source, joint), group in data.groupby(["source", "joint"], sort=False):
        for name, rows in group.groupby("q_name", sort=False):
            rows = rows.sort_values("time_s")
            channels.append(
                dict(
                    source=source,
                    joint=joint,
                    axis=str(rows.iloc[0]["axis"]),
                    q_name=name,
                    time=rows.time_s.to_numpy(dtype=float),
                    values=rows.value.to_numpy(dtype=float),
                    definition="native_coordinate",
                    sign_multiplier=int(rows.iloc[0].get("sign_multiplier", 1)),
                    sign_status=str(
                        rows.iloc[0].get("sign_status", "pending_model_evidence")
                    ),
                    positive_direction=str(rows.iloc[0].get("positive_direction", "")),
                    plane=str(rows.iloc[0].get("sign_plane", "")),
                )
            )
    result = []
    for channel in channels:
        curves = []
        used = []
        for cycle in cycles:
            values, status = normalize_cycle(
                channel["time"],
                channel["values"],
                cycle,
                n_points=int(report.get("points_per_cycle", 101)),
                max_gap_s=float(report.get("maximum_interpolation_gap_s", 0.05)),
                angular=True,
            )
            if status != "ok" or not np.isfinite(values).all():
                continue
            # Equivalent +/-360 branches must not cancel between repetitions.
            if curves:
                values = values - 360 * np.round(np.median(values - curves[0]) / 360)
            curves.append(values)
            used.append(cycle["cycle_id"])
        if curves:
            result.append(
                {k: v for k, v in channel.items() if k not in {"time", "values"}}
                | dict(
                    phase_percent=np.linspace(0, 100, len(curves[0])),
                    mean=np.mean(curves, axis=0),
                    cycle_ids=used,
                    n_cycles=len(curves),
                    side=side,
                )
            )
    return result


JOINT_LABELS = {"hip": "Hip", "knee": "Knee", "ankle": "Ankle"}


def positive_squat_peak(values, task):
    """Reverse an entire mean curve if its largest absolute peak is negative.

    Display-only: preserve offsets and curve shape, never take elementwise abs.
    Other tasks retain their native sign.
    """
    values = np.asarray(values, dtype=float)
    valid = values[np.isfinite(values)]
    sign = (
        -1 if task == "squat" and valid.size and valid[np.argmax(abs(valid))] < 0 else 1
    )
    return values * sign, sign


def draw_selected_leg_means(
    figure,
    trial_dir,
    report,
    excluded=(),
    *,
    side="right",
    joint="hip",
    curves=None,
):
    """One joint per Figure, with sagittal, frontal and transverse panels."""
    if joint not in JOINT_LABELS:
        raise ValueError(f"Unknown joint: {joint}")
    figure.clear()
    figure.set_layout_engine("constrained")
    axes = np.atleast_1d(figure.subplots(1, 3))
    if curves is None:
        path = Path(trial_dir) / "lower_limb_kinematics.npz"
        raw = read_table_npz(path) if path.is_file() else None
        curves = (
            selected_leg_means(raw, report, excluded, side=side, trial_dir=trial_dir)
            if raw is not None
            else []
        )
    colors = {"motive": "#0ea5e9", "biobuddy": "#22a447", "captury": "#f97316"}
    available_sources = {c["source"] for c in curves}
    for ax, plane in zip(axes, PLANE_AXES):
        found = [
            c
            for c in curves
            if c["joint"] == joint and c["axis"] == PLANE_AXES[plane].get(c["source"])
        ]
        for curve in found:
            values = curve["mean"]
            sign = int(curve.get("sign_multiplier", 1))
            source = {"motive": "Motive", "biobuddy": "BioBuddy", "captury": "Captury"}[
                curve["source"]
            ]
            status = curve.get("sign_status", "pending_model_evidence")
            note = (
                f"×{sign:+d}"
                if status == "model_reference_polarity"
                else "signe natif · à vérifier"
            )
            label = f"{source} · {curve['q_name']}\n(n={curve['n_cycles']}) · {note}"
            ax.plot(
                curve["phase_percent"],
                values,
                color=colors[curve["source"]],
                label=label,
                linewidth=1.8,
            )
        missing = available_sources - {c["source"] for c in found}
        title = plane
        if missing:
            title += "\nIndisponible :\n" + ", ".join(sorted(missing))
        ax.set_title(title, fontsize=10)
        positive = sorted({c.get("positive_direction", "") for c in found} - {""})
        ax.set_ylabel(
            "°; + " + "/".join(positive) if positive else "°; signe natif à vérifier"
        )
        ax.set_xlim(0, 100)
        ax.grid(alpha=0.25)
        if found:
            ax.legend(fontsize=7, loc="best")
        else:
            ax.text(
                0.5,
                0.5,
                "Aucun cycle sélectionné valide",
                ha="center",
                transform=ax.transAxes,
                fontsize=8,
            )
    for ax in axes:
        ax.set_xlabel("Cycle normalisé (%)")
    figure.suptitle(
        f"{JOINT_LABELS[joint]} — {side} — moyennes sélectionnées\n"
        "Signes fixes · Fukuchi et al. (2018) · DOI: "
        + SIGN_REFERENCE["doi"]
        + "\nCoordonnées natives; équivalence anatomique non établie",
        fontsize=9,
    )
    return list(axes)


def draw_joint_figures(figures, trial_dir, report, excluded=(), *, side="right"):
    """Update three independent Figures using exactly the same checked cycles."""
    path = Path(trial_dir) / "lower_limb_kinematics.npz"
    curves = (
        selected_leg_means(
            read_table_npz(path), report, excluded, side=side, trial_dir=trial_dir
        )
        if path.is_file()
        else []
    )
    return {
        joint: draw_selected_leg_means(
            figures[joint],
            trial_dir,
            report,
            excluded,
            side=side,
            joint=joint,
            curves=curves,
        )
        for joint in JOINT_LABELS
    }
