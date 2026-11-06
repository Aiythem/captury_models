"""Review normalized task cycles and export the selected subset on request."""

from pathlib import Path
from native_angle_signs import (
    correct_native_sign_table,
    SIGN_FIELDS,
    SIGN_REFERENCE,
    profiles_for_trial,
)


def cycle_options(report):
    """Return independently selectable left/right views of each event interval."""
    return [
        {**cycle, "side": side, "selection_key": side + "::" + cycle["cycle_id"]}
        for cycle in report.get("cycles", [])
        for side in (("left", "right") if cycle["side"] == "both" else (cycle["side"],))
    ]


def selected_cycle_keys(report, excluded):
    excluded = set(excluded)
    return {
        c["selection_key"]
        for c in cycle_options(report)
        if c["selection_key"] not in excluded and c["cycle_id"] not in excluded
    }


def cycle_row_keys(dataframe):
    return dataframe.side.astype(str) + "::" + dataframe.cycle_id.astype(str)


def draw_gait_cycles(
    axes, dataframe, filters, *, excluded_cycle_ids=(), task="gait", trial_dir=None
):
    axes.clear()
    selected = correct_native_sign_table(dataframe, trial_dir)
    if "cycle_id" in selected:
        selected = selected[
            ~selected.cycle_id.isin(excluded_cycle_ids)
            & ~cycle_row_keys(selected).isin(excluded_cycle_ids)
        ]
    for key, value in filters.items():
        if key in selected.columns:
            selected = selected[selected[key].astype(str) == str(value)]
    if selected.empty:
        axes.set_title("Aucun cycle complet disponible")
        return
    # A panel shows one source/side/DoF, with all its cycles on the same axes.
    first = selected.iloc[0]
    for key in ("source", "side", "joint", "q_name"):
        selected = selected[selected[key] == first[key]]
    plotted = 0
    for cycle_id, rows in selected.groupby("cycle_id", sort=True):
        rows = rows.sort_values("phase_percent")
        if not (rows.normalization_status == "ok").all():
            continue
        (line,) = axes.plot(rows.phase_percent, rows.value, label=cycle_id)
        if "toe_off_percent" in rows and rows.toe_off_percent.notna().any():
            axes.axvline(
                float(rows.toe_off_percent.dropna().iloc[0]),
                color=line.get_color(),
                linestyle=":",
                alpha=0.4,
            )
        if (
            task == "squat"
            and "peak_percent" in rows
            and rows.peak_percent.notna().any()
        ):
            axes.axvline(
                float(rows.peak_percent.dropna().iloc[0]),
                color=line.get_color(),
                linestyle=":",
                alpha=0.5,
            )
        plotted += 1
    axes.set_xlabel(
        {
            "gait": "Marche — contact au contact suivant (%)",
            "squat": "Squat — debout → flexion → debout (%)",
            "jump": "Saut — décollage → réception (%)",
        }.get(task, "Cycle (%)")
    )
    positive = str(first.get("positive_direction", ""))
    axes.set_ylabel(
        str(first["unit"])
        + ("; + " + positive if positive else "; signe natif à vérifier")
    )
    axes.set_xlim(0, 100)
    axes.grid(alpha=0.3)
    title = f"{first['trial']} / {first['source']} / {first['side']} / {first['joint']} / {first['q_name']}"
    if plotted:
        axes.legend(title="Cycles")
    else:
        reasons = {
            "gap_exceeds_limit": "lacune trop longue",
            "outside_source_coverage": "couverture temporelle insuffisante",
        }
        title += "\nIndisponible: " + ", ".join(
            reasons.get(status, status)
            for status in sorted(set(selected.normalization_status))
        )
    note = (
        f"×{int(first.sign_multiplier):+d}"
        if first.sign_status == "model_reference_polarity"
        else "signe natif · à vérifier"
    )
    axes.set_title(title + "\n" + note + " · Fukuchi et al. (2018)")


def cycle_selection(trial_dir):
    """Exclusions apply only to the exact reviewed detection report."""
    import json, hashlib

    directory = Path(trial_dir)
    report_path = directory / "gait_cycles.json"
    if not report_path.is_file():
        return {}, [], ""
    signature = hashlib.sha256(report_path.read_bytes()).hexdigest()
    report = json.loads(report_path.read_text())
    excluded = []
    path = directory / "cycle_selection.json"
    if path.is_file():
        saved = json.loads(path.read_text())
        if saved.get("report_sha256") == signature:
            excluded = saved.get("excluded_cycle_ids", [])
    return report, excluded, signature


def save_cycle_selection(trial_dir, excluded, signature):
    import json

    (Path(trial_dir) / "cycle_selection.json").write_text(
        json.dumps(
            {"report_sha256": signature, "excluded_cycle_ids": sorted(excluded)},
            indent=2,
        )
    )


def normalized_export_points(raw, normalized, metrics, report):
    """Rebuild every exported native channel/cycle on 101 points numbered 1..101.

    Use original samples rather than resampling an already interpolated cache.
    Missing coverage/gaps retain normalize_cycle's unavailable status and NaNs.
    """
    import numpy as np
    import pandas as pd
    from gait_cycles import normalize_cycle

    keys = ["source", "side", "joint", "q_name", "cycle_id"]
    intervals = {(c["side"], c["cycle_id"]): c for c in cycle_options(report)}
    rows, summaries = [], []
    for identity, group in normalized.groupby(keys, sort=False):
        source, side, joint, name, cycle_id = identity
        channel = raw[
            (raw.source == source)
            & (raw.side == side)
            & (raw.joint == joint)
            & (raw.q_name == name)
        ].sort_values("time_s")
        if channel.empty:
            raise ValueError(
                f"Raw samples unavailable for 101-point export: {source}/{side}/{name}. Re-run normalization."
            )
        cycle = intervals[(side, cycle_id)]
        values, status = normalize_cycle(
            channel.time_s.to_numpy(),
            channel.value.to_numpy(),
            cycle,
            n_points=101,
            max_gap_s=float(report.get("maximum_interpolation_gap_s", 0.05)),
            angular=str(group.iloc[0]["unit"]) == "deg",
        )
        template = group.iloc[0].to_dict()
        template.update(normalization_status=status)
        for point, (phase, time, value) in enumerate(
            zip(
                np.linspace(0, 100, 101),
                np.linspace(cycle["start_s"], cycle["end_s"], 101),
                values,
            ),
            1,
        ):
            rows.append(
                {
                    **template,
                    "point_index": point,
                    "phase_percent": phase,
                    "time_s": time,
                    "value": value,
                }
            )
        match = metrics
        for key, value in zip(keys, identity):
            match = match[match[key] == value]
        summary = match.iloc[0].to_dict() if not match.empty else {**template}
        summary.update(
            mean_value=float(np.mean(values)) if status == "ok" else np.nan,
            rom_value=float(np.ptp(values)) if status == "ok" else np.nan,
            normalization_status=status,
        )
        summaries.append(summary)
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def copy_gait_exports(trial_dir, destination, *, prefix=None, included_cycle_ids=None):
    """Materialize CSVs only on explicit export, using the reviewed cycle subset."""
    import json
    import numpy as np
    from gui_graphs import read_table_npz
    from gait_cycles import RAW_FIELDS, CYCLE_FIELDS, METRIC_FIELDS

    trial_dir, destination = Path(trial_dir), Path(destination)
    report, excluded, _ = cycle_selection(trial_dir)
    if not report:
        raise FileNotFoundError("Normaliser cet essai avant l'export.")
    options = cycle_options(report)
    allowed = {c["selection_key"] for c in options}
    if included_cycle_ids is None:
        selected = selected_cycle_keys(report, excluded)
    else:
        requested = set(included_cycle_ids)
        valid_ids = {c["cycle_id"] for c in options} | allowed
        if not requested <= valid_ids:
            raise ValueError("Unknown selected cycle.")
        selected = {
            c["selection_key"]
            for c in options
            if c["selection_key"] in requested or c["cycle_id"] in requested
        }
    if not selected:
        raise ValueError("Sélectionner au moins un cycle détecté pour l'export.")
    prefix = prefix if prefix is not None else trial_dir.name
    if (
        not prefix.strip()
        or prefix in {".", ".."}
        or any(c in prefix for c in "/\\\x00")
    ):
        raise ValueError("Le nom doit être un préfixe de fichier, sans chemin.")
    raw = read_table_npz(trial_dir / "lower_limb_kinematics.npz")
    normalized = read_table_npz(trial_dir / "gait_cycle_timeseries.npz")
    metrics = read_table_npz(trial_dir / "gait_cycle_metrics.npz")
    if normalized.empty or metrics.empty:
        raise ValueError("Aucune courbe normalisée disponible.")
    normalized = normalized[cycle_row_keys(normalized).isin(selected)]
    metrics = metrics[cycle_row_keys(metrics).isin(selected)]
    if normalized.empty:
        raise ValueError("Aucune courbe pour les cycles sélectionnés.")
    cycles = [c for c in options if c["selection_key"] in selected]
    normalized, metrics = normalized_export_points(raw, normalized, metrics, report)
    # Raw exports also exclude unselected intervals for each side.
    if not raw.empty:
        keep = np.zeros(len(raw), dtype=bool)
        for cycle in cycles:
            side = (
                np.ones(len(raw), dtype=bool)
                if cycle["side"] == "both"
                else raw.side.eq(cycle["side"])
            )
            keep |= side & raw.time_s.between(cycle["start_s"], cycle["end_s"])
        raw = raw[keep]
    profiles = profiles_for_trial(trial_dir)
    raw = correct_native_sign_table(raw, profiles=profiles)
    normalized = correct_native_sign_table(normalized, profiles=profiles)
    metrics = correct_native_sign_table(metrics, profiles=profiles)
    tables = [
        ("lower_limb_kinematics", raw, RAW_FIELDS),
        ("lower_limb_gait_cycles", normalized, CYCLE_FIELDS),
        ("gait_cycle_metrics", metrics, METRIC_FIELDS),
    ]
    for joint in ("hip", "knee", "ankle"):
        tables += [
            (
                joint + "_kinematics",
                raw[raw.joint == joint] if not raw.empty else raw,
                RAW_FIELDS,
            ),
            (
                joint + "_gait_cycles",
                normalized[normalized.joint == joint],
                CYCLE_FIELDS,
            ),
        ]
    destination.mkdir(parents=True, exist_ok=True)
    exported = []
    for name, table, fields in tables:
        if "phase_percent" in fields:
            fields = tuple(fields) + ("point_index",)
        if report.get("task", "gait") != "gait":
            name = name.replace("_gait_cycles", "_cycles").replace(
                "gait_cycle_metrics", "cycle_metrics"
            )
        target = destination / f"{prefix}_{name}.csv"
        table.reindex(
            columns=tuple(fields)
            + tuple(f for f in SIGN_FIELDS if f in table.columns and f not in fields)
        ).to_csv(target, index=False)
        exported.append(target)
    report = {
        **report,
        "cycles": cycles,
        "included_cycle_ids": sorted({c["cycle_id"] for c in cycles}),
        "included_cycle_keys": sorted(selected),
        "excluded_cycle_keys": sorted(allowed - selected),
        "excluded_cycle_ids": sorted(
            {c["cycle_id"] for c in options} - {c["cycle_id"] for c in cycles}
        ),
        "export_prefix": prefix,
        "cached_points_per_cycle": report.get("points_per_cycle"),
        "points_per_cycle": 101,
        "point_index_range": [1, 101],
        "phase_percent_range": [0, 100],
        "sign_convention": {
            "reference": SIGN_REFERENCE,
            "profiles": profiles,
            "operation": "value = raw_value * sign_multiplier; ROM unchanged",
            "ankle_transverse": "unchanged_pending_mapping",
            "scope": "native coordinates only; not anatomical harmonization",
        },
        "valid_curve_cycles": int((metrics.normalization_status == "ok").sum()),
        "unavailable_curve_cycles": int((metrics.normalization_status != "ok").sum()),
    }
    target = destination / f"{prefix}_gait_cycles.json"
    target.write_text(json.dumps(report, indent=2))
    exported.append(target)
    return exported
