"""Small presentation helpers for trial-level run reports in the GUI."""

from __future__ import annotations

from typing import Any, Mapping


def _nested(mapping: Mapping[str, Any], *keys: str, default: object = "") -> object:
    value: object = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return default
        value = value.get(key, default)
    return value


def _model_line(report: Mapping[str, Any], model_key: str, label: str) -> str | None:
    model = _nested(report, "models", model_key, default={})
    if not isinstance(model, Mapping) or not model:
        return None
    source_kind = str(model.get("source_kind", "?"))
    policy = _nested(model, "root_offset_policy", "selected_policy", default="?")
    score = _nested(model, "root_offset_policy", "score_mm", default=None)
    line = f"{label}: {source_kind}, root offset {policy}"
    if isinstance(score, (float, int)):
        line += f" ({score:.2f} mm)"
    return line


def summarize_run_report(report: Mapping[str, Any]) -> str:
    """Return a compact human-readable summary of automatic analysis choices."""

    if not report:
        return "Aucun rapport sélectionné."
    lines: list[str] = []
    trial = report.get("trial")
    if trial:
        lines.append(f"Essai: {trial}")
    gait = report.get("gait_normalization", {})
    if gait and gait.get("status") not in {"disabled", "static_trial"}:
        event_label = (
            "contacts CSV"
            if gait.get("method") == "supplied_initial_contacts"
            else (
                "Ghoussayni (vitesse sagittale talon/avant-pied)"
                if gait.get("method", "").startswith("ghoussayni")
                else "contacts estimés par marqueurs"
            )
        )
        lines.append(
            f"Cycles candidats complets 0–100 %: {len(gait.get('cycles', []))}; {event_label}"
        )
        if gait.get("unavailable_curve_cycles"):
            lines.append(
                f"Courbes/cycles indisponibles (couverture ou lacunes): {gait['unavailable_curve_cycles']}"
            )
    if report.get("workflow") in {"source_comparison", "motive_biobuddy"}:
        labels = {"captury": "Captury", "motive": "Motive", "biobuddy": "BioBuddy"}
        available = report.get("available_sources", [])
        lines.append(
            "Sources utilisées: "
            + (", ".join(labels.get(s, s) for s in available) or "aucune")
        )
        for source, selection in report.get("source_selection", {}).items():
            if selection.get("status") == "unavailable":
                reasons = {
                    "missing_recording": "enregistrement absent",
                    "missing_selected_skeleton_export": "export du squelette absent",
                    "missing_biobuddy_model": "modèle non fourni",
                    "missing_motive_marker_recording_for_biobuddy_ik": "marqueurs Motive requis pour l'IK",
                    "missing_source_in_static_calibration": "source absente de la calibration statique",
                    "missing_compatible_kinematics": "IK compatible indisponible pour ce modèle/essai",
                }
                reason = selection.get("reason", "")
                lines.append(
                    f"{labels.get(source, source)} indisponible: {reasons.get(reason, reason)}"
                )
        lines.append(
            "Courbes q natives; écarts des origines/repères segmentaires diagnostiques."
        )
    axis = report.get("axis_conversion")
    if axis:
        if isinstance(axis, Mapping):
            descriptions: list[str] = []
            for system in ("captury", "motive"):
                details = axis.get(system)
                if not isinstance(details, Mapping):
                    continue
                own = str(details.get("own_c3d_vertical_axis", "?"))
                common = str(details.get("common_comparison_vertical_axis", "+Z"))
                descriptions.append(f"{system.title()} C3D {own} -> commun {common}")
            if descriptions:
                lines.append("Axes: " + "; ".join(descriptions))
            else:
                lines.append(f"Axes: {axis}")
        else:
            lines.append(f"Axe modèle -> C3D: {axis}")
    for model_key, label in (("captury", "Captury"), ("motive", "Motive")):
        model_line = _model_line(report, model_key, label)
        if model_line:
            lines.append(model_line)
    alignment_status = _nested(report, "alignment", "status", default="")
    marker_method = _nested(
        report,
        "alignment",
        "motive_model_to_c3d_markers",
        "method",
        default="",
    )
    if alignment_status or marker_method:
        suffix = f", marqueurs Motive: {marker_method}" if marker_method else ""
        lines.append(f"Recalage: {alignment_status}{suffix}")
    segment_status = _nested(report, "segment_rotations", "status", default="")
    requested = _nested(report, "segment_rotations", "requested_reference", default="")
    effective = _nested(report, "segment_rotations", "effective_reference", default="")
    if requested or effective or segment_status:
        if requested and effective:
            line = f"Référence segments: {requested} -> {effective}"
        else:
            line = f"Référence segments: {requested or effective}"
        if segment_status:
            line += f" ({segment_status})"
        lines.append(line)
    corrections = _nested(
        report, "segment_orientation_corrections", "applied", default=[]
    )
    if corrections:
        lines.append(f"Corrections segments: {', '.join(map(str, corrections))}")
    synchronization = report.get("temporal_synchronization", {})
    if isinstance(synchronization, Mapping) and synchronization:
        status = synchronization.get("status", "?")
        lag_s = synchronization.get("lag_s", 0.0)
        estimated_lag_s = synchronization.get("estimated_lag_s")
        line = f"Synchronisation: {status}, lag appliqué {float(lag_s):+.6f} s"
        if isinstance(estimated_lag_s, (float, int)) and estimated_lag_s != lag_s:
            line += f", estimé {float(estimated_lag_s):+.6f} s"
        correlation = synchronization.get("correlation_after")
        prominence = synchronization.get("peak_prominence")
        if isinstance(correlation, (float, int)):
            line += f", r={float(correlation):.3f}"
        if isinstance(prominence, (float, int)):
            line += f", proéminence={float(prominence):.3f}"
        lines.append(line)
    metric_quality = report.get("metric_quality", {})
    if isinstance(metric_quality, Mapping) and metric_quality:
        total = metric_quality.get("waveforms")
        eligible = metric_quality.get("eligible_waveforms")
        if isinstance(total, int) and isinstance(eligible, int):
            lines.append(f"Forme cinématique: {eligible}/{total} courbes éligibles")
        status_counts = metric_quality.get("status_counts", {})
        if isinstance(status_counts, Mapping):
            low_reference = status_counts.get("low_reference_amplitude", 0)
            low_test = status_counts.get("low_test_amplitude", 0)
            insufficient_pairs = status_counts.get("insufficient_pairs", 0)
            insufficient_coverage = status_counts.get("insufficient_coverage", 0)
            unknown_native = status_counts.get("unknown_native_scale", 0)
            details: list[str] = []
            if isinstance(low_reference, int) and low_reference:
                details.append(f"faible amplitude référence: {low_reference}")
            if isinstance(low_test, int) and low_test:
                details.append(f"faible amplitude test: {low_test}")
            if isinstance(insufficient_pairs, int) and insufficient_pairs:
                details.append(f"paires insuffisantes: {insufficient_pairs}")
            if isinstance(insufficient_coverage, int) and insufficient_coverage:
                details.append(f"couverture insuffisante: {insufficient_coverage}")
            if isinstance(unknown_native, int) and unknown_native:
                details.append(f"seuil natif inconnu: {unknown_native}")
            if details:
                lines.append("Garde-fous: " + ", ".join(details))
    sensitivity = report.get("metric_sensitivity", {})
    if isinstance(sensitivity, Mapping) and sensitivity:
        root = sensitivity.get("root_translation", {})
        if isinstance(root, Mapping):
            for system in ("captury", "motive"):
                system_root = root.get(system, {})
                if not isinstance(system_root, Mapping):
                    continue
                score = system_root.get("score_difference_mm")
                if system_root.get("status") == "computed" and isinstance(
                    score, (float, int)
                ):
                    lines.append(
                        "Sensibilité offset translation racine "
                        f"{system.capitalize()}: {float(score):.2f} mm"
                    )
        temporal = sensitivity.get("temporal_lag", {})
        if isinstance(temporal, Mapping) and temporal.get("status") == "computed":
            improvement = temporal.get("normalized_rmse_improvement")
            if isinstance(improvement, (float, int)):
                lines.append(
                    "Sensibilité synchronisation: gain RMSE normalisée "
                    f"{float(improvement):+.4f}"
                )
    return (
        "\n".join(lines) if lines else "Rapport disponible, aucun choix critique listé."
    )
