"""One comparison entry point with source availability resolved for each trial.

The paired capture adapter retains its established spatial/time calibration.
The available-source adapter emits independent native results when a pair is
absent. Adapter choice is internal; selection, calibration lifetime and reports
share this workflow.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from available_source_results import compare_available_trial

SOURCES = ("captury", "motive", "biobuddy")


def requested_sources(args):
    return tuple(getattr(args, "sources", SOURCES))


def source_selection(bundle, args):
    requested = requested_sources(args)
    result = {}
    for source in SOURCES:
        reason = None
        if source not in requested:
            result[source] = {"status": "not_selected"}
            continue
        if source == "biobuddy":
            model = getattr(args, "biobuddy_biomod", None)
            if model is None or not Path(model).is_file():
                reason = "missing_biobuddy_model"
            elif not bundle.motive_c3d:
                reason = "missing_motive_marker_recording_for_biobuddy_ik"
        elif not getattr(bundle, f"{source}_c3d", None):
            reason = "missing_recording"
        else:
            kinds = (
                ("bvh", "fbx") if args.model_source == "auto" else (args.model_source,)
            )
            if not any(getattr(bundle, f"{source}_{kind}", None) for kind in kinds):
                reason = "missing_selected_skeleton_export"
        result[source] = (
            {"status": "unavailable", "reason": reason}
            if reason
            else {"status": "available"}
        )
    return result


class ComparisonWorkflow:
    def __init__(self, static_bundle, args, *, api):
        self.static_bundle = static_bundle
        self.args = args
        self.api = api
        self.calibrations = {}
        self.static_reports = []

    def process(self, bundle, out_root):
        selection = source_selection(bundle, self.args)
        static_selection = source_selection(self.static_bundle, self.args)
        for source in ("captury", "motive"):
            if (
                selection[source]["status"] == "available"
                and static_selection[source]["status"] != "available"
            ):
                selection[source] = {
                    "status": "unavailable",
                    "reason": "missing_source_in_static_calibration",
                }
        active = tuple(s for s in SOURCES if selection[s]["status"] == "available")
        args = copy.copy(self.args)
        args.sources = list(active)
        if "biobuddy" not in active:
            args.biobuddy_biomod = None
            args.run_ik_batch = False
        paired = "captury" in active and "motive" in active
        adapter = self.api.compare_trial if paired else compare_available_trial
        key = tuple(s for s in active if s != "biobuddy")

        def run(trial, root, calibration=None):
            if paired:
                return adapter(
                    trial, root, args, static_alignment_transform=calibration
                )
            return adapter(trial, root, args, api=self.api, calibration=calibration)

        if not active:
            report, _ = compare_available_trial(bundle, out_root, args, api=self.api)
            report["status"] = "unavailable_no_selected_source"
        else:
            if key not in self.calibrations and bundle.name != self.static_bundle.name:
                calibration_root = (
                    self.args.out_dir
                    / "_static_alignment"
                    / ("_".join(key) or "biobuddy")
                )
                static_report, transform = run(self.static_bundle, calibration_root)
                self.calibrations[key] = transform
                static_report["recording_trial"] = self.static_bundle.name
                static_report["trial"] = (
                    f"calibration:{'_'.join(key) or 'biobuddy'}:{self.static_bundle.name}"
                )
                static_report["workflow"] = "source_comparison"
                static_report["requested_sources"] = list(active)
                self.static_reports.append(static_report)
                static_path = (
                    calibration_root
                    / self.api.safe_name(self.static_bundle.name)
                    / "run_report.json"
                )
                static_path.write_text(json.dumps(static_report, indent=2))
            report, transform = run(bundle, out_root, self.calibrations.get(key))
            if bundle.name == self.static_bundle.name:
                self.calibrations[key] = transform
            available = report.get("available_sources")
            if available is None:
                available = [s for s in active if s != "biobuddy"]
                if report.get("biobuddy_ik_batch", {}).get("status") in (
                    "ok",
                    "cached",
                ):
                    available.append("biobuddy")
            report["available_sources"] = available
            for source in active:
                if source not in available:
                    selection[source] = {
                        "status": "unavailable",
                        "reason": "missing_compatible_kinematics",
                        "details": (
                            report.get("biobuddy_ik_batch", {})
                            if source == "biobuddy"
                            else {}
                        ),
                    }
        report.update(
            workflow="source_comparison",
            requested_sources=list(requested_sources(self.args)),
            source_selection=selection,
        )
        directory = out_root / self.api.safe_name(bundle.name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "run_report.json").write_text(json.dumps(report, indent=2))
        print(
            f"{bundle.name}: "
            + ", ".join(f"{s}={v['status']}" for s, v in selection.items()),
            flush=True,
        )
        return report
