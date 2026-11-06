import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from compare_p6_motive_captury import write_table_npz
from captury_biobuddy_gui import CapturyBioBuddyGui
from gait_cycles import export_gait_results
from gui_commands import build_p6_auto_analysis_args
from gui_gait import copy_gait_exports, draw_gait_cycles
from gui_graphs import read_table_npz, values_for_display
from tests.test_gait_cycles import contact_rows
from isb_compliance_report import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
    load_curve_from_manifest,
)


class GuiGaitTests(unittest.TestCase):
    def test_missing_trial_npz_clears_previous_gait_plot(self):
        figure = Figure()
        axes = figure.subplots()
        axes.plot([0, 100], [1, 2])
        with tempfile.TemporaryDirectory() as tmp:
            gui = SimpleNamespace(
                graph_panels={
                    "gait": {"figure": figure, "axes": axes, "canvas": Mock()}
                },
                _graph_output_root=lambda: Path(tmp),
                _graph_csv_path=lambda _: Path(tmp) / "summary.csv",
                _read_csv_or_empty=lambda _: pd.DataFrame(
                    {"trial": ["Walk"], "mean_value": [1.0]}
                ),
            )
            CapturyBioBuddyGui._draw_metric_graph(
                gui, "gait", [{"filters": {"trial": "Walk"}, "metric": "mean_value"}]
            )
        self.assertEqual(len(axes.lines), 0)
        self.assertIn("absentes", axes.get_title())

    def test_export_all_dofs_csv_npz_plot_and_copy(self):
        time = np.arange(0.0, 4.01, 0.01)
        runs = [
            SimpleNamespace(
                system=source,
                source_kind=kind,
                q_names=names,
                q_units=["rad"] * len(names),
                time=time,
                q=np.tile(time, (len(names), 1)),
                unit_scale_to_m=1.0,
            )
            for source, kind, names in (
                (
                    "motive",
                    "bvh",
                    [
                        "LeftUpLeg_Xrotation",
                        "RightUpLeg_Yrotation",
                        "LeftLeg_Zrotation",
                        "LeftFoot_Yrotation",
                    ],
                ),
                (
                    "biobuddy",
                    "motive_57_ik",
                    [
                        "LThigh_RotX",
                        "RThigh_RotY",
                        "LShank_RotZ",
                        "LFoot_RotZ",
                        "LFoot_RotX",
                    ],
                ),
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "Walk"
            directory.mkdir()
            report, outputs = export_gait_results(
                directory,
                "Walk",
                runs,
                contact_rows(),
                enabled=True,
                write_npz=write_table_npz,
                export_csv=True,
            )
            self.assertEqual(len(report["cycles"]), 6)
            raw = pd.read_csv(outputs["lower_limb_kinematics"])
            self.assertEqual(raw.q_name.nunique(), 9)
            normalized = pd.read_csv(outputs["lower_limb_gait_cycles"])
            npz = read_table_npz(outputs["gait_cycle_timeseries"])
            self.assertEqual(len(normalized), len(npz))
            self.assertEqual(set(normalized.source), {"motive", "biobuddy"})
            manifest = build_reproducibility_manifest(
                directory, [{"trial": "Walk", "outputs": outputs}]
            )
            manifest_path = write_reproducibility_manifest(directory, manifest)
            recipe = next(
                r
                for r in manifest["curve_recipes"]
                if r["artifact_id"] == "Walk/gait_cycle_timeseries"
                and r["y_key"] == "value"
            )
            self.assertIn("cycle_id", recipe["filter_keys"])
            curve = load_curve_from_manifest(
                manifest_path,
                recipe["curve_id"],
                filters=recipe["filter_combinations"][0],
            )
            self.assertEqual(len(curve["x"]), 101)
            self.assertEqual(curve["y_unit"], "deg")
            for _, curve in normalized.groupby(["source", "q_name", "cycle_id"]):
                self.assertEqual(len(curve), 101)
                self.assertEqual(curve.phase_percent.iloc[0], 0.0)
                self.assertEqual(curve.phase_percent.iloc[-1], 100.0)
                np.testing.assert_allclose(
                    curve.value, np.degrees(curve.time_s), atol=1e-10
                )
            for joint in ("hip", "knee", "ankle"):
                data = pd.read_csv(outputs[joint + "_gait_cycles"])
                self.assertEqual(set(data.joint), {joint})
            axes = Figure().subplots()
            draw_gait_cycles(
                axes,
                npz,
                {"trial": "Walk", "source": "motive", "side": "left", "joint": "hip"},
            )
            self.assertEqual(len(axes.lines), 3)
            self.assertEqual(axes.get_xlim(), (0.0, 100.0))
            self.assertEqual(axes.get_ylabel(), "deg; signe natif à vérifier")
            copied = copy_gait_exports(directory, Path(tmp) / "export")
            self.assertEqual(len(copied), 10)
            self.assertTrue((Path(tmp) / "export/Walk_hip_gait_cycles.csv").is_file())
            # Disabling normalization must clear old normalized files, not show stale cycles.
            export_gait_results(
                directory,
                "Walk",
                runs,
                contact_rows(),
                enabled=False,
                write_npz=write_table_npz,
                export_csv=True,
            )
            self.assertTrue(pd.read_csv(outputs["hip_gait_cycles"]).empty)
            self.assertEqual(
                json.loads(Path(outputs["gait_cycles"]).read_text())["status"],
                "disabled",
            )

    def test_auto_analysis_command_includes_cycle_options(self):
        args = build_p6_auto_analysis_args(
            {
                "p6_normalize_gait": True,
                "p6_gait_events_csv": "contacts.csv",
                "p6_gait_min_cycle_s": "0.4",
                "p6_gait_max_cycle_s": "2.5",
                "p6_gait_max_gap_s": "0.05",
                "p6_gait_velocity_threshold_m_s": "0.5",
                "p6_gait_filter_hz": "7.0",
            },
            "Walk",
        )
        self.assertIn("--normalize-gait", args)
        self.assertEqual(args[args.index("--gait-events-csv") + 1], "contacts.csv")
        self.assertIn("--gait-max-gap-s", args)
        self.assertEqual(args[args.index("--gait-velocity-threshold-m-s") + 1], "0.5")
        self.assertEqual(args[args.index("--gait-filter-hz") + 1], "7.0")

    def test_toe_off_is_drawn_at_exported_phase(self):
        axes = Figure().subplots()
        frame = pd.DataFrame(
            {
                "trial": ["Walk"] * 3,
                "source": ["motive"] * 3,
                "side": ["left"] * 3,
                "joint": ["knee"] * 3,
                "q_name": ["LeftLeg_Xrotation"] * 3,
                "cycle_id": ["left_001"] * 3,
                "normalization_status": ["ok"] * 3,
                "phase_percent": [0.0, 50.0, 100.0],
                "value": [5.0, 10.0, 5.0],
                "unit": ["deg"] * 3,
                "toe_off_percent": [62.0] * 3,
            }
        )
        draw_gait_cycles(axes, frame, {})
        self.assertEqual(len(axes.lines), 2)
        np.testing.assert_allclose(axes.lines[1].get_xdata(), [62.0, 62.0])

    def test_motive_native_rotation_names_are_displayed_in_degrees(self):
        np.testing.assert_allclose(
            values_for_display(
                np.array([np.pi]), "motive", q_name="LeftUpLeg_Xrotation"
            ),
            [180.0],
        )


if __name__ == "__main__":
    unittest.main()
