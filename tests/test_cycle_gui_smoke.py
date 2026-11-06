"""Opt-in Tk integration check using generated cycles: CAPTURY_GUI_SMOKE=1."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd


@unittest.skipUnless(os.environ.get("CAPTURY_GUI_SMOKE") == "1", "requires Tk desktop")
class CycleGuiSmokeTests(unittest.TestCase):
    def test_checkbox_mean_figures_and_named_101_point_export(self):
        from captury_biobuddy_gui import CapturyBioBuddyGui
        from compare_p6_motive_captury import write_table_npz
        from gait_cycles import export_gait_results
        from gui_gait import cycle_selection
        from gui_graphs import read_table_npz

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / "Trial"
            directory.mkdir()
            time = np.arange(301) / 100
            names = [
                side + joint + "_Xrotation"
                for side in ("Left", "Right")
                for joint in ("UpLeg", "Leg", "Foot")
            ]
            run = SimpleNamespace(
                system="motive",
                source_kind="bvh",
                q_names=names,
                q_units=["rad"] * len(names),
                time=time,
                q=np.tile(np.sin(time), (len(names), 1)),
                unit_scale_to_m=0.01,
            )
            cycles = [
                dict(
                    cycle_id=f"squat_{i}",
                    cycle_index=i,
                    side="both",
                    start_s=start,
                    end_s=start + 1,
                    duration_s=1,
                )
                for i, start in enumerate((0, 2), 1)
            ]
            export_gait_results(
                directory,
                "Trial",
                [run],
                [],
                enabled=True,
                task="squat",
                cycle_report=dict(
                    cycles=cycles,
                    rejected_cycles=[],
                    method="synthetic",
                    status="diagnostic_only",
                ),
                write_npz=write_table_npz,
            )
            metrics = read_table_npz(directory / "gait_cycle_metrics.npz")
            write_table_npz(
                root / "all_gait_cycle_metrics.npz", metrics.to_dict("records")
            )

            def restore(app):
                app.vars["p6_auto_analyze"].set(False)
                app.vars["p6_data_root"].set(str(root))
                app.vars["p6_out_dir"].set(str(root))
                app.vars["p6_cycle_task"].set("squat")
                app.vars["selected_trial"].set("Trial")

            with patch.object(
                CapturyBioBuddyGui, "_restore_gui_session_state", restore
            ), patch.object(CapturyBioBuddyGui, "_refresh_trial_inventory"):
                app = CapturyBioBuddyGui()
            try:
                app.update()
                self.assertEqual(len(app.cycle_check_vars), 4)
                app.cycle_side_frames["right"].winfo_children()[0].invoke()
                app.update()
                self.assertIn("right::squat_1", cycle_selection(directory)[1])
                self.assertNotIn("left::squat_1", cycle_selection(directory)[1])
                windows = app._open_cycle_mean_window()
                app.update()
                self.assertEqual(set(windows), {"hip", "knee", "ankle"})
                for window in windows.values():
                    self.assertEqual(len(window.figure.axes), 3)
                    self.assertTrue(
                        any(
                            "n=1" in line.get_label()
                            for axis in window.figure.axes
                            for line in axis.lines
                        )
                    )
                destination = root / "export"
                destination.mkdir()
                with patch(
                    "tkinter.simpledialog.askstring", return_value="selected"
                ), patch(
                    "captury_biobuddy_gui.filedialog.askdirectory",
                    return_value=str(destination),
                ):
                    app._export_gait_csvs()
                exported = pd.read_csv(destination / "selected_lower_limb_cycles.csv")
                self.assertFalse(
                    (
                        (exported.side == "right") & (exported.cycle_id == "squat_1")
                    ).any()
                )
                for _, group in exported.groupby(
                    ["source", "side", "joint", "q_name", "cycle_id"]
                ):
                    np.testing.assert_array_equal(group.point_index, np.arange(1, 102))
                    np.testing.assert_array_equal(group.phase_percent, np.arange(101))
                self.assertIn("Export terminé", app.status_var.get())
            finally:
                app.destroy()
