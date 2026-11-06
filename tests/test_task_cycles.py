import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from task_cycles import (
    excursion_cycles,
    squat_cycles,
    jump_cycles,
    reviewed_task_cycles,
)
from gait_cycles import export_gait_results
from gui_gait import copy_gait_exports, cycle_selection, save_cycle_selection
from compare_p6_motive_captury import write_table_npz
from gui_graphs import read_table_npz


class TaskCycleTests(unittest.TestCase):
    def test_squat_geometry_and_complete_boundaries(self):
        t = np.arange(101) / 50
        angle = 60 * np.sin(np.pi * t / 2) ** 2
        centres = {}
        for side in ("Left", "Right"):
            centres[side + "UpLeg"] = np.tile([[0], [1], [0]], (1, len(t)))
            centres[side + "Leg"] = np.zeros((3, len(t)))
            a = np.radians(angle)
            centres[side + "Foot"] = np.array([np.sin(a), -np.cos(a), np.zeros(len(t))])
        run = SimpleNamespace(system="motive", time=t, centres_native=centres)
        report = squat_cycles([run], min_duration_s=0.4, max_duration_s=5)
        self.assertEqual(len(report["cycles"]), 1)
        self.assertAlmostEqual(report["cycles"][0]["peak_s"], 1)
        self.assertEqual(report["cycles"][0]["side"], "both")
        centres["RightLeg"][:, 40:60] = np.nan
        self.assertEqual(
            squat_cycles([run], min_duration_s=0.4, max_duration_s=5)["cycles"], []
        )

    def test_squat_supports_fbx_segment_names(self):
        from kinematic_conventions import (
            load_kinematic_conventions,
            segment_source_names,
        )

        time = np.arange(301) / 100
        angle = np.radians(60 * np.sin(np.pi * time / 3) ** 2)
        names = segment_source_names(
            load_kinematic_conventions(), "motive_model", "fbx"
        )
        centres = {}
        for side in ("left", "right"):
            centres[names[f"{side}_thigh"]] = np.array(
                [np.zeros_like(time), np.ones_like(time), np.zeros_like(time)]
            )
            centres[names[f"{side}_shank"]] = np.zeros((3, len(time)))
            centres[names[f"{side}_foot"]] = np.array(
                [np.sin(angle), -np.cos(angle), np.zeros_like(time)]
            )
        run = SimpleNamespace(
            system="motive", source_kind="fbx", centres_native=centres, time=time
        )
        report = squat_cycles([run], min_duration_s=0.4, max_duration_s=5)
        self.assertEqual(len(report["cycles"]), 1)
        del centres[names["left_shank"]]
        with self.assertRaisesRegex(ValueError, "left hip, knee and ankle"):
            squat_cycles([run])

    def test_jump_requires_all_four_markers_and_ground_boundaries(self):
        t = np.arange(101) / 100
        p = np.zeros((3, 4, len(t)))
        p[2, :, 30:70] = 100
        report = jump_cycles(["LFCC", "LFM2", "RFCC", "RFM2"], p, t)
        self.assertEqual(len(report["cycles"]), 1)
        self.assertAlmostEqual(report["cycles"][0]["start_s"], 0.29)
        self.assertAlmostEqual(report["cycles"][0]["end_s"], 0.70)
        p[2, 0, :] = 0
        self.assertEqual(
            jump_cycles(["LFCC", "LFM2", "RFCC", "RFM2"], p, t)["cycles"], []
        )
        with self.assertRaises(ValueError):
            jump_cycles(["LFCC", "LFM2", "RFCC", "OTHER"], p, t)

    def test_jump_export_respects_c3d_residual_validity(self):
        from unittest.mock import patch
        from compare_p6_motive_captury import write_trial_gait_exports

        times = np.arange(101) / 100
        points = np.zeros((3, 4, len(times)))
        points[2, :, 30:70] = 100
        residuals = np.zeros(points.shape[1:])
        bundle = SimpleNamespace(
            name="Jump", participant="Synthetic", motive_c3d=Path("jump.c3d")
        )
        args = SimpleNamespace(
            cycle_task="jump",
            cycle_detection="automatic",
            normalize_gait=True,
            static_trial="Static",
            gait_min_cycle_s=0.1,
            gait_max_cycle_s=2.5,
            gait_max_gap_s=0.05,
            jump_height_mm=30,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch(
                "compare_p6_motive_captury.read_c3d_points_mm",
                return_value=(
                    ["LFCC", "LFM2", "RFCC", "RFM2"],
                    points,
                    residuals,
                    times,
                ),
            ):
                valid, _ = write_trial_gait_exports(
                    bundle, Path(tmp), args, [], [], None, None
                )
                self.assertEqual(len(valid["cycles"]), 1)
                residuals[0, 29] = (
                    -1
                )  # finite coordinates, but invalid take-off boundary
                invalid, _ = write_trial_gait_exports(
                    bundle, Path(tmp), args, [], [], None, None
                )
                self.assertEqual(invalid["cycles"], [])
                self.assertTrue(invalid["rejected_cycles"])

    def test_partial_excursions_and_timestamp_gaps_are_rejected(self):
        report = excursion_cycles(
            np.arange(6) * 0.1,
            [50, 50, 0, 50, 50, 50],
            task="squat",
            threshold=15,
            max_gap_s=0.2,
        )
        self.assertEqual(report["cycles"], [])
        report = excursion_cycles(
            [0, 0.1, 0.9, 1], [0, 50, 50, 0], task="squat", threshold=15
        )
        self.assertEqual(report["rejected_cycles"][0]["reason"], "missing_event_data")

    def test_reviewed_intervals_reject_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text("trial,start_s,end_s\nSquat,0,2\nSquat,1,3\n")
            with self.assertRaises(ValueError):
                reviewed_task_cycles(
                    path, "Squat", "squat", min_duration_s=0.4, max_duration_s=10
                )

    def test_no_automatic_csv_export_and_selected_named_export(self):
        t = np.arange(401) / 100
        run = SimpleNamespace(
            system="motive",
            source_kind="bvh",
            q_names=["LeftLeg_Xrotation", "RightLeg_Xrotation"],
            q_units=["rad", "rad"],
            time=t,
            q=np.array([t, t]),
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
            for i, start in enumerate([0, 2], 1)
        ]
        report = dict(
            cycles=cycles, rejected_cycles=[], method="test", status="diagnostic_only"
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "Squat"
            directory.mkdir()
            export_gait_results(
                directory,
                "Squat",
                [run],
                [],
                enabled=True,
                task="squat",
                cycle_report=report,
                write_npz=write_table_npz,
            )
            self.assertEqual(list(directory.glob("*.csv")), [])
            normalized = read_table_npz(directory / "gait_cycle_timeseries.npz")
            self.assertEqual(set(normalized.side), {"left", "right"})
            _, _, signature = cycle_selection(directory)
            save_cycle_selection(directory, ["squat_1"], signature)
            files = copy_gait_exports(
                directory, Path(tmp) / "export", prefix="My_session"
            )
            self.assertEqual(len(files), 10)
            data = pd.read_csv(Path(tmp) / "export/My_session_lower_limb_cycles.csv")
            self.assertEqual(set(data.cycle_id), {"squat_2"})
            raw = pd.read_csv(Path(tmp) / "export/My_session_lower_limb_kinematics.csv")
            self.assertTrue(raw.time_s.between(2, 3).all())
            save_cycle_selection(directory, ["squat_1", "right::squat_2"], signature)
            copy_gait_exports(directory, Path(tmp) / "left_export", prefix="LeftOnly")
            left_only = pd.read_csv(
                Path(tmp) / "left_export/LeftOnly_lower_limb_cycles.csv"
            )
            self.assertEqual(set(left_only.side), {"left"})
            self.assertEqual(set(left_only.cycle_id), {"squat_2"})
            raw_left = pd.read_csv(
                Path(tmp) / "left_export/LeftOnly_lower_limb_kinematics.csv"
            )
            self.assertEqual(set(raw_left.side), {"left"})
            with self.assertRaises(ValueError):
                copy_gait_exports(directory, Path(tmp) / "export", prefix="../bad")
            save_cycle_selection(directory, ["squat_1", "squat_2"], signature)
            with self.assertRaises(ValueError):
                copy_gait_exports(directory, Path(tmp) / "export")
            # Recomputed event boundaries invalidate the old review selection.
            path = directory / "gait_cycles.json"
            payload = json.loads(path.read_text())
            payload["cycles"][0]["end_s"] = 0.9
            path.write_text(json.dumps(payload))
            self.assertEqual(cycle_selection(directory)[1], [])


if __name__ == "__main__":
    unittest.main()
