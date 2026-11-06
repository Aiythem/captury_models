import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from matplotlib.figure import Figure
from gui_cycle_means import (
    selected_leg_means,
    draw_selected_leg_means,
    draw_joint_figures,
    positive_squat_peak,
    PLANE_AXES,
    JOINT_LABELS,
)
from compare_p6_motive_captury import write_table_npz
from gui_graphs import GRAPH_CONFIGS


def fixture():
    cycles = [
        dict(
            side="both",
            cycle_id=f"squat_{i}",
            cycle_index=i,
            start_s=start,
            end_s=start + 1,
            duration_s=1,
        )
        for i, start in [(1, 0), (2, 2)]
    ]
    report = dict(
        task="squat",
        cycles=cycles,
        points_per_cycle=11,
        maximum_interpolation_gap_s=0.11,
    )
    rows = []
    for source in ("motive", "biobuddy"):
        for joint in ("hip", "knee", "ankle"):
            for side in ("right", "left"):
                for start, value in [(0, 10), (2, 30)]:
                    for t in np.linspace(start, start + 1, 11):
                        rows.append(
                            dict(
                                source=source,
                                joint=joint,
                                side=side,
                                q_name=joint + "_RotX",
                                axis="X",
                                unit="deg",
                                time_s=t,
                                value=value if side == "right" else 100,
                            )
                        )
    return pd.DataFrame(rows), report


class CycleMeanTests(unittest.TestCase):
    def test_equal_cycle_weights_and_independent_right_selection(self):
        raw, report = fixture()
        curves = selected_leg_means(raw, report, ["left::squat_1"])
        self.assertEqual(len(curves), 6)
        for curve in curves:
            np.testing.assert_allclose(curve["mean"], 20)
            self.assertEqual(curve["n_cycles"], 2)
        for curve in selected_leg_means(raw, report, ["right::squat_1"]):
            np.testing.assert_allclose(curve["mean"], 30)
            self.assertEqual(curve["cycle_ids"], ["squat_2"])
        self.assertEqual(
            selected_leg_means(raw, report, ["right::squat_1", "right::squat_2"]), []
        )

    def test_squat_knee_preserves_original_rightleg_channels(self):
        time = np.linspace(0, 2, 101)
        angle = 126 * np.sin(np.pi * time / 2) ** 2
        euler = Rotation.from_euler("X", angle[:, None], degrees=True).as_euler(
            "ZXY", degrees=True
        )
        rows = [
            dict(
                source="motive",
                joint="knee",
                side="right",
                q_name="RightLeg_" + axis + "rotation",
                axis=axis,
                unit="deg",
                time_s=t,
                value=v,
            )
            for i, axis in enumerate("ZXY")
            for t, v in zip(time, euler[:, i])
        ]
        report = dict(
            task="squat",
            cycles=[
                dict(
                    side="both",
                    cycle_id="squat_1",
                    cycle_index=1,
                    start_s=0,
                    end_s=2,
                    duration_s=2,
                )
            ],
            points_per_cycle=101,
        )
        curves = selected_leg_means(pd.DataFrame(rows), report)
        self.assertEqual(
            {c["q_name"] for c in curves}, {"RightLeg_" + a + "rotation" for a in "ZXY"}
        )
        for c in curves:
            self.assertEqual(c["definition"], "native_coordinate")
            original = euler[:, "ZXY".index(c["axis"])]
            expected = np.degrees(np.unwrap(np.radians(original)))
            np.testing.assert_allclose(c["mean"], expected, atol=1e-10)

    def test_non_hinge_rotation_is_not_mislabeled_as_flexion(self):
        rows = [
            dict(
                source="motive",
                joint="knee",
                side="right",
                q_name="RightLeg_" + axis + "rotation",
                axis=axis,
                unit="deg",
                time_s=t,
                value=10,
            )
            for axis in "ZXY"
            for t in np.linspace(0, 1, 101)
        ]
        report = dict(
            cycles=[dict(side="right", cycle_id="a", start_s=0, end_s=1, duration_s=1)]
        )
        curves = selected_leg_means(pd.DataFrame(rows), report)
        self.assertEqual(len(curves), 3)
        self.assertTrue(all(c["definition"] == "native_coordinate" for c in curves))

    def test_complete_cycles_only_and_angular_branch_alignment(self):
        raw, report = fixture()
        raw.loc[(raw.time_s >= 2), "value"] += 360
        curves = selected_leg_means(raw, report)
        np.testing.assert_allclose(curves[0]["mean"], 20)
        raw.loc[(raw.time_s > 0.3) & (raw.time_s < 0.7), "value"] = np.nan
        curves = selected_leg_means(raw, report)
        self.assertTrue(all(c["n_cycles"] == 1 for c in curves))

    def test_plot_contains_both_sources_and_no_rom_choice(self):
        raw, report = fixture()
        raw = pd.concat(
            [raw.assign(axis=a, q_name=raw.q_name + a) for a in "XYZ"],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_table_npz(
                Path(tmp) / "lower_limb_kinematics.npz", raw.to_dict("records")
            )
            figure = Figure(figsize=(12, 9))
            axes = draw_selected_leg_means(figure, tmp, report)
            self.assertEqual(len(axes), 3)
            self.assertTrue(all(len(ax.lines) == 2 for ax in axes))
            labels = [line.get_label() for line in axes[0].lines]
            self.assertTrue(any("Motive" in label for label in labels))
            self.assertTrue(any("BioBuddy" in label for label in labels))
        self.assertNotIn("rom_value", GRAPH_CONFIGS["gait"]["metrics"])

    def test_positive_peak_reverses_whole_curve_only_for_squat(self):
        source = np.array([-90.0, 10.0, -20.0])
        shown, sign = positive_squat_peak(source, "squat")
        self.assertEqual(sign, -1)
        np.testing.assert_array_equal(shown, [90.0, -10.0, 20.0])
        np.testing.assert_array_equal(source, [-90.0, 10.0, -20.0])
        np.testing.assert_array_equal(positive_squat_peak(source, "gait")[0], source)

    def test_three_joint_figures_each_with_three_horizontal_planes(self):
        raw, report = fixture()
        # Provide every native axis for both sources.
        raw = pd.concat(
            [raw.assign(axis=a, q_name=raw.q_name + a) for a in "XYZ"],
            ignore_index=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_table_npz(
                Path(tmp) / "lower_limb_kinematics.npz", raw.to_dict("records")
            )
            figures = {joint: Figure(figsize=(13.5, 5)) for joint in JOINT_LABELS}
            axes = draw_joint_figures(figures, tmp, report)
            self.assertEqual(len({id(f) for f in figures.values()}), 3)
            for joint, label in JOINT_LABELS.items():
                self.assertEqual(len(axes[joint]), 3)
                self.assertTrue(all(len(ax.lines) == 2 for ax in axes[joint]))
                self.assertIn(label, figures[joint]._suptitle.get_text())
                self.assertEqual(
                    [ax.get_title() for ax in axes[joint]], list(PLANE_AXES)
                )
                self.assertTrue(
                    all(
                        ax.get_subplotspec().get_gridspec().get_geometry() == (1, 3)
                        for ax in axes[joint]
                    )
                )
                for ax in axes[joint]:
                    self.assertTrue(
                        all(max(line.get_ydata()) >= 0 for line in ax.lines)
                    )


if __name__ == "__main__":
    unittest.main()
