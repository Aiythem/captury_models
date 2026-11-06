import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from native_angle_signs import (
    positive_axis,
    channel_rule,
    correct_native_sign_table,
    reference_basis,
    SIGN_REFERENCE,
)


class NativeSignTests(unittest.TestCase):
    def test_positive_axes_on_both_sides(self):
        # X anterior, Y proximal, Z subject-right; positive knee bends posteriorly.
        basis = np.eye(3)
        for side, mirror in [("right", 1), ("left", -1)]:
            for joint in ["hip", "knee", "ankle"]:
                np.testing.assert_equal(
                    positive_axis(joint, side, "sagittal", basis),
                    [0, 0, -1 if joint == "knee" else 1],
                )
                np.testing.assert_equal(
                    positive_axis(joint, side, "frontal", basis), [mirror, 0, 0]
                )
                if joint != "ankle":
                    np.testing.assert_equal(
                        positive_axis(joint, side, "transverse", basis), [0, mirror, 0]
                    )
            self.assertIsNone(positive_axis("ankle", side, "transverse", basis))

    def test_model_axis_polarity_and_oblique_unknown(self):
        r = channel_rule("knee", "right", "sagittal", np.eye(3), np.array([0, 0, 1]))
        self.assertEqual(r["sign_multiplier"], -1)
        self.assertEqual(r["positive_direction"], "flexion")
        self.assertEqual(
            channel_rule("hip", "left", "frontal", np.eye(3), np.array([1, 0, 0]))[
                "sign_multiplier"
            ],
            -1,
        )
        r = channel_rule(
            "hip", "right", "sagittal", np.eye(3), np.array([1, 0, 1]) / np.sqrt(2)
        )
        self.assertEqual(r["sign_status"], "pending_oblique_axis")
        self.assertEqual(r["sign_multiplier"], 1)
        self.assertEqual(
            channel_rule("ankle", "left", "transverse", np.eye(3), np.array([0, 1, 0]))[
                "sign_status"
            ],
            "pending_ankle_transverse_mapping",
        )

    def test_reference_basis_is_right_handed_and_rotation_invariant(self):
        p = {
            "right_thigh": np.array([0, 1, 1]),
            "left_thigh": np.array([0, 1, -1]),
            "right_foot": np.array([0, 0, 1]),
            "left_foot": np.array([0, 0, -1]),
        }
        B = reference_basis(p)
        np.testing.assert_allclose(B, np.eye(3))
        from scipy.spatial.transform import Rotation

        R = Rotation.from_rotvec([0.4, -0.2, 0.3]).as_matrix()
        np.testing.assert_allclose(reference_basis({k: R @ v for k, v in p.items()}), R)
        self.assertAlmostEqual(np.linalg.det(B), 1)

    def rows(self):
        return pd.DataFrame(
            [
                dict(
                    source=s,
                    side="right",
                    joint="knee",
                    axis="Z",
                    q_name="RShank_RotZ",
                    coordinate_kind="rotation",
                    unit="deg",
                    value=v,
                    mean_value=v,
                    rom_value=50.0,
                )
                for s in ["motive", "biobuddy", "captury"]
                for v in [-40.0, 10.0, np.nan]
            ]
        )

    def profiles(self):
        return {
            s: {
                "model_sha256": "sha-" + s,
                "channels": {
                    "RShank_RotZ": dict(
                        sign_multiplier=-1,
                        sign_status="model_reference_polarity",
                        positive_direction="flexion",
                        plane="sagittal",
                        axis_projection=-1.0,
                    )
                },
            }
            for s in ["motive", "biobuddy", "captury"]
        }

    def test_sign_only_idempotent_all_sources_and_exports_metadata(self):
        df = self.rows()
        before = df.copy(deep=True)
        corrected = correct_native_sign_table(df, profiles=self.profiles())
        pd.testing.assert_frame_equal(df, before)
        np.testing.assert_allclose(corrected.value, -df.value, equal_nan=True)
        np.testing.assert_allclose(corrected.mean_value, -df.mean_value, equal_nan=True)
        np.testing.assert_allclose(corrected.raw_value, df.value, equal_nan=True)
        np.testing.assert_allclose(corrected.rom_value, df.rom_value)
        pd.testing.assert_frame_equal(
            correct_native_sign_table(corrected, profiles=self.profiles()), corrected
        )
        self.assertEqual(set(corrected.sign_reference), {SIGN_REFERENCE["doi"]})

    def test_missing_source_and_translation_are_unchanged(self):
        df = self.rows()
        df.loc[0, "coordinate_kind"] = "translation"
        df.loc[0, "unit"] = "mm"
        profiles = self.profiles()
        del profiles["captury"]
        out = correct_native_sign_table(df, profiles=profiles)
        self.assertEqual(out.iloc[0].value, -40)
        np.testing.assert_allclose(
            out[out.source == "captury"].value,
            df[df.source == "captury"].value,
            equal_nan=True,
        )
        self.assertEqual(
            set(out[out.source == "captury"].sign_status), {"pending_model_evidence"}
        )

    def test_export_point_indices_restart_and_missing_gaps_remain_missing(self):
        from gui_gait import normalized_export_points

        raw = pd.DataFrame(
            [
                dict(
                    source="motive",
                    side="right",
                    joint="knee",
                    q_name="RightLeg_Xrotation",
                    unit="deg",
                    time_s=t,
                    value=v,
                )
                for t, v in [(0, 0), (0.5, 10), (1, 20), (2, 30), (3, 40)]
            ]
        )
        cycles = [
            dict(side="right", cycle_id="first", start_s=0, end_s=1),
            dict(side="right", cycle_id="second", start_s=2, end_s=3),
        ]
        normalized = pd.DataFrame(
            [
                {
                    **raw.iloc[0].to_dict(),
                    "cycle_id": c["cycle_id"],
                    "phase_percent": 0,
                    "normalization_status": "ok",
                }
                for c in cycles
            ]
        )
        metrics = normalized.assign(mean_value=0.0, rom_value=0.0)
        for task in ["gait", "squat", "jump"]:
            output, summary = normalized_export_points(
                raw,
                normalized,
                metrics,
                dict(task=task, cycles=cycles, maximum_interpolation_gap_s=0.6),
            )
            for _, group in output.groupby("cycle_id"):
                np.testing.assert_equal(group.point_index, np.arange(1, 102))
                np.testing.assert_equal(group.phase_percent, np.arange(101))
            first = output[output.cycle_id == "first"]
            np.testing.assert_allclose(first.value, np.linspace(0, 20, 101))
            second = output[output.cycle_id == "second"]
            self.assertTrue(second.value.isna().all())
            self.assertEqual(set(second.normalization_status), {"gap_exceeds_limit"})
            self.assertTrue(
                summary[summary.cycle_id == "second"].mean_value.isna().all()
            )

    def test_actual_model_perturbation_aliases_and_stale_evidence(self):
        from native_angle_signs import profiles_for_trial, _model_profile
        import hashlib

        model_text = (
            "version 4\nsegment Pelvis\ntranslations xyz\nrotations zxy\nendsegment\n"
        )
        for prefix, z in [("R", 0.1), ("L", -0.1)]:
            for segment, parent, offset in [
                ("Thigh", "Pelvis", (0, 1, z)),
                ("Shank", prefix + "Thigh", (0, -0.4, 0)),
                ("Foot", prefix + "Shank", (0, -0.4, 0)),
            ]:
                model_text += f"segment {prefix}{segment}\nparent {parent}\nRT 0 0 0 xyz {offset[0]} {offset[1]} {offset[2]}\nrotations zxy\nendsegment\n"
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model = directory / "model.bioMod"
            model.write_text(model_text)
            sha = hashlib.sha256(model.read_bytes()).hexdigest()
            report = {
                "models": {"biobuddy": {"path": str(model)}},
                "biobuddy_ik_batch": {
                    "cache": {"payload": {"biomod": {"sha256": sha}}}
                },
            }
            report_path = directory / "run_report.json"
            report_path.write_text(json.dumps(report))
            profile = profiles_for_trial(directory)["biobuddy"]
            self.assertEqual(profile["model_sha256"], sha)
            self.assertEqual(profile["channels"]["RShank_RotZ"]["sign_multiplier"], -1)
            self.assertEqual(profile["channels"]["LThigh_RotX"]["sign_multiplier"], -1)
            np.testing.assert_allclose(
                profile["channels"]["RThigh_RotZ"]["native_axis_at_q0"],
                [0, 0, 1],
                atol=1e-8,
            )
            df = self.rows()
            df = df[df.source == "biobuddy"].copy()
            df["q_name"] = "RShank_Zrotation"
            out = correct_native_sign_table(df, directory)
            np.testing.assert_allclose(out.value, -df.value, equal_nan=True)
            self.assertEqual(model.read_text(), model_text)
            report["q_reexpression"] = {"enabled": True}
            report_path.write_text(json.dumps(report))
            self.assertEqual(profiles_for_trial(directory), {})
            del report["q_reexpression"]
            report_path.write_text(json.dumps(report))
            model.write_text(model_text + "\n// modified model\n")
            self.assertEqual(profiles_for_trial(directory), {})

    def test_csv_individual_plot_and_selected_mean_share_fixed_signs(self):
        from gui_gait import copy_gait_exports, draw_gait_cycles
        from gui_cycle_means import selected_leg_means
        from compare_p6_motive_captury import write_table_npz

        cycle = dict(
            side="right",
            cycle_id="right_001",
            cycle_index=1,
            start_s=0.0,
            end_s=1.0,
            duration_s=1.0,
        )
        report = dict(
            task="gait",
            cycles=[cycle],
            points_per_cycle=3,
            maximum_interpolation_gap_s=0.6,
        )
        raw = self.rows().dropna().iloc[:0].copy()
        raw = pd.DataFrame(
            [
                dict(
                    source=source,
                    side="right",
                    joint="knee",
                    axis="Z",
                    q_name="RShank_RotZ",
                    coordinate_kind="rotation",
                    unit="deg",
                    coordinate_semantics="native_model_dof",
                    trial="Walk",
                    time_s=t,
                    value=v,
                )
                for source in ["motive", "biobuddy", "captury"]
                for t, v in [(0.0, -5.0), (0.5, -40.0), (1.0, -10.0)]
            ]
        )
        normalized = raw.assign(
            cycle_id="right_001",
            phase_percent=raw.time_s * 100,
            normalization_status="ok",
        )
        metrics = (
            raw.groupby("source", as_index=False)
            .first()
            .drop(columns=["value", "time_s"])
            .assign(
                cycle_id="right_001",
                mean_value=-55 / 3,
                rom_value=35.0,
                normalization_status="ok",
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "Walk"
            directory.mkdir()
            (directory / "gait_cycles.json").write_text(json.dumps(report))
            for name, table in [
                ("lower_limb_kinematics", raw),
                ("gait_cycle_timeseries", normalized),
                ("gait_cycle_metrics", metrics),
            ]:
                write_table_npz(directory / (name + ".npz"), table.to_dict("records"))
            with patch("gui_gait.profiles_for_trial", return_value=self.profiles()):
                files = copy_gait_exports(directory, Path(tmp) / "csv", prefix="named")
            exported = pd.read_csv(Path(tmp) / "csv/named_lower_limb_gait_cycles.csv")
            np.testing.assert_allclose(exported.value, -exported.raw_value)
            exported_metrics = pd.read_csv(
                Path(tmp) / "csv/named_gait_cycle_metrics.csv"
            )
            np.testing.assert_allclose(
                exported_metrics.mean_value, -exported_metrics.raw_mean_value
            )
            np.testing.assert_allclose(exported_metrics.rom_value, 35)
            curves = selected_leg_means(
                correct_native_sign_table(raw, profiles=self.profiles()), report
            )
            self.assertEqual(
                {c["source"] for c in curves}, {"captury", "motive", "biobuddy"}
            )
            for curve in curves:
                export_curve = exported[exported.source == curve["source"]]
                self.assertEqual(len(export_curve), 101)
                np.testing.assert_equal(export_curve.point_index, np.arange(1, 102))
                np.testing.assert_equal(export_curve.phase_percent, np.arange(101))
                expected = export_curve.value.to_numpy()[[0, 50, 100]]
                np.testing.assert_allclose(curve["mean"], expected)
                axes = Figure().subplots()
                draw_gait_cycles(
                    axes,
                    correct_native_sign_table(normalized, profiles=self.profiles()),
                    {"source": curve["source"]},
                )
                np.testing.assert_allclose(axes.lines[0].get_ydata(), expected)
            metadata = json.loads(
                (Path(tmp) / "csv/named_gait_cycles.json").read_text()
            )
            self.assertEqual(
                metadata["sign_convention"]["reference"]["doi"], "10.7717/peerj.4640"
            )

    def test_plot_retains_negative_peaks_without_task_dependent_flip(self):
        from gui_cycle_means import draw_selected_leg_means

        curve = dict(
            source="biobuddy",
            joint="knee",
            axis="Z",
            q_name="RShank_RotZ",
            side="right",
            phase_percent=np.array([0, 50, 100]),
            mean=np.array([-5.0, -50.0, -10.0]),
            n_cycles=1,
            sign_multiplier=1,
            sign_status="model_reference_polarity",
            positive_direction="flexion",
            plane="sagittal",
        )
        for task in ["squat", "gait", "jump"]:
            axes = draw_selected_leg_means(
                Figure(), Path("/tmp"), {"task": task}, joint="knee", curves=[curve]
            )
            np.testing.assert_equal(axes[0].lines[0].get_ydata(), curve["mean"])
            self.assertIn("flexion", axes[0].get_ylabel())


if __name__ == "__main__":
    unittest.main()
