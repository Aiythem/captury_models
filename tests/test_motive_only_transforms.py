"""Synthetic end-to-end contracts for the Motive/BioBuddy result path."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import compare_p6_motive_captury as comparison
from gui_graphs import read_table_npz
from gui_trial_viewer import load_joint_centre_chain_data
from available_source_results import compare_available_trial as compare_motive_trial


class MotiveOnlyTransformTests(unittest.TestCase):
    def make_runs(self, root, shift):
        time = np.array([0.0, 1.0, 2.0])
        native_cm = np.array([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0], [0.0, 1.0, 0.0]])
        native_cm[0] += shift
        thorax_cm = native_cm + np.array([[0.0], [20.0], [0.0]])
        # Explicit expected result: Y-up -> Z-up, then +90 degree lab yaw,
        # then translation [100, 200, 300] mm, with cm -> mm conversion.
        expected = np.vstack(
            (10 * native_cm[2] + 100, 10 * native_cm[0] + 200, 10 * native_cm[1] + 300)
        )
        expected_thorax = expected + np.array([[0.0], [0.0], [200.0]])
        offset_mm = np.array([[5.0], [-7.0], [11.0]])
        rotations = np.repeat(np.eye(3)[:, :, None], len(time), axis=2)

        def run(system, names, centres, scale):
            return comparison.ModelRun(
                system=system,
                source_kind="bvh" if system == "motive" else "motive_57_ik",
                biomod_path=root / f"{system}.bioMod",
                q=np.array([[0.1, 0.2, 0.3]]),
                q_names=["Pelvis_rotX"],
                q_units=["rad"],
                time=time,
                joint_names=names,
                centres_native=centres,
                rotations_native={name: rotations.copy() for name in names},
                unit_scale_to_m=scale,
                mesh_report={},
                root_offset_policy={
                    "selected_mode": "subtract_static_offset_from_root_q"
                },
            )

        motive = run(
            "motive", ["Hips", "Spine3"], {"Hips": native_cm, "Spine3": thorax_cm}, 0.01
        )
        biobuddy = run(
            "biobuddy",
            ["Pelvis", "Thorax"],
            {
                "Pelvis": (expected + offset_mm) / 1000,
                "Thorax": (expected_thorax + offset_mm) / 1000,
            },
            1.0,
        )
        return motive, biobuddy, expected, expected_thorax, offset_mm

    def test_static_transform_is_frozen_and_biobuddy_stays_in_c3d_frame(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = comparison.parse_args()
            args.out_dir = root
            args.static_trial = "P16_Static"
            args.cut_mode = "full"
            static_runs = self.make_runs(root, shift=0)
            dynamic_runs = self.make_runs(root, shift=20)
            api = SimpleNamespace(**vars(comparison))
            api.build_model_run_with_rotation_audit = Mock(
                side_effect=[
                    (static_runs[0], {}),
                    (dynamic_runs[0], {}),
                ]
            )
            # Row-vector +90 degree yaw. A second fit must never happen.
            api.model_to_motive_marker_alignment = Mock(
                side_effect=[
                    (
                        np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                        np.array([100.0, 200.0, 300.0]),
                        {"status": "ok"},
                    )
                ]
            )
            api.run_biobuddy_ik_for_trial = Mock(
                side_effect=[
                    (static_runs[1], {"status": "ok"}),
                    (dynamic_runs[1], {"status": "ok"}),
                ]
            )
            api.detect_trial_events_and_contacts = Mock(return_value=({}, []))
            api.analyze_motive_occlusions = Mock(return_value=([], None))
            api.biobuddy_dimension_rows = Mock(return_value=([], {"status": "ok"}))

            def bundle(name):
                return comparison.TrialBundle(
                    name,
                    None,
                    None,
                    None,
                    root / f"{name}.c3d",
                    root / f"{name}.bvh",
                    None,
                    "P16",
                )

            static_report, calibration = compare_motive_trial(
                bundle("P16_Static"),
                root,
                args,
                api=api,
            )
            # A changed GUI preference must not replace the policy fixed on Static.
            args.motive_root_offset_mode = "keep"
            dynamic_report, reused = compare_motive_trial(
                bundle("P16_Marche_001"),
                root,
                args,
                api=api,
                calibration=calibration,
            )
            api.model_to_motive_marker_alignment.assert_called_once()
            self.assertIs(reused, calibration)
            self.assertEqual(
                api.build_model_run_with_rotation_audit.call_args.args[4], "subtract"
            )
            self.assertEqual(static_report["alignment"], dynamic_report["alignment"])
            self.assertTrue(
                all(
                    call.kwargs["cache_only"]
                    for call in api.run_biobuddy_ik_for_trial.call_args_list
                )
            )

            trial_dir = root / "P16_Marche_001"
            centres = read_table_npz(trial_dir / "joint_centre_timeseries.npz")
            for name, expected in (
                ("pelvis", dynamic_runs[2]),
                ("thorax", dynamic_runs[3]),
            ):
                rows = centres[centres.joint == name]
                np.testing.assert_allclose(
                    rows[[f"motive_{a}_mm" for a in "xyz"]], expected.T
                )
                np.testing.assert_allclose(
                    rows[[f"biobuddy_{a}_mm" for a in "xyz"]],
                    (expected + dynamic_runs[4]).T,
                )
                np.testing.assert_allclose(
                    rows.biobuddy_distance_mm, np.sqrt(5**2 + 7**2 + 11**2)
                )
            self.assertFalse(any("captury" in column for column in centres.columns))
            # Identity BioBuddy frames versus the composed cyclic Motive basis:
            # trace=0 gives a 120 degree SO(3) deviation, not 90 or 180 degrees.
            segment_curves = read_table_npz(
                trial_dir / "segment_rotation_timeseries.npz"
            )
            np.testing.assert_allclose(segment_curves.global_deg, 120.0, atol=1e-10)
            self.assertEqual(set(segment_curves.source), {"motive"})
            self.assertEqual(set(segment_curves.reference), {"biobuddy"})

            chain = load_joint_centre_chain_data(
                trial_dir / "joint_centre_timeseries.npz"
            )
            self.assertEqual(set(chain.layers), {"motive", "biobuddy"})
            self.assertEqual(chain.edges, [("pelvis", "thorax")])
            np.testing.assert_allclose(
                chain.layers["biobuddy"]["pelvis"],
                (dynamic_runs[2] + dynamic_runs[4]).T,
            )
            q = read_table_npz(trial_dir / "kinematics_q_timeseries.npz")
            self.assertEqual(set(q.source), {"motive", "biobuddy"})
            self.assertNotIn("difference", q.columns)
            saved = json.loads((trial_dir / "motive_static_alignment.json").read_text())
            self.assertEqual(saved, calibration)


if __name__ == "__main__":
    unittest.main()
