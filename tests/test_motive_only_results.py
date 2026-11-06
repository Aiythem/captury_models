import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import compare_p6_motive_captury as comparison
from captury_biobuddy_gui import CapturyBioBuddyGui, inventory_p6_dataset
from gui_graphs import read_table_npz, values_for_display
from gui_trial_viewer import load_joint_centre_chain_data
from isb_compliance_report import build_reproducibility_manifest


class MotiveOnlyResultsTests(unittest.TestCase):
    def test_motive_only_p16_is_discovered_without_fabricating_captury(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Motive").mkdir()
            for name in ("P16_Marche_001.c3d", "P16_Marche_001_Skeleton 001.bvh"):
                (root / "Motive" / name).touch()
            trials = comparison.discover_trials(root)
            inventory = inventory_p6_dataset(root)
        self.assertEqual([trial.name for trial in trials], ["P16_Marche_001"])
        self.assertIsNone(trials[0].captury_c3d)
        self.assertEqual(set(inventory["P16_Marche_001"]), {"Motive"})

    def test_native_q_curves_keep_sources_separate_even_when_names_match(self):
        from available_source_results import native_q_rows

        runs = [
            SimpleNamespace(
                system=source,
                q_names=["Pelvis_rotX"],
                q_units=["rad"],
                time=np.array([0.0, 0.01]),
                q=np.array([[value, value + 0.2]]),
            )
            for source, value in (("motive", 0.1), ("biobuddy", 0.3))
        ]
        summary, curves = native_q_rows("P16_Marche_001", runs)
        self.assertEqual(len(summary), 2)
        self.assertEqual({row["source"] for row in summary}, {"motive", "biobuddy"})
        self.assertTrue(all("difference" not in row for row in curves))
        self.assertTrue(all("bias_rad" not in row for row in summary))
        self.assertAlmostEqual(summary[0]["rom_rad"], 0.2)
        np.testing.assert_allclose(
            values_for_display(np.array([np.pi]), "rom_rad"), [180.0]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kinematics_q_timeseries.npz"
            comparison.write_table_npz(path, curves)
            restored = read_table_npz(path)
        self.assertEqual(set(restored.source), {"motive", "biobuddy"})
        self.assertNotIn("captury", restored.columns)

    def test_motive_biobuddy_skeleton_and_calibration_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "joint_centre_timeseries.npz"
            rows = [
                {
                    "joint": joint,
                    "time": 0.0,
                    **{
                        f"{source}_{axis}_mm": float(i)
                        for source in ("motive", "biobuddy")
                        for i, axis in enumerate("xyz")
                    },
                }
                for joint in ("pelvis", "thorax")
            ]
            comparison.write_table_npz(path, rows)
            chain = load_joint_centre_chain_data(path)
            self.assertEqual(set(chain.layers), {"motive", "biobuddy"})
            self.assertEqual(chain.edges, [("pelvis", "thorax")])
            alignment = root / "motive_static_alignment.json"
            alignment.write_text("{}")
            manifest = build_reproducibility_manifest(
                root,
                [
                    {
                        "trial": "P16_Static",
                        "outputs": {
                            "joint_centre_timeseries": str(path),
                            "motive_static_alignment": str(alignment),
                        },
                    }
                ],
            )
            self.assertEqual(
                manifest["transformations"]["P16_Static"]["spatial"],
                "P16_Static/motive_static_alignment",
            )

    def test_gui_resolves_p16_static_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "P16_Static.c3d"
            path.touch()
            gui = SimpleNamespace(
                vars={"p6_static_trial": SimpleNamespace(get=lambda: "Static")},
                trial_inventory={"P16_Static": {"Motive": {"c3d": path}}},
            )
            self.assertEqual(CapturyBioBuddyGui._static_motive_c3d_path(gui), path)

    def test_failed_static_alignment_is_not_published_as_comparison(self):
        from available_source_results import (
            compare_available_trial as compare_motive_trial,
        )

        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            args = comparison.parse_args()
            bundle = SimpleNamespace(name="Static", motive_c3d=Path("static.c3d"))
            run = SimpleNamespace(
                centres_native={}, unit_scale_to_m=0.01, time=np.array([0.0, 0.01])
            )
            with patch.object(
                comparison,
                "build_model_run_with_rotation_audit",
                return_value=(run, {}),
            ), patch.object(
                comparison,
                "model_to_motive_marker_alignment",
                return_value=(
                    np.eye(3),
                    np.zeros(3),
                    {"status": "insufficient_points"},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Cannot align Motive static"):
                    compare_motive_trial(bundle, Path(tmp), args, api=comparison)


if __name__ == "__main__":
    unittest.main()
