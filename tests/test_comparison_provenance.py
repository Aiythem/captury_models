from __future__ import annotations

import unittest
import argparse
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from gui_graphs import read_table_npz

import compare_p6_motive_captury as comparison


class ComparisonProvenanceTests(unittest.TestCase):
    def test_file_sha256_hashes_sidecar_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.json"
            path.write_bytes(b"phase-4-evidence")

            digest = comparison.file_sha256(path)

        self.assertEqual(digest, hashlib.sha256(b"phase-4-evidence").hexdigest())

    def test_cache_fingerprint_tracks_scientific_implementation_files(self) -> None:
        bundle = comparison.TrialBundle(
            "Static",
            Path("c.c3d"),
            Path("c.bvh"),
            None,
            Path("m.c3d"),
            Path("m.bvh"),
            None,
        )
        args = argparse.Namespace(
            model_source="bvh",
            model_to_c3d_axis="auto",
            captury_unit_scale_to_m=None,
            motive_unit_scale_to_m=None,
            biobuddy_biomod=None,
            biobuddy_unit_scale_to_m=1.0,
            root_offset_mode="auto",
            angle_label_regex="angle",
            c3d_angle_unit="deg",
            landmark_map=None,
            segment_reference="biobuddy",
            captury_reorient_thigh_y_from_cor=False,
            rotate_body_segments_180_x=False,
            reexpress_rotations_zxy=False,
            disable_static_model_alignment=False,
            disable_motive_marker_alignment=False,
            joint_filter=[],
            no_mesh=True,
            max_mesh_points=0,
            run_ik_batch=False,
            ik_max_frames=0,
            cut_mode="full",
            time_start=None,
            time_end=None,
            no_figures=True,
            audit_bvh_fbx_rotations=True,
            bvh_fbx_max_p95_geodesic_deg=5.0,
        )
        versions = {"kinematic_rotations.py": 1}

        def fingerprint(path):
            if path is None:
                return None
            path = Path(path)
            return {"path": path.name, "version": versions.get(path.name, 0)}

        with patch.object(comparison, "file_fingerprint", side_effect=fingerprint):
            first = comparison.trial_cache_fingerprint(bundle, args)
            versions["kinematic_rotations.py"] = 2
            second = comparison.trial_cache_fingerprint(bundle, args)

        implementation = first["payload"]["implementation"]
        self.assertEqual(
            set(implementation),
            {
                "comparison_batch",
                "available_source_results",
                "comparison_workflow",
                "gait_cycles",
                "task_cycles",
                "ghoussayni",
                "kinematic_conventions_registry",
                "kinematic_conventions_code",
                "kinematic_rotations_code",
                "joint_kinematics_registry",
                "joint_kinematics_code",
                "isb_segment_audit_code",
                "isb_compliance_report_code",
                "spatial_calibration_code",
                "mocap_alignment_code",
                "captury_c3d_angle_decoder",
                "captury_c3d_angle_registry",
                "captury_frame_calibration_code",
                "captury_q_landmarks",
                "mocap_coordinate_frames_code",
                "biobuddy_ik_code",
                "temporal_synchronization_code",
                "comparison_metrics_code",
            },
        )
        self.assertNotEqual(first["digest"], second["digest"])
        args.gait_velocity_threshold_m_s = 0.25
        with patch.object(comparison, "file_fingerprint", side_effect=fingerprint):
            threshold_changed = comparison.trial_cache_fingerprint(bundle, args)
        self.assertNotEqual(second["digest"], threshold_changed["digest"])
        args.gait_filter_hz = 6.0
        with patch.object(comparison, "file_fingerprint", side_effect=fingerprint):
            filter_changed = comparison.trial_cache_fingerprint(bundle, args)
        self.assertNotEqual(threshold_changed["digest"], filter_changed["digest"])

    def test_biobuddy_ik_outputs_must_exist_before_trial_cache_is_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            npz = root / "ik.npz"
            summary = root / "ik.json"
            npz.write_bytes(b"npz")
            summary.write_text("{}", encoding="utf-8")
            report = {
                "biobuddy_ik_batch": {
                    "outputs": {"npz": str(npz), "summary": str(summary)}
                }
            }

            self.assertTrue(comparison.biobuddy_ik_outputs_complete(report))
            npz.unlink()
            self.assertFalse(comparison.biobuddy_ik_outputs_complete(report))

        self.assertFalse(comparison.biobuddy_ik_outputs_complete({}))

    def test_rotation_audit_npz_roundtrip_including_empty_comparison(self) -> None:
        comparison_data = {
            "timeseries": {
                "pelvis": {
                    "time": np.asarray([0.0, 0.5]),
                    "rotation_vector_rad": np.asarray(
                        [[0.1, 0.2], [0.0, 0.0], [0.0, 0.0]]
                    ),
                    "geodesic_deg": np.rad2deg(np.asarray([0.1, 0.2])),
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            paths = comparison.write_bvh_fbx_rotation_audit(
                Path(tmp), "captury", {"status": "blocked"}, comparison_data
            )
            dataframe = read_table_npz(paths["timeseries"])
            empty_paths = comparison.write_bvh_fbx_rotation_audit(
                Path(tmp), "motive", {"status": "blocked"}, {}
            )
            empty = read_table_npz(empty_paths["timeseries"])

        self.assertEqual(list(dataframe["segment"]), ["pelvis", "pelvis"])
        np.testing.assert_allclose(dataframe["time_s"], [0.0, 0.5])
        np.testing.assert_allclose(dataframe["deviation_x_deg"], np.rad2deg([0.1, 0.2]))
        np.testing.assert_allclose(dataframe["geodesic_deg"], np.rad2deg([0.1, 0.2]))
        self.assertTrue(empty.empty)

    def test_biobuddy_ik_batch_does_not_add_historical_motive_bvh_ik(self) -> None:
        bundle = comparison.TrialBundle(
            "Static",
            Path("captury.c3d"),
            Path("captury.bvh"),
            Path("captury.fbx"),
            Path("motive.c3d"),
            Path("motive.bvh"),
            Path("motive.fbx"),
        )
        args = argparse.Namespace(
            model_source="fbx",
            biobuddy_biomod=None,
            landmark_map=None,
            occlusions_only=False,
            run_ik_batch=True,
        )

        inputs = comparison.comparison_input_files([bundle], args)

        self.assertNotIn("Static/motive/ik_bvh", inputs)
        self.assertEqual(
            inputs["implementation/biobuddy_ik_code"],
            comparison.SCIENTIFIC_IMPLEMENTATION_FILES["biobuddy_ik_code"],
        )

    def test_rotation_audit_tracks_bvh_and_fbx_for_both_systems(self) -> None:
        bundle = comparison.TrialBundle(
            "Static",
            Path("captury.c3d"),
            Path("captury.bvh"),
            Path("captury.fbx"),
            Path("motive.c3d"),
            Path("motive.bvh"),
            Path("motive.fbx"),
        )
        args = argparse.Namespace(
            model_source="bvh",
            audit_bvh_fbx_rotations=True,
            biobuddy_biomod=None,
            landmark_map=None,
            occlusions_only=False,
            run_ik_batch=False,
        )

        inputs = comparison.comparison_input_files([bundle], args)

        self.assertEqual(inputs["Static/captury/bvh"], Path("captury.bvh"))
        self.assertEqual(inputs["Static/captury/fbx"], Path("captury.fbx"))
        self.assertEqual(inputs["Static/motive/bvh"], Path("motive.bvh"))
        self.assertEqual(inputs["Static/motive/fbx"], Path("motive.fbx"))

    def test_reports_expose_generated_artifacts_used_downstream(self) -> None:
        reports = [
            {
                "trial": "Static",
                "models": {
                    "captury": {"biomod": "/tmp/captury.bioMod"},
                    "motive": {"biomod": "/tmp/motive.bioMod"},
                },
                "outputs": {
                    "skin_marker_correspondence_proposal": "/tmp/proposal.json",
                    "spatial_calibration": "/tmp/spatial_calibration.json",
                    "joint_kinematics_d4_d6": "/tmp/joint_d4_d6.json",
                    "joint_kinematics_d4_d6_timeseries": "/tmp/joint_d4_d6.npz",
                    "captury_c3d_angle_decode": "/tmp/captury_angles.json",
                    "captury_c3d_angle_metrics": "/tmp/captury_angles.csv",
                    "captury_c3d_angle_timeseries": "/tmp/captury_angles.npz",
                },
                "biobuddy_ik_batch": {
                    "outputs": {
                        "npz": "/tmp/biobuddy_ik.npz",
                        "summary": "/tmp/biobuddy_ik.json",
                    }
                },
                "skin_marker_correspondence": {"map_source": "automatic_proposal"},
                "bvh_fbx_rotation_audit": {
                    "captury": {
                        "artifacts": {
                            "summary": "/tmp/captury_audit.json",
                            "timeseries": "/tmp/captury_audit.npz",
                        },
                        "generated_biomods": {},
                    },
                    "motive": {},
                },
            }
        ]

        artifacts = comparison.comparison_derived_artifacts(reports)

        self.assertEqual(
            artifacts,
            {
                "Static/captury/generated_biomod": Path("/tmp/captury.bioMod"),
                "Static/motive/generated_biomod": Path("/tmp/motive.bioMod"),
                "Static/captury/rotation_audit_summary": Path(
                    "/tmp/captury_audit.json"
                ),
                "Static/captury/rotation_audit_timeseries": Path(
                    "/tmp/captury_audit.npz"
                ),
                "Static/markers/automatic_proposal": Path("/tmp/proposal.json"),
                "Static/alignment/spatial_calibration": Path(
                    "/tmp/spatial_calibration.json"
                ),
                "Static/kinematics/d4_d6_json": Path("/tmp/joint_d4_d6.json"),
                "Static/kinematics/d4_d6_timeseries": Path("/tmp/joint_d4_d6.npz"),
                "Static/kinematics/captury_c3d_angle_decode": Path(
                    "/tmp/captury_angles.json"
                ),
                "Static/kinematics/captury_c3d_angle_metrics": Path(
                    "/tmp/captury_angles.csv"
                ),
                "Static/kinematics/captury_c3d_angle_timeseries": Path(
                    "/tmp/captury_angles.npz"
                ),
                "Static/biobuddy/ik_npz": Path("/tmp/biobuddy_ik.npz"),
                "Static/biobuddy/ik_summary": Path("/tmp/biobuddy_ik.json"),
            },
        )

    def test_batch_isb_audit_artifacts_are_included_in_provenance(self) -> None:
        artifacts = comparison.comparison_derived_artifacts(
            [],
            batch_artifacts={
                "scientific/isb_d1_d3_json": Path("/tmp/isb_d1_d3_audit.json"),
                "scientific/isb_d1_d3_table": Path("/tmp/isb_d1_d3_audit.npz"),
            },
        )

        self.assertEqual(
            artifacts,
            {
                "scientific/isb_d1_d3_json": Path("/tmp/isb_d1_d3_audit.json"),
                "scientific/isb_d1_d3_table": Path("/tmp/isb_d1_d3_audit.npz"),
            },
        )

    def test_hidden_static_calibration_trial_is_included_once(self) -> None:
        static = comparison.TrialBundle(
            "Static",
            Path("static_cap.c3d"),
            Path("static_cap.bvh"),
            None,
            Path("static_mot.c3d"),
            Path("static_mot.bvh"),
            None,
        )
        walk = comparison.TrialBundle(
            "Marche_001",
            Path("walk_cap.c3d"),
            Path("walk_cap.bvh"),
            None,
            Path("walk_mot.c3d"),
            Path("walk_mot.bvh"),
            None,
        )

        provenance_trials = comparison.provenance_trials_with_static(
            [walk], [static, walk], "Static"
        )

        self.assertEqual(
            [trial.name for trial in provenance_trials], ["Marche_001", "Static"]
        )
        self.assertEqual(
            comparison.provenance_trials_with_static(
                [static, walk], [static, walk], "Static"
            ),
            [static, walk],
        )


if __name__ == "__main__":
    unittest.main()
