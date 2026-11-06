from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np

    from compare_p6_motive_captury import (
        TrialBundle,
        _source_joint_articulations,
        comparison_input_files,
        provenance_trials_with_static,
        captury_flat_trial_name,
        c3d_angle_scale_to_deg,
        centre_metric_rows,
        compose_row_alignment,
        compute_model_segment_rotations_native,
        discover_flat_trials,
        dimension_rows_from_centres,
        euler_matrix_from_sequence,
        euler_zxy_from_matrix,
        file_fingerprint,
        finite_range,
        foot_contact_from_markers,
        infer_participant_identifier,
        marker_proxy_centres_from_c3d,
        metric_quality_report,
        metric_sensitivity_report,
        normalize_summary_source,
        marker_indices_by_clean_label,
        model_to_c3d_matrix,
        motive_flat_trial_name,
        occlusion_rows_from_points,
        orient_segment_y_from_cor,
        propose_marker_correspondences_from_points,
        reexpress_rotational_q_from_segment_rotations,
        required_trial_outputs,
        resolve_cut_window,
        rotation_deviation_vector,
        rotations_to_c3d,
        rotate_segment_frames_180_x,
        root_alignment_score_mm,
        sanitize_channel_name,
        segment_rotation_metric_rows,
        segment_relative_rotation_curves,
        split_static_calibration_trial,
        spatial_calibration_from_report,
        static_transform_from_report,
        requested_model_to_own_c3d_axis,
        requested_root_offset_mode,
        time_window_mask,
        trial_cache_fingerprint,
        required_output_may_be_empty,
        yaw_alignment_rows,
    )
except ImportError as exc:  # pragma: no cover - depends on optional scientific env
    TrialBundle = None
    _source_joint_articulations = None
    comparison_input_files = None
    provenance_trials_with_static = None
    captury_flat_trial_name = None
    c3d_angle_scale_to_deg = None
    centre_metric_rows = None
    compose_row_alignment = None
    compute_model_segment_rotations_native = None
    discover_flat_trials = None
    dimension_rows_from_centres = None
    euler_matrix_from_sequence = None
    euler_zxy_from_matrix = None
    file_fingerprint = None
    finite_range = None
    foot_contact_from_markers = None
    infer_participant_identifier = None
    marker_proxy_centres_from_c3d = None
    metric_quality_report = None
    metric_sensitivity_report = None
    normalize_summary_source = None
    marker_indices_by_clean_label = None
    model_to_c3d_matrix = None
    motive_flat_trial_name = None
    occlusion_rows_from_points = None
    orient_segment_y_from_cor = None
    propose_marker_correspondences_from_points = None
    reexpress_rotational_q_from_segment_rotations = None
    required_trial_outputs = None
    resolve_cut_window = None
    rotation_deviation_vector = None
    rotations_to_c3d = None
    rotate_segment_frames_180_x = None
    root_alignment_score_mm = None
    sanitize_channel_name = None
    segment_rotation_metric_rows = None
    segment_relative_rotation_curves = None
    split_static_calibration_trial = None
    spatial_calibration_from_report = None
    static_transform_from_report = None
    requested_model_to_own_c3d_axis = None
    requested_root_offset_mode = None
    time_window_mask = None
    trial_cache_fingerprint = None
    required_output_may_be_empty = None
    yaw_alignment_rows = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@unittest.skipIf(
    IMPORT_ERROR is not None, f"optional dependencies missing: {IMPORT_ERROR}"
)
class FlatTrialDiscoveryTests(unittest.TestCase):
    def test_source_joint_articulations_preserve_canonical_and_exporter_names(
        self,
    ) -> None:
        assert _source_joint_articulations is not None

        right_hip = _source_joint_articulations("captury_model", "fbx")["right_hip"]

        self.assertEqual(right_hip["proximal"], "Hips")
        self.assertEqual(right_hip["distal"], "RightUpLeg")
        self.assertEqual(right_hip["proximal_segment_id"], "pelvis")
        self.assertEqual(right_hip["distal_segment_id"], "right_thigh")

    def test_finite_range_preserves_missing_trajectories_without_warning(
        self,
    ) -> None:
        assert finite_range is not None

        ranges = finite_range(
            np.asarray([[1.0, 4.0], [np.nan, np.nan], [-2.0, 3.0]]), axis=1
        )

        np.testing.assert_allclose(ranges[[0, 2]], [3.0, 5.0])
        self.assertTrue(np.isnan(ranges[1]))

    def test_missing_foot_markers_never_create_true_contacts(self) -> None:
        assert foot_contact_from_markers is not None
        contacts, z, speed, available = foot_contact_from_markers(
            np.zeros((3, 2, 5)), [], 0.01
        )

        self.assertFalse(available)
        self.assertFalse(np.any(contacts))
        self.assertTrue(np.all(np.isnan(z)))
        self.assertTrue(np.all(np.isnan(speed)))

    def test_participant_identifier_is_inferred_only_when_sources_agree(self) -> None:
        assert infer_participant_identifier is not None

        self.assertEqual(
            infer_participant_identifier(
                Path("Captury/Walk_P6.c3d"), Path("Motive/P6_Walk.c3d")
            ),
            "P6",
        )
        self.assertIsNone(
            infer_participant_identifier(
                Path("Captury/Walk_P6.c3d"), Path("Motive/P7_Walk.c3d")
            )
        )
        self.assertIsNone(infer_participant_identifier(Path("Captury/Walk.c3d")))

    def test_nan_source_is_replaced_before_population_aggregation(self) -> None:
        assert normalize_summary_source is not None
        rows = [{"source": np.nan}, {}, {"source": "biobuddy_ik"}]

        normalize_summary_source(rows, "captury")

        self.assertEqual(
            [row["source"] for row in rows], ["captury", "captury", "biobuddy_ik"]
        )

    def test_metric_quality_report_counts_ineligible_waveforms(self) -> None:
        assert metric_quality_report is not None
        report = metric_quality_report(
            [
                {"waveform_status": "ok", "shape_metrics_eligible": True},
                {
                    "waveform_status": "low_reference_amplitude",
                    "shape_metrics_eligible": False,
                },
                {
                    "waveform_status": "insufficient_coverage",
                    "shape_metrics_eligible": False,
                },
                {"source": "captury_c3d", "c3d_mean_deg": 12.0},
            ]
        )

        self.assertEqual(report["waveforms"], 3)
        self.assertEqual(report["eligible_waveforms"], 1)
        self.assertEqual(report["ineligible_waveforms"], 2)
        self.assertEqual(report["status_counts"]["low_reference_amplitude"], 1)

    def test_sensitivity_report_uses_only_computed_counterfactuals(self) -> None:
        assert metric_sensitivity_report is not None
        report = metric_sensitivity_report(
            root_policies={
                "captury": {
                    "score_mm_subtract_static_offset": 10.0,
                    "score_mm_keep_file_translation": 25.0,
                    "selected_mode": "subtract_static_offset_from_root_q",
                }
            },
            temporal={
                "normalized_rmse_before": 0.4,
                "normalized_rmse_after": 0.2,
                "estimated_lag_s": 0.1,
                "lag_s": 0.1,
            },
            evaluation_centre_rows=[{"median_error_mm": 30.0}],
            calibration_centre_rows=[{"median_error_mm": 5.0}],
            rotation_audits={},
        )

        self.assertEqual(report["root_translation"]["captury"]["status"], "computed")
        self.assertAlmostEqual(
            report["root_translation"]["captury"]["score_difference_mm"], 15.0
        )
        self.assertEqual(report["temporal_lag"]["status"], "computed")
        self.assertEqual(report["bvh_fbx_source"]["status"], "not_computed")

    def test_segment_rotation_metrics_name_global_so3_error_as_geodesic(self) -> None:
        reference = np.repeat(np.eye(3)[:, :, None], 2, axis=2)
        test = reference.copy()
        angle = np.deg2rad(10.0)
        test[:, :, 1] = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        rows, timeseries, _report = segment_rotation_metric_rows(
            "Walk",
            {"biobuddy": {"pelvis": reference}, "captury": {"pelvis": test}},
            {"biobuddy": np.asarray([0.0, 1.0]), "captury": np.asarray([0.0, 1.0])},
            "biobuddy",
        )

        self.assertAlmostEqual(rows[0]["max_geodesic_deg"], 10.0)
        self.assertAlmostEqual(timeseries[1]["geodesic_deg"], 10.0)

    def test_segment_rotation_metrics_use_slerp_at_different_frame_rates(self) -> None:
        def rotation_z(angle: float) -> np.ndarray:
            return np.asarray(
                [
                    [np.cos(angle), -np.sin(angle), 0.0],
                    [np.sin(angle), np.cos(angle), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

        reference = np.stack(
            [rotation_z(angle) for angle in (0.0, np.pi / 4.0, np.pi / 2.0)],
            axis=2,
        )
        source = np.stack([rotation_z(0.0), rotation_z(np.pi / 2.0)], axis=2)

        rows, timeseries, _report = segment_rotation_metric_rows(
            "Walk",
            {"biobuddy": {"pelvis": reference}, "captury": {"pelvis": source}},
            {
                "biobuddy": np.asarray([0.0, 0.5, 1.0]),
                "captury": np.asarray([0.0, 1.0]),
            },
            "biobuddy",
        )

        self.assertAlmostEqual(rows[0]["max_geodesic_deg"], 0.0, places=10)
        np.testing.assert_allclose(
            [row["geodesic_deg"] for row in timeseries], 0.0, atol=1e-10
        )

    def test_missing_biobuddy_segment_reference_never_falls_back_to_motive(
        self,
    ) -> None:
        summary, timeseries, report = segment_rotation_metric_rows(
            "Static",
            {
                "captury": {},
                "motive": {"pelvis": np.eye(3)[:, :, None]},
                "biobuddy": {},
            },
            {
                "captury": np.asarray([]),
                "motive": np.asarray([0.0]),
                "biobuddy": np.asarray([]),
            },
            "biobuddy",
        )

        self.assertEqual(summary, [])
        self.assertEqual(timeseries, [])
        self.assertEqual(report["status"], "missing_reference")
        self.assertNotIn("effective_reference", report)

    def test_static_calibration_trial_is_removed_from_dynamic_order(self) -> None:
        assert split_static_calibration_trial is not None
        static = TrialBundle(
            "Static", Path("cap.c3d"), None, None, Path("mot.c3d"), None, None
        )
        walk = TrialBundle(
            "Marche_001",
            Path("cap_walk.c3d"),
            None,
            None,
            Path("mot_walk.c3d"),
            None,
            None,
        )

        calibration, dynamic = split_static_calibration_trial([walk, static], "Static")

        self.assertIs(calibration, static)
        self.assertEqual([bundle.name for bundle in dynamic], ["Marche_001"])

    def test_root_policy_defaults_keep_captury_and_auto_score_motive(self) -> None:
        assert requested_root_offset_mode is not None
        args = argparse.Namespace(
            root_offset_mode="auto",
            captury_root_offset_mode="keep",
            motive_root_offset_mode=None,
        )

        self.assertEqual(requested_root_offset_mode(args, "captury", None), "keep")
        self.assertEqual(requested_root_offset_mode(args, "motive", None), "auto")

    def test_manual_model_axis_mode_must_name_the_system(self) -> None:
        assert requested_model_to_own_c3d_axis is not None

        legacy = argparse.Namespace(model_to_c3d_axis="identity")
        with self.assertRaises(ValueError):
            requested_model_to_own_c3d_axis(legacy, "captury")

        explicit = argparse.Namespace(
            model_to_c3d_axis="auto",
            captury_model_to_c3d_axis="identity",
            motive_model_to_c3d_axis="y_up_to_z_up",
        )
        self.assertEqual(
            requested_model_to_own_c3d_axis(explicit, "captury"), "identity"
        )
        self.assertEqual(
            requested_model_to_own_c3d_axis(explicit, "motive"), "y_up_to_z_up"
        )

    def test_comparison_inputs_include_only_selected_model_files(self) -> None:
        assert TrialBundle is not None
        assert comparison_input_files is not None

        bundle = TrialBundle(
            name="Static",
            captury_c3d=Path("captury.c3d"),
            captury_bvh=Path("captury.bvh"),
            captury_fbx=Path("captury.fbx"),
            motive_c3d=Path("motive.c3d"),
            motive_bvh=Path("motive.bvh"),
            motive_fbx=Path("motive.fbx"),
        )
        args = argparse.Namespace(
            model_source="fbx",
            biobuddy_biomod=Path("model.bioMod"),
            landmark_map=Path("markers.json"),
            occlusions_only=False,
        )

        inputs = comparison_input_files([bundle], args)

        self.assertEqual(
            inputs,
            {
                "Static/captury/c3d": Path("captury.c3d"),
                "Static/captury/fbx": Path("captury.fbx"),
                "Static/motive/c3d": Path("motive.c3d"),
                "Static/motive/fbx": Path("motive.fbx"),
                "biobuddy/biomod": Path("model.bioMod"),
                "markers/landmark_map": Path("markers.json"),
            },
        )

    def test_provenance_includes_hidden_static_calibration_trial(self) -> None:
        assert TrialBundle is not None
        assert provenance_trials_with_static is not None

        static = TrialBundle(
            "Static",
            Path("static_cap.c3d"),
            Path("static_cap.bvh"),
            None,
            Path("static_mot.c3d"),
            Path("static_mot.bvh"),
            None,
        )
        walk = TrialBundle(
            "Marche_001",
            Path("walk_cap.c3d"),
            Path("walk_cap.bvh"),
            None,
            Path("walk_mot.c3d"),
            Path("walk_mot.bvh"),
            None,
        )

        provenance_trials = provenance_trials_with_static(
            [walk], [static, walk], "Static"
        )

        self.assertEqual(
            [trial.name for trial in provenance_trials], ["Marche_001", "Static"]
        )
        self.assertEqual(
            provenance_trials_with_static([static, walk], [static, walk], "Static"),
            [static, walk],
        )

    def test_flat_trial_names_strip_system_specific_suffixes(self) -> None:
        assert captury_flat_trial_name is not None
        assert motive_flat_trial_name is not None

        self.assertEqual(captury_flat_trial_name(Path("Static_P6.c3d")), "Static")
        self.assertEqual(
            motive_flat_trial_name(Path("P6_Static_Skeleton 001.bvh")), "Static"
        )
        self.assertEqual(
            motive_flat_trial_name(Path("P6_Marche_001.c3d")), "Marche_001"
        )

    def test_required_trial_outputs_use_npz_for_timeseries(self) -> None:
        assert required_trial_outputs is not None

        outputs = [path.name for path in required_trial_outputs(Path("out"), "Static")]

        self.assertIn("joint_centre_timeseries.npz", outputs)
        self.assertIn("alignment_calibration_centre_metrics.csv", outputs)
        self.assertIn("alignment_calibration_centre_timeseries.npz", outputs)
        self.assertIn("spatial_calibration.json", outputs)
        self.assertIn("kinematics_q_timeseries.npz", outputs)
        self.assertIn("captury_c3d_angle_metrics.csv", outputs)
        self.assertIn("captury_c3d_angle_timeseries.npz", outputs)
        self.assertIn("segment_rotation_metrics.csv", outputs)
        self.assertIn("segment_rotation_timeseries.npz", outputs)
        self.assertIn("joint_kinematics_d4_d6.json", outputs)
        self.assertIn("joint_kinematics_d4_d6.npz", outputs)
        self.assertIn("skin_marker_correspondence_timeseries.npz", outputs)
        self.assertIn("temporal_synchronization.json", outputs)
        self.assertIn("temporal_synchronization_timeseries.npz", outputs)
        self.assertIn("temporal_phase_normalized.npz", outputs)
        self.assertIn("phase_normalized_scientific_timeseries.npz", outputs)
        self.assertIn("contact_cycles.json", outputs)
        self.assertIn("metric_quality.json", outputs)
        self.assertIn("metric_sensitivity.json", outputs)
        self.assertNotIn("joint_centre_timeseries.csv", outputs)
        self.assertNotIn("kinematics_q_timeseries.csv", outputs)
        self.assertNotIn("segment_rotation_timeseries.csv", outputs)
        self.assertNotIn("joint_kinematics_d4_d6.csv", outputs)

        audit_outputs = [
            str(path)
            for path in required_trial_outputs(
                Path("out"), "Static", include_rotation_audit=True
            )
        ]
        self.assertIn("out/captury/bvh_fbx_rotation_audit.npz", audit_outputs)
        self.assertIn("out/motive/bvh_fbx_rotation_audit.json", audit_outputs)

    def test_empty_skin_marker_metrics_are_valid_cache_outputs(self) -> None:
        assert required_output_may_be_empty is not None

        self.assertTrue(
            required_output_may_be_empty(Path("skin_marker_correspondence_metrics.csv"))
        )
        self.assertFalse(required_output_may_be_empty(Path("run_report.json")))

    def test_dimension_rows_from_centres_supports_biobuddy_source(self) -> None:
        assert dimension_rows_from_centres is not None

        centres_mm = {
            "LThigh": np.asarray([[0.0], [0.0], [0.0]]),
            "LShank": np.asarray([[0.0], [400.0], [0.0]]),
        }

        rows = dimension_rows_from_centres(
            "Static", "biobuddy", "motive_57", centres_mm
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["system"], "biobuddy")
        self.assertEqual(rows[0]["source_kind"], "motive_57")
        self.assertEqual(rows[0]["dimension"], "left_thigh")
        self.assertAlmostEqual(rows[0]["median_length_mm"], 400.0)

    def test_marker_indices_support_duplicate_number_suffixes(self) -> None:
        assert marker_indices_by_clean_label is not None

        lookup = marker_indices_by_clean_label(["Q_Hip", "Q_Knee", "Q_Hip"])

        self.assertEqual(lookup["Q_Hip"], [0, 2])
        self.assertEqual(lookup["Q_Hip#1"], [0])
        self.assertEqual(lookup["Q_Hip#2"], [2])

    def test_compose_row_alignment_matches_sequential_transforms(self) -> None:
        assert compose_row_alignment is not None

        first_rotation = np.asarray(
            [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        first_translation = np.asarray([10.0, 20.0, 30.0])
        second_rotation = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
        )
        second_translation = np.asarray([-5.0, 2.0, 8.0])
        points = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        rotation, translation = compose_row_alignment(
            first_rotation, first_translation, second_rotation, second_translation
        )

        sequential = (
            points @ first_rotation + first_translation
        ) @ second_rotation + second_translation
        composed = points @ rotation + translation
        np.testing.assert_allclose(composed, sequential)

    def test_marker_correspondence_proposal_pairs_nearest_aligned_skin_markers(
        self,
    ) -> None:
        assert propose_marker_correspondences_from_points is not None

        time = np.asarray([0.0, 1.0])
        motive_points = np.asarray(
            [
                [[0.0, 0.0], [100.0, 100.0], [200.0, 200.0], [300.0, 300.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        captury_points = motive_points[:, [0, 1, 3], :].copy()
        captury_points[:, 0, :] += 500.0
        pairs, report = propose_marker_correspondences_from_points(
            ["Skeleton_001_LIAS", "Skeleton_001_RIAS", "CAPJC_Hips", "LANK"],
            motive_points,
            time,
            ["Q_A", "Q_B", "Q_C"],
            captury_points,
            time,
            np.eye(3),
            np.zeros(3),
            max_median_error_mm=1.0,
        )

        self.assertEqual(report["selected_count"], 2)
        self.assertEqual(
            [(row["reference"][0], row["test"][0]) for row in pairs],
            [("RIAS", "Q_B"), ("LANK", "Q_C")],
        )

    def test_marker_proposal_applies_captury_lag_before_interpolation(self) -> None:
        assert propose_marker_correspondences_from_points is not None

        motive_time = np.asarray([0.0, 0.5, 1.0, 1.5])
        captury_time = motive_time - 0.5
        trajectory = np.asarray(
            [motive_time * 100.0, motive_time**2, motive_time * 0.0]
        )
        motive_points = trajectory[:, None, :]
        captury_points = trajectory[:, None, :].copy()

        synchronized, _ = propose_marker_correspondences_from_points(
            ["LIAS"],
            motive_points,
            motive_time,
            ["Q_A"],
            captury_points,
            captury_time,
            np.eye(3),
            np.zeros(3),
            captury_lag_s=0.5,
            max_median_error_mm=1e-9,
        )
        unsynchronized, _ = propose_marker_correspondences_from_points(
            ["LIAS"],
            motive_points,
            motive_time,
            ["Q_A"],
            captury_points,
            captury_time,
            np.eye(3),
            np.zeros(3),
            captury_lag_s=0.0,
            max_median_error_mm=1e-9,
        )

        self.assertEqual(len(synchronized), 1)
        self.assertEqual(unsynchronized, [])

    def test_rotation_deviation_vector_reports_axis_angle_components(self) -> None:
        assert rotation_deviation_vector is not None

        angle = np.deg2rad(10.0)
        rotation_x = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )

        vector = rotation_deviation_vector(np.eye(3), rotation_x)

        np.testing.assert_allclose(vector, [angle, 0.0, 0.0], atol=1e-10)

    def test_centre_metrics_can_exclude_reserved_calibration_centres(self) -> None:
        assert centre_metric_rows is not None

        time = np.asarray([0.0, 1.0])
        captury = {
            "Hips": np.asarray([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            "LeftLeg": np.asarray([[2.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
        }
        motive = {
            "Hips": np.zeros((3, 2)),
            "LeftLeg": np.zeros((3, 2)),
        }

        summary, timeseries = centre_metric_rows(
            "Static",
            captury,
            motive,
            time,
            time,
            excluded_joints={"Hips"},
        )

        self.assertEqual([row["joint"] for row in summary], ["LeftLeg"])
        self.assertEqual(sorted(set(row["joint"] for row in timeseries)), ["LeftLeg"])

    def test_rotation_deviation_vector_is_finite_at_180_degrees(self) -> None:
        assert rotation_deviation_vector is not None

        rotation_x_180 = np.diag([1.0, -1.0, -1.0])
        vector = rotation_deviation_vector(np.eye(3), rotation_x_180)

        self.assertTrue(np.all(np.isfinite(vector)))
        self.assertAlmostEqual(np.linalg.norm(vector), np.pi, places=12)

    def test_rotations_to_c3d_canonicalizes_small_fbx_scale_roundoff(self) -> None:
        assert rotations_to_c3d is not None

        raw = np.diag([1.0 + 2.0e-6, 1.0 - 1.0e-6, 1.0])[:, :, None]
        converted = rotations_to_c3d({"pelvis": raw}, "y_up_to_z_up")["pelvis"]

        np.testing.assert_allclose(
            converted[:, :, 0].T @ converted[:, :, 0], np.eye(3), atol=1e-12
        )
        self.assertAlmostEqual(np.linalg.det(converted[:, :, 0]), 1.0, places=12)

    def test_global_jcs_rotation_block_is_stored_without_transposition(self) -> None:
        assert compute_model_segment_rotations_native is not None

        expected = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        class FakeTransform:
            def to_array(self):
                transform = np.eye(4)
                transform[:3, :3] = expected
                return transform

        class FakeModel:
            def nbQ(self):
                return 1

            def nbSegment(self):
                return 2

            def segment(self, index):
                names = ("root", "pelvis")

                class Name:
                    def to_string(self):
                        return names[index]

                class Segment:
                    def name(self):
                        return Name()

                return Segment()

            def globalJCS(self, _q, index):
                self.last_index = index
                return FakeTransform()

        class FakeBiorbd:
            def __init__(self):
                self.model = FakeModel()

            def Model(self, _path):
                return self.model

        import compare_p6_motive_captury as comparison
        from unittest.mock import patch

        fake_biorbd = FakeBiorbd()
        with patch.object(comparison, "require_biorbd", return_value=fake_biorbd):
            rotations = compute_model_segment_rotations_native(
                Path("synthetic.bioMod"), np.asarray([[0.25]]), {"pelvis"}
            )

        np.testing.assert_allclose(rotations["pelvis"][:, :, 0], expected)
        self.assertEqual(fake_biorbd.model.last_index, 1)

    def test_biorbd_global_jcs_matches_known_static_biomod_rotation(self) -> None:
        assert compute_model_segment_rotations_native is not None

        biomod = """version 4
segment root
    parent base
    RTinMatrix 1
    RT
        1 0 0 0
        0 1 0 0
        0 0 1 0
        0 0 0 1
endsegment
segment pelvis
    parent root
    RTinMatrix 1
    RT
        0 -1 0 0
        1 0 0 0
        0 0 1 0
        0 0 0 1
endsegment
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "known_rotation.bioMod"
            path.write_text(biomod, encoding="utf-8")
            rotations = compute_model_segment_rotations_native(
                path, np.empty((0, 1)), {"pelvis"}
            )

        expected = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(rotations["pelvis"][:, :, 0], expected)

    def test_rotate_segment_frames_180_x_changes_local_x_and_y_axes(self) -> None:
        assert rotate_segment_frames_180_x is not None

        corrected = rotate_segment_frames_180_x({"Thigh": np.eye(3)[:, :, None]})[
            "Thigh"
        ]

        np.testing.assert_allclose(corrected[:, :, 0], np.diag([-1.0, -1.0, 1.0]))

    def test_euler_zxy_roundtrip_from_rotation_matrix(self) -> None:
        assert euler_matrix_from_sequence is not None
        assert euler_zxy_from_matrix is not None

        expected = np.deg2rad([20.0, -10.0, 35.0])
        rotation = euler_matrix_from_sequence(expected, "ZXY")

        actual = euler_zxy_from_matrix(rotation)

        np.testing.assert_allclose(actual, expected, atol=1e-10)
        np.testing.assert_allclose(
            euler_matrix_from_sequence(actual, "ZXY"), rotation, atol=1e-10
        )

    def test_reexpress_rotational_q_uses_corrected_segment_matrix(self) -> None:
        assert euler_matrix_from_sequence is not None
        assert reexpress_rotational_q_from_segment_rotations is not None

        q_names = ["Thigh_rotZ", "Thigh_rotX", "Thigh_rotY", "Thigh_transX"]
        original_q = np.asarray([[0.0], [0.0], [0.0], [123.0]])
        original_rotation = euler_matrix_from_sequence(
            np.deg2rad([5.0, 8.0, 11.0]), "ZXY"
        )
        corrected_rotation = original_rotation @ np.diag([-1.0, -1.0, 1.0])

        q, report = reexpress_rotational_q_from_segment_rotations(
            original_q,
            q_names,
            {"Thigh": corrected_rotation[:, :, None]},
            "ZXY",
        )

        np.testing.assert_allclose(
            euler_matrix_from_sequence(q[:3, 0], "ZXY"),
            corrected_rotation,
            atol=1e-10,
        )
        self.assertEqual(float(q[3, 0]), 123.0)
        self.assertEqual(report["changed_segments"], ["Thigh"])

    def test_orient_segment_y_from_cor_uses_proximal_to_distal_direction(
        self,
    ) -> None:
        assert orient_segment_y_from_cor is not None

        rotations = {"LeftUpLeg": np.eye(3)[:, :, None]}
        centres = {
            "LeftUpLeg": np.asarray([[0.0], [0.0], [0.0]]),
            "LeftLeg": np.asarray([[0.0], [0.0], [2.0]]),
        }

        corrected = orient_segment_y_from_cor(
            rotations, centres, "LeftUpLeg", "LeftUpLeg", "LeftLeg"
        )

        np.testing.assert_allclose(
            corrected["LeftUpLeg"][:, 1, 0], [0.0, 0.0, 1.0], atol=1e-10
        )

    def test_segment_relative_rotation_curves_extract_joint_rotation(self) -> None:
        assert segment_relative_rotation_curves is not None

        angle = np.deg2rad(15.0)
        rotation_z = np.asarray(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        curves = segment_relative_rotation_curves(
            {
                "LeftUpLeg": np.eye(3)[:, :, None],
                "LeftLeg": rotation_z[:, :, None],
            }
        )

        np.testing.assert_allclose(
            curves["SegRel_LeftKnee"][:, 0], [0.0, 0.0, angle], atol=1e-10
        )

    def test_segment_metrics_project_small_biorbd_rotation_roundoff(self) -> None:
        assert segment_rotation_metric_rows is not None
        noisy = np.eye(3)
        noisy[0, 0] += 1.0e-6

        rows, timeseries, report = segment_rotation_metric_rows(
            "Static",
            {
                "motive": {"Pelvis": np.eye(3)[:, :, None]},
                "biobuddy": {"Pelvis": noisy[:, :, None]},
            },
            {"motive": np.asarray([0.0]), "biobuddy": np.asarray([0.0])},
            "motive",
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(timeseries), 1)
        self.assertLess(rows[0]["max_global_deg"], 1e-8)

    def test_segment_metrics_do_not_repeat_endpoints_outside_time_overlap(
        self,
    ) -> None:
        assert segment_rotation_metric_rows is not None
        rotations = np.repeat(np.eye(3)[:, :, None], 3, axis=2)

        rows, timeseries, report = segment_rotation_metric_rows(
            "Walk",
            {
                "motive": {"Pelvis": rotations},
                "captury": {"Pelvis": rotations[:, :, 1:]},
            },
            {
                "motive": np.asarray([0.0, 1.0, 2.0]),
                "captury": np.asarray([1.0, 2.0]),
            },
            "motive",
        )

        self.assertEqual(report["status"], "ok")
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(timeseries), 2)
        self.assertEqual([row["time"] for row in timeseries], [1.0, 2.0])

    def test_c3d_angle_scale_to_deg_handles_rad_and_deg(self) -> None:
        assert c3d_angle_scale_to_deg is not None

        self.assertEqual(c3d_angle_scale_to_deg("deg"), 1.0)
        self.assertAlmostEqual(c3d_angle_scale_to_deg("rad"), 180.0 / np.pi)

    def test_sanitize_channel_name_keeps_angle_q_names_valid(self) -> None:
        assert sanitize_channel_name is not None

        self.assertEqual(
            sanitize_channel_name("R Hip.angle", "fallback"), "R_Hip_angle"
        )
        self.assertEqual(sanitize_channel_name("123", "fallback"), "_123")
        self.assertEqual(sanitize_channel_name(" ", "fallback"), "fallback")

    def test_root_alignment_score_uses_nearest_marker_distances_in_mm(self) -> None:
        assert root_alignment_score_mm is not None

        time = np.asarray([0.0, 1.0])
        centres = {"Hips": np.asarray([[0.0, 10.0], [0.0, 0.0], [0.0, 0.0]])}
        markers = np.asarray(
            [
                [[1.0, 11.0], [100.0, 100.0]],
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]
        )

        score = root_alignment_score_mm(centres, time, markers, time)

        self.assertAlmostEqual(score, 1.0)

    def test_root_alignment_score_ignores_frames_outside_overlap(self) -> None:
        assert root_alignment_score_mm is not None
        source_time = np.asarray([1.0, 2.0])
        c3d_time = np.asarray([0.0, 0.5, 1.0, 2.0])
        centres = {"Hips": np.zeros((3, 2))}
        markers = np.asarray(
            [
                [[0.0, 0.0, 100.0, 100.0]],
                [[0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ]
        )

        score = root_alignment_score_mm(
            centres, source_time, markers, c3d_time, max_frames=4
        )

        self.assertAlmostEqual(score, 100.0)

    def test_discover_flat_trials_retains_missing_source_trials(self) -> None:
        assert discover_flat_trials is not None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captury = root / "Captury"
            motive = root / "Motive"
            captury.mkdir()
            motive.mkdir()

            for suffix in ("c3d", "bvh", "fbx"):
                (captury / f"Static_P6.{suffix}").write_text(
                    "captury", encoding="utf-8"
                )
            (motive / "P6_Static.c3d").write_text("motive", encoding="utf-8")
            (motive / "P6_Static.fbx").write_text("motive", encoding="utf-8")
            (motive / "P6_Static_Skeleton 001.bvh").write_text(
                "motive", encoding="utf-8"
            )

            (captury / "MissingMotive_P6.c3d").write_text("captury", encoding="utf-8")
            (motive / "P6_MissingCaptury.c3d").write_text("motive", encoding="utf-8")

            trials = discover_flat_trials(root)

            self.assertEqual(
                [trial.name for trial in trials],
                ["MissingCaptury", "MissingMotive", "Static"],
            )
            self.assertIsNone(trials[0].captury_c3d)
            self.assertIsNone(trials[1].motive_c3d)
            bundle = trials[2]
            self.assertEqual(bundle.captury_c3d.name, "Static_P6.c3d")
            self.assertEqual(bundle.captury_bvh.name, "Static_P6.bvh")
            self.assertEqual(bundle.captury_fbx.name, "Static_P6.fbx")
            self.assertEqual(bundle.motive_c3d.name, "P6_Static.c3d")
            self.assertEqual(bundle.motive_bvh.name, "P6_Static_Skeleton 001.bvh")
            self.assertEqual(bundle.motive_fbx.name, "P6_Static.fbx")

    def test_auto_axis_uses_current_y_up_to_z_up_conversion(self) -> None:
        assert model_to_c3d_matrix is not None

        np.testing.assert_allclose(
            model_to_c3d_matrix("auto"), model_to_c3d_matrix("y_up_to_z_up")
        )

    def test_time_window_mask_keeps_requested_seconds(self) -> None:
        assert time_window_mask is not None

        mask = time_window_mask(np.asarray([0.0, 0.5, 1.0, 1.5, 2.0]), 0.5, 1.5)

        np.testing.assert_array_equal(mask, [False, True, True, True, False])

    def test_resolve_cut_window_can_use_detected_movement(self) -> None:
        assert resolve_cut_window is not None

        start, end, mode = resolve_cut_window(
            "movement",
            None,
            None,
            {"movement_start_time": 0.25, "movement_end_time": 1.5},
        )

        self.assertEqual((start, end, mode), (0.25, 1.5, "movement"))

    def test_trial_cache_fingerprint_changes_when_input_changes(self) -> None:
        assert TrialBundle is not None
        assert trial_cache_fingerprint is not None

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {}
            for name in (
                "captury.c3d",
                "captury.bvh",
                "captury.fbx",
                "motive.c3d",
                "motive.bvh",
                "motive.fbx",
            ):
                paths[name] = root / name
                paths[name].write_text(name, encoding="utf-8")
            bundle = TrialBundle(
                name="Static",
                captury_c3d=paths["captury.c3d"],
                captury_bvh=paths["captury.bvh"],
                captury_fbx=paths["captury.fbx"],
                motive_c3d=paths["motive.c3d"],
                motive_bvh=paths["motive.bvh"],
                motive_fbx=paths["motive.fbx"],
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
                cut_mode="manual",
                time_start=None,
                time_end=None,
                temporal_sync_mode="auto",
                manual_lag_s=None,
                max_lag_s=0.5,
                phase_normalization_points=101,
                no_figures=True,
            )

            first = trial_cache_fingerprint(bundle, args)
            args.audit_bvh_fbx_rotations = True
            audited = trial_cache_fingerprint(bundle, args)
            args.bvh_fbx_max_p95_geodesic_deg = 2.0
            stricter_audit = trial_cache_fingerprint(bundle, args)
            args.audit_bvh_fbx_rotations = False
            args.bvh_fbx_max_p95_geodesic_deg = 5.0
            args.captury_reorient_thigh_y_from_cor = True
            fourth = trial_cache_fingerprint(bundle, args)
            args.captury_reorient_thigh_y_from_cor = False
            args.reexpress_rotations_zxy = True
            fifth = trial_cache_fingerprint(bundle, args)
            args.reexpress_rotations_zxy = False
            args.temporal_sync_mode = "none"
            unsynchronized = trial_cache_fingerprint(bundle, args)
            args.temporal_sync_mode = "auto"
            args.root_offset_mode = "keep"
            third = trial_cache_fingerprint(bundle, args)
            args.root_offset_mode = "auto"
            paths["motive.c3d"].write_text("changed", encoding="utf-8")
            second = trial_cache_fingerprint(bundle, args)

            self.assertNotEqual(first["digest"], second["digest"])
            self.assertNotEqual(first["digest"], audited["digest"])
            self.assertNotEqual(audited["digest"], stricter_audit["digest"])
            self.assertNotEqual(first["digest"], third["digest"])
            self.assertNotEqual(first["digest"], fourth["digest"])
            self.assertNotEqual(first["digest"], fifth["digest"])
            self.assertNotEqual(first["digest"], unsynchronized["digest"])

    def test_file_fingerprint_records_missing_files(self) -> None:
        assert file_fingerprint is not None

        with tempfile.TemporaryDirectory() as tmp:
            fingerprint = file_fingerprint(Path(tmp) / "missing.c3d")

        self.assertFalse(fingerprint["exists"])

    def test_static_transform_from_report_returns_numpy_arrays(self) -> None:
        assert static_transform_from_report is not None

        transform = static_transform_from_report(
            {
                "alignment": {
                    "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    "translation_mm": [1.0, 2.0, 3.0],
                }
            }
        )

        self.assertIsNotNone(transform)
        rotation, translation = transform
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(translation, [1.0, 2.0, 3.0])

    def test_spatial_calibration_report_freezes_system_root_policies(self) -> None:
        assert spatial_calibration_from_report is not None
        assert requested_root_offset_mode is not None

        report = {
            "spatial_calibration": {
                "schema_version": 1,
                "status": "ok",
                "static_trial": "Static",
                "calibration_centres": [
                    "Hips",
                    "Head",
                    "LeftShoulder",
                    "RightShoulder",
                ],
                "evaluation_centres": ["LeftLeg", "RightLeg"],
                "root_translation_policy": {
                    "captury": "keep",
                    "motive": "subtract",
                },
                "captury_to_motive": {
                    "rotation": np.eye(3).tolist(),
                    "translation_mm": [1.0, 2.0, 3.0],
                },
                "motive_to_c3d": {
                    "rotation": np.eye(3).tolist(),
                    "translation_mm": [4.0, 5.0, 6.0],
                },
            }
        }
        calibration = spatial_calibration_from_report(report)
        self.assertIsNotNone(calibration)
        args = argparse.Namespace(
            root_offset_mode="auto",
            captury_root_offset_mode="subtract",
            motive_root_offset_mode="keep",
        )

        self.assertEqual(
            requested_root_offset_mode(args, "captury", calibration), "keep"
        )
        self.assertEqual(
            requested_root_offset_mode(args, "motive", calibration), "subtract"
        )

    def test_occlusion_rows_use_vectorized_missing_points_and_residuals(self) -> None:
        assert occlusion_rows_from_points is not None

        points = np.asarray(
            [
                [[0.0, np.nan, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
                [[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]],
            ]
        )
        residuals = np.asarray([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])

        rows = occlusion_rows_from_points(
            "Static", ["Skeleton_001_A", "B"], points, residuals
        )

        self.assertEqual(rows[0]["marker"], "A")
        self.assertEqual(rows[0]["missing_frames"], 2)
        self.assertAlmostEqual(rows[0]["missing_percent"], 100.0 * 2.0 / 3.0)
        self.assertEqual(rows[1]["missing_frames"], 0)

    def test_yaw_alignment_rows_recovers_vertical_axis_rotation(self) -> None:
        assert yaw_alignment_rows is not None

        moving = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
                [0.0, 200.0, 0.0],
                [100.0, 200.0, 500.0],
            ]
        )
        angle = np.deg2rad(90.0)
        row_rotation = np.asarray(
            [
                [np.cos(angle), np.sin(angle), 0.0],
                [-np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.asarray([10.0, -20.0, 30.0])
        reference = moving @ row_rotation + translation

        rotation, recovered_translation, report = yaw_alignment_rows(reference, moving)

        self.assertEqual(report["status"], "ok")
        np.testing.assert_allclose(rotation, row_rotation, atol=1e-12)
        np.testing.assert_allclose(recovered_translation, translation, atol=1e-12)
        np.testing.assert_allclose(moving @ rotation + recovered_translation, reference)

    def test_marker_proxy_centres_from_c3d_uses_clean_motive_labels(self) -> None:
        assert marker_proxy_centres_from_c3d is not None

        labels = [
            "Skeleton_001_LFLE",
            "Skeleton_001_LFME",
            "Skeleton_001_RFLE",
            "Skeleton_001_RFME",
        ]
        points = np.asarray(
            [
                [[0.0, 2.0], [2.0, 4.0], [10.0, 12.0], [14.0, 16.0]],
                [[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [14.0, 14.0]],
                [
                    [100.0, 100.0],
                    [102.0, 102.0],
                    [110.0, 110.0],
                    [114.0, 114.0],
                ],
            ]
        )

        proxies = marker_proxy_centres_from_c3d(labels, points)

        np.testing.assert_allclose(
            proxies["LeftLeg"], [[1.0, 3.0], [1.0, 1.0], [101.0, 101.0]]
        )
        np.testing.assert_allclose(
            proxies["RightLeg"], [[12.0, 14.0], [12.0, 12.0], [112.0, 112.0]]
        )

    def test_joint_filter_keeps_full_timeseries_for_visualization(self) -> None:
        assert centre_metric_rows is not None

        time = np.asarray([0.0, 1.0])
        captury = {
            "Hips": np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            "LeftLeg": np.asarray([[0.0, 1.0], [10.0, 10.0], [0.0, 0.0]]),
            "Spine": np.asarray([[0.0, 1.0], [20.0, 20.0], [0.0, 0.0]]),
        }
        motive = {
            "Hips": np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
            "LeftLeg": np.asarray([[0.0, 1.0], [11.0, 11.0], [0.0, 0.0]]),
            "Spine": np.asarray([[0.0, 1.0], [21.0, 21.0], [0.0, 0.0]]),
        }

        summary_rows, timeseries_rows = centre_metric_rows(
            "Walk", captury, motive, time, time, ["Leg"]
        )

        self.assertEqual({row["joint"] for row in summary_rows}, {"LeftLeg"})
        self.assertEqual(
            {row["joint"] for row in timeseries_rows},
            {"Hips", "LeftLeg", "Spine"},
        )


if __name__ == "__main__":
    unittest.main()
