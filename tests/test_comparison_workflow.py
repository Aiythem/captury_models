import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import compare_p6_motive_captury as c
from gui_commands import build_p6_args


class FlexibleWorkflowTests(unittest.TestCase):
    def make_bundle(self, root, name, systems):
        paths = {}
        for system in systems:
            for ext in ("c3d", "bvh"):
                p = root / f"{name}_{system}.{ext}"
                p.touch()
                paths[f"{system}_{ext}"] = p
        return c.TrialBundle(
            name,
            paths.get("captury_c3d"),
            paths.get("captury_bvh"),
            None,
            paths.get("motive_c3d"),
            paths.get("motive_bvh"),
            None,
        )

    def test_mixed_folder_keeps_unpaired_trials(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for system in ("Motive", "Captury"):
                (root / system).mkdir()
            for name in ("Static", "Walk", "Squat"):
                (root / "Motive" / f"{name}.c3d").touch()
            for name in ("Static", "Walk", "Jump"):
                (root / "Captury" / f"{name}.c3d").touch()
            self.assertEqual(
                {b.name for b in c.discover_trials(root)},
                {"Static", "Walk", "Squat", "Jump"},
            )

    def test_explicit_gui_selection_is_passed_to_same_command(self):
        args = build_p6_args(
            {
                "p6_source_motive": True,
                "p6_source_biobuddy": True,
                "p6_source_captury": False,
            }
        )
        start = args.index("--sources") + 1
        self.assertEqual(args[start : start + 2], ["motive", "biobuddy"])
        self.assertNotIn("captury", args[start : start + 2])

    def test_source_selection_and_calibration_are_per_trial(self):
        from comparison_workflow import ComparisonWorkflow

        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.out_dir = root
            args.sources = ["captury", "motive"]
            static = self.make_bundle(root, "Static", args.sources)
            walk = self.make_bundle(root, "Walk", args.sources)
            squat = self.make_bundle(root, "Squat", ["motive"])
            api = SimpleNamespace(**vars(c))

            def result(bundle, output, *a, **kw):
                (output / bundle.name).mkdir(parents=True, exist_ok=True)
                return {"trial": bundle.name, "outputs": {}}, {"static": bundle.name}

            api.compare_trial = Mock(side_effect=result)
            with patch(
                "comparison_workflow.compare_available_trial", side_effect=result
            ) as single:
                workflow = ComparisonWorkflow(static, args, api=api)
                a = workflow.process(walk, root)
                b = workflow.process(squat, root)
                workflow.process(squat, root)
            self.assertEqual(a["workflow"], b["workflow"])
            self.assertEqual(a["requested_sources"], ["captury", "motive"])
            self.assertEqual(b["source_selection"]["captury"]["status"], "unavailable")
            self.assertEqual(api.compare_trial.call_count, 2)  # static then walk
            self.assertEqual(single.call_count, 3)  # static once, squat twice

    def test_disabled_biobuddy_never_reaches_processor(self):
        from comparison_workflow import ComparisonWorkflow

        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.sources = ["motive"]
            args.biobuddy_biomod = root / "model.bioMod"
            args.biobuddy_biomod.touch()
            args.out_dir = root
            b = self.make_bundle(root, "Static", ["motive"])
            with patch(
                "comparison_workflow.compare_available_trial",
                return_value=({"trial": "Static"}, {}),
            ) as processor:
                ComparisonWorkflow(b, args, api=c).process(b, root)
            self.assertIsNone(processor.call_args.args[2].biobuddy_biomod)
            self.assertFalse(processor.call_args.args[2].run_ik_batch)

    def test_biobuddy_uses_motive_markers_without_requiring_motive_skeleton(self):
        from comparison_workflow import source_selection

        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.sources = ["biobuddy"]
            args.biobuddy_biomod = root / "model.bioMod"
            args.biobuddy_biomod.touch()
            b = self.make_bundle(root, "Static", ["motive"])
            b.motive_bvh = None
            statuses = source_selection(b, args)
            self.assertEqual(statuses["biobuddy"]["status"], "available")
            self.assertEqual(statuses["motive"]["status"], "not_selected")

    def test_individual_and_two_source_results_clear_previous_selection(self):
        from available_source_results import compare_available_trial
        from tests.test_motive_only_transforms import MotiveOnlyTransformTests
        from gui_graphs import read_table_npz
        import copy
        import json

        fixture = MotiveOnlyTransformTests()
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.out_dir = root
            args.static_trial = "Static"
            args.disable_motive_marker_alignment = True
            args.cut_mode = (
                "movement"  # Static always uses full, including Captury-only.
            )
            base, bio, *_ = fixture.make_runs(root, 0)
            base.root_offset_policy = {"selected_mode": "keep_root_q_as_file"}
            api = SimpleNamespace(**vars(c))
            api.build_model_run_with_rotation_audit = Mock(
                side_effect=lambda bundle, source, *a: (
                    c.replace(base, system=source),
                    {},
                )
            )
            api.run_biobuddy_ik_for_trial = Mock(return_value=(bio, {"status": "ok"}))
            api.detect_trial_events_and_contacts = Mock(return_value=({}, []))
            api.analyze_motive_occlusions = Mock(return_value=([], None))
            api.biobuddy_dimension_rows = Mock(return_value=([], {"status": "ok"}))
            b = self.make_bundle(root, "Static", ("captury", "motive"))
            directory = root / "Static"
            for sources in (
                ["motive", "biobuddy"],
                ["motive"],
                ["biobuddy"],
                ["captury"],
                ["captury", "biobuddy"],
                [],
            ):
                with self.subTest(sources=sources):
                    args.sources = sources
                    directory.mkdir(exist_ok=True)
                    (directory / "temporal_synchronization_timeseries.npz").write_bytes(
                        b"stale"
                    )
                    (directory / "trial_events_contacts.csv").write_text("stale")
                    report, _ = compare_available_trial(b, root, args, api=api)
                    self.assertEqual(set(report["available_sources"]), set(sources))
                    table = read_table_npz(directory / "kinematics_q_timeseries.npz")
                    self.assertEqual(
                        set(table.source) if not table.empty else set(), set(sources)
                    )
                    self.assertFalse(
                        (directory / "temporal_synchronization_timeseries.npz").exists()
                    )
                    if "captury" in sources or not sources:
                        self.assertEqual(
                            (directory / "trial_events_contacts.csv").read_text(), ""
                        )
                    if not sources:
                        self.assertTrue(
                            read_table_npz(
                                directory / "joint_centre_timeseries.npz"
                            ).empty
                        )

    def test_source_with_no_static_is_reported_and_does_not_use_dynamic_fit(self):
        from comparison_workflow import ComparisonWorkflow

        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.sources = ["captury", "motive"]
            args.out_dir = root
            static = self.make_bundle(root, "Static", ["motive"])
            dynamic = self.make_bundle(root, "Walk", ["motive", "captury"])

            def result(bundle, output, *a, **kw):
                (output / bundle.name).mkdir(parents=True, exist_ok=True)
                return {"trial": bundle.name, "available_sources": ["motive"]}, {}

            with patch(
                "comparison_workflow.compare_available_trial", side_effect=result
            ), patch.object(c, "compare_trial") as paired:
                workflow = ComparisonWorkflow(static, args, api=c)
                report = workflow.process(dynamic, root)
            paired.assert_not_called()
            self.assertEqual(
                report["source_selection"]["captury"]["reason"],
                "missing_source_in_static_calibration",
            )
            self.assertTrue(
                workflow.static_reports[0]["trial"].startswith("calibration:motive:")
            )

    def test_occlusions_skip_absent_motive(self):
        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.out_dir = root
            bundle = self.make_bundle(root, "Jump", ["captury"])
            with patch.object(c, "analyze_motive_occlusions") as analyze:
                self.assertEqual(c.run_occlusions_only([bundle], args), [])
            analyze.assert_not_called()

    def test_empty_selection_does_not_launch_auto_analysis(self):
        from captury_biobuddy_gui import CapturyBioBuddyGui

        gui = SimpleNamespace(
            vars={
                f"p6_source_{s}": SimpleNamespace(get=lambda: False)
                for s in ("captury", "motive", "biobuddy")
            },
            pending_auto_analysis=True,
            status_var=Mock(),
        )
        CapturyBioBuddyGui._run_selected_trial_auto_analysis(gui)
        self.assertFalse(gui.pending_auto_analysis)
        gui.status_var.set.assert_called_once()

    def test_three_sources_use_the_same_workflow_with_paired_calibration(self):
        from comparison_workflow import ComparisonWorkflow

        with tempfile.TemporaryDirectory() as tmp, patch("sys.argv", ["compare"]):
            root = Path(tmp)
            args = c.parse_args()
            args.out_dir = root
            args.biobuddy_biomod = root / "model.bioMod"
            args.biobuddy_biomod.touch()
            bundle = self.make_bundle(root, "Static", ["captury", "motive"])
            api = SimpleNamespace(**vars(c))
            api.compare_trial = Mock(
                return_value=(
                    {"trial": "Static", "biobuddy_ik_batch": {"status": "ok"}},
                    {},
                )
            )
            with patch("comparison_workflow.compare_available_trial") as partial:
                report = ComparisonWorkflow(bundle, args, api=api).process(bundle, root)
            partial.assert_not_called()
            self.assertEqual(
                set(report["available_sources"]), {"captury", "motive", "biobuddy"}
            )
            self.assertEqual(report["workflow"], "source_comparison")
            self.assertEqual(
                api.compare_trial.call_args.args[2].biobuddy_biomod,
                args.biobuddy_biomod,
            )
