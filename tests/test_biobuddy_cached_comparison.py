import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import compare_p6_motive_captury as comparison
from run_biobuddy_c3d_ik import ANGLE_LABEL_REGEX


class CachedBioBuddyComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / "model.bioMod"
        self.c3d = self.root / "trial.c3d"
        self.model.write_text("model")
        self.c3d.write_text("markers")
        self.bundle = comparison.TrialBundle(
            "Trial", self.c3d, None, None, self.c3d, None, None
        )

    def fingerprint(self):
        return comparison.biobuddy_ik_cache_fingerprint(
            self.model, self.c3d, 1.0, 0, ANGLE_LABEL_REGEX
        )

    def write_cache(self):
        fingerprint = self.fingerprint()
        directory = self.root / "cache" / fingerprint["digest"]
        directory.mkdir(parents=True)
        npz = directory / "inverse_kinematics_trf.npz"
        np.savez(npz, time=[0.0, 0.01], q=[[0.1, 0.2]], q_names=["Pelvis_rotX"])
        report = {
            "status": "ok",
            "cache": fingerprint,
            "outputs": {"npz": str(npz)},
            "artifacts": {"npz_sha256": comparison.file_sha256(npz)},
        }
        (directory / "inverse_kinematics_trf_summary.json").write_text(
            json.dumps(report)
        )
        return npz

    def load(self):
        return comparison.run_biobuddy_ik_for_trial(
            self.bundle,
            self.root / "trial",
            self.model,
            1.0,
            0,
            ANGLE_LABEL_REGEX,
            self.root / "cache",
            cache_only=True,
        )

    def test_reuses_valid_ik_without_running_solver(self):
        npz = self.write_cache()
        with patch.object(comparison, "run_direct_biobuddy_ik") as solve, patch.object(
            comparison, "model_run_from_biobuddy_ik_npz", return_value="loaded"
        ) as load:
            run, report = self.load()
        solve.assert_not_called()
        load.assert_called_once_with(self.model, npz, unit_scale_to_m=1.0)
        self.assertEqual(run, "loaded")
        self.assertTrue(report["cache"]["hit"])

    def test_missing_stale_or_corrupt_cache_never_runs_solver(self):
        with patch.object(comparison, "run_direct_biobuddy_ik") as solve:
            self.assertIsNone(self.load()[0])
            npz = self.write_cache()
            self.model.write_text("changed model")
            self.assertIsNone(self.load()[0])
            self.model.write_text("model")
            npz.write_bytes(b"corrupt")
            self.assertIsNone(self.load()[0])
        solve.assert_not_called()

    def test_paired_report_keeps_cached_ik_when_solver_is_disabled(self):
        # Execute the report-publication block from the paired adapter without
        # running numerical calibration: the cached source must remain reported.
        import ast
        import inspect
        from types import SimpleNamespace

        function = ast.parse(inspect.getsource(comparison.compare_trial)).body[0]
        blocks = [
            statement
            for statement in function.body
            if any(
                isinstance(child, ast.Assign)
                and any(
                    ast.unparse(target) == "report['biobuddy_ik_batch']"
                    for target in child.targets
                )
                for child in ast.walk(statement)
            )
        ]
        self.assertEqual(len(blocks), 1)
        report = {}
        cached = {
            "status": "ok",
            "cache": {"hit": True},
            "outputs": {"npz": "/tmp/ik.npz", "summary": "/tmp/ik.json"},
        }
        namespace = {
            "args": SimpleNamespace(run_ik_batch=False),
            "report": report,
            "biobuddy_ik_report": cached,
        }
        exec(
            compile(ast.Module(body=blocks, type_ignores=[]), "paired_report", "exec"),
            namespace,
        )
        self.assertEqual(report.get("biobuddy_ik_batch"), cached)
        artifacts = comparison.comparison_derived_artifacts(
            [dict(report, trial="Walk")]
        )
        self.assertEqual(artifacts["Walk/biobuddy/ik_npz"], Path("/tmp/ik.npz"))

    def test_trial_cache_changes_when_ik_becomes_available(self):
        with patch(
            "sys.argv",
            [
                "compare",
                "--data-root",
                str(self.root),
                "--out-dir",
                str(self.root),
                "--biobuddy-biomod",
                str(self.model),
            ],
        ):
            args = comparison.parse_args()
        # The comparison uses this fixed subdirectory for cached IK.
        (self.root / "biobuddy_ik_cache").symlink_to(
            self.root / "cache", target_is_directory=True
        )
        before = comparison.trial_cache_fingerprint(self.bundle, args)
        self.write_cache()
        after = comparison.trial_cache_fingerprint(self.bundle, args)
        self.assertNotEqual(before["digest"], after["digest"])


if __name__ == "__main__":
    unittest.main()
