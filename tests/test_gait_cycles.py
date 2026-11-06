import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gait_cycles import (
    cycles_from_contacts,
    cycles_from_event_csv,
    lower_limb_channels,
    normalize_cycle,
    export_gait_results,
)


def contact_rows():
    times = np.arange(0.0, 4.01, 0.01)
    rows = []
    for t in times:
        row = {"time": float(t)}
        for side, offset in (("left", 0.3), ("right", 0.8)):
            row[f"{side}_contact"] = any(
                start <= t < start + 0.3 for start in np.arange(offset, 4.0, 1.0)
            )
            row[f"{side}_contact_available"] = True
            row[f"{side}_foot_z_mm"] = 0.0
            row[f"{side}_foot_speed_mm_s"] = 0.0
        rows.append(row)
    return rows


class GaitCycleTests(unittest.TestCase):
    def test_multiple_same_foot_cycles_and_exact_normalization(self):
        report = cycles_from_contacts(contact_rows())
        cycles = report["cycles"]
        self.assertEqual(len(cycles), 6)
        self.assertEqual({c["side"] for c in cycles}, {"left", "right"})
        t = np.arange(0.0, 4.01, 0.01)
        values, status = normalize_cycle(t, 2 * t, cycles[0], n_points=101)
        self.assertEqual(status, "ok")
        self.assertEqual(len(values), 101)
        self.assertAlmostEqual(values[0], 2 * cycles[0]["start_s"])
        self.assertAlmostEqual(values[-1], 2 * cycles[0]["end_s"])

    def test_boundary_contact_is_not_an_observed_initial_contact(self):
        rows = contact_rows()
        for row in rows[:50]:
            row["left_contact"] = True
        report = cycles_from_contacts(rows)
        self.assertTrue(all(c["start_s"] > 0.0 for c in report["cycles"]))

    def test_contact_chatter_is_merged_but_occlusion_is_not(self):
        rows = contact_rows()
        for row in rows[40:43]:
            row["left_contact"] = False
        self.assertEqual(len(cycles_from_contacts(rows)["cycles"]), 6)
        for row in rows[80:90]:
            row["left_foot_z_mm"] = np.nan
        report = cycles_from_contacts(rows)
        self.assertEqual(len([c for c in report["cycles"] if c["side"] == "left"]), 2)
        self.assertTrue(
            any(
                c["reason"] == "missing_contact_data" for c in report["rejected_cycles"]
            )
        )

    def test_selected_window_keeps_only_complete_cycles(self):
        report = cycles_from_contacts(contact_rows(), start_s=0.4, end_s=2.5)
        self.assertTrue(
            all(c["start_s"] >= 0.4 and c["end_s"] <= 2.5 for c in report["cycles"])
        )

    def test_signal_gap_and_missing_endpoint_are_not_filled(self):
        t = np.arange(0.0, 1.01, 0.01)
        cycle = {"start_s": 0.0, "end_s": 1.0}
        y = t.copy()
        y[40:60] = np.nan
        values, status = normalize_cycle(t, y, cycle)
        self.assertEqual(status, "gap_exceeds_limit")
        self.assertTrue(np.isnan(values).all())
        values, status = normalize_cycle(t[10:], t[10:], cycle)
        self.assertEqual(status, "outside_source_coverage")

    def test_rotation_wrap_uses_continuous_angle_branch(self):
        t = np.linspace(0.0, 1.0, 101)
        y = np.linspace(179.0, 181.0, 101)
        y[y > 180.0] -= 360.0
        values, status = normalize_cycle(
            t, y, {"start_s": 0.0, "end_s": 1.0}, angular=True
        )
        self.assertEqual(status, "ok")
        np.testing.assert_allclose(values, np.linspace(179.0, 181.0, 101), atol=1e-10)

    def test_manual_events_validate_side_duplicates_and_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            path.write_text(
                "trial,side,time_s\nWalk,left,0\nWalk,left,1\nOther,left,2\n"
            )
            report = cycles_from_event_csv(path, "Walk", start_s=0, end_s=1)
            self.assertEqual(len(report["cycles"]), 1)
            path.write_text("side,time_s\nleft,0\nleft,0\n")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                cycles_from_event_csv(path, "Walk")
            path.write_text("side,time_s\nunknown,0\n")
            with self.assertRaises(ValueError):
                cycles_from_event_csv(path, "Walk")

    def test_all_available_dofs_and_units_preserved_by_source(self):
        run = SimpleNamespace(
            system="biobuddy",
            source_kind="motive_57_ik",
            q_names=[
                "LThigh_RotX",
                "LShank_RotZ",
                "LFoot_RotZ",
                "LFoot_RotX",
                "LFoot_TransX",
                "Head_RotX",
            ],
            q_units=["rad"] * 4 + ["native", "rad"],
            time=np.array([0.0, 0.01]),
            q=np.ones((6, 2)),
            unit_scale_to_m=0.01,
        )
        channels = lower_limb_channels([run])
        self.assertEqual(len(channels), 5)
        self.assertEqual(len([c for c in channels if c["joint"] == "ankle"]), 3)
        np.testing.assert_allclose(channels[0]["values"], np.degrees([1.0, 1.0]))
        np.testing.assert_allclose(channels[-1]["values"], [10.0, 10.0])
        self.assertEqual(channels[-1]["unit"], "mm")


if __name__ == "__main__":
    unittest.main()
