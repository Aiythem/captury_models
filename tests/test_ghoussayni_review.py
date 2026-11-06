"""Independent gap and event-sequence checks for Ghoussayni normalization."""

import unittest

import numpy as np

from gait_cycles import cycles_from_contacts
from ghoussayni import METHOD, detect_ghoussayni
from tests.test_ghoussayni import walking_markers


class GhoussayniReviewTests(unittest.TestCase):
    def test_timestamp_hole_rejects_crossing_cycle_without_recovery_event(self):
        time, labels, points = walking_markers()
        keep = ~((time >= 0.8) & (time <= 1.0))
        report, rows = detect_ghoussayni(time[keep], labels, points[:, :, keep])
        cycles = cycles_from_contacts(rows)
        self.assertFalse(
            any(0.8 <= t <= 1.02 for t in report["initial_contacts_s"]["left"])
        )
        self.assertTrue(
            any(
                c["side"] == "left" and c["reason"] == "missing_contact_data"
                for c in cycles["rejected_cycles"]
            )
        )
        self.assertTrue(
            all(not (c["start_s"] < 0.8 < c["end_s"]) for c in cycles["cycles"])
        )

    def test_pelvis_occlusion_invalidates_both_sides(self):
        time, labels, points = walking_markers()
        gap = (time >= 1.8) & (time <= 2.0)
        points[:, 0, gap] = np.nan
        _, rows = detect_ghoussayni(time, labels, points)
        for side in ("left", "right"):
            self.assertTrue(
                all(
                    not rows[i][f"{side}_contact_available"]
                    for i in np.flatnonzero(gap)
                )
            )
        report = cycles_from_contacts(rows)
        self.assertEqual(
            {
                c["side"]
                for c in report["rejected_cycles"]
                if c["reason"] == "missing_contact_data"
            },
            {"left", "right"},
        )

    def test_exactly_one_strictly_interior_toe_off_is_required(self):
        time = np.arange(0, 2.01, 0.1)
        for off_indices, accepted in (
            ([], False),
            ([5], True),
            ([5, 7], False),
            ([0], False),
            ([10], False),
        ):
            with self.subTest(off_indices=off_indices):
                rows = [
                    {
                        "time": float(t),
                        "event_method": METHOD,
                        "left_contact_available": True,
                        "left_initial_contact": i in (0, 10),
                        "left_toe_off": i in off_indices,
                    }
                    for i, t in enumerate(time)
                ]
                report = cycles_from_contacts(rows)
                self.assertEqual(len(report["cycles"]), int(accepted))
                if accepted:
                    self.assertEqual(report["cycles"][0]["toe_off_percent"], 50.0)
                else:
                    self.assertEqual(
                        report["rejected_cycles"][0]["reason"],
                        "missing_or_multiple_toe_offs",
                    )
                    self.assertEqual(report["status"], "no_complete_cycle")

    def test_irregular_observation_spacing_is_rejected(self):
        time, labels, points = walking_markers()
        time[50] += 0.001
        with self.assertRaisesRegex(ValueError, "uniformly sampled"):
            detect_ghoussayni(time, labels, points)


if __name__ == "__main__":
    unittest.main()
