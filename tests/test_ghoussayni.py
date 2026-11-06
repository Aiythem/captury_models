"""Known sagittal velocities, independent of joint-angle waveform shape."""

import unittest

import numpy as np

from ghoussayni import detect_ghoussayni
from gait_cycles import cycles_from_contacts


def walking_markers():
    time = np.arange(0, 4.001, 0.005)
    labels = ["LIAS", "RIAS", "LIPS", "RIPS", "LFCC", "LFM2", "RFCC", "RFM2"]
    points = np.zeros((3, len(labels), len(time)))
    for i, (x, y) in enumerate(((100, 100), (100, -100), (-100, 100), (-100, -100))):
        points[0, i] = 1000 * time + x
        points[1, i] = y
        points[2, i] = 900
    for i, phase in enumerate((0, 0.2, 0.5, 0.7), 4):
        points[0, i] = 800 * time + 700 / (2 * np.pi) * np.sin(
            2 * np.pi * (time - phase)
        )
        # Large lateral velocity must not enter sagittal speed.
        points[1, i] = 2000 * time
        points[2, i] = 100
    return time, labels, points


class GhoussayniTests(unittest.TestCase):
    def test_vertical_velocity_counts_in_sagittal_speed(self):
        time, labels, points = walking_markers()
        points[2, 4:] += 600 * time
        report, _ = detect_ghoussayni(time, labels, points)
        self.assertEqual(report["initial_contacts_s"], {"left": [], "right": []})

    def test_known_heel_and_toe_crossings_and_complete_cycles(self):
        time, labels, points = walking_markers()
        report, rows = detect_ghoussayni(time, labels, points)
        expected = np.arccos((500 - 800) / 700) / (2 * np.pi)
        np.testing.assert_allclose(
            report["initial_contacts_s"]["left"], expected + np.arange(4), atol=0.01
        )
        # The last event includes the known zero-phase filter endpoint transient.
        np.testing.assert_allclose(
            report["toe_offs_s"]["left"], 1 - expected + 0.2 + np.arange(4), atol=0.015
        )
        cycles = cycles_from_contacts(rows)
        self.assertEqual(len(cycles["cycles"]), 6)
        self.assertIn("ghoussayni", cycles["method"])
        for cycle in cycles["cycles"]:
            self.assertAlmostEqual(cycle["duration_s"], 1.0, places=2)
            self.assertTrue(cycle["start_s"] < cycle["toe_off_s"] < cycle["end_s"])
        self.assertEqual(report["velocity_threshold_m_s"], 0.5)

    def test_yaw_and_translation_do_not_change_events(self):
        time, labels, points = walking_markers()
        original, _ = detect_ghoussayni(time, labels, points)
        angle = 0.83
        rot = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        transformed = (
            np.einsum("ij,jkt->ikt", rot, points)
            + np.array([500, -900, 250])[:, None, None]
        )
        changed, _ = detect_ghoussayni(time, labels, transformed)
        self.assertEqual(original["initial_contacts_s"], changed["initial_contacts_s"])

    def test_occlusion_does_not_create_contact_or_bridge_cycles(self):
        time, labels, points = walking_markers()
        points[:, 4, (time > 0.8) & (time < 1.0)] = np.nan
        _, rows = detect_ghoussayni(time, labels, points)
        report = cycles_from_contacts(rows)
        self.assertTrue(
            any(
                c["reason"] == "missing_contact_data" for c in report["rejected_cycles"]
            )
        )
        self.assertFalse(
            any(0.8 < t < 1.1 for t in report["initial_contacts_s"]["left"])
        )

    def test_missing_or_ambiguous_marker_never_uses_average_foot(self):
        time, labels, points = walking_markers()
        labels[4] = "LFM5"
        report, rows = detect_ghoussayni(time, labels, points)
        self.assertFalse(report["sides"]["left"]["available"])
        self.assertEqual(report["initial_contacts_s"]["left"], [])
        self.assertTrue(report["sides"]["right"]["available"])
        labels[4] = "LFCC"
        labels.append("LFCC")
        report, _ = detect_ghoussayni(
            time, labels, np.concatenate((points, points[:, 4:5]), axis=1)
        )
        self.assertFalse(report["sides"]["left"]["available"])

    def test_stationary_recording_has_no_invented_initial_contact(self):
        time, labels, points = walking_markers()
        points[:] = points[:, :, :1]
        report, rows = detect_ghoussayni(time, labels, points)
        self.assertEqual(report["initial_contacts_s"], {"left": [], "right": []})
        self.assertEqual(cycles_from_contacts(rows)["cycles"], [])

    def test_bad_threshold_and_cutoff_are_rejected(self):
        time, labels, points = walking_markers()
        for kwargs in (
            {"threshold_m_s": 0},
            {"threshold_m_s": np.nan},
            {"cutoff_hz": 120},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                detect_ghoussayni(time, labels, points, **kwargs)


if __name__ == "__main__":
    unittest.main()
