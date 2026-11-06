"""Independent boundary and rejection contracts for full gait normalization."""

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from compare_p6_motive_captury import write_table_npz
from gait_cycles import cycles_from_contacts, export_gait_results, normalize_cycle
from gui_graphs import read_table_npz


def observed_contacts():
    rows = []
    for time in np.arange(0.0, 4.01, 0.01):
        row = {"time": float(time)}
        for side in ("left", "right"):
            row.update(
                {
                    f"{side}_contact_available": True,
                    f"{side}_foot_z_mm": 0.0,
                    f"{side}_foot_speed_mm_s": 0.0,
                    f"{side}_contact": any(
                        start <= time < start + 0.3 for start in (0.3, 1.3, 2.3, 3.3)
                    ),
                }
            )
        rows.append(row)
    return rows


class GaitCycleReviewTests(unittest.TestCase):
    def test_contact_exactly_at_window_start_retains_observed_onset(self):
        report = cycles_from_contacts(observed_contacts(), start_s=0.3, end_s=1.3)
        self.assertEqual(len(report["cycles"]), 2)
        for cycle in report["cycles"]:
            self.assertAlmostEqual(cycle["start_s"], 0.3)
            self.assertAlmostEqual(cycle["end_s"], 1.3)

    def test_occlusion_recovery_is_not_a_new_contact(self):
        rows = observed_contacts()
        # The true onset at .3 s is hidden; the first visible contact at .4 s
        # must not be exported as a replacement initial contact.
        for row in rows:
            if 0.25 <= row["time"] < 0.4:
                row["left_foot_speed_mm_s"] = np.nan
        report = cycles_from_contacts(rows)
        self.assertNotIn(0.4, report["initial_contacts_s"]["left"])
        self.assertTrue(
            all(c["start_s"] >= 1.3 for c in report["cycles"] if c["side"] == "left")
        )

    def test_missing_contact_timestamps_reject_cross_gap_cycle(self):
        rows = [row for row in observed_contacts() if not 0.8 <= row["time"] <= 0.95]
        report = cycles_from_contacts(rows)
        rejected = report["rejected_cycles"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            all(row["reason"] == "missing_contact_data" for row in rejected)
        )

    def test_invalid_source_clocks_fail_instead_of_silently_reordering(self):
        cycle = {"start_s": 0.0, "end_s": 1.0}
        for times in ([0.0, 0.0, 1.0], [0.0, 1.0, 0.5], [0.0, np.nan, 1.0]):
            with self.subTest(time=times), self.assertRaisesRegex(
                ValueError, "strictly increasing"
            ):
                normalize_cycle(times, [0.0, 0.5, 1.0], cycle)

    def test_exports_share_cycle_clock_and_preserve_unavailable_curve_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = root / "events.csv"
            events.write_text("side,time_s\nleft,0.5\nleft,1.5\n")
            runs = []
            for source, dt, name in (
                ("motive", 0.01, "LeftUpLeg_Xrotation"),
                ("captury", 0.02, "LeftUpLeg_Xrotation"),
                ("biobuddy", 0.01, "LThigh_RotX"),
            ):
                native_time = np.arange(0.0, 2.01, dt)
                # Captury has already been placed on the Motive clock by the
                # pipeline; normalization must not introduce another offset.
                time = native_time + (0.1 if source == "captury" else 0.0)
                values = np.radians(10.0 * time)
                if source == "biobuddy":
                    values[(time >= 0.9) & (time <= 1.1)] = np.nan
                runs.append(
                    SimpleNamespace(
                        system=source,
                        source_kind="motive_57_ik" if source == "biobuddy" else "bvh",
                        q_names=[name],
                        q_units=["rad"],
                        q=values[None, :],
                        time=time,
                        unit_scale_to_m=1.0,
                    )
                )
            report, outputs = export_gait_results(
                root,
                "Walk",
                runs,
                [],
                enabled=True,
                events_csv=events,
                participant="P16",
                write_npz=write_table_npz,
                export_csv=True,
            )
            self.assertEqual(report["valid_curve_cycles"], 2)
            self.assertEqual(report["unavailable_curve_cycles"], 1)
            dataframe = read_table_npz(outputs["gait_cycle_timeseries"])
            self.assertEqual(len(dataframe), 303)
            for source, values in dataframe.groupby("source"):
                self.assertEqual(set(values.cycle_id), {"left_001"})
                np.testing.assert_allclose(values.phase_percent, np.arange(101))
                np.testing.assert_allclose(values.time_s, np.linspace(0.5, 1.5, 101))
                self.assertEqual(set(values.unit), {"deg"})
                if source == "biobuddy":
                    self.assertTrue(values.value.isna().all())
                    self.assertEqual(
                        set(values.normalization_status), {"gap_exceeds_limit"}
                    )
                else:
                    np.testing.assert_allclose(
                        values.value, np.linspace(5.0, 15.0, 101)
                    )
            with Path(outputs["hip_gait_cycles"]).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 303)
            self.assertEqual(
                {row["normalization_status"] for row in rows},
                {"ok", "gap_exceeds_limit"},
            )
            with Path(outputs["knee_gait_cycles"]).open(newline="") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])


if __name__ == "__main__":
    unittest.main()
