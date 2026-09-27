from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from analyze_m9 import _expand_campaign_jobs, _scenario_matrix_rows
from run_m9_campaign import _expand_jobs


class M9CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[3]
        cls.campaign = yaml.safe_load(
            (root / "07_Испытания" / "campaign.yaml").read_text(encoding="utf-8")
        )

    def test_campaign_has_98_unique_preregistered_trials(self) -> None:
        executed = _expand_jobs(self.campaign)
        analyzed = _expand_campaign_jobs(self.campaign)
        self.assertEqual(len(executed), 98)
        self.assertEqual(len({job["trial_id"] for job in executed}), 98)
        self.assertEqual(
            [job["trial_id"] for job in executed],
            [job["trial_id"] for job in analyzed],
        )
        six_object = next(job for job in executed if job["trial_id"] == "v02_two_per_color")
        self.assertEqual(six_object["scenario"], "m9_six_object_grid")
        outer_trials = {job["trial_id"] for job in executed if "V12" in job["tags"]}
        self.assertTrue({"v12_outer_left", "v12_outer_right"}.issubset(outer_trials))
        matrix_rows = _scenario_matrix_rows(self.campaign)
        self.assertEqual(len(matrix_rows), 98)
        self.assertEqual([row["trial_id"] for row in matrix_rows],
                         [job["trial_id"] for job in executed])
        adjacent = next(job for job in executed if job["trial_id"] == "v08_adjacent_pair")
        self.assertEqual(adjacent["expected_kind"], "safe_skip")
        self.assertEqual(adjacent["expected_track_status"], "SKIPPED_UNREACHABLE")
        self.assertEqual(adjacent["expected_failure_reason"], "BRANCH_DISCONTINUITY")
        self.assertEqual(adjacent["required_perception_checks"][
            "first_frame_merged_candidate_split_count"], 2)

    def test_seeded_series_have_paired_controls_and_frozen_expected_outcomes(self) -> None:
        jobs = _expand_jobs(self.campaign)
        baseline = [job for job in jobs if job["experiment_id"] == "V03_RANDOM_MIXED"]
        friction = [job for job in jobs if job["experiment_id"] == "V10_GRIP_FRICTION"]
        self.assertEqual(len(baseline), 30)
        self.assertEqual(len(friction), 10)
        self.assertEqual({job["seed"] for job in friction},
                         set(self.campaign["seed_sets"]["paired_sensor_and_friction"]))
        blocked = next(job for job in jobs if job["trial_id"] == "v05_blocked_plus_reachable")
        self.assertEqual(blocked["expected_object_outcomes"]["object_red_02"], "SORTED")
        self.assertEqual(blocked["expected_object_outcomes"]["object_green_01"],
                         "SKIPPED_UNREACHABLE")


if __name__ == "__main__":
    unittest.main()
