from __future__ import annotations

import math
import hashlib
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze_m9 import (
    analyze,
    _controller_target_at,
    _expected_object_outcome_failures,
    _object_outcomes_accounted,
    _placement_confirmation_failures,
    _input_snapshot_sha256,
    _motion_commands_for_skipped_tracks,
    _perception_requirement_failures,
    _write_csv,
    classification_metrics,
    classify_contact_pair,
    wilson_interval,
)


class M9AnalysisTests(unittest.TestCase):
    def test_analyzer_indexes_run_by_registered_trial_id_not_run_folder_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_path = root / "campaign.yaml"
            campaign_path.write_text(
                "campaign_id: test-campaign\n"
                "primary_acceptance:\n"
                "  minimum_complete_batch_rate: 0.9\n"
                "  required_seeds: 1\n"
                "  minimum_complete_batches: 1\n",
                encoding="utf-8",
            )
            run_dir = root / "runs" / "scenario-generated-folder-1234"
            run_dir.mkdir(parents=True)
            (run_dir / "campaign_trial.json").write_text(
                json.dumps({"trial_id": "registered-trial-1"}), encoding="utf-8"
            )
            job = {
                "trial_id": "registered-trial-1",
                "experiment_id": "FIXED_CASE",
                "variant_id": "fixed",
                "scenario": "one-object",
                "expected_kind": "safe_completion",
                "tags": [],
            }
            metric = {
                "trial_id": "registered-trial-1",
                "experiment_id": "FIXED_CASE",
                "variant_id": "fixed",
                "test_pass": True,
                "wrong_bin_count": 0,
                "forbidden_contact_episodes": 0,
                "position_limit_violation_samples": 0,
                "force_limit_violation_samples": 0,
                "simulation_time_s": 0.02,
                "active_objects": 0,
                "sorted_correctly": 0,
                "_object_records": [],
            }
            tables, plots = root / "tables", root / "plots"
            legacy_tables, legacy_plots = root / "legacy-tables", root / "legacy-plots"
            with (
                patch("analyze_m9._expand_campaign_jobs", return_value=[job]),
                patch("analyze_m9._verify_trial_integrity"),
                patch("analyze_m9._trial_metrics", return_value=(metric, [])),
                patch("analyze_m9._series_summary", return_value=[]),
                patch("analyze_m9._paired_comparisons", return_value=([], [])),
                patch("analyze_m9._draw_rate_plot"),
                patch("analyze_m9._draw_confusion_matrix"),
                patch("analyze_m9.TABLES_ROOT", legacy_tables),
                patch("analyze_m9.PLOTS_ROOT", legacy_plots),
                patch("analyze_m9.RESULTS_PATH", root / "results.json"),
            ):
                result = analyze(
                    campaign_path,
                    root / "runs",
                    tables_root=tables,
                    plots_root=plots,
                )
            self.assertEqual(result["analyzed_trials"], 1)
            self.assertEqual(result["missing_trials"], [])
            self.assertTrue((tables / "Результаты_по_прогонам.csv").is_file())
            self.assertTrue(plots.is_dir())
            self.assertFalse(legacy_tables.exists())
            self.assertFalse(legacy_plots.exists())

    def test_csv_serializes_nested_values_as_deterministic_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            _write_csv(path, [{"checks": {"z": [1, 2], "a": True}}])
            with path.open(encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(json.loads(row["checks"]), {"a": True, "z": [1, 2]})
            self.assertEqual(row["checks"], '{"a":true,"z":[1,2]}')

    def test_preregistered_perception_checks_are_enforced(self) -> None:
        requirements = {
            "first_frame_detection_count": 2,
            "first_frame_merged_candidate_split_count": 2,
            "first_frame_correct_classification_count": 2,
        }
        observed = dict(requirements)
        self.assertEqual(_perception_requirement_failures(requirements, observed), [])
        observed["first_frame_merged_candidate_split_count"] = 1
        self.assertEqual(
            _perception_requirement_failures(requirements, observed),
            ["first_frame_merged_candidate_split_count: expected 2, observed 1"],
        )
        with self.assertRaises(ValueError):
            _perception_requirement_failures({"hidden_truth": 1}, observed)

    def test_integrated_mjcf_hash_is_line_ending_independent_but_other_inputs_are_not(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.xml"
            path.write_bytes(b"<mujoco>\r\n<body/>\r\n</mujoco>\r\n")
            canonical = _input_snapshot_sha256("m7_integrated_model", path)
            expected = hashlib.sha256(b"<mujoco>\n<body/>\n</mujoco>\n").hexdigest()
            self.assertEqual(canonical, expected)
            self.assertNotEqual(canonical, _input_snapshot_sha256("mjcf", path))

    def test_wilson_interval_uses_independent_run_counts(self) -> None:
        lower, upper = wilson_interval(10, 10)
        self.assertAlmostEqual(lower, 0.7224672, places=6)
        self.assertAlmostEqual(upper, 1.0, places=15)
        empty = wilson_interval(0, 0)
        self.assertTrue(math.isnan(empty[0]))
        with self.assertRaises(ValueError):
            wilson_interval(11, 10)

    def test_classification_metrics_count_unknown_missing_and_false_positives(self) -> None:
        metrics = classification_metrics({
            ("RED", "RED"): 7,
            ("RED", "UNKNOWN"): 1,
            ("RED", "MISSING"): 1,
            ("GREEN", "RED"): 1,
            ("NO_OBJECT", "RED"): 1,
        })
        red = next(row for row in metrics if row["class_label"] == "RED")
        self.assertEqual(red["support_objects"], 9)
        self.assertEqual(red["predicted_objects_including_false_positives"], 9)
        self.assertAlmostEqual(red["precision"], 7 / 9)
        self.assertAlmostEqual(red["recall"], 7 / 9)
        self.assertEqual(red["unknown_abstentions"], 1)
        self.assertEqual(red["missing_detections"], 1)

    def test_contact_classifier_only_allows_current_target_during_grasp_cycle(self) -> None:
        self.assertEqual(
            classify_contact_pair("object_red_01_collision", "table_top_collision"),
            "object_support",
        )
        self.assertEqual(
            classify_contact_pair("tray_RED_floor_collision", "object_red_01_collision"),
            "tray_support",
        )
        self.assertEqual(
            classify_contact_pair("finger_left_collision", "object_red_01_collision",
                                  target_body="object_red_01", controller_state="GRASP"),
            "intended_grasp",
        )
        self.assertEqual(
            classify_contact_pair("finger_left_collision", "object_green_01_collision",
                                  target_body="object_red_01", controller_state="GRASP"),
            "forbidden",
        )
        self.assertEqual(
            classify_contact_pair("finger_right_collision", "object_red_01_collision",
                                  target_body="object_red_01", controller_state="APPROACH"),
            "forbidden",
        )
        self.assertEqual(
            classify_contact_pair("object_red_01_collision", "object_blue_01_collision"),
            "forbidden",
        )

    def test_contact_time_is_associated_with_public_track_and_cycle(self) -> None:
        events = [
            {"event_id": 1, "event_type": "TARGET_SELECTED", "simulation_time_s": 0.1,
             "track_id": 7},
            {"event_id": 2, "event_type": "STATE_ENTERED", "simulation_time_s": 0.2,
             "state": "DESCEND"},
            {"event_id": 3, "event_type": "STATE_ENTERED", "simulation_time_s": 0.5,
             "state": "GRASP"},
            {"event_id": 4, "event_type": "CYCLE_CLOSED", "simulation_time_s": 1.0},
        ]
        mapping = {"7": "object_red_01"}
        self.assertEqual(_controller_target_at(events, mapping, 0.7), ("object_red_01", "GRASP"))
        self.assertEqual(_controller_target_at(events, mapping, 1.1), (None, "GRASP"))

    def test_confirmation_validation_uses_exact_track_to_object_association(self) -> None:
        tracks = {
            "1": {"status": "PLACED_CONTROLLER_CONFIRMED", "class_label": "RED"},
            "2": {"status": "PLACED_CONTROLLER_CONFIRMED", "class_label": "RED"},
        }
        objects = [
            {"track_id": 1, "body_name": "object_red_01", "ground_truth_class": "RED",
             "physical_sort_success": True},
            {"track_id": 2, "body_name": "object_red_02", "ground_truth_class": "RED",
             "physical_sort_success": False},
        ]
        self.assertEqual(_placement_confirmation_failures(tracks, objects), ["2"])

    def test_safe_completion_requires_sort_or_reason_coded_skip(self) -> None:
        records = [
            {"ground_truth_class": "RED", "physical_sort_success": True,
             "controller_status": "PLACED_CONTROLLER_CONFIRMED", "controller_class": "RED"},
            {"ground_truth_class": "BLUE", "physical_sort_success": False,
             "controller_status": "SKIPPED_UNREACHABLE", "controller_class": "BLUE",
             "controller_failure_reasons": "COLLISION_GRASP"},
        ]
        self.assertTrue(_object_outcomes_accounted(records))
        records[1]["controller_failure_reasons"] = ""
        self.assertFalse(_object_outcomes_accounted(records))
        self.assertEqual(_expected_object_outcome_failures(
            {"object_red_01": "SORTED", "object_blue_01": "SKIPPED_UNREACHABLE"},
            [
                {**records[0], "body_name": "object_red_01"},
                {**records[1], "body_name": "object_blue_01",
                 "controller_failure_reasons": "COLLISION_GRASP"},
            ],
        ), [])

    def test_mixed_partial_does_not_move_on_skipped_track_cycle(self) -> None:
        tracks = {"11": {"status": "SKIPPED_UNREACHABLE"}}
        events = [
            {"event_id": 1, "event_type": "TARGET_SELECTED", "track_id": 11,
             "simulation_time_s": 0.1},
            {"event_id": 2, "event_type": "PLAN_REJECTED", "track_id": 11,
             "simulation_time_s": 0.2},
            {"event_id": 3, "event_type": "STATE_ENTERED", "state": "OBSERVE",
             "simulation_time_s": 0.3},
            {"event_id": 4, "event_type": "TARGET_SELECTED", "track_id": 12,
             "simulation_time_s": 0.4},
            {"event_id": 5, "event_type": "EXECUTION_COMMAND_ISSUED", "track_id": 12,
             "simulation_time_s": 0.5},
        ]
        self.assertEqual(_motion_commands_for_skipped_tracks(tracks, events), 0)
        events.insert(3, {"event_id": 6, "event_type": "EXECUTION_COMMAND_ISSUED",
                          "track_id": 11, "simulation_time_s": 0.35})
        self.assertEqual(_motion_commands_for_skipped_tracks(tracks, events), 1)


if __name__ == "__main__":
    unittest.main()
