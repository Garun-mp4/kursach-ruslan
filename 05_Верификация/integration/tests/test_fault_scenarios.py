from __future__ import annotations

import sys
import json
import math
import tempfile
import unittest
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04_Программа"))

from integration.runner import _disable_gripper_actuator, _load_model, _run_one  # noqa: E402
from integration.scene import load_scenario  # noqa: E402


class FaultScenarioIntegrationTests(unittest.TestCase):
    def run_scenario(self, scenario: str, **faults):
        temporary = tempfile.TemporaryDirectory(prefix="m7-fault-")
        self.addCleanup(temporary.cleanup)
        manifest, report = _run_one(
            scenario_name=scenario,
            mode="batch",
            out_root=Path(temporary.name),
            profile="nominal",
            **faults,
        )
        return manifest, report, Path(temporary.name) / manifest["run_id"]

    def test_gripper_actuator_fault_is_a_physical_zero_force_override(self) -> None:
        scenario = load_scenario("single_red_center")
        model, _data, _physics, _generated = _load_model(scenario, "nominal")
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                        "act_gripper_left")
        self.assertGreaterEqual(actuator_id, 0)
        original = tuple(map(float, model.actuator_forcerange[actuator_id]))
        self.assertNotEqual(original, (0.0, 0.0))
        self.assertEqual(_disable_gripper_actuator(model), original)
        self.assertEqual(tuple(map(float, model.actuator_forcerange[actuator_id])), (0.0, 0.0))

    def test_full_red_zone_is_detected_from_public_rgb_and_skipped(self) -> None:
        manifest, report, _run_dir = self.run_scenario("zone_full_red")
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "DONE")
        self.assertEqual(
            summary["initial_occupied_slots_from_public_rgb"],
            ["RED:0", "RED:1", "RED:2"],
        )
        self.assertEqual(len(summary["track_records"]), 1)
        record = next(iter(summary["track_records"].values()))
        self.assertEqual(record["status"], "SKIPPED_FULL_ZONE")
        self.assertEqual(record["attempts"], 0)
        self.assertEqual(report["physical_sort_success_count"], 0)
        self.assertFalse(report["controller_done_and_physical_success"])

    def test_unknown_color_is_confirmed_and_skipped_without_camera_fault(self) -> None:
        manifest, report, run_dir = self.run_scenario("unknown_cyan_object")
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "DONE")
        self.assertEqual(summary["terminal_result"]["skipped_unknown"], 1)
        record = next(iter(summary["track_records"].values()))
        self.assertEqual(record["class_label"], "UNKNOWN")
        self.assertEqual(record["status"], "SKIPPED_UNKNOWN")
        self.assertEqual(record["attempts"], 0)
        event_reasons = {
            event["reason_code"] for event in map(
                json.loads, (run_dir / "controller" / "controller_events.jsonl")
                .read_text(encoding="utf-8").splitlines()
            )
        }
        self.assertNotIn("CAMERA_FAILURE_LIMIT", event_reasons)
        self.assertIn("UNKNOWN_COLOR_CONFIRMED", event_reasons)
        self.assertEqual(report["physical_sort_success_count"], 0)

    def test_blocked_grasp_route_is_rejected_from_public_perception(self) -> None:
        manifest, report, run_dir = self.run_scenario("blocked_grasp_route")
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "DONE")
        target = next(item for item in summary["track_records"].values()
                      if item["class_label"] == "GREEN")
        self.assertEqual(target["status"], "SKIPPED_UNREACHABLE")
        self.assertIn("COLLISION_GRASP", target["failure_reasons"])
        self.assertEqual(target["grasp_attempts"], 0)
        self.assertFalse(manifest["controller_summary"]["executor"]["active"])
        self.assertEqual(report["physical_sort_success_count"], 0)
        truth = json.loads((run_dir / "evaluator_private" / "scene_truth.json")
                           .read_text(encoding="utf-8"))
        by_body = {item["body_name"]: item for item in report["items"]}
        for item in truth["active_objects"]:
            final_xy = by_body[item["body_name"]]["final_xyz_m"][:2]
            self.assertTrue(all(math.isclose(actual, expected, abs_tol=1e-8)
                                for actual, expected in zip(final_xy, item["initial_xy_m"])))

    def test_camera_failure_stops_before_motion(self) -> None:
        manifest, report, _run_dir = self.run_scenario("single_red_center",
                                                       camera_failure=True)
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        self.assertEqual(summary["terminal_result"]["reason_code"],
                         "CAMERA_FAILURE_LIMIT")
        self.assertIsNone(summary["executor"]["command_id"])
        self.assertEqual(report["physical_sort_success_count"], 0)

    def test_stale_camera_frames_are_rejected_before_motion(self) -> None:
        manifest, report, run_dir = self.run_scenario(
            "single_red_center", stale_camera_age_s=1.0
        )
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        self.assertEqual(summary["terminal_result"]["reason_code"],
                         "CAMERA_FAILURE_LIMIT")
        self.assertIsNone(summary["executor"]["command_id"])
        self.assertEqual(report["physical_sort_success_count"], 0)
        snapshots = [json.loads(line) for line in
                     (run_dir / "controller" / "perception_snapshots.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
        stale_inputs = [item for item in snapshots if item["view"] == "input"]
        self.assertEqual(len(stale_inputs), 3)
        self.assertTrue(all(item["batch_status"] == "STALE" for item in stale_inputs))
        self.assertTrue(all(item["frame_age_s"] > 0.1 for item in stale_inputs))
        injected = [json.loads(line) for line in
                    (run_dir / "evaluator_private" / "fault_injection_events.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(injected), 3)
        self.assertTrue(all(item["fault"] == "STALE_INPUT_CAMERA_FRAME"
                            for item in injected))

    def test_failed_gripper_actuation_is_not_counted_as_a_grasp(self) -> None:
        manifest, report, _run_dir = self.run_scenario(
            "single_red_center", disable_gripper_actuator=True
        )
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        record = next(iter(summary["track_records"].values()))
        self.assertEqual(record["status"], "SAFE_FAILURE")
        self.assertIn("ACTUATOR_TRACKING_TIMEOUT", record["failure_reasons"])
        self.assertEqual(record["grasp_attempts"], 0)
        self.assertIsNone(report["evaluator_grasp_slip"]["object_red_01"]
                          ["first_bilateral_contact_time_s"])
        self.assertFalse(report["controller_done_and_physical_success"])

    def test_low_friction_causes_a_physical_failed_grasp(self) -> None:
        manifest, report, _run_dir = self.run_scenario(
            "single_red_center", gripper_friction_scale=0.001
        )
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        record = next(iter(summary["track_records"].values()))
        self.assertEqual(record["status"], "SAFE_FAILURE")
        self.assertIn("NO_HOLD_AFTER_TEST_LIFT", record["failure_reasons"])
        self.assertEqual(record["grasp_attempts"], 1)
        grasp = report["evaluator_grasp_slip"]["object_red_01"]
        self.assertGreater(grasp["bilateral_contact_samples"], 0)
        self.assertGreater(grasp["maximum_relative_slip_m"], 0.01)
        self.assertEqual(report["physical_sort_success_count"], 0)
        self.assertFalse(report["controller_done_and_physical_success"])

    def test_transfer_payload_loss_is_detected_without_false_success(self) -> None:
        manifest, report, run_dir = self.run_scenario(
            "single_red_center", payload_loss_after_hold=True,
            payload_loss_impulse_Ns=0.12,
        )
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        self.assertEqual(summary["terminal_result"]["reason_code"],
                         "PUBLIC_HOLD_SIGNAL_LOST")
        self.assertEqual(report["physical_sort_success_count"], 0)
        self.assertFalse(report["controller_done_and_physical_success"])
        fault_path = run_dir / "evaluator_private" / "fault_injection_events.jsonl"
        injected = [json.loads(line) for line in fault_path.read_text(encoding="utf-8")
                    .splitlines()]
        self.assertEqual(len(injected), 1)
        self.assertEqual(injected[0]["fault"], "EXTERNAL_PAYLOAD_IMPULSE")
        self.assertEqual(injected[0]["trigger"],
                         "TRANSFER_WITH_BILATERAL_PUBLIC_TOUCH")

    def test_emergency_stop_preserves_confirmed_held_payload(self) -> None:
        manifest, report, run_dir = self.run_scenario(
            "single_red_center", emergency_stop_after_hold=True
        )
        summary = manifest["controller_summary"]
        terminal = summary["terminal_result"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        self.assertEqual(terminal["reason_code"], "EMERGENCY_STOP")
        self.assertIs(terminal["holding_state"], True)
        self.assertIs(terminal["no_automatic_release"], True)
        grasp = report["evaluator_grasp_slip"]["object_red_01"]
        self.assertGreater(grasp["bilateral_contact_samples"], 0)
        truth = json.loads((run_dir / "evaluator_private" / "scene_truth.json")
                           .read_text(encoding="utf-8"))
        item = report["items"][0]
        self.assertEqual(item["body_name"], truth["active_objects"][0]["body_name"])
        self.assertGreater(item["final_xyz_m"][2], 0.04)
        self.assertFalse(report["controller_done_and_physical_success"])

    def test_release_actuator_fault_fails_closed_even_if_object_settles(self) -> None:
        manifest, report, run_dir = self.run_scenario(
            "single_red_center", release_actuator_failure=True
        )
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "SAFE_STOP")
        record = next(iter(summary["track_records"].values()))
        self.assertEqual(record["status"], "SAFE_FAILURE")
        self.assertIn("EXECUTION_FAILED_WITH_POSSIBLE_PAYLOAD", record["failure_reasons"])
        self.assertEqual(report["schema_version"], "M7-evaluator-v1.1")
        self.assertEqual(report["physical_sort_success_count"], 1)
        self.assertFalse(report["controller_done_and_physical_success"])
        injected = [json.loads(line) for line in
                    (run_dir / "evaluator_private" / "fault_injection_events.jsonl")
                    .read_text(encoding="utf-8").splitlines()]
        self.assertEqual(injected[0]["fault"], "GRIPPER_RELEASE_ACTUATOR_DISABLED")

    def test_invalid_placement_view_does_not_confirm_a_sort(self) -> None:
        manifest, report, run_dir = self.run_scenario(
            "single_red_center", placement_camera_failure_at_verify=True
        )
        summary = manifest["controller_summary"]
        self.assertEqual(summary["final_state"], "DONE")
        terminal = summary["terminal_result"]
        self.assertEqual(terminal["controller_confirmed_placements"], 0)
        record = next(iter(summary["track_records"].values()))
        self.assertEqual(record["status"], "SAFE_FAILURE")
        self.assertIn("PLACE_VERIFICATION_TIMEOUT", record["failure_reasons"])
        self.assertEqual(report["physical_sort_success_count"], 1)
        self.assertFalse(report["controller_done_and_physical_success"])
        snapshots = [json.loads(line) for line in
                     (run_dir / "controller" / "perception_snapshots.jsonl")
                     .read_text(encoding="utf-8").splitlines()]
        placement = [item for item in snapshots if item["view"] == "placement"]
        self.assertTrue(any(item["valid"] is False for item in placement))


if __name__ == "__main__":
    unittest.main()
