from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from controller.types import (
    ExecutionFeedback, ExecutionStatus, RecoveryRequest, RecoveryStatus, State,
)
from integration.actuator import MuJoCoPositionExecutor
from integration.recovery import recovery_feedback_from_execution
from planning.records import Plan, TrajectorySample


class RecoveryFeedbackTests(unittest.TestCase):
    def feedback(self, status=ExecutionStatus.COMPLETED, command_id="RECOVERY:4"):
        return ExecutionFeedback(command_id, status, 2.0)

    def convert(self, *, feedback=None, q=(0.0, 0.4, 0.08, 0.0), grip=0.02,
                touches=(0.0, 0.0)):
        return recovery_feedback_from_execution(
            request_id=4,
            feedback=feedback or self.feedback(),
            simulation_time_s=2.0,
            joint_state=q,
            gripper_position_m=grip,
            touch_forces_N=touches,
            touch_threshold_N=0.001,
        )

    def test_completed_actuator_and_clear_public_touch_sensors_complete_recovery(self):
        result = self.convert()
        self.assertEqual(result.status, RecoveryStatus.COMPLETED)
        self.assertFalse(result.holding)
        self.assertTrue(result.arm_state_valid)
        self.assertTrue(result.gripper_state_known)

    def test_failed_actuator_does_not_claim_recovery(self):
        result = self.convert(feedback=self.feedback(ExecutionStatus.FAILED))
        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertFalse(result.holding)

    def test_any_remaining_finger_contact_fails_closed(self):
        result = self.convert(touches=(0.002, 0.0))
        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertIsNone(result.holding)
        self.assertEqual(result.reason_code, "RECOVERY_GRIPPER_CONTACT_NOT_CLEAR")

    def test_two_contacts_are_reported_as_still_holding(self):
        result = self.convert(touches=(0.002, 0.003))
        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertTrue(result.holding)

    def test_invalid_joint_state_fails_closed(self):
        result = self.convert(q=(0.0, float("nan"), 0.08, 0.0))
        self.assertEqual(result.status, RecoveryStatus.FAILED)
        self.assertFalse(result.arm_state_valid)
        self.assertIsNone(result.holding)

    def test_mismatched_request_cannot_complete_recovery(self):
        with self.assertRaisesRegex(ValueError, "not correlated"):
            self.convert(feedback=self.feedback(command_id="RECOVERY:5"))


def sample(time_s, phase, q, grip=0.02):
    return TrajectorySample(
        time_s=time_s, phase=phase, q=q, dq=(0.0, 0.0, 0.0, 0.0),
        ddq=(0.0, 0.0, 0.0, 0.0), tcp_xyzyaw=(0.0, 0.0, 0.1, 0.0),
        branch_id="ELBOW_POSITIVE", gripper_m=grip, payload_mode="NONE",
    )


def recovery_plan():
    return Plan(
        plan_id="checked-plan", ssot_version="M5-v1.3", ssot_sha256="s",
        model_sha256="m", perception_frame_id=3, perception_time_s=0.1,
        target_track_id=8, target_class="RED", slot_id="RED:1",
        selected_ik_branch="ELBOW_POSITIVE", resolved_tool_yaw_rad=0.0,
        phases=("PREGRASP", "DESCENT", "GRASP_CLOSE"), waypoints=(),
        samples=(
            sample(0.0, "PREGRASP_ROUTE", (0.0, 0.4, 0.08, 0.0)),
            sample(1.0, "PREGRASP", (0.1, 0.5, 0.08, 0.0)),
            sample(2.0, "DESCENT", (0.2, 0.6, 0.02, 0.0), 0.02),
        ), events=(), duration_s=2.0, path_length_m=0.2,
        minimum_clearance_m=0.01, clearance_lower_bound_m=0.009,
        minimum_clearance_phase="PREGRASP_ROUTE", vertical_xy_error_max_m=0.0,
        vertical_yaw_error_max_rad=0.0, maximum_abs_velocity_ratio=0.5,
        maximum_abs_acceleration_ratio=0.5, planner_strategy="test",
    )


class ActuatorRecoveryPathTests(unittest.TestCase):
    def executor(self, latest_time=2.0):
        executor = object.__new__(MuJoCoPositionExecutor)
        executor._command = None
        executor._latest_plan_time_s = latest_time
        executor.config = SimpleNamespace(recovery_gripper_open_s=0.5)
        executor.joint_state = Mock(return_value=(0.2, 0.6, 0.02, 0.0))
        executor.start = Mock()
        return executor

    def request(self, plan=None):
        return RecoveryRequest(
            request_id=4, reason_code="TEST_NO_PAYLOAD",
            action="OPEN_AND_RETREAT_NO_PAYLOAD", plan=plan or recovery_plan(),
            simulation_time_s=2.0,
        )

    def test_recovery_reverses_checked_plan_prefix_and_opens_before_retreat(self):
        executor = self.executor()
        started, reason = executor.start_recovery(self.request(), 2.0)
        self.assertTrue(started)
        self.assertEqual(reason, "REVERSED_M5_CHECKED_PREFIX_TO_PREGRASP")
        command = executor.start.call_args.args[0]
        self.assertEqual(command.state, State.RECOVER)
        self.assertEqual(command.phase_names, ("M7_RECOVERY_RETREAT",))
        samples = command.plan.samples
        self.assertEqual([item.time_s for item in samples], [0.0, 0.5, 1.5])
        self.assertEqual(samples[0].q, executor.joint_state.return_value)
        self.assertEqual(samples[0].q, samples[1].q)
        self.assertEqual(samples[-1].q, (0.1, 0.5, 0.08, 0.0))
        self.assertEqual(samples[-1].phase, "M7_RECOVERY_RETREAT")

    def test_recovery_without_pregrasp_anchor_fails_closed(self):
        plan = replace(recovery_plan(), samples=(sample(0.0, "DESCENT", (0, 0, 0, 0)),))
        executor = self.executor(latest_time=0.0)
        started, reason = executor.start_recovery(self.request(plan), 1.0)
        self.assertFalse(started)
        self.assertEqual(reason, "NO_CHECKED_PREGRASP_RECOVERY_ANCHOR")
        executor.start.assert_not_called()

    def test_recovery_beyond_time_budget_fails_closed(self):
        plan = replace(recovery_plan(), samples=(
            sample(0.0, "PREGRASP", (0, 0, 0, 0)),
            sample(5.0, "DESCENT", (0, 0, 0, 0)),
        ))
        executor = self.executor(latest_time=5.0)
        started, reason = executor.start_recovery(self.request(plan), 1.0)
        self.assertFalse(started)
        self.assertEqual(reason, "CHECKED_RETREAT_EXCEEDS_M6_RECOVERY_BUDGET")
        executor.start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
