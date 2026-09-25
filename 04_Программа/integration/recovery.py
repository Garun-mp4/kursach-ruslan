from __future__ import annotations

import math
from collections.abc import Sequence

from controller.types import (
    ExecutionFeedback, ExecutionStatus, RecoveryFeedback, RecoveryStatus,
)


def recovery_feedback_from_execution(
    *,
    request_id: int,
    feedback: ExecutionFeedback,
    simulation_time_s: float,
    joint_state: Sequence[float],
    gripper_position_m: float,
    touch_forces_N: tuple[float, float],
    touch_threshold_N: float,
) -> RecoveryFeedback:
    """Translate terminal actuator status and public touch sensors into M6 recovery evidence."""
    if request_id < 1 or feedback.command_id != f"RECOVERY:{request_id}":
        raise ValueError("Recovery feedback is not correlated to the active request")
    if feedback.status == ExecutionStatus.RUNNING:
        raise ValueError("Recovery feedback requires terminal actuator status")
    values = (*joint_state, gripper_position_m, *touch_forces_N,
              simulation_time_s, touch_threshold_N)
    finite = all(math.isfinite(float(value)) for value in values)
    arm_valid = len(joint_state) == 4 and all(math.isfinite(float(value)) for value in joint_state)
    gripper_known = math.isfinite(float(gripper_position_m))
    if not finite or touch_threshold_N <= 0 or any(force < 0 for force in touch_forces_N):
        return RecoveryFeedback(
            request_id=request_id,
            status=RecoveryStatus.FAILED,
            simulation_time_s=simulation_time_s if math.isfinite(simulation_time_s) else 0.0,
            arm_state_valid=arm_valid,
            gripper_state_known=gripper_known,
            holding=None,
            reason_code="RECOVERY_SENSOR_STATE_INVALID",
        )

    left_contact, right_contact = (force >= touch_threshold_N for force in touch_forces_N)
    if left_contact or right_contact:
        holding: bool | None = True if left_contact and right_contact else None
        reason = "RECOVERY_GRIPPER_CONTACT_NOT_CLEAR"
        status = RecoveryStatus.FAILED
    else:
        holding = False
        reason = feedback.reason_code
        status = (
            RecoveryStatus.COMPLETED
            if feedback.status == ExecutionStatus.COMPLETED
            else RecoveryStatus.FAILED
        )
    if not arm_valid or not gripper_known:
        status = RecoveryStatus.FAILED
        reason = "RECOVERY_JOINT_OR_GRIPPER_STATE_INVALID"
    return RecoveryFeedback(
        request_id=request_id,
        status=status,
        simulation_time_s=simulation_time_s,
        arm_state_valid=arm_valid,
        gripper_state_known=gripper_known,
        holding=holding,
        reason_code=reason,
    )
