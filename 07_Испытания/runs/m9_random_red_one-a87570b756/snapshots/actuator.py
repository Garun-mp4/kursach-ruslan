from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import math

import mujoco
import numpy as np

from controller.types import (ExecutionCommand, ExecutionFeedback, ExecutionStatus,
                              RecoveryRequest, State)
from planning.records import TrajectorySample

from .config import RuntimeConfig

ARM_JOINTS = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")
ACTUATORS = ("act_j1_shoulder", "act_j2_elbow", "act_j3_lift", "act_j4_wrist",
             "act_gripper_left")


@dataclass(frozen=True)
class ExecutorSnapshot:
    command_id: str | None
    phase: str | None
    active: bool
    max_joint_error: tuple[float, float, float, float]
    gripper_error_m: float
    terminal_status: str | None
    terminal_reason: str | None


class MuJoCoPositionExecutor:
    """Executes M5 samples through finite-force MuJoCo position actuators."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 config: RuntimeConfig, *, gripper_preload_position_offset_m: float = 0.0):
        self.model, self.data, self.config = model, data, config
        if not math.isfinite(gripper_preload_position_offset_m) or gripper_preload_position_offset_m < 0:
            raise ValueError("Gripper preload offset must be finite and nonnegative")
        self._gripper_preload_position_offset_m = float(gripper_preload_position_offset_m)
        self._joint_qadr = tuple(self._resolve_joint_qadr(name) for name in ARM_JOINTS)
        self._gripper_qadr = self._resolve_joint_qadr("j5_finger_left")
        self._actuator_ids = tuple(self._resolve_actuator_id(name) for name in ACTUATORS)
        self._joint_dadr = tuple(self._resolve_joint_dadr(name) for name in ARM_JOINTS)
        self._gripper_dadr = self._resolve_joint_dadr("j5_finger_left")
        self._command: ExecutionCommand | None = None
        self._samples: tuple[TrajectorySample, ...] = ()
        self._sample_times: tuple[float, ...] = ()
        self._group_start_s = 0.0
        self._planned_end_s = 0.0
        self._settle_deadline_s = 0.0
        self._stable_since_s: float | None = None
        self._last_targets = np.zeros(5, dtype=float)
        self._last_planned_gripper_position = 0.0
        self._last_planned_dq = np.zeros(4, dtype=float)
        self._last_planned_ddq = np.zeros(4, dtype=float)
        self._last_planned_tcp: tuple[float, float, float, float] | None = None
        self._last_planned_gripper_velocity = 0.0
        self._last_planned_gripper_acceleration = 0.0
        self._phase: str | None = None
        self._max_error = np.zeros(4, dtype=float)
        self._max_gripper_error = 0.0
        self._last_feedback: ExecutionFeedback | None = None
        self._terminal_reason: str | None = None
        self._last_sample_index = 0
        self._latest_plan_time_s = 0.0
        initial_q = self.joint_state()
        self._set_targets(np.asarray((*initial_q, self.gripper_position_m()), dtype=float))

    def _resolve_joint_qadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"M7 actuator model is missing joint {name!r}")
        return int(self.model.jnt_qposadr[jid])

    def _resolve_actuator_id(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if actuator_id < 0:
            raise ValueError(f"M7 actuator model is missing actuator {name!r}")
        return int(actuator_id)

    def _resolve_joint_dadr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"M7 actuator model is missing joint {name!r}")
        return int(self.model.jnt_dofadr[jid])

    def joint_state(self) -> tuple[float, float, float, float]:
        return tuple(float(self.data.qpos[index]) for index in self._joint_qadr)

    def joint_velocity(self) -> tuple[float, float, float, float]:
        values = []
        for name, qadr in zip(ARM_JOINTS, self._joint_qadr):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            dadr = int(self.model.jnt_dofadr[jid])
            values.append(float(self.data.qvel[dadr]))
        return tuple(values)

    def gripper_position_m(self) -> float:
        return float(self.data.qpos[self._gripper_qadr])

    def _set_targets(self, targets: np.ndarray) -> None:
        if targets.shape != (5,) or not np.all(np.isfinite(targets)):
            raise ValueError("M7 actuator targets must be five finite coordinates")
        for index, actuator_id in enumerate(self._actuator_ids):
            low, high = map(float, self.model.actuator_ctrlrange[actuator_id])
            if targets[index] < low - 1e-10 or targets[index] > high + 1e-10:
                raise ValueError(f"Actuator target outside model control range: index={index}")
        self.data.ctrl[np.asarray(self._actuator_ids, dtype=int)] = targets
        self._last_targets = targets.copy()

    def start(self, command: ExecutionCommand, now_s: float) -> None:
        if self._command is not None:
            raise RuntimeError("Cannot replace an active M7 execution command")
        if not math.isfinite(now_s):
            raise ValueError("Execution start time must be finite")
        allowed = set(command.phase_names)
        selected = tuple(sample for sample in command.plan.samples if sample.phase in allowed)
        if not selected:
            raise ValueError(f"M5 plan contains no samples for {command.phase_names}")
        if tuple(dict.fromkeys(sample.phase for sample in selected)) != command.phase_names:
            raise ValueError("M5 sample phases do not match the ordered M6 command phases")
        times = tuple(float(sample.time_s) for sample in selected)
        if any(not math.isfinite(t) for t in times) or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError("M5 command samples have invalid timestamps")
        self._command = command
        self._samples = selected
        self._sample_times = times
        gripper_actuator_id = self._actuator_ids[4]
        low, high = map(float, self.model.actuator_ctrlrange[gripper_actuator_id])
        for sample in selected:
            physical_target = float(sample.gripper_m)
            actuator_target = physical_target + (
                self._gripper_preload_position_offset_m
                if sample.payload_mode == "carried" else 0.0
            )
            if not low <= actuator_target <= high:
                raise ValueError("M7 gripper preload exceeds the physical actuator control range")
        self._group_start_s = now_s
        self._planned_end_s = now_s + max(0.0, times[-1] - times[0])
        self._settle_deadline_s = self._planned_end_s + self.config.actuator_settle_timeout_s
        self._stable_since_s = None
        self._phase = selected[0].phase
        self._max_error[:] = 0.0
        self._max_gripper_error = 0.0
        self._terminal_reason = None
        self._last_sample_index = 0
        self._latest_plan_time_s = times[0]
        self._last_planned_gripper_position = float(selected[0].gripper_m)
        self._last_feedback = ExecutionFeedback(
            command.command_id, ExecutionStatus.RUNNING, now_s,
            gripper_position_m=self.gripper_position_m(),
        )

    def _sample_at(self, absolute_plan_time_s: float) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, float, float,
        tuple[float, float, float, float], int, str,
    ]:
        index = bisect_right(self._sample_times, absolute_plan_time_s) - 1
        index = max(0, min(index, len(self._samples) - 1))
        if index + 1 >= len(self._samples):
            sample = self._samples[index]
            return (
                np.asarray((*sample.q, sample.gripper_m), dtype=float),
                np.asarray(sample.dq, dtype=float), np.asarray(sample.ddq, dtype=float),
                float(sample.gripper_velocity_m_s), float(sample.gripper_acceleration_m_s2),
                tuple(map(float, sample.tcp_xyzyaw)), index, sample.phase,
            )
        first, second = self._samples[index], self._samples[index + 1]
        span = float(second.time_s - first.time_s)
        alpha = 0.0 if span <= 1e-12 else float(np.clip((absolute_plan_time_s - first.time_s) / span, 0.0, 1.0))
        q = np.asarray(first.q, dtype=float) * (1.0 - alpha) + np.asarray(second.q, dtype=float) * alpha
        grip = float(first.gripper_m) * (1.0 - alpha) + float(second.gripper_m) * alpha
        dq = np.asarray(first.dq, dtype=float) * (1.0 - alpha) + np.asarray(second.dq, dtype=float) * alpha
        ddq = np.asarray(first.ddq, dtype=float) * (1.0 - alpha) + np.asarray(second.ddq, dtype=float) * alpha
        tcp = tuple(
            float(a) * (1.0 - alpha) + float(b) * alpha
            for a, b in zip(first.tcp_xyzyaw, second.tcp_xyzyaw)
        )
        grip_velocity = first.gripper_velocity_m_s * (1.0 - alpha) + second.gripper_velocity_m_s * alpha
        grip_acceleration = first.gripper_acceleration_m_s2 * (1.0 - alpha) + second.gripper_acceleration_m_s2 * alpha
        return (
            np.asarray((*q, grip), dtype=float), dq, ddq,
            float(grip_velocity), float(grip_acceleration), tcp, index,
            first.phase if alpha < 1.0 else second.phase,
        )

    def apply(self, now_s: float) -> None:
        if self._command is None:
            return
        elapsed = max(0.0, now_s - self._group_start_s)
        plan_time = self._sample_times[0] + elapsed
        self._latest_plan_time_s = min(plan_time, self._sample_times[-1])
        (targets, planned_dq, planned_ddq, grip_velocity, grip_acceleration,
         planned_tcp, sample_index, phase) = self._sample_at(plan_time)
        self._last_sample_index = sample_index
        self._phase = phase
        self._last_planned_dq = planned_dq
        self._last_planned_ddq = planned_ddq
        self._last_planned_tcp = planned_tcp
        self._last_planned_gripper_velocity = grip_velocity
        self._last_planned_gripper_acceleration = grip_acceleration
        self._last_planned_gripper_position = float(targets[4])
        if self._samples[sample_index].payload_mode == "carried":
            targets[4] += self._gripper_preload_position_offset_m
        self._set_targets(targets)

    def start_recovery(self, request: RecoveryRequest, now_s: float) -> tuple[bool, str]:
        """Open the physical gripper, then reverse the checked M5 prefix to PREGRASP."""
        if self._command is not None:
            return False, "ACTIVE_EXECUTION_NOT_CANCELLED"
        if request.action != "OPEN_AND_RETREAT_NO_PAYLOAD" or request.plan is None:
            return False, "UNSUPPORTED_OR_PLANLESS_RECOVERY"
        plan = request.plan
        plan_times = tuple(float(sample.time_s) for sample in plan.samples)
        current_index = bisect_right(plan_times, self._latest_plan_time_s + 1e-9) - 1
        current_index = max(0, min(current_index, len(plan.samples) - 1))
        anchors = [index for index, sample in enumerate(plan.samples[:current_index + 1])
                   if sample.phase == "PREGRASP"]
        if not anchors:
            return False, "NO_CHECKED_PREGRASP_RECOVERY_ANCHOR"
        anchor_index = anchors[-1]
        if current_index <= anchor_index:
            return False, "ALREADY_AT_OR_BEFORE_RECOVERY_ANCHOR"
        reverse_duration = plan_times[current_index] - plan_times[anchor_index]
        total_duration = self.config.recovery_gripper_open_s + reverse_duration
        if not math.isfinite(total_duration) or total_duration > 4.0:
            return False, "CHECKED_RETREAT_EXCEEDS_M6_RECOVERY_BUDGET"
        current_q = self.joint_state()
        current_tcp = plan.samples[current_index].tcp_xyzyaw
        opened = float(plan.samples[0].gripper_m)
        recovery_samples = [replace(
            plan.samples[current_index], time_s=0.0, phase="M7_RECOVERY_RETREAT",
            q=current_q, gripper_m=opened, dq=(0.0, 0.0, 0.0, 0.0),
            ddq=(0.0, 0.0, 0.0, 0.0), gripper_velocity_m_s=0.0,
            gripper_acceleration_m_s2=0.0, payload_mode="unheld", event=None,
        )]
        recovery_samples.append(replace(
            recovery_samples[0], time_s=self.config.recovery_gripper_open_s,
        ))
        elapsed = self.config.recovery_gripper_open_s
        for index in range(current_index - 1, anchor_index - 1, -1):
            elapsed += plan_times[index + 1] - plan_times[index]
            recovery_samples.append(replace(
                plan.samples[index], time_s=elapsed, phase="M7_RECOVERY_RETREAT",
                gripper_m=opened, gripper_velocity_m_s=0.0,
                gripper_acceleration_m_s2=0.0, payload_mode="unheld", event=None,
            ))
        recovery_plan = replace(
            plan, plan_id=f"{plan.plan_id}:recovery:{request.request_id}",
            phases=("M7_RECOVERY_RETREAT",), samples=tuple(recovery_samples),
            duration_s=elapsed,
        )
        recovery_command = ExecutionCommand(
            command_id=f"RECOVERY:{request.request_id}", state=State.RECOVER,
            phase_names=("M7_RECOVERY_RETREAT",), plan=recovery_plan,
            simulation_time_s=now_s,
        )
        self.start(recovery_command, now_s)
        self._phase = "M7_RECOVERY_RETREAT"
        return True, "REVERSED_M5_CHECKED_PREFIX_TO_PREGRASP"

    def _position_errors(self) -> tuple[np.ndarray, float]:
        actual = np.asarray(self.joint_state(), dtype=float)
        arm_error = np.abs(self._last_targets[:4] - actual)
        grip_error = abs(self._last_planned_gripper_position - self.gripper_position_m())
        self._max_error = np.maximum(self._max_error, arm_error)
        self._max_gripper_error = max(self._max_gripper_error, grip_error)
        return arm_error, grip_error

    def _gripper_position_tracking_required(self) -> bool:
        """A carried-object grip is force-preloaded; its physical jaw coordinate is contact-limited."""
        if not self._samples:
            return True
        return self._samples[self._last_sample_index].payload_mode != "carried"

    def telemetry_snapshot(self) -> dict[str, object]:
        """Expose actuator-level commanded and measured coordinates for logging."""
        actual_q = np.asarray(self.joint_state(), dtype=float)
        actual_dq = np.asarray(self.joint_velocity(), dtype=float)
        actual_ddq = np.asarray([self.data.qacc[index] for index in self._joint_dadr], dtype=float)
        commanded_q = self._last_targets[:4].copy()
        position_error = commanded_q - actual_q
        velocity_error = self._last_planned_dq - actual_dq
        return {
            "commanded_q": tuple(map(float, commanded_q)),
            "actual_q": tuple(map(float, actual_q)),
            "position_error": tuple(map(float, position_error)),
            "absolute_position_error": tuple(map(float, np.abs(position_error))),
            "planned_dq": tuple(map(float, self._last_planned_dq)),
            "actual_dq": tuple(map(float, actual_dq)),
            "velocity_error": tuple(map(float, velocity_error)),
            "planned_ddq": tuple(map(float, self._last_planned_ddq)),
            "actual_ddq": tuple(map(float, actual_ddq)),
            "commanded_tcp_xyzyaw": self._last_planned_tcp,
            "commanded_gripper_m": float(self._last_targets[4]),
            "planned_gripper_m": float(self._last_planned_gripper_position),
            "actual_gripper_m": self.gripper_position_m(),
            "planned_gripper_velocity_m_s": float(self._last_planned_gripper_velocity),
            "planned_gripper_acceleration_m_s2": float(self._last_planned_gripper_acceleration),
            "actual_gripper_velocity_m_s": float(self.data.qvel[self._gripper_dadr]),
            "actuator_effort": tuple(float(self.data.actuator_force[index])
                                      for index in self._actuator_ids),
        }

    def feedback(self, now_s: float) -> ExecutionFeedback | None:
        if self._command is None:
            return self._last_feedback
        command = self._command
        errors, grip_error = self._position_errors()
        if now_s >= self._planned_end_s:
            within_tolerance = (
                errors[0] <= self.config.joint_position_tolerance_rad
                and errors[1] <= self.config.joint_position_tolerance_rad
                and errors[2] <= self.config.slide_position_tolerance_m
                and errors[3] <= self.config.joint_position_tolerance_rad
                and (
                    not self._gripper_position_tracking_required()
                    or grip_error <= self.config.gripper_position_tolerance_m
                )
            )
            if within_tolerance:
                if self._stable_since_s is None:
                    self._stable_since_s = now_s
                if now_s - self._stable_since_s >= self.config.actuator_settle_stable_s:
                    self._last_feedback = ExecutionFeedback(
                        command.command_id, ExecutionStatus.COMPLETED, now_s,
                        gripper_position_m=self.gripper_position_m(),
                    )
                    self._command = None
                    return self._last_feedback
            else:
                self._stable_since_s = None
            if now_s >= self._settle_deadline_s:
                self._terminal_reason = "ACTUATOR_TRACKING_TIMEOUT"
                self._last_feedback = ExecutionFeedback(
                    command.command_id, ExecutionStatus.FAILED, now_s,
                    reason_code=self._terminal_reason,
                    gripper_position_m=self.gripper_position_m(),
                )
                self._command = None
                return self._last_feedback
        self._last_feedback = ExecutionFeedback(
            command.command_id, ExecutionStatus.RUNNING, now_s,
            gripper_position_m=self.gripper_position_m(),
        )
        return self._last_feedback

    def hold_current(self, *, preserve_gripper: bool = True) -> None:
        current = np.asarray(self.joint_state(), dtype=float)
        grip = self._last_targets[4] if preserve_gripper else self.gripper_position_m()
        self._command = None
        self._last_targets = np.asarray((*current, grip), dtype=float)
        self._last_planned_gripper_position = self.gripper_position_m()
        self._last_planned_dq[:] = 0.0
        self._last_planned_ddq[:] = 0.0
        self._last_planned_gripper_velocity = 0.0
        self._last_planned_gripper_acceleration = 0.0
        self._set_targets(self._last_targets)

    def snapshot(self) -> ExecutorSnapshot:
        return ExecutorSnapshot(
            command_id=self._command.command_id if self._command else (
                self._last_feedback.command_id if self._last_feedback else None
            ),
            phase=self._phase,
            active=self._command is not None,
            max_joint_error=tuple(map(float, self._max_error)),
            gripper_error_m=float(self._max_gripper_error),
            terminal_status=self._last_feedback.status.value if self._last_feedback else None,
            terminal_reason=self._terminal_reason,
        )
