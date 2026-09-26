from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import mujoco
import numpy as np

from controller.config import ControllerConfig
from controller.controller import SortController
from controller.types import (
    ControllerInput, ExecutionStatus, GraspEvidence, PlanResponse, RecoveryFeedback,
    RecoveryStatus, State, SystemHealth,
)
from kinematics.scara import forward_kinematics
from planning.planner import Planner

from .actuator import MuJoCoPositionExecutor
from .config import ROOT, RuntimeConfig, load_ssot, value
from .placement import PublicPlacementVerifier
from .recovery import recovery_feedback_from_execution
from .sensors import PublicSensorPipeline, SensorSnapshot


class IntegrationError(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SorterRuntime:
    """Joins M4 perception, M5 planner, M6 controller, and physical M2 actuators."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, *, run_id: str,
                 runtime_config: RuntimeConfig, output_dir: Path,
                 viewer: Any | None = None, sensor_pipeline: PublicSensorPipeline | None = None,
                 camera_fault_injection: bool = False,
                 placement_camera_failure_at_verify: bool = False,
                 emergency_stop_after_hold: bool = False,
                 payload_loss_body_name: str | None = None,
                 payload_loss_impulse_Ns: float = 0.012,
                 release_actuator_failure: bool = False,
                 physics_expected: dict[str, Any] | None = None,
                 evaluator_sample: Callable[[float], None] | None = None):
        self.model, self.data = model, data
        self.run_id = run_id
        self.runtime_config = runtime_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.viewer = viewer
        self._evaluator_sample = evaluator_sample
        self.ssot = load_ssot()
        expected_timestep = float((physics_expected or {}).get(
            "timestep_s", value(self.ssot, "environment.timestep_s")
        ))
        if not math.isclose(float(model.opt.timestep), expected_timestep, rel_tol=0.0, abs_tol=1e-12):
            raise IntegrationError("Physics timestep differs from the configured M7 run profile")
        expected_iterations = int((physics_expected or {}).get(
            "solver_iterations", value(self.ssot, "environment.solver_iterations")
        ))
        if int(model.opt.iterations) != expected_iterations:
            raise IntegrationError("Solver iteration count differs from the configured M7 run profile")
        expected_tolerance = float((physics_expected or {}).get(
            "solver_tolerance", value(self.ssot, "environment.solver_tolerance")
        ))
        if not math.isclose(float(model.opt.tolerance), expected_tolerance, rel_tol=0.0, abs_tol=1e-14):
            raise IntegrationError("Solver tolerance differs from the configured M7 run profile")
        expected_impratio = float(value(self.ssot, "environment.impratio"))
        if not math.isclose(float(model.opt.impratio), expected_impratio,
                            rel_tol=0.0, abs_tol=1e-12):
            raise IntegrationError("Friction impedance ratio differs from the M7 configuration")
        expected_noslip = int(value(self.ssot, "environment.noslip_iterations"))
        if int(model.opt.noslip_iterations) != expected_noslip:
            raise IntegrationError("NoSlip iterations differ from the M7 configuration")
        ticks = runtime_config.control_period_s / float(model.opt.timestep)
        if not math.isclose(ticks, round(ticks), rel_tol=0.0, abs_tol=1e-9):
            raise IntegrationError("Control period must be an integer multiple of the physics step")
        self.physics_steps_per_control = int(round(ticks))
        self.executor = MuJoCoPositionExecutor(
            model, data, runtime_config,
            gripper_preload_position_offset_m=float(value(
                self.ssot, "robot.gripper_preload_position_offset_m"
            )),
        )
        self.sensors = sensor_pipeline or PublicSensorPipeline(model)
        self.placement_verifier = PublicPlacementVerifier(self.ssot, runtime_config)
        initial_snapshot = self.sensors.capture(
            data, simulation_time_s=float(data.time), valid=True
        )
        initial_occupied_slots = self.placement_verifier.initial_occupied_slots(
            initial_snapshot.placement_batch
        )
        self.initial_occupied_slots = initial_occupied_slots
        self.placement_verifier.observe(initial_snapshot.placement_batch)
        self.controller_config = ControllerConfig.load()
        self.planner = Planner()
        self.controller = SortController(
            self.controller_config, run_id=run_id,
            initial_occupied_slots=initial_occupied_slots,
        )
        self._pending_plan_response: PlanResponse | None = None
        self._pending_recovery_feedback: RecoveryFeedback | None = None
        self._active_recovery_request_id: int | None = None
        self._pending_observation_request = None
        self._last_served_observation_request_id: int | None = None
        self._last_new_snapshot: SensorSnapshot | None = None
        self._last_camera_capture_s = float(data.time)
        self._last_placement_evidence = None
        self._latest_touch = (0.0, 0.0)
        self._event_path = self.output_dir / "controller_events.jsonl"
        self._telemetry_path = self.output_dir / "telemetry.csv"
        self._perception_path = self.output_dir / "perception_snapshots.jsonl"
        self._event_stream = self._event_path.open("w", encoding="utf-8", newline="\n")
        self._telemetry_stream = self._telemetry_path.open("w", encoding="utf-8", newline="\n")
        self._perception_stream = self._perception_path.open("w", encoding="utf-8", newline="\n")
        joints = ("j1", "j2", "j3", "j4")
        self._telemetry_columns = ["simulation_time_s", "physics_step_index", "state",
                                   "active_command_id", "phase"]
        for prefix in ("commanded", "actual", "position_error", "absolute_position_error",
                       "planned_dq", "actual_dq", "velocity_error", "planned_ddq", "actual_ddq"):
            self._telemetry_columns.extend(f"{prefix}_{joint}" for joint in joints)
        self._telemetry_columns.extend((
            "commanded_tcp_x_m", "commanded_tcp_y_m", "commanded_tcp_z_m", "commanded_tcp_yaw_rad",
            "estimated_tcp_x_m", "estimated_tcp_y_m", "estimated_tcp_z_m", "estimated_tcp_yaw_rad",
            "planned_gripper_m", "commanded_gripper_m", "actual_gripper_m",
            "planned_gripper_velocity_m_s",
            "actual_gripper_velocity_m_s", "planned_gripper_acceleration_m_s2",
            "touch_left_N", "touch_right_N", "peak_abs_joint_error",
            "peak_abs_error_j1", "peak_abs_error_j2", "peak_abs_error_j3", "peak_abs_error_j4",
            "actuator_force_j1_Nm", "actuator_force_j2_Nm", "actuator_force_j3_N",
            "actuator_force_j4_Nm", "actuator_force_gripper_N",
        ))
        self._telemetry_stream.write(",".join(self._telemetry_columns) + "\n")
        self._last_telemetry_s = -math.inf
        self._last_output = None
        self._camera_fault_injection = camera_fault_injection
        self._placement_camera_failure_at_verify = placement_camera_failure_at_verify
        self._emergency_stop_after_hold_pending = emergency_stop_after_hold
        self._payload_loss_body_id = -1
        self._payload_loss_impulse_Ns = float(payload_loss_impulse_Ns)
        self._payload_loss_remaining_steps = 0
        self._payload_loss_force_N = 0.0
        self._payload_loss_triggered = False
        self._fault_injection_events: list[dict[str, Any]] = []
        self._release_actuator_failure_pending = release_actuator_failure
        if payload_loss_body_name is not None:
            if not math.isfinite(self._payload_loss_impulse_Ns) or self._payload_loss_impulse_Ns <= 0:
                raise ValueError("Payload-loss impulse must be finite and positive")
            self._payload_loss_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, payload_loss_body_name
            )
            if self._payload_loss_body_id <= 0:
                raise IntegrationError(f"Unknown payload-loss body {payload_loss_body_name!r}")
        if emergency_stop_after_hold and payload_loss_body_name is not None:
            raise ValueError("Emergency-stop and payload-loss injections are mutually exclusive")
        self._initial_sensor_snapshot = initial_snapshot
        self._plan_directory = self.output_dir / "plans"

    def _sensor_values(self) -> tuple[float, float]:
        values: list[float] = []
        for name in ("touch_left", "touch_right"):
            sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id < 0:
                raise IntegrationError(f"M2 contact sensor {name!r} is missing")
            address = int(self.model.sensor_adr[sensor_id])
            dimension = int(self.model.sensor_dim[sensor_id])
            if dimension != 1:
                raise IntegrationError(f"Unexpected touch sensor dimension for {name}")
            values.append(max(0.0, float(self.data.sensordata[address])))
        self._latest_touch = (values[0], values[1])
        return self._latest_touch

    def _grasp_evidence(self, now_s: float) -> GraspEvidence:
        left, right = self._sensor_values()
        threshold = self.runtime_config.touch_contact_threshold_N
        holding = left >= threshold and right >= threshold
        return GraspEvidence(
            holding=holding,
            simulation_time_s=now_s,
            source="FINGER_CONTACT_SENSOR",
            valid=True,
        )

    def _capture_if_requested(self, now_s: float) -> tuple[Any, Any] | tuple[None, None]:
        state = self.controller.state
        request = self._pending_observation_request
        request_due = (
            request is not None
            and request.request_id != self._last_served_observation_request_id
            and now_s + 1e-12 >= request.simulation_time_s + self.runtime_config.observation_settle_s
        )
        periodic_place = (
            state == State.VERIFY_PLACE
            and now_s - self._last_camera_capture_s >= self.runtime_config.telemetry_period_s - 1e-12
        )
        if not request_due and not periodic_place:
            return None, None
        snapshot = self.sensors.capture(
            self.data,
            simulation_time_s=now_s,
            valid=not self._camera_fault_injection,
            placement_valid=not (
                self._placement_camera_failure_at_verify and state == State.VERIFY_PLACE
            ),
        )
        self._last_camera_capture_s = now_s
        self._last_new_snapshot = snapshot
        self._write_perception_snapshot(snapshot)
        if request_due:
            if snapshot.frame.frame_id < request.minimum_frame_id:
                raise IntegrationError("Camera violated M6 minimum_frame_id contract")
            if snapshot.placement_frame.frame_id < request.minimum_frame_id:
                raise IntegrationError("Placement camera violated M6 minimum_frame_id contract")
            self._last_served_observation_request_id = request.request_id
            self._pending_observation_request = None
        if state == State.VERIFY_PLACE:
            placement_evidence = self.placement_verifier.evidence(snapshot.placement_batch)
            self._last_placement_evidence = placement_evidence
        else:
            self.placement_verifier.observe(snapshot.placement_batch)
            placement_evidence = None
        return snapshot.input_batch, placement_evidence

    def _write_perception_snapshot(self, snapshot: SensorSnapshot) -> None:
        for view_name, frame, batch in (
            ("input", snapshot.frame, snapshot.input_batch),
            ("placement", snapshot.placement_frame, snapshot.placement_batch),
        ):
            self._perception_stream.write(json.dumps({
                "view": view_name,
                "frame_id": frame.frame_id,
                "simulation_time_s": frame.simulation_time_s,
                "valid": frame.valid,
                "invalid_reason": frame.invalid_reason,
                "camera_config_id": frame.camera_config_id,
                "resolution_px": frame.resolution_px,
                "batch_status": batch.status,
                "batch_reason": batch.reason,
                "photometric_gain_rgb": batch.photometric_gain_rgb,
                "detections": [asdict(item) for item in batch.detections],
            }, ensure_ascii=False, allow_nan=False, default=str) + "\n")

    def _controller_input(self, now_s: float, batch, placement_evidence) -> ControllerInput:
        q = self.executor.joint_state()
        grip = self.executor.gripper_position_m()
        health_valid = all(math.isfinite(value) for value in (*q, grip))
        try:
            tcp = forward_kinematics(q, self.planner.context.arm).pose
            tcp_xy = (float(tcp.x_m), float(tcp.y_m))
        except ValueError:
            tcp_xy = None
            health_valid = False
        feedback = self.executor.feedback(now_s)
        recovery_feedback = self._pending_recovery_feedback
        execution_feedback = feedback
        if self._active_recovery_request_id is not None:
            expected_command_id = f"RECOVERY:{self._active_recovery_request_id}"
            if feedback is not None and feedback.command_id == expected_command_id:
                if feedback.status != ExecutionStatus.RUNNING:
                    left, right = self._sensor_values()
                    recovery_feedback = recovery_feedback_from_execution(
                        request_id=self._active_recovery_request_id,
                        feedback=feedback,
                        simulation_time_s=now_s,
                        joint_state=self.executor.joint_state(),
                        gripper_position_m=self.executor.gripper_position_m(),
                        touch_forces_N=(left, right),
                        touch_threshold_N=self.runtime_config.touch_contact_threshold_N,
                    )
                    self._active_recovery_request_id = None
                    execution_feedback = None
            elif feedback is not None and feedback.status != ExecutionStatus.RUNNING:
                raise IntegrationError(
                    "Terminal actuator feedback does not match the active M7 recovery request"
                )
            else:
                execution_feedback = None
        return ControllerInput(
            simulation_time_s=now_s,
            health=SystemHealth(True, health_valid, math.isfinite(grip)),
            current_q=q if health_valid else None,
            current_gripper_m=grip if math.isfinite(grip) else None,
            current_tcp_xy_m=tcp_xy,
            detection_batch=batch,
            plan_response=self._pending_plan_response,
            execution_feedback=execution_feedback,
            grasp_evidence=self._grasp_evidence(now_s),
            placement_evidence=placement_evidence,
            recovery_feedback=recovery_feedback,
        )

    def _write_controller_events(self, output) -> None:
        for event in output.events:
            document = asdict(event)
            document["state"] = event.state.value
            self._event_stream.write(json.dumps(document, ensure_ascii=False, allow_nan=False) + "\n")

    def _handle_output(self, output, now_s: float) -> None:
        self._last_output = output
        self._write_controller_events(output)
        if output.observation_requests:
            self._pending_observation_request = output.observation_requests[-1]
        if output.safety_hold_requested:
            self.executor.hold_current(preserve_gripper=True)
        if output.plan_requests:
            if len(output.plan_requests) != 1:
                raise IntegrationError("M6 emitted more than one pending M5 request")
            request = output.plan_requests[0]
            started = time.perf_counter()
            result = self.planner.plan_cycle(
                detections=request.detections,
                target_track_id=request.target_track_id,
                current_q=request.current_q,
                current_gripper_m=request.current_gripper_m,
                available_slot_ids=request.available_slot_ids,
                now_time_s=request.simulation_time_s,
            )
            plan_artifacts = None
            phase_transitions: list[str] = []
            if result.plan is not None:
                plan = result.plan
                self._plan_directory.mkdir(parents=True, exist_ok=True)
                plan_stem = f"m5_plan_{request.request_id}"
                plan_csv = self._plan_directory / f"{plan_stem}.csv"
                plan_json = self._plan_directory / f"{plan_stem}.json"
                with plan_csv.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow((
                        "time_s", "phase", "q1_rad", "q2_rad", "q3_m", "q4_rad",
                        "dq1_rad_s", "dq2_rad_s", "dq3_m_s", "dq4_rad_s",
                        "ddq1_rad_s2", "ddq2_rad_s2", "ddq3_m_s2", "ddq4_rad_s2",
                        "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_yaw_rad", "gripper_m",
                        "payload_mode", "event",
                    ))
                    for sample in plan.samples:
                        writer.writerow((
                            sample.time_s, sample.phase, *sample.q, *sample.dq,
                            *sample.ddq, *sample.tcp_xyzyaw, sample.gripper_m,
                            sample.payload_mode, sample.event or "",
                        ))
                payload = result.to_dict()
                payload["plan"].pop("samples", None)
                payload["plan"].pop("clearance_profile", None)
                plan_json.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=str) + "\n",
                    encoding="utf-8",
                )
                for sample in plan.samples:
                    if not phase_transitions or phase_transitions[-1] != sample.phase:
                        phase_transitions.append(sample.phase)
                plan_artifacts = {
                    "sampled_trajectory_csv": str(plan_csv.relative_to(self.output_dir)),
                    "plan_metadata_json": str(plan_json.relative_to(self.output_dir)),
                }
            self._pending_plan_response = PlanResponse(request.request_id, result)
            self._event_stream.write(json.dumps({
                "event_type": "M5_PLAN_RESPONSE",
                "simulation_time_s": now_s,
                "request_id": request.request_id,
                "plan_code": result.code.value,
                "plan_id": result.plan.plan_id if result.plan else None,
                "planner_wall_time_s": time.perf_counter() - started,
                "diagnostics": result.diagnostics,
                "phase_transition_sequence": phase_transitions,
                "plan_artifacts": plan_artifacts,
            }, ensure_ascii=False, allow_nan=False, default=str) + "\n")
        if output.execution_commands:
            if len(output.execution_commands) != 1:
                raise IntegrationError("M6 emitted more than one active motion command")
            self.executor.start(output.execution_commands[0], now_s)
        if output.recovery_requests:
            if len(output.recovery_requests) != 1:
                raise IntegrationError("M6 emitted more than one pending recovery request")
            request = output.recovery_requests[0]
            started, reason = self.executor.start_recovery(request, now_s)
            if started:
                self._active_recovery_request_id = request.request_id
                self._pending_recovery_feedback = None
            else:
                self.executor.hold_current(preserve_gripper=True)
                q_now = self.executor.joint_state()
                self._pending_recovery_feedback = RecoveryFeedback(
                    request.request_id, RecoveryStatus.FAILED, now_s,
                    arm_state_valid=all(math.isfinite(value) for value in q_now),
                    gripper_state_known=math.isfinite(self.executor.gripper_position_m()),
                    holding=None,
                    reason_code=reason,
                )
        else:
            self._pending_recovery_feedback = None

    def _write_telemetry(self, now_s: float) -> None:
        if now_s - self._last_telemetry_s + 1e-12 < self.runtime_config.telemetry_period_s:
            return
        actuator = self.executor.telemetry_snapshot()
        left, right = self._sensor_values()
        snapshot = self.executor.snapshot()
        try:
            estimated_tcp = forward_kinematics(actuator["actual_q"], self.planner.context.arm).pose
            estimated_tcp_values = (estimated_tcp.x_m, estimated_tcp.y_m, estimated_tcp.z_m,
                                    estimated_tcp.yaw_rad)
        except (TypeError, ValueError):
            estimated_tcp_values = ("", "", "", "")
        commanded_tcp = actuator["commanded_tcp_xyzyaw"]
        commanded_tcp_values = commanded_tcp if commanded_tcp is not None else ("", "", "", "")
        values: list[object] = [
            now_s, round(now_s / float(self.model.opt.timestep)), self.controller.state.value,
            snapshot.command_id or "", snapshot.phase or "",
            *actuator["commanded_q"], *actuator["actual_q"],
            *actuator["position_error"], *actuator["absolute_position_error"],
            *actuator["planned_dq"], *actuator["actual_dq"], *actuator["velocity_error"],
            *actuator["planned_ddq"], *actuator["actual_ddq"],
            *commanded_tcp_values, *estimated_tcp_values,
            actuator["planned_gripper_m"], actuator["commanded_gripper_m"],
            actuator["actual_gripper_m"],
            actuator["planned_gripper_velocity_m_s"], actuator["actual_gripper_velocity_m_s"],
            actuator["planned_gripper_acceleration_m_s2"], left, right,
            max(snapshot.max_joint_error), *snapshot.max_joint_error,
            *actuator["actuator_effort"],
        ]
        self._telemetry_stream.write(",".join(
            str(item) if isinstance(item, str) else f"{float(item):.9g}" for item in values
        ) + "\n")
        self._last_telemetry_s = now_s
        if self._evaluator_sample is not None:
            self._evaluator_sample(now_s)

    def _apply_payload_loss_fault(self) -> None:
        if self._payload_loss_body_id < 0:
            return
        timestep = float(self.model.opt.timestep)
        if not self._payload_loss_triggered and self.controller.state == State.TRANSFER:
            left, right = self._sensor_values()
            if min(left, right) >= self.runtime_config.touch_contact_threshold_N:
                duration_s = 0.02
                self._payload_loss_remaining_steps = max(1, round(duration_s / timestep))
                actual_duration_s = self._payload_loss_remaining_steps * timestep
                self._payload_loss_force_N = self._payload_loss_impulse_Ns / actual_duration_s
                self._payload_loss_triggered = True
                self._fault_injection_events.append({
                    "fault": "EXTERNAL_PAYLOAD_IMPULSE",
                    "simulation_time_s": float(self.data.time),
                    "body_name": mujoco.mj_id2name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, self._payload_loss_body_id
                    ),
                    "world_impulse_xyz_Ns": [self._payload_loss_impulse_Ns, 0.0, 0.0],
                    "duration_s": actual_duration_s,
                    "trigger": "TRANSFER_WITH_BILATERAL_PUBLIC_TOUCH",
                })

        wrench = self.data.xfrc_applied[self._payload_loss_body_id]
        wrench[:] = 0.0
        if self._payload_loss_remaining_steps > 0:
            wrench[0] = self._payload_loss_force_N
            self._payload_loss_remaining_steps -= 1

    def _apply_release_actuator_fault(self) -> None:
        if not self._release_actuator_failure_pending or self.controller.state != State.RELEASE:
            return
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper_left"
        )
        if actuator_id < 0:
            raise IntegrationError("M7 model is missing the gripper position actuator")
        original_range = tuple(map(float, self.model.actuator_forcerange[actuator_id]))
        self.model.actuator_forcerange[actuator_id] = (0.0, 0.0)
        self._release_actuator_failure_pending = False
        self._fault_injection_events.append({
            "fault": "GRIPPER_RELEASE_ACTUATOR_DISABLED",
            "simulation_time_s": float(self.data.time),
            "original_forcerange_N": original_range,
            "trigger": "CONTROLLER_RELEASE_STATE",
        })

    def run(self, *, max_simulation_time_s: float, max_control_ticks: int | None = None) -> dict[str, Any]:
        if not math.isfinite(max_simulation_time_s) or max_simulation_time_s <= 0:
            raise ValueError("Maximum simulation time must be finite and positive")
        start_s = float(self.data.time)
        control_ticks = 0
        started_wall = time.perf_counter()
        while float(self.data.time) - start_s < max_simulation_time_s:
            if max_control_ticks is not None and control_ticks >= max_control_ticks:
                break
            now_s = float(self.data.time)
            batch, placement_evidence = self._capture_if_requested(now_s)
            controller_input = self._controller_input(now_s, batch, placement_evidence)
            if (
                self._emergency_stop_after_hold_pending
                and self.controller.state == State.TRANSFER
                and controller_input.grasp_evidence is not None
                and controller_input.grasp_evidence.holding is True
            ):
                controller_input = replace(controller_input, emergency_stop=True)
                self._emergency_stop_after_hold_pending = False
                self._fault_injection_events.append({
                    "fault": "EMERGENCY_STOP_WHILE_HOLDING",
                    "simulation_time_s": now_s,
                    "trigger": "TRANSFER_WITH_BILATERAL_PUBLIC_TOUCH",
                })
            output = self.controller.step(controller_input)
            self._pending_plan_response = None
            self._pending_recovery_feedback = None
            self._handle_output(output, now_s)
            control_ticks += 1
            self._write_telemetry(now_s)
            if self.viewer is not None:
                if not self.viewer.is_running():
                    break
                self.viewer.sync()
            if output.state in {State.DONE, State.SAFE_STOP}:
                break
            for _ in range(self.physics_steps_per_control):
                step_time = float(self.data.time)
                self._apply_release_actuator_fault()
                self.executor.apply(step_time)
                self._apply_payload_loss_fault()
                mujoco.mj_step(self.model, self.data)
            if self.viewer is not None:
                self.viewer.sync()
        elapsed_wall = time.perf_counter() - started_wall
        self._write_telemetry(float(self.data.time))
        self._event_stream.flush()
        self._telemetry_stream.flush()
        return {
            "run_id": self.run_id,
            "final_state": self.controller.state.value,
            "simulation_time_s": float(self.data.time),
            "simulated_elapsed_s": float(self.data.time) - start_s,
            "control_ticks": control_ticks,
            "wall_time_s": elapsed_wall,
            "terminal_result": self._last_output.terminal_result if self._last_output else None,
            "track_records": {
                str(key): {
                    "class_label": record.class_label,
                    "status": record.status.value,
                    "attempts": record.attempts,
                    "grasp_attempts": record.grasp_attempts,
                    "failure_reasons": list(record.failure_reasons),
                }
                for key, record in self.controller.track_records.items()
            },
            "slot_states": [asdict(slot) for slot in self.controller.slot_states],
            "executor": asdict(self.executor.snapshot()),
            "controller_event_count": len(self.controller.event_log),
            "controller_events_path": self._event_path.name,
            "telemetry_path": self._telemetry_path.name,
            "perception_snapshots_path": self._perception_path.name,
            "initial_occupied_slots_from_public_rgb": list(self.initial_occupied_slots),
            "fault_injection_events": list(self._fault_injection_events),
        }

    def close(self) -> None:
        self._event_stream.close()
        self._telemetry_stream.close()
        self._perception_stream.close()
        self.sensors.close()
