from __future__ import annotations

from dataclasses import replace
import math
from typing import Iterable

from perception.types import Detection, DetectionBatch
from planning.records import Phase, Plan, PlanCode, PlanResult

from .config import ControllerConfig
from .fsm_spec import EXECUTION_PHASES, PLAN_REQUIRED_PHASES, TRANSITIONS
from .reservations import ReservationError, ReservationManager
from .types import (
    ControllerEvent,
    ControllerInput,
    ControllerOutput,
    EvidenceStatus,
    ExecutionCommand,
    ExecutionStatus,
    ObservationRequest,
    PlanRequest,
    RecoveryRequest,
    RecoveryStatus,
    SlotSnapshot,
    SlotStatus,
    State,
    TrackRecord,
    TrackStatus,
)


class IllegalTransitionError(RuntimeError):
    pass


class ControllerContractError(RuntimeError):
    pass


PLAN_RETRYABLE = frozenset({
    PlanCode.TIMEOUT,
    PlanCode.STALE_DETECTION,
    PlanCode.SCENE_UNCERTAIN,
})
PLAN_FATAL = frozenset({PlanCode.INVALID_INPUT, PlanCode.INVALID_BATCH, PlanCode.MODEL_MISMATCH})
PLAN_SLOT_CONFLICT = frozenset({PlanCode.NO_AVAILABLE_SLOT, PlanCode.SLOT_INVALID, PlanCode.SLOT_OCCUPIED})


class SortController:
    """Deterministic high-level sorter; all physical work is delegated to M5/M7 ports."""

    def __init__(
        self,
        config: ControllerConfig | None = None,
        *,
        run_id: str = "M6-controlled",
        initial_occupied_slots: Iterable[str] | None = None,
    ):
        if not run_id:
            raise ValueError("run_id must be a non-empty stable identifier")
        self.config = config or ControllerConfig.load()
        if initial_occupied_slots is None:
            initial_occupied_slots = tuple(self.config.p("initial_occupied_slots"))
        self.reservations = ReservationManager(
            self.config.slot_classes, tuple(initial_occupied_slots)
        )
        self.run_id = run_id
        self.state = State.INIT
        self._state_started_s: float | None = None
        self._batch_started_s: float | None = None
        self._last_time_s: float | None = None
        self._event_clock_s = 0.0
        self._events: list[ControllerEvent] = []
        self._event_id = 0
        self._tick_events: list[ControllerEvent] = []
        self._observation_requests: list[ObservationRequest] = []
        self._plan_requests: list[PlanRequest] = []
        self._execution_commands: list[ExecutionCommand] = []
        self._recovery_requests: list[RecoveryRequest] = []
        self._request_id = 0
        self._command_id = 0
        self._track_records: dict[int, TrackRecord] = {}
        self._latest_batch: DetectionBatch | None = None
        self._last_frame_id = -1
        self._last_place_frame_id = -1
        self._empty_scene_frames = 0
        self._camera_failures = 0
        self._scene_uncertainty_frames = 0
        self._active_track_id: int | None = None
        self._active_class: str | None = None
        self._active_slot_id: str | None = None
        self._source_xy: tuple[float, float] | None = None
        self._selection_tcp_xy: tuple[float, float] | None = None
        self._selection_q: tuple[float, float, float, float] | None = None
        self._selection_gripper_m: float | None = None
        self._plan: Plan | None = None
        self._pending_plan_input: tuple[tuple[float, float, float, float], float] | None = None
        self._pending_plan_request_id: int | None = None
        self._active_command_id: str | None = None
        self._completed_command_at_s: float | None = None
        self._lift_completed_at_s: float | None = None
        self._release_completed_at_s: float | None = None
        self._observe_request_frame_id = -1
        self._recovery_requires_feedback = False
        self._recovery_request_id: int | None = None
        self._recovery_action: str | None = None
        self._recovery_next_state = State.OBSERVE
        self._recovery_reason = ""
        self._recovery_attempts = 0
        self._cycle_deadline_s: float | None = None
        self._cycle_started_s: float | None = None
        self._holding_state: bool | None = None
        self._holding_evidence_time_s: float | None = None
        self._cycle_outcome: str | None = None
        self._terminal_result: dict | None = None
        self._placed_slot_by_track: dict[int, str] = {}
        self._safety_hold_requested = False

    @property
    def event_log(self) -> tuple[ControllerEvent, ...]:
        return tuple(self._events)

    @property
    def track_records(self) -> dict[int, TrackRecord]:
        return {
            track_id: replace(record, failure_reasons=list(record.failure_reasons))
            for track_id, record in self._track_records.items()
        }

    @property
    def slot_states(self) -> tuple[SlotSnapshot, ...]:
        return self.reservations.snapshots()

    @property
    def active_command_id(self) -> str | None:
        return self._active_command_id

    @property
    def active_plan(self) -> Plan | None:
        return self._plan

    def step(self, inputs: ControllerInput) -> ControllerOutput:
        self._tick_events = []
        self._observation_requests = []
        self._plan_requests = []
        self._execution_commands = []
        self._recovery_requests = []

        raw_now = inputs.simulation_time_s
        if not math.isfinite(raw_now):
            self._event_clock_s = self._last_time_s if self._last_time_s is not None else 0.0
            self._safe_stop("INVALID_SIMULATION_TIME", {"received": repr(raw_now)})
            return self._output()
        now = float(raw_now)
        self._event_clock_s = now
        if self._last_time_s is not None and now < self._last_time_s - 1e-12:
            self._safe_stop("SIMULATION_TIME_REGRESSION", {
                "previous_s": self._last_time_s, "received_s": now,
            })
            return self._output()
        self._last_time_s = now
        if self._batch_started_s is None:
            self._batch_started_s = now
            self._state_started_s = now
            self._emit("STATE_ENTERED", "CONTROLLER_START", details={"run_id": self.run_id})

        self._accept_public_holding_signal(inputs, now)
        if inputs.emergency_stop and self.state not in {State.SAFE_STOP, State.DONE}:
            self._safe_stop("EMERGENCY_STOP", {"holding_state": self._holding_state})
            return self._output()

        if self.state == State.SAFE_STOP:
            self._handle_safe_stop_reset(inputs, now)
            self._assert_invariants()
            return self._output()
        if self.state == State.DONE:
            return self._output()

        if self._batch_started_s is not None and now - self._batch_started_s >= float(
            self.config.p("batch_timeout_s")
        ):
            self._safe_stop("BATCH_TIMEOUT", {"elapsed_s": now - self._batch_started_s})
            self._assert_invariants()
            return self._output()
        if self._cycle_deadline_s is not None and now >= self._cycle_deadline_s:
            self._on_active_cycle_timeout(now, "CYCLE_TIMEOUT")
            self._assert_invariants()
            return self._output()

        state_before = self.state
        if self.state == State.INIT:
            self._handle_init(inputs, now)
        elif self.state == State.OBSERVE:
            self._handle_observe(inputs, now)
        elif self.state == State.SELECT:
            self._handle_select(inputs, now)
        elif self.state == State.PLAN:
            self._handle_plan(inputs, now)
        elif self.state == State.VERIFY_HOLD:
            self._handle_verify_hold(inputs, now)
        elif self.state in EXECUTION_PHASES:
            self._handle_execution(inputs, now)
        elif self.state == State.VERIFY_PLACE:
            self._handle_verify_place(inputs, now)
        elif self.state == State.RECOVER:
            self._handle_recover(inputs, now)
        else:
            raise ControllerContractError(f"No handler implemented for {self.state.value}")

        if self.state == state_before and self._state_started_s is not None:
            timeout = self._state_timeout_s()
            if timeout is not None and now - self._state_started_s >= timeout:
                self._handle_state_timeout(now)
        self._assert_invariants()
        return self._output()

    def _output(self) -> ControllerOutput:
        return ControllerOutput(
            state=self.state,
            events=tuple(self._tick_events),
            observation_requests=tuple(self._observation_requests),
            plan_requests=tuple(self._plan_requests),
            execution_commands=tuple(self._execution_commands),
            recovery_requests=tuple(self._recovery_requests),
            safety_hold_requested=self.state == State.SAFE_STOP or self._safety_hold_requested,
            terminal_result=self._terminal_result,
        )

    def _emit(
        self,
        event_type: str,
        reason_code: str,
        *,
        track_id: int | None = None,
        slot_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        self._event_id += 1
        event = ControllerEvent(
            self._event_id, event_type, self._event_clock_s, self.state, reason_code,
            track_id, slot_id, details or {},
        )
        self._events.append(event)
        self._tick_events.append(event)

    def _transition(self, target: State, reason_code: str, details: dict | None = None) -> None:
        source = self.state
        if target not in TRANSITIONS[source]:
            raise IllegalTransitionError(f"Illegal FSM transition {source.value} -> {target.value}")
        self.state = target
        self._state_started_s = self._event_clock_s
        self._completed_command_at_s = None
        self._emit("STATE_TRANSITION", reason_code, details={
            "from": source.value, "to": target.value, **(details or {}),
        })
        self._emit("STATE_ENTERED", reason_code, details=details or {})
        self._enter_state(target)

    def _enter_state(self, target: State) -> None:
        if target == State.OBSERVE:
            self._issue_observation_request()
        elif target == State.PLAN:
            self._create_plan_request()
        elif target in {State.GRASP, State.VERIFY_HOLD, State.RELEASE}:
            self._holding_state = None
            self._holding_evidence_time_s = None
            if target == State.VERIFY_HOLD:
                self._lift_completed_at_s = None
                self._active_command_id = None
            if target in EXECUTION_PHASES:
                self._issue_execution_command(target)
        elif target in EXECUTION_PHASES:
            self._issue_execution_command(target)
        elif target == State.VERIFY_PLACE:
            self._issue_observation_request()
        elif target == State.RECOVER:
            self._enter_recovery()
        elif target == State.SAFE_STOP:
            self._safety_hold_requested = True
        elif target == State.DONE:
            self._terminal_result = self._build_terminal_result()

    def _issue_observation_request(self) -> None:
        self._request_id += 1
        self._observe_request_frame_id = self._last_frame_id + 1
        request = ObservationRequest(
            self._request_id, self._observe_request_frame_id, self._event_clock_s
        )
        self._observation_requests.append(request)
        self._emit("OBSERVATION_REQUESTED", "FRESH_FRAME_REQUIRED",
                   details={"request_id": request.request_id,
                            "minimum_frame_id": request.minimum_frame_id})

    def _create_plan_request(self) -> None:
        if (
            self._latest_batch is None or self._active_track_id is None
            or self._active_class is None or self._active_slot_id is None
            or self._selection_q is None or self._selection_gripper_m is None
        ):
            self._safe_stop("PLAN_REQUEST_CONTEXT_MISSING")
            return
        self._request_id += 1
        request = PlanRequest(
            request_id=self._request_id,
            target_track_id=self._active_track_id,
            target_class=self._active_class,
            reserved_slot_id=self._active_slot_id,
            detections=self._latest_batch,
            current_q=self._selection_q,
            current_gripper_m=self._selection_gripper_m,
            available_slot_ids=(self._active_slot_id,),
            simulation_time_s=self._event_clock_s,
        )
        self._pending_plan_request_id = request.request_id
        self._plan_requests.append(request)
        self._emit("PLAN_REQUESTED", "M5_FULL_CYCLE_PREFLIGHT",
                   track_id=self._active_track_id, slot_id=self._active_slot_id,
                   details={"request_id": request.request_id,
                            "frame_id": self._latest_batch.frame_id})

    def _issue_execution_command(self, state: State) -> None:
        if self._plan is None:
            self._safe_stop("EXECUTION_WITHOUT_VALID_PLAN", {"state": state.value})
            return
        phase_names = EXECUTION_PHASES[state]
        if not set(phase_names).issubset(set(self._plan.phases)):
            self._safe_stop("PLAN_PHASE_MISSING", {"state": state.value})
            return
        self._command_id += 1
        command_id = f"{self.run_id}:{self._command_id:04d}:{state.value}"
        command = ExecutionCommand(
            command_id, state, phase_names, self._plan, self._event_clock_s
        )
        self._active_command_id = command_id
        self._completed_command_at_s = None
        self._execution_commands.append(command)
        self._emit("EXECUTION_COMMAND_ISSUED", "M5_PHASE_GROUP",
                   track_id=self._active_track_id, slot_id=self._active_slot_id,
                   details={"command_id": command_id, "phase_names": list(phase_names)})

    def _enter_recovery(self) -> None:
        if not self._recovery_requires_feedback:
            self._emit("RECOVERY_NO_MOTION_REQUIRED", self._recovery_reason or "NO_MOTION",
                       track_id=self._active_track_id, slot_id=self._active_slot_id)
            return
        if self._recovery_attempts >= int(self.config.p("max_recovery_attempts")):
            self._mark_recovery_failure("RECOVERY_BUDGET_EXHAUSTED")
            self._safe_stop("RECOVERY_BUDGET_EXHAUSTED",
                            {"recovery_reason": self._recovery_reason})
            return
        self._recovery_attempts += 1
        self._request_id += 1
        self._recovery_request_id = self._request_id
        request = RecoveryRequest(
            request_id=self._request_id,
            reason_code=self._recovery_reason,
            action=self._recovery_action or "STOP_AND_HOLD",
            plan=self._plan,
            simulation_time_s=self._event_clock_s,
        )
        self._recovery_requests.append(request)
        self._emit("RECOVERY_REQUESTED", request.reason_code,
                   track_id=self._active_track_id, slot_id=self._active_slot_id,
                   details={"request_id": request.request_id, "action": request.action})

    def _handle_init(self, inputs: ControllerInput, now: float) -> None:
        if inputs.emergency_stop:
            self._safe_stop("EMERGENCY_STOP")
            return
        if not inputs.health.dependencies_ready:
            self._emit("INIT_WAIT", "DEPENDENCY_NOT_READY")
            return
        if not inputs.health.arm_state_valid or not self._valid_q(inputs.current_q) \
                or not self._valid_xy(inputs.current_tcp_xy_m):
            self._emit("INIT_WAIT", "ARM_STATE_NOT_VALID")
            return
        if not inputs.health.gripper_state_known or not self._finite(inputs.current_gripper_m):
            self._emit("INIT_WAIT", "GRIPPER_STATE_NOT_KNOWN")
            return
        if abs(float(inputs.current_gripper_m) - self.config.gripper_home_m) > float(
            self.config.p("gripper_position_tolerance_m")
        ):
            self._safe_stop("GRIPPER_NOT_AT_HOME",
                            {"gripper_m": inputs.current_gripper_m,
                             "home_m": self.config.gripper_home_m})
            return
        evidence = inputs.grasp_evidence
        if evidence is None or evidence.holding is not False or not self._fresh_grasp_evidence(
            evidence, now, minimum_time_s=now - float(self.config.p("hold_evidence_max_age_s"))
        ):
            self._emit("INIT_WAIT", "NO_FRESH_EMPTY_GRIPPER_EVIDENCE")
            return
        if self.reservations.reserved_track_id is not None or self.reservations.reserved_slot_id is not None:
            self._safe_stop("INIT_RESERVATION_NOT_EMPTY")
            return
        self._holding_state = False
        self._transition(State.OBSERVE, "STARTUP_READY")

    def _handle_observe(self, inputs: ControllerInput, now: float) -> None:
        batch = inputs.detection_batch
        if batch is None:
            return
        validated, reason = self._validate_batch(batch, now, require_new_frame=True)
        if reason == "NON_NEW_FRAME":
            self._emit("OBSERVATION_IGNORED", reason,
                       details={"frame_id": batch.frame_id, "last_frame_id": self._last_frame_id})
            return
        if reason is not None:
            self._record_camera_failure(reason, {"frame_id": getattr(batch, "frame_id", None)})
            return
        assert validated is not None
        self._latest_batch = validated
        self._last_frame_id = validated.frame_id
        self._camera_failures = 0
        if not validated.detections:
            self._empty_scene_frames += 1
            self._emit("EMPTY_SCENE_OBSERVED", "VALID_EMPTY_INPUT_ROI",
                       details={"frame_id": validated.frame_id,
                                "consecutive_frames": self._empty_scene_frames})
        else:
            self._empty_scene_frames = 0
        if self._empty_scene_frames >= int(self.config.p("empty_scene_confirm_frames")):
            self._transition(State.DONE, "BATCH_EMPTY_CONFIRMED")
        else:
            self._transition(State.SELECT, "FRESH_PERCEPTION_ACCEPTED",
                             {"frame_id": validated.frame_id,
                              "detection_count": len(validated.detections)})

    def _handle_select(self, inputs: ControllerInput, now: float) -> None:
        if self._latest_batch is None:
            self._safe_stop("SELECT_WITHOUT_OBSERVATION")
            return
        if self._latest_batch.frame_id != self._last_frame_id:
            self._safe_stop("SELECT_FRAME_CONTEXT_MISMATCH")
            return
        if self._latest_batch.detections and any(d.status == "OCCLUDED"
                                                 for d in self._latest_batch.detections):
            self._scene_uncertainty_frames += 1
            self._emit("SCENE_INCOMPLETE", "OCCLUDED_TRACK_BLOCKS_SAFE_PLANNING",
                       details={"frame_id": self._latest_batch.frame_id,
                                "consecutive_frames": self._scene_uncertainty_frames})
            if self._scene_uncertainty_frames >= int(self.config.p("max_scene_uncertainty_frames")):
                self._safe_stop("SCENE_UNCERTAIN_LIMIT")
            else:
                self._transition(State.OBSERVE, "REOBSERVE_INCOMPLETE_SCENE")
            return
        self._scene_uncertainty_frames = 0
        candidates: list[Detection] = []
        unknown_pending = False
        for detection in self._latest_batch.detections:
            assert detection.track_id is not None
            record = self._track_records.setdefault(
                detection.track_id, TrackRecord(detection.track_id, detection.class_label)
            )
            record.last_seen_frame_id = self._latest_batch.frame_id
            if detection.status == "UNKNOWN" or detection.class_label == "UNKNOWN":
                record.class_label = "UNKNOWN"
                if record.last_unknown_frame_id != self._latest_batch.frame_id:
                    record.unknown_observations += 1
                    record.last_unknown_frame_id = self._latest_batch.frame_id
                    self._emit("UNKNOWN_COLOR_OBSERVED", detection.reason or "COLOR_UNKNOWN",
                               track_id=detection.track_id,
                               details={"frame_id": self._latest_batch.frame_id,
                                        "observations": record.unknown_observations})
                if record.status not in _TERMINAL_TRACK_STATUSES:
                    if record.unknown_observations >= int(self.config.p("unknown_confirm_frames")):
                        record.status = TrackStatus.SKIPPED_UNKNOWN
                        record.failure_reasons.append("UNKNOWN_COLOR_CONFIRMED")
                        self._emit("TRACK_SKIPPED", "UNKNOWN_COLOR_CONFIRMED",
                                   track_id=detection.track_id)
                    else:
                        unknown_pending = True
                continue
            record.class_label = detection.class_label
            if record.status in _TERMINAL_TRACK_STATUSES or record.status == TrackStatus.RESERVED:
                continue
            if detection.class_label not in self.config.class_to_zone:
                record.status = TrackStatus.SKIPPED_UNKNOWN
                record.failure_reasons.append("UNSUPPORTED_COLOR_CLASS")
                self._emit("TRACK_SKIPPED", "UNSUPPORTED_COLOR_CLASS",
                           track_id=detection.track_id)
                continue
            if record.attempts >= int(self.config.p("max_cycle_attempts_per_track")):
                record.status = TrackStatus.SAFE_FAILURE
                record.failure_reasons.append("TRACK_ATTEMPT_BUDGET_EXHAUSTED")
                self._emit("TRACK_SKIPPED", "TRACK_ATTEMPT_BUDGET_EXHAUSTED",
                           track_id=detection.track_id)
                continue
            candidates.append(detection)

        ranked = self._rank_candidates(candidates, inputs.current_tcp_xy_m)
        for _, detection in ranked:
            free_slots = self.reservations.free_slots(detection.class_label)
            if not free_slots:
                record = self._track_records[detection.track_id]  # type: ignore[index]
                record.status = TrackStatus.SKIPPED_FULL_ZONE
                record.failure_reasons.append("ZONE_CAPACITY_EXHAUSTED")
                self._emit("TRACK_SKIPPED", "ZONE_CAPACITY_EXHAUSTED",
                           track_id=detection.track_id,
                           details={"class_label": detection.class_label})
                continue
            if not self._valid_q(inputs.current_q) or not self._finite(inputs.current_gripper_m):
                self._safe_stop("SELECT_MISSING_ARM_STATE")
                return
            slot_id = free_slots[0]
            try:
                self.reservations.reserve(int(detection.track_id), slot_id)
            except ReservationError as exc:
                self._safe_stop("RESERVATION_CONFLICT", {"message": str(exc)})
                return
            record = self._track_records[int(detection.track_id)]
            record.status = TrackStatus.RESERVED
            record.attempts += 1
            self._active_track_id = int(detection.track_id)
            self._active_class = detection.class_label
            self._active_slot_id = slot_id
            self._source_xy = tuple(map(float, detection.xy_base_m))
            self._selection_tcp_xy = tuple(map(float, inputs.current_tcp_xy_m)) \
                if self._valid_xy(inputs.current_tcp_xy_m) else None
            self._selection_q = tuple(map(float, inputs.current_q))  # type: ignore[arg-type]
            self._selection_gripper_m = float(inputs.current_gripper_m)
            self._pending_plan_input = (self._selection_q, self._selection_gripper_m)
            self._plan = None
            self._cycle_outcome = None
            self._plan_requests.clear()
            self._emit("TARGET_SELECTED", "DETERMINISTIC_LOWEST_COST",
                       track_id=self._active_track_id, slot_id=slot_id,
                       details={"score": ranked[[item[1].track_id for item in ranked].index(detection.track_id)][0],
                                "class_label": detection.class_label,
                                "attempt": record.attempts})
            self._transition(State.PLAN, "TRACK_AND_SLOT_RESERVED")
            return

        if unknown_pending:
            self._transition(State.OBSERVE, "UNKNOWN_REQUIRES_CONFIRMATION")
        elif not self._latest_batch.detections:
            self._transition(State.OBSERVE, "EMPTY_SCENE_CONFIRMATION_PENDING")
        else:
            self._transition(State.DONE, "NO_ELIGIBLE_TRACKS_REMAIN")

    def _rank_candidates(
        self,
        candidates: list[Detection],
        current_tcp_xy_m: tuple[float, float] | None,
    ) -> list[tuple[float, Detection]]:
        if not candidates:
            return []
        distances = {
            int(d.track_id): (
                math.dist(d.xy_base_m, current_tcp_xy_m)
                if self._valid_xy(current_tcp_xy_m) else 0.0
            )
            for d in candidates
        }
        maximum_distance = max(max(distances.values()), 1e-9)
        w_conf, w_xy, w_yaw, w_motion, w_fail = self.config.score_weights
        ranked: list[tuple[float, Detection]] = []
        for detection in candidates:
            record = self._track_records[int(detection.track_id)]
            xy_ratio = min(1.0, float(detection.position_sigma_m) /
                           self.config.position_sigma_limit_m)
            yaw_ratio = min(1.0, float(detection.yaw_sigma_rad) /
                            self.config.yaw_sigma_limit_rad)
            motion_ratio = distances[int(detection.track_id)] / maximum_distance
            failure_ratio = min(1.0, len(record.failure_reasons) /
                                int(self.config.p("max_cycle_attempts_per_track")))
            score = (
                w_conf * (1.0 - float(detection.confidence))
                + w_xy * xy_ratio
                + w_yaw * yaw_ratio
                + w_motion * motion_ratio
                + w_fail * failure_ratio
            )
            ranked.append((score, detection))
        ranked.sort(key=lambda item: (item[0], int(item[1].track_id)))
        return ranked

    def _handle_plan(self, inputs: ControllerInput, now: float) -> None:
        response = inputs.plan_response
        if response is None:
            return
        if response.request_id != self._pending_plan_request_id:
            self._emit("PLAN_RESPONSE_IGNORED", "STALE_REQUEST_ID",
                       details={"received": response.request_id,
                                "pending": self._pending_plan_request_id})
            return
        result = response.result
        if result.code == PlanCode.SUCCESS:
            if result.plan is None:
                self._safe_stop("PLAN_SUCCESS_WITHOUT_PLAN")
                return
            if self._latest_batch is None or not self._batch_is_fresh(self._latest_batch, now):
                self._handle_plan_failure(PlanResult(
                    PlanCode.STALE_DETECTION,
                    "Perception snapshot expired while awaiting M5.",
                    None, None, {"controller_rejected_stale_plan_response": True},
                ), now)
                return
            validation_error = self._validate_plan(result.plan)
            if validation_error is not None:
                self._safe_stop("M5_PLAN_CONTRACT_MISMATCH", {"detail": validation_error})
                return
            self._plan = result.plan
            self._cycle_started_s = now
            self._cycle_deadline_s = now + (
                result.plan.duration_s * float(self.config.p("cycle_timeout_factor"))
                + float(self.config.p("cycle_timeout_margin_s"))
            )
            self._emit("PLAN_ACCEPTED", "FULL_M5_PREFLIGHT_SUCCESS",
                       track_id=self._active_track_id, slot_id=self._active_slot_id,
                       details={"plan_id": result.plan.plan_id,
                                "duration_s": result.plan.duration_s,
                                "cycle_deadline_s": self._cycle_deadline_s})
            self._transition(State.APPROACH, "VALID_FULL_PLAN")
            return
        self._handle_plan_failure(result, now)

    def _validate_plan(self, plan: Plan) -> str | None:
        if self._active_track_id is None or self._active_class is None or self._active_slot_id is None:
            return "active target/slot is absent"
        if plan.target_track_id != self._active_track_id:
            return "plan target track does not match reserved M4 track"
        if plan.target_class != self._active_class:
            return "plan target class does not match M4 class"
        if plan.slot_id != self._active_slot_id:
            return "plan slot does not match reserved placement slot"
        if plan.ssot_version != self.config.config_version:
            return "plan SSOT version differs from controller config"
        if plan.ssot_sha256 != self.config.ssot_sha256:
            return "plan SSOT hash differs from controller config"
        if plan.model_sha256 != self.config.model_sha256:
            return "plan model hash differs from controller config"
        if self._latest_batch is None or (
            plan.perception_frame_id != self._latest_batch.frame_id
            or not math.isclose(plan.perception_time_s, self._latest_batch.simulation_time_s,
                                rel_tol=0.0, abs_tol=1e-9)
        ):
            return "plan was not produced from the reserved fresh perception snapshot"
        if not plan.plan_id or not math.isfinite(plan.duration_s) or plan.duration_s <= 0:
            return "plan identity or duration is invalid"
        expected_phases = tuple(phase.value for phase in Phase)
        if tuple(plan.phases) != expected_phases:
            return "plan phase sequence differs from the complete ordered M5 phase contract"
        phase_order = {phase: index for index, phase in enumerate(expected_phases)}
        sample_phase_indexes = []
        for sample in plan.samples:
            if sample.phase not in phase_order:
                return "trajectory contains a phase outside the M5 phase contract"
            sample_phase_indexes.append(phase_order[sample.phase])
            if (
                len(sample.q) != 4
                or any(not math.isfinite(float(value)) for value in sample.q)
                or len(sample.tcp_xyzyaw) != 4
                or any(not math.isfinite(float(value)) for value in sample.tcp_xyzyaw)
                or not math.isfinite(float(sample.gripper_m))
            ):
                return "trajectory contains malformed joint, TCP, or gripper coordinates"
        if not PLAN_REQUIRED_PHASES.issubset(set(plan.phases)) or not PLAN_REQUIRED_PHASES.issubset(
            {sample.phase for sample in plan.samples}
        ):
            return "full pick/place/retreat phase set is incomplete"
        if any(b < a for a, b in zip(sample_phase_indexes, sample_phase_indexes[1:])):
            return "trajectory samples move backwards through the M5 phase sequence"
        times = [float(sample.time_s) for sample in plan.samples]
        if not times or any(not math.isfinite(value) for value in times):
            return "trajectory contains no finite sample times"
        if any(b < a for a, b in zip(times, times[1:])):
            return "trajectory sample times are not monotonic"
        if abs(times[-1] - plan.duration_s) > max(1e-6, plan.duration_s * 1e-6):
            return "plan duration and final sample time disagree"
        if any(event.track_id != self._active_track_id or event.slot_id != self._active_slot_id
               for event in plan.events):
            return "plan events do not share the reserved track/slot"
        required_markers = {
            "OBJECT_ATTACHED": Phase.LIFT.value,
            "OBJECT_RELEASED": Phase.RELEASE.value,
        }
        for marker, phase_name in required_markers.items():
            matching_events = [event for event in plan.events if event.event == marker]
            matching_samples = [sample for sample in plan.samples if sample.event == marker]
            if len(matching_events) != 1 or len(matching_samples) != 1:
                return f"plan requires exactly one {marker} event and trajectory marker"
            event = matching_events[0]
            sample = matching_samples[0]
            if (
                event.phase != phase_name or sample.phase != phase_name
                or not math.isfinite(float(event.time_s))
                or not math.isclose(event.time_s, sample.time_s, rel_tol=0.0, abs_tol=1e-9)
            ):
                return f"{marker} event is inconsistent with its M5 phase/sample"
        attach_time = next(event.time_s for event in plan.events
                           if event.event == "OBJECT_ATTACHED")
        release_time = next(event.time_s for event in plan.events
                            if event.event == "OBJECT_RELEASED")
        if attach_time >= release_time:
            return "object attachment must precede release in the M5 plan"
        return None

    def _handle_plan_failure(self, result: PlanResult, now: float) -> None:
        code = result.code
        record = self._active_record()
        if code in PLAN_FATAL:
            self._safe_stop("FATAL_PLANNER_CONTRACT_ERROR",
                            {"plan_code": code.value, "reason": result.reason})
            return
        if code in PLAN_RETRYABLE and record is not None:
            failed_track = self._active_track_id
            failed_slot = self._active_slot_id
            can_retry = record.plan_retries < int(self.config.p("max_plan_retries"))
            record.failure_reasons.append(code.value)
            if can_retry:
                record.plan_retries += 1
                record.status = TrackStatus.PENDING
                self._emit("PLAN_RETRY_SCHEDULED", code.value,
                           track_id=self._active_track_id, slot_id=self._active_slot_id,
                           details={"retry": record.plan_retries,
                                    "max_retries": self.config.p("max_plan_retries")})
            else:
                record.status = TrackStatus.SKIPPED_UNREACHABLE
                self._emit("TRACK_SKIPPED", f"PLAN_{code.value}_RETRY_EXHAUSTED",
                           track_id=self._active_track_id)
            self._release_active_reservations(f"PLAN_{code.value}")
            self._recovery_reason = code.value
            self._recovery_requires_feedback = False
            self._recovery_next_state = State.OBSERVE
            self._transition(State.RECOVER, f"PLAN_{code.value}",
                             {"retryable": can_retry, "phase": result.phase})
            return

        failed_track = self._active_track_id
        failed_slot = self._active_slot_id
        if record is not None:
            record.failure_reasons.append(code.value)
            if code in PLAN_SLOT_CONFLICT:
                record.status = TrackStatus.SKIPPED_FULL_ZONE
                if code in {PlanCode.SLOT_OCCUPIED, PlanCode.SLOT_INVALID} and self._active_slot_id:
                    try:
                        self.reservations.quarantine(
                            self._active_slot_id, int(self._active_track_id), f"PLANNER_{code.value}"
                        )
                        self.reservations.release_track(int(self._active_track_id))
                    except ReservationError:
                        pass
                    self._release_active_reservations(f"PLANNER_{code.value}")
                else:
                    self._release_active_reservations(f"PLANNER_{code.value}")
            else:
                record.status = TrackStatus.SKIPPED_UNREACHABLE
                self._release_active_reservations(f"PLANNER_{code.value}")
        self._emit("PLAN_REJECTED", code.value, track_id=failed_track,
                   slot_id=failed_slot,
                   details={"reason": result.reason, "phase": result.phase,
                            "diagnostics": result.diagnostics})
        self._recovery_reason = code.value
        self._recovery_requires_feedback = False
        self._recovery_next_state = State.OBSERVE
        self._transition(State.RECOVER, f"PLAN_{code.value}")

    def _handle_execution(self, inputs: ControllerInput, now: float) -> None:
        feedback = inputs.execution_feedback
        if self.state in {State.TRANSFER, State.PLACE}:
            if self._holding_state is not True:
                reason = (
                    "PUBLIC_HOLD_SIGNAL_LOST" if self._holding_state is False
                    else "PUBLIC_HOLD_SIGNAL_STALE_OR_UNKNOWN"
                )
                self._handle_lost_payload(now, reason)
                return
        if feedback is None:
            return
        if feedback.command_id != self._active_command_id:
            self._emit("EXECUTION_FEEDBACK_IGNORED", "COMMAND_ID_MISMATCH",
                       details={"received": feedback.command_id,
                                "pending": self._active_command_id})
            return
        if not self._fresh_timestamp(feedback.simulation_time_s, now,
                                     minimum_time_s=self._state_started_s or now):
            self._emit("EXECUTION_FEEDBACK_IGNORED", "STALE_OR_FUTURE_FEEDBACK",
                       details={"feedback_time_s": feedback.simulation_time_s})
            return
        if feedback.status == ExecutionStatus.FAILED:
            self._handle_execution_failure(feedback, now)
            return
        if feedback.status == ExecutionStatus.RUNNING:
            return
        if feedback.status != ExecutionStatus.COMPLETED:
            self._safe_stop("UNKNOWN_EXECUTION_STATUS")
            return
        if self.state == State.APPROACH:
            self._transition(State.DESCEND, "APPROACH_COMPLETE")
        elif self.state == State.DESCEND:
            self._transition(State.GRASP, "DESCENT_COMPLETE")
        elif self.state == State.GRASP:
            self._transition(State.VERIFY_HOLD, "GRIPPER_CLOSE_COMMAND_COMPLETE")
        elif self.state == State.TRANSFER:
            if not self._fresh_grasp_evidence(
                inputs.grasp_evidence, now, minimum_time_s=feedback.simulation_time_s
            ) or inputs.grasp_evidence.holding is not True:  # type: ignore[union-attr]
                self._emit("TRANSFER_WAITING_FOR_HOLD_EVIDENCE", "FRESH_HOLD_REQUIRED",
                           track_id=self._active_track_id)
                self._completed_command_at_s = feedback.simulation_time_s
                return
            self._holding_state = True
            self._transition(State.PLACE, "TRANSFER_COMPLETE_HOLD_CONFIRMED")
        elif self.state == State.PLACE:
            if not self._fresh_grasp_evidence(
                inputs.grasp_evidence, now, minimum_time_s=feedback.simulation_time_s
            ) or inputs.grasp_evidence.holding is not True:  # type: ignore[union-attr]
                self._emit("PLACE_WAITING_FOR_HOLD_EVIDENCE", "FRESH_HOLD_REQUIRED",
                           track_id=self._active_track_id)
                self._completed_command_at_s = feedback.simulation_time_s
                return
            self._holding_state = True
            self._transition(State.RELEASE, "PLACE_DESCENT_COMPLETE")
        elif self.state == State.RELEASE:
            if not self._fresh_grasp_evidence(
                inputs.grasp_evidence, now, minimum_time_s=feedback.simulation_time_s
            ) or inputs.grasp_evidence.holding is not False:  # type: ignore[union-attr]
                self._release_completed_at_s = feedback.simulation_time_s
                self._completed_command_at_s = feedback.simulation_time_s
                self._emit("RELEASE_WAITING_FOR_EMPTY_GRIPPER", "FRESH_RELEASE_EVIDENCE_REQUIRED",
                           track_id=self._active_track_id)
                return
            self._holding_state = False
            self._release_completed_at_s = feedback.simulation_time_s
            self._transition(State.VERIFY_PLACE, "RELEASE_COMMAND_AND_SENSOR_CONFIRMED")
        elif self.state == State.RETREAT:
            self._finish_cycle(now)

    def _handle_verify_hold(self, inputs: ControllerInput, now: float) -> None:
        if self._lift_completed_at_s is None:
            feedback = inputs.execution_feedback
            if feedback is None:
                return
            if feedback.command_id != self._active_command_id:
                self._emit("EXECUTION_FEEDBACK_IGNORED", "COMMAND_ID_MISMATCH",
                           details={"received": feedback.command_id,
                                    "pending": self._active_command_id})
                return
            if not self._fresh_timestamp(feedback.simulation_time_s, now,
                                         minimum_time_s=self._state_started_s or now):
                return
            if feedback.status == ExecutionStatus.FAILED:
                self._handle_execution_failure(feedback, now)
                return
            if feedback.status != ExecutionStatus.COMPLETED:
                return
            self._lift_completed_at_s = feedback.simulation_time_s
            self._active_command_id = None
            self._state_started_s = now
            self._issue_observation_request()
            self._emit("TEST_LIFT_COMPLETE", "WAITING_FOR_PUBLIC_HOLD_AND_SOURCE_EVIDENCE",
                       track_id=self._active_track_id)
            return
        evidence = inputs.grasp_evidence
        if evidence is None or not self._fresh_grasp_evidence(
            evidence, now, minimum_time_s=self._lift_completed_at_s
        ):
            return
        if evidence.holding is False:
            self._holding_state = False
            self._handle_grasp_failure(now, "NO_HOLD_AFTER_TEST_LIFT")
            return
        if evidence.holding is not True:
            return
        if inputs.detection_batch is None:
            return
        batch, reason = self._validate_batch(inputs.detection_batch, now, require_new_frame=True)
        if reason == "NON_NEW_FRAME":
            return
        if reason is not None or batch is None:
            return
        self._latest_batch = batch
        self._last_frame_id = batch.frame_id
        target_at_source = any(
            math.dist(d.xy_base_m, self._source_xy) <= self.config.tracking_max_distance_m
            for d in batch.detections
        ) if self._source_xy is not None else True
        if target_at_source:
            self._safe_stop("HOLD_SENSOR_CONFLICTS_WITH_SOURCE_OBSERVATION",
                            {"frame_id": batch.frame_id})
            return
        self._holding_state = True
        self._emit("HOLD_CONFIRMED", "FINGER_SIGNAL_AND_SOURCE_ABSENCE",
                   track_id=self._active_track_id,
                   details={"source": evidence.source, "frame_id": batch.frame_id})
        self._transition(State.TRANSFER, "HOLD_CONFIRMED_AFTER_TEST_LIFT")

    def _handle_verify_place(self, inputs: ControllerInput, now: float) -> None:
        evidence = inputs.placement_evidence
        if evidence is None:
            return
        if evidence.frame_id <= self._last_place_frame_id or evidence.frame_id < self._observe_request_frame_id:
            self._emit("PLACE_EVIDENCE_IGNORED", "NON_NEW_PUBLIC_FRAME",
                       details={"frame_id": evidence.frame_id,
                                "minimum_frame_id": self._observe_request_frame_id})
            return
        if evidence.source != "PUBLIC_CAMERA_PERCEPTION":
            self._emit("PLACE_EVIDENCE_IGNORED", "FORBIDDEN_EVIDENCE_SOURCE",
                       details={"source": evidence.source})
            return
        if not self._fresh_timestamp(evidence.simulation_time_s, now,
                                     minimum_time_s=self._release_completed_at_s or now):
            self._emit("PLACE_EVIDENCE_IGNORED", "STALE_OR_FUTURE_PLACE_EVIDENCE",
                       details={"frame_id": evidence.frame_id,
                                "timestamp_s": evidence.simulation_time_s})
            return
        self._last_place_frame_id = evidence.frame_id
        if evidence.status == EvidenceStatus.AMBIGUOUS or evidence.status == EvidenceStatus.NOT_CONFIRMED:
            self._emit("PLACE_EVIDENCE_INCONCLUSIVE", evidence.reason_code or evidence.status.value,
                       track_id=self._active_track_id, slot_id=self._active_slot_id,
                       details={"frame_id": evidence.frame_id})
            return
        if evidence.status != EvidenceStatus.CONFIRMED:
            self._emit("PLACE_EVIDENCE_IGNORED", "UNKNOWN_PLACE_EVIDENCE_STATUS")
            return
        confidence_valid = (
            evidence.confidence is not None and math.isfinite(evidence.confidence)
            and evidence.confidence >= float(self.config.p("place_evidence_min_confidence"))
        )
        expected = (
            evidence.observed_zone_id == self.config.class_to_zone.get(self._active_class or "")
            and evidence.observed_slot_id == self._active_slot_id
            and evidence.observed_class == self._active_class
            and confidence_valid
        )
        if expected:
            self._confirm_placement()
            self._transition(State.RETREAT, "PUBLIC_PLACE_EVIDENCE_MATCHED")
            return
        self._quarantine_active_slot("PLACE_EVIDENCE_MISMATCH")
        if evidence.observed_slot_id in self.config.slot_classes:
            try:
                if evidence.observed_slot_id != self._active_slot_id \
                        and self.reservations.status(evidence.observed_slot_id) == SlotStatus.FREE:
                    self.reservations.quarantine(
                        evidence.observed_slot_id, int(self._active_track_id), "OBSERVED_WRONG_PLACEMENT"
                    )
            except ReservationError:
                pass
        record = self._active_record()
        if record is not None:
            record.status = TrackStatus.SAFE_FAILURE
            record.failure_reasons.append("PLACE_EVIDENCE_MISMATCH")
        self._cycle_outcome = "SAFE_FAILURE"
        self._emit("PLACEMENT_NOT_CONFIRMED", "PLACE_EVIDENCE_MISMATCH",
                   track_id=self._active_track_id, slot_id=self._active_slot_id,
                   details={"observed_zone_id": evidence.observed_zone_id,
                            "observed_slot_id": evidence.observed_slot_id,
                            "observed_class": evidence.observed_class,
                            "confidence": evidence.confidence})
        self._transition(State.RETREAT, "UNCONFIRMED_PLACE_RETREAT")

    def _handle_recover(self, inputs: ControllerInput, now: float) -> None:
        if not self._recovery_requires_feedback:
            self._recovery_requires_feedback = False
            self._transition(self._recovery_next_state, "BOUNDED_RECOVERY_COMPLETE")
            return
        feedback = inputs.recovery_feedback
        if feedback is None:
            return
        if feedback.request_id != self._recovery_request_id:
            self._emit("RECOVERY_FEEDBACK_IGNORED", "STALE_REQUEST_ID",
                       details={"received": feedback.request_id,
                                "pending": self._recovery_request_id})
            return
        if not self._fresh_timestamp(feedback.simulation_time_s, now,
                                     minimum_time_s=self._state_started_s or now):
            return
        if (
            feedback.status != RecoveryStatus.COMPLETED
            or not feedback.arm_state_valid
            or not feedback.gripper_state_known
            or feedback.holding is not False
        ):
            self._mark_recovery_failure("RECOVERY_FAILED_OR_STATE_UNKNOWN")
            self._safe_stop("RECOVERY_FAILED_OR_STATE_UNKNOWN",
                            {"reason_code": feedback.reason_code,
                             "holding": feedback.holding})
            return
        self._holding_state = False
        self._release_active_reservations("RECOVERY_COMPLETED_NO_PAYLOAD")
        self._recovery_requires_feedback = False
        self._recovery_attempts = 0
        self._active_command_id = None
        self._plan = None
        self._cycle_deadline_s = None
        self._transition(self._recovery_next_state, "RECOVERY_COMPLETED")

    def _handle_execution_failure(self, feedback, now: float) -> None:
        reason = feedback.reason_code or "EXECUTION_FAILED"
        payload_possible = (
            self.state in {State.GRASP, State.VERIFY_HOLD}
            and self._holding_state is not False
        )
        if (
            self._holding_state is True or payload_possible
            or self.state in {State.TRANSFER, State.PLACE, State.RELEASE}
        ):
            self._safe_stop("EXECUTION_FAILED_WITH_POSSIBLE_PAYLOAD",
                            {"state": self.state.value, "reason": reason})
            return
        record = self._active_record()
        if record is not None:
            record.failure_reasons.append(reason)
            if record.attempts >= int(self.config.p("max_cycle_attempts_per_track")):
                record.status = TrackStatus.SAFE_FAILURE
                self._cycle_outcome = "SAFE_FAILURE"
            else:
                record.status = TrackStatus.PENDING
        self._recovery_reason = reason
        self._recovery_action = "OPEN_AND_RETREAT_NO_PAYLOAD"
        self._recovery_requires_feedback = True
        self._recovery_next_state = State.OBSERVE
        self._emit("EXECUTION_FAILED", reason, track_id=self._active_track_id,
                   slot_id=self._active_slot_id, details={"state": self.state.value})
        self._release_active_reservations(f"EXECUTION_FAILURE_NO_PAYLOAD_{reason}")
        self._transition(State.RECOVER, "EXECUTION_FAILURE_NO_PAYLOAD")

    def _handle_grasp_failure(self, now: float, reason: str) -> None:
        record = self._active_record()
        if record is None:
            self._safe_stop("GRASP_FAILURE_WITHOUT_TRACK")
            return
        record.grasp_attempts += 1
        record.failure_reasons.append(reason)
        retry = (
            record.grasp_attempts < int(self.config.p("max_grasp_attempts"))
            and record.attempts < int(self.config.p("max_cycle_attempts_per_track"))
        )
        record.status = TrackStatus.PENDING if retry else TrackStatus.SAFE_FAILURE
        self._cycle_outcome = None if retry else "SAFE_FAILURE"
        self._recovery_reason = reason
        self._recovery_action = "OPEN_AND_RETREAT_NO_PAYLOAD"
        self._recovery_requires_feedback = True
        self._recovery_next_state = State.OBSERVE
        self._emit("GRASP_FAILED", reason, track_id=self._active_track_id,
                   slot_id=self._active_slot_id,
                   details={"grasp_attempt": record.grasp_attempts,
                            "retry": retry,
                            "max_attempts": self.config.p("max_grasp_attempts")})
        self._release_active_reservations(f"GRASP_FAILURE_NO_PAYLOAD_{reason}")
        self._transition(State.RECOVER, "GRASP_NOT_HELD")

    def _handle_lost_payload(self, now: float, reason: str) -> None:
        if self._active_slot_id is not None and self._active_track_id is not None:
            self._quarantine_active_slot(reason)
        record = self._active_record()
        if record is not None:
            record.status = TrackStatus.SAFE_FAILURE
            record.failure_reasons.append(reason)
        self._cycle_outcome = "SAFE_FAILURE"
        self._safe_stop(reason, {"state": self.state.value,
                                 "track_id": self._active_track_id,
                                 "slot_id": self._active_slot_id})

    def _mark_recovery_failure(self, reason: str) -> None:
        track_id = self._active_track_id
        if track_id is None and self._plan is not None:
            track_id = self._plan.target_track_id
        record = self._track_records.get(track_id) if track_id is not None else None
        if record is not None and record.status not in _TERMINAL_TRACK_STATUSES:
            record.status = TrackStatus.SAFE_FAILURE
            record.failure_reasons.append(reason)
        self._cycle_outcome = "SAFE_FAILURE"

    def _handle_state_timeout(self, now: float) -> None:
        if self.state == State.OBSERVE:
            self._record_camera_failure("OBSERVATION_TIMEOUT", {})
        elif self.state == State.INIT:
            self._safe_stop("INIT_TIMEOUT")
        elif self.state == State.SELECT:
            self._safe_stop("SELECT_TIMEOUT")
        elif self.state == State.PLAN:
            synthetic = PlanResult(
                PlanCode.TIMEOUT, "No M5 response before simulation-time deadline.",
                None, None, {"controller_timeout": True},
            )
            self._handle_plan_failure(synthetic, self._event_clock_s)
        elif self.state == State.VERIFY_PLACE:
            self._quarantine_active_slot("PLACE_VERIFICATION_TIMEOUT")
            record = self._active_record()
            if record is not None:
                record.status = TrackStatus.SAFE_FAILURE
                record.failure_reasons.append("PLACE_VERIFICATION_TIMEOUT")
            self._cycle_outcome = "SAFE_FAILURE"
            self._emit("PLACE_NOT_VERIFIED", "PLACE_VERIFICATION_TIMEOUT",
                       track_id=self._active_track_id, slot_id=self._active_slot_id)
            self._transition(State.RETREAT, "PLACE_UNVERIFIED_SAFE_RETREAT")
        elif self.state == State.VERIFY_HOLD:
            self._safe_stop("HOLD_EVIDENCE_TIMEOUT_UNKNOWN",
                            {"holding_state": self._holding_state,
                             "lift_completed_at_s": self._lift_completed_at_s})
        elif self.state == State.RELEASE:
            self._safe_stop("RELEASE_STATE_UNCERTAIN",
                            {"holding_state": self._holding_state})
        elif self.state == State.RECOVER:
            self._mark_recovery_failure("RECOVERY_TIMEOUT")
            self._safe_stop("RECOVERY_TIMEOUT",
                            {"recovery_reason": self._recovery_reason})
        elif self.state in EXECUTION_PHASES:
            self._on_active_cycle_timeout(now, f"{self.state.value}_TIMEOUT")
        else:
            self._safe_stop("UNHANDLED_STATE_TIMEOUT", {"state": self.state.value})

    def _on_active_cycle_timeout(self, now: float, reason: str) -> None:
        if self._holding_state is not False:
            self._safe_stop(reason, {"holding_state": self._holding_state})
            return
        record = self._active_record()
        if record is not None:
            record.failure_reasons.append(reason)
            record.status = TrackStatus.SAFE_FAILURE
        self._cycle_outcome = "SAFE_FAILURE"
        self._recovery_reason = reason
        self._recovery_action = "OPEN_AND_RETREAT_NO_PAYLOAD"
        self._recovery_requires_feedback = True
        self._recovery_next_state = State.OBSERVE
        self._emit("CYCLE_TIMEOUT_NO_PAYLOAD", reason,
                   track_id=self._active_track_id, slot_id=self._active_slot_id)
        self._release_active_reservations(f"CYCLE_TIMEOUT_NO_PAYLOAD_{reason}")
        self._transition(State.RECOVER, reason)

    def _record_camera_failure(self, reason: str, details: dict) -> None:
        self._camera_failures += 1
        self._emit("PERCEPTION_FAILURE", reason,
                   details={**details, "consecutive_failures": self._camera_failures})
        if self._camera_failures >= int(self.config.p("max_camera_failures")):
            self._safe_stop("CAMERA_FAILURE_LIMIT", {"last_reason": reason})
            return
        self._state_started_s = self._event_clock_s
        self._issue_observation_request()

    def _validate_batch(
        self, batch: DetectionBatch, now: float, *, require_new_frame: bool
    ) -> tuple[DetectionBatch | None, str | None]:
        if not isinstance(batch, DetectionBatch):
            return None, "INVALID_BATCH_TYPE"
        if not isinstance(batch.frame_id, int) or isinstance(batch.frame_id, bool) or batch.frame_id < 0:
            return None, "INVALID_FRAME_ID"
        if require_new_frame and batch.frame_id <= self._last_frame_id:
            return None, "NON_NEW_FRAME"
        if not self._finite(batch.simulation_time_s):
            return None, "INVALID_FRAME_TIMESTAMP"
        age = now - float(batch.simulation_time_s)
        freshness_limit = min(self.config.sensor_max_age_s, self.config.planner_max_detection_age_s)
        if age < -1e-12 or age > freshness_limit:
            return None, "STALE_OR_FUTURE_FRAME"
        if batch.status == "NO_CANDIDATES":
            if batch.detections:
                return None, "EMPTY_STATUS_WITH_DETECTIONS"
            return batch, None
        if batch.status != "OK" or not batch.detections:
            return None, "INVALID_OR_INCONSISTENT_BATCH_STATUS"
        groups: dict[int, list[Detection]] = {}
        for detection in batch.detections:
            if not isinstance(detection.track_id, int) or isinstance(detection.track_id, bool):
                return None, "TRACK_ID_MISSING_OR_INVALID"
            if detection.frame_id != batch.frame_id:
                return None, "DETECTION_FRAME_MISMATCH"
            stamp_age = now - float(detection.simulation_time_s)
            if not math.isfinite(stamp_age) or stamp_age < -1e-12 or stamp_age > freshness_limit:
                return None, "STALE_OR_FUTURE_DETECTION"
            if detection.frame_age_s is not None and (
                not math.isfinite(detection.frame_age_s)
                or detection.frame_age_s < 0
                or detection.frame_age_s > freshness_limit
            ):
                return None, "DETECTION_FRAME_AGE_INVALID"
            if detection.status not in {"VALID", "UNKNOWN", "OCCLUDED"}:
                return None, "UNSUPPORTED_DETECTION_STATUS"
            if detection.status == "VALID" and detection.class_label not in self.config.class_to_zone:
                return None, "VALID_TRACK_HAS_UNSUPPORTED_CLASS"
            if detection.status == "UNKNOWN" and detection.class_label != "UNKNOWN":
                return None, "UNKNOWN_STATUS_CLASS_MISMATCH"
            if len(detection.xy_base_m) != 2 or not all(
                math.isfinite(float(value)) for value in detection.xy_base_m
            ):
                return None, "INVALID_TRACK_XY"
            fields = (detection.confidence, detection.position_sigma_m, detection.yaw_sigma_rad)
            if any(not math.isfinite(float(value)) for value in fields):
                return None, "INVALID_TRACK_QUALITY"
            if not 0 <= detection.confidence <= 1 or detection.position_sigma_m < 0 \
                    or detection.yaw_sigma_rad < 0:
                return None, "TRACK_QUALITY_OUT_OF_RANGE"
            if detection.status in {"VALID", "UNKNOWN"} and (
                detection.yaw_base_rad is None or not math.isfinite(float(detection.yaw_base_rad))
            ):
                return None, "MISSING_TRACK_ORIENTATION"
            groups.setdefault(detection.track_id, []).append(detection)

        normalized: list[Detection] = []
        for track_id, duplicates in groups.items():
            if len(duplicates) == 1:
                normalized.append(duplicates[0])
                continue
            labels = {item.class_label for item in duplicates}
            if len(labels) > 1 or any(
                math.dist(a.xy_base_m, b.xy_base_m) > self.config.tracking_max_distance_m
                for i, a in enumerate(duplicates) for b in duplicates[i + 1:]
            ):
                return None, "CONFLICTING_DUPLICATE_TRACK"
            selected = min(duplicates, key=lambda item: (
                -float(item.confidence), float(item.position_sigma_m),
                float(item.yaw_sigma_rad), tuple(item.xy_base_m),
            ))
            normalized.append(selected)
            self._emit("DUPLICATE_TRACK_COLLAPSED", "SAME_M4_TRACK_ID",
                       track_id=track_id,
                       details={"input_detections": len(duplicates),
                                "selected_frame_id": selected.frame_id})
        normalized.sort(key=lambda item: item.track_id or -1)
        return replace(batch, detections=tuple(normalized)), None

    def _batch_is_fresh(self, batch: DetectionBatch, now: float) -> bool:
        if not self._finite(batch.simulation_time_s):
            return False
        age = now - float(batch.simulation_time_s)
        limit = min(self.config.sensor_max_age_s, self.config.planner_max_detection_age_s)
        return -1e-12 <= age <= limit

    def _validate_plan_response_age(self, timestamp_s: float, now: float) -> bool:
        return self._fresh_timestamp(timestamp_s, now,
                                     minimum_time_s=self._state_started_s or now)

    def _fresh_grasp_evidence(self, evidence, now: float, *, minimum_time_s: float) -> bool:
        if evidence is None or not evidence.valid:
            return False
        if evidence.source not in {"FINGER_CONTACT_SENSOR", "GRIPPER_EFFORT_PROXY"}:
            return False
        return self._fresh_timestamp(
            evidence.simulation_time_s, now,
            max_age_s=float(self.config.p("hold_evidence_max_age_s")),
            minimum_time_s=minimum_time_s,
        )

    def _accept_public_holding_signal(self, inputs: ControllerInput, now: float) -> None:
        evidence = inputs.grasp_evidence
        if evidence is not None and evidence.holding is not None and self._fresh_grasp_evidence(
            evidence, now, minimum_time_s=now - float(self.config.p("hold_evidence_max_age_s"))
        ):
            self._holding_state = evidence.holding
            self._holding_evidence_time_s = float(evidence.simulation_time_s)
        elif evidence is not None and evidence.source not in {
            "FINGER_CONTACT_SENSOR", "GRIPPER_EFFORT_PROXY"
        }:
            self._emit("SENSOR_EVIDENCE_IGNORED", "FORBIDDEN_HOLD_EVIDENCE_SOURCE",
                       details={"source": evidence.source})
        if (
            self._holding_evidence_time_s is not None
            and now - self._holding_evidence_time_s > float(self.config.p("hold_evidence_max_age_s"))
        ):
            self._holding_state = None
            self._holding_evidence_time_s = None

    def _fresh_timestamp(
        self, timestamp_s: float, now: float, *, minimum_time_s: float, max_age_s: float | None = None
    ) -> bool:
        if not math.isfinite(timestamp_s):
            return False
        age = now - timestamp_s
        max_age = self.config.sensor_max_age_s if max_age_s is None else max_age_s
        return (
            timestamp_s >= minimum_time_s - 1e-12
            and -1e-12 <= age <= max_age
        )

    def _phase_duration(self, names: tuple[str, ...]) -> float:
        if self._plan is None:
            return 0.0
        by_phase: dict[str, list[float]] = {}
        for sample in self._plan.samples:
            by_phase.setdefault(sample.phase, []).append(float(sample.time_s))
        duration = 0.0
        for name in names:
            times = by_phase.get(name, [])
            if times:
                duration += max(times) - min(times)
        return duration

    def _state_timeout_s(self) -> float | None:
        if self.state == State.INIT:
            return float(self.config.p("init_timeout_s"))
        if self.state == State.OBSERVE:
            return float(self.config.p("observation_timeout_s"))
        if self.state == State.SELECT:
            return float(self.config.p("select_timeout_s"))
        if self.state == State.PLAN:
            return float(self.config.p("plan_response_timeout_s"))
        if self.state == State.RECOVER:
            return float(self.config.p("recovery_timeout_s"))
        if self.state == State.VERIFY_PLACE:
            return float(self.config.p("place_verification_timeout_s"))
        if self.state == State.VERIFY_HOLD and self._lift_completed_at_s is not None:
            return float(self.config.p("hold_verification_timeout_s"))
        if self.state == State.RELEASE and self._release_completed_at_s is not None:
            return float(self.config.p("gripper_release_confirmation_timeout_s"))
        if self.state in EXECUTION_PHASES:
            duration = self._phase_duration(EXECUTION_PHASES[self.state])
            return max(
                float(self.config.p("execution_timeout_min_s")),
                duration * float(self.config.p("execution_timeout_factor"))
                + float(self.config.p("execution_timeout_margin_s")),
            )
        return None

    def _validate_q(self, q) -> bool:
        return q is not None and len(q) == 4 and all(self._finite(item) for item in q)

    @staticmethod
    def _valid_q(q) -> bool:
        return q is not None and len(q) == 4 and all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in q
        )

    @staticmethod
    def _valid_xy(xy) -> bool:
        return xy is not None and len(xy) == 2 and all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in xy
        )

    @staticmethod
    def _finite(value) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))

    def _active_record(self) -> TrackRecord | None:
        if self._active_track_id is None:
            return None
        return self._track_records.get(self._active_track_id)

    def _release_active_reservations(self, reason: str) -> None:
        if self._active_track_id is None:
            return
        track_id = self._active_track_id
        slot_id = self._active_slot_id
        if slot_id is not None and self.reservations.status(slot_id) == SlotStatus.RESERVED:
            self.reservations.release_cycle(track_id, slot_id, reason)
        elif self.reservations.reserved_track_id == track_id:
            self.reservations.release_track(track_id)
        self._active_track_id = None
        self._active_class = None
        self._active_slot_id = None
        self._source_xy = None
        self._selection_tcp_xy = None
        self._selection_q = None
        self._selection_gripper_m = None
        self._pending_plan_input = None
        self._pending_plan_request_id = None

    def _quarantine_active_slot(self, reason: str) -> None:
        if self._active_track_id is None or self._active_slot_id is None:
            return
        if self.reservations.status(self._active_slot_id) == SlotStatus.RESERVED:
            self.reservations.quarantine_cycle(
                self._active_track_id, self._active_slot_id, reason
            )

    def _confirm_placement(self) -> None:
        if self._active_track_id is None or self._active_slot_id is None:
            self._safe_stop("PLACEMENT_CONFIRMATION_WITHOUT_RESERVATION")
            return
        track_id, slot_id = self._active_track_id, self._active_slot_id
        self.reservations.complete_cycle(track_id, slot_id, "PUBLIC_PERCEPTION_CONFIRMED")
        record = self._track_records[track_id]
        record.status = TrackStatus.PLACED_CONTROLLER_CONFIRMED
        self._placed_slot_by_track[track_id] = slot_id
        self._cycle_outcome = "PLACED_CONTROLLER_CONFIRMED"
        self._emit("PLACEMENT_CONTROLLER_CONFIRMED", "PUBLIC_PERCEPTION_MATCHED_CLASS_ZONE_SLOT",
                   track_id=track_id, slot_id=slot_id)
        self._active_track_id = None
        self._active_class = None
        self._active_slot_id = None
        self._source_xy = None
        self._selection_tcp_xy = None
        self._selection_q = None
        self._selection_gripper_m = None
        self._pending_plan_request_id = None

    def _finish_cycle(self, now: float) -> None:
        if self._active_track_id is not None:
            record = self._active_record()
            if record is not None:
                record.status = TrackStatus.SAFE_FAILURE
                record.failure_reasons.append("RETREAT_COMPLETED_WITHOUT_PLACE_CONFIRMATION")
            self._release_active_reservations("RETREAT_FINISHED_UNCONFIRMED")
            self._cycle_outcome = "SAFE_FAILURE"
        self._emit("CYCLE_CLOSED", self._cycle_outcome or "SAFE_FAILURE",
                   details={"cycle_duration_s": (
                       None if self._cycle_started_s is None else now - self._cycle_started_s
                   )})
        self._plan = None
        self._cycle_deadline_s = None
        self._cycle_started_s = None
        self._active_command_id = None
        self._recovery_attempts = 0
        self._transition(State.OBSERVE, "CYCLE_CLOSED_REOBSERVE")

    def _safe_stop(self, reason: str, details: dict | None = None) -> None:
        if self.state == State.DONE:
            return
        if self.state != State.SAFE_STOP:
            if self._holding_state is False and self._active_track_id is not None:
                track_id = self._active_track_id
                record = self._track_records.get(track_id)
                self._release_active_reservations(f"SAFE_STOP_NO_PAYLOAD_{reason}")
                if record is not None and record.status == TrackStatus.RESERVED:
                    record.status = TrackStatus.SAFE_FAILURE
                    record.failure_reasons.append(reason)
            try:
                self._transition(State.SAFE_STOP, reason, details)
            except IllegalTransitionError:
                self.state = State.SAFE_STOP
                self._state_started_s = self._event_clock_s
                self._safety_hold_requested = True
                self._emit("STATE_TRANSITION_FORCED_SAFE", reason,
                           details={"from": self.state.value, **(details or {})})
            self._active_command_id = None
            self._terminal_result = {
                "run_id": self.run_id,
                "status": "SAFE_STOP",
                "reason_code": reason,
                "simulation_time_s": self._event_clock_s,
                "holding_state": self._holding_state,
                "active_track_id": self._active_track_id,
                "reserved_slot_id": self.reservations.reserved_slot_id,
                "no_automatic_release": True,
            }
        else:
            self._emit("SAFE_STOP_REASON_RETAINED", reason, details=details)
        self._safety_hold_requested = True

    def _handle_safe_stop_reset(self, inputs: ControllerInput, now: float) -> None:
        if not (inputs.operator_reset and inputs.payload_safely_resolved):
            return
        if inputs.emergency_stop or not inputs.health.dependencies_ready:
            self._emit("RESET_REJECTED", "INTERLOCK_OR_DEPENDENCY_NOT_READY")
            return
        if not inputs.health.arm_state_valid or not inputs.health.gripper_state_known \
                or not self._valid_q(inputs.current_q) or not self._finite(inputs.current_gripper_m):
            self._emit("RESET_REJECTED", "ARM_OR_GRIPPER_STATE_UNKNOWN")
            return
        if self._holding_state is not False:
            self._emit("RESET_REJECTED", "GRIPPER_NOT_CONFIRMED_EMPTY")
            return
        self.reservations.reset_reserved("OPERATOR_RESET_QUARANTINE")
        self._active_track_id = None
        self._active_class = None
        self._active_slot_id = None
        self._plan = None
        self._latest_batch = None
        self._track_records = {}
        self._last_frame_id = -1
        self._empty_scene_frames = 0
        self._camera_failures = 0
        self._batch_started_s = now
        self._terminal_result = None
        self._safety_hold_requested = False
        self._cycle_deadline_s = None
        self._transition(State.INIT, "OPERATOR_RESET_AFTER_SAFE_DISPOSITION")

    def _build_terminal_result(self) -> dict:
        counts = {status.value: 0 for status in TrackStatus}
        for record in self._track_records.values():
            counts[record.status.value] += 1
        return {
            "run_id": self.run_id,
            "status": "DONE",
            "simulation_time_s": self._event_clock_s,
            "controller_track_status_counts": counts,
            "controller_confirmed_placements": counts[TrackStatus.PLACED_CONTROLLER_CONFIRMED.value],
            "safe_failures": counts[TrackStatus.SAFE_FAILURE.value],
            "skipped_unknown": counts[TrackStatus.SKIPPED_UNKNOWN.value],
            "skipped_unreachable": counts[TrackStatus.SKIPPED_UNREACHABLE.value],
            "skipped_full_zone": counts[TrackStatus.SKIPPED_FULL_ZONE.value],
            "evaluator_result": None,
            "note": "DONE means the batch controller terminated; it does not imply 100% sort success.",
            "slot_states": [
                {"slot_id": slot.slot_id, "status": slot.status.value}
                for slot in self.reservations.snapshots()
            ],
        }

    def _assert_invariants(self) -> None:
        self.reservations.assert_invariants()
        reserved_track = self.reservations.reserved_track_id
        if reserved_track is not None:
            record = self._track_records.get(reserved_track)
            if record is None or record.status != TrackStatus.RESERVED:
                raise ControllerContractError("Reserved M4 track must have RESERVED lifecycle status")
            if self._active_track_id != reserved_track:
                raise ControllerContractError("Reservation owner differs from active controller track")
        if self.state == State.DONE and (
            self.reservations.reserved_track_id is not None
            or self.reservations.reserved_slot_id is not None
        ):
            raise ControllerContractError("DONE may not retain an active reservation")
        if self.state == State.SAFE_STOP and self._holding_state is True:
            if self._active_track_id is None or self.reservations.reserved_slot_id is None:
                raise ControllerContractError("SAFE_STOP with a held payload must preserve cycle reservations")
        if self.state in {State.TRANSFER, State.PLACE}:
            if self._holding_state is not True or self._holding_evidence_time_s is None:
                raise ControllerContractError(
                    f"{self.state.value} requires fresh public evidence of a held payload"
                )
            evidence_age = self._event_clock_s - self._holding_evidence_time_s
            if evidence_age < -1e-12 or evidence_age > float(
                self.config.p("hold_evidence_max_age_s")
            ):
                raise ControllerContractError(
                    f"{self.state.value} has stale public hold evidence"
                )
        if self.state == State.PLACE and (
            self._active_track_id is None or self._active_slot_id is None
            or self.reservations.reserved_track_id != self._active_track_id
            or self.reservations.status(self._active_slot_id) != SlotStatus.RESERVED
        ):
            raise ControllerContractError("PLACE requires the active track's reserved slot")
        if self.state == State.SAFE_STOP and self._active_command_id is not None:
            raise ControllerContractError("SAFE_STOP may not retain an active execution command")
        if self.state == State.DONE and self._active_command_id is not None:
            raise ControllerContractError("DONE may not retain an active execution command")
        for track_id, slot_id in self._placed_slot_by_track.items():
            record = self._track_records[track_id]
            if record.status == TrackStatus.PLACED_CONTROLLER_CONFIRMED:
                if self.reservations.status(slot_id) != SlotStatus.OCCUPIED:
                    raise ControllerContractError("Controller-confirmed placement must retain OCCUPIED slot")

    def _active_track_id_or_none(self) -> int | None:
        return self._active_track_id


_TERMINAL_TRACK_STATUSES = frozenset({
    TrackStatus.PLACED_CONTROLLER_CONFIRMED,
    TrackStatus.SKIPPED_UNKNOWN,
    TrackStatus.SKIPPED_UNREACHABLE,
    TrackStatus.SKIPPED_FULL_ZONE,
    TrackStatus.SAFE_FAILURE,
    TrackStatus.FAILED,
})
