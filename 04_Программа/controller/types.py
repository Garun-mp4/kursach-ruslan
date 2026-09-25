from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from perception.types import DetectionBatch
from planning.records import Plan, PlanResult


class State(str, Enum):
    INIT = "INIT"
    OBSERVE = "OBSERVE"
    SELECT = "SELECT"
    PLAN = "PLAN"
    APPROACH = "APPROACH"
    DESCEND = "DESCEND"
    GRASP = "GRASP"
    VERIFY_HOLD = "VERIFY_HOLD"
    TRANSFER = "TRANSFER"
    PLACE = "PLACE"
    RELEASE = "RELEASE"
    VERIFY_PLACE = "VERIFY_PLACE"
    RETREAT = "RETREAT"
    RECOVER = "RECOVER"
    SAFE_STOP = "SAFE_STOP"
    DONE = "DONE"


class TrackStatus(str, Enum):
    PENDING = "PENDING"
    RESERVED = "RESERVED"
    PLACED_CONTROLLER_CONFIRMED = "PLACED_CONTROLLER_CONFIRMED"
    SKIPPED_UNKNOWN = "SKIPPED_UNKNOWN"
    SKIPPED_UNREACHABLE = "SKIPPED_UNREACHABLE"
    SKIPPED_FULL_ZONE = "SKIPPED_FULL_ZONE"
    SAFE_FAILURE = "SAFE_FAILURE"
    FAILED = "FAILED"


class SlotStatus(str, Enum):
    FREE = "FREE"
    RESERVED = "RESERVED"
    OCCUPIED = "OCCUPIED"
    QUARANTINED = "QUARANTINED"


class ExecutionStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class EvidenceStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"


class RecoveryStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class SystemHealth:
    dependencies_ready: bool
    arm_state_valid: bool
    gripper_state_known: bool


@dataclass(frozen=True)
class ExecutionFeedback:
    command_id: str
    status: ExecutionStatus
    simulation_time_s: float
    reason_code: str | None = None
    gripper_position_m: float | None = None


@dataclass(frozen=True)
class GraspEvidence:
    """Public sensor evidence. It deliberately carries no simulator object state."""

    holding: bool | None
    simulation_time_s: float
    source: str
    valid: bool = True


@dataclass(frozen=True)
class PlacementEvidence:
    """Perception-derived evidence for a released item and a public bin/slot."""

    status: EvidenceStatus
    simulation_time_s: float
    source: str
    frame_id: int = -1
    observed_zone_id: str | None = None
    observed_slot_id: str | None = None
    observed_class: str | None = None
    confidence: float | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class PlanRequest:
    request_id: int
    target_track_id: int
    target_class: str
    reserved_slot_id: str
    detections: DetectionBatch
    current_q: tuple[float, float, float, float]
    current_gripper_m: float
    available_slot_ids: tuple[str, ...]
    simulation_time_s: float


@dataclass(frozen=True)
class PlanResponse:
    request_id: int
    result: PlanResult


@dataclass(frozen=True)
class ObservationRequest:
    request_id: int
    minimum_frame_id: int
    simulation_time_s: float


@dataclass(frozen=True)
class ExecutionCommand:
    command_id: str
    state: State
    phase_names: tuple[str, ...]
    plan: Plan
    simulation_time_s: float


@dataclass(frozen=True)
class RecoveryRequest:
    request_id: int
    reason_code: str
    action: str
    plan: Plan | None
    simulation_time_s: float


@dataclass(frozen=True)
class RecoveryFeedback:
    request_id: int
    status: RecoveryStatus
    simulation_time_s: float
    arm_state_valid: bool
    gripper_state_known: bool
    holding: bool | None
    reason_code: str | None = None


@dataclass(frozen=True)
class ControllerInput:
    simulation_time_s: float
    health: SystemHealth
    current_q: tuple[float, float, float, float] | None
    current_gripper_m: float | None
    current_tcp_xy_m: tuple[float, float] | None
    detection_batch: DetectionBatch | None = None
    plan_response: PlanResponse | None = None
    execution_feedback: ExecutionFeedback | None = None
    grasp_evidence: GraspEvidence | None = None
    placement_evidence: PlacementEvidence | None = None
    recovery_feedback: RecoveryFeedback | None = None
    emergency_stop: bool = False
    operator_reset: bool = False
    payload_safely_resolved: bool = False


@dataclass(frozen=True)
class ControllerEvent:
    event_id: int
    event_type: str
    simulation_time_s: float
    state: State
    reason_code: str
    track_id: int | None = None
    slot_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ControllerOutput:
    state: State
    events: tuple[ControllerEvent, ...]
    observation_requests: tuple[ObservationRequest, ...] = ()
    plan_requests: tuple[PlanRequest, ...] = ()
    execution_commands: tuple[ExecutionCommand, ...] = ()
    recovery_requests: tuple[RecoveryRequest, ...] = ()
    safety_hold_requested: bool = False
    terminal_result: dict[str, Any] | None = None


@dataclass
class TrackRecord:
    track_id: int
    class_label: str
    status: TrackStatus = TrackStatus.PENDING
    attempts: int = 0
    grasp_attempts: int = 0
    plan_retries: int = 0
    unknown_observations: int = 0
    last_unknown_frame_id: int | None = None
    failure_reasons: list[str] = field(default_factory=list)
    last_seen_frame_id: int = -1


@dataclass(frozen=True)
class SlotSnapshot:
    slot_id: str
    class_label: str
    status: SlotStatus
    owner_track_id: int | None = None
    reason_code: str | None = None
