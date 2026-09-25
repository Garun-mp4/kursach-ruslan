from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


class Phase(str, Enum):
    INITIAL_APPROACH = "INITIAL_APPROACH"
    ORIENTATION_ALIGN = "ORIENTATION_ALIGN"
    PREGRASP_ROUTE = "PREGRASP_ROUTE"
    PREGRASP = "PREGRASP"
    DESCENT = "DESCENT"
    GRASP_CLOSE = "GRASP_CLOSE"
    LIFT = "LIFT"
    TRANSFER = "TRANSFER"
    PREPLACE = "PREPLACE"
    PLACE_DESCENT = "PLACE_DESCENT"
    RELEASE = "RELEASE"
    RETREAT = "RETREAT"
    SAFE_RETURN = "SAFE_RETURN"
    SAFE_HOME = "SAFE_HOME"


class PlanCode(str, Enum):
    SUCCESS = "SUCCESS"
    INVALID_INPUT = "INVALID_INPUT"
    INVALID_BATCH = "INVALID_BATCH"
    STALE_DETECTION = "STALE_DETECTION"
    UNSUPPORTED_CLASS = "UNSUPPORTED_CLASS"
    SCENE_UNCERTAIN = "SCENE_UNCERTAIN"
    NO_AVAILABLE_SLOT = "NO_AVAILABLE_SLOT"
    SLOT_INVALID = "SLOT_INVALID"
    SLOT_OCCUPIED = "SLOT_OCCUPIED"
    TARGET_OUT_OF_SCOPE = "TARGET_OUT_OF_SCOPE"
    IK_UNREACHABLE = "IK_UNREACHABLE"
    NEAR_SINGULARITY = "NEAR_SINGULARITY"
    BRANCH_DISCONTINUITY = "BRANCH_DISCONTINUITY"
    JOINT_LIMIT = "JOINT_LIMIT"
    COLLISION_AT_START = "COLLISION_AT_START"
    COLLISION_PATH = "COLLISION_PATH"
    COLLISION_GRASP = "COLLISION_GRASP"
    COLLISION_PAYLOAD = "COLLISION_PAYLOAD"
    NO_DETERMINISTIC_ROUTE = "NO_DETERMINISTIC_ROUTE"
    VERTICAL_PATH_ERROR = "VERTICAL_PATH_ERROR"
    VELOCITY_LIMIT = "VELOCITY_LIMIT"
    ACCELERATION_LIMIT = "ACCELERATION_LIMIT"
    TIMEOUT = "TIMEOUT"
    MODEL_MISMATCH = "MODEL_MISMATCH"
    SCENE_CAPACITY_EXCEEDED = "SCENE_CAPACITY_EXCEEDED"


@dataclass(frozen=True)
class CartesianWaypoint:
    phase: str
    pose_xyzyaw: tuple[float, float, float, float]
    q: tuple[float, float, float, float]
    branch_id: str
    symmetry_index: int
    gripper_m: float
    payload_mode: str
    event: str | None = None


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    phase: str
    q: tuple[float, float, float, float]
    dq: tuple[float, float, float, float]
    ddq: tuple[float, float, float, float]
    tcp_xyzyaw: tuple[float, float, float, float]
    branch_id: str
    gripper_m: float
    payload_mode: str
    gripper_velocity_m_s: float = 0.0
    gripper_acceleration_m_s2: float = 0.0
    event: str | None = None


@dataclass(frozen=True)
class PlanEvent:
    time_s: float
    event: str
    phase: str
    track_id: int
    slot_id: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ClearanceSample:
    time_s: float
    phase: str
    payload_mode: str
    narrowphase_minimum_m: float
    conservative_lower_bound_m: float
    checked_pair_count: int
    narrowphase_pair_count: int


@dataclass(frozen=True)
class Plan:
    plan_id: str
    ssot_version: str
    ssot_sha256: str
    model_sha256: str
    perception_frame_id: int
    perception_time_s: float
    target_track_id: int
    target_class: str
    slot_id: str
    selected_ik_branch: str
    resolved_tool_yaw_rad: float
    phases: tuple[str, ...]
    waypoints: tuple[CartesianWaypoint, ...]
    samples: tuple[TrajectorySample, ...]
    events: tuple[PlanEvent, ...]
    duration_s: float
    path_length_m: float
    minimum_clearance_m: float
    clearance_lower_bound_m: float
    minimum_clearance_phase: str
    vertical_xy_error_max_m: float
    vertical_yaw_error_max_rad: float
    maximum_abs_velocity_ratio: float
    maximum_abs_acceleration_ratio: float
    planner_strategy: str
    collision_sample_step_m: float = 0.0
    clearance_profile: tuple[ClearanceSample, ...] = ()


@dataclass(frozen=True)
class PlanResult:
    code: PlanCode
    reason: str
    phase: str | None
    plan: Plan | None
    diagnostics: dict[str, Any]

    @property
    def success(self) -> bool:
        return self.code == PlanCode.SUCCESS and self.plan is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
