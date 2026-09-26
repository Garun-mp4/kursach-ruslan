"""Pure FK/IK and workspace utilities for the M2 SCARA R-R-P-R arm.

The module contains no MuJoCo calls. MuJoCo is used only by the independent
verification runner to compare its compiled site transforms with this math.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping
import math

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SSOT = PROJECT_ROOT / "02_Спецификация" / "параметры_системы.yaml"
TAU = 2.0 * math.pi


class IKStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NEAR_SINGULARITY = "NEAR_SINGULARITY"
    UNREACHABLE_GEOMETRY = "UNREACHABLE_GEOMETRY"
    JOINT_LIMIT_VIOLATION = "JOINT_LIMIT_VIOLATION"
    Z_LIMIT_VIOLATION = "Z_LIMIT_VIOLATION"
    ORIENTATION_UNREACHABLE = "ORIENTATION_UNREACHABLE"
    INVALID_TARGET = "INVALID_TARGET"
    CURRENT_CONFIGURATION_INVALID = "CURRENT_CONFIGURATION_INVALID"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"


@dataclass(frozen=True)
class Pose:
    """Reduced SCARA task pose in BASE: XYZ in metres, yaw in radians."""

    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float

    def as_array(self) -> np.ndarray:
        return np.asarray((self.x_m, self.y_m, self.z_m, self.yaw_rad), dtype=float)


@dataclass(frozen=True)
class ArmConfig:
    """Immutable kinematic subset read from the project SSOT."""

    ssot_version: str
    l1_m: float
    l2_m: float
    base_world_xyz_m: tuple[float, float, float]
    shoulder_z_m: float
    zero_xyz_m: tuple[float, float, float]
    wrist_center_in_zslide_m: tuple[float, float, float]
    tcp_in_wrist_m: tuple[float, float, float]
    j1_limits_rad: tuple[float, float]
    j2_limits_rad: tuple[float, float]
    j3_limits_m: tuple[float, float]
    j4_limits_rad: tuple[float, float]
    observation_q: tuple[float, float, float, float]
    symmetry_period_rad: float
    numeric_cos_tolerance: float
    joint_limit_tolerance_rad: float
    joint_limit_tolerance_m: float
    candidate_dedup_tolerance: float
    position_acceptance_m: float
    yaw_acceptance_rad: float
    near_singular_abs_sin_q2: float

    @property
    def q_limits(self) -> tuple[tuple[float, float], ...]:
        return (
            self.j1_limits_rad,
            self.j2_limits_rad,
            self.j3_limits_m,
            self.j4_limits_rad,
        )

    @property
    def characteristic_spans(self) -> np.ndarray:
        return np.asarray([hi - lo for lo, hi in self.q_limits], dtype=float)

    @property
    def tcp_z_zero_m(self) -> float:
        return (
            self.base_world_xyz_m[2]
            + self.shoulder_z_m
            + self.zero_xyz_m[2]
            + self.wrist_center_in_zslide_m[2]
            + self.tcp_in_wrist_m[2]
        )


@dataclass(frozen=True)
class FKResult:
    pose: Pose
    transform_base_tcp: np.ndarray
    frames_base: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class IKCandidate:
    q: tuple[float, float, float, float]
    branch_id: str
    symmetry_index: int
    resolved_target_yaw_rad: float
    status: IKStatus
    abs_sin_q2: float
    manipulability_m2: float
    condition_number_xy: float
    normalized_joint_cost: float
    minimum_normalized_limit_margin: float


@dataclass(frozen=True)
class IKResult:
    status: IKStatus
    reason: str
    candidates: tuple[IKCandidate, ...]
    selected: IKCandidate | None
    position_residual_m: float | None
    yaw_residual_rad: float | None
    numeric_boundary_adjustment: bool = False


def _finite_vector(values: Iterable[float], length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(tuple(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


def _parameter(parameters: Mapping[str, dict], key: str):
    try:
        return parameters[key]["value"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"SSOT is missing parameter {key}") from exc


def load_config(path: str | Path = DEFAULT_SSOT) -> ArmConfig:
    """Load and validate the kinematics inputs; no geometric values are copied here."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("parameters"), dict):
        raise ValueError("SSOT must contain a parameters mapping")
    version = str(document.get("config_version", ""))
    if version not in {"M2-v1.0", "M3-v1.0", "M4-v1.1", "M4-v1.2", "M4-v1.3", "M5-v1.0", "M5-v1.1", "M5-v1.2", "M5-v1.3", "M7-v1.0", "M7-v1.1", "M7-v1.2", "M7-v1.3", "M7-v1.4", "M7-v1.5", "M7-v1.6", "M7-v1.7", "M7-v1.8"}:
        raise ValueError(f"Unsupported SSOT config_version {version!r}")
    p = document["parameters"]

    def scalar(key: str) -> float:
        value = float(_parameter(p, key))
        if not math.isfinite(value):
            raise ValueError(f"SSOT parameter {key} must be finite")
        return value

    def triple(key: str) -> tuple[float, float, float]:
        result = _finite_vector(_parameter(p, key), 3, key)
        return tuple(float(v) for v in result)

    def limits(key: str) -> tuple[float, float]:
        result = _finite_vector(_parameter(p, key), 2, key)
        if result[0] >= result[1]:
            raise ValueError(f"SSOT parameter {key} must have min < max")
        return float(result[0]), float(result[1])

    zero = triple("robot.zero_xyz_m")
    # M2 MJCF declares these offsets as zero. Fail closed if the geometry is later
    # changed without updating the model and this transformation chain together.
    if not np.allclose(zero, (0.0, 0.0, 0.0), rtol=0.0, atol=0.0):
        raise ValueError("M2 kinematics assumes robot.zero_xyz_m == [0, 0, 0]; revise model and FK together")
    symmetry_range = _finite_vector(_parameter(p, "object.yaw_range_rad"), 2, "object.yaw_range_rad")
    symmetry_period = float(symmetry_range[1] - symmetry_range[0])
    if not (0.0 < symmetry_period <= TAU):
        raise ValueError("object.yaw_range_rad must define a positive symmetry period")
    pose = _parameter(p, "camera.safe_observation_joint_pose")
    observation = tuple(float(pose[name]) for name in ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist"))
    if not all(math.isfinite(v) for v in observation):
        raise ValueError("SSOT observation pose must be finite")

    cfg = ArmConfig(
        ssot_version=version,
        l1_m=scalar("robot.link1_length_m"),
        l2_m=scalar("robot.link2_length_m"),
        base_world_xyz_m=triple("robot.base_world_position_m"),
        shoulder_z_m=scalar("robot.shoulder_z_m"),
        zero_xyz_m=zero,
        wrist_center_in_zslide_m=triple("robot.wrist_center_in_zslide_m"),
        tcp_in_wrist_m=triple("robot.tcp_in_wrist_m"),
        j1_limits_rad=limits("robot.j1_range_rad"),
        j2_limits_rad=limits("robot.j2_range_rad"),
        j3_limits_m=limits("robot.j3_range_m"),
        j4_limits_rad=limits("robot.j4_range_rad"),
        observation_q=observation,
        symmetry_period_rad=symmetry_period,
        numeric_cos_tolerance=scalar("kinematics.numeric_cos_tolerance"),
        joint_limit_tolerance_rad=scalar("kinematics.joint_limit_tolerance_rad"),
        joint_limit_tolerance_m=scalar("kinematics.joint_limit_tolerance_m"),
        candidate_dedup_tolerance=scalar("kinematics.candidate_dedup_tolerance"),
        position_acceptance_m=scalar("kinematics.position_acceptance_m"),
        yaw_acceptance_rad=scalar("kinematics.yaw_acceptance_rad"),
        near_singular_abs_sin_q2=scalar("kinematics.near_singular_abs_sin_q2"),
    )
    if min(cfg.l1_m, cfg.l2_m) <= 0.0 or min(cfg.characteristic_spans) <= 0.0:
        raise ValueError("Link lengths and joint spans must be positive")
    if not 0.0 < cfg.numeric_cos_tolerance < 1e-6:
        raise ValueError("kinematics.numeric_cos_tolerance is outside the supported range")
    if not 0.0 < cfg.joint_limit_tolerance_rad < 1e-9 or not 0.0 < cfg.joint_limit_tolerance_m < 1e-9:
        raise ValueError("kinematics joint boundary tolerances are outside the supported range")
    if not 0.0 < cfg.candidate_dedup_tolerance < 1e-6:
        raise ValueError("kinematics.candidate_dedup_tolerance is outside the supported range")
    if not 0.0 < cfg.position_acceptance_m < 1e-3:
        raise ValueError("kinematics.position_acceptance_m must be positive and below 1 mm")
    if not 0.0 < cfg.yaw_acceptance_rad < 1e-4:
        raise ValueError("kinematics.yaw_acceptance_rad must be positive and below 1e-4 rad")
    if not 0.0 < cfg.near_singular_abs_sin_q2 < 1.0:
        raise ValueError("kinematics.near_singular_abs_sin_q2 must be in (0, 1)")
    if not check_joint_limits(cfg.observation_q, cfg):
        raise ValueError("SSOT camera observation pose violates the arm limits")
    return cfg


def _translation(x: float, y: float, z: float) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, 3] = (x, y, z)
    return transform


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    return transform


def wrap_angle(angle_rad: float) -> float:
    """Canonical representative in [-pi, pi], preserving +pi when supplied positive."""
    wrapped = (float(angle_rad) + math.pi) % TAU - math.pi
    if math.isclose(wrapped, -math.pi, rel_tol=0.0, abs_tol=2e-15) and angle_rad > 0.0:
        return math.pi
    return wrapped


def periodic_difference(a_rad: float, b_rad: float, period_rad: float = TAU) -> float:
    return (float(a_rad) - float(b_rad) + period_rad / 2.0) % period_rad - period_rad / 2.0


def check_joint_limits(q: Iterable[float], config: ArmConfig) -> bool:
    """Check physical coordinates as given; never wrap an out-of-range joint."""
    try:
        coordinates = _finite_vector(q, 4, "q=[q1,q2,q3,q4]")
    except ValueError:
        return False
    return all(lo <= float(v) <= hi
               for v, (lo, hi) in zip(coordinates, config.q_limits))


def forward_kinematics(q: Iterable[float], config: ArmConfig) -> FKResult:
    """Manual homogeneous-transform FK. q = [q1 rad, q2 rad, q3 m, q4 rad]."""
    q1, q2, q3, q4 = _finite_vector(q, 4, "q=[q1,q2,q3,q4]")
    if not check_joint_limits((q1, q2, q3, q4), config):
        raise ValueError("Joint configuration violates physical joint limits")

    bx, by, bz = config.base_world_xyz_m
    zx, zy, zz = config.zero_xyz_m
    t_base_j1 = _translation(bx + zx, by + zy, bz + config.shoulder_z_m + zz) @ _rotation_z(q1)
    t_j1_j2 = _translation(config.l1_m, 0.0, 0.0) @ _rotation_z(q2)
    t_j2_j3 = _translation(config.l2_m, 0.0, 0.0) @ _translation(0.0, 0.0, -q3)
    t_j3_j4 = _translation(*config.wrist_center_in_zslide_m) @ _rotation_z(q4)
    t_j4_tcp = _translation(*config.tcp_in_wrist_m)

    t_base_j2 = t_base_j1 @ t_j1_j2
    t_base_j3 = t_base_j2 @ t_j2_j3
    t_base_j4 = t_base_j3 @ t_j3_j4
    t_base_tcp = t_base_j4 @ t_j4_tcp
    yaw = wrap_angle(math.atan2(t_base_tcp[1, 0], t_base_tcp[0, 0]))
    pose = Pose(*map(float, t_base_tcp[:3, 3]), yaw)
    frames = {
        "BASE": np.eye(4, dtype=float),
        "J1": t_base_j1,
        "J2": t_base_j2,
        "J3": t_base_j3,
        "J4": t_base_j4,
        "TCP": t_base_tcp,
    }
    return FKResult(pose, t_base_tcp, frames)


def jacobian(q: Iterable[float], config: ArmConfig) -> np.ndarray:
    """Analytic task Jacobian for [x,y,z,yaw] versus [q1,q2,q3,q4]."""
    q1, q2, q3, q4 = _finite_vector(q, 4, "q=[q1,q2,q3,q4]")
    if not check_joint_limits((q1, q2, q3, q4), config):
        raise ValueError("Joint configuration violates physical joint limits")
    a = q1
    b = q1 + q2
    l1, l2 = config.l1_m, config.l2_m
    return np.asarray(
        [
            [-l1 * math.sin(a) - l2 * math.sin(b), -l2 * math.sin(b), 0.0, 0.0],
            [l1 * math.cos(a) + l2 * math.cos(b), l2 * math.cos(b), 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def classify_singularity(q: Iterable[float], config: ArmConfig) -> dict[str, float | bool | str]:
    q1, q2, q3, q4 = _finite_vector(q, 4, "q=[q1,q2,q3,q4]")
    if not check_joint_limits((q1, q2, q3, q4), config):
        raise ValueError("Joint configuration violates physical joint limits")
    abs_sin = abs(math.sin(q2))
    planar_j = jacobian((q1, q2, q3, q4), config)[:2, :2]
    singular_values = np.linalg.svd(planar_j, compute_uv=False)
    smallest = float(singular_values[-1])
    condition = math.inf if smallest <= np.finfo(float).eps else float(singular_values[0] / smallest)
    return {
        "classification": "NEAR_SINGULARITY" if abs_sin <= config.near_singular_abs_sin_q2 else "REGULAR",
        "abs_sin_q2": abs_sin,
        "manipulability_m2": config.l1_m * config.l2_m * abs_sin,
        "sigma_min_xy_m_per_rad": smallest,
        "condition_number_xy": condition,
        "near_singular": abs_sin <= config.near_singular_abs_sin_q2,
    }


def workspace_bounds(config: ArmConfig) -> dict[str, float]:
    """Exact radial bounds for unconstrained-link geometry and limited q2 workspace."""
    l1, l2 = config.l1_m, config.l2_m
    geom_inner, geom_outer = abs(l1 - l2), l1 + l2
    q2_lo, q2_hi = config.j2_limits_rad
    furthest_fold = max(abs(q2_lo), abs(q2_hi))
    cos_fold = math.cos(furthest_fold)
    limited_inner = math.sqrt(max(0.0, l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * cos_fold))
    max_abs_q2 = max(abs(q2_lo), abs(q2_hi))
    if q2_lo <= 0.0 <= q2_hi:
        limited_outer = geom_outer
    else:
        limited_outer = max(
            math.sqrt(l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * math.cos(q2_lo)),
            math.sqrt(l1 * l1 + l2 * l2 + 2.0 * l1 * l2 * math.cos(q2_hi)),
        )
    del max_abs_q2  # retained only as an explicit reminder of the limit calculation above
    q3_lo, q3_hi = config.j3_limits_m
    z_low = config.tcp_z_zero_m - q3_hi
    z_high = config.tcp_z_zero_m - q3_lo
    return {
        "geometric_r_min_m": geom_inner,
        "geometric_r_max_m": geom_outer,
        "joint_limited_r_min_m": limited_inner,
        "joint_limited_r_max_m": limited_outer,
        "tcp_z_min_m": z_low,
        "tcp_z_max_m": z_high,
    }


def _angle_representatives(angle_rad: float, limits: tuple[float, float], tolerance: float) -> list[float]:
    """Enumerate actual 2pi-equivalent coordinates that lie within physical limits."""
    lo, hi = limits
    k_min = math.ceil((lo - angle_rad - tolerance) / TAU)
    k_max = math.floor((hi - angle_rad + tolerance) / TAU)
    result: list[float] = []
    for k in range(k_min, k_max + 1):
        value = angle_rad + k * TAU
        if lo - tolerance <= value <= hi + tolerance:
            if value < lo:
                value = lo
            elif value > hi:
                value = hi
            result.append(value)
    return result


def _target_pose(target: Pose | Iterable[float]) -> np.ndarray:
    values = target.as_array() if isinstance(target, Pose) else _finite_vector(target, 4, "target=[x,y,z,yaw]")
    return _finite_vector(values, 4, "target=[x,y,z,yaw]")


def _min_normalized_limit_margin(q: tuple[float, float, float, float], config: ArmConfig) -> float:
    margins = []
    for value, (lo, hi) in zip(q, config.q_limits):
        span = hi - lo
        margins.extend(((value - lo) / span, (hi - value) / span))
    return float(min(margins))


def _deduplicate(candidates: list[IKCandidate], config: ArmConfig) -> list[IKCandidate]:
    unique: list[IKCandidate] = []
    for candidate in candidates:
        q = np.asarray(candidate.q) / config.characteristic_spans
        if any(np.max(np.abs(q - np.asarray(existing.q) / config.characteristic_spans)) <= config.candidate_dedup_tolerance
               for existing in unique):
            continue
        unique.append(candidate)
    return unique


def inverse_kinematics(
    target: Pose | Iterable[float],
    config: ArmConfig,
    current_q: Iterable[float] | None = None,
    *,
    allow_object_symmetry: bool = True,
) -> IKResult:
    """Analytic two-link SCARA IK with explicit candidate and failure diagnostics."""
    try:
        pose = _target_pose(target)
    except (ValueError, TypeError):
        return IKResult(IKStatus.INVALID_TARGET, "Target must be four finite values [x_m,y_m,z_m,yaw_rad].", (), None, None, None)

    try:
        reference = np.asarray(config.observation_q if current_q is None else tuple(current_q), dtype=float)
    except (TypeError, ValueError):
        reference = np.asarray([], dtype=float)
    if reference.shape != (4,) or not np.all(np.isfinite(reference)) or not check_joint_limits(reference, config):
        return IKResult(IKStatus.CURRENT_CONFIGURATION_INVALID, "Current q is non-finite, malformed, or outside physical limits.", (), None, None, None)

    x, y, z, target_yaw = map(float, pose)
    q3 = config.tcp_z_zero_m - z
    q3_lo, q3_hi = config.j3_limits_m
    z_numeric_tol = config.joint_limit_tolerance_m
    if q3 < q3_lo - z_numeric_tol or q3 > q3_hi + z_numeric_tol:
        return IKResult(IKStatus.Z_LIMIT_VIOLATION, f"Required q3={q3:.12g} m is outside [{q3_lo:.12g}, {q3_hi:.12g}] m.", (), None, None, None)
    boundary_adjustment = False
    if q3 < q3_lo:
        q3, boundary_adjustment = q3_lo, True
    elif q3 > q3_hi:
        q3, boundary_adjustment = q3_hi, True

    radius_sq = x * x + y * y
    c2 = (radius_sq - config.l1_m**2 - config.l2_m**2) / (2.0 * config.l1_m * config.l2_m)
    epsilon = config.numeric_cos_tolerance
    if c2 < -1.0 - epsilon or c2 > 1.0 + epsilon:
        return IKResult(IKStatus.UNREACHABLE_GEOMETRY, f"Planar target requires cos(q2)={c2:.12g}, outside [-1,1].", (), None, None, None)
    if c2 < -1.0 or c2 > 1.0:
        c2 = min(1.0, max(-1.0, c2))
        boundary_adjustment = True

    alpha = math.acos(c2)
    q2_limits = config.j2_limits_rad
    q2_options = [(alpha, "ELBOW_POSITIVE"), (-alpha, "ELBOW_NEGATIVE")]
    q2_valid: list[tuple[float, str]] = []
    for q2_value, branch in q2_options:
        if q2_limits[0] - config.joint_limit_tolerance_rad <= q2_value <= q2_limits[1] + config.joint_limit_tolerance_rad:
            q2_value = min(q2_limits[1], max(q2_limits[0], q2_value))
            q2_valid.append((q2_value, branch))
    if not q2_valid:
        return IKResult(IKStatus.JOINT_LIMIT_VIOLATION, "Geometric IK exists, but every elbow branch violates the J2 physical range.", (), None, None, None, boundary_adjustment)

    yaw_values: list[tuple[float, int]] = []
    if allow_object_symmetry:
        period = config.symmetry_period_rad
        canonical = target_yaw % period
        order = int(round(TAU / period))
        if not math.isclose(order * period, TAU, rel_tol=0.0, abs_tol=1e-9):
            return IKResult(IKStatus.NUMERICAL_FAILURE, "Object symmetry period must divide 2*pi.", (), None, None, None, boundary_adjustment)
        yaw_values = [(canonical + k * period, k) for k in range(order)]
    else:
        yaw_values = [(wrap_angle(target_yaw), 0)]

    candidates: list[IKCandidate] = []
    q1_lohi = config.j1_limits_rad
    q4_lohi = config.j4_limits_rad
    for q2_value, branch in q2_valid:
        q1_raw = math.atan2(y, x) - math.atan2(
            config.l2_m * math.sin(q2_value),
            config.l1_m + config.l2_m * math.cos(q2_value),
        )
        q1_values = _angle_representatives(q1_raw, q1_lohi, config.joint_limit_tolerance_rad)
        if not q1_values:
            continue
        for q1_value in q1_values:
            for resolved_yaw, symmetry_index in yaw_values:
                q4_raw = resolved_yaw - q1_value - q2_value
                q4_values = _angle_representatives(q4_raw, q4_lohi, config.joint_limit_tolerance_rad)
                for q4_value in q4_values:
                    q = (float(q1_value), float(q2_value), float(q3), float(q4_value))
                    if not check_joint_limits(q, config):
                        continue
                    singularity = classify_singularity(q, config)
                    delta_scaled = (np.asarray(q) - reference) / config.characteristic_spans
                    cost = float(np.dot(delta_scaled, delta_scaled))
                    candidates.append(
                        IKCandidate(
                            q=q,
                            branch_id=branch,
                            symmetry_index=symmetry_index,
                            resolved_target_yaw_rad=float(resolved_yaw),
                            status=IKStatus.NEAR_SINGULARITY if singularity["near_singular"] else IKStatus.SUCCESS,
                            abs_sin_q2=float(singularity["abs_sin_q2"]),
                            manipulability_m2=float(singularity["manipulability_m2"]),
                            condition_number_xy=float(singularity["condition_number_xy"]),
                            normalized_joint_cost=cost,
                            minimum_normalized_limit_margin=_min_normalized_limit_margin(q, config),
                        )
                    )

    candidates = _deduplicate(candidates, config)
    if not candidates:
        if q2_valid:
            return IKResult(IKStatus.ORIENTATION_UNREACHABLE, "Elbow and position solutions satisfy limits, but no wrist coordinate satisfies its physical range.", (), None, None, None, boundary_adjustment)
        return IKResult(IKStatus.JOINT_LIMIT_VIOLATION, "No complete joint solution satisfies physical limits.", (), None, None, None, boundary_adjustment)

    branch_order = {"ELBOW_POSITIVE": 0, "ELBOW_NEGATIVE": 1}
    candidates.sort(
        key=lambda candidate: (
            candidate.normalized_joint_cost,
            candidate.status == IKStatus.NEAR_SINGULARITY,
            -candidate.minimum_normalized_limit_margin,
            branch_order.get(candidate.branch_id, 9),
            candidate.symmetry_index,
            candidate.q,
        )
    )
    selected = candidates[0]
    selected_fk = forward_kinematics(selected.q, config)
    position_residual = float(np.linalg.norm(selected_fk.transform_base_tcp[:3, 3] - pose[:3]))
    yaw_period = config.symmetry_period_rad if allow_object_symmetry else TAU
    yaw_residual = abs(periodic_difference(selected_fk.pose.yaw_rad, target_yaw, yaw_period))
    if (position_residual > config.position_acceptance_m
            or yaw_residual > config.yaw_acceptance_rad):
        return IKResult(
            IKStatus.NUMERICAL_FAILURE,
            "Analytic candidate failed the configured FK residual acceptance limits; it is returned for diagnosis, not accepted as a solution.",
            tuple(candidates),
            selected,
            position_residual,
            yaw_residual,
            boundary_adjustment,
        )
    return IKResult(
        status=selected.status,
        reason="Pose solved analytically; selected candidate is ranked by range-normalized direct joint displacement from the supplied (or SSOT observation) configuration.",
        candidates=tuple(candidates),
        selected=selected,
        position_residual_m=position_residual,
        yaw_residual_rad=yaw_residual,
        numeric_boundary_adjustment=boundary_adjustment,
    )
