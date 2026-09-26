from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import math
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "04_Программа"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from kinematics import scara

SSOT_PATH = ROOT / "02_Спецификация" / "параметры_системы.yaml"
MODEL_PATH = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"


@dataclass(frozen=True)
class PlanningContext:
    ssot: dict[str, Any]
    parameters: dict[str, Any]
    arm: scara.ArmConfig
    config_version: str
    ssot_sha256: str
    model_sha256: str

    def p(self, key: str) -> Any:
        try:
            return self.parameters[key]["value"]
        except KeyError as exc:
            raise ValueError(f"SSOT is missing planning parameter {key}") from exc

    @property
    def motion_velocity_limits(self) -> tuple[float, float, float, float]:
        return tuple(float(self.p(k)) for k in (
            "robot.j1_velocity_rad_s", "robot.j2_velocity_rad_s",
            "robot.j3_velocity_m_s", "robot.j4_velocity_rad_s"))

    @property
    def motion_acceleration_limits(self) -> tuple[float, float, float, float]:
        return tuple(float(self.p(k)) for k in (
            "robot.j1_acceleration_rad_s2", "robot.j2_acceleration_rad_s2",
            "robot.j3_acceleration_m_s2", "robot.j4_acceleration_rad_s2"))


def load_context(ssot_path: str | Path = SSOT_PATH, model_path: str | Path = MODEL_PATH) -> PlanningContext:
    ssot_path, model_path = Path(ssot_path), Path(model_path)
    ssot = yaml.safe_load(ssot_path.read_text(encoding="utf-8-sig"))
    if not isinstance(ssot, dict) or not isinstance(ssot.get("parameters"), dict):
        raise ValueError("SSOT is malformed")
    version = str(ssot.get("config_version", ""))
    if version not in {"M5-v1.0", "M5-v1.1", "M5-v1.2", "M5-v1.3", "M7-v1.0", "M7-v1.1", "M7-v1.2", "M7-v1.3", "M7-v1.4", "M7-v1.5", "M7-v1.6", "M7-v1.7", "M7-v1.8"}:
        raise ValueError(f"Planner requires an M5-frozen SSOT, got {version!r}")
    for key, meta in ssot["parameters"].items():
        if not isinstance(meta, dict) or not all(k in meta for k in ("value", "unit", "origin", "basis", "owner")):
            raise ValueError(f"SSOT parameter metadata is incomplete: {key}")
    required = (
        "planning.static_clearance_m", "planning.perceived_object_clearance_m",
        "planning.path_sweep_sample_step_m", "planning.distance_numeric_tolerance_m",
        "planning.safe_transfer_tcp_z_m", "planning.max_normalized_joint_step",
        "planning.branch_singularity_weight", "planning.plan_timeout_s",
        "planning.motion_profile", "planning.contact_speed_scale",
        "planning.parameterization_safety_scale",
        "planning.transfer_ring_radial_clearance_m", "planning.maximum_waypoint_refinements",
        "planning.minimum_clearance_report_cap_m", "planning.allowed_gripper_object_contact_phases",
        "planning.self_collision_clearance_m", "planning.maximum_detection_age_s",
        "planning.position_sigma_limit_m", "planning.yaw_sigma_limit_rad",
        "planning.grasp_entry_clearance_m", "planning.grasp_finger_support_clearance_m",
        "planning.minimum_abs_sin_q2",
        "planning.transfer_arc_max_angle_rad", "planning.maximum_cartesian_waypoint_step_m",
        "planning.maximum_waypoint_refinements",
        "robot.object_pick_center_z_m", "robot.object_place_center_z_m", "robot.object_pick_center_z_m",
        "cell.tray_centers_xy_m", "cell.tray_slot_x_offsets_m", "cell.tray_slot_y_offset_m",
        "robot.finger_open_center_offset_m", "robot.finger_thickness_m", "robot.finger_range_m",
        "robot.finger_home_m", "robot.finger_velocity_m_s", "robot.finger_acceleration_m_s2",
        "robot.link1_length_m", "robot.link2_length_m", "object.size_xyz_m",
        "kinematics.yaw_acceptance_rad",
        "robot.j1_velocity_rad_s", "robot.j2_velocity_rad_s", "robot.j3_velocity_m_s",
        "robot.j4_velocity_rad_s", "robot.j1_acceleration_rad_s2", "robot.j2_acceleration_rad_s2",
        "robot.j3_acceleration_m_s2", "robot.j4_acceleration_rad_s2", "collision.allowed_pairs",
    )
    for key in required:
        if key not in ssot["parameters"]:
            raise ValueError(f"SSOT is missing required M5 parameter {key}")
    grasp_table_clearance = float(ssot["parameters"]["planning.grasp_finger_support_clearance_m"]["value"])
    static_clearance = float(ssot["parameters"]["planning.static_clearance_m"]["value"])
    numeric_tolerance = float(ssot["parameters"]["planning.distance_numeric_tolerance_m"]["value"])
    if not all(math.isfinite(value) for value in (grasp_table_clearance, static_clearance, numeric_tolerance)):
        raise ValueError("Collision clearances must be finite")
    if grasp_table_clearance < numeric_tolerance or grasp_table_clearance >= static_clearance:
        raise ValueError("Grasp finger-table clearance must be at least the numeric tolerance and below the static clearance")
    p = ssot["parameters"]
    values = {key: item["value"] for key, item in p.items()}
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    actual_model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    # The model is generated from the same SSOT. Its own compiled hash is checked at startup
    # by the M2 validator and the M5 acceptance runner, not by a duplicated mutable hash here.
    arm = scara.load_config(ssot_path)
    return PlanningContext(
        ssot=ssot, parameters=p, arm=arm, config_version=version,
        ssot_sha256=hashlib.sha256(ssot_path.read_bytes()).hexdigest(),
        model_sha256=actual_model_hash,
    )
