from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib
import math

import yaml


ROOT = Path(__file__).resolve().parents[2]
SSOT_PATH = ROOT / "02_Спецификация" / "параметры_системы.yaml"
MODEL_PATH = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"


@dataclass(frozen=True)
class ControllerConfig:
    policy_version: str
    config_version: str
    ssot_sha256: str
    model_sha256: str
    parameters: dict[str, Any]
    slot_ids: tuple[str, ...]
    slot_classes: dict[str, str]
    class_to_zone: dict[str, str]
    sensor_max_age_s: float
    tracking_max_distance_m: float
    planner_max_detection_age_s: float
    position_sigma_limit_m: float
    yaw_sigma_limit_rad: float
    gripper_home_m: float
    score_weights: tuple[float, float, float, float, float]

    def p(self, key: str) -> Any:
        try:
            return self.parameters[f"controller.{key}"]
        except KeyError as exc:
            raise ValueError(f"SSOT is missing controller.{key}") from exc

    @classmethod
    def load(cls, ssot_path: str | Path = SSOT_PATH,
             model_path: str | Path = MODEL_PATH) -> "ControllerConfig":
        ssot_path, model_path = Path(ssot_path), Path(model_path)
        raw = ssot_path.read_bytes()
        ssot = yaml.safe_load(raw.decode("utf-8-sig"))
        if not isinstance(ssot, dict) or not isinstance(ssot.get("parameters"), dict):
            raise ValueError("SSOT is malformed")
        policy_version = str(ssot.get("project", {}).get("controller_policy_version", ""))
        if policy_version != "M6-v1.0":
            raise ValueError(f"Controller requires policy M6-v1.0, got {policy_version!r}")
        params = ssot["parameters"]
        values: dict[str, Any] = {}
        for key, record in params.items():
            if not isinstance(record, dict) or not all(
                field in record for field in ("value", "unit", "origin", "basis", "owner")
            ):
                raise ValueError(f"SSOT parameter metadata is incomplete: {key}")
            if key.startswith("controller."):
                values[key] = record["value"]
        required = (
            "observation_timeout_s", "select_timeout_s", "plan_response_timeout_s",
            "init_timeout_s", "recovery_timeout_s", "place_verification_timeout_s",
            "hold_verification_timeout_s", "gripper_release_confirmation_timeout_s",
            "empty_scene_confirm_frames",
            "unknown_confirm_frames", "max_camera_failures", "max_plan_retries",
            "max_grasp_attempts", "max_cycle_attempts_per_track",
            "execution_timeout_factor", "execution_timeout_margin_s",
            "execution_timeout_min_s", "cycle_timeout_factor",
            "cycle_timeout_margin_s", "batch_timeout_s",
            "hold_evidence_max_age_s", "place_evidence_min_confidence",
            "gripper_position_tolerance_m",
            "selection_weight_confidence", "selection_weight_xy_uncertainty",
            "selection_weight_yaw_uncertainty", "selection_weight_motion",
            "selection_weight_failures", "max_recovery_attempts",
            "max_scene_uncertainty_frames",
            "initial_occupied_slots",
        )
        for key in required:
            if f"controller.{key}" not in values:
                raise ValueError(f"SSOT is missing controller.{key}")
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        runtime = ssot.get("perception", {}).get("runtime_config", {})
        sensor_max_age = float(runtime["sensor_interface"]["maximum_age_s"])
        tracking_max_distance = float(runtime["tracking"]["max_distance_m"])
        planning = params
        planner_max_age = float(planning["planning.maximum_detection_age_s"]["value"])
        sigma_xy = float(planning["planning.position_sigma_limit_m"]["value"])
        sigma_yaw = float(planning["planning.yaw_sigma_limit_rad"]["value"])
        home = float(planning["robot.finger_home_m"]["value"])
        offsets = planning["cell.tray_slot_x_offsets_m"]["value"]
        trays = planning["cell.tray_centers_xy_m"]["value"]
        slot_ids: list[str] = []
        slot_classes: dict[str, str] = {}
        for color in ("RED", "GREEN", "BLUE"):
            if color not in trays:
                raise ValueError(f"SSOT has no tray center for {color}")
            for index in range(len(offsets)):
                slot_id = f"{color}:{index}"
                slot_ids.append(slot_id)
                slot_classes[slot_id] = color
        score_weights = tuple(float(values[f"controller.selection_weight_{key}"]) for key in (
            "confidence", "xy_uncertainty", "yaw_uncertainty", "motion", "failures"
        ))
        if any(not math.isfinite(weight) or weight < 0 for weight in score_weights):
            raise ValueError("Controller selection weights must be finite and non-negative")
        if not math.isclose(sum(score_weights), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("Controller selection weights must sum to 1")
        positive = (
            sensor_max_age, tracking_max_distance, planner_max_age, sigma_xy, sigma_yaw,
            *[float(values[f"controller.{key}"]) for key in (
                "observation_timeout_s", "select_timeout_s", "plan_response_timeout_s",
                "init_timeout_s", "recovery_timeout_s", "place_verification_timeout_s",
                "hold_verification_timeout_s", "gripper_release_confirmation_timeout_s",
                "execution_timeout_factor", "execution_timeout_margin_s",
                "execution_timeout_min_s", "cycle_timeout_factor",
                "cycle_timeout_margin_s", "batch_timeout_s", "hold_evidence_max_age_s",
                "gripper_position_tolerance_m"
            )],
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Controller timeouts and inherited limits must be finite and positive")
        for key in required:
            value = values[f"controller.{key}"]
            if key.startswith("max_") or key.endswith("_frames"):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"Controller count {key} must be a positive integer")
        if not isinstance(values["controller.initial_occupied_slots"], list):
            raise ValueError("controller.initial_occupied_slots must be a list of public slot IDs")
        confidence = float(values["controller.place_evidence_min_confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Placement evidence confidence must be in [0, 1]")
        return cls(
            policy_version=policy_version,
            config_version=str(ssot.get("config_version", "")),
            ssot_sha256=hashlib.sha256(raw).hexdigest(),
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            parameters=values,
            slot_ids=tuple(slot_ids),
            slot_classes=slot_classes,
            class_to_zone={"RED": "RED", "GREEN": "GREEN", "BLUE": "BLUE"},
            sensor_max_age_s=sensor_max_age,
            tracking_max_distance_m=tracking_max_distance,
            planner_max_detection_age_s=planner_max_age,
            position_sigma_limit_m=sigma_xy,
            yaw_sigma_limit_rad=sigma_yaw,
            gripper_home_m=home,
            score_weights=score_weights,
        )
