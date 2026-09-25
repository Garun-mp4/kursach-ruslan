from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
PROGRAM_DIR = ROOT / "04_Программа"
SSOT_PATH = ROOT / "02_Спецификация" / "параметры_системы.yaml"
MODEL_PATH = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
RUNTIME_CONFIG_PATH = Path(__file__).with_name("runtime_config.yaml")


@dataclass(frozen=True)
class RuntimeConfig:
    control_period_s: float
    telemetry_period_s: float
    observation_settle_s: float
    actuator_settle_timeout_s: float
    actuator_settle_stable_s: float
    recovery_gripper_open_s: float
    joint_position_tolerance_rad: float
    slide_position_tolerance_m: float
    gripper_position_tolerance_m: float
    touch_contact_threshold_N: float
    placement_slot_acceptance_radius_m: float
    evaluator_rest_height_tolerance_m: float
    evaluator_linear_speed_tolerance_m_s: float
    evaluator_angular_speed_tolerance_rad_s: float
    renderer_width_px: int
    renderer_height_px: int
    randomized_scene_min_object_spacing_m: float
    randomized_scene_max_attempts: int
    default_max_simulation_time_s: float

    @classmethod
    def load(cls, path: str | Path = RUNTIME_CONFIG_PATH) -> "RuntimeConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("schema_version") != "M7-v1.0":
            raise ValueError("Unsupported M7 runtime configuration")
        values = raw.get("runtime")
        if not isinstance(values, dict):
            raise ValueError("M7 runtime section is missing")
        fields = cls.__dataclass_fields__
        if set(values) != set(fields):
            missing = sorted(set(fields) - set(values))
            extra = sorted(set(values) - set(fields))
            raise ValueError(f"M7 runtime fields differ from schema; missing={missing}, extra={extra}")
        config = cls(**values)
        for name in fields:
            value = getattr(config, name)
            if isinstance(value, bool):
                raise ValueError(f"M7 runtime value {name} cannot be boolean")
            if isinstance(value, int):
                if value < 1:
                    raise ValueError(f"M7 runtime integer {name} must be positive")
            elif not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"M7 runtime value {name} must be positive")
        return config


def load_ssot(path: str | Path = SSOT_PATH) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(raw, dict) or raw.get("config_version") not in {"M5-v1.0", "M5-v1.1", "M5-v1.2", "M5-v1.3", "M7-v1.0", "M7-v1.1"}:
        raise ValueError("M7 requires the accepted M5 SSOT baseline")
    for section in ("parameters", "project"):
        if not isinstance(raw.get(section), dict):
            raise ValueError(f"SSOT section {section!r} is missing")
    return raw


def value(ssot: dict[str, Any], key: str) -> Any:
    try:
        return ssot["parameters"][key]["value"]
    except KeyError as exc:
        raise ValueError(f"SSOT is missing parameter {key!r}") from exc
