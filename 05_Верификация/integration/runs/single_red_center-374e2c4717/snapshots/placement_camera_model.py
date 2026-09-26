from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from .config import MODEL_PATH, load_ssot, value


CAMERA_NAME = "placement_rgb"


def placement_camera_values(ssot: dict | None = None) -> dict:
    config = ssot or load_ssot()
    p = {
        "name": str(value(config, "camera.placement_name")),
        "config_id": str(value(config, "camera.placement_config_id")),
        "position_world_m": np.asarray(
            value(config, "camera.placement_position_world_m"), dtype=np.float64
        ),
        "target_world_m": np.asarray(
            value(config, "camera.placement_target_world_m"), dtype=np.float64
        ),
        "fovy_deg": float(value(config, "camera.placement_fovy_deg")),
        "resolution_px": tuple(map(int, value(config, "camera.placement_resolution_px"))),
        "max_age_s": float(value(config, "camera.placement_max_age_s")),
        "object_top_plane_z_m": float(
            value(config, "camera.placement_object_top_plane_z_m")
        ),
        "nominal_object_area_px": float(
            value(config, "camera.placement_nominal_object_area_px")
        ),
        "anchor_xy_bounds_m": np.asarray(
            value(config, "camera.placement_anchor_xy_bounds_m"), dtype=np.float64
        ),
    }
    if p["name"] != CAMERA_NAME:
        raise ValueError(f"Placement camera name must remain {CAMERA_NAME!r}")
    if p["position_world_m"].shape != (3,) or p["target_world_m"].shape != (3,):
        raise ValueError("Placement camera position and target must be 3D vectors")
    if p["anchor_xy_bounds_m"].shape != (2, 2):
        raise ValueError("Placement camera normalization anchor must be XY bounds")
    if len(p["resolution_px"]) != 2 or min(p["resolution_px"]) < 1:
        raise ValueError("Placement camera resolution must contain positive width and height")
    if not 1.0 < p["fovy_deg"] < 179.0 or p["max_age_s"] <= 0:
        raise ValueError("Placement camera FOV and frame-age limit are invalid")
    if p["nominal_object_area_px"] <= 0:
        raise ValueError("Expected projected object area must be positive")
    if not np.all(np.isfinite(p["position_world_m"])) or not np.all(
        np.isfinite(p["target_world_m"])
    ):
        raise ValueError("Placement camera pose must be finite")
    if np.linalg.norm(p["target_world_m"] - p["position_world_m"]) < 1e-9:
        raise ValueError("Placement camera target must differ from its position")
    return p


def camera_quaternion_wxyz(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    back = -direction
    right = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-9:
        raise ValueError("Placement camera optical axis cannot be parallel to world up")
    right /= right_norm
    up = np.cross(back, right)
    up /= np.linalg.norm(up)
    rotation_world_from_camera = np.column_stack((right, up, back))
    quaternion = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quaternion, rotation_world_from_camera.ravel())
    return quaternion


def integrated_mjcf_text(base_xml: str | None = None, ssot: dict | None = None) -> str:
    source = base_xml if base_xml is not None else MODEL_PATH.read_text(encoding="utf-8")
    if f'name="{CAMERA_NAME}"' in source:
        raise ValueError(f"M2 source already contains the M7 camera {CAMERA_NAME!r}")
    marker = "</worldbody>"
    if source.count(marker) != 1:
        raise ValueError("M2 MJCF must contain exactly one worldbody closing tag")
    camera = placement_camera_values(ssot)
    pos = camera["position_world_m"]
    target = camera["target_world_m"]
    quat = camera_quaternion_wxyz(pos, target)
    attrs = (
        f'name="{CAMERA_NAME}" '
        f'pos="{" ".join(format(float(v), ".17g") for v in pos)}" '
        f'quat="{" ".join(format(float(v), ".17g") for v in quat)}" '
        f'fovy="{format(camera["fovy_deg"], ".17g")}" mode="fixed"'
    )
    return source.replace(marker, f'    <camera {attrs} />\n  {marker}')


def load_integrated_model(ssot: dict | None = None) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(integrated_mjcf_text(ssot=ssot))
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME) < 0:
        raise RuntimeError("M7 integrated model did not compile its placement RGB camera")
    return model


def intrinsic_matrix(width: int, height: int, fovy_deg: float) -> np.ndarray:
    focal = height / (2.0 * math.tan(math.radians(fovy_deg) / 2.0))
    return np.array(
        [[focal, 0.0, (width - 1.0) / 2.0],
         [0.0, focal, (height - 1.0) / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
