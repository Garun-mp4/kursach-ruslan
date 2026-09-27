from __future__ import annotations

import json
from pathlib import Path

import cv2
import mujoco
import numpy as np
from PIL import Image

from .config import ROOT, load_ssot, value
from .placement_camera import CAMERA_NAME, intrinsic_matrix, load_integrated_model, placement_camera_values


OUTPUT_DIR = ROOT / "05_Верификация" / "integration" / "sensors"


def _project_xy(data: mujoco.MjData, camera_id: int, intrinsics: np.ndarray,
                x_m: float, y_m: float, z_m: float) -> tuple[float, float]:
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3)
    position = np.asarray(data.cam_xpos[camera_id], dtype=np.float64)
    camera_xyz = rotation.T @ (np.asarray([x_m, y_m, z_m]) - position)
    if not np.all(np.isfinite(camera_xyz)) or camera_xyz[2] >= -1e-9:
        raise ValueError("Calibration point is behind the placement camera")
    depth = -float(camera_xyz[2])
    # MuJoCo camera coordinates use +Y up, while RGB arrays use v down.
    return (
        float(intrinsics[0, 0] * camera_xyz[0] / depth + intrinsics[0, 2]),
        float(intrinsics[1, 2] - intrinsics[1, 1] * camera_xyz[1] / depth),
    )


def _empty_cell_state(model: mujoco.MjModel, data: mujoco.MjData,
                      ssot: dict) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_observation")
    if key_id < 0:
        raise ValueError("M2 model has no home_observation keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    object_size = np.asarray(value(ssot, "object.size_xyz_m"), dtype=np.float64)
    table_size = np.asarray(value(ssot, "cell.table_size_xy_m"), dtype=np.float64)
    table_z = float(value(ssot, "cell.table_top_z_m"))
    park_x = table_size[0] / 2.0 - object_size[0] / 2.0
    body_names = [
        f"object_{color}_{index:02d}"
        for index in (1, 2)
        for color in ("red", "green", "blue")
    ]
    park_y = np.linspace(-0.30, 0.30, len(body_names))
    for name, y_m in zip(body_names, park_y, strict=True):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"M2 model is missing body {name!r}")
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Background object {name!r} must have a free joint")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr:qpos_adr + 7] = (
            park_x, float(y_m), table_z + object_size[2] / 2.0, 1.0, 0.0, 0.0, 0.0
        )
    mujoco.mj_forward(model, data)


def build_placement_camera_assets(output_dir: Path = OUTPUT_DIR) -> dict:
    ssot = load_ssot()
    camera = placement_camera_values(ssot)
    model = load_integrated_model(ssot)
    data = mujoco.MjData(model)
    _empty_cell_state(model, data, ssot)

    width, height = camera["resolution_px"]
    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=CAMERA_NAME)
        background_rgb = np.asarray(renderer.render())
    finally:
        renderer.close()
    if background_rgb.shape != (height, width, 3) or background_rgb.dtype != np.uint8:
        raise ValueError("Placement background render has an invalid RGB shape or dtype")

    intrinsics = intrinsic_matrix(width, height, camera["fovy_deg"])
    z_plane = float(camera["object_top_plane_z_m"])
    trays = value(ssot, "cell.tray_centers_xy_m")
    tray_size = np.asarray(value(ssot, "cell.tray_outer_size_xy_m"), dtype=np.float64)
    x_bounds = (
        min(float(trays[name][0]) for name in ("RED", "GREEN", "BLUE")) - tray_size[0] / 2.0,
        max(float(trays[name][0]) for name in ("RED", "GREEN", "BLUE")) + tray_size[0] / 2.0,
    )
    y_bounds = (
        min(float(trays[name][1]) for name in ("RED", "GREEN", "BLUE")) - tray_size[1] / 2.0,
        max(float(trays[name][1]) for name in ("RED", "GREEN", "BLUE")) + tray_size[1] / 2.0,
    )
    grid_x = np.linspace(*x_bounds, 7)
    grid_y = np.linspace(*y_bounds, 7)
    calibration_xy = np.asarray([(x, y) for y in grid_y for x in grid_x], dtype=np.float64)
    calibration_uv = np.asarray([
        _project_xy(data, camera_id, intrinsics, float(x), float(y), z_plane)
        for x, y in calibration_xy
    ], dtype=np.float64)
    homography, _ = cv2.findHomography(calibration_uv, calibration_xy, method=0)
    if homography is None or not np.all(np.isfinite(homography)):
        raise ValueError("Could not compute placement-camera planar homography")
    homography /= homography[2, 2]

    verification_xy = np.asarray([
        (x, y)
        for y in np.linspace(y_bounds[0], y_bounds[1], 11)[1::2]
        for x in np.linspace(x_bounds[0], x_bounds[1], 13)[1::2]
    ], dtype=np.float64)
    verification_uv = np.asarray([
        _project_xy(data, camera_id, intrinsics, float(x), float(y), z_plane)
        for x, y in verification_xy
    ], dtype=np.float64)
    mapped_xy = cv2.perspectiveTransform(
        verification_uv.reshape(-1, 1, 2), homography
    ).reshape(-1, 2)
    errors_m = np.linalg.norm(mapped_xy - verification_xy, axis=1)
    if not np.all(np.isfinite(errors_m)):
        raise ValueError("Placement-camera calibration verification produced non-finite error")

    quaternion = np.asarray(model.cam_quat[camera_id], dtype=np.float64)
    calibration = {
        "schema_version": "M7-placement-calibration-v1.0",
        "calibration_id": "M7-HOMOGRAPHY-PLACEMENT-TRAY-TOP-v1.0",
        "camera_config_id": camera["config_id"],
        "method": "known_static_camera_pose_projected_world_grid",
        "source_frame": f'{CAMERA_NAME} RGB pixel coordinates (u right, v down, top-left)',
        "destination_frame": "BASE XY in meters",
        "camera_name": CAMERA_NAME,
        "camera_position_world_m": camera["position_world_m"].tolist(),
        "camera_quaternion_wxyz": quaternion.tolist(),
        "camera_target_world_m": camera["target_world_m"].tolist(),
        "fovy_deg": camera["fovy_deg"],
        "resolution_px": [width, height],
        "intrinsics_fx_fy_cx_cy_px": [
            float(intrinsics[0, 0]), float(intrinsics[1, 1]),
            float(intrinsics[0, 2]), float(intrinsics[1, 2]),
        ],
        "object_top_plane_z_m": z_plane,
        "calibration_point_count": int(len(calibration_xy)),
        "calibration_points_base_xy_m": calibration_xy.tolist(),
        "calibration_points_detected_uv_px": calibration_uv.tolist(),
        "pixel_to_base_xy_homography": homography.tolist(),
        "base_xy_to_pixel_homography": np.linalg.inv(homography).tolist(),
        "verification_point_count": int(len(verification_xy)),
        "verification_metrics": {
            "rmse_m": float(np.sqrt(np.mean(np.square(errors_m)))),
            "p95_m": float(np.percentile(errors_m, 95)),
            "max_m": float(np.max(errors_m)),
        },
        "validation_note": (
            "Analytic virtual-camera projection and a held-out planar grid; this validates "
            "the simulated camera transform, not a physical camera calibration."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "placement_camera_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    np.save(output_dir / "placement_background_rgb.npy", background_rgb, allow_pickle=False)
    Image.fromarray(background_rgb, mode="RGB").save(output_dir / "placement_background_preview.png")
    return calibration


def main() -> int:
    result = build_placement_camera_assets()
    metrics = result["verification_metrics"]
    print(
        "M7 placement-camera assets generated: "
        f"points={result['calibration_point_count']}, "
        f"held_out_max={metrics['max_m']:.3e} m, "
        f"config={result['camera_config_id']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
