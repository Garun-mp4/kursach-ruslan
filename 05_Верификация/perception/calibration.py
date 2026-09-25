from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np

from camera.mujoco_adapter import MuJoCoRGBAdapter
from rig import calibration_grid, hide_diagnostic_sites, hide_robot_geometries, marker_board_xml


def _render_marker_points(root: Path, points: list[tuple[float, float]], z_top_m: float,
                          config: dict[str, Any]) -> np.ndarray:
    xml = hide_robot_geometries(hide_diagnostic_sites(
        marker_board_xml(root, points, z_top_m=z_top_m)
    ))
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    width, height = map(int, config["sensor_interface"]["resolution_px"])
    renderer = mujoco.Renderer(model, width=width, height=height)
    adapter = MuJoCoRGBAdapter(model, renderer, "overhead_rgb", width, height,
                               config["sensor_interface"]["camera_config_id"])
    frame = adapter.capture(data, simulation_time_s=0.0)
    if not frame.valid or frame.rgb is None:
        renderer.close()
        raise RuntimeError("Calibration renderer returned invalid image")
    rgb = frame.rgb.copy()
    renderer.close()
    return rgb


def detect_yellow_fiducials(rgb: np.ndarray, expected_count: int,
                           expected_uv: np.ndarray | None = None) -> np.ndarray:
    """Detect visible yellow fiducial centers from RGB pixels, not renderer metadata."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected uint8 RGB frame")
    mask = ((image[:, :, 0] >= 150) & (image[:, :, 1] >= 120) &
            (image[:, :, 2] <= 115)).astype(np.uint8)
    count, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for area, center in zip(stats[1:, cv2.CC_STAT_AREA], centers[1:]):
        if 10 <= int(area) <= 100:
            found.append((float(center[0]), float(center[1])))
    if len(found) != expected_count:
        raise RuntimeError(f"Expected {expected_count} calibration markers, detected {len(found)}")
    if expected_uv is not None:
        expected = np.asarray(expected_uv, dtype=np.float64).reshape(-1, 2)
        if len(expected) != expected_count:
            raise ValueError("expected_uv count does not match expected_count")
        points = np.asarray(found, dtype=np.float64)
        distances = np.linalg.norm(expected[:, None, :] - points[None, :, :], axis=2)
        assigned: list[np.ndarray | None] = [None] * expected_count
        pairs = sorted((float(distances[i, j]), i, j)
                       for i in range(expected_count) for j in range(expected_count))
        used_i: set[int] = set()
        used_j: set[int] = set()
        for distance, i, j in pairs:
            if i not in used_i and j not in used_j:
                if distance > 12.0:
                    raise RuntimeError(f"Calibration marker association residual {distance:.2f}px")
                assigned[i] = points[j]
                used_i.add(i)
                used_j.add(j)
        return np.asarray(assigned, dtype=np.float64)
    found.sort(key=lambda p: (p[1], p[0]))
    return np.asarray(found, dtype=np.float64)


def run_calibration(root: Path, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    z_top = float(config["calibration"]["object_top_plane_z_m"])
    calibration_xy = calibration_grid()
    calibration_rgb = _render_marker_points(root, calibration_xy, z_top, config)
    cv2.imwrite(str(output_dir / "calibration_raw_rgb.png"), cv2.cvtColor(calibration_rgb, cv2.COLOR_RGB2BGR))
    K = config["camera"]["intrinsics_fx_fy_cx_cy_px"]
    position = config["camera"]["position_world_m"]
    def nominal_uv(xy: list[tuple[float, float]]) -> np.ndarray:
        denom = float(position[2]) - z_top
        return np.asarray([[float(K[0]) * x / denom + float(K[2]),
                            -float(K[1]) * y / denom + float(K[3])]
                           for x, y in xy], dtype=np.float64)
    calibration_uv = detect_yellow_fiducials(
        calibration_rgb, len(calibration_xy), nominal_uv(calibration_xy)
    )
    H_pixel_to_xy, inliers = cv2.findHomography(
        calibration_uv.astype(np.float32), np.asarray(calibration_xy, dtype=np.float32), method=0
    )
    if H_pixel_to_xy is None or not np.all(np.isfinite(H_pixel_to_xy)):
        raise RuntimeError("Homography fitting failed")
    H_pixel_to_xy = H_pixel_to_xy / H_pixel_to_xy[2, 2]
    H_xy_to_pixel = np.linalg.inv(H_pixel_to_xy)

    # A second, non-overlapping set spanning center, edges, input and trays.
    xs = [-0.455, -0.335, -0.205, -0.075, 0.065, 0.195, 0.325, 0.455]
    ys = [-0.405, -0.295, -0.185, -0.075, 0.045, 0.155, 0.265, 0.405]
    validation_xy = [(x, y) for y in reversed(ys) for x in xs]
    calibration_set = {tuple(round(v, 8) for v in p) for p in calibration_xy}
    if any(tuple(round(v, 8) for v in p) in calibration_set for p in validation_xy):
        raise AssertionError("Calibration/validation points overlap")
    validation_rgb = _render_marker_points(root, validation_xy, z_top, config)
    cv2.imwrite(str(output_dir / "calibration_validation_rgb.png"), cv2.cvtColor(validation_rgb, cv2.COLOR_RGB2BGR))
    validation_uv = detect_yellow_fiducials(
        validation_rgb, len(validation_xy), nominal_uv(validation_xy)
    )
    predicted_xy = cv2.perspectiveTransform(
        validation_uv.astype(np.float64).reshape(-1, 1, 2), H_pixel_to_xy
    ).reshape(-1, 2)
    truth_xy = np.asarray(validation_xy, dtype=np.float64)
    errors = predicted_xy - truth_xy
    planar = np.linalg.norm(errors, axis=1)
    artifact = {
        "calibration_id": "M4-HOMOGRAPHY-TOP-PLANE-v1.1",
        "method": "planar_homography_from_image_fiducials",
        "source_frame": "overhead_rgb rendered RGB image pixel coordinates (u right, v down, top-left)",
        "destination_frame": "BASE XY in meters",
        "object_top_plane_z_m": z_top,
        "plane_definition": "top face plane of the M2 cuboid object; table_top_z + object.size_xyz_m[2]",
        "calibration_point_count": len(calibration_xy),
        "calibration_points_base_xy_m": calibration_xy,
        "calibration_points_detected_uv_px": calibration_uv.tolist(),
        "validation_point_count": len(validation_xy),
        "validation_points_base_xy_m": validation_xy,
        "validation_points_detected_uv_px": validation_uv.tolist(),
        "pixel_to_base_xy_homography": H_pixel_to_xy.tolist(),
        "base_xy_to_pixel_homography": H_xy_to_pixel.tolist(),
        "validation_metrics": {
            "n": len(planar),
            "mean_x_error_m": float(np.mean(errors[:, 0])),
            "mean_y_error_m": float(np.mean(errors[:, 1])),
            "rmse_x_m": float(np.sqrt(np.mean(errors[:, 0] ** 2))),
            "rmse_y_m": float(np.sqrt(np.mean(errors[:, 1] ** 2))),
            "mean_planar_error_m": float(np.mean(planar)),
            "p95_planar_error_m": float(np.percentile(planar, 95)),
            "max_planar_error_m": float(np.max(planar)),
            "max_error_point_index": int(np.argmax(planar)),
        },
        "nominal_intrinsics_from_M2_px": config["camera"]["intrinsics_fx_fy_cx_cy_px"],
        "association_note": "M2 intrinsics are used only to associate visually detected calibration fiducials with the printed calibration grid; homography is fitted from detected pixel centers and known planar BASE coordinates.",
        "camera_config_id": config["sensor_interface"]["camera_config_id"],
        "opencv_version": cv2.__version__,
        "mujoco_version": mujoco.__version__,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "camera_calibration.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savetxt(output_dir / "calibration_validation_errors.csv",
               np.column_stack([truth_xy, validation_uv, predicted_xy, errors, planar]),
               delimiter=",", header="x_true_m,y_true_m,u_px,v_px,x_est_m,y_est_m,dx_m,dy_m,planar_m",
               comments="", fmt="%.10g")
    return artifact
