from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any

import mujoco
import numpy as np
import yaml

from camera.frame import CameraFrame
from camera.mujoco_adapter import MuJoCoRGBAdapter
from perception.detector import RGBObjectPerception
from perception.geometry import PlanarCalibration
from perception.types import DetectionBatch

from .config import ROOT, load_ssot
from .placement_camera import CAMERA_NAME, placement_camera_values


def apply_rgb_perturbation(
    frame: CameraFrame,
    *,
    gain: float,
    noise_sigma: float,
    rng: np.random.Generator,
) -> CameraFrame:
    """Apply a reproducible image-plane gain/noise perturbation to a public RGB frame."""
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("RGB gain must be finite and positive")
    if not math.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("RGB noise sigma must be finite and nonnegative")
    if not frame.valid or frame.rgb is None:
        return frame
    if gain == 1.0 and noise_sigma == 0.0:
        return frame
    values = frame.rgb.astype(np.float32) * float(gain)
    if noise_sigma:
        values += rng.normal(0.0, float(noise_sigma), values.shape)
    rgb = np.ascontiguousarray(np.clip(np.rint(values), 0, 255).astype(np.uint8))
    rgb.setflags(write=False)
    return replace(frame, rgb=rgb)


def offset_planar_calibration(
    calibration: PlanarCalibration,
    bias_xy_m: tuple[float, float],
) -> PlanarCalibration:
    """Return a calibration whose public pixel-to-world estimates have a fixed XY bias."""
    dx, dy = map(float, bias_xy_m)
    if not all(math.isfinite(value) for value in (dx, dy)):
        raise ValueError("Calibration bias must be finite")
    if dx == 0.0 and dy == 0.0:
        return calibration
    homography = np.asarray(calibration.pixel_to_base_xy, dtype=np.float64).copy()
    homography[0, :] += dx * homography[2, :]
    homography[1, :] += dy * homography[2, :]
    inverse = np.linalg.inv(homography)
    return PlanarCalibration(
        homography,
        inverse,
        calibration.object_top_z_m,
        f"{calibration.calibration_id}+bias({dx:.6g},{dy:.6g})m",
    )


@dataclass(frozen=True)
class SensorSnapshot:
    frame: CameraFrame
    placement_frame: CameraFrame
    input_batch: DetectionBatch
    placement_batch: DetectionBatch


class PublicSensorPipeline:
    """Separate overhead input and oblique placement RGB views; never exposes scene truth."""

    def __init__(self, model: mujoco.MjModel, *, renderer: mujoco.Renderer | None = None,
                 rgb_gain: float = 1.0, rgb_noise_sigma: float = 0.0,
                 noise_seed: int = 0,
                 input_calibration_bias_xy_m: tuple[float, float] = (0.0, 0.0)):
        if not math.isfinite(rgb_gain) or rgb_gain <= 0:
            raise ValueError("RGB gain must be finite and positive")
        if not math.isfinite(rgb_noise_sigma) or rgb_noise_sigma < 0:
            raise ValueError("RGB noise sigma must be finite and nonnegative")
        if isinstance(noise_seed, bool) or not isinstance(noise_seed, (int, np.integer)):
            raise ValueError("RGB noise seed must be an integer")
        self.rgb_gain = float(rgb_gain)
        self.rgb_noise_sigma = float(rgb_noise_sigma)
        self.noise_seed = int(noise_seed)
        self._noise_rng = np.random.default_rng(self.noise_seed)
        self.model = model
        self.ssot = load_ssot()
        perception_dir = ROOT / "04_Программа" / "perception"
        verification_dir = ROOT / "05_Верификация" / "perception"
        placement_assets_dir = ROOT / "05_Верификация" / "integration" / "sensors"
        self.config_path = perception_dir / "perception_config.yaml"
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        calibration_path = placement_assets_dir / "placement_camera_calibration.json"
        background_path = placement_assets_dir / "placement_background_rgb.npy"
        if not calibration_path.is_file() or not background_path.is_file():
            raise FileNotFoundError(
                "M7 placement-camera calibration and empty-cell RGB background are required"
            )
        self.input_calibration = PlanarCalibration.from_json(
            str(verification_dir / "calibration" / "camera_calibration.json")
        )
        self.input_calibration = offset_planar_calibration(
            self.input_calibration, input_calibration_bias_xy_m
        )
        self.input_background = np.load(
            verification_dir / "validation" / "background_reference_rgb.npy",
            allow_pickle=False,
        )
        placement_camera = placement_camera_values(self.ssot)
        self.placement_calibration = PlanarCalibration.from_json(str(calibration_path))
        calibration_metadata = json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_metadata.get("camera_config_id") != placement_camera["config_id"]:
            raise ValueError("Placement calibration identity disagrees with the SSOT camera view")
        if calibration_metadata.get("camera_name") != CAMERA_NAME:
            raise ValueError("Placement calibration belongs to a different MJCF camera")
        if tuple(calibration_metadata.get("resolution_px", ())) != placement_camera["resolution_px"]:
            raise ValueError("Placement calibration resolution disagrees with the SSOT camera view")
        if not np.isclose(
            self.placement_calibration.object_top_z_m,
            placement_camera["object_top_plane_z_m"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("Placement calibration plane disagrees with the SSOT tray geometry")
        self.placement_background = np.load(background_path, allow_pickle=False)
        width, height = map(int, self.config["sensor_interface"]["resolution_px"])
        placement_width, placement_height = placement_camera["resolution_px"]
        if (width, height) != (placement_width, placement_height):
            raise ValueError("M4 input and M7 placement views must share a renderer resolution")
        if int(model.ncam) != 2:
            raise ValueError("M7 expects the M2 overhead and M7 placement RGB cameras")
        input_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_rgb")
        placement_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if min(input_camera_id, placement_camera_id) < 0:
            raise ValueError("M7 model is missing a named input or placement RGB camera")
        self.renderer = renderer or mujoco.Renderer(model, height=height, width=width)
        self.camera = MuJoCoRGBAdapter(
            model=model,
            renderer=self.renderer,
            camera_name="overhead_rgb",
            width=width,
            height=height,
            camera_config_id=str(self.config["sensor_interface"]["camera_config_id"]),
        )
        self.placement_camera = MuJoCoRGBAdapter(
            model=model,
            renderer=self.renderer,
            camera_name=CAMERA_NAME,
            width=width,
            height=height,
            camera_config_id=placement_camera["config_id"],
        )
        self.input_perception = RGBObjectPerception(
            self.config, self.input_calibration, self.input_background
        )
        placement_config: dict[str, Any] = copy.deepcopy(self.config)
        placement_config["sensor_interface"]["camera_config_id"] = placement_camera["config_id"]
        placement_config["sensor_interface"]["maximum_age_s"] = placement_camera["max_age_s"]
        placement_config["sensor_interface"]["resolution_px"] = [width, height]
        trays = self.ssot["parameters"]["cell.tray_centers_xy_m"]["value"]
        tray_size = self.ssot["parameters"]["cell.tray_outer_size_xy_m"]["value"]
        half_x, half_y = float(tray_size[0]) / 2, float(tray_size[1]) / 2
        centers = [tuple(map(float, trays[color])) for color in ("RED", "GREEN", "BLUE")]
        placement_config["regions"]["input_xy_bounds_m"] = [
            [min(x for x, _ in centers) - half_x, max(x for x, _ in centers) + half_x],
            [min(y for _, y in centers) - half_y, max(y for _, y in centers) + half_y],
        ]
        placement_config["photometric_normalization"]["anchor_xy_bounds_m"] = (
            placement_camera["anchor_xy_bounds_m"].tolist()
        )
        placement_config["candidate_detection"]["nominal_projected_object_area_px"] = (
            placement_camera["nominal_object_area_px"]
        )
        self.placement_perception = RGBObjectPerception(
            placement_config, self.placement_calibration, self.placement_background
        )
        self._last_valid_input_frame: CameraFrame | None = None
        self._performance = {
            "capture_count": 0,
            "render_wall_total_s": 0.0,
            "render_wall_max_s": 0.0,
            "perception_wall_total_s": 0.0,
            "perception_wall_max_s": 0.0,
        }

    def reset_performance_counters(self) -> None:
        for key in self._performance:
            self._performance[key] = 0.0 if key.endswith("_s") else 0

    def performance_summary(self) -> dict[str, float | int]:
        count = int(self._performance["capture_count"])
        render_total = float(self._performance["render_wall_total_s"])
        perception_total = float(self._performance["perception_wall_total_s"])
        return {
            "capture_count": count,
            "render_wall_total_s": render_total,
            "render_wall_average_s": render_total / count if count else 0.0,
            "render_wall_max_s": float(self._performance["render_wall_max_s"]),
            "perception_wall_total_s": perception_total,
            "perception_wall_average_s": perception_total / count if count else 0.0,
            "perception_wall_max_s": float(self._performance["perception_wall_max_s"]),
        }

    def capture(self, data: mujoco.MjData, *, simulation_time_s: float,
                valid: bool = True, placement_valid: bool | None = None,
                stale_input_age_s: float | None = None) -> SensorSnapshot:
        if stale_input_age_s is not None and (
            not np.isfinite(stale_input_age_s) or stale_input_age_s <= 0
        ):
            raise ValueError("Stale input frame age must be finite and positive")
        render_started = time.perf_counter()
        current_frame = self.camera.capture(
            data, simulation_time_s=simulation_time_s, valid=valid
        )
        placement_frame = self.placement_camera.capture(
            data,
            simulation_time_s=simulation_time_s,
            valid=valid if placement_valid is None else placement_valid,
        )
        current_frame = apply_rgb_perturbation(
            current_frame, gain=self.rgb_gain,
            noise_sigma=self.rgb_noise_sigma, rng=self._noise_rng,
        )
        placement_frame = apply_rgb_perturbation(
            placement_frame, gain=self.rgb_gain,
            noise_sigma=self.rgb_noise_sigma, rng=self._noise_rng,
        )
        render_elapsed = time.perf_counter() - render_started
        if (current_frame.frame_id != placement_frame.frame_id
                or current_frame.simulation_time_s != placement_frame.simulation_time_s):
            raise RuntimeError("Input and placement cameras lost synchronized frame sequencing")
        if stale_input_age_s is not None:
            if not valid or self._last_valid_input_frame is None:
                raise ValueError("Stale-frame injection requires a preceding valid RGB frame")
            frame = replace(
                self._last_valid_input_frame,
                frame_id=current_frame.frame_id,
                simulation_time_s=float(simulation_time_s) - float(stale_input_age_s),
            )
        else:
            frame = current_frame
        if current_frame.valid:
            self._last_valid_input_frame = current_frame
        perception_started = time.perf_counter()
        input_batch = self.input_perception.detect(
            frame, now_simulation_time_s=simulation_time_s
        )
        placement_batch = self.placement_perception.detect(
            placement_frame, now_simulation_time_s=simulation_time_s
        )
        perception_elapsed = time.perf_counter() - perception_started
        self._performance["capture_count"] += 1
        self._performance["render_wall_total_s"] += render_elapsed
        self._performance["render_wall_max_s"] = max(
            self._performance["render_wall_max_s"], render_elapsed
        )
        self._performance["perception_wall_total_s"] += perception_elapsed
        self._performance["perception_wall_max_s"] = max(
            self._performance["perception_wall_max_s"], perception_elapsed
        )
        return SensorSnapshot(frame, placement_frame, input_batch, placement_batch)

    def close(self) -> None:
        self.renderer.close()
