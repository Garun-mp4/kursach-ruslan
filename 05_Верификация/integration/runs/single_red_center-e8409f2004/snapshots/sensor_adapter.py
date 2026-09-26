from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
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


@dataclass(frozen=True)
class SensorSnapshot:
    frame: CameraFrame
    placement_frame: CameraFrame
    input_batch: DetectionBatch
    placement_batch: DetectionBatch


class PublicSensorPipeline:
    """Separate overhead input and oblique placement RGB views; never exposes scene truth."""

    def __init__(self, model: mujoco.MjModel, *, renderer: mujoco.Renderer | None = None):
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

    def capture(self, data: mujoco.MjData, *, simulation_time_s: float,
                valid: bool = True, placement_valid: bool | None = None) -> SensorSnapshot:
        frame = self.camera.capture(
            data, simulation_time_s=simulation_time_s, valid=valid
        )
        placement_frame = self.placement_camera.capture(
            data,
            simulation_time_s=simulation_time_s,
            valid=valid if placement_valid is None else placement_valid,
        )
        if (frame.frame_id != placement_frame.frame_id
                or frame.simulation_time_s != placement_frame.simulation_time_s):
            raise RuntimeError("Input and placement cameras lost synchronized frame sequencing")
        input_batch = self.input_perception.detect(
            frame, now_simulation_time_s=simulation_time_s
        )
        placement_batch = self.placement_perception.detect(
            placement_frame, now_simulation_time_s=simulation_time_s
        )
        return SensorSnapshot(frame, placement_frame, input_batch, placement_batch)

    def close(self) -> None:
        self.renderer.close()
