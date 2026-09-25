from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SensorSnapshot:
    frame: CameraFrame
    input_batch: DetectionBatch
    placement_batch: DetectionBatch


class PublicSensorPipeline:
    """RGB camera plus two M4 detector views; exposes images/estimates, never scene truth."""

    def __init__(self, model: mujoco.MjModel, *, renderer: mujoco.Renderer | None = None):
        self.model = model
        self.ssot = load_ssot()
        perception_dir = ROOT / "04_Программа" / "perception"
        verification_dir = ROOT / "05_Верификация" / "perception"
        self.config_path = perception_dir / "perception_config.yaml"
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        calibration_path = verification_dir / "calibration" / "camera_calibration.json"
        background_path = verification_dir / "validation" / "background_reference_rgb.npy"
        if not calibration_path.is_file() or not background_path.is_file():
            raise FileNotFoundError("M4 calibration and RGB background artifacts are required by M7")
        self.calibration = PlanarCalibration.from_json(str(calibration_path))
        self.background = np.load(background_path, allow_pickle=False)
        width, height = map(int, self.config["sensor_interface"]["resolution_px"])
        if int(model.ncam) != 1:
            raise ValueError("M7 expects the single calibrated M2 overhead RGB camera")
        camera_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, 0)
        if not camera_name:
            raise ValueError("M2 overhead camera has no model name")
        self.renderer = renderer or mujoco.Renderer(model, height=height, width=width)
        self.camera = MuJoCoRGBAdapter(
            model=model,
            renderer=self.renderer,
            camera_name=camera_name,
            width=width,
            height=height,
            camera_config_id=str(self.config["sensor_interface"]["camera_config_id"]),
        )
        self.input_perception = RGBObjectPerception(
            self.config, self.calibration, self.background
        )
        placement_config: dict[str, Any] = yaml.safe_load(
            yaml.safe_dump(self.config, allow_unicode=True, sort_keys=False)
        )
        trays = self.ssot["parameters"]["cell.tray_centers_xy_m"]["value"]
        tray_size = self.ssot["parameters"]["cell.tray_outer_size_xy_m"]["value"]
        half_x, half_y = float(tray_size[0]) / 2, float(tray_size[1]) / 2
        centers = [tuple(map(float, trays[color])) for color in ("RED", "GREEN", "BLUE")]
        placement_config["regions"]["input_xy_bounds_m"] = [
            [min(x for x, _ in centers) - half_x, max(x for x, _ in centers) + half_x],
            [min(y for _, y in centers) - half_y, max(y for _, y in centers) + half_y],
        ]
        self.placement_perception = RGBObjectPerception(
            placement_config, self.calibration, self.background
        )

    def capture(self, data: mujoco.MjData, *, simulation_time_s: float,
                valid: bool = True) -> SensorSnapshot:
        frame = self.camera.capture(
            data, simulation_time_s=simulation_time_s, valid=valid
        )
        input_batch = self.input_perception.detect(
            frame, now_simulation_time_s=simulation_time_s
        )
        placement_batch = self.placement_perception.detect(
            frame, now_simulation_time_s=simulation_time_s
        )
        return SensorSnapshot(frame, input_batch, placement_batch)

    def close(self) -> None:
        self.renderer.close()
