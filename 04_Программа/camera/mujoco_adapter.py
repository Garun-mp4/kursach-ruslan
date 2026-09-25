from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .frame import CameraFrame


@dataclass
class MuJoCoRGBAdapter:
    """Render and expose only an RGB image and timing/configuration metadata."""

    model: mujoco.MjModel
    renderer: mujoco.Renderer
    camera_name: str
    width: int
    height: int
    camera_config_id: str
    frame_counter: int = 0

    def capture(
        self,
        data: mujoco.MjData,
        *,
        simulation_time_s: float,
        valid: bool = True,
        noise_sigma: float = 0.0,
        noise_seed: int | None = None,
    ) -> CameraFrame:
        self.frame_counter += 1
        if not valid:
            return CameraFrame(None, self.frame_counter, float(simulation_time_s), False,
                               self.camera_config_id, "SENSOR_INVALID")
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = np.asarray(self.renderer.render())
        if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
            return CameraFrame(None, self.frame_counter, float(simulation_time_s), False,
                               self.camera_config_id, "MALFORMED_RENDER")
        if noise_sigma < 0:
            raise ValueError("noise_sigma must be non-negative")
        if noise_sigma:
            rng = np.random.default_rng(noise_seed)
            noisy = rgb.astype(np.float32) + rng.normal(0.0, noise_sigma, rgb.shape)
            rgb = np.clip(np.rint(noisy), 0, 255).astype(np.uint8)
        else:
            rgb = rgb.copy()
        rgb.setflags(write=False)
        return CameraFrame(rgb, self.frame_counter, float(simulation_time_s), True,
                           self.camera_config_id)

