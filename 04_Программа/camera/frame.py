from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraFrame:
    """Public sensor message; intentionally contains no simulator object metadata."""

    rgb: np.ndarray | None
    frame_id: int
    simulation_time_s: float
    valid: bool
    camera_config_id: str
    invalid_reason: str | None = None

    @property
    def resolution_px(self) -> tuple[int, int] | None:
        """Image size as (width, height), or None when the sensor has no payload."""
        if self.rgb is None or self.rgb.ndim != 3:
            return None
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])

    def age_s(self, now_simulation_time_s: float) -> float:
        """Signed age; negative values indicate a timestamp from the future."""
        return float(now_simulation_time_s) - float(self.simulation_time_s)

