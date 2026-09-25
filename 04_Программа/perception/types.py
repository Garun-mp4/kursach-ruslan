from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    track_id: int | None
    class_label: str
    xy_base_m: tuple[float, float]
    yaw_base_rad: float | None
    confidence: float
    position_sigma_m: float
    yaw_sigma_rad: float
    status: str
    reason: str | None
    frame_id: int
    simulation_time_s: float
    area_px: float
    bbox_xywh_px: tuple[int, int, int, int]
    center_uv_px: tuple[float, float] = (0.0, 0.0)
    frame_age_s: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionBatch:
    status: str
    reason: str | None
    frame_id: int
    simulation_time_s: float
    detections: tuple[Detection, ...]
    candidate_mask: Any = None
    color_masks: dict[str, Any] = field(default_factory=dict)
    photometric_gain_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
