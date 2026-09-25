from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(matrix, dtype=np.float64)).reshape(-1, 2)


def canonical_square_yaw(yaw_rad: float) -> float:
    """Represent square orientation on [0, pi/2); quarter-turns are equivalent."""
    period = np.pi / 2.0
    return float((yaw_rad + period) % period)


def square_yaw_error(a_rad: float, b_rad: float) -> float:
    period = np.pi / 2.0
    delta = (float(a_rad) - float(b_rad) + period / 2.0) % period - period / 2.0
    return abs(float(delta))


@dataclass(frozen=True)
class PlanarCalibration:
    pixel_to_base_xy: np.ndarray
    base_xy_to_pixel: np.ndarray
    object_top_z_m: float
    calibration_id: str

    @classmethod
    def from_json(cls, path: str) -> "PlanarCalibration":
        import json
        from pathlib import Path

        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        h = np.asarray(obj["pixel_to_base_xy_homography"], dtype=np.float64)
        return cls(h, np.linalg.inv(h), float(obj["object_top_plane_z_m"]),
                   str(obj["calibration_id"]))

    def pixel_to_xy(self, u: float, v: float) -> tuple[float, float]:
        xy = transform_points(self.pixel_to_base_xy, np.array([[u, v]], dtype=float))[0]
        return float(xy[0]), float(xy[1])

    def xy_to_pixel(self, x: float, y: float) -> tuple[float, float]:
        uv = transform_points(self.base_xy_to_pixel, np.array([[x, y]], dtype=float))[0]
        return float(uv[0]), float(uv[1])

