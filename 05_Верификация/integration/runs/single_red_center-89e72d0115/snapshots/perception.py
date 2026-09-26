from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from camera.frame import CameraFrame
from .geometry import PlanarCalibration, canonical_square_yaw, transform_points
from .tracker import GreedyTrackManager
from .types import Detection, DetectionBatch


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict) or cfg.get("config_version") not in {"M4-v1.1", "M4-v1.2", "M4-v1.3"}:
        raise ValueError("Unsupported or missing perception configuration version")
    return cfg


def _poly_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


class RGBObjectPerception:
    """Perception receives CameraFrame only; it never imports simulator APIs."""

    def __init__(self, config: dict[str, Any], calibration: PlanarCalibration,
                 background_rgb: np.ndarray) -> None:
        self.config = config
        self.calibration = calibration
        self.background_rgb = np.asarray(background_rgb, dtype=np.uint8).copy()
        self.height, self.width = self.background_rgb.shape[:2]
        if self.background_rgb.shape != (self.height, self.width, 3):
            raise ValueError("Background reference must be HxWx3 RGB")
        roi = config["regions"]["input_xy_bounds_m"]
        x0, x1 = map(float, roi[0])
        y0, y1 = map(float, roi[1])
        self.input_poly_uv = transform_points(
            calibration.base_xy_to_pixel,
            np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float),
        )
        self.input_mask = _poly_mask((self.height, self.width), self.input_poly_uv)
        anchor = config["photometric_normalization"]["anchor_xy_bounds_m"]
        ax0, ax1 = map(float, anchor[0])
        ay0, ay1 = map(float, anchor[1])
        anchor_poly = transform_points(
            calibration.base_xy_to_pixel,
            np.array([[ax0, ay0], [ax1, ay0], [ax1, ay1], [ax0, ay1]], dtype=float),
        )
        self.anchor_mask = _poly_mask((self.height, self.width), anchor_poly) > 0
        if not np.any(self.anchor_mask):
            raise ValueError("Photometric anchor projects outside the image")
        self._background_lab = cv2.cvtColor(self.background_rgb, cv2.COLOR_RGB2LAB)
        self.tracker = GreedyTrackManager(
            max_distance_m=float(config["tracking"]["max_distance_m"]),
            max_missed_frames=int(config["tracking"]["max_missed_frames"]),
        )
        self._kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(config["candidate_detection"]["open_kernel_px"])
        )
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, tuple(config["candidate_detection"]["close_kernel_px"])
        )

    def detect(self, frame: CameraFrame, *, now_simulation_time_s: float | None = None) -> DetectionBatch:
        if not frame.valid or frame.rgb is None:
            return DetectionBatch("NO_FRAME", frame.invalid_reason or "INVALID_SENSOR_FRAME",
                                  frame.frame_id, frame.simulation_time_s, ())
        expected_camera_id = str(self.config["sensor_interface"]["camera_config_id"])
        if frame.camera_config_id != expected_camera_id:
            return DetectionBatch("INVALID", "CAMERA_CONFIG_MISMATCH", frame.frame_id,
                                  frame.simulation_time_s, ())
        if frame.frame_id < 0 or not np.isfinite(frame.simulation_time_s):
            return DetectionBatch("INVALID", "INVALID_FRAME_TIMESTAMP", frame.frame_id,
                                  frame.simulation_time_s, ())
        image = np.asarray(frame.rgb)
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            return DetectionBatch("INVALID", "MALFORMED_RGB_FRAME", frame.frame_id,
                                  frame.simulation_time_s, ())
        freshness_s = float(self.config["sensor_interface"]["maximum_age_s"])
        frame_age_s: float | None = None
        if now_simulation_time_s is not None:
            frame_age_s = frame.age_s(now_simulation_time_s)
            if frame_age_s < 0:
                return DetectionBatch("INVALID", "FUTURE_FRAME_TIMESTAMP", frame.frame_id,
                                      frame.simulation_time_s, ())
            if frame_age_s > freshness_s:
                return DetectionBatch("STALE", "FRAME_AGE_EXCEEDED", frame.frame_id,
                                      frame.simulation_time_s, ())

        normalized, gain = self._normalize_illumination(image)
        current_lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB).astype(np.float32)
        background_lab = self._background_lab.astype(np.float32)
        delta = np.linalg.norm(current_lab - background_lab, axis=2)
        threshold = float(self.config["candidate_detection"]["lab_delta_threshold"])
        binary = ((delta >= threshold) & (self.input_mask > 0)).astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._kernel_open)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel_close)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv = cv2.cvtColor(normalized, cv2.COLOR_RGB2HSV)
        masks = {name: self._class_mask(hsv, bounds)
                 for name, bounds in self.config["color_classifier"]["hsv_ranges"].items()}
        known_union = np.zeros((self.height, self.width), dtype=np.uint8)
        for class_mask in masks.values():
            known_union = cv2.bitwise_or(known_union, class_mask)
        masks["UNKNOWN"] = cv2.bitwise_and(binary, cv2.bitwise_not(known_union))
        found: list[Detection] = []
        cd = self.config["candidate_detection"]
        min_area = float(cd["min_area_px"])
        max_area = float(cd["max_area_px"])
        max_aspect = float(cd["max_aspect_ratio"])
        min_aspect = float(cd["min_aspect_ratio"])
        expected_area = float(cd["nominal_projected_object_area_px"])
        erode_size = int(self.config["color_classifier"]["interior_erode_kernel_px"])
        erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        hsv_cfg = self.config["color_classifier"]
        minimum_confidence = float(hsv_cfg["minimum_confidence"])
        minimum_margin = float(hsv_cfg["minimum_class_margin"])

        split_cfg = self.config.get("merged_candidate_split", {
            "enabled": True,
            "minimum_component_area_ratio": 0.40,
            "maximum_component_area_ratio": 1.40,
            "minimum_components": 2,
        })
        process_contours: list[tuple[np.ndarray, bool]] = []
        for parent in contours:
            parent_area = float(cv2.contourArea(parent))
            if (not split_cfg.get("enabled", False) or
                parent_area <= expected_area * float(cd["merged_candidate_area_ratio"])):
                process_contours.append((parent, False))
                continue
            parent_mask = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(parent_mask, [parent], -1, 255, thickness=cv2.FILLED)
            minimum_component_area = max(
                min_area,
                expected_area * float(split_cfg["minimum_component_area_ratio"]),
            )
            maximum_component_area = expected_area * float(split_cfg["maximum_component_area_ratio"])
            split_components: list[np.ndarray] = []
            for class_name, class_mask in masks.items():
                if class_name == "UNKNOWN":
                    continue
                within_parent = cv2.bitwise_and(class_mask, parent_mask)
                class_contours, _ = cv2.findContours(
                    within_parent, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for component in class_contours:
                    component_area = float(cv2.contourArea(component))
                    if not (minimum_component_area <= component_area <= maximum_component_area):
                        continue
                    component_rect = cv2.minAreaRect(component)
                    rw, rh = [max(float(v), 1e-6) for v in component_rect[1]]
                    component_aspect = max(rw, rh) / min(rw, rh)
                    if min_aspect <= component_aspect <= max_aspect:
                        split_components.append(component)
            if len(split_components) >= int(split_cfg["minimum_components"]):
                process_contours.extend((component, True) for component in split_components)
            else:
                process_contours.append((parent, False))

        for contour, split_from_merged_candidate in process_contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            rect = cv2.minAreaRect(contour)
            rw, rh = [max(float(v), 1e-6) for v in rect[1]]
            aspect = max(rw, rh) / min(rw, rh)
            x, y, w, h = cv2.boundingRect(contour)
            filled = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.drawContours(filled, [contour], -1, 255, thickness=cv2.FILLED)
            interior = cv2.erode(filled, erode)
            if cv2.countNonZero(interior) < 6:
                interior = filled
            pixel_count = max(1, cv2.countNonZero(interior))
            scores = {label: float(cv2.countNonZero(cv2.bitwise_and(mask, interior)) / pixel_count)
                      for label, mask in masks.items()}
            ordered_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            best_label, best_score = ordered_scores[0]
            second_score = ordered_scores[1][1] if len(ordered_scores) > 1 else 0.0
            margin = best_score - second_score
            confidence = float(np.clip(best_score * min(1.0, margin / max(minimum_margin, 1e-6)), 0, 1))
            known = best_score >= minimum_confidence and margin >= minimum_margin
            class_label = best_label if known else "UNKNOWN"
            reason: str | None = None if known else (
                "LOW_CLASS_SEPARATION" if best_score >= minimum_confidence else "NO_CONFIDENT_HSV_CLASS"
            )

            moments = cv2.moments(contour)
            if abs(moments["m00"]) > 1e-9:
                centroid_uv = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            else:
                centroid_uv = (float(rect[0][0]), float(rect[0][1]))
            rect_center_uv = (float(rect[0][0]), float(rect[0][1]))
            orientation_contour = contour
            color_core_centroid_uv = centroid_uv
            color_core_rect_center_uv = rect_center_uv
            if known:
                core_mask = cv2.bitwise_and(masks[best_label], filled)
                core_contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if core_contours:
                    core = max(core_contours, key=cv2.contourArea)
                    if cv2.contourArea(core) >= max(12.0, min_area * 0.15):
                        orientation_contour = core
                        core_moments = cv2.moments(core)
                        if abs(core_moments["m00"]) > 1e-9:
                            color_core_centroid_uv = (
                                core_moments["m10"] / core_moments["m00"],
                                core_moments["m01"] / core_moments["m00"],
                            )
                        core_rect = cv2.minAreaRect(core)
                        color_core_rect_center_uv = (float(core_rect[0][0]), float(core_rect[0][1]))
            center_method = self.config["pose_estimation"]["center_method"]
            center_options = {
                "contour_centroid": centroid_uv,
                "min_area_rect_center": rect_center_uv,
                "color_core_centroid": color_core_centroid_uv,
                "color_core_rect_center": color_core_rect_center_uv,
            }
            center_uv = center_options.get(center_method, centroid_uv)
            center_xy = self.calibration.pixel_to_xy(*center_uv)

            contour_uv = orientation_contour.reshape(-1, 2).astype(np.float32)
            contour_xy = transform_points(self.calibration.pixel_to_base_xy, contour_uv)
            rect_world = cv2.minAreaRect(contour_xy.astype(np.float32).reshape(-1, 1, 2))
            world_box = cv2.boxPoints(rect_world)
            edges = np.roll(world_box, -1, axis=0) - world_box
            lengths = np.linalg.norm(edges, axis=1)
            longest = edges[int(np.argmax(lengths))]
            yaw = canonical_square_yaw(float(np.arctan2(longest[1], longest[0])))

            status = "VALID" if known else "UNKNOWN"
            if aspect < min_aspect or aspect > max_aspect:
                status, reason = "OCCLUDED", "CANDIDATE_SHAPE_OUTSIDE_SQUARE_BOUNDS"
            elif area < expected_area * float(cd["occlusion_area_ratio"]):
                status, reason = "OCCLUDED", "VISIBLE_AREA_BELOW_OCCLUSION_LIMIT"
            elif area > expected_area * float(cd["merged_candidate_area_ratio"]):
                status, reason = "OCCLUDED", "OVERSIZED_OR_MERGED_CANDIDATE"

            rect_world_center = transform_points(
                self.calibration.pixel_to_base_xy, np.asarray([rect_center_uv], dtype=np.float64)
            )[0]
            centroid_world = transform_points(
                self.calibration.pixel_to_base_xy, np.asarray([centroid_uv], dtype=np.float64)
            )[0]
            found.append(Detection(
                track_id=None,
                class_label=class_label,
                xy_base_m=(float(center_xy[0]), float(center_xy[1])),
                yaw_base_rad=float(yaw),
                confidence=confidence,
                position_sigma_m=float(self.config["uncertainty"]["position_sigma_m"]),
                yaw_sigma_rad=float(self.config["uncertainty"]["yaw_sigma_rad"]),
                status=status,
                reason=reason,
                frame_id=frame.frame_id,
                simulation_time_s=frame.simulation_time_s,
                area_px=area,
                bbox_xywh_px=(int(x), int(y), int(w), int(h)),
                center_uv_px=(float(center_uv[0]), float(center_uv[1])),
                frame_age_s=frame_age_s,
                diagnostics={
                    "hsv_scores": scores,
                    "class_margin": margin,
                    "aspect_ratio": aspect,
                    "centroid_xy_base_m": [float(v) for v in centroid_world],
                    "min_area_rect_center_xy_base_m": [float(v) for v in rect_world_center],
                    "color_core_centroid_xy_base_m": [float(v) for v in transform_points(
                        self.calibration.pixel_to_base_xy,
                        np.asarray([color_core_centroid_uv], dtype=np.float64))[0]],
                    "color_core_rect_center_xy_base_m": [float(v) for v in transform_points(
                        self.calibration.pixel_to_base_xy,
                        np.asarray([color_core_rect_center_uv], dtype=np.float64))[0]],
                    "centroid_uv_px": [float(v) for v in centroid_uv],
                    "rect_center_uv_px": [float(v) for v in rect_center_uv],
                    "split_from_merged_candidate": bool(split_from_merged_candidate),
                },
            ))
        found.sort(key=lambda d: (d.xy_base_m[0], d.xy_base_m[1]))
        tracked = self.tracker.update(tuple(found))
        return DetectionBatch("OK" if found else "NO_CANDIDATES", None if found else "EMPTY_INPUT_ROI",
                              frame.frame_id, frame.simulation_time_s, tracked, binary,
                              masks, tuple(float(v) for v in gain))

    def _normalize_illumination(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        reference = np.median(self.background_rgb[self.anchor_mask].astype(np.float32), axis=0)
        observed = np.median(image[self.anchor_mask].astype(np.float32), axis=0)
        gain = reference / np.maximum(observed, 1.0)
        lower, upper = self.config["photometric_normalization"]["gain_clip"]
        gain = np.clip(gain, float(lower), float(upper))
        corrected = np.clip(np.rint(image.astype(np.float32) * gain[None, None, :]), 0, 255)
        return corrected.astype(np.uint8), gain

    @staticmethod
    def _class_mask(hsv: np.ndarray, bounds: dict[str, Any]) -> np.ndarray:
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        hue_mask = np.zeros(hue.shape, dtype=bool)
        for low, high in bounds["hue_intervals"]:
            hue_mask |= (hue >= int(low)) & (hue <= int(high))
        mask = hue_mask & (sat >= int(bounds["saturation_min"])) & (val >= int(bounds["value_min"]))
        return mask.astype(np.uint8) * 255
