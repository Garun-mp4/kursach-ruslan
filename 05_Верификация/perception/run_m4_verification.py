from __future__ import annotations

import copy
import ast
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROGRAM_DIR = ROOT / "04_Программа"
sys.path.insert(0, str(PROGRAM_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibration import run_calibration  # noqa: E402
from camera.frame import CameraFrame  # noqa: E402
from evaluator import CLASSES, MATRIX_LABELS, class_metrics, confusion_matrix, match_detections  # noqa: E402
from perception.detector import RGBObjectPerception, load_config  # noqa: E402
from perception.geometry import PlanarCalibration, square_yaw_error  # noqa: E402
from rig import GroundTruthObject, PerceptionRig  # noqa: E402


VERIFICATION = ROOT / "05_Верификация" / "perception"
CALIBRATION_DIR = VERIFICATION / "calibration"
TUNING_DIR = VERIFICATION / "tuning"
VALIDATION_DIR = VERIFICATION / "validation"
RESULTS_DIR = ROOT / "99_Рабочие_материалы" / "m4_perception"
CONFIG_PATH = PROGRAM_DIR / "perception" / "perception_config.yaml"
SSOT_PATH = ROOT / "02_Спецификация" / "параметры_системы.yaml"

KNOWN_RGBA = {
    "RED": (0.90, 0.06, 0.04, 1.0),
    "GREEN": (0.04, 0.78, 0.08, 1.0),
    "BLUE": (0.04, 0.12, 0.92, 1.0),
}
UNKNOWN_RGBA = {
    "gray": (0.35, 0.37, 0.39, 1.0),
    "pale_gray": (0.55, 0.57, 0.58, 1.0),
    "yellow": (0.90, 0.72, 0.04, 1.0),
    "magenta": (0.78, 0.06, 0.72, 1.0),
    "cyan": (0.04, 0.72, 0.78, 1.0),
}
LIGHTING_LEVELS = (0.75, 1.0, 1.25)
M4_VERSION = "M4-v1.3"
_CONFIG_VERSION_PATTERN = re.compile(r"^M(?P<milestone>\d+)-v(?P<major>\d+)\.(?P<minor>\d+)$")
TUNING_SEEDS = tuple(range(20260401, 20260425))
# Earlier validation seeds were inspected while debugging the perception pipeline and
# are development data. This frozen final set is disjoint from tuning and all earlier
# validation samples; do not alter it after the acceptance run starts.
VALIDATION_SEEDS = tuple(range(20265000, 20265068))


def load_ssot() -> dict[str, Any]:
    return yaml.safe_load(SSOT_PATH.read_text(encoding="utf-8-sig"))


def p_value(ssot: dict[str, Any], key: str) -> Any:
    return ssot["parameters"][key]["value"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rgb_variant(label: str, shade: float) -> tuple[float, float, float, float]:
    rgba = np.asarray(KNOWN_RGBA[label], dtype=float)
    rgb = np.clip(rgba[:3] * float(shade), 0.0, 0.98)
    return float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0


def _dataset_objects(ssot: dict[str, Any], seed: int, *, tuning: bool) -> tuple[list[GroundTruthObject], float, float]:
    rng = np.random.default_rng(seed)
    p = lambda key: p_value(ssot, key)
    if not tuning and VALIDATION_SEEDS[0] <= seed < VALIDATION_SEEDS[0] + 8:
        edge_positions = [(-0.112, -0.25), (0.112, -0.25), (0.0, -0.322), (0.0, -0.178),
                          (-0.105, -0.318), (0.105, -0.318), (-0.105, -0.182), (0.105, -0.182)]
        labels = ("RED", "GREEN", "BLUE", "RED", "GREEN", "BLUE", "GREEN", "BLUE")
        yaws = (0.0, math.pi/4, math.pi/2, 0.2, 0.5, 1.1, 0.9, 0.3)
        idx = seed - VALIDATION_SEEDS[0]
        x, y = edge_positions[idx]
        label = labels[idx]
        obj = GroundTruthObject(f"seed-{seed}-edge", label, x, y, yaws[idx], KNOWN_RGBA[label])
        return [obj], LIGHTING_LEVELS[idx % 3], 1.0
    xs = list(p("cell.input_slot_x_offsets_m"))
    ys = [float(p("cell.input_center_xy_m")[1]) + float(v) for v in p("cell.input_row_y_offsets_m")]
    positions = [(float(x), float(y)) for y in ys for x in xs]
    rng.shuffle(positions)
    classes = ["RED", "GREEN", "BLUE", "RED", "GREEN", "BLUE"]
    rng.shuffle(classes)
    variants = list(UNKNOWN_RGBA.keys())
    objects: list[GroundTruthObject] = []
    for i, (label, (x0, y0)) in enumerate(zip(classes, positions)):
        x = x0 + float(rng.uniform(-0.004, 0.004))
        y = y0 + float(rng.uniform(-0.004, 0.004))
        # Dedicated boundary samples remain inside the assigned input ROI.
        if not tuning and i == 0 and seed % 10 == 0:
            x = -0.106 if (seed // 10) % 2 == 0 else 0.106
        yaw = float(rng.uniform(0.0, math.pi / 2.0))
        if i < 3 and seed % 5 == 0:
            yaw = (0.0, math.pi / 4.0, math.pi / 2.0)[i]
        shade = float(rng.choice([0.72, 0.82, 0.93, 1.0, 1.10, 1.20, 1.28]))
        rgba = _rgb_variant(label, shade)
        true_label = label
        # Tuning and validation both contain explicit unknown colors; their seeds are disjoint.
        replace_unknown = ((seed % 4 == 0 and i == 5) or
                           (tuning and seed % 6 == 0 and i == 4))
        if replace_unknown:
            unknown_name = variants[(seed + i) % len(variants)]
            rgba = UNKNOWN_RGBA[unknown_name]
            true_label = "UNKNOWN"
        objects.append(GroundTruthObject(
            evaluator_key=f"seed-{seed}-item-{i}", class_label=true_label,
            x_m=x, y_m=y, yaw_rad=yaw, rgba=rgba,
        ))
    if tuning:
        lighting = float(rng.choice([0.80, 0.90, 1.0, 1.12, 1.20]))
        noise = float(rng.choice([0.0, 1.0, 2.0, 3.0]))
    else:
        lighting = LIGHTING_LEVELS[(seed - VALIDATION_SEEDS[0]) % len(LIGHTING_LEVELS)]
        noise = (0.0, 1.5, 3.0)[(seed // 3) % 3]
    return objects, lighting, noise


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f"Could not write image {path}")


def _build_sample_cache(ssot: dict[str, Any], *, tuning: bool) -> tuple[np.ndarray, list[dict[str, Any]]]:
    seeds = TUNING_SEEDS if tuning else VALIDATION_SEEDS
    rig = PerceptionRig(ROOT, ssot)
    background_frame = rig.render_empty()
    if not background_frame.valid or background_frame.rgb is None:
        rig.close()
        raise RuntimeError("Empty scene background render failed")
    background = background_frame.rgb.copy()
    records: list[dict[str, Any]] = []
    try:
        for index, seed in enumerate(seeds):
            objects, light, noise = _dataset_objects(ssot, seed, tuning=tuning)
            rig.set_objects(objects, lighting_scale=light)
            frame = rig.capture(sim_time_s=float(index) * 0.02, noise_sigma=noise,
                                noise_seed=seed ^ 0x5A5A)
            if not frame.valid or frame.rgb is None:
                raise RuntimeError(f"Invalid {('tuning' if tuning else 'validation')} render at seed {seed}")
            records.append({
                "seed": seed, "frame": frame, "truth": objects,
                "lighting_scale": light, "noise_sigma": noise,
                "dataset": "tuning" if tuning else "validation",
            })
    finally:
        rig.close()
    return background, records


def _metrics_for_dataset(records: list[dict[str, Any]], detector: RGBObjectPerception,
                         *, max_match_distance_m: float = 0.025) -> dict[str, Any]:
    tp = fp = fn = 0
    color_rows: list[tuple[str, str]] = []
    localization: list[dict[str, Any]] = []
    candidate_localization: list[dict[str, Any]] = []
    yaw_errors: list[float] = []
    candidate_yaw_errors: list[float] = []
    centroid_errors: list[float] = []
    rect_center_errors: list[float] = []
    frame_rows: list[dict[str, Any]] = []
    class_error_scenes: list[dict[str, Any]] = []
    unmatched_truth: list[dict[str, Any]] = []
    false_positive_details: list[dict[str, Any]] = []
    truth_contrast: dict[tuple[int, int], float] = {}
    truth_coverage: dict[tuple[int, int], float] = {}
    for item in records:
        batch = detector.detect(item["frame"])
        detections = list(batch.detections)
        normalized, gain = detector._normalize_illumination(item["frame"].rgb)
        normalized_background = np.clip(
            detector.background_rgb.astype(np.float32) * gain[None, None, :], 0, 255
        ).astype(np.uint8)
        normalized_lab = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(normalized_background, cv2.COLOR_RGB2LAB).astype(np.float32)
        for gi, gt in enumerate(item["truth"]):
            u, v = detector.calibration.xy_to_pixel(gt.x_m, gt.y_m)
            ix, iy = int(round(u)), int(round(v))
            crop = (slice(max(0, iy - 3), min(normalized.shape[0], iy + 4)),
                    slice(max(0, ix - 3), min(normalized.shape[1], ix + 4)))
            delta = np.linalg.norm(normalized_lab[crop] - background_lab[crop], axis=2)
            truth_contrast[(int(item["seed"]), gi)] = float(np.median(delta)) if delta.size else 0.0
            size_x, size_y = map(float, detector.config["candidate_detection"]["object_size_xy_m"])
            local_corners = np.asarray([
                [-size_x/2, -size_y/2], [size_x/2, -size_y/2],
                [size_x/2, size_y/2], [-size_x/2, size_y/2],
            ], dtype=float)
            c, s = math.cos(gt.yaw_rad), math.sin(gt.yaw_rad)
            rotation = np.asarray([[c, -s], [s, c]], dtype=float)
            footprint_base = local_corners @ rotation.T + np.asarray([gt.x_m, gt.y_m])
            footprint_uv = cv2.perspectiveTransform(
                footprint_base.astype(np.float64).reshape(-1, 1, 2),
                detector.calibration.base_xy_to_pixel.astype(np.float64),
            ).reshape(-1, 2)
            x0 = max(0, int(math.floor(float(np.min(footprint_uv[:, 0])))))
            x1 = min(normalized.shape[1], int(math.ceil(float(np.max(footprint_uv[:, 0])))) + 1)
            y0 = max(0, int(math.floor(float(np.min(footprint_uv[:, 1])))))
            y1 = min(normalized.shape[0], int(math.ceil(float(np.max(footprint_uv[:, 1])))) + 1)
            if x1 <= x0 or y1 <= y0:
                truth_coverage[(int(item["seed"]), gi)] = 0.0
            else:
                footprint_local = np.rint(footprint_uv - np.asarray([x0, y0])).astype(np.int32)
                footprint_mask = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
                cv2.fillPoly(footprint_mask, [footprint_local], 255)
                local_delta = np.linalg.norm(normalized_lab[y0:y1, x0:x1] -
                                             background_lab[y0:y1, x0:x1], axis=2)
                pixels = footprint_mask > 0
                truth_coverage[(int(item["seed"]), gi)] = (
                    float(np.count_nonzero(local_delta[pixels] >=
                                           float(detector.config["candidate_detection"]["lab_delta_threshold"])) /
                          max(1, np.count_nonzero(pixels)))
                )
        matches, missed_truth, false_detections = match_detections(
            detections, item["truth"], max_match_distance_m
        )
        tp += len(matches)
        fn += len(missed_truth)
        fp += len(false_detections)
        for di in sorted(false_detections):
            det = detections[di]
            false_positive_details.append({
                "seed": int(item["seed"]), "detection_index": int(di),
                "class_label": det.class_label, "status": det.status, "reason": det.reason,
                "x_base_m": det.xy_base_m[0], "y_base_m": det.xy_base_m[1],
                "confidence": det.confidence, "bbox_xywh_px": det.bbox_xywh_px,
            })
        color_rows.extend((item["truth"][gi].class_label, "NO_DETECTION") for gi in missed_truth)
        for gi in sorted(missed_truth):
            gt = item["truth"][gi]
            nearby_rejections = [
                (float(np.linalg.norm(np.asarray(det.xy_base_m) - [gt.x_m, gt.y_m])), det)
                for det in detections if det.status != "VALID"
            ]
            nearby_rejections = [pair for pair in nearby_rejections
                                 if pair[0] <= max_match_distance_m]
            nearest_rejection = min(nearby_rejections, key=lambda pair: pair[0]) if nearby_rejections else None
            unmatched_truth.append({"seed": item["seed"], "truth_index": gi,
                                    "truth_key": gt.evaluator_key, "true_class": gt.class_label,
                                    "local_contrast_delta_e76": truth_contrast[(int(item["seed"]), gi)],
                                    "foreground_coverage_ratio": truth_coverage[(int(item["seed"]), gi)],
                                    "nearby_safe_rejection_distance_m": nearest_rejection[0] if nearest_rejection else None,
                                    "nearby_safe_rejection_status": nearest_rejection[1].status if nearest_rejection else None,
                                    "nearby_safe_rejection_reason": nearest_rejection[1].reason if nearest_rejection else None,
                                    "x_true_m": gt.x_m, "y_true_m": gt.y_m,
                                    "yaw_true_rad": gt.yaw_rad,
                                    "lighting_scale": item["lighting_scale"],
                                    "noise_sigma": item["noise_sigma"]})
        errors_this_frame: list[float] = []
        for match in matches:
            gt = item["truth"][match.gt_index]
            det = detections[match.detection_index]
            color_rows.append((gt.class_label, det.class_label))
            dx = det.xy_base_m[0] - gt.x_m
            dy = det.xy_base_m[1] - gt.y_m
            pos = math.hypot(dx, dy)
            errors_this_frame.append(pos)
            diagnostic = det.diagnostics
            centroid_xy = diagnostic["centroid_xy_base_m"]
            rect_xy = diagnostic["min_area_rect_center_xy_base_m"]
            centroid_errors.append(math.hypot(centroid_xy[0] - gt.x_m, centroid_xy[1] - gt.y_m))
            rect_center_errors.append(math.hypot(rect_xy[0] - gt.x_m, rect_xy[1] - gt.y_m))
            yaw_error = square_yaw_error(float(det.yaw_base_rad or 0.0), gt.yaw_rad)
            candidate_yaw_errors.append(yaw_error)
            localization_row = {
                "seed": item["seed"], "truth_key": gt.evaluator_key,
                "true_class": gt.class_label, "predicted_class": det.class_label,
                "local_contrast_delta_e76": truth_contrast[(int(item["seed"]), match.gt_index)],
                "foreground_coverage_ratio": truth_coverage[(int(item["seed"]), match.gt_index)],
                "x_true_m": gt.x_m, "y_true_m": gt.y_m,
                "x_est_m": det.xy_base_m[0], "y_est_m": det.xy_base_m[1],
                "dx_m": dx, "dy_m": dy, "planar_error_m": pos,
                "yaw_true_mod_pi2_rad": gt.yaw_rad % (math.pi / 2.0),
                "yaw_est_mod_pi2_rad": float(det.yaw_base_rad or 0.0),
                "yaw_error_mod_pi2_rad": yaw_error,
                "centroid_planar_error_m": centroid_errors[-1],
                "min_rect_center_planar_error_m": rect_center_errors[-1],
                "color_core_centroid_planar_error_m": _planar_from_xy(
                    diagnostic["color_core_centroid_xy_base_m"], gt.x_m, gt.y_m
                ),
                "color_core_rect_center_planar_error_m": _planar_from_xy(
                    diagnostic["color_core_rect_center_xy_base_m"], gt.x_m, gt.y_m
                ),
                "status": det.status, "reason": det.reason,
                "confidence": det.confidence, "area_px": det.area_px,
            }
            candidate_localization.append(localization_row)
            if det.status == "VALID":
                localization.append(localization_row)
                yaw_errors.append(yaw_error)
            if gt.class_label != det.class_label:
                class_error_scenes.append({"seed": item["seed"], "gt": gt, "detection": det,
                                           "frame": item["frame"], "batch": batch})
        frame_rows.append({
            "seed": item["seed"], "dataset": item["dataset"],
            "lighting_scale": item["lighting_scale"], "noise_sigma": item["noise_sigma"],
            "gt_count": len(item["truth"]), "candidate_count": len(detections),
            "matched": len(matches), "false_positive": len(false_detections),
            "false_negative": len(missed_truth), "batch_status": batch.status,
            "max_xy_error_m": max(errors_this_frame) if errors_this_frame else None,
        })
        item["batch"] = batch
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    matrix = confusion_matrix(color_rows)
    counts = Counter(label for label, _ in color_rows)
    coverage_threshold = float(detector.config["candidate_detection"]["occlusion_area_ratio"])
    supported_localization = [row for row in candidate_localization
                              if float(row["foreground_coverage_ratio"]) >= coverage_threshold]
    supported_unmatched = [row for row in unmatched_truth
                           if float(row["foreground_coverage_ratio"]) >= coverage_threshold]
    supported_tp = len(supported_localization)
    supported_fn = len(supported_unmatched)
    safely_rejected_misses = sum(
        1 for row in supported_unmatched if row["nearby_safe_rejection_reason"] in {
            "OVERSIZED_OR_MERGED_CANDIDATE", "VISIBLE_AREA_BELOW_OCCLUSION_LIMIT",
            "CANDIDATE_SHAPE_OUTSIDE_SQUARE_BOUNDS", "LOW_CLASS_SEPARATION",
            "NO_CONFIDENT_HSV_CLASS",
        }
    )
    unaccounted_supported_fn = supported_fn - safely_rejected_misses
    supported_fp = fp  # conservative: every unmatched detection counts against supported precision
    supported_precision = supported_tp / (supported_tp + supported_fp) if supported_tp + supported_fp else 0.0
    supported_recall = supported_tp / (supported_tp + supported_fn) if supported_tp + supported_fn else 0.0
    supported_valid = sum(1 for row in supported_localization if row["status"] == "VALID")
    unmatched_valid_predictions = sum(1 for row in false_positive_details if row["status"] == "VALID")
    rejected_false_candidates = sum(1 for row in false_positive_details if row["status"] != "VALID")
    return {
        "object_detection": {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
                             "recall": recall, "f1": 2*precision*recall/(precision+recall)
                             if precision + recall else 0.0},
        "color_confusion_matrix_counts": matrix,
        "color_metrics": class_metrics(matrix),
        "color_sample_counts": dict(counts),
        "localization": localization,
        "candidate_localization": candidate_localization,
        "detection_status_counts": dict(Counter(
            detection.status
            for item in records for detection in item["batch"].detections
        )),
        "xy_planar_error_summary_m": _summarize([r["planar_error_m"] for r in localization]),
        "x_signed_error_summary_m": _summarize([r["dx_m"] for r in localization]),
        "y_signed_error_summary_m": _summarize([r["dy_m"] for r in localization]),
        "yaw_error_summary_rad": _summarize(yaw_errors),
        "candidate_yaw_error_summary_rad": _summarize(candidate_yaw_errors),
        "center_estimator_comparison": {
            "contour_centroid_planar_error_m": _summarize(centroid_errors),
            "min_area_rectangle_center_planar_error_m": _summarize(rect_center_errors),
            "color_core_centroid_planar_error_m": _summarize(
                [r["color_core_centroid_planar_error_m"] for r in localization]
            ),
            "color_core_rectangle_center_planar_error_m": _summarize(
                [r["color_core_rect_center_planar_error_m"] for r in localization]
            ),
        },
        "frame_rows": frame_rows,
        "class_error_scenes": class_error_scenes,
        "unmatched_truth": unmatched_truth,
        "false_positive_details": false_positive_details,
        "operating_domain": {
            "definition": "At least the configured visibility fraction of the projected 28x28 mm object-top footprint has per-pixel Lab Euclidean contrast ΔE*ab above the frozen candidate threshold; coverage is estimated only by the evaluator from ground truth and RGB frame.",
            "foreground_contrast_threshold_delta_e76": float(detector.config["candidate_detection"]["lab_delta_threshold"]),
            "minimum_foreground_coverage_ratio": coverage_threshold,
            "supported_object_count": supported_tp + supported_fn,
            "supported_matched_count": supported_tp,
            "supported_false_negative_count": supported_fn,
            "supported_misses_with_nearby_safe_rejection": safely_rejected_misses,
            "unaccounted_supported_false_negative_count": unaccounted_supported_fn,
            "supported_false_positive_count": supported_fp,
            "supported_precision": supported_precision,
            "supported_recall": supported_recall,
            "supported_valid_count": supported_valid,
            "supported_rejected_count": sum(
                1 for row in supported_localization if row["status"] != "VALID"
            ),
            "unmatched_valid_prediction_count": unmatched_valid_predictions,
            "unmatched_rejected_candidate_count": rejected_false_candidates,
            "out_of_domain_object_count": sum(
                1 for value in truth_coverage.values() if value < coverage_threshold
            ),
            "contrast_by_object": [
                {"seed": seed, "truth_index": gi, "median_local_contrast_delta_e76": truth_contrast[(seed, gi)],
                 "foreground_coverage_ratio": value, "in_declared_domain": value >= coverage_threshold}
                for (seed, gi), value in sorted(truth_coverage.items())
            ],
        },
    }


def _summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    a = np.asarray(values, dtype=float)
    return {"n": int(len(a)), "mean": float(np.mean(a)), "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)), "max": float(np.max(a))}


def _planar_from_xy(point: list[float] | tuple[float, float], x: float, y: float) -> float:
    return float(math.hypot(float(point[0]) - x, float(point[1]) - y))


def _tuning_score(metrics: dict[str, Any]) -> float:
    detection_f1 = float(metrics["object_detection"]["f1"])
    class_recalls = [float(metrics["color_metrics"][cls]["recall"])
                     for cls in CLASSES if int(metrics["color_metrics"][cls]["support"]) > 0]
    macro_recall = float(np.mean(class_recalls)) if class_recalls else 0.0
    return 0.5 * detection_f1 + 0.5 * macro_recall


def _fit_tuning_hsv(records: list[dict[str, Any]], calibration: PlanarCalibration,
                    base_cfg: dict[str, Any]) -> dict[str, Any]:
    """Use only labeled tuning samples to estimate circular hue intervals and S/V floors."""
    p = lambda key: base_cfg["camera"][key]
    K = np.asarray(p("intrinsics_fx_fy_cx_cy_px"), dtype=float)
    cam_z = float(p("position_world_m")[2])
    z_top = float(base_cfg["calibration"]["object_top_plane_z_m"])
    hue_samples: dict[str, list[float]] = {k: [] for k in ("RED", "GREEN", "BLUE")}
    sat_samples: dict[str, list[float]] = {k: [] for k in hue_samples}
    val_samples: dict[str, list[float]] = {k: [] for k in hue_samples}
    for item in records:
        hsv = cv2.cvtColor(item["frame"].rgb, cv2.COLOR_RGB2HSV)
        for gt in item["truth"]:
            if gt.class_label == "UNKNOWN":
                continue
            u = K[0] * gt.x_m / (cam_z - z_top) + K[2]
            v = -K[1] * gt.y_m / (cam_z - z_top) + K[3]
            ix, iy = int(round(u)), int(round(v))
            patch = hsv[max(0, iy-3):iy+4, max(0, ix-3):ix+4]
            if patch.size == 0:
                continue
            hue_samples[gt.class_label].extend(patch[:, :, 0].astype(float).ravel().tolist())
            sat_samples[gt.class_label].extend(patch[:, :, 1].astype(float).ravel().tolist())
            val_samples[gt.class_label].extend(patch[:, :, 2].astype(float).ravel().tolist())

    ranges: dict[str, Any] = {}
    for label in hue_samples:
        hue = np.asarray(hue_samples[label], dtype=float)
        angle = hue * (2.0 * math.pi / 180.0)
        mean_angle = math.atan2(float(np.mean(np.sin(angle))), float(np.mean(np.cos(angle))))
        mean_hue = (mean_angle % (2.0 * math.pi)) * 180.0 / (2.0 * math.pi)
        signed_delta = (hue - mean_hue + 90.0) % 180.0 - 90.0
        lower = float(np.percentile(signed_delta, 1.0)) - 4.0
        upper = float(np.percentile(signed_delta, 99.0)) + 4.0
        ranges[label] = {
            "hue_intervals": _cyclic_hue_intervals(mean_hue + lower, mean_hue + upper),
            "saturation_min": max(35, int(math.floor(np.percentile(sat_samples[label], 1.0) - 20))),
            "value_min": max(12, int(math.floor(np.percentile(val_samples[label], 1.0) - 18))),
        }
    output = copy.deepcopy(base_cfg)
    output["color_classifier"]["hsv_ranges"] = ranges
    return output


def _cyclic_hue_intervals(low_raw: float, high_raw: float) -> list[list[int]]:
    width = min(179.0, high_raw - low_raw)
    center = ((low_raw + high_raw) / 2.0) % 180.0
    low = center - width / 2.0
    high = center + width / 2.0
    if low < 0:
        return [[0, int(math.ceil(high))], [int(math.floor(low + 180)), 179]]
    if high >= 180:
        return [[0, int(math.ceil(high - 180))], [int(math.floor(low)), 179]]
    return [[int(math.floor(low)), int(math.ceil(high))]]


def _tune_config(background: np.ndarray, tuning_records: list[dict[str, Any]],
                  calibration: PlanarCalibration, initial_cfg: dict[str, Any],
                  tuning_empty_records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    base = _fit_tuning_hsv(tuning_records, calibration, initial_cfg)
    center_comparison = {"contour_centroid": [], "min_area_rectangle_center": [],
                         "color_core_centroid": [], "color_core_rectangle_center": []}
    provisional = copy.deepcopy(base)
    provisional["pose_estimation"]["center_method"] = "contour_centroid"
    probe = RGBObjectPerception(provisional, calibration, background)
    probe_metrics = _metrics_for_dataset(tuning_records, probe)
    for row in probe_metrics["localization"]:
        center_comparison["contour_centroid"].append(row["centroid_planar_error_m"])
        center_comparison["min_area_rectangle_center"].append(row["min_rect_center_planar_error_m"])
        center_comparison["color_core_centroid"].append(row["color_core_centroid_planar_error_m"])
        center_comparison["color_core_rectangle_center"].append(row["color_core_rect_center_planar_error_m"])
    center_p95 = {name: float(_summarize(values)["p95"] or 0.0)
                  for name, values in center_comparison.items()}
    selected_estimator = min(center_p95, key=center_p95.get)
    chosen_center = {"color_core_rectangle_center": "color_core_rect_center"}.get(
        selected_estimator, selected_estimator
    )
    base["pose_estimation"]["center_method"] = chosen_center

    sweep_rows: list[dict[str, Any]] = []
    for lab_threshold in (16.0, 22.0, 28.0, 34.0):
        for min_confidence in (0.36, 0.48, 0.60):
            cfg = copy.deepcopy(base)
            cfg["candidate_detection"]["lab_delta_threshold"] = lab_threshold
            cfg["color_classifier"]["minimum_confidence"] = min_confidence
            detector = RGBObjectPerception(cfg, calibration, background)
            metrics = _metrics_for_dataset(tuning_records, detector)
            score = _tuning_score(metrics)
            sweep_rows.append({
                "lab_delta_threshold": lab_threshold,
                "minimum_confidence": min_confidence,
                "score": score,
                "detection_f1": metrics["object_detection"]["f1"],
                "color_recall": metrics["color_metrics"],
                "xy_p95_m": metrics["xy_planar_error_summary_m"]["p95"],
            })
    chosen = max(sweep_rows, key=lambda row: (row["score"], row["detection_f1"], -row["lab_delta_threshold"], -row["minimum_confidence"]))
    final = copy.deepcopy(base)
    final["candidate_detection"]["lab_delta_threshold"] = chosen["lab_delta_threshold"]
    final["color_classifier"]["minimum_confidence"] = chosen["minimum_confidence"]
    final["config_version"] = M4_VERSION

    geometry_sweep: list[dict[str, Any]] = []
    for kernel_size in (2, 3):
        for min_area in (8.0, 20.0, 35.0, 55.0):
            candidate = copy.deepcopy(final)
            candidate["candidate_detection"]["open_kernel_px"] = [kernel_size, kernel_size]
            candidate["candidate_detection"]["min_area_px"] = min_area
            candidate_metrics = _metrics_for_dataset(
                tuning_records, RGBObjectPerception(candidate, calibration, background)
            )
            empty_detector = RGBObjectPerception(candidate, calibration, background)
            empty_false_candidates = sum(
                len(empty_detector.detect(row["frame"]).detections)
                for row in tuning_empty_records
            )
            geometry_sweep.append({
                "open_kernel_px": kernel_size, "min_area_px": min_area,
                "precision": candidate_metrics["object_detection"]["precision"],
                "recall": candidate_metrics["object_detection"]["recall"],
                "f1": candidate_metrics["object_detection"]["f1"],
                "unknown_recall": candidate_metrics["color_metrics"]["UNKNOWN"]["recall"],
                "xy_p95_m": candidate_metrics["xy_planar_error_summary_m"]["p95"],
                "empty_scene_frames": len(tuning_empty_records),
                "empty_scene_false_candidates": int(empty_false_candidates),
            })
    selected_geometry = max(geometry_sweep, key=lambda row: (
        row["empty_scene_false_candidates"] == 0,
        -row["empty_scene_false_candidates"],
        row["f1"], row["precision"], row["unknown_recall"],
        -(row["xy_p95_m"] or float("inf")), row["min_area_px"],
        row["open_kernel_px"] == 3,
    ))
    final["candidate_detection"]["open_kernel_px"] = [selected_geometry["open_kernel_px"]] * 2
    final["candidate_detection"]["min_area_px"] = selected_geometry["min_area_px"]
    report = {
        "tuning_seed_ids": [int(item["seed"]) for item in tuning_records],
        "tuning_frame_count": len(tuning_records),
        "tuning_object_count": int(sum(len(item["truth"]) for item in tuning_records)),
        "tuning_split_hash": hashlib.sha256(",".join(map(str, TUNING_SEEDS)).encode()).hexdigest(),
        "tuning_empty_scene_seed_ids": [int(item["seed"]) for item in tuning_empty_records],
        "tuning_empty_scene_seed_hash": hashlib.sha256(
            ",".join(str(item["seed"]) for item in tuning_empty_records).encode()
        ).hexdigest(),
        "hsv_ranges_fit_from_tuning": final["color_classifier"]["hsv_ranges"],
        "center_estimator_p95_comparison_m": {
            "contour_centroid": _summarize(center_comparison["contour_centroid"]),
            "min_area_rectangle_center": _summarize(center_comparison["min_area_rectangle_center"]),
            "color_core_centroid": _summarize(center_comparison["color_core_centroid"]),
            "color_core_rectangle_center": _summarize(center_comparison["color_core_rectangle_center"]),
        },
        "selected_center_method": chosen_center,
        "grid_search": sweep_rows,
        "selected_parameters": {
            "lab_delta_threshold": final["candidate_detection"]["lab_delta_threshold"],
            "minimum_confidence": final["color_classifier"]["minimum_confidence"],
            "open_kernel_px": final["candidate_detection"]["open_kernel_px"],
            "min_area_px": final["candidate_detection"]["min_area_px"],
        },
        "candidate_geometry_sweep": geometry_sweep,
        "selected_candidate_geometry": selected_geometry,
    }
    return final, report


def _annotate(rgb: np.ndarray, batch: Any, truth: list[GroundTruthObject], calibration: PlanarCalibration) -> np.ndarray:
    image = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    for gt in truth:
        u, v = calibration.xy_to_pixel(gt.x_m, gt.y_m)
        cv2.drawMarker(image, (int(round(u)), int(round(v))), (255, 255, 255),
                       cv2.MARKER_CROSS, 11, 1)
    for det in batch.detections:
        x, y, w, h = det.bbox_xywh_px
        color = (20, 210, 20) if det.status == "VALID" else (20, 170, 255)
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 1)
        cv2.putText(image, f"{det.class_label}/{det.track_id} {det.confidence:.2f}",
                    (x, max(12, y-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        u, v = calibration.xy_to_pixel(*det.xy_base_m)
        cv2.circle(image, (int(round(u)), int(round(v))), 2, color, -1)
    return image


def _save_dataset_outputs(records: list[dict[str, Any]], metrics: dict[str, Any],
                          dataset_dir: Path, calibration: PlanarCalibration) -> None:
    for name in ("raw_rgb", "candidate_masks", "color_masks", "overlays"):
        (dataset_dir / name).mkdir(parents=True, exist_ok=True)
    detection_json = []
    detection_csv: list[dict[str, Any]] = []
    frame_by_seed = {item["seed"]: item for item in records}
    for item in records:
        frame = item["frame"]
        batch = item["batch"]
        key = f"{item['seed']}"
        _save_rgb(dataset_dir / "raw_rgb" / f"frame_{key}.png", frame.rgb)
        cv2.imwrite(str(dataset_dir / "candidate_masks" / f"candidate_{key}.png"), batch.candidate_mask)
        for cls, mask in batch.color_masks.items():
            cv2.imwrite(str(dataset_dir / "color_masks" / f"{cls.lower()}_{key}.png"), mask)
        overlay = _annotate(frame.rgb, batch, item["truth"], calibration)
        cv2.imwrite(str(dataset_dir / "overlays" / f"overlay_{key}.png"), overlay)
        detection_json.append({
            "seed": item["seed"], "frame_id": frame.frame_id,
            "simulation_time_s": frame.simulation_time_s,
            "valid": frame.valid, "camera_config_id": frame.camera_config_id,
            "photometric_gain_rgb": batch.photometric_gain_rgb,
            "batch_status": batch.status, "reason": batch.reason,
            "detections": [asdict(detection) for detection in batch.detections],
        })
        for detection in batch.detections:
            detection_csv.append({
                "seed": item["seed"], "frame_id": frame.frame_id,
                "track_id": detection.track_id, "class_label": detection.class_label,
                "x_base_m": detection.xy_base_m[0], "y_base_m": detection.xy_base_m[1],
                "yaw_base_rad_mod_pi2": detection.yaw_base_rad,
                "confidence": detection.confidence,
                "position_sigma_m": detection.position_sigma_m,
                "yaw_sigma_rad": detection.yaw_sigma_rad,
                "status": detection.status, "reason": detection.reason,
                "area_px": detection.area_px,
                "bbox_x_px": detection.bbox_xywh_px[0], "bbox_y_px": detection.bbox_xywh_px[1],
                "bbox_w_px": detection.bbox_xywh_px[2], "bbox_h_px": detection.bbox_xywh_px[3],
            })
    (dataset_dir / "detections.json").write_text(
        json.dumps(detection_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    evaluator_manifest = [{
        "seed": item["seed"],
        "lighting_scale": item["lighting_scale"],
        "noise_sigma": item["noise_sigma"],
        "truth_objects": [asdict(gt) for gt in item["truth"]],
    } for item in records]
    (dataset_dir / "evaluator_manifest.json").write_text(
        json.dumps(evaluator_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (dataset_dir / "detections.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detection_csv[0]) if detection_csv else ["seed"])
        writer.writeheader()
        writer.writerows(detection_csv)
    with (dataset_dir / "frame_metrics.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        rows = metrics["frame_rows"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["seed"])
        writer.writeheader()
        writer.writerows(rows)
    with (dataset_dir / "localization_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        rows = metrics["localization"]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["seed"])
        writer.writeheader()
        writer.writerows(rows)


def _write_confusion_csv(path: Path, matrix: dict[str, dict[str, int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["actual\\predicted", *CLASSES, "NO_DETECTION"])
        for actual in CLASSES:
            writer.writerow([actual, *[matrix.get(actual, {}).get(pred, 0) for pred in MATRIX_LABELS]])


def _save_worst_cases(metrics: dict[str, Any], records: list[dict[str, Any]],
                      out_dir: Path, calibration: PlanarCalibration) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = sorted(metrics["localization"], key=lambda row: row["planar_error_m"], reverse=True)
    choices: list[tuple[str, dict[str, Any]]] = []
    for row in errors[:3]:
        choices.append(("largest_xy_error", row))
    rejected_errors = sorted(
        (row for row in metrics["candidate_localization"] if row["status"] != "VALID"),
        key=lambda row: row["planar_error_m"], reverse=True,
    )
    for row in rejected_errors[:3]:
        choices.append(("rejected_candidate_large_residual", row))
    if metrics["class_error_scenes"]:
        for err in metrics["class_error_scenes"][:3]:
            choices.append(("classification_error", {
                "seed": err["seed"], "truth_key": err["gt"].evaluator_key,
                "true_class": err["gt"].class_label,
                "predicted_class": err["detection"].class_label,
                "planar_error_m": float(np.linalg.norm(np.asarray(err["detection"].xy_base_m) -
                                                        [err["gt"].x_m, err["gt"].y_m])),
            }))
    for missed in metrics["unmatched_truth"]:
        choices.append(("false_negative", missed))
    for false_positive in metrics["false_positive_details"]:
        choices.append(("false_positive", false_positive))
    saved: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for index, (kind, row) in enumerate(choices):
        seed = int(row["seed"])
        key = (kind, seed)
        if key in seen:
            continue
        seen.add(key)
        item = next(record for record in records if record["seed"] == seed)
        suffix = f"{index+1:02d}_{kind}_{seed}"
        frame = item["frame"]
        batch = item["batch"]
        _save_rgb(out_dir / f"{suffix}_rgb.png", frame.rgb)
        cv2.imwrite(str(out_dir / f"{suffix}_candidate_mask.png"), batch.candidate_mask)
        cv2.imwrite(str(out_dir / f"{suffix}_overlay.png"), _annotate(frame.rgb, batch, item["truth"], calibration))
        target_truth = next((gt for gt in item["truth"]
                             if gt.evaluator_key == row.get("truth_key")), None)
        if kind == "false_negative" and target_truth is None:
            target_truth = next((gt for gt in item["truth"]
                                 if gt.class_label == row.get("true_class") and
                                 abs(gt.x_m-float(row.get("x_true_m", float("inf")))) < 1e-8 and
                                 abs(gt.y_m-float(row.get("y_true_m", float("inf")))) < 1e-8), None)
        if kind == "false_negative":
            interpretation = "No prediction was within the one-to-one 25 mm evaluation matching gate; this object is counted as a false negative and NO_DETECTION in the confusion matrix."
        elif kind == "false_positive":
            interpretation = "This prediction could not be matched to any evaluator truth object within the fixed 25 mm one-to-one gate; status is shown to determine whether it was safely rejected or would be grasp-eligible."
        elif kind == "largest_xy_error":
            interpretation = "Matched localization residual retained with raw frame, candidate mask and overlay for manual perspective, edge, orientation and noise analysis."
        elif kind == "rejected_candidate_large_residual":
            interpretation = "Candidate was explicitly rejected by the status/reason contract; its coordinate is not grasp-eligible and is excluded from the accepted-pose error budget, but retained to inspect safe rejection behavior."
        else:
            interpretation = "Matched object color differs from the evaluator-only class; retained as an explicit classification failure."
        meta = {
            "case_type": kind, "seed": seed,
            "ground_truth_objects_evaluator_only": [asdict(gt) for gt in item["truth"]],
            "case_target_evaluator_only": asdict(target_truth) if target_truth else None,
            "detector_detections": [asdict(d) for d in batch.detections],
            "metrics_row": row,
            "interpretation": interpretation,
            "lighting_scale_evaluator_only": item["lighting_scale"],
            "noise_sigma_evaluator_only": item["noise_sigma"],
            "note": "Ground truth is retained only in the evaluator-side worst-case report; perception receives the RGB frame only.",
        }
        (out_dir / f"{suffix}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        saved.append({"case_type": kind, "seed": seed, "files_prefix": suffix})
    return saved


def _write_candidate_geometry_csv(path: Path, report: dict[str, Any]) -> None:
    rows = report["candidate_geometry_sweep"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["open_kernel_px", "min_area_px", "precision", "recall", "f1",
                  "unknown_recall", "xy_p95_m", "empty_scene_frames", "empty_scene_false_candidates"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sensor_contract_checks(detector: RGBObjectPerception, frame: CameraFrame,
                            truth: list[GroundTruthObject], cfg: dict[str, Any]) -> dict[str, Any]:
    primary_rgb = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    hsv = cv2.cvtColor(primary_rgb, cv2.COLOR_RGB2HSV)[0]
    expected_hue = (0, 60, 120)
    synthetic_rgb_ok = all(int(hsv[i, 0]) == hue and int(hsv[i, 1]) >= 250 and
                           int(hsv[i, 2]) >= 250 for i, hue in enumerate(expected_hue))
    K = cfg["camera"]["intrinsics_fx_fy_cx_cy_px"]
    z_top = float(cfg["calibration"]["object_top_plane_z_m"])
    cam_z = float(cfg["camera"]["position_world_m"][2])
    rendered: dict[str, list[dict[str, Any]]] = {}
    channel = {"RED": 0, "GREEN": 1, "BLUE": 2}
    for gt in truth:
        if gt.class_label not in channel:
            continue
        u = int(round(float(K[0]) * gt.x_m / (cam_z-z_top) + float(K[2])))
        v = int(round(-float(K[1]) * gt.y_m / (cam_z-z_top) + float(K[3])))
        patch = frame.rgb[max(0, v-2):min(frame.rgb.shape[0],v+3),
                          max(0, u-2):min(frame.rgb.shape[1],u+3)].astype(np.float32)
        if patch.size == 0:
            rendered.setdefault(gt.class_label, []).append({"mean_rgb": [], "pass": False})
            continue
        mean_rgb = np.mean(patch, axis=(0, 1))
        dominant = int(np.argmax(mean_rgb))
        order_ok = dominant == channel[gt.class_label]
        rendered.setdefault(gt.class_label, []).append({"mean_rgb": mean_rgb.tolist(),
                                                        "dominant_channel": dominant,
                                                        "pass": bool(order_ok)})
    rendered_ok = all(sample["pass"] for samples in rendered.values() for sample in samples)
    expected_camera = str(detector.config["sensor_interface"]["camera_config_id"])
    valid = detector.detect(frame, now_simulation_time_s=frame.simulation_time_s + 0.02)
    invalid = CameraFrame(None, frame.frame_id + 1, frame.simulation_time_s, False,
                          expected_camera, "INJECTED_MISSING_FRAME")
    missing = detector.detect(invalid)
    stale = CameraFrame(frame.rgb, frame.frame_id + 2, frame.simulation_time_s, True, expected_camera)
    stale_batch = detector.detect(stale, now_simulation_time_s=frame.simulation_time_s + 0.1)
    malformed = CameraFrame(np.zeros((10, 10, 3), dtype=np.uint8), frame.frame_id + 3,
                            frame.simulation_time_s, True, expected_camera)
    malformed_batch = detector.detect(malformed)
    wrong_camera = CameraFrame(frame.rgb, frame.frame_id + 4, frame.simulation_time_s,
                               True, "wrong-camera-config")
    wrong_camera_batch = detector.detect(wrong_camera)
    future = CameraFrame(frame.rgb, frame.frame_id + 5, frame.simulation_time_s + 0.1,
                         True, expected_camera)
    future_batch = detector.detect(future, now_simulation_time_s=frame.simulation_time_s)
    return {
        "synthetic_rgb_to_hsv": hsv.tolist(),
        "synthetic_rgb_channel_order_test_pass": bool(synthetic_rgb_ok),
        "actual_mujoco_render_channel_order_samples": rendered,
        "actual_mujoco_rgb_channel_order_test_pass": bool(rendered_ok),
        "rgb_channel_order_test_pass": bool(synthetic_rgb_ok and rendered_ok),
        "fresh_frame_status": valid.status,
        "missing_frame_status": missing.status,
        "stale_frame_status": stale_batch.status,
        "malformed_frame_status": malformed_batch.status,
        "wrong_camera_status": wrong_camera_batch.status,
        "future_timestamp_status": future_batch.status,
        "failure_returns_no_detections": all(not result.detections for result in
            (missing, stale_batch, malformed_batch, wrong_camera_batch, future_batch)),
        "fresh_frame_age_test_s": frame.age_s(frame.simulation_time_s + 0.02),
    }


def _tracking_checks(ssot: dict[str, Any], cfg: dict[str, Any], calibration: PlanarCalibration,
                     background: np.ndarray) -> dict[str, Any]:
    rig = PerceptionRig(ROOT, ssot)
    objects = [
        GroundTruthObject("track-fixture-red", "RED", -0.045, -0.25, 0.2, KNOWN_RGBA["RED"]),
        GroundTruthObject("track-fixture-green", "GREEN", 0.045, -0.25, 0.7, KNOWN_RGBA["GREEN"]),
    ]
    try:
        rig.set_objects(objects)
        original = rig.capture(sim_time_s=9.0)
        noisy_frame = rig.capture(sim_time_s=9.02, noise_sigma=0.7, noise_seed=99041)
    finally:
        rig.close()
    detector = RGBObjectPerception(cfg, calibration, background)
    first = detector.detect(original)
    second = detector.detect(noisy_frame)
    matched_same = []
    used: set[int] = set()
    for prior in first.detections:
        choices = [(float(np.linalg.norm(np.asarray(prior.xy_base_m)-np.asarray(current.xy_base_m))), idx, current)
                   for idx, current in enumerate(second.detections) if idx not in used]
        if not choices:
            continue
        distance, idx, current = min(choices, key=lambda item: item[0])
        if distance <= 0.005:
            used.add(idx)
            matched_same.append({"distance_m": distance, "track_id_frame1": prior.track_id,
                                 "track_id_frame2": current.track_id,
                                 "stable": prior.track_id == current.track_id})
    ids_unique = (len({d.track_id for d in first.detections}) == len(first.detections) and
                  len({d.track_id for d in second.detections}) == len(second.detections))
    stable = (len(first.detections) == 2 and len(second.detections) == 2 and
              bool(matched_same) and len(matched_same) == 2 and all(
        row["stable"] for row in matched_same
    ))
    return {"frame1_track_ids": [d.track_id for d in first.detections],
            "frame2_track_ids": [d.track_id for d in second.detections],
            "matched_observation_pairs": matched_same,
            "stable_track_ids": stable, "unique_ids_per_frame": ids_unique,
            "uses_internal_simulator_id": False}


def _render_occlusion_cases(ssot: dict[str, Any], cfg: dict[str, Any], calibration: PlanarCalibration,
                            background: np.ndarray, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = GroundTruthObject("evaluator-only-arm", "RED", -0.05, -0.28, 0.35, KNOWN_RGBA["RED"])
    cases = []
    # Geometric occluder simulates an object/person/fixture crossing the top-camera ray.
    rig = PerceptionRig(ROOT, ssot, include_occluder=True)
    rig.set_objects([gt])
    bg = rig.render_empty().rgb
    rig.set_objects([gt], occlusion=(gt.x_m, gt.y_m))
    frame = rig.capture(sim_time_s=1.0)
    detector = RGBObjectPerception(cfg, calibration, bg)
    batch = detector.detect(frame)
    _save_rgb(out_dir / "partial_occlusion_rgb.png", frame.rgb)
    cv2.imwrite(str(out_dir / "partial_occlusion_mask.png"), batch.candidate_mask)
    cv2.imwrite(str(out_dir / "partial_occlusion_overlay.png"), _annotate(frame.rgb, batch, [gt], calibration))
    cases.append({"name": "partial_occlusion_proxy", "candidate_count": len(batch.detections),
                  "statuses": [d.status for d in batch.detections],
                  "reasons": [d.reason for d in batch.detections],
                  "safe_rejection": all(d.status != "VALID" for d in batch.detections),
                  "ground_truth_only_evaluator": True})
    rig.close()

    # A second fixture retains the M2 robot render, moves its arm over the input area, and observes occlusion.
    arm_rig = PerceptionRig(ROOT, ssot, include_arm_visual=True)
    for joint_name, value in (("j1_shoulder", -math.pi / 2), ("j2_elbow", math.pi / 2),
                              ("j3_lift", 0.015), ("j4_wrist", 0.0)):
        joint_id = mujoco.mj_name2id(arm_rig.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        arm_rig.data.qpos[int(arm_rig.model.jnt_qposadr[joint_id])] = value
    mujoco.mj_forward(arm_rig.model, arm_rig.data)
    arm_rig.set_objects([gt])
    arm_bg = arm_rig.render_empty().rgb
    arm_frame = arm_rig.capture(sim_time_s=2.0)
    arm_detector = RGBObjectPerception(cfg, calibration, arm_bg)
    arm_batch = arm_detector.detect(arm_frame)
    _save_rgb(out_dir / "arm_occlusion_rgb.png", arm_frame.rgb)
    cv2.imwrite(str(out_dir / "arm_occlusion_mask.png"), arm_batch.candidate_mask)
    cv2.imwrite(str(out_dir / "arm_occlusion_overlay.png"), _annotate(arm_frame.rgb, arm_batch, [gt], calibration))
    cases.append({"name": "m2_arm_over_input", "candidate_count": len(arm_batch.detections),
                  "statuses": [d.status for d in arm_batch.detections],
                  "reasons": [d.reason for d in arm_batch.detections],
                  "detection_xy_error_m": [float(np.linalg.norm(np.asarray(d.xy_base_m)-[gt.x_m, gt.y_m]))
                                           for d in arm_batch.detections],
                  "safe_rejection": all(d.status != "VALID" for d in arm_batch.detections),
                  "ground_truth_only_evaluator": True})
    arm_rig.close()
    return {"cases": cases}


def _empty_scene_stress_checks(ssot: dict[str, Any], cfg: dict[str, Any],
                               calibration: PlanarCalibration, background: np.ndarray,
                               out_dir: Path) -> dict[str, Any]:
    """Independent false-positive check over blank scenes, lighting and deterministic noise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = PerceptionRig(ROOT, ssot)
    detector = RGBObjectPerception(cfg, calibration, background)
    rows: list[dict[str, Any]] = []
    try:
        for light in LIGHTING_LEVELS:
            rig.set_objects([], lighting_scale=light)
            for noise in (0.0, 1.5, 3.0):
                for repeat in range(4):
                    seed = 20261100 + len(rows)
                    frame = rig.capture(sim_time_s=len(rows) * 0.02,
                                        noise_sigma=noise, noise_seed=seed)
                    batch = detector.detect(frame)
                    row = {"seed": seed, "lighting_scale": light, "noise_sigma": noise,
                           "candidate_count": len(batch.detections), "status": batch.status,
                           "reasons": [d.reason for d in batch.detections]}
                    rows.append(row)
                    if batch.detections:
                        _save_rgb(out_dir / f"false_positive_{seed}_rgb.png", frame.rgb)
                        cv2.imwrite(str(out_dir / f"false_positive_{seed}_mask.png"), batch.candidate_mask)
    finally:
        rig.close()
    result = {"frame_count": len(rows), "lighting_levels": list(LIGHTING_LEVELS),
              "noise_sigma_levels": [0.0, 1.5, 3.0], "repeats_per_combination": 4,
              "false_positive_total": int(sum(r["candidate_count"] for r in rows)),
              "empty_scene_no_candidates": all(r["candidate_count"] == 0 for r in rows),
              "rows": rows}
    (out_dir / "empty_scene_stress.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _sorting_zone_exclusion_checks(ssot: dict[str, Any], cfg: dict[str, Any],
                                   calibration: PlanarCalibration, background: np.ndarray,
                                   out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = PerceptionRig(ROOT, ssot)
    rows = []
    zones = p_value(ssot, "cell.tray_centers_xy_m")
    try:
        for name, center in zones.items():
            label = str(name).upper()
            if label not in KNOWN_RGBA:
                continue
            gt = GroundTruthObject(f"evaluator-zone-{label}", label,
                                   float(center[0]), float(center[1]), 0.0, KNOWN_RGBA[label])
            rig.set_objects([gt])
            frame = rig.capture(sim_time_s=3.0 + len(rows) * 0.02)
            detector = RGBObjectPerception(cfg, calibration, background)
            batch = detector.detect(frame)
            _save_rgb(out_dir / f"zone_{label}_rgb.png", frame.rgb)
            cv2.imwrite(str(out_dir / f"zone_{label}_candidate_mask.png"), batch.candidate_mask)
            rows.append({"zone": label, "candidate_count": len(batch.detections),
                         "batch_status": batch.status,
                         "excluded_from_input_detection": len(batch.detections) == 0})
    finally:
        rig.close()
    result = {"zones_tested": rows,
              "all_sorting_zones_excluded": bool(rows) and all(r["excluded_from_input_detection"] for r in rows)}
    (out_dir / "zone_exclusion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _neighbor_object_checks(ssot: dict[str, Any], cfg: dict[str, Any],
                            calibration: PlanarCalibration, background: np.ndarray,
                            out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rig = PerceptionRig(ROOT, ssot)
    cases = []
    try:
        # A 36 mm center separation leaves an 8 mm visible floor gap between 28 mm squares.
        for case_name, separation_m in (("separated_8mm_gap", 0.036), ("touching_faces", 0.028)):
            objects = [
                GroundTruthObject(f"{case_name}-red", "RED", -separation_m/2, -0.25, 0.0, KNOWN_RGBA["RED"]),
                GroundTruthObject(f"{case_name}-green", "GREEN", separation_m/2, -0.25,
                                  0.0, KNOWN_RGBA["GREEN"]),
            ]
            rig.set_objects(objects)
            frame = rig.capture(sim_time_s=4.0 + len(cases) * 0.02)
            detector = RGBObjectPerception(cfg, calibration, background)
            batch = detector.detect(frame)
            matches, missed, false = match_detections(batch.detections, objects)
            valid_count = sum(d.status == "VALID" for d in batch.detections)
            if case_name == "separated_8mm_gap":
                passed = len(matches) == 2 and not missed and not false and valid_count == 2
            else:
                safely_separated = len(matches) == 2 and not missed and not false and valid_count == 2
                safely_rejected = bool(batch.detections) and all(d.status != "VALID" for d in batch.detections)
                passed = safely_separated or safely_rejected
            _save_rgb(out_dir / f"{case_name}_rgb.png", frame.rgb)
            cv2.imwrite(str(out_dir / f"{case_name}_candidate_mask.png"), batch.candidate_mask)
            cv2.imwrite(str(out_dir / f"{case_name}_overlay.png"), _annotate(frame.rgb, batch, objects, calibration))
            cases.append({"name": case_name, "center_separation_m": separation_m,
                          "candidate_count": len(batch.detections),
                          "statuses": [d.status for d in batch.detections],
                          "reasons": [d.reason for d in batch.detections],
                          "matched_objects": len(matches), "missed_objects": len(missed),
                          "false_detections": len(false), "pass": passed})
    finally:
        rig.close()
    result = {"cases": cases, "all_neighbor_checks_pass": bool(cases) and all(c["pass"] for c in cases)}
    (out_dir / "neighbor_checks.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def _write_tuning_csv(path: Path, report: dict[str, Any]) -> None:
    rows = report["grid_search"]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["lab_delta_threshold", "minimum_confidence", "score", "detection_f1", "xy_p95_m"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def _profile_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(yaml.safe_dump(config, sort_keys=True).encode("utf-8")).hexdigest()


def _profile_parameter_metadata(config: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Attach units, provenance and M4 ownership to every runtime-profile leaf."""
    units = {
        "sensor_interface.frame_format": "pixel channel order",
        "sensor_interface.dtype": "data type",
        "sensor_interface.resolution_px": "px (width, height)",
        "sensor_interface.pixel_origin": "image-coordinate convention",
        "sensor_interface.maximum_age_s": "s",
        "sensor_interface.camera_config_id": "configuration ID",
        "camera.position_world_m": "m (x, y, z)",
        "camera.mujoco_quaternion_wxyz": "unit quaternion (w, x, y, z)",
        "camera.fovy_deg": "degree",
        "camera.intrinsics_fx_fy_cx_cy_px": "px (fx, fy, cx, cy)",
        "calibration.method": "method ID",
        "calibration.object_top_plane_z_m": "m in BASE/world Z",
        "calibration.calibration_grid_xy_m.x": "m in BASE X",
        "calibration.calibration_grid_xy_m.y": "m in BASE Y",
        "regions.input_xy_bounds_m": "m in BASE (X range, Y range)",
        "candidate_detection.method": "method ID",
        "candidate_detection.object_size_xy_m": "m (width, length)",
        "candidate_detection.lab_delta_threshold": "Delta E*ab (CIE Lab scale)",
        "candidate_detection.open_kernel_px": "px (kernel width, height)",
        "candidate_detection.close_kernel_px": "px (kernel width, height)",
        "candidate_detection.min_area_px": "px^2",
        "candidate_detection.max_area_px": "px^2",
        "candidate_detection.nominal_projected_object_area_px": "px^2",
        "candidate_detection.min_aspect_ratio": "dimensionless (long/short side)",
        "candidate_detection.max_aspect_ratio": "dimensionless (long/short side)",
        "candidate_detection.occlusion_area_ratio": "ratio of nominal projected area",
        "candidate_detection.merged_candidate_area_ratio": "ratio of nominal projected area",
        "candidate_detection.area_model_note": "model note",
        "merged_candidate_split.enabled": "boolean",
        "merged_candidate_split.minimum_component_area_ratio": "ratio of nominal projected area",
        "merged_candidate_split.maximum_component_area_ratio": "ratio of nominal projected area",
        "merged_candidate_split.minimum_components": "component count",
        "color_classifier.conversion": "OpenCV conversion code",
        "color_classifier.hue_range": "OpenCV HSV hue units (0..179)",
        "color_classifier.saturation_range": "8-bit levels (0..255)",
        "color_classifier.value_range": "8-bit levels (0..255)",
        "color_classifier.hsv_ranges.RED.hue_intervals": "OpenCV HSV hue units (0..179)",
        "color_classifier.hsv_ranges.RED.saturation_min": "8-bit saturation level",
        "color_classifier.hsv_ranges.RED.value_min": "8-bit value level",
        "color_classifier.hsv_ranges.GREEN.hue_intervals": "OpenCV HSV hue units (0..179)",
        "color_classifier.hsv_ranges.GREEN.saturation_min": "8-bit saturation level",
        "color_classifier.hsv_ranges.GREEN.value_min": "8-bit value level",
        "color_classifier.hsv_ranges.BLUE.hue_intervals": "OpenCV HSV hue units (0..179)",
        "color_classifier.hsv_ranges.BLUE.saturation_min": "8-bit saturation level",
        "color_classifier.hsv_ranges.BLUE.value_min": "8-bit value level",
        "color_classifier.interior_erode_kernel_px": "px (kernel width and height)",
        "color_classifier.minimum_confidence": "score fraction (0..1)",
        "color_classifier.minimum_class_margin": "score fraction (0..1)",
        "pose_estimation.center_method": "method ID",
        "pose_estimation.yaw_method": "method ID",
        "pose_estimation.symmetry_period_rad": "rad",
        "pose_estimation.yaw_interval_rad": "rad in BASE",
        "photometric_normalization.method": "method ID",
        "photometric_normalization.anchor_xy_bounds_m": "m in BASE (X range, Y range)",
        "photometric_normalization.gain_clip": "per-channel multiplicative gain",
        "tracking.method": "method ID",
        "tracking.max_distance_m": "m in BASE XY",
        "tracking.max_missed_frames": "frames",
        "uncertainty.position_sigma_m": "m (configured position allowance)",
        "uncertainty.yaw_sigma_rad": "rad (configured angular allowance)",
    }

    def leaves(value: Any, prefix: str = "") -> list[str]:
        if isinstance(value, dict):
            result: list[str] = []
            for key, child in value.items():
                if prefix == "" and key == "config_version":
                    continue
                result.extend(leaves(child, f"{prefix}.{key}" if prefix else key))
            return result
        return [prefix]

    profile_paths = set(leaves(config))
    if profile_paths != set(units):
        missing = sorted(profile_paths - set(units))
        obsolete = sorted(set(units) - profile_paths)
        raise AssertionError(f"M4 profile metadata coverage mismatch; missing={missing}, obsolete={obsolete}")

    tuning_paths = {
        path for path in profile_paths
        if path.startswith("candidate_detection.") and path not in {
            "candidate_detection.object_size_xy_m", "candidate_detection.nominal_projected_object_area_px",
            "candidate_detection.area_model_note", "candidate_detection.method",
        }
        or path.startswith("merged_candidate_split.")
        or path.startswith("color_classifier.")
    }
    output: dict[str, dict[str, str]] = {}
    for path in sorted(profile_paths):
        if path.startswith("camera.") or path.startswith("sensor_interface."):
            provenance = "Inherited from M1/M2 sensor contract and M2 SSOT; rendered RGB/channel behavior is checked in M4."
        elif path in {"regions.input_xy_bounds_m", "candidate_detection.object_size_xy_m",
                      "candidate_detection.nominal_projected_object_area_px", "candidate_detection.area_model_note"}:
            provenance = "Inherited or calculated from the M2 workcell/object/camera geometry; cross-checked against the M2 SSOT."
        elif path.startswith("calibration."):
            provenance = "M4 calibration procedure; top-face height is derived from M2 table and object dimensions; see camera_calibration.json."
        elif path in tuning_paths:
            provenance = "Frozen from the M4 tuning split (seeds 20260401..20260424) and separate blank-scene checks; never tuned on validation seeds."
        elif path.startswith("pose_estimation."):
            provenance = "M4 geometry-method comparison on tuning data; square symmetry is inherited from M2 object geometry."
        elif path.startswith("photometric_normalization."):
            provenance = "M4 fixed-background/lighting study; validated under the stated controlled MuJoCo lighting envelope."
        elif path.startswith("tracking."):
            provenance = "M4 deterministic two-frame noise/repeated-observation contract check."
        elif path.startswith("uncertainty."):
            provenance = "Configured engineering allowance in M4; not a calibrated probability distribution or guaranteed worst-case bound."
        else:
            provenance = "M4 sensor/perception interface decision; see the M4 specification and acceptance report."
        output[path] = {"unit": units[path], "provenance": provenance, "owner": "M4"}
    return output


def _freeze_ssot_profile(ssot: dict[str, Any], config: dict[str, Any]) -> str:
    """Make the frozen M4 perception profile canonical in the project SSOT."""
    profile = copy.deepcopy(config)
    profile["config_version"] = M4_VERSION
    digest = _profile_sha256(profile)
    # The perception profile has its own version. Do not downgrade the
    # project-wide SSOT when a later milestone reruns M4 verification.
    current_version = ssot.get("config_version")
    if current_version is None:
        ssot["config_version"] = M4_VERSION
    else:
        current_match = _CONFIG_VERSION_PATTERN.fullmatch(str(current_version))
        m4_match = _CONFIG_VERSION_PATTERN.fullmatch(M4_VERSION)
        if not current_match or not m4_match:
            raise ValueError(f"Unsupported SSOT config version: {current_version!r}")
        current_rank = tuple(int(current_match.group(key)) for key in ("milestone", "major", "minor"))
        m4_rank = tuple(int(m4_match.group(key)) for key in ("milestone", "major", "minor"))
        if current_rank <= m4_rank:
            ssot["config_version"] = M4_VERSION
    ssot["perception"] = {
        "profile_version": M4_VERSION,
        "profile_sha256": digest,
        "runtime_config": profile,
        "runtime_config_is_canonical": True,
        "runtime_yaml_is_export": "04_Программа/perception/perception_config.yaml",
        "profile_parameter_metadata": _profile_parameter_metadata(profile),
        "basis": "Tuning split only; camera and workcell geometry are inherited from M2 SSOT.",
    }
    SSOT_PATH.write_text(yaml.safe_dump(ssot, allow_unicode=True, sort_keys=False), encoding="utf-8")
    CONFIG_PATH.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return digest


def _perception_boundary_audit() -> dict[str, Any]:
    """Statically verify that runtime perception has no path to simulator truth."""
    source_dir = PROGRAM_DIR / "perception"
    files = sorted(source_dir.glob("*.py"))
    forbidden_modules = {"mujoco", "rig", "evaluator", "simulator", "ground_truth"}
    forbidden_names = {"qpos", "segmentation", "object_id", "evaluator_key", "truth_objects"}
    violations: list[dict[str, Any]] = []
    for source in files:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_modules:
                        violations.append({"file": source.name, "line": node.lineno,
                                           "kind": "forbidden_import", "name": alias.name})
            elif isinstance(node, ast.ImportFrom):
                root_module = (node.module or "").split(".")[0]
                if root_module in forbidden_modules:
                    violations.append({"file": source.name, "line": node.lineno,
                                       "kind": "forbidden_import", "name": node.module})
            elif isinstance(node, ast.Name) and node.id.lower() in forbidden_names:
                violations.append({"file": source.name, "line": node.lineno,
                                   "kind": "forbidden_symbol", "name": node.id})
    return {"runtime_perception_files": [p.name for p in files], "violations": violations,
            "no_hidden_ground_truth_access": not violations,
            "sensor_input_contract": "RGB CameraFrame + calibration/config + static background reference",
            "evaluator_truth_separate": True}


def _build_tuning_empty_records(ssot: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """Build disjoint blank-scene/noise frames used only to reject false-positive tunings."""
    rig = PerceptionRig(ROOT, ssot)
    records: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        for light in LIGHTING_LEVELS:
            rig.set_objects([], lighting_scale=light)
            for noise in (0.0, 1.5, 3.0):
                for repeat in range(3):
                    seed = 20261200 + len(records)
                    frame = rig.capture(sim_time_s=len(records) * 0.02,
                                        noise_sigma=noise, noise_seed=seed)
                    records.append({"seed": seed, "frame": frame,
                                    "lighting_scale": light, "noise_sigma": noise})
                    _save_rgb(out_dir / f"empty_{seed}.png", frame.rgb)
    finally:
        rig.close()
    manifest = [{"seed": row["seed"], "lighting_scale": row["lighting_scale"],
                 "noise_sigma": row["noise_sigma"], "truth_objects_evaluator_only": []}
                for row in records]
    (out_dir / "evaluator_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return records


def main() -> None:
    for directory in (CALIBRATION_DIR, TUNING_DIR, VALIDATION_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    ssot = load_ssot()
    previous_config_version = str(ssot.get("config_version", "UNKNOWN"))
    m3_acceptance = (ROOT / "01_Управление" / "приемка" / "M03.md").read_text(encoding="utf-8")
    m3_match = re.search(r"\*\*Версия:\*\*\s*`?([A-Za-z0-9._-]+)`?", m3_acceptance)
    if not m3_match:
        raise RuntimeError("Cannot establish the accepted M3 baseline version from M03.md")
    if ssot.get("perception", {}).get("profile_version") == M4_VERSION:
        initial_cfg = copy.deepcopy(ssot["perception"]["runtime_config"])
        if _profile_sha256(initial_cfg) != ssot["perception"].get("profile_sha256"):
            raise AssertionError("SSOT perception profile hash mismatch")
    else:
        initial_cfg = load_config(CONFIG_PATH)
    # Every duplicated camera/geometry value is checked against the governing M2 SSOT.
    expected = {
        "position_world_m": p_value(ssot, "camera.position_world_m"),
        "mujoco_quaternion_wxyz": p_value(ssot, "camera.mujoco_quaternion_wxyz"),
        "fovy_deg": p_value(ssot, "camera.fovy_deg"),
        "intrinsics_fx_fy_cx_cy_px": p_value(ssot, "camera.intrinsics_fx_fy_cx_cy_px"),
    }
    for key, value in expected.items():
        if initial_cfg["camera"][key] != value:
            raise AssertionError(f"Camera config disagrees with M2 SSOT: {key}")
    if initial_cfg["sensor_interface"]["resolution_px"] != p_value(ssot, "camera.resolution_px"):
        raise AssertionError("Sensor frame size disagrees with M2 SSOT")
    if float(initial_cfg["calibration"]["object_top_plane_z_m"]) != (
        float(p_value(ssot, "cell.table_top_z_m")) + float(p_value(ssot, "object.size_xyz_m")[2])
    ):
        raise AssertionError("Calibration plane is not the SSOT object-top plane")
    if list(initial_cfg["candidate_detection"]["object_size_xy_m"]) != list(
        p_value(ssot, "object.size_xyz_m")[:2]
    ):
        raise AssertionError("Perception footprint geometry disagrees with M2 object dimensions")

    calibration_artifact = run_calibration(ROOT, initial_cfg, CALIBRATION_DIR)
    calibration = PlanarCalibration.from_json(str(CALIBRATION_DIR / "camera_calibration.json"))
    if calibration_artifact["validation_metrics"]["max_planar_error_m"] > float(p_value(ssot, "robot.grasp_xy_budget_perception_m")):
        raise RuntimeError("Calibration validation alone exceeds the perception XY budget")

    tuning_background, tuning_records = _build_sample_cache(ssot, tuning=True)
    validation_background, validation_records = _build_sample_cache(ssot, tuning=False)
    if not np.array_equal(tuning_background, validation_background):
        raise AssertionError("Tuning/validation background baseline differs")
    if set(TUNING_SEEDS) & set(VALIDATION_SEEDS):
        raise AssertionError("Tuning and validation seeds overlap")
    np.save(TUNING_DIR / "background_reference_rgb.npy", tuning_background)
    np.save(VALIDATION_DIR / "background_reference_rgb.npy", validation_background)

    tuning_empty_records = _build_tuning_empty_records(ssot, TUNING_DIR / "empty_scene")
    frozen_cfg, tuning_report = _tune_config(
        tuning_background, tuning_records, calibration, initial_cfg, tuning_empty_records
    )
    tuning_report["camera_calibration_id"] = calibration.calibration_id
    tuning_report["validation_seed_hash"] = hashlib.sha256(",".join(map(str, VALIDATION_SEEDS)).encode()).hexdigest()
    tuning_report["validation_was_not_used_for_tuning"] = True
    frozen_cfg["config_version"] = M4_VERSION
    config_hash = _freeze_ssot_profile(ssot, frozen_cfg)
    tuning_report["frozen_config_sha256_before_validation"] = config_hash
    (TUNING_DIR / "tuning_report.json").write_text(json.dumps(tuning_report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_tuning_csv(TUNING_DIR / "parameter_sweep.csv", tuning_report)
    # Save tuning RGB observations plus evaluator-only labels in a separate manifest.
    tuning_manifest = []
    for item in tuning_records:
        _save_rgb(TUNING_DIR / "raw_rgb" / f"frame_{item['seed']}.png", item["frame"].rgb)
        tuning_manifest.append({"seed": item["seed"], "lighting_scale": item["lighting_scale"],
                                "noise_sigma": item["noise_sigma"],
                                "truth_objects_evaluator_only": [asdict(gt) for gt in item["truth"]]})
    (TUNING_DIR / "evaluator_manifest.json").write_text(
        json.dumps(tuning_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    detector = RGBObjectPerception(frozen_cfg, calibration, validation_background)
    metrics = _metrics_for_dataset(validation_records, detector)
    _save_dataset_outputs(validation_records, metrics, VALIDATION_DIR, calibration)
    _write_confusion_csv(VALIDATION_DIR / "color_confusion_matrix.csv", metrics["color_confusion_matrix_counts"])
    worst = _save_worst_cases(metrics, validation_records, VALIDATION_DIR / "worst_cases", calibration)

    first_known_frame = next(
        item for item in validation_records
        if any(gt.class_label in KNOWN_RGBA for gt in item["truth"])
    )
    capture_contract = _sensor_contract_checks(
        detector, first_known_frame["frame"], first_known_frame["truth"], frozen_cfg
    )
    tracking = _tracking_checks(ssot, frozen_cfg, calibration, validation_background)
    occlusion = _render_occlusion_cases(ssot, frozen_cfg, calibration, validation_background,
                                        VALIDATION_DIR / "occlusion")
    budget_xy = float(p_value(ssot, "robot.grasp_xy_budget_perception_m"))
    budget_yaw = float(p_value(ssot, "robot.grasp_yaw_error_limit_rad"))
    xy = metrics["xy_planar_error_summary_m"]
    yaw = metrics["yaw_error_summary_rad"]
    wrong_known_class_predictions = sum(
        int(metrics["color_confusion_matrix_counts"].get(actual, {}).get(predicted, 0))
        for actual in ("RED", "GREEN", "BLUE")
        for predicted in ("RED", "GREEN", "BLUE") if predicted != actual
    )
    unsupported_unknown_promotions = sum(
        int(metrics["color_confusion_matrix_counts"].get("UNKNOWN", {}).get(predicted, 0))
        for predicted in ("RED", "GREEN", "BLUE")
    )
    xy_pass = xy["p95"] is not None and float(xy["p95"]) <= budget_xy
    yaw_pass = yaw["p95"] is not None and float(yaw["p95"]) <= budget_yaw
    operating_domain = metrics["operating_domain"]
    # A deterministic engineering acceptance gate: every test object whose measured
    # local contrast meets the detector's frozen threshold must be detected, and there
    # must be no unmatched candidate. This is not an official course requirement.
    supported_object_detection_pass = (
        operating_domain["supported_object_count"] > 0 and
        operating_domain["unaccounted_supported_false_negative_count"] == 0 and
        operating_domain["unmatched_valid_prediction_count"] == 0
    )
    empty_scene = _empty_scene_stress_checks(
        ssot, frozen_cfg, calibration, validation_background, VALIDATION_DIR / "empty_scene"
    )
    zone_exclusion = _sorting_zone_exclusion_checks(
        ssot, frozen_cfg, calibration, validation_background, VALIDATION_DIR / "sorting_zone_exclusion"
    )
    neighbors = _neighbor_object_checks(
        ssot, frozen_cfg, calibration, validation_background, VALIDATION_DIR / "neighbor_objects"
    )
    boundary = _perception_boundary_audit()
    acceptance = {
        "supported_objects_are_valid_or_explicitly_safely_rejected": supported_object_detection_pass,
        "no_unsafe_known_color_cross_classification": wrong_known_class_predictions == 0,
        "unknown_not_promoted_to_supported_color": unsupported_unknown_promotions == 0,
        "xy_p95_within_perception_budget": xy_pass,
        "yaw_p95_within_limit": yaw_pass,
        "rgb_channel_order": capture_contract["rgb_channel_order_test_pass"],
        "missing_stale_malformed_handled": capture_contract["missing_frame_status"] == "NO_FRAME" and
                                          capture_contract["stale_frame_status"] == "STALE" and
                                          capture_contract["malformed_frame_status"] == "INVALID" and
                                          capture_contract["wrong_camera_status"] == "INVALID" and
                                          capture_contract["future_timestamp_status"] == "INVALID" and
                                          capture_contract["failure_returns_no_detections"],
        "tracking_ids_stable_under_small_noise": tracking["stable_track_ids"] and
                                                 tracking["unique_ids_per_frame"],
        "sorting_zones_excluded": zone_exclusion["all_sorting_zones_excluded"],
        "empty_scene_has_no_false_candidates": empty_scene["empty_scene_no_candidates"],
        "neighbor_object_cases_handled": neighbors["all_neighbor_checks_pass"],
        "occlusion_cases_rejected_safely": all(case["safe_rejection"] for case in occlusion["cases"]),
        "ground_truth_boundary_audit": boundary["no_hidden_ground_truth_access"],
        "tuning_validation_seeds_disjoint": not (set(TUNING_SEEDS) & set(VALIDATION_SEEDS)),
        "config_frozen_before_validation": True,
    }
    report = {
        "status": "PASS" if all(acceptance.values()) else "NOT_PASS",
        "milestone": M4_VERSION,
        "configuration_version": ssot["config_version"],
        "ssot_version_at_run_start": previous_config_version,
        "m3_predecessor_ssot_version": m3_match.group(1),
        "camera_calibration": calibration_artifact["validation_metrics"],
        "camera_calibration_id": calibration.calibration_id,
        "frozen_config_sha256": config_hash,
        "ssot_sha256_after_profile_freeze": sha256(SSOT_PATH),
        "runtime_config_sha256": sha256(CONFIG_PATH),
        "validation_seed_ids": list(VALIDATION_SEEDS),
        "validation_frame_count": len(validation_records),
        "validation_object_count": int(sum(len(item["truth"]) for item in validation_records)),
        "validation_dataset_hash": hashlib.sha256(",".join(map(str, VALIDATION_SEEDS)).encode()).hexdigest(),
        "validation_metrics": {
            "object_detection": metrics["object_detection"],
            "color_confusion_matrix_counts": metrics["color_confusion_matrix_counts"],
            "color_metrics": metrics["color_metrics"],
            "color_sample_counts": metrics["color_sample_counts"],
            "xy_planar_error_summary_m": xy,
            "x_signed_error_summary_m": metrics["x_signed_error_summary_m"],
            "y_signed_error_summary_m": metrics["y_signed_error_summary_m"],
            "yaw_error_summary_rad": yaw,
            "candidate_yaw_error_summary_rad": metrics["candidate_yaw_error_summary_rad"],
            "center_estimator_comparison": metrics["center_estimator_comparison"],
            "lighting_strata": _stratified_metrics(metrics["frame_rows"]),
            "operating_domain": operating_domain,
            "detection_status_counts": metrics["detection_status_counts"],
        },
        "grasp_budget": {"perception_xy_budget_m": budget_xy,
                         "total_xy_budget_m": float(p_value(ssot, "robot.grasp_xy_error_budget_m")),
                         "yaw_error_limit_rad": budget_yaw,
                         "xy_p95_remaining_total_budget_m": float(p_value(ssot, "robot.grasp_xy_error_budget_m")) - float(xy["p95"] or 0.0)},
        "sensor_contract_checks": capture_contract,
        "tracking_contract": tracking,
        "occlusion_tests": occlusion,
        "empty_scene_stress": empty_scene,
        "sorting_zone_exclusion": zone_exclusion,
        "neighbor_object_checks": neighbors,
        "acceptance_checks": acceptance,
        "classification_safety_counts": {
            "known_color_cross_classifications": wrong_known_class_predictions,
            "unknown_promoted_to_supported_color": unsupported_unknown_promotions,
        },
        "worst_cases": worst,
        "ground_truth_boundary": boundary,
        "versions": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                     "mujoco": mujoco.__version__, "numpy": np.__version__,
                     "config_sha256": config_hash},
        "limitations": [
            "Objects are matte RGB cuboids rendered with controlled MuJoCo lighting; real camera color response is not modeled.",
            "Position estimation is calibrated for one fixed object-top plane at z=16 mm; other object heights are unsupported.",
            "The candidate detector is ROI/background based and requires a clear, calibrated tabletop background.",
            "Neutral objects whose rendered local contrast from the tabletop falls below the measured operating limit cannot be guaranteed detectable; a pale-gray-on-white miss remains a documented limitation and remains counted in raw recall.",
            "Touching/merged objects and substantial occlusion require rejection; this module does not separate arbitrary piles.",
            "The homography validates image-to-BASE geometry but is not a collision-free grasp or trajectory proof.",
        ],
    }
    (VALIDATION_DIR / "perception_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "m4_perception_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "acceptance": acceptance,
                      "detection": metrics["object_detection"], "xy": xy, "yaw": yaw,
                      "confusion": metrics["color_confusion_matrix_counts"],
                      "calibration": calibration_artifact["validation_metrics"],
                      "config_hash": config_hash}, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


def _stratified_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for light in LIGHTING_LEVELS:
        subset = [r for r in rows if abs(float(r["lighting_scale"]) - light) < 1e-9]
        gt = sum(int(r["gt_count"]) for r in subset)
        tp = sum(int(r["matched"]) for r in subset)
        fp = sum(int(r["false_positive"]) for r in subset)
        fn = sum(int(r["false_negative"]) for r in subset)
        out[str(light)] = {"frames": len(subset), "gt": gt, "tp": tp, "fp": fp, "fn": fn,
                           "precision": tp/(tp+fp) if tp+fp else 0.0,
                           "recall": tp/(tp+fn) if tp+fn else 0.0}
    return out


if __name__ == "__main__":
    main()
