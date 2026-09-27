from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_PATH = ROOT / "07_Испытания" / "campaign.yaml"
RUNS_ROOT = ROOT / "07_Испытания" / "runs"
TABLES_ROOT = ROOT / "07_Испытания" / "tables"
PLOTS_ROOT = ROOT / "07_Испытания" / "plots"
RESULTS_PATH = ROOT / "07_Испытания" / "campaign_results.json"
COLORS = ("RED", "GREEN", "BLUE")
PREDICTION_LABELS = (*COLORS, "UNKNOWN", "MISSING")
JOINTS = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")
ACTUATORS = (
    ("act_j1_shoulder", "actuator_force_j1_Nm"),
    ("act_j2_elbow", "actuator_force_j2_Nm"),
    ("act_j3_lift", "actuator_force_j3_N"),
    ("act_j4_wrist", "actuator_force_j4_Nm"),
    ("act_gripper_left", "actuator_force_gripper_N"),
)
SNAPSHOT_INPUT_FILES = {
    "ssot": "ssot.yaml",
    "mjcf": "mjcf.xml",
    "runtime_config": "runtime_config.yaml",
    "scenario": "scenario.yaml",
    "perception_config": "perception_config.yaml",
    "camera_calibration": "camera_calibration.json",
    "background_reference": "background_reference.npy",
    "placement_camera_calibration": "placement_camera_calibration.json",
    "placement_camera_background": "placement_camera_background.npy",
    "runner": "runner.py",
    "placement_camera_model": "placement_camera_model.py",
    "placement_camera_asset_builder": "placement_camera_asset_builder.py",
    "runtime": "runtime.py",
    "recording": "recording.py",
    "actuator": "actuator.py",
    "scene_generator": "scene_generator.py",
    "sensor_adapter": "sensor_adapter.py",
    "placement_adapter": "placement_adapter.py",
    "interactive_display": "interactive_display.py",
    "controller": "controller.py",
    "planner": "planner.py",
    "perception": "perception.py",
    "evaluator": "evaluator.py",
    "evaluator_telemetry": "evaluator_telemetry.py",
    "m7_integrated_model": "m7_integrated_model.xml",
}


def _sha256(path: Path, *, decompress_gzip: bool = False) -> str:
    opener = gzip.open if decompress_gzip else open
    digest = hashlib.sha256()
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_snapshot_sha256(key: str, path: Path) -> str:
    """Use canonical LF text for the generated MJCF; hash every other snapshot byte-for-byte."""
    if key == "m7_integrated_model":
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n"))
        return digest.hexdigest()
    return _sha256(path)


def wilson_interval(successes: int, observations: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for independent Bernoulli units, in [0, 1]."""
    if observations < 0 or successes < 0 or successes > observations:
        raise ValueError("Wilson counts require 0 <= successes <= observations")
    if observations == 0:
        return (math.nan, math.nan)
    p = successes / observations
    z2 = z * z
    denominator = 1.0 + z2 / observations
    center = (p + z2 / (2.0 * observations)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / observations + z2 / (4.0 * observations**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def classification_metrics(
    matrix: dict[tuple[str, str], int], classes: Iterable[str] = COLORS,
) -> list[dict[str, Any]]:
    """Compute one-vs-rest metrics, counting unmatched detections as false positives."""
    class_list = tuple(classes)
    truth_labels = (*class_list, "NO_OBJECT")
    rows: list[dict[str, Any]] = []
    for label in class_list:
        true_positive = matrix.get((label, label), 0)
        support = sum(matrix.get((label, predicted), 0) for predicted in PREDICTION_LABELS)
        predicted_count = sum(matrix.get((truth, label), 0) for truth in truth_labels)
        precision = true_positive / predicted_count if predicted_count else None
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall > 0
            else 0.0 if precision == 0.0 or recall == 0.0 else None
        )
        rows.append({
            "class_label": label,
            "true_positive": true_positive,
            "support_objects": support,
            "predicted_objects_including_false_positives": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "unknown_abstentions": matrix.get((label, "UNKNOWN"), 0),
            "missing_detections": matrix.get((label, "MISSING"), 0),
        })
    return rows


INTENDED_GRASP_STATES = frozenset({
    "DESCEND", "GRASP", "VERIFY_HOLD", "TRANSFER", "PLACE", "RELEASE", "RETREAT",
})


def classify_contact_pair(
    geom1: str, geom2: str, *, target_body: str | None = None,
    controller_state: str | None = None,
) -> str:
    """Permit finger contact only with the current target during grasp/carry phases."""
    names = (str(geom1), str(geom2))
    object_geometries = [
        name for name in names
        if name.startswith("object_") and name.endswith("_collision")
    ]
    if len(object_geometries) != 1:
        return "forbidden"
    object_geometry = object_geometries[0]
    object_name = object_geometry.removesuffix("_collision")
    other = names[1] if names[0] == object_geometry else names[0]
    if other == "table_top_collision":
        return "object_support"
    if other.startswith("tray_") and other.endswith("_floor_collision"):
        return "tray_support"
    if other.startswith("tray_") and "_wall_collision" in other:
        return "tray_boundary"
    if other in {"finger_left_collision", "finger_right_collision"}:
        return (
            "intended_grasp"
            if object_name == target_body and controller_state in INTENDED_GRASP_STATES
            else "forbidden"
        )
    return "forbidden"


def _read_text(path: Path) -> str:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return stream.read()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(_read_text(path))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(_read_text(path).splitlines()))


def _resolve_output(run_dir: Path, relative: str) -> Path:
    path = run_dir / relative
    if path.exists():
        return path
    if path.suffix != ".gz" and path.with_name(path.name + ".gz").exists():
        return path.with_name(path.name + ".gz")
    raise FileNotFoundError(f"Missing M7/M9 run output: {path}")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _controller_target_at(
    controller_events: list[dict[str, Any]], track_to_body: dict[str, str], time_s: float,
) -> tuple[str | None, str | None]:
    """Resolve a controller public track to a target body at a physics event time."""
    current_track: str | None = None
    current_state: str | None = None
    ordered = sorted(controller_events, key=lambda row: (
        float(row.get("simulation_time_s", math.inf)), int(row.get("event_id", 0)),
    ))
    for event in ordered:
        event_time = _number(event.get("simulation_time_s"))
        if event_time is None or event_time > time_s:
            break
        if event.get("track_id") is not None:
            current_track = str(event["track_id"])
        if event.get("event_type") == "STATE_ENTERED":
            current_state = str(event.get("state", ""))
        if event.get("event_type") == "CYCLE_CLOSED":
            current_track = None
    target_body = track_to_body.get(current_track) if current_track is not None else None
    return target_body, current_state


def _placement_confirmation_failures(
    track_records: dict[str, dict[str, Any]], object_records: list[dict[str, Any]],
) -> list[str]:
    by_track = {
        str(row["track_id"]): row for row in object_records
        if row.get("track_id") is not None and row.get("body_name") is not None
    }
    failures: list[str] = []
    for track_id, record in track_records.items():
        if record.get("status") != "PLACED_CONTROLLER_CONFIRMED":
            continue
        row = by_track.get(str(track_id))
        if (row is None or row.get("physical_sort_success") is not True
                or record.get("class_label") != row.get("ground_truth_class")):
            failures.append(str(track_id))
    return failures


def _object_outcomes_accounted(
    object_records: list[dict[str, Any]], *, allow_safe_failure: bool = False,
) -> bool:
    """Every matched truth object must be physically sorted or reason-coded skipped."""
    for row in object_records:
        if row.get("ground_truth_class") not in COLORS:
            continue
        if (row.get("physical_sort_success") is True
                and row.get("controller_status") == "PLACED_CONTROLLER_CONFIRMED"
                and row.get("controller_class") == row.get("ground_truth_class")):
            continue
        reasons = row.get("controller_failure_reasons", "")
        if (str(row.get("controller_status", "")).startswith("SKIPPED_")
                and bool(reasons)):
            continue
        if (allow_safe_failure and row.get("controller_status") == "SAFE_FAILURE"
                and bool(reasons)):
            continue
        return False
    return True


def _expected_object_outcome_failures(
    expected: dict[str, str], object_records: list[dict[str, Any]],
) -> list[str]:
    by_body = {str(row.get("body_name")): row for row in object_records if row.get("body_name")}
    failures: list[str] = []
    for body, outcome in expected.items():
        row = by_body.get(body)
        if outcome == "SORTED":
            valid = bool(
                row and row.get("physical_sort_success") is True
                and row.get("controller_status") == "PLACED_CONTROLLER_CONFIRMED"
                and row.get("controller_class") == row.get("ground_truth_class")
            )
        else:
            valid = bool(row and row.get("controller_status") == outcome
                         and row.get("controller_failure_reasons"))
        if not valid:
            failures.append(body)
    return failures


def _motion_commands_for_skipped_tracks(
    track_records: dict[str, dict[str, Any]], controller_events: list[dict[str, Any]],
) -> int:
    """Count execution commands inside cycles whose selected track ends as safely skipped."""
    skipped = {
        str(track_id) for track_id, record in track_records.items()
        if str(record.get("status", "")).startswith("SKIPPED_")
    }
    if not skipped:
        return 0
    current_track: str | None = None
    count = 0
    for event in sorted(controller_events, key=lambda row: (
        float(row.get("simulation_time_s", math.inf)), int(row.get("event_id", 0)),
    )):
        event_type = event.get("event_type")
        if event_type == "TARGET_SELECTED" and event.get("track_id") is not None:
            current_track = str(event["track_id"])
        elif event_type == "CYCLE_CLOSED":
            current_track = None
        elif event_type == "EXECUTION_COMMAND_ISSUED" and current_track in skipped:
            count += 1
    return count


def _fsm_state_durations(
    controller_events: list[dict[str, Any]], end_time_s: float | None = None,
) -> dict[str, float]:
    entered = sorted(
        (event for event in controller_events if event.get("event_type") == "STATE_ENTERED"),
        key=lambda event: (float(event.get("simulation_time_s", math.inf)),
                           int(event.get("event_id", 0))),
    )
    durations: dict[str, float] = {}
    for index, event in enumerate(entered):
        start = _number(event.get("simulation_time_s"))
        if start is None:
            continue
        next_time = _number(entered[index + 1].get("simulation_time_s")) \
            if index + 1 < len(entered) else end_time_s
        if next_time is None:
            next_time = start
        if next_time is None or next_time < start:
            continue
        state = str(event.get("state", "UNKNOWN"))
        durations[state] = durations.get(state, 0.0) + next_time - start
    return durations


def _match_detections(
    truths: list[dict[str, Any]], detections: list[dict[str, Any]], max_distance_m: float = 0.04,
) -> tuple[list[tuple[int, int, float]], set[int], set[int]]:
    """Maximum-cardinality, minimum-distance assignment for the small camera scene."""
    if not truths or not detections:
        return [], set(range(len(truths))), set(range(len(detections)))
    distances: dict[tuple[int, int], float] = {}
    for truth_index, truth in enumerate(truths):
        xy = truth.get("initial_xy_m")
        if not isinstance(xy, (list, tuple)) or len(xy) != 2:
            continue
        for detection_index, detection in enumerate(detections):
            point = detection.get("xy_base_m")
            if isinstance(point, (list, tuple)) and len(point) == 2:
                distance = math.dist(tuple(map(float, xy)), tuple(map(float, point)))
                if math.isfinite(distance):
                    distances[(truth_index, detection_index)] = distance
    # With at most six model objects, exhaustive assignments are clearer and safer than
    # introducing another dependency for the evaluator-only matching step.
    best: tuple[int, float, tuple[tuple[int, int, float], ...]] = (-1, math.inf, ())
    def visit(ti: int, used: set[int], pairs: list[tuple[int, int, float]]) -> None:
        nonlocal best
        if ti == len(truths):
            candidate = (len(pairs), sum(item[2] for item in pairs), tuple(pairs))
            if candidate[0] > best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                best = candidate
            return
        visit(ti + 1, used, pairs)
        for di in range(len(detections)):
            distance = distances.get((ti, di))
            if di not in used and distance is not None and distance <= max_distance_m:
                used.add(di)
                pairs.append((ti, di, distance))
                visit(ti + 1, used, pairs)
                pairs.pop()
                used.remove(di)
    visit(0, set(), [])
    matched = list(best[2])
    truth_matched = {ti for ti, _, _ in matched}
    detection_matched = {di for _, di, _ in matched}
    return matched, set(range(len(truths))) - truth_matched, set(range(len(detections))) - detection_matched


def _snapshot_perception(
    run_dir: Path, manifest: dict[str, Any], truth: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    snapshots_path = _resolve_output(run_dir, manifest["relative_outputs"]["perception_snapshots"])
    snapshots = _read_jsonl(snapshots_path)
    inputs = sorted((item for item in snapshots if item.get("view") == "input"),
                    key=lambda item: (float(item.get("simulation_time_s", math.inf)),
                                      int(item.get("frame_id", 0))))
    first = inputs[0] if inputs else {"detections": [], "batch_status": "NO_INPUT_SNAPSHOT"}
    truths = list(truth.get("active_objects", []))
    detections = list(first.get("detections", []))
    matches, missing, false_positive = _match_detections(truths, detections)
    object_records: list[dict[str, Any]] = []
    detection_track_to_body: dict[str, str] = {}
    errors_xy: list[float] = []
    errors_yaw: list[float] = []
    for truth_index, detection_index, distance in matches:
        expected, observed = truths[truth_index], detections[detection_index]
        body = str(expected["body_name"])
        track_id = observed.get("track_id")
        if track_id is not None:
            detection_track_to_body[str(track_id)] = body
        yaw_error = None
        if _number(expected.get("yaw_rad")) is not None and _number(observed.get("yaw_base_rad")) is not None:
            # Square objects are invariant under quarter turns; compare on the pi/2 period.
            period = math.pi / 2.0
            delta = (float(observed["yaw_base_rad"]) - float(expected["yaw_rad"]) + period / 2.0) % period - period / 2.0
            yaw_error = abs(delta)
            errors_yaw.append(yaw_error)
        errors_xy.append(distance)
        object_records.append({
            "body_name": body,
            "ground_truth_class": expected.get("class_label"),
            "initial_x_m": expected["initial_xy_m"][0],
            "initial_y_m": expected["initial_xy_m"][1],
            "initial_yaw_rad": expected.get("yaw_rad"),
            "track_id": track_id,
            "detected_class": observed.get("class_label", "UNKNOWN"),
            "color_outcome": (
                "CORRECT" if expected.get("class_label") in COLORS
                and observed.get("class_label") == expected.get("class_label")
                else "UNKNOWN_ABSTENTION" if expected.get("class_label") in COLORS
                and observed.get("class_label") == "UNKNOWN"
                else "MISCLASSIFICATION" if expected.get("class_label") in COLORS
                and observed.get("class_label") in COLORS
                else "NOT_APPLICABLE"
            ),
            "detection_status": observed.get("status"),
            "confidence": observed.get("confidence"),
            "merged_candidate_split": bool(
                (observed.get("diagnostics") or {}).get("split_from_merged_candidate", False)
            ),
            "xy_error_m": distance,
            "yaw_error_rad_symmetry_90deg": yaw_error,
            "frame_id": first.get("frame_id"),
            "simulation_time_s": first.get("simulation_time_s"),
        })
    for truth_index in missing:
        expected = truths[truth_index]
        object_records.append({
            "body_name": expected.get("body_name"),
            "ground_truth_class": expected.get("class_label"),
            "initial_x_m": (expected.get("initial_xy_m") or [None, None])[0],
            "initial_y_m": (expected.get("initial_xy_m") or [None, None])[1],
            "initial_yaw_rad": expected.get("yaw_rad"),
            "track_id": None,
            "detected_class": "MISSING",
            "color_outcome": "MISSING_DETECTION",
            "detection_status": first.get("batch_status"),
            "confidence": None,
            "merged_candidate_split": False,
            "xy_error_m": None,
            "yaw_error_rad_symmetry_90deg": None,
            "frame_id": first.get("frame_id"),
            "simulation_time_s": first.get("simulation_time_s"),
        })
    for detection_index in false_positive:
        detection = detections[detection_index]
        object_records.append({
            "body_name": None,
            "ground_truth_class": "NO_OBJECT",
            "initial_x_m": None,
            "initial_y_m": None,
            "initial_yaw_rad": None,
            "track_id": detection.get("track_id"),
            "detected_class": detection.get("class_label", "UNKNOWN"),
            "color_outcome": "FALSE_POSITIVE",
            "detection_status": detection.get("status"),
            "confidence": detection.get("confidence"),
            "merged_candidate_split": bool(
                (detection.get("diagnostics") or {}).get("split_from_merged_candidate", False)
            ),
            "xy_error_m": None,
            "yaw_error_rad_symmetry_90deg": None,
            "frame_id": first.get("frame_id"),
            "simulation_time_s": first.get("simulation_time_s"),
        })
    perception_summary = {
        "first_input_snapshot_status": first.get("batch_status"),
        "first_input_frame_id": first.get("frame_id"),
        "first_input_snapshot_time_s": _number(first.get("simulation_time_s")),
        "truth_objects": len(truths),
        "first_frame_detection_count": len(detections),
        "first_frame_merged_candidate_split_count": sum(
            bool((detection.get("diagnostics") or {}).get("split_from_merged_candidate", False))
            for detection in detections
        ),
        "first_frame_correct_classification_count": sum(
            row.get("color_outcome") == "CORRECT" for row in object_records
        ),
        "matched_objects": len(matches),
        "missing_objects": len(missing),
        "false_positive_detections": len(false_positive),
        "mean_xy_error_m": statistics.fmean(errors_xy) if errors_xy else None,
        "max_xy_error_m": max(errors_xy) if errors_xy else None,
        "mean_yaw_error_rad": statistics.fmean(errors_yaw) if errors_yaw else None,
        "max_yaw_error_rad": max(errors_yaw) if errors_yaw else None,
    }
    return perception_summary, object_records, detection_track_to_body


def _perception_requirement_failures(
    requirements: dict[str, Any], perception: dict[str, Any],
) -> list[str]:
    """Evaluate preregistered first-frame perception assertions for a trial."""
    if not isinstance(requirements, dict):
        raise ValueError("required_perception_checks must be a mapping")
    failures: list[str] = []
    for metric_name, expected in requirements.items():
        if metric_name not in {
            "first_frame_detection_count",
            "first_frame_merged_candidate_split_count",
            "first_frame_correct_classification_count",
        }:
            raise ValueError(f"Unsupported preregistered perception check: {metric_name}")
        if isinstance(expected, bool) or not isinstance(expected, int) or expected < 0:
            raise ValueError(f"Perception check {metric_name} must be a nonnegative integer")
        actual = perception.get(metric_name)
        if actual != expected:
            failures.append(f"{metric_name}: expected {expected}, observed {actual}")
    return failures


def _contact_metrics(
    run_dir: Path, manifest: dict[str, Any], track_to_body: dict[str, str],
    controller_events: list[dict[str, Any]],
) -> dict[str, Any]:
    path = _resolve_output(run_dir, manifest["relative_outputs"]["evaluator_contact_events"])
    events = _read_jsonl(path)
    counts = {key: 0 for key in ("object_support", "tray_support", "tray_boundary", "intended_grasp", "forbidden")}
    max_penetration = 0.0
    max_force = 0.0
    for event in events:
        target_body, controller_state = _controller_target_at(
            controller_events, track_to_body, float(event.get("start_time_s", 0.0)),
        )
        category = classify_contact_pair(
            event.get("geom1", ""), event.get("geom2", ""),
            target_body=target_body, controller_state=controller_state,
        )
        counts[category] += 1
        minimum = _number(event.get("minimum_distance_m"))
        if minimum is not None:
            max_penetration = max(max_penetration, max(0.0, -minimum))
        normal = _number(event.get("maximum_normal_force_N")) or 0.0
        tangent = _number(event.get("maximum_tangent_force_N")) or 0.0
        max_force = max(max_force, normal, tangent)
    return {
        "contact_episode_count": len(events),
        "contact_object_support_episodes": counts["object_support"],
        "contact_tray_support_episodes": counts["tray_support"],
        "contact_tray_boundary_episodes": counts["tray_boundary"],
        "contact_intended_grasp_episodes": counts["intended_grasp"],
        "forbidden_contact_episodes": counts["forbidden"],
        "maximum_contact_penetration_m": max_penetration,
        "maximum_contact_force_N": max_force,
        "contact_sampling_hz": round(1.0 / float(manifest["physics"]["timestep_s"])),
    }


def _motion_metrics(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    telemetry = _read_csv(_resolve_output(run_dir, manifest["relative_outputs"]["controller_telemetry"]))
    events = _read_jsonl(_resolve_output(run_dir, manifest["relative_outputs"]["controller_events"]))
    xys: list[tuple[float, float, float]] = []
    path_length = 0.0
    last_xyz: tuple[float, float, float] | None = None
    max_joint_error = 0.0
    joint_path = {name: 0.0 for name in JOINTS}
    last_joint: dict[str, float] = {}
    max_limit_excess = 0.0
    position_violation_count = 0
    force_violation_count = 0
    model_path = run_dir / "snapshots" / "m7_integrated_model.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    joint_ranges: dict[str, tuple[float, float]] = {}
    for name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or not bool(model.jnt_limited[joint_id]):
            raise ValueError(f"Could not verify model limit for joint {name}")
        joint_ranges[name] = tuple(map(float, model.jnt_range[joint_id]))
    force_ranges: dict[str, tuple[float, float]] = {}
    for actuator_name, _column in ACTUATORS:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0 or not bool(model.actuator_forcelimited[actuator_id]):
            raise ValueError(f"Could not verify force limit for actuator {actuator_name}")
        force_ranges[actuator_name] = tuple(map(float, model.actuator_forcerange[actuator_id]))

    for row in telemetry:
        xyz_values = tuple(_number(row.get(f"estimated_tcp_{axis}_m")) for axis in "xyz")
        if all(value is not None for value in xyz_values):
            xyz = tuple(float(value) for value in xyz_values)
            if last_xyz is not None:
                path_length += math.dist(last_xyz, xyz)
            last_xyz = xyz
            time_s = _number(row.get("simulation_time_s"))
            if time_s is not None:
                xys.append((time_s, xyz[0], xyz[1]))
        peak = _number(row.get("peak_abs_joint_error"))
        if peak is not None:
            max_joint_error = max(max_joint_error, peak)
        for name in JOINTS:
            # Telemetry column names use j1..j4 aliases, while model names retain full names.
            alias = {"j1_shoulder": "j1", "j2_elbow": "j2", "j3_lift": "j3", "j4_wrist": "j4"}[name]
            actual = _number(row.get(f"actual_{alias}"))
            if actual is None:
                continue
            if name in last_joint:
                joint_path[name] += abs(actual - last_joint[name])
            last_joint[name] = actual
            low, high = joint_ranges[name]
            excess = max(low - actual, actual - high, 0.0)
            if excess > 1e-6:
                position_violation_count += 1
                max_limit_excess = max(max_limit_excess, excess)
        for actuator_name, column in ACTUATORS:
            actual_force = _number(row.get(column))
            if actual_force is None:
                continue
            low, high = force_ranges[actuator_name]
            excess = max(low - actual_force, actual_force - high, 0.0)
            if excess > 1e-6:
                force_violation_count += 1
                max_limit_excess = max(max_limit_excess, excess)

    state_entries = [event for event in events if event.get("event_type") == "STATE_ENTERED"]
    motion_commands = sum(event.get("event_type") == "EXECUTION_COMMAND_ISSUED" for event in events)
    transitions = sum(event.get("event_type") == "STATE_TRANSITION" for event in events)
    return {
        "tcp_path_length_m": path_length,
        "tcp_xy_samples": xys,
        "joint_path_abs_rad_or_m": joint_path,
        "max_peak_joint_tracking_error_rad_or_m": max_joint_error,
        "position_limit_violation_samples": position_violation_count,
        "force_limit_violation_samples": force_violation_count,
        "max_limit_excess_in_channel_unit": max_limit_excess,
        "execution_command_count": motion_commands,
        "fsm_state_entries": len(state_entries),
        "fsm_transitions": transitions,
        "state_entries": state_entries,
        "telemetry_rows": len(telemetry),
        "event_rows": len(events),
    }


def _plan_metrics(run_dir: Path) -> dict[str, Any]:
    plan_dir = run_dir / "controller" / "plans"
    plans: list[dict[str, Any]] = []
    if plan_dir.is_dir():
        for path in sorted(plan_dir.glob("m5_plan_*.json*")):
            document = _read_json(path)
            plan = document.get("plan") or {}
            plans.append({
                "file": path.name,
                "code": document.get("code"),
                "reason": document.get("reason"),
                "phase": document.get("phase"),
                "minimum_clearance_m": plan.get("minimum_clearance_m"),
                "clearance_lower_bound_m": plan.get("clearance_lower_bound_m"),
                "minimum_clearance_phase": plan.get("minimum_clearance_phase"),
                "collision_sample_step_m": plan.get("collision_sample_step_m"),
                "duration_s": plan.get("duration_s"),
                "path_length_m": plan.get("path_length_m"),
                "selected_ik_branch": plan.get("selected_ik_branch"),
            })
    accepted = [plan for plan in plans if plan.get("code") == "SUCCESS"]
    lower_bounds = [_number(plan.get("clearance_lower_bound_m")) for plan in accepted]
    lower_bounds = [value for value in lower_bounds if value is not None]
    return {
        "plan_request_count": len(plans),
        "plan_accepted_count": len(accepted),
        "plan_rejected_count": sum(plan.get("code") != "SUCCESS" for plan in plans),
        "min_accepted_plan_clearance_lower_bound_m": min(lower_bounds) if lower_bounds else None,
        "accepted_plans": accepted,
        "all_plans": plans,
    }


def _trial_metrics(run_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    trial = _read_json(run_dir / "campaign_trial.json")
    manifest = _read_json(run_dir / "manifest.json")
    report_path = _resolve_output(run_dir, manifest["relative_outputs"]["evaluator_report"])
    report = _read_json(report_path)
    truth = _read_json(run_dir / "evaluator_private" / "scene_truth.json")
    perception, object_records, track_to_body = _snapshot_perception(run_dir, manifest, truth)
    perception_requirement_failures = _perception_requirement_failures(
        trial.get("required_perception_checks", {}), perception,
    )
    item_by_body = {item.get("body_name"): item for item in report.get("items", [])}
    controller = manifest.get("controller_summary", {})
    track_records = report.get("controller_track_records") or controller.get("track_records", {})
    slip_by_body = report.get("evaluator_grasp_slip", {})
    for row in object_records:
        body = row.get("body_name")
        item = item_by_body.get(body, {})
        mapped = None
        if row.get("track_id") is not None:
            mapped = track_records.get(str(row["track_id"]))
        row.update({
            "controller_status": (mapped or {}).get("status"),
            "controller_class": (mapped or {}).get("class_label"),
            "controller_failure_reasons": ";".join((mapped or {}).get("failure_reasons", [])),
            "final_x_m": (item.get("final_xyz_m") or [None, None, None])[0],
            "final_y_m": (item.get("final_xyz_m") or [None, None, None])[1],
            "final_z_m": (item.get("final_xyz_m") or [None, None, None])[2],
            "final_zone": item.get("nearest_zone"),
            "final_slot": item.get("nearest_slot_id"),
            "settled": item.get("settled"),
            "resting_on_tray": item.get("resting_on_tray"),
            "physical_sort_success": item.get("physical_sort_success", False),
            "horizontal_slot_error_m": item.get("horizontal_slot_error_m"),
            "controller_attempts": (mapped or {}).get("attempts"),
            "controller_grasp_attempts": (mapped or {}).get("grasp_attempts"),
            "bilateral_contact_samples": (slip_by_body.get(body) or {}).get("bilateral_contact_samples"),
            "first_bilateral_contact_time_s": (slip_by_body.get(body) or {}).get("first_bilateral_contact_time_s"),
            "maximum_relative_slip_m": (slip_by_body.get(body) or {}).get("maximum_relative_slip_m"),
        })
    controller_events = _read_jsonl(
        _resolve_output(run_dir, manifest["relative_outputs"]["controller_events"])
    )
    contacts = _contact_metrics(run_dir, manifest, track_to_body, controller_events)
    motion = _motion_metrics(run_dir, manifest)
    plans = _plan_metrics(run_dir)
    true_items = list(report.get("items", []))
    sorted_items = [item for item in true_items if item.get("physical_sort_success") is True]
    wrong_bin_items = [
        item for item in true_items
        if item.get("nearest_zone") is not None
        and item.get("nearest_zone") != item.get("ground_truth_class")
    ]
    track_statuses = [str(record.get("status", "")) for record in track_records.values()]
    skipped_count = sum(status.startswith("SKIPPED") for status in track_statuses)
    skipped_target_motion_commands = _motion_commands_for_skipped_tracks(
        track_records, controller_events,
    )
    confirmation_failures = _placement_confirmation_failures(track_records, object_records)
    state = str(controller.get("final_state", "UNKNOWN"))
    expected_kind = str(trial["expected_kind"])
    expected_reason = trial.get("expected_reason")
    terminal_reason = (controller.get("terminal_result") or {}).get("reason_code")
    forbidden = contacts["forbidden_contact_episodes"]
    limits = motion["position_limit_violation_samples"] + motion["force_limit_violation_samples"]
    all_safe = forbidden == 0 and limits == 0 and not wrong_bin_items and not confirmation_failures
    sort_all_pass = (
        bool(true_items) and len(sorted_items) == len(true_items)
        and state == "DONE" and report.get("controller_done_and_physical_success") is True
    )
    if expected_kind == "sort_all":
        test_pass = sort_all_pass and all_safe
    elif expected_kind == "safe_completion":
        test_pass = state == "DONE" and all_safe and _object_outcomes_accounted(object_records)
    elif expected_kind == "safe_skip":
        expected_status = trial.get("expected_track_status")
        status_ok = bool(track_records) and all(
            status.startswith("SKIPPED")
            and (expected_status is None or status == expected_status)
            for status in track_statuses
        )
        reason = trial.get("expected_failure_reason")
        reason_ok = reason is None or all(
            reason in record.get("failure_reasons", []) for record in track_records.values()
        )
        object_failures = _expected_object_outcome_failures(
            trial.get("expected_object_outcomes", {}), object_records,
        )
        test_pass = (
            bool(true_items) and state == "DONE" and status_ok and reason_ok
            and skipped_count == len(true_items) and not sorted_items
            and not object_failures
            and perception["false_positive_detections"] == 0
            and motion["execution_command_count"] == 0 and all_safe
        )
    elif expected_kind == "no_task":
        test_pass = not true_items and state == "DONE" and motion["execution_command_count"] == 0 and all_safe
    elif expected_kind == "safe_stop":
        test_pass = (
            state == "SAFE_STOP" and motion["execution_command_count"] == 0
            and (expected_reason is None or terminal_reason == expected_reason) and all_safe
        )
    elif expected_kind == "safe_stop_no_false_confirmation":
        test_pass = (
            state == "SAFE_STOP" and not report.get("controller_done_and_physical_success", False)
            and bool(terminal_reason)
            and not confirmation_failures
            and all(record.get("status") != "PLACED_CONTROLLER_CONFIRMED"
                    for record in track_records.values()) and all_safe
        )
    elif expected_kind == "mixed_partial":
        object_failures = _expected_object_outcome_failures(
            trial.get("expected_object_outcomes", {}), object_records,
        )
        test_pass = (
            state == "DONE" and len(sorted_items) >= 1 and not object_failures
            and all_safe and _object_outcomes_accounted(object_records)
            and skipped_target_motion_commands == 0
        )
    elif expected_kind == "contained_fault":
        fault_contained = (
            state == "SAFE_STOP" and bool(terminal_reason)
        ) or (
            state == "DONE" and _object_outcomes_accounted(object_records, allow_safe_failure=True)
        )
        test_pass = (
            fault_contained and all_safe
            and not confirmation_failures
            and any(value not in (None, False, 0, 0.0, "")
                    for value in manifest.get("fault_injection", {}).values())
        )
    else:
        raise ValueError(f"Unknown expected_kind {expected_kind!r} for {trial['trial_id']}")
    test_pass = bool(test_pass and not perception_requirement_failures)

    performance = controller.get("performance", {})
    object_count = len(true_items)
    matched_color_rows = [row for row in object_records if row.get("ground_truth_class") in COLORS]
    color_correct_count = sum(row.get("detected_class") == row.get("ground_truth_class")
                              for row in matched_color_rows)
    unknown_abstentions = sum(row.get("detected_class") == "UNKNOWN" for row in matched_color_rows)
    misclassifications = sum(row.get("detected_class") in COLORS
                             and row.get("detected_class") != row.get("ground_truth_class")
                             for row in matched_color_rows)
    attempt_count = sum(int(record.get("attempts", 0)) for record in track_records.values())
    grasp_attempt_count = sum(int(record.get("grasp_attempts", 0)) for record in track_records.values())
    bilateral_objects = [
        value for value in slip_by_body.values()
        if int(value.get("bilateral_contact_samples") or 0) > 0
    ]
    slip_values = [_number(value.get("maximum_relative_slip_m"))
                   for value in bilateral_objects]
    slip_values = [value for value in slip_values if value is not None]
    state_durations = _fsm_state_durations(
        controller_events, _number(report.get("simulation_time_s")),
    )
    placement_times = [
        float(event["simulation_time_s"]) for event in controller_events
        if event.get("event_type") == "PLACEMENT_CONTROLLER_CONFIRMED"
        and _number(event.get("simulation_time_s")) is not None
    ]
    metric = {
        "trial_id": trial["trial_id"],
        "experiment_id": trial["experiment_id"],
        "variant_id": trial["variant_id"],
        "tags": trial.get("tags", []),
        "run_id": manifest["run_id"],
        "scenario_id": manifest["scenario_id"],
        "scenario_seed": manifest.get("scenario_seed"),
        "expected_kind": expected_kind,
        "test_pass": bool(test_pass),
        "final_state": state,
        "terminal_reason": terminal_reason,
        "expected_reason": expected_reason,
        "controller_reports_done": report.get("controller_reports_done"),
        "controller_confirmed_tracks": report.get("controller_confirmed_track_count"),
        "controller_false_confirmation_tracks": ";".join(confirmation_failures),
        "active_objects": object_count,
        "sorted_correctly": len(sorted_items),
        "wrong_bin_count": len(wrong_bin_items),
        "unplaced_or_unsettled_count": object_count - len(sorted_items),
        "skipped_track_count": skipped_count,
        "execution_commands_for_skipped_targets": skipped_target_motion_commands,
        "controller_attempt_count": attempt_count,
        "controller_grasp_attempt_count": grasp_attempt_count,
        "hold_confirmed_event_count": sum(event.get("event_type") == "HOLD_CONFIRMED"
                                           for event in controller_events),
        "placement_confirmed_event_count": len(placement_times),
        "time_to_first_placement_confirmation_s": min(placement_times) if placement_times else None,
        "fsm_state_durations_s": state_durations,
        "simulation_time_s": report.get("simulation_time_s"),
        "wall_time_s": manifest.get("wall_clock_elapsed_s"),
        "simulation_to_wall_ratio": performance.get("simulation_to_wall_time_ratio"),
        "tcp_path_length_m": motion["tcp_path_length_m"],
        "joint_path_abs_rad_or_m": motion["joint_path_abs_rad_or_m"],
        "max_joint_tracking_error_rad_or_m": motion["max_peak_joint_tracking_error_rad_or_m"],
        "position_limit_violation_samples": motion["position_limit_violation_samples"],
        "force_limit_violation_samples": motion["force_limit_violation_samples"],
        "max_limit_excess_in_channel_unit": motion["max_limit_excess_in_channel_unit"],
        "execution_command_count": motion["execution_command_count"],
        "fsm_state_entries": motion["fsm_state_entries"],
        "fsm_transitions": motion["fsm_transitions"],
        "perception_truth_objects": perception["truth_objects"],
        "perception_first_frame_detection_count": perception["first_frame_detection_count"],
        "perception_first_frame_merged_split_count": perception[
            "first_frame_merged_candidate_split_count"
        ],
        "perception_first_frame_correct_classification_count": perception[
            "first_frame_correct_classification_count"
        ],
        "perception_requirement_failures": perception_requirement_failures,
        "perception_matched_objects": perception["matched_objects"],
        "perception_missing_objects": perception["missing_objects"],
        "perception_false_positives": perception["false_positive_detections"],
        "perception_color_correct_count": color_correct_count,
        "perception_unknown_abstention_count": unknown_abstentions,
        "perception_misclassification_count": misclassifications,
        "perception_missing_detection_count": perception["missing_objects"],
        "perception_first_view_time_s": perception.get("first_input_snapshot_time_s"),
        "perception_mean_xy_error_m": perception["mean_xy_error_m"],
        "perception_max_xy_error_m": perception["max_xy_error_m"],
        "perception_mean_yaw_error_rad": perception["mean_yaw_error_rad"],
        "perception_max_yaw_error_rad": perception["max_yaw_error_rad"],
        "first_input_batch_status": perception["first_input_snapshot_status"],
        "contact_episode_count": contacts["contact_episode_count"],
        "object_support_contact_episodes": contacts["contact_object_support_episodes"],
        "tray_support_contact_episodes": contacts["contact_tray_support_episodes"],
        "tray_boundary_contact_episodes": contacts["contact_tray_boundary_episodes"],
        "intended_grasp_contact_episodes": contacts["contact_intended_grasp_episodes"],
        "forbidden_contact_episodes": forbidden,
        "maximum_contact_penetration_m": contacts["maximum_contact_penetration_m"],
        "maximum_contact_force_N": contacts["maximum_contact_force_N"],
        "contact_sampling_hz": contacts["contact_sampling_hz"],
        "objects_with_bilateral_grasp_contact": len(bilateral_objects),
        "maximum_relative_slip_m": max(slip_values) if slip_values else None,
        "m5_plan_requests": plans["plan_request_count"],
        "m5_plans_accepted": plans["plan_accepted_count"],
        "m5_plans_rejected": plans["plan_rejected_count"],
        "min_accepted_plan_clearance_lower_bound_m": plans["min_accepted_plan_clearance_lower_bound_m"],
        "rgb_noise_sigma_8bit": manifest.get("m9_public_input_conditions", {}).get("rgb_noise_sigma_8bit", 0.0),
        "lighting_scale": manifest.get("m9_public_input_conditions", {}).get("lighting_scale", 1.0),
        "rgb_gain": manifest.get("m9_public_input_conditions", {}).get("rgb_gain", 1.0),
        "calibration_bias_x_m": (manifest.get("m9_public_input_conditions", {}).get("input_calibration_bias_xy_m") or [0, 0])[0],
        "calibration_bias_y_m": (manifest.get("m9_public_input_conditions", {}).get("input_calibration_bias_xy_m") or [0, 0])[1],
        "gripper_friction_scale": manifest.get("fault_injection", {}).get("gripper_friction_scale"),
        "fault_injection": manifest.get("fault_injection", {}),
        "input_hashes_sha256": manifest.get("input_hashes_sha256", {}),
    }
    for record in object_records:
        record.update({
            "trial_id": trial["trial_id"],
            "run_id": manifest["run_id"],
            "experiment_id": trial["experiment_id"],
            "variant_id": trial["variant_id"],
        })
    metric["_truth"] = truth
    metric["_perception"] = perception
    metric["_object_records"] = object_records
    metric["_contacts"] = contacts
    metric["_motion"] = motion
    metric["_plans"] = plans
    metric["_report"] = report
    metric["_manifest"] = manifest
    metric["_run_dir"] = str(run_dir)
    return metric, object_records


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def _series_summary(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for metric in metrics:
        groups.setdefault((metric["experiment_id"], metric["variant_id"]), []).append(metric)
    summaries: list[dict[str, Any]] = []
    for (experiment, variant), rows in sorted(groups.items()):
        n = len(rows)
        successes = sum(bool(row["test_pass"]) for row in rows)
        lower, upper = wilson_interval(successes, n)
        object_n = sum(int(row["active_objects"]) for row in rows)
        sorted_n = sum(int(row["sorted_correctly"]) for row in rows)
        sum_s = [float(row["simulation_time_s"]) for row in rows if _number(row["simulation_time_s"]) is not None]
        wall_s = [float(row["wall_time_s"]) for row in rows if _number(row["wall_time_s"]) is not None]
        summaries.append({
            "experiment_id": experiment,
            "variant_id": variant,
            "runs": n,
            "test_passes": successes,
            "test_pass_rate": successes / n if n else None,
            "test_pass_rate_wilson_95_low": lower,
            "test_pass_rate_wilson_95_high": upper,
            "active_objects": object_n,
            "correctly_sorted_objects": sorted_n,
            "object_sort_rate": sorted_n / object_n if object_n else None,
            "wrong_bin_count": sum(int(row["wrong_bin_count"]) for row in rows),
            "forbidden_contact_episodes": sum(int(row["forbidden_contact_episodes"]) for row in rows),
            "joint_force_limit_violation_samples": sum(
                int(row["position_limit_violation_samples"]) + int(row["force_limit_violation_samples"])
                for row in rows
            ),
            "median_simulation_time_s": statistics.median(sum_s) if sum_s else None,
            "median_wall_time_s": statistics.median(wall_s) if wall_s else None,
            "median_tcp_path_length_m": statistics.median(float(row["tcp_path_length_m"]) for row in rows),
            "minimum_sampled_plan_clearance_lower_bound_m": min(
                (float(row["min_accepted_plan_clearance_lower_bound_m"]) for row in rows
                 if _number(row["min_accepted_plan_clearance_lower_bound_m"]) is not None),
                default=None,
            ),
        })
    return summaries


def _paired_comparisons(metrics: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair registered perturbations by the common scene/noise seed, not object count."""
    specifications = (
        ("V07_RGB_NOISE", "sigma_0", "V07_RGB_NOISE", "sigma_4", "rgb_noise_sigma_4"),
        ("V07_RGB_NOISE", "sigma_0", "V07_RGB_NOISE", "sigma_8", "rgb_noise_sigma_8"),
        ("V07_RGB_NOISE", "sigma_0", "V10_GRIP_FRICTION", "scale_0_001", "gripper_friction_scale_0_001"),
    )
    indexed = {
        (row["experiment_id"], row["variant_id"], int(row["scenario_seed"])): row
        for row in metrics if row.get("scenario_seed") is not None
    }
    pair_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for base_experiment, base_variant, test_experiment, test_variant, label in specifications:
        differences: list[int] = []
        sort_differences: list[int] = []
        paired = 0
        base_passes = 0
        test_passes = 0
        for (experiment, variant, seed), base in sorted(indexed.items()):
            if (experiment, variant) != (base_experiment, base_variant):
                continue
            test = indexed.get((test_experiment, test_variant, seed))
            if test is None:
                continue
            paired += 1
            base_passes += bool(base["test_pass"])
            test_passes += bool(test["test_pass"])
            delta_pass = int(bool(test["test_pass"])) - int(bool(base["test_pass"]))
            delta_sorted = int(test["sorted_correctly"]) - int(base["sorted_correctly"])
            differences.append(delta_pass)
            sort_differences.append(delta_sorted)
            pair_rows.append({
                "comparison": label,
                "seed": seed,
                "control_trial_id": base["trial_id"],
                "test_trial_id": test["trial_id"],
                "control_test_pass": base["test_pass"],
                "test_test_pass": test["test_pass"],
                "test_pass_difference": delta_pass,
                "control_sorted_objects": base["sorted_correctly"],
                "test_sorted_objects": test["sorted_correctly"],
                "sorted_object_difference": delta_sorted,
                "control_final_state": base["final_state"],
                "test_final_state": test["final_state"],
            })
        summaries.append({
            "comparison": label,
            "paired_seed_runs": paired,
            "control_passes": base_passes,
            "test_passes": test_passes,
            "mean_test_pass_difference": statistics.fmean(differences) if differences else None,
            "mean_sorted_object_difference": statistics.fmean(sort_differences) if sort_differences else None,
            "discordant_pairs": sum(value != 0 for value in differences),
        })
    return summaries, pair_rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(
            _json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return ""
    return value


def _verify_trial_integrity(
    run_dir: Path, campaign_path: Path, expected: dict[str, Any],
) -> None:
    trial = _read_json(run_dir / "campaign_trial.json")
    manifest_path = run_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    campaign_hash = _sha256(campaign_path)
    if trial.get("campaign_sha256") != campaign_hash:
        raise ValueError(f"Campaign hash mismatch in {run_dir.name}")
    for key in (
        "trial_id", "experiment_id", "variant_id", "scenario", "scenario_dir", "seed",
        "expected_kind", "expected_reason", "expected_track_status",
        "expected_failure_reason", "expected_object_outcomes",
        "required_perception_checks", "tags", "parameters",
        "paired_control",
    ):
        if trial.get(key) != expected.get(key):
            raise ValueError(f"Registered campaign field {key!r} differs for {run_dir.name}")
    if (trial.get("run_id") != run_dir.name or manifest.get("run_id") != run_dir.name
            or trial.get("run_manifest_sha256") != _sha256(manifest_path)):
        raise ValueError(f"Run identity or manifest hash mismatch in {run_dir.name}")
    if manifest.get("scenario_id") != expected.get("scenario"):
        raise ValueError(f"Scenario ID mismatch in {run_dir.name}")
    if expected.get("seed") is not None and manifest.get("scenario_seed") != expected["seed"]:
        raise ValueError(f"Seed mismatch in {run_dir.name}")
    if trial.get("input_hashes_sha256") != manifest.get("input_hashes_sha256"):
        raise ValueError(f"Input hashes differ between trial record and manifest in {run_dir.name}")
    input_hashes = manifest.get("input_hashes_sha256", {})
    if set(input_hashes) != set(SNAPSHOT_INPUT_FILES):
        raise ValueError(f"Unexpected simulator input hash set in {run_dir.name}")
    for key, filename in SNAPSHOT_INPUT_FILES.items():
        snapshot = run_dir / "snapshots" / filename
        if not snapshot.is_file() or _input_snapshot_sha256(key, snapshot) != input_hashes[key]:
            raise ValueError(f"Simulator input snapshot checksum mismatch for {key} in {run_dir.name}")

    recorded_hashes = trial.get("evidence_hashes_sha256")
    if not isinstance(recorded_hashes, dict) or not recorded_hashes:
        raise ValueError(f"Run has no evidence hashes: {run_dir.name}")
    compressed = trial.get("compressed_evidence", {})
    for relative, expected_hash in recorded_hashes.items():
        raw_path = (run_dir / relative).resolve()
        if not raw_path.is_relative_to(run_dir.resolve()):
            raise ValueError(f"Evidence path escapes run directory: {relative}")
        if raw_path.is_file():
            observed_hash = _sha256(raw_path)
        else:
            gzip_path = raw_path.with_name(raw_path.name + ".gz")
            if not gzip_path.is_file():
                raise FileNotFoundError(f"Missing raw or compressed evidence: {raw_path}")
            observed_hash = _sha256(gzip_path, decompress_gzip=True)
            compressed_record = compressed.get(relative)
            if not compressed_record or compressed_record.get("path") != gzip_path.relative_to(run_dir).as_posix():
                raise ValueError(f"Compressed evidence metadata mismatch for {relative}")
            if compressed_record.get("compressed_sha256") != _sha256(gzip_path):
                raise ValueError(f"Compressed evidence checksum mismatch for {relative}")
        if observed_hash != expected_hash:
            raise ValueError(f"Evidence checksum mismatch for {relative} in {run_dir.name}")


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _draw_rate_plot(rows: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in rows if row["runs"] >= 2]
    width, height = 1600, 920
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label = _font(19)
    small = _font(16)
    left, right, top, bottom = 475, 1490, 90, 770
    for tick in range(0, 11):
        value = tick / 10
        x = left + int(value * (right - left))
        draw.line((x, top, x, bottom), fill="#D8E0E4", width=1)
        draw.text((x - 16, bottom + 12), f"{value:.0%}", font=small, fill="#445761")
    row_height = min(78, (bottom - top) // max(1, len(rows)))
    for index, row in enumerate(rows):
        center_y = top + index * row_height + row_height // 2
        y0, y1 = center_y - 19, center_y + 19
        name = f"{row['experiment_id']} / {row['variant_id']}"
        draw.text((60, center_y - 11), name[:39], font=label, fill="#213843")
        rate = float(row["test_pass_rate"] or 0.0)
        x1 = left + int(rate * (right - left))
        draw.rounded_rectangle((left, y0, x1, y1), radius=7, fill="#E7B83E")
        low = float(row["test_pass_rate_wilson_95_low"] or 0.0)
        high = float(row["test_pass_rate_wilson_95_high"] or 0.0)
        ci_left = left + int(low * (right - left))
        ci_right = left + int(high * (right - left))
        draw.line((ci_left, (y0 + y1) // 2, ci_right, (y0 + y1) // 2), fill="#233D4A", width=3)
        draw.line((ci_left, (y0 + y1) // 2 - 6, ci_left, (y0 + y1) // 2 + 6), fill="#233D4A", width=2)
        draw.line((ci_right, (y0 + y1) // 2 - 6, ci_right, (y0 + y1) // 2 + 6), fill="#233D4A", width=2)
        draw.text((min(x1 + 12, right - 62), y0 + 3), f"{row['test_passes']}/{row['runs']}", font=small, fill="#162A35")
    draw.text((left, 835), "Полоса — доля прогонов, выполнивших заранее заданный исход; усики — 95% интервал Уилсона.",
              font=small, fill="#445761")
    draw.text((left, 862), "Для V10 ожидаемый исход — безопасное ограничение отказа, а не успешная сортировка.",
              font=small, fill="#445761")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _draw_confusion_matrix(matrix: dict[tuple[str, str], int], path: Path) -> None:
    rows = (*COLORS, "NO_OBJECT")
    columns = PREDICTION_LABELS
    cell_w, cell_h = 205, 110
    left, top = 250, 145
    image = Image.new("RGB", (left + cell_w * len(columns) + 80, top + cell_h * (len(rows) + 1) + 70), "white")
    draw = ImageDraw.Draw(image)
    label = _font(20)
    small = _font(18)
    draw.text((left, top - 72), "Метка первого кадра →", font=small, fill="#445761")
    draw.text((65, top - 4), "Истинный класс", font=small, fill="#445761")
    draw.text((left, top + cell_h * len(rows) + 18),
              "Строка NO_OBJECT показывает несопоставленные детекции; столбцы — метки первого кадра.",
              font=small, fill="#445761")
    for ci, column in enumerate(columns):
        draw.text((left + ci * cell_w + 20, top - 44), column, font=label, fill="#213843")
    maximum = max(matrix.values(), default=1) or 1
    for ri, row in enumerate(rows):
        draw.text((65, top + ri * cell_h + 38), row, font=label, fill="#213843")
        for ci, column in enumerate(columns):
            count = matrix.get((row, column), 0)
            strength = int(245 - 110 * count / maximum)
            fill = (strength, min(244, strength + 10), min(248, strength + 14))
            x0, y0 = left + ci * cell_w, top + ri * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 4, y0 + cell_h - 4), fill=fill, outline="#C6D3D9")
            draw.text((x0 + cell_w // 2 - 10, y0 + cell_h // 2 - 14), str(count), font=label, fill="#132C38")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _draw_trajectory(
    metric: dict[str, Any], path: Path, ssot: dict[str, Any], *, outcome_label: str,
) -> None:
    width, height = 1400, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label, small = _font(18), _font(15)
    xmin, xmax, ymin, ymax = -0.52, 0.52, -0.44, 0.44
    l, r, t, b = 130, 1320, 70, 1000
    def xy(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        return (int(l + (x - xmin) / (xmax - xmin) * (r - l)),
                int(b - (y - ymin) / (ymax - ymin) * (b - t)))
    for x in np.arange(xmin, xmax + 1e-9, 0.1):
        xx, _ = xy((float(x), ymin))
        draw.line((xx, t, xx, b), fill="#EEF1F2", width=1)
        draw.text((xx - 15, b + 8), f"{x:.1f}", font=small, fill="#566972")
    for y in np.arange(ymin, ymax + 1e-9, 0.1):
        _, yy = xy((xmin, float(y)))
        draw.line((l, yy, r, yy), fill="#EEF1F2", width=1)
        draw.text((l - 48, yy - 8), f"{y:.1f}", font=small, fill="#566972")
    draw.text((l + (r - l) // 2 - 18, b + 24), "x, м", font=small, fill="#566972")
    draw.text((l - 42, t - 20), "y, м", font=small, fill="#566972")
    trays = ssot["parameters"]["cell.tray_centers_xy_m"]["value"]
    offsets = ssot["parameters"]["cell.tray_slot_x_offsets_m"]["value"]
    tray_y_offset = float(ssot["parameters"]["cell.tray_slot_y_offset_m"]["value"])
    palette = {"RED": "#C6483A", "GREEN": "#2C8A62", "BLUE": "#3478B8"}
    for color in COLORS:
        cx, cy = map(float, trays[color])
        # Show tray extent from the SSOT rather than a hand-drawn decorative icon.
        sx, sy = map(float, ssot["parameters"]["cell.tray_outer_size_xy_m"]["value"])
        p0, p1 = xy((cx - sx / 2, cy - sy / 2)), xy((cx + sx / 2, cy + sy / 2))
        draw.rectangle((p0[0], p1[1], p1[0], p0[1]), outline=palette[color], width=3)
        draw.text((p0[0], p0[1] + 6), color, font=small, fill=palette[color])
        for offset in offsets:
            slot = xy((cx + float(offset), cy + tray_y_offset))
            draw.ellipse((slot[0] - 4, slot[1] - 4, slot[0] + 4, slot[1] + 4), fill=palette[color])
    truth = metric["_truth"]
    report_items = {item.get("body_name"): item for item in metric["_report"].get("items", [])}
    for item in truth.get("active_objects", []):
        color = item["class_label"]
        point = xy(tuple(map(float, item["initial_xy_m"])))
        draw.ellipse((point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8),
                     fill=palette[color], outline="#172C36", width=2)
        final = report_items.get(item["body_name"], {}).get("final_xyz_m")
        if final and len(final) >= 2:
            end = xy((float(final[0]), float(final[1])))
            draw.line((point[0], point[1], end[0], end[1]), fill=palette[color], width=2)
            draw.ellipse((end[0] - 5, end[1] - 5, end[0] + 5, end[1] + 5), fill=palette[color])
    path_points = [(xy((x, y))) for _, x, y in metric["_motion"]["tcp_xy_samples"]]
    if len(path_points) > 1:
        # Thin dense traces to keep the exported chart readable.
        stride = max(1, len(path_points) // 600)
        draw.line(path_points[::stride] + [path_points[-1]], fill="#293F4A", width=2)
    draw.text((l, 1070), "TCP по телеметрии; цветные отрезки соединяют начальные и конечные позиции объектов по оценщику.",
              font=small, fill="#445761")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _draw_joint_error(metric: dict[str, Any], path: Path) -> None:
    run_dir = Path(metric["_run_dir"])
    manifest = metric["_manifest"]
    rows = _read_csv(_resolve_output(run_dir, manifest["relative_outputs"]["controller_telemetry"]))
    times = [float(value) for row in rows
             if (value := _number(row.get("simulation_time_s"))) is not None]
    max_t = max(times, default=1.0) or 1.0
    angular = [
        (f"J{joint}", f"peak_abs_error_j{joint}", color)
        for joint, color in ((1, "#2F7792"), (2, "#D77B32"), (4, "#5D7C4A"))
    ]
    vertical = [("J3 · ползун", "peak_abs_error_j3", "#7655A5")]
    width, height = 1500, 790
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    label, small = _font(18), _font(16)
    left, right = 135, 1430
    panels = [
        ("Поворотные оси J1/J2/J4, рад", angular, 75, 325, 1.0),
        ("Вертикальный ползун J3, мм", vertical, 395, 645, 1000.0),
    ]
    for panel_title, series, top, bottom, unit_scale in panels:
        draw.text((left, top - 32), panel_title, font=label, fill="#213843")
        max_y = max((
            (value or 0.0) * unit_scale
            for row in rows for _, field, _ in series
            if (value := _number(row.get(field))) is not None
        ), default=0.001)
        max_y = max(max_y * 1.12, 0.001)
        for tick in range(5):
            ratio = tick / 4
            y = bottom - int(ratio * (bottom - top))
            draw.line((left, y, right, y), fill="#E5EAEC", width=1)
            draw.text((24, y - 9), f"{ratio * max_y:.3g}", font=small, fill="#52656E")
        for tick in range(5):
            ratio = tick / 4
            x = left + int(ratio * (right - left))
            draw.line((x, top, x, bottom), fill="#F0F3F4", width=1)
            draw.text((x - 14, bottom + 7), f"{ratio * max_t:.0f}", font=small, fill="#52656E")
        for series_index, (series_name, field, color) in enumerate(series):
            points: list[tuple[int, int]] = []
            for row in rows:
                time_value = _number(row.get("simulation_time_s"))
                error_value = _number(row.get(field))
                if time_value is None or error_value is None:
                    continue
                x = int(left + time_value / max_t * (right - left))
                y = int(bottom - (error_value * unit_scale) / max_y * (bottom - top))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=color, width=2)
            legend_x = 840 + series_index * 165
            legend_y = top - 28
            draw.line((legend_x, legend_y + 11, legend_x + 24, legend_y + 11), fill=color, width=3)
            draw.text((legend_x + 31, legend_y), series_name, font=small, fill="#445761")
    draw.text((left, 750), f"Модельное время, с (0–{max_t:.1f}); вертикальные и угловые ошибки показаны раздельно.",
              font=small, fill="#445761")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def _registered_trial_directories(runs_root: Path) -> dict[str, Path]:
    trial_dirs: dict[str, Path] = {}
    for record_path in sorted(runs_root.glob("*/campaign_trial.json")):
        trial = _read_json(record_path)
        trial_id = trial.get("trial_id")
        if not isinstance(trial_id, str) or not trial_id:
            raise ValueError(f"Campaign trial record has no valid trial_id: {record_path}")
        if trial_id in trial_dirs:
            raise ValueError(f"Duplicate campaign trial_id {trial_id!r} under {runs_root}")
        trial_dirs[trial_id] = record_path.parent
    return trial_dirs


def analyze(campaign_path: Path = CAMPAIGN_PATH, runs_root: Path = RUNS_ROOT,
            tables_root: Path = TABLES_ROOT, plots_root: Path = PLOTS_ROOT,
            require_all_jobs: bool = True) -> dict[str, Any]:
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    expected_jobs = {job["trial_id"]: job for job in _expand_campaign_jobs(campaign)}
    trial_dirs = _registered_trial_directories(runs_root)
    unknown = sorted(set(trial_dirs) - set(expected_jobs))
    missing = sorted(set(expected_jobs) - set(trial_dirs))
    if unknown:
        raise ValueError(f"Unregistered trial IDs found under campaign runs: {unknown}")
    if require_all_jobs and missing:
        raise ValueError(f"Campaign is incomplete; missing {len(missing)} preregistered trials: {missing[:5]}")
    metrics: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    for trial_id, run_dir in trial_dirs.items():
        expected = expected_jobs[trial_id]
        _verify_trial_integrity(run_dir, campaign_path, expected)
        trial, _ = _trial_metrics(run_dir)
        if trial["experiment_id"] != expected["experiment_id"] or trial["variant_id"] != expected["variant_id"]:
            raise ValueError(f"Trial metadata no longer matches campaign for {trial_id}")
        metrics.append(trial)
        object_records.extend(trial["_object_records"])
    metrics.sort(key=lambda item: item["trial_id"])
    series = _series_summary(metrics)
    paired_summaries, paired_rows = _paired_comparisons(metrics)

    baseline_confusion: dict[tuple[str, str], int] = {}
    confusion_by_condition: dict[tuple[str, str, str, str], int] = {}
    for record in object_records:
        truth_label = str(record.get("ground_truth_class") or "NO_OBJECT")
        pred = str(record.get("detected_class") or "MISSING")
        if truth_label in (*COLORS, "NO_OBJECT"):
            pred = pred if pred in PREDICTION_LABELS else "UNKNOWN"
            condition_key = (
                str(record.get("experiment_id", "UNKNOWN")),
                str(record.get("variant_id", "UNKNOWN")), truth_label, pred,
            )
            confusion_by_condition[condition_key] = confusion_by_condition.get(condition_key, 0) + 1
            if record.get("experiment_id") == "V03_RANDOM_MIXED":
                baseline_confusion[(truth_label, pred)] = baseline_confusion.get((truth_label, pred), 0) + 1
    confusion_rows = [
        {"ground_truth": color,
         **{pred: baseline_confusion.get((color, pred), 0) for pred in PREDICTION_LABELS}}
        for color in (*COLORS, "NO_OBJECT")
    ]
    baseline_classification = classification_metrics(baseline_confusion)
    confusion_condition_rows = [
        {"experiment_id": experiment, "variant_id": variant,
         "ground_truth": truth, "prediction": prediction, "count": count}
        for (experiment, variant, truth, prediction), count in sorted(confusion_by_condition.items())
    ]

    baseline = [row for row in metrics if row["experiment_id"] == "V03_RANDOM_MIXED"]
    baseline_color_records = [
        row for row in object_records
        if row.get("experiment_id") == "V03_RANDOM_MIXED"
        and row.get("ground_truth_class") in COLORS
    ]
    baseline_color_correct = sum(
        row.get("detected_class") == row.get("ground_truth_class")
        for row in baseline_color_records
    )
    baseline_unknown = sum(row.get("detected_class") == "UNKNOWN"
                           for row in baseline_color_records)
    baseline_misclassified = sum(
        row.get("detected_class") in COLORS
        and row.get("detected_class") != row.get("ground_truth_class")
        for row in baseline_color_records
    )
    baseline_successes = sum(bool(row["test_pass"]) for row in baseline)
    ci_low, ci_high = wilson_interval(baseline_successes, len(baseline))
    required_rate = float(campaign["primary_acceptance"]["minimum_complete_batch_rate"])
    primary_pass = (
        len(baseline) == int(campaign["primary_acceptance"]["required_seeds"])
        and baseline_successes >= int(campaign["primary_acceptance"]["minimum_complete_batches"])
        and sum(int(row["wrong_bin_count"]) for row in baseline) == 0
        and sum(int(row["forbidden_contact_episodes"]) for row in baseline) == 0
        and sum(int(row["position_limit_violation_samples"]) + int(row["force_limit_violation_samples"])
                for row in baseline) == 0
    )
    failed_trials = [row["trial_id"] for row in metrics if not row["test_pass"]]
    overall_pass = primary_pass and not failed_trials and not missing
    results = {
        "schema_version": "M9-analysis-v1",
        "campaign_id": campaign["campaign_id"],
        "campaign_sha256": _sha256(campaign_path),
        "analyzer_source_sha256": _sha256(Path(__file__).resolve()),
        "analyzed_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "scope": "virtual MuJoCo simulation only; no physical prototype tests",
        "registered_trials": len(expected_jobs),
        "analyzed_trials": len(metrics),
        "missing_trials": missing,
        "expected_outcome_trials_passed": sum(bool(row["test_pass"]) for row in metrics),
        "expected_outcome_trials_failed": failed_trials,
        "primary_baseline": {
            "experiment_id": "V03_RANDOM_MIXED",
            "independent_unit": "three-object seeded batch",
            "n": len(baseline),
            "complete_batches": baseline_successes,
            "complete_batch_rate": baseline_successes / len(baseline) if baseline else None,
            "wilson_95_interval": [ci_low, ci_high],
            "target_rate": required_rate,
            "target_complete_batches": int(campaign["primary_acceptance"]["minimum_complete_batches"]),
            "wrong_bin_count": sum(int(row["wrong_bin_count"]) for row in baseline),
            "forbidden_contact_episodes": sum(int(row["forbidden_contact_episodes"]) for row in baseline),
            "joint_force_limit_violation_samples": sum(
                int(row["position_limit_violation_samples"]) + int(row["force_limit_violation_samples"])
                for row in baseline
            ),
            "pass": primary_pass,
        },
        "baseline_first_view_color": {
            "object_observations": len(baseline_color_records),
            "correct_classifications": baseline_color_correct,
            "unknown_abstentions": baseline_unknown,
            "misclassifications": baseline_misclassified,
            "missing_detections": sum(row.get("detected_class") == "MISSING"
                                      for row in baseline_color_records),
            "per_class_metrics": baseline_classification,
            "interpretation": "object observations pooled across seeded batches; objects within a batch are not independent runs",
        },
        "overall_expected_outcome_pass": overall_pass,
        "series": series,
        "paired_comparisons": paired_summaries,
        "metric_definitions": {
            "physical_sort_success": "M7 independent post-run evaluator true class + correct settled tray placement",
            "xy_error": "distance between initial evaluator truth and first public input detection after one-to-one XY association",
            "yaw_error": "absolute difference modulo pi/2 for square-object symmetry",
            "contact_sampling": "one record per geom-pair episode, pair availability sampled each physics step",
            "planned_clearance": "M5 sampled lower bound at per-plan collision sample step; not a continuous dynamic guarantee",
            "unit_of_confidence_interval": "independent run or seeded batch, not individual objects within a batch",
        },
        "limitations": [
            "All outcomes are from the MuJoCo model and the registered virtual sensor perturbations.",
            "A finite seed campaign does not establish behavior outside its sampled initial conditions.",
            "M5 sampled clearance and 500 Hz contact episodes do not guarantee a continuous minimum distance.",
            "No conclusion is made about a physical prototype, real camera calibration, or real gripper friction.",
        ],
    }
    tables_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    scenario_rows = _scenario_matrix_rows(campaign)
    _write_csv(tables_root / "Матрица_сценариев.csv", scenario_rows)
    flat_metrics = [
        {key: value for key, value in metric.items() if not key.startswith("_")}
        for metric in metrics
    ]
    _write_csv(tables_root / "Результаты_по_прогонам.csv", flat_metrics)
    object_columns = list(object_records[0]) if object_records else []
    _write_csv(tables_root / "Результаты_по_объектам.csv", object_records, object_columns)
    _write_csv(tables_root / "Сводка_по_сериям.csv", series)
    _write_csv(tables_root / "Сравнение_парных_seed.csv", paired_rows)
    _write_csv(tables_root / "Матрица_ошибок_цвета_V03.csv", confusion_rows)
    _write_csv(tables_root / "Метрики_классификации_V03.csv", baseline_classification)
    _write_csv(tables_root / "Матрица_цветов_по_условиям.csv", confusion_condition_rows)
    RESULTS_PATH.write_text(json.dumps(_json_safe(results), ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    _draw_rate_plot(series, plots_root / "Выполнение_серий.png")
    _draw_confusion_matrix(baseline_confusion, plots_root / "Матрица_цветов_V03.png")
    successful_baseline = [row for row in baseline if row["test_pass"]]
    if baseline:
        if successful_baseline:
            chosen = sorted(successful_baseline, key=lambda row: float(row["simulation_time_s"]))[
                (len(successful_baseline) - 1) // 2
            ]
            representative_rule = (
                "lower median simulation time among successful V03 batches; not claimed statistically typical"
            )
            outcome_label = "полная партия"
        else:
            chosen = sorted(
                baseline,
                key=lambda row: (-int(row["sorted_correctly"]), float(row["simulation_time_s"])),
            )[0]
            representative_rule = (
                "best observed V03 outcome, then shortest simulation time; failure example only"
            )
            outcome_label = "лучшая неполная партия"
        ssot = yaml.safe_load((ROOT / "02_Спецификация" / "параметры_системы.yaml").read_text(encoding="utf-8-sig"))
        _draw_trajectory(
            chosen, plots_root / "Траектория_представительного_прогона.png", ssot,
            outcome_label=outcome_label,
        )
        _draw_joint_error(chosen, plots_root / "Ошибка_слеживания_суставов.png")
        results["representative_run"] = {
            "run_id": chosen["run_id"],
            "test_pass": chosen["test_pass"],
            "rule": representative_rule,
        }
        results["representative_video_candidate"] = (
            {"run_id": chosen["run_id"], "rule": representative_rule}
            if successful_baseline else None
        )
        RESULTS_PATH.write_text(json.dumps(_json_safe(results), ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    return results


def _expand_campaign_jobs(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = campaign.get("seed_sets", {})
    jobs: list[dict[str, Any]] = []
    for experiment in campaign.get("experiments", []):
        for variant in experiment.get("variants", []):
            for seed in seeds[experiment["seeds"]]:
                jobs.append({
                    "trial_id": f"{experiment['experiment_id']}__{variant['variant_id']}__{int(seed)}",
                    "experiment_id": experiment["experiment_id"],
                    "variant_id": variant["variant_id"],
                    "scenario": experiment["scenario"],
                    "scenario_dir": experiment.get("scenario_dir"),
                    "seed": int(seed),
                    "expected_kind": variant["expected_kind"],
                    "expected_reason": variant.get("expected_reason"),
                    "expected_track_status": variant.get("expected_track_status"),
                    "expected_failure_reason": variant.get("expected_failure_reason"),
                    "expected_object_outcomes": dict(variant.get("expected_object_outcomes", {})),
                    "required_perception_checks": dict(variant.get("required_perception_checks", {})),
                    "tags": list(variant.get("tags", [])),
                    "parameters": dict(variant.get("parameters", {})),
                    "paired_control": experiment.get("paired_control"),
                })
    for fixed in campaign.get("fixed_trials", []):
        jobs.append({
            "trial_id": fixed["trial_id"],
            "experiment_id": fixed["trial_id"],
            "variant_id": "fixed",
            "scenario": fixed["scenario"],
            "scenario_dir": fixed.get("scenario_dir"),
            "seed": fixed.get("seed"),
            "expected_kind": fixed["expected_kind"],
            "expected_reason": fixed.get("expected_reason"),
            "expected_track_status": fixed.get("expected_track_status"),
            "expected_failure_reason": fixed.get("expected_failure_reason"),
            "expected_object_outcomes": dict(fixed.get("expected_object_outcomes", {})),
            "required_perception_checks": dict(fixed.get("required_perception_checks", {})),
            "tags": list(fixed.get("tags", [])),
            "parameters": dict(fixed.get("parameters", {})),
            "paired_control": None,
        })
    return jobs


def _scenario_matrix_rows(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "trial_id": job["trial_id"],
            "scenario_families": ";".join(job.get("tags", [])),
            "experiment_id": job["experiment_id"],
            "variant_id": job["variant_id"],
            "scenario_id": job["scenario"],
            "seed": job.get("seed"),
            "expected_kind": job["expected_kind"],
            "expected_reason": job.get("expected_reason"),
            "expected_track_status": job.get("expected_track_status"),
            "expected_failure_reason": job.get("expected_failure_reason"),
            "expected_object_outcomes": job.get("expected_object_outcomes", {}),
            "required_perception_checks": job.get("required_perception_checks", {}),
            "parameters": job.get("parameters", {}),
            "paired_control": job.get("paired_control"),
        }
        for job in _expand_campaign_jobs(campaign)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Recalculate M9 metrics and plots from raw run evidence")
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN_PATH)
    parser.add_argument("--runs", type=Path, default=RUNS_ROOT)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Analyze available registered trials without claiming campaign acceptance")
    args = parser.parse_args()
    result = analyze(args.campaign.resolve(), args.runs.resolve(), require_all_jobs=not args.allow_incomplete)
    print(json.dumps({
        "campaign_id": result["campaign_id"],
        "analyzed_trials": result["analyzed_trials"],
        "missing_trials": len(result["missing_trials"]),
        "primary_baseline": result["primary_baseline"],
        "overall_expected_outcome_pass": result["overall_expected_outcome_pass"],
        "results_path": RESULTS_PATH.relative_to(ROOT).as_posix(),
    }, ensure_ascii=False, indent=2))
    return 0 if not result["missing_trials"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
