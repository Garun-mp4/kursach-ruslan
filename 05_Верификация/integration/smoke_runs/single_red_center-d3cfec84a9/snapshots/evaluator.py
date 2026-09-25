from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from integration.config import MODEL_PATH, RuntimeConfig, load_ssot, value


def _body_qpos_address(model: mujoco.MjModel, body_name: str) -> tuple[int, int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Evaluator cannot find body {body_name!r}")
    joint_id = int(model.body_jntadr[body_id])
    if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Evaluator requires a free joint for {body_name!r}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def evaluate_run(run_dir: Path) -> dict[str, Any]:
    """Post-run evaluator. Ground truth is read only after controller termination."""
    private_dir = run_dir / "evaluator_private"
    truth = json.loads((private_dir / "scene_truth.json").read_text(encoding="utf-8"))
    state = np.load(private_dir / "final_state.npz", allow_pickle=False)
    qpos, qvel = state["qpos"], state["qvel"]
    final_time_s = float(state["simulation_time_s"].item())
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    if len(qpos) != model.nq or len(qvel) != model.nv:
        raise ValueError("Final state dimensions do not match the evaluated M2 model")
    ssot = load_ssot()
    object_height = float(value(ssot, "object.size_xyz_m")[2])
    tray_floor_top = float(value(ssot, "cell.tray_floor_thickness_m"))
    target_z = tray_floor_top + object_height / 2
    slots = []
    trays = value(ssot, "cell.tray_centers_xy_m")
    offsets = value(ssot, "cell.tray_slot_x_offsets_m")
    y_offset = float(value(ssot, "cell.tray_slot_y_offset_m"))
    for color in ("RED", "GREEN", "BLUE"):
        for index, offset in enumerate(offsets):
            slots.append({
                "slot_id": f"{color}:{index}",
                "color": color,
                "xy": (float(trays[color][0]) + float(offset),
                       float(trays[color][1]) + y_offset),
            })
    runtime_config = RuntimeConfig.load()
    slot_tolerance = runtime_config.placement_slot_acceptance_radius_m
    evaluated = []
    for item in truth["active_objects"]:
        qadr, dadr = _body_qpos_address(model, item["body_name"])
        xyz = tuple(map(float, qpos[qadr:qadr + 3]))
        linear_velocity = tuple(map(float, qvel[dadr:dadr + 3]))
        angular_velocity = tuple(map(float, qvel[dadr + 3:dadr + 6]))
        ranked = sorted(
            (math.dist(xyz[:2], slot["xy"]), slot) for slot in slots
        )
        distance, nearest = ranked[0]
        in_slot = distance <= slot_tolerance
        resting = abs(xyz[2] - target_z) <= runtime_config.evaluator_rest_height_tolerance_m
        settled = (
            math.sqrt(sum(v * v for v in linear_velocity))
            <= runtime_config.evaluator_linear_speed_tolerance_m_s
            and math.sqrt(sum(v * v for v in angular_velocity))
            <= runtime_config.evaluator_angular_speed_tolerance_rad_s
        )
        correct = bool(in_slot and nearest["color"] == item["class_label"] and resting and settled)
        evaluated.append({
            "body_name": item["body_name"],
            "ground_truth_class": item["class_label"],
            "final_xyz_m": xyz,
            "linear_velocity_m_s": linear_velocity,
            "angular_velocity_rad_s": angular_velocity,
            "nearest_slot_id": nearest["slot_id"] if in_slot else None,
            "nearest_zone": nearest["color"] if in_slot else None,
            "horizontal_slot_error_m": distance,
            "resting_on_tray": resting,
            "settled": settled,
            "physical_sort_success": correct,
        })
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    controller_summary = manifest.get("controller_summary", {})
    controller_records = controller_summary.get("track_records", {})
    report = {
        "schema_version": "M7-evaluator-v1.0",
        "run_id": manifest["run_id"],
        "evaluation_source": "post-run MuJoCo free-joint state plus private scene ground truth",
        "simulation_time_s": final_time_s,
        "controller_final_state": controller_summary.get("final_state"),
        "controller_track_records": controller_records,
        "controller_reports_done": controller_summary.get("final_state") == "DONE",
        "items": evaluated,
        "physical_sort_success_count": sum(bool(item["physical_sort_success"]) for item in evaluated),
        "physical_sort_failure_count": sum(not item["physical_sort_success"] for item in evaluated),
        "all_active_objects_physically_sorted": bool(evaluated) and all(
            item["physical_sort_success"] for item in evaluated
        ),
        "controller_done_and_physical_success": (
            controller_summary.get("final_state") == "DONE"
            and bool(evaluated)
            and all(item["physical_sort_success"] for item in evaluated)
        ),
        "limitations": [
            "The evaluator uses simulator ground truth only after the controller run has ended.",
            "This is a virtual physical-model result, not a physical prototype test.",
            "Slot and settling tolerances are project-level integration checks, not M9 campaign acceptance criteria.",
        ],
    }
    output_path = run_dir / "evaluator" / "report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate one completed M7 run.")
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_run(args.run_dir.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
