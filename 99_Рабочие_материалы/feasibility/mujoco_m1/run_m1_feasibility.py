from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)


def load_model(path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def named_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    ident = mujoco.mj_name2id(model, kind, name)
    if ident < 0:
        raise RuntimeError(f"Missing {kind.name} named {name!r}")
    return ident


def joint_probe(filename: str, joint_name: str, actuator_name: str,
                target: float, expected_kind: int) -> dict[str, Any]:
    model, data = load_model(MODELS / filename)
    joint_id = named_id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    actuator_id = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    qpos_address = int(model.jnt_qposadr[joint_id])
    qvel_address = int(model.jnt_dofadr[joint_id])
    lower, upper = map(float, model.jnt_range[joint_id])
    if int(model.jnt_type[joint_id]) != expected_kind:
        raise AssertionError(f"{joint_name} has unexpected type {model.jnt_type[joint_id]}")
    data.ctrl[actuator_id] = target
    start = time.perf_counter()
    samples: list[float] = []
    for _ in range(round(1.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        samples.append(float(data.qpos[qpos_address]))
    elapsed = time.perf_counter() - start
    observed = float(data.qpos[qpos_address])
    velocity = float(data.qvel[qvel_address])
    recent = np.asarray(samples[-max(2, round(0.1 / model.opt.timestep)):], dtype=np.float64)
    error = abs(observed - target)
    passed = (
        lower - 1e-6 <= observed <= upper + 1e-6
        and error <= (0.025 if expected_kind == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.004)
        and abs(velocity) <= (0.08 if expected_kind == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.02)
        and float(np.ptp(recent)) <= (0.012 if expected_kind == int(mujoco.mjtJoint.mjJNT_HINGE) else 0.002)
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "model": filename,
        "joint": joint_name,
        "kind": "revolute" if expected_kind == int(mujoco.mjtJoint.mjJNT_HINGE) else "prismatic",
        "target": target,
        "observed": observed,
        "absolute_error": error,
        "velocity": velocity,
        "limit": [lower, upper],
        "within_limits": lower - 1e-6 <= observed <= upper + 1e-6,
        "settle_peak_to_peak_last_0_1s": float(np.ptp(recent)),
        "simulation_seconds": 1.0,
        "wall_seconds": elapsed,
        "simulated_seconds_per_wall_second": 1.0 / max(elapsed, 1e-9),
        "criterion": "target error, joint limits, low final velocity and bounded terminal variation"
    }


def camera_probe() -> dict[str, Any]:
    model, data = load_model(MODELS / "rgb_camera_probe.xml")
    renderer = mujoco.Renderer(model, height=480, width=640)
    mujoco.mj_forward(model, data)
    try:
        renderer.update_scene(data, camera="overhead")
        start = time.perf_counter()
        frame = renderer.render().copy()
        elapsed = time.perf_counter() - start
    finally:
        renderer.close()
    if frame.ndim != 3 or frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise AssertionError(f"Unexpected RGB frame: shape={frame.shape}, dtype={frame.dtype}")
    red_mask = (
        (frame[:, :, 0].astype(np.int16) > frame[:, :, 1].astype(np.int16) * 1.45)
        & (frame[:, :, 0].astype(np.int16) > frame[:, :, 2].astype(np.int16) * 1.35)
        & (frame[:, :, 0] > 55)
    )
    red_pixels = int(np.count_nonzero(red_mask))
    if red_pixels < 100:
        raise AssertionError(f"Expected rendered red object; only {red_pixels} red-dominant pixels")
    path = OUT / "rgb_camera_frame.png"
    imageio.imwrite(path, frame)
    return {
        "status": "PASS",
        "api": "mujoco.Renderer.update_scene(..., camera='overhead').render()",
        "render_mode": "offscreen OpenGL, no visible GUI window",
        "frame_shape_h_w_channels": list(frame.shape),
        "dtype": str(frame.dtype),
        "channel_order": "RGB as returned by MuJoCo Renderer",
        "red_dominant_pixels": red_pixels,
        "wall_seconds": elapsed,
        "frame_file": str(path.relative_to(ROOT)),
        "criterion": "640x480 uint8 RGB array, rendered object detected by RGB channel ordering, frame saved"
    }


def gripper_contacts(model: mujoco.MjModel, data: mujoco.MjData,
                     obj_id: int, left_id: int, right_id: int) -> tuple[bool, bool, int]:
    left = False
    right = False
    total = int(data.ncon)
    for index in range(data.ncon):
        c = data.contact[index]
        pair = {int(c.geom1), int(c.geom2)}
        if obj_id in pair and left_id in pair:
            left = True
        if obj_id in pair and right_id in pair:
            right = True
    return left, right, total


def run_grasp_trial(name: str, friction: float, finger_force: float,
                    timestep: float, record_video: bool = False) -> dict[str, Any]:
    template = (MODELS / "gripper_probe_template.xml").read_text(encoding="utf-8")
    xml = (
        template
        .replace("__TIMESTEP__", f"{timestep:.6f}")
        .replace("__FRICTION__", f"{friction:.6f}")
        .replace("__FORCE__", f"{finger_force:.6f}")
    )
    generated_model = OUT / f"gripper_{name}.xml"
    generated_model.write_text(xml, encoding="utf-8")
    model, data = load_model(generated_model)
    lift_id = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "lift_position")
    left_act = named_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_position")
    obj_body = named_id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
    obj_geom = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    left_geom = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger")
    right_geom = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "right_finger")
    floor_geom = named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    object_dof = int(model.jnt_dofadr[named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")])
    lift_qpos = int(model.jnt_qposadr[named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "lift")])
    left_qpos = int(model.jnt_qposadr[named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_finger_slide")])
    right_qpos = int(model.jnt_qposadr[named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_finger_slide")])
    half_height = 0.020
    phase_steps = {
        "settle": round(0.50 / timestep),
        "close": round(0.80 / timestep),
        "lift": round(1.20 / timestep),
        "release": round(1.00 / timestep),
    }
    renderer = None
    writer = None
    video_path = OUT / "gripper_contact_probe.mp4"
    if record_video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        mujoco.mj_forward(model, data)
        writer = imageio.get_writer(
            video_path, fps=25, codec="libx264", quality=7, macro_block_size=16
        )
    capture_every = max(1, round(0.04 / timestep))
    frame_count = 0
    bilateral_during_close = False
    bilateral_during_lift = False
    left_contact_steps = 0
    right_contact_steps = 0
    max_contacts = 0
    z_lift: list[float] = []
    floor_touch_at_release = False
    all_start = time.perf_counter()

    def step_phase(label: str, count: int, lift_target: float, finger_target: float) -> None:
        nonlocal frame_count, bilateral_during_close, bilateral_during_lift
        nonlocal left_contact_steps, right_contact_steps, max_contacts, floor_touch_at_release
        data.ctrl[lift_id] = lift_target
        data.ctrl[left_act] = finger_target
        for step_index in range(count):
            mujoco.mj_step(model, data)
            left, right, ncon = gripper_contacts(model, data, obj_geom, left_geom, right_geom)
            max_contacts = max(max_contacts, ncon)
            if label == "close":
                bilateral_during_close = bilateral_during_close or (left and right)
            elif label == "lift":
                bilateral_during_lift = bilateral_during_lift or (left and right)
                left_contact_steps += int(left)
                right_contact_steps += int(right)
                z_lift.append(float(data.xpos[obj_body][2]))
            elif label == "release":
                if any(
                    {int(data.contact[i].geom1), int(data.contact[i].geom2)} == {obj_geom, floor_geom}
                    for i in range(data.ncon)
                ):
                    floor_touch_at_release = True
            if writer is not None and renderer is not None and step_index % capture_every == 0:
                renderer.update_scene(data, camera="gripper_view")
                frame = renderer.render().copy()
                writer.append_data(frame)
                frame_count += 1
                if label == "lift" and step_index == count // 2:
                    imageio.imwrite(OUT / "gripper_contact_lift.png", frame)
                if label == "release" and step_index >= count - capture_every:
                    imageio.imwrite(OUT / "gripper_contact_released.png", frame)

    try:
        step_phase("settle", phase_steps["settle"], 0.0, 0.0)
        z_ground = float(data.xpos[obj_body][2])
        q_initial = float(data.qpos[lift_qpos])
        step_phase("close", phase_steps["close"], 0.0, 0.020)
        z_after_close = float(data.xpos[obj_body][2])
        step_phase("lift", phase_steps["lift"], 0.090, 0.020)
        z_after_lift = float(data.xpos[obj_body][2])
        if record_video:
            # Additional hold segment makes the transition visible in the diagnostic clip.
            step_phase("hold", round(0.25 / timestep), 0.090, 0.020)
        step_phase("release", phase_steps["release"], 0.090, 0.0)
        z_final = float(data.xpos[obj_body][2])
        z_velocity_final = float(data.qvel[object_dof + 2])
        left_final, right_final, final_contact_count = gripper_contacts(
            model, data, obj_geom, left_geom, right_geom
        )
        lift_final = float(data.qpos[lift_qpos])
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    lift_samples = np.asarray(z_lift, dtype=np.float64)
    tail_count = max(1, round(min(0.35, 0.25 / timestep)))
    tail_samples = lift_samples[-tail_count:] if lift_samples.size else np.asarray([])
    hold_height_threshold = z_ground + 0.045
    hold_fraction_tail = (
        float(np.mean(tail_samples > hold_height_threshold)) if tail_samples.size else 0.0
    )
    height_gain = z_after_lift - z_ground
    bilateral = bilateral_during_close and bilateral_during_lift
    physically_lifted = height_gain >= 0.045 and hold_fraction_tail >= 0.80
    released_to_floor = (
        abs(z_final - half_height) <= 0.008
        and abs(z_velocity_final) <= 0.05
        and not left_final
        and not right_final
        and floor_touch_at_release
    )
    finger_sync_error = abs(float(data.qpos[left_qpos]) - float(data.qpos[right_qpos]))
    has_one_finger_mimic = (
        int(np.count_nonzero(model.eq_type == int(mujoco.mjtEq.mjEQ_JOINT))) == 1
    )
    object_weld_count = int(np.count_nonzero(model.eq_type == int(mujoco.mjtEq.mjEQ_WELD)))
    # Joint equality models the mechanical finger linkage; the object has no weld/attach.
    no_attach_constraint = object_weld_count == 0
    successful = bilateral and physically_lifted and released_to_floor and no_attach_constraint and has_one_finger_mimic and finger_sync_error <= 0.002
    sim_duration = (sum(phase_steps.values()) + (round(0.25 / timestep) if record_video else 0)) * timestep
    elapsed = time.perf_counter() - all_start
    result = {
        "trial": name,
        "friction_coefficient": friction,
        "max_finger_actuator_force_N": finger_force,
        "timestep_s": timestep,
        "status": "PASS" if successful else "FAIL",
        "bilateral_object_finger_contact_during_close": bilateral_during_close,
        "bilateral_object_finger_contact_during_lift": bilateral_during_lift,
        "left_contact_lift_steps": left_contact_steps,
        "right_contact_lift_steps": right_contact_steps,
        "max_simultaneous_contact_pairs": max_contacts,
        "object_z_after_settle_m": z_ground,
        "object_z_after_close_m": z_after_close,
        "object_z_after_lift_m": z_after_lift,
        "object_lift_gain_m": height_gain,
        "held_fraction_of_last_lift_window": hold_fraction_tail,
        "lift_joint_final_qpos_m": lift_final,
        "object_z_after_release_m": z_final,
        "object_vertical_velocity_after_release_m_s": z_velocity_final,
        "object_floor_contact_observed_after_release": floor_touch_at_release,
        "object_left_finger_contact_at_end": left_final,
        "object_right_finger_contact_at_end": right_final,
        "model_equality_constraints_count": int(model.neq),
        "symmetric_finger_mimic_constraint_count": int(np.count_nonzero(model.eq_type == int(mujoco.mjtEq.mjEQ_JOINT))),
        "object_weld_constraint_count": object_weld_count,
        "symmetric_finger_position_error_m": finger_sync_error,
        "one_actuated_gripper_coordinate": has_one_finger_mimic,
        "no_attach_or_weld_constraint": no_attach_constraint,
        "simulation_duration_s": sim_duration,
        "wall_seconds": elapsed,
        "simulation_to_wall_ratio": sim_duration / max(elapsed, 1e-9),
        "recorded_frames": frame_count,
        "video_file": str(video_path.relative_to(ROOT)) if record_video else None,
        "criterion": "one actuated gripper coordinate drives mechanically coupled fingers; bilateral physical contact, object rises >=45 mm and stays held in the terminal lift window; opening releases it to the floor; no object weld/attach constraint"
    }
    if q_initial != 0.0:
        result["initial_lift_joint_qpos_m"] = q_initial
    return result


def verify_video() -> dict[str, Any]:
    if not (OUT / "gripper_contact_probe.mp4").exists():
        raise FileNotFoundError("MP4 was not generated")
    path = OUT / "gripper_contact_probe.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    version = subprocess.run(
        [ffmpeg, "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    reader = imageio.get_reader(path, format="ffmpeg")
    try:
        meta = reader.get_meta_data()
        count = sum(1 for _ in reader)
    finally:
        reader.close()
    passed = count >= 50 and meta.get("size") == (640, 480)
    return {
        "status": "PASS" if passed else "FAIL",
        "file": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "decoded_frame_count": count,
        "decoded_size": meta.get("size"),
        "fps": meta.get("fps"),
        "duration_s": meta.get("duration"),
        "ffmpeg": version,
        "full_decode_completed": True,
        "criterion": "MP4 encoded from simulator camera frames and decoded to completion with expected dimensions"
    }


def write_csv(results: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for key in ("revolute", "prismatic", "rgb_camera", "grasp_baseline",
                "grasp_baseline_headless", "grasp_low_friction", "grasp_low_force", "grasp_coarse_timestep",
                "mp4"):
        value = results[key]
        row = {"test": key}
        if isinstance(value, dict):
            row.update(value)
        rows.append(row)
    path = OUT / "feasibility_results.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    results: dict[str, Any] = {
        "metadata": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "imageio": metadata.version("imageio"),
            "imageio_ffmpeg": imageio_ffmpeg.__version__,
            "random_seed": None,
            "seed_note": "All feasibility scenes are deterministic and contain no randomized sampling.",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "scope": "isolated M1 tool-feasibility probes only; not robot-system tests or course results"
        }
    }
    results["revolute"] = joint_probe(
        "revolute_probe.xml", "probe_hinge", "hinge_position", 0.55,
        int(mujoco.mjtJoint.mjJNT_HINGE)
    )
    results["prismatic"] = joint_probe(
        "prismatic_probe.xml", "probe_slide", "slide_position", 0.12,
        int(mujoco.mjtJoint.mjJNT_SLIDE)
    )
    results["rgb_camera"] = camera_probe()
    results["grasp_baseline"] = run_grasp_trial(
        "baseline", friction=0.9, finger_force=5.0, timestep=0.002, record_video=True
    )
    results["grasp_baseline_headless"] = run_grasp_trial(
        "baseline_headless", friction=0.9, finger_force=5.0, timestep=0.002
    )
    results["grasp_low_friction"] = run_grasp_trial(
        "low_friction", friction=0.005, finger_force=5.0, timestep=0.002
    )
    results["grasp_low_force"] = run_grasp_trial(
        "low_force", friction=0.9, finger_force=0.05, timestep=0.002
    )
    results["grasp_coarse_timestep"] = run_grasp_trial(
        "coarse_timestep", friction=0.9, finger_force=5.0, timestep=0.01
    )
    results["mp4"] = verify_video()
    # A negative-grasp result is a successful test outcome only when it fails to lift
    # under deliberately inadequate friction/force; it is never counted as a course run.
    for key, expectation in (("grasp_low_friction", "low normal-direction tangential capacity must fail to lift"), ("grasp_low_force", "insufficient actuator force must fail to lift")):
        trial = results[key]
        trial["expected_negative_case"] = expectation
        trial["protocol_verdict"] = "PASS_EXPECTED_NEGATIVE" if trial["status"] == "FAIL" and trial["object_lift_gain_m"] < 0.01 else "FAIL_NEGATIVE_CASE_NOT_REPRODUCED"
    coarse = results["grasp_coarse_timestep"]
    coarse["protocol_verdict"] = "LIMITATION_DETECTED" if coarse["status"] == "FAIL" and coarse["object_lift_gain_m"] < 0.01 else "NO_LIMITATION_AT_TESTED_COARSE_STEP"
    results["negative_case_expectations"] = {
        "low_friction_did_not_pass_hold_and_release": results["grasp_low_friction"]["status"] == "FAIL",
        "low_force_did_not_pass_hold_and_release": results["grasp_low_force"]["status"] == "FAIL"
    }
    results["overall_status"] = "PASS" if (
        results["revolute"]["status"] == "PASS"
        and results["prismatic"]["status"] == "PASS"
        and results["rgb_camera"]["status"] == "PASS"
        and results["grasp_baseline"]["status"] == "PASS"
        and results["grasp_baseline_headless"]["status"] == "PASS"
        and results["grasp_low_friction"]["status"] == "FAIL"
        and results["grasp_low_force"]["status"] == "FAIL"
        and results["grasp_coarse_timestep"]["status"] in {"PASS", "FAIL"}
        and results["mp4"]["status"] == "PASS"
        and all(results["negative_case_expectations"].values())
    ) else "FAIL"
    json_path = OUT / "feasibility_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if results["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
