from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any

import mujoco
import numpy as np
import yaml

from .config import MODEL_PATH, ROOT, RUNTIME_CONFIG_PATH, SSOT_PATH, RuntimeConfig, load_ssot, value
from .runtime import SorterRuntime, sha256
from .scene import SceneGenerator, load_scenario

EVALUATOR_PATH = ROOT / "05_Верификация" / "integration" / "evaluator.py"
M4_CONFIG = ROOT / "04_Программа" / "perception" / "perception_config.yaml"
M4_CALIBRATION = ROOT / "05_Верификация" / "perception" / "calibration" / "camera_calibration.json"
M4_BACKGROUND = ROOT / "05_Верификация" / "perception" / "validation" / "background_reference_rgb.npy"


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _snapshot(run_dir: Path, scenario_path: Path) -> dict[str, str]:
    files = {
        "ssot": SSOT_PATH,
        "mjcf": MODEL_PATH,
        "runtime_config": RUNTIME_CONFIG_PATH,
        "scenario": scenario_path,
        "perception_config": M4_CONFIG,
        "camera_calibration": M4_CALIBRATION,
        "background_reference": M4_BACKGROUND,
        "runner": Path(__file__),
        "runtime": Path(__file__).with_name("runtime.py"),
        "actuator": Path(__file__).with_name("actuator.py"),
        "scene_generator": Path(__file__).with_name("scene.py"),
        "sensor_adapter": Path(__file__).with_name("sensors.py"),
        "placement_adapter": Path(__file__).with_name("placement.py"),
        "controller": ROOT / "04_Программа" / "controller" / "controller.py",
        "planner": ROOT / "04_Программа" / "planning" / "planner.py",
        "perception": ROOT / "04_Программа" / "perception" / "detector.py",
        "evaluator": EVALUATOR_PATH,
    }
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"Required M7 provenance input is missing: {source}")
        destination = snapshot_dir / f"{name}{source.suffix}"
        shutil.copy2(source, destination)
        hashes[name] = sha256(source)
    return hashes


def _apply_profile(model: mujoco.MjModel, profile: str, ssot: dict) -> dict[str, Any]:
    baseline = {
        "timestep_s": float(value(ssot, "environment.timestep_s")),
        "solver_iterations": int(value(ssot, "environment.solver_iterations")),
        "solver_tolerance": float(value(ssot, "environment.solver_tolerance")),
        "gravity_m_s2": float(value(ssot, "environment.gravity_m_s2")),
    }
    profiles = {
        "nominal": {},
        "fine_timestep": {"timestep_s": 0.001},
        "coarse_timestep": {"timestep_s": 0.005},
        "solver_50": {"solver_iterations": 50},
        "solver_150": {"solver_iterations": 150},
    }
    if profile not in profiles:
        raise ValueError(f"Unsupported physics profile {profile!r}")
    if not np.isclose(model.opt.timestep, baseline["timestep_s"], atol=1e-12, rtol=0):
        raise ValueError("Compiled M2 MJCF timestep disagrees with SSOT")
    if int(model.opt.iterations) != baseline["solver_iterations"]:
        raise ValueError("Compiled M2 solver iterations disagree with SSOT")
    if not np.isclose(model.opt.tolerance, baseline["solver_tolerance"], atol=1e-14, rtol=0):
        raise ValueError("Compiled M2 solver tolerance disagrees with SSOT")
    if not np.allclose(model.opt.gravity, (0.0, 0.0, -baseline["gravity_m_s2"]), atol=1e-9):
        raise ValueError("Compiled M2 gravity disagrees with SSOT")
    effective = {**baseline, **profiles[profile]}
    model.opt.timestep = effective["timestep_s"]
    model.opt.iterations = effective["solver_iterations"]
    effective["profile"] = profile
    effective["solver"] = str(model.opt.solver)
    effective["integrator"] = str(model.opt.integrator)
    effective["contact_condim"] = int(value(ssot, "contact.condim"))
    return effective


def _load_model(scenario: dict[str, Any], profile: str) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, Any], Any]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_observation")
    if key_id < 0:
        raise ValueError("M2 model is missing the home_observation reset keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    ssot = load_ssot()
    physics = _apply_profile(model, profile, ssot)
    runtime_config = RuntimeConfig.load()
    generated = SceneGenerator(ssot, runtime_config).generate(model, data, scenario)
    for body_name, rgba in scenario.get("render_rgba_overrides", {}).items():
        geom_name = f"{body_name}_visual"
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise ValueError(f"Scenario visual override references missing geom {geom_name}")
        rgba_values = np.asarray(rgba, dtype=float)
        if rgba_values.shape != (4,) or not np.all(np.isfinite(rgba_values)) or np.any(rgba_values < 0) or np.any(rgba_values > 1):
            raise ValueError(f"Invalid render RGBA override for {body_name}")
        model.geom_rgba[geom_id] = rgba_values
    mujoco.mj_forward(model, data)
    return model, data, physics, generated


def _run_one(*, scenario_name: str, mode: str, out_root: Path, profile: str,
             max_simulation_time_s: float | None = None,
             camera_failure: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario_path = ROOT / "04_Программа" / "integration" / "scenarios" / f"{scenario_name}.yaml"
    scenario = load_scenario(scenario_name)
    run_id = f"{scenario_name}-{uuid.uuid4().hex[:10]}"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    hashes = _snapshot(run_dir, scenario_path)
    model, data, physics, generated = _load_model(scenario, profile)
    private_dir = run_dir / "evaluator_private"
    private_dir.mkdir()
    _json_write(private_dir / "scene_truth.json", {
        "scenario_id": generated.scenario_id,
        "seed": generated.seed,
        "active_objects": [
            {"body_name": item.body_name, "class_label": item.class_label,
             "initial_xy_m": item.initial_xy_m, "yaw_rad": item.yaw_rad}
            for item in generated.active_objects
        ],
        "inactive_bodies": list(generated.inactive_bodies),
    })
    state_path = private_dir / "final_state.npz"
    runtime_config = RuntimeConfig.load()
    effective_timeout = max_simulation_time_s or runtime_config.default_max_simulation_time_s
    viewer_context = None
    viewer = None
    if mode == "interactive":
        from mujoco import viewer as mujoco_viewer
        viewer_context = mujoco_viewer.launch_passive(model, data)
        viewer = viewer_context
    runtime = SorterRuntime(
        model, data, run_id=run_id, runtime_config=runtime_config,
        output_dir=run_dir / "controller", viewer=viewer,
        camera_fault_injection=camera_failure, physics_expected=physics,
    )
    started_utc = datetime.now(timezone.utc).isoformat()
    wall_started = time.perf_counter()
    try:
        controller_summary = runtime.run(max_simulation_time_s=effective_timeout)
        np.savez_compressed(state_path, qpos=np.asarray(data.qpos), qvel=np.asarray(data.qvel),
                            simulation_time_s=np.asarray(float(data.time)))
    finally:
        runtime.close()
        if viewer_context is not None:
            viewer_context.close()
    manifest: dict[str, Any] = {
        "schema_version": "M7-run-manifest-v1.0",
        "run_id": run_id,
        "scenario_id": scenario_name,
        "scenario_seed": int(scenario["seed"]),
        "mode": mode,
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_elapsed_s": time.perf_counter() - wall_started,
        "runtime_versions": {"python": sys.version, "mujoco": mujoco.__version__,
                              "numpy": np.__version__},
        "physics": physics,
        "runtime_configuration": yaml.safe_load(RUNTIME_CONFIG_PATH.read_text(encoding="utf-8")),
        "fault_injection": {"camera_failure": camera_failure,
                            "render_rgba_overrides": scenario.get("render_rgba_overrides", {})},
        "initial_occupied_slots_from_public_rgb": None,
        "input_hashes_sha256": hashes,
        "controller_summary": controller_summary,
        "initial_occupied_slots_from_public_rgb": controller_summary.get(
            "initial_occupied_slots_from_public_rgb", []
        ),
        "private_evaluator_inputs": {
            "ground_truth": "evaluator_private/scene_truth.json",
            "final_state": "evaluator_private/final_state.npz",
        },
        "relative_outputs": {
            "controller_events": "controller/controller_events.jsonl",
            "controller_telemetry": "controller/telemetry.csv",
            "physics_contacts": "controller/physics_contacts.jsonl",
            "evaluator_report": "evaluator/report.json",
        },
    }
    _json_write(run_dir / "manifest.json", manifest)
    evaluator_cmd = [sys.executable, str(EVALUATOR_PATH), "--run-dir", str(run_dir)]
    env = os.environ.copy()
    program_path = str(ROOT / "04_Программа")
    env["PYTHONPATH"] = program_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    evaluated = subprocess.run(evaluator_cmd, cwd=ROOT, env=env, text=True,
                               capture_output=True, check=False)
    if evaluated.returncode != 0:
        raise RuntimeError(f"M7 evaluator failed: {evaluated.stderr[-4000:]}")
    report = json.loads((run_dir / "evaluator" / "report.json").read_text(encoding="utf-8"))
    manifest["evaluator_summary"] = report
    _json_write(run_dir / "manifest.json", manifest)
    return manifest, report


def main() -> int:
    parser = argparse.ArgumentParser(description="M7 physical MuJoCo integration runner")
    parser.add_argument("--mode", choices=("batch", "interactive"), default="batch")
    parser.add_argument("--scenario", default="single_red_center")
    parser.add_argument("--physics-profile", choices=("nominal", "fine_timestep", "coarse_timestep", "solver_50", "solver_150"), default="nominal")
    parser.add_argument("--out", type=Path, default=ROOT / "05_Верификация" / "integration" / "runs")
    parser.add_argument("--max-simulation-time", type=float)
    parser.add_argument("--camera-failure", action="store_true")
    args = parser.parse_args()
    manifest, report = _run_one(
        scenario_name=args.scenario, mode=args.mode, out_root=args.out.resolve(),
        profile=args.physics_profile, max_simulation_time_s=args.max_simulation_time,
        camera_failure=args.camera_failure,
    )
    print(json.dumps({"run_id": manifest["run_id"], "state": manifest["controller_summary"]["final_state"],
                      "controller_summary": manifest["controller_summary"],
                      "evaluator_summary": report}, ensure_ascii=False, indent=2))
    return 0 if manifest["controller_summary"]["final_state"] == "DONE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
