"""Verify the M4 arm-occlusion contract against the current M2 model geometry."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "04_Программа"
M4_VERIFICATION = ROOT / "05_Верификация" / "perception"
OUT = ROOT / "05_Верификация" / "planning" / "perception_compatibility"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))
if str(M4_VERIFICATION) not in sys.path:
    sys.path.insert(0, str(M4_VERIFICATION))

from perception.detector import RGBObjectPerception, load_config  # noqa: E402
from perception.geometry import PlanarCalibration  # noqa: E402
from rig import GroundTruthObject, PerceptionRig  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run() -> dict:
    ssot_path = ROOT / "02_Спецификация" / "параметры_системы.yaml"
    model_path = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
    config_path = PROGRAM / "perception" / "perception_config.yaml"
    calibration_path = M4_VERIFICATION / "calibration" / "camera_calibration.json"
    background_path = M4_VERIFICATION / "validation" / "background_reference_rgb.npy"

    ssot = yaml.safe_load(ssot_path.read_text(encoding="utf-8-sig"))
    config = load_config(config_path)
    calibration = PlanarCalibration.from_json(str(calibration_path))
    background = np.load(background_path)

    # Same M4 evaluator-only fixture: a supported red cuboid under the arm pose
    # used by the original occlusion validation. Truth never enters perception.
    gt = GroundTruthObject("evaluator-only-current-m2", "RED", -0.05, -0.28,
                           0.35, (0.90, 0.06, 0.04, 1.0))
    rig = PerceptionRig(ROOT, ssot, include_arm_visual=True)
    try:
        for joint_name, value in (("j1_shoulder", -math.pi / 2),
                                  ("j2_elbow", math.pi / 2),
                                  ("j3_lift", 0.015), ("j4_wrist", 0.0)):
            joint_id = mujoco.mj_name2id(rig.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise RuntimeError(f"Current M2 model lacks {joint_name}")
            rig.data.qpos[int(rig.model.jnt_qposadr[joint_id])] = value
        mujoco.mj_forward(rig.model, rig.data)
        rig.set_objects([gt])
        arm_background = rig.render_empty().rgb
        frame = rig.capture(sim_time_s=2.0)
        detector = RGBObjectPerception(config, calibration, arm_background)
        batch = detector.detect(frame)
    finally:
        rig.close()

    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame.rgb, mode="RGB").save(OUT / "current_m2_arm_occlusion_rgb.png")
    Image.fromarray(batch.candidate_mask).save(OUT / "current_m2_arm_occlusion_mask.png")
    accepted = [d for d in batch.detections if d.status == "VALID"]
    result = {
        "status": "PASS" if not accepted else "FAIL",
        "test": "M4 frozen perception arm-occlusion check against current M2 geometry",
        "m4_profile_version": config.get("config_version"),
        "m4_calibration_id": calibration.calibration_id,
        "ssot_version": ssot.get("config_version"),
        "ssot_sha256": sha256(ssot_path),
        "current_m2_model_sha256": sha256(model_path),
        "original_m4_model_sha256": "d5445d8a6a4cf1a38f533be697d6cd0748d457e4de40d6a16d5240752957c9e2",
        "m4_profile_sha256": sha256(config_path),
        "calibration_sha256": sha256(calibration_path),
        "background_sha256": sha256(background_path),
        "fixture": {"class_label_evaluator_only": gt.class_label,
                    "xy_base_m_evaluator_only": [gt.x_m, gt.y_m],
                    "yaw_rad_evaluator_only": gt.yaw_rad,
                    "observation_joints": {"j1_shoulder": -math.pi/2,
                                           "j2_elbow": math.pi/2,
                                           "j3_lift": 0.015,
                                           "j4_wrist": 0.0}},
        "frame": {"frame_id": frame.frame_id,
                  "simulation_time_s": frame.simulation_time_s,
                  "camera_config_id": frame.camera_config_id,
                  "valid": frame.valid},
        "batch_status": batch.status,
        "detections": [asdict(d) for d in batch.detections],
        "accepted_valid_count": len(accepted),
        "truth_boundary": "Evaluator fixture is used only to place/score the occlusion case; the perception API receives RGB CameraFrame, calibration and background only.",
        "interpretation": "Confirms the frozen M4 detector still rejects the M4 arm-occlusion fixture using the corrected M2 wrist/finger geometry. This targeted regression does not rerun the full M4 classification/localization campaign.",
        "artifacts": ["current_m2_arm_occlusion_rgb.png", "current_m2_arm_occlusion_mask.png"],
    }
    (OUT / "current_m2_arm_occlusion.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if result["status"] != "PASS":
        raise AssertionError("M4 perception accepted an object in the updated M2 arm-occlusion fixture")
    print(json.dumps({k: result[k] for k in ("status", "m4_profile_version", "ssot_version",
                                            "current_m2_model_sha256", "batch_status",
                                            "accepted_valid_count", "artifacts")},
                     ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    run()
