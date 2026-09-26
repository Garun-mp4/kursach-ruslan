from __future__ import annotations

import sys
import unittest
from pathlib import Path

import mujoco

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04_Программа"))

from integration.config import RuntimeConfig, load_ssot, value
from integration.placement_camera import load_integrated_model
from integration.scene import SceneGenerator, load_scenario
from integration.sensors import PublicSensorPipeline
from kinematics.scara import load_config as load_arm_config
from planning.context import load_context
from planning.planner import Planner


class MixedSceneStartupPlanTests(unittest.TestCase):
    def test_public_green_detection_in_seeded_mixed_scene_is_plannable_from_home(self):
        ssot = load_ssot()
        runtime_config = RuntimeConfig.load()
        model = load_integrated_model(ssot)
        data = mujoco.MjData(model)
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_observation")
        self.assertGreaterEqual(home_id, 0)
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        SceneGenerator(ssot, runtime_config).generate(
            model, data, load_scenario("mixed_random_three")
        )

        sensors = PublicSensorPipeline(model)
        try:
            snapshot = sensors.capture(data, simulation_time_s=float(data.time))
        finally:
            sensors.close()

        self.assertTrue(snapshot.frame.valid)
        self.assertEqual(snapshot.input_batch.status, "OK")
        green = [item for item in snapshot.input_batch.detections
                 if item.class_label == "GREEN" and item.status == "VALID"]
        self.assertEqual(len(green), 1, snapshot.input_batch.detections)

        arm = load_arm_config()
        joint_names = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")
        current_q = tuple(
            float(data.qpos[int(model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )])])
            for name in joint_names
        )
        self.assertEqual(current_q, arm.observation_q)
        available_slots = tuple(
            f"{color}:{index}"
            for color in ("RED", "GREEN", "BLUE")
            for index in range(3)
        )
        result = Planner(load_context()).plan_cycle(
            snapshot.input_batch,
            green[0].track_id,
            current_q,
            float(value(ssot, "robot.finger_home_m")),
            available_slots,
            snapshot.input_batch.simulation_time_s,
        )

        self.assertTrue(
            result.success,
            f"The M4-observed green object in the baseline mixed scene must be plannable "
            f"from the configured home pose; got {result.code.value} during {result.phase}: "
            f"{result.reason}; diagnostics={result.diagnostics}",
        )


if __name__ == "__main__":
    unittest.main()
