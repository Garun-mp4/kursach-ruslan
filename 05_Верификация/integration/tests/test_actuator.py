from __future__ import annotations

import unittest
from types import SimpleNamespace

import mujoco

from integration.actuator import MuJoCoPositionExecutor
from integration.config import MODEL_PATH, RuntimeConfig


class PositionActuatorPhysicsTests(unittest.TestCase):
    def test_vertical_axis_holds_target_against_gravity_with_finite_effort(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        executor = MuJoCoPositionExecutor(model, data, RuntimeConfig.load())

        j3_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_j3_lift")
        self.assertGreater(j3_id, -1)
        self.assertAlmostEqual(float(model.actuator_gainprm[j3_id, 0]), 3000.0)
        self.assertAlmostEqual(float(model.actuator_biasprm[j3_id, 2]), -20.0)
        self.assertTrue(bool(model.actuator_forcelimited[j3_id]))
        self.assertEqual(tuple(map(float, model.actuator_forcerange[j3_id])), (-4.0, 4.0))

        target_m = 0.043
        targets = executor._last_targets.copy()
        targets[2] = target_m
        executor._set_targets(targets)
        max_force_N = 0.0
        for _ in range(3000):
            mujoco.mj_step(model, data)
            max_force_N = max(max_force_N, abs(float(data.actuator_force[j3_id])))

        self.assertLessEqual(max_force_N, 4.0 + 1e-9)
        self.assertLessEqual(abs(executor.joint_state()[2] - target_m), 0.0005)

    def test_gripper_preload_stays_inside_joint_range_and_force_limit(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        executor = MuJoCoPositionExecutor(
            model, data, RuntimeConfig.load(), gripper_preload_position_offset_m=0.011,
        )
        gripper_joint = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "j5_finger_left"
        )
        gripper_actuator = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper_left"
        )
        self.assertEqual(tuple(map(float, model.jnt_range[gripper_joint])), (0.0, 0.027))
        self.assertEqual(tuple(map(float, model.actuator_ctrlrange[gripper_actuator])), (0.0, 0.027))

        sample = SimpleNamespace(
            phase="LIFT", time_s=0.0, q=executor.joint_state(),
            dq=(0.0, 0.0, 0.0, 0.0), ddq=(0.0, 0.0, 0.0, 0.0),
            gripper_m=0.015998, gripper_velocity_m_s=0.0,
            gripper_acceleration_m_s2=0.0, tcp_xyzyaw=(0.0, 0.0, 0.0, 0.0),
            payload_mode="carried", event=None,
        )
        command = SimpleNamespace(
            command_id="test-carried-preload", phase_names=("LIFT",),
            plan=SimpleNamespace(samples=(sample,)),
        )
        executor.start(command, 0.0)
        executor.apply(0.0)
        self.assertAlmostEqual(float(data.ctrl[gripper_actuator]), 0.026998, places=8)
        self.assertFalse(executor._gripper_position_tracking_required())

        maximum_effort = 0.0
        completion = None
        for _ in range(1000):
            mujoco.mj_step(model, data)
            maximum_effort = max(maximum_effort, abs(float(data.actuator_force[gripper_actuator])))
            completion = executor.feedback(float(data.time))
        self.assertLessEqual(executor.gripper_position_m(), 0.027 + 1e-9)
        self.assertLessEqual(maximum_effort, 1.2 + 1e-9)
        self.assertIsNotNone(completion)
        self.assertEqual(completion.status.value, "COMPLETED")

    def test_unheld_gripper_still_requires_position_tracking(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)
        executor = MuJoCoPositionExecutor(model, data, RuntimeConfig.load())
        executor._samples = (SimpleNamespace(payload_mode="unheld"),)
        executor._last_sample_index = 0
        self.assertTrue(executor._gripper_position_tracking_required())


if __name__ == "__main__":
    unittest.main()
