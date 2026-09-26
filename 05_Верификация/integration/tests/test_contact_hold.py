from __future__ import annotations

import unittest

import mujoco
import numpy as np

from integration.config import MODEL_PATH, load_ssot, value


class ContactHoldPhysicsTests(unittest.TestCase):
    def _measure_stationary_grasp_drift(self, noslip_iterations: int | None) -> tuple[float, float, float]:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        if noslip_iterations is not None:
            model.opt.noslip_iterations = noslip_iterations
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, 0)

        arm_joints = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")
        arm_actuators = (
            "act_j1_shoulder", "act_j2_elbow", "act_j3_lift", "act_j4_wrist",
        )
        for joint_name, actuator_name in zip(arm_joints, arm_actuators):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
            data.ctrl[actuator_id] = data.qpos[int(model.jnt_qposadr[joint_id])]

        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
        object_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object_red_01")
        object_joint_id = int(model.body_jntadr[object_id])
        object_qpos_adr = int(model.jnt_qposadr[object_joint_id])
        object_dof_adr = int(model.jnt_dofadr[object_joint_id])
        mujoco.mj_forward(model, data)

        ssot = load_ssot()
        wrist_rotation = np.asarray(data.xmat[wrist_id], dtype=float).reshape(3, 3)
        wrist_position = np.asarray(data.xpos[wrist_id], dtype=float)
        object_center_local = np.asarray(
            [0.0, 0.0, float(value(ssot, "robot.finger_center_in_wrist_m")[2])],
            dtype=float,
        )
        object_position = wrist_position + wrist_rotation @ object_center_local
        object_quaternion = np.asarray(data.xquat[wrist_id], dtype=float)
        data.qpos[object_qpos_adr:object_qpos_adr + 7] = [
            *object_position, *object_quaternion,
        ]

        jaw_contact_position = (
            float(value(ssot, "robot.finger_open_center_offset_m"))
            - (float(value(ssot, "robot.finger_thickness_m"))
               + float(value(ssot, "object.size_xyz_m")[1])) / 2.0
        )
        for joint_name in ("j5_finger_left", "j6_finger_right"):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[int(model.jnt_qposadr[joint_id])] = jaw_contact_position
            data.qvel[int(model.jnt_dofadr[joint_id])] = 0.0

        gripper_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_gripper_left",
        )
        data.ctrl[gripper_id] = float(value(ssot, "robot.finger_range_m")[1])
        mujoco.mj_forward(model, data)

        # Let the finite-force servo close against the object before measuring drift.
        for _ in range(round(0.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
        reference = np.asarray(data.xpos[object_id], dtype=float).copy()
        start_rotation = np.asarray(data.xmat[wrist_id], dtype=float).reshape(3, 3)
        relative_start = start_rotation.T @ (
            reference - np.asarray(data.xpos[wrist_id], dtype=float)
        )

        for _ in range(round(2.0 / model.opt.timestep)):
            mujoco.mj_step(model, data)

        end_rotation = np.asarray(data.xmat[wrist_id], dtype=float).reshape(3, 3)
        relative_end = end_rotation.T @ (
            np.asarray(data.xpos[object_id], dtype=float)
            - np.asarray(data.xpos[wrist_id], dtype=float)
        )
        touch_ids = tuple(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            for name in ("touch_left", "touch_right")
        )
        touch_forces = tuple(
            float(data.sensordata[int(model.sensor_adr[sensor_id])])
            for sensor_id in touch_ids
        )
        self.assertGreater(min(touch_forces), 0.001)
        self.assertLessEqual(abs(float(data.actuator_force[gripper_id])),
                             float(value(ssot, "robot.gripper_actuator_total_force_N")) + 1e-9)
        return (
            float(np.linalg.norm(relative_end - relative_start)),
            float(relative_end[2] - relative_start[2]),
            min(touch_forces),
        )

    def test_configured_noslip_solver_prevents_slow_grasp_creep(self):
        configured_iterations = int(value(load_ssot(), "environment.noslip_iterations"))
        self.assertGreater(configured_iterations, 0)

        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.assertEqual(model.opt.noslip_iterations, configured_iterations)
        regularized_drift, _, _ = self._measure_stationary_grasp_drift(0)
        configured_drift, vertical_drift, minimum_touch = self._measure_stationary_grasp_drift(
            None,
        )

        self.assertGreater(regularized_drift, 0.001)
        self.assertLess(configured_drift, 0.0001)
        self.assertLess(abs(vertical_drift), 0.0001)
        self.assertGreater(minimum_touch, 0.001)


if __name__ == "__main__":
    unittest.main()
