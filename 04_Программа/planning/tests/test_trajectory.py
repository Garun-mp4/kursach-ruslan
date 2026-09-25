from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROGRAM = Path(__file__).resolve().parents[2]
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from planning.context import load_context
from planning.trajectory import build_trajectory
from planning.records import CartesianWaypoint, Phase
from planning.run_m5_verification import tcp_kinematic_metrics
from kinematics.scara import forward_kinematics, jacobian


class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = load_context()

    def test_quintic_phase_parameterization_respects_joint_and_gripper_limits(self):
        ctx = self.context
        q0 = tuple(ctx.arm.observation_q)
        q1 = list(q0)
        q1[2] = 0.060
        q1 = tuple(q1)
        qg_open = float(ctx.p("robot.finger_home_m"))
        qg_closed = 0.010
        p0 = forward_kinematics(q0, ctx.arm).pose
        p1 = forward_kinematics(q1, ctx.arm).pose

        def waypoint(phase, q, pose, gripper, payload="unheld", event=None):
            return CartesianWaypoint(phase, (pose.x_m, pose.y_m, pose.z_m, pose.yaw_rad),
                                     q, "ELBOW_POSITIVE", 0, gripper, payload, event)

        points = (
            waypoint(Phase.INITIAL_APPROACH.value, q0, p0, qg_open),
            waypoint(Phase.DESCENT.value, q1, p1, qg_open),
            waypoint(Phase.GRASP_CLOSE.value, q1, p1, qg_closed,
                     event="GRIPPER_CLOSE_COMPLETE"),
            waypoint(Phase.LIFT.value, q1, p1, qg_closed, "carried", "OBJECT_ATTACHED"),
        )
        samples, metrics = build_trajectory(points, ctx)

        self.assertGreater(len(samples), 10)
        self.assertAlmostEqual(samples[0].time_s, 0.0, places=12)
        self.assertAlmostEqual(samples[-1].time_s, metrics["duration_s"], places=9)
        self.assertTrue(all(b.time_s >= a.time_s for a, b in zip(samples, samples[1:])))
        self.assertLessEqual(metrics["max_velocity_ratio"], 1.0 + 1e-10)
        self.assertLessEqual(metrics["max_acceleration_ratio"], 1.0 + 1e-10)
        self.assertLessEqual(metrics["max_swept_sample_bound_m"],
                             float(ctx.p("planning.path_sweep_sample_step_m")) + 1e-9)
        self.assertEqual([s.event for s in samples if s.event],
                         ["GRIPPER_CLOSE_COMPLETE", "OBJECT_ATTACHED"])
        self.assertTrue(all(ctx.arm.q_limits[i][0] - 1e-12 <= s.q[i] <=
                            ctx.arm.q_limits[i][1] + 1e-12
                            for s in samples for i in range(4)))
        finger_low, finger_high = map(float, ctx.p("robot.finger_range_m"))
        self.assertTrue(all(finger_low <= s.gripper_m <= finger_high for s in samples))

        descent = [s for s in samples if s.phase == Phase.DESCENT.value]
        self.assertTrue(descent)
        start_xy = descent[0].tcp_xyzyaw[:2]
        xy_error = max(math.dist(start_xy, sample.tcp_xyzyaw[:2]) for sample in descent)
        self.assertLessEqual(xy_error, float(ctx.p("planning.distance_numeric_tolerance_m")))

    def test_invalid_or_nonpositive_sweep_step_is_rejected(self):
        ctx = self.context
        q = tuple(ctx.arm.observation_q)
        pose = forward_kinematics(q, ctx.arm).pose
        point = CartesianWaypoint(Phase.INITIAL_APPROACH.value,
                                  (pose.x_m, pose.y_m, pose.z_m, pose.yaw_rad),
                                  q, "ELBOW_POSITIVE", 0,
                                  float(ctx.p("robot.finger_home_m")), "unheld")
        with self.assertRaises(ValueError):
            build_trajectory((point, point), ctx, sample_step_override_m=0.0)

    def test_tcp_fk_velocity_and_acceleration_match_jacobian_chain_rule(self):
        ctx = self.context
        q = np.asarray(ctx.arm.observation_q, dtype=float)
        dq = np.asarray((0.12, -0.18, 0.008, 0.21), dtype=float)
        ddq = np.asarray((0.24, -0.31, 0.016, 0.19), dtype=float)
        sample = type("Sample", (), {"q": tuple(q), "dq": tuple(dq), "ddq": tuple(ddq)})()
        metrics = tcp_kinematic_metrics(sample, ctx)
        fk_jacobian = jacobian(q, ctx.arm)
        expected_velocity = fk_jacobian[:3, :] @ dq
        self.assertAlmostEqual(metrics["tcp_linear_speed_m_s"],
                               float(np.linalg.norm(expected_velocity)), places=12)
        self.assertAlmostEqual(metrics["tcp_yaw_rate_rad_s"],
                               float(fk_jacobian[3, :] @ dq), places=12)

        h = 1e-4
        def position(coordinates):
            pose = forward_kinematics(coordinates, ctx.arm).pose
            return np.asarray((pose.x_m, pose.y_m, pose.z_m))

        p0 = position(q)
        p_plus = position(q+h*dq)
        p_minus = position(q-h*dq)
        expected_acceleration = (p_plus-2*p0+p_minus)/(h*h) + fk_jacobian[:3, :] @ ddq
        self.assertAlmostEqual(metrics["tcp_linear_acceleration_m_s2"],
                               float(np.linalg.norm(expected_acceleration)), places=5)
        self.assertLessEqual(metrics["tcp_linear_speed_m_s"],
                             metrics["tcp_linear_speed_joint_limit_bound_m_s"] + 1e-12)
        self.assertLessEqual(metrics["tcp_linear_acceleration_m_s2"],
                             metrics["tcp_linear_acceleration_joint_limit_bound_m_s2"] + 1e-12)
        self.assertLessEqual(abs(metrics["tcp_yaw_rate_rad_s"]),
                             metrics["tcp_yaw_rate_joint_limit_bound_rad_s"] + 1e-12)
        self.assertLessEqual(abs(metrics["tcp_yaw_acceleration_rad_s2"]),
                             metrics["tcp_yaw_acceleration_joint_limit_bound_rad_s2"] + 1e-12)


if __name__ == "__main__":
    unittest.main()
