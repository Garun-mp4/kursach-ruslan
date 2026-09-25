"""Deterministic unit and integration tests for the M3 analytic kinematics."""
from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

import mujoco
import numpy as np

from scara import (
    DEFAULT_SSOT,
    IKStatus,
    FKResult,
    Pose,
    check_joint_limits,
    classify_singularity,
    forward_kinematics,
    inverse_kinematics,
    jacobian,
    load_config,
    periodic_difference,
    workspace_bounds,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
ARM_JOINTS = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")


class ScaraKinematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(DEFAULT_SSOT)

    def test_fk_zero_configuration_matches_link_sum_and_vertical_offset(self) -> None:
        cfg = self.config
        result = forward_kinematics((0.0, 0.0, 0.015, 0.0), cfg)
        self.assertAlmostEqual(result.pose.x_m, cfg.l1_m + cfg.l2_m, delta=1e-14)
        self.assertAlmostEqual(result.pose.y_m, 0.0, delta=1e-14)
        self.assertAlmostEqual(result.pose.z_m, cfg.tcp_z_zero_m - 0.015, delta=1e-14)
        self.assertAlmostEqual(result.pose.yaw_rad, 0.0, delta=1e-14)

    def test_fk_quarter_turn_and_wrist_cancel(self) -> None:
        cfg = self.config
        result = forward_kinematics((0.0, math.pi / 2.0, 0.040, -math.pi / 2.0), cfg)
        self.assertAlmostEqual(result.pose.x_m, cfg.base_world_xyz_m[0] + cfg.l1_m, delta=1e-14)
        self.assertAlmostEqual(result.pose.y_m, cfg.base_world_xyz_m[1] + cfg.l2_m, delta=1e-14)
        self.assertAlmostEqual(result.pose.z_m, cfg.tcp_z_zero_m - 0.040, delta=1e-14)
        self.assertAlmostEqual(result.pose.yaw_rad, 0.0, delta=1e-14)

    def test_fk_home_matches_manual_coordinates(self) -> None:
        cfg = self.config
        q1, q2, q3, q4 = cfg.observation_q
        result = forward_kinematics(cfg.observation_q, cfg)
        expected_x = cfg.l1_m * math.cos(q1) + cfg.l2_m * math.cos(q1 + q2)
        expected_y = cfg.l1_m * math.sin(q1) + cfg.l2_m * math.sin(q1 + q2)
        self.assertAlmostEqual(result.pose.x_m, expected_x, delta=1e-14)
        self.assertAlmostEqual(result.pose.y_m, expected_y, delta=1e-14)
        self.assertAlmostEqual(result.pose.z_m, cfg.tcp_z_zero_m - q3, delta=1e-14)
        self.assertAlmostEqual(result.pose.yaw_rad, periodic_difference(q1 + q2 + q4, 0.0), delta=1e-14)

    def test_transform_chain_ends_at_tcp_and_contains_all_frames(self) -> None:
        result = forward_kinematics((0.4, -0.8, 0.03, 0.2), self.config)
        self.assertEqual(set(result.frames_base), {"BASE", "J1", "J2", "J3", "J4", "TCP"})
        np.testing.assert_allclose(result.transform_base_tcp, result.frames_base["TCP"], atol=0.0, rtol=0.0)
        self.assertAlmostEqual(float(np.linalg.det(result.transform_base_tcp[:3, :3])), 1.0, places=14)

    def test_fk_rejects_out_of_limit_joint_without_wrapping(self) -> None:
        with self.assertRaises(ValueError):
            forward_kinematics((2.0 * math.pi, 0.0, 0.02, 0.0), self.config)
        self.assertFalse(check_joint_limits((0.0, math.radians(160), 0.02, 0.0), self.config))

    def test_ik_returns_both_elbow_branches_and_selects_nearest_current(self) -> None:
        q = (0.45, 0.9, 0.035, -0.25)
        target = forward_kinematics(q, self.config).pose
        result = inverse_kinematics(target, self.config, current_q=q, allow_object_symmetry=False)
        self.assertIn(result.status, {IKStatus.SUCCESS, IKStatus.NEAR_SINGULARITY})
        self.assertEqual({c.branch_id for c in result.candidates}, {"ELBOW_POSITIVE", "ELBOW_NEGATIVE"})
        self.assertIsNotNone(result.selected)
        np.testing.assert_allclose(result.selected.q, q, atol=1e-12, rtol=0.0)
        self.assertLessEqual(result.position_residual_m, self.config.position_acceptance_m)
        self.assertLessEqual(result.yaw_residual_rad, self.config.yaw_acceptance_rad)

    def test_ik_square_object_symmetry_selects_equivalent_yaw(self) -> None:
        q = (0.2, 0.7, 0.03, 0.4)
        fk = forward_kinematics(q, self.config)
        target = Pose(fk.pose.x_m, fk.pose.y_m, fk.pose.z_m, fk.pose.yaw_rad + math.pi / 2.0)
        result = inverse_kinematics(target, self.config, current_q=q, allow_object_symmetry=True)
        self.assertIsNotNone(result.selected)
        self.assertLessEqual(result.yaw_residual_rad, self.config.yaw_acceptance_rad)
        self.assertAlmostEqual(result.selected.q[3], q[3], delta=1e-12)

    def test_ik_residual_outside_acceptance_is_reported_as_numerical_failure(self) -> None:
        q = (0.2, 0.7, 0.03, 0.4)
        actual = forward_kinematics(q, self.config)
        shifted = actual.transform_base_tcp.copy()
        shifted[0, 3] += 2.0 * self.config.position_acceptance_m
        incorrect = FKResult(
            Pose(actual.pose.x_m, actual.pose.y_m, actual.pose.z_m,
                 actual.pose.yaw_rad + 2.0 * self.config.yaw_acceptance_rad),
            shifted,
            actual.frames_base,
        )
        with patch("scara.forward_kinematics", return_value=incorrect):
            result = inverse_kinematics(actual.pose, self.config, current_q=q, allow_object_symmetry=False)
        self.assertEqual(result.status, IKStatus.NUMERICAL_FAILURE)
        self.assertIsNotNone(result.selected)
        self.assertGreater(result.position_residual_m, self.config.position_acceptance_m)
        self.assertGreater(result.yaw_residual_rad, self.config.yaw_acceptance_rad)

    def test_ik_reports_orientation_unreachable_for_a_limited_wrist(self) -> None:
        q = (0.2, 0.7, 0.03, 0.4)
        target = forward_kinematics(q, self.config).pose
        limited_wrist = replace(self.config, j4_limits_rad=(-0.1, 0.1))
        result = inverse_kinematics(
            target, limited_wrist, current_q=limited_wrist.observation_q,
            allow_object_symmetry=False,
        )
        self.assertEqual(result.status, IKStatus.ORIENTATION_UNREACHABLE)
        self.assertIsNone(result.selected)

    def test_ik_distant_and_geometric_hole_targets_are_rejected(self) -> None:
        cfg = self.config
        too_far = inverse_kinematics(Pose(0.430001, 0.0, 0.08, 0.0), cfg)
        too_close = inverse_kinematics(Pose(0.029, 0.0, 0.08, 0.0), cfg)
        self.assertEqual(too_far.status, IKStatus.UNREACHABLE_GEOMETRY)
        self.assertEqual(too_close.status, IKStatus.UNREACHABLE_GEOMETRY)
        self.assertIsNone(too_far.selected)
        self.assertIsNone(too_close.selected)

    def test_ik_accepts_only_roundoff_sized_outer_boundary_excess(self) -> None:
        cfg = self.config
        tolerated = inverse_kinematics(Pose(0.43 + 1e-13, 0.0, 0.08, 0.0), cfg)
        rejected = inverse_kinematics(Pose(0.43 + 1e-6, 0.0, 0.08, 0.0), cfg)
        self.assertIsNotNone(tolerated.selected)
        self.assertTrue(tolerated.numeric_boundary_adjustment)
        self.assertEqual(rejected.status, IKStatus.UNREACHABLE_GEOMETRY)

    def test_ik_accepts_only_roundoff_sized_z_boundary_excess(self) -> None:
        z_floor = workspace_bounds(self.config)["tcp_z_min_m"]
        tolerated = inverse_kinematics(Pose(0.30, 0.0, z_floor - 1e-13, 0.0), self.config)
        rejected = inverse_kinematics(Pose(0.30, 0.0, z_floor - 1e-6, 0.0), self.config)
        self.assertIsNotNone(tolerated.selected)
        self.assertTrue(tolerated.numeric_boundary_adjustment)
        self.assertEqual(rejected.status, IKStatus.Z_LIMIT_VIOLATION)

    def test_ik_geometrically_possible_but_joint_limited_inner_region_is_diagnosed(self) -> None:
        result = inverse_kinematics(Pose(0.05, 0.0, 0.08, 0.0), self.config)
        self.assertEqual(result.status, IKStatus.JOINT_LIMIT_VIOLATION)
        self.assertEqual(result.candidates, ())

    def test_ik_outer_and_inner_joint_limited_boundaries(self) -> None:
        cfg = self.config
        bounds = workspace_bounds(cfg)
        outer = inverse_kinematics(Pose(bounds["joint_limited_r_max_m"], 0.0, 0.08, 0.0), cfg)
        inner = inverse_kinematics(Pose(bounds["joint_limited_r_min_m"], 0.0, 0.08, 0.0), cfg)
        self.assertIsNotNone(outer.selected)
        self.assertIsNotNone(inner.selected)
        self.assertAlmostEqual(outer.selected.q[1], 0.0, delta=1e-7)
        self.assertAlmostEqual(abs(inner.selected.q[1]), abs(cfg.j2_limits_rad[1]), delta=1e-7)
        self.assertEqual(outer.status, IKStatus.NEAR_SINGULARITY)

    def test_ik_z_boundaries_and_outside_are_not_silently_clamped(self) -> None:
        cfg = self.config
        bounds = workspace_bounds(cfg)
        for z in (bounds["tcp_z_min_m"], bounds["tcp_z_max_m"]):
            result = inverse_kinematics(Pose(0.30, 0.0, z, 0.0), cfg)
            self.assertIsNotNone(result.selected)
        below = inverse_kinematics(Pose(0.30, 0.0, bounds["tcp_z_min_m"] - 1e-6, 0.0), cfg)
        above = inverse_kinematics(Pose(0.30, 0.0, bounds["tcp_z_max_m"] + 1e-6, 0.0), cfg)
        self.assertEqual(below.status, IKStatus.Z_LIMIT_VIOLATION)
        self.assertEqual(above.status, IKStatus.Z_LIMIT_VIOLATION)

    def test_ik_full_wrist_range_boundary_and_no_symmetry(self) -> None:
        cfg = self.config
        q = (0.2, 0.6, 0.03, math.pi)
        pose = forward_kinematics(q, cfg).pose
        result = inverse_kinematics(pose, cfg, current_q=q, allow_object_symmetry=False)
        self.assertIsNotNone(result.selected)
        self.assertAlmostEqual(abs(result.selected.q[3]), math.pi, delta=1e-12)

    def test_ik_reports_invalid_nan_and_inf_targets(self) -> None:
        for target in ((math.nan, 0.0, 0.08, 0.0), (0.2, math.inf, 0.08, 0.0), (0.2, 0.0, 0.08, math.nan)):
            result = inverse_kinematics(target, self.config)
            self.assertEqual(result.status, IKStatus.INVALID_TARGET)
            self.assertIsNone(result.selected)

    def test_ik_reports_invalid_current_configuration(self) -> None:
        target = Pose(0.30, 0.0, 0.08, 0.0)
        result = inverse_kinematics(target, self.config, current_q=(0.0, math.radians(160), 0.02, 0.0))
        self.assertEqual(result.status, IKStatus.CURRENT_CONFIGURATION_INVALID)
        malformed = inverse_kinematics(target, self.config, current_q=(0.0, 0.0))
        self.assertEqual(malformed.status, IKStatus.CURRENT_CONFIGURATION_INVALID)

    def test_near_straight_is_flagged_but_not_rejected(self) -> None:
        q = (0.4, 0.0, 0.02, 0.1)
        diagnostic = classify_singularity(q, self.config)
        self.assertTrue(diagnostic["near_singular"])
        result = inverse_kinematics(forward_kinematics(q, self.config).pose, self.config, current_q=q)
        self.assertEqual(result.status, IKStatus.NEAR_SINGULARITY)
        self.assertIsNotNone(result.selected)

    def test_folded_configuration_is_outside_physical_j2_range(self) -> None:
        q2 = math.pi
        radius = math.sqrt(self.config.l1_m**2 + self.config.l2_m**2 + 2 * self.config.l1_m * self.config.l2_m * math.cos(q2))
        result = inverse_kinematics(Pose(radius, 0.0, 0.08, 0.0), self.config)
        self.assertEqual(result.status, IKStatus.JOINT_LIMIT_VIOLATION)

    def test_analytic_jacobian_matches_finite_differences(self) -> None:
        rng = np.random.default_rng(20260924)
        for _ in range(40):
            q = tuple(float(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)))
                      for lo, hi in self.config.q_limits)
            analytic = jacobian(q, self.config)
            numeric = np.zeros((4, 4), dtype=float)
            step = 1e-7
            for i in range(4):
                plus, minus = list(q), list(q)
                plus[i] += step
                minus[i] -= step
                p_plus = forward_kinematics(plus, self.config).pose
                p_minus = forward_kinematics(minus, self.config).pose
                delta = p_plus.as_array() - p_minus.as_array()
                delta[3] = periodic_difference(p_plus.yaw_rad, p_minus.yaw_rad)
                numeric[:, i] = delta / (2.0 * step)
            np.testing.assert_allclose(analytic, numeric, rtol=2e-6, atol=2e-8)

    def test_engine_parity_for_seeded_joint_states(self) -> None:
        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        rng = np.random.default_rng(71103)
        for _ in range(80):
            q = tuple(float(rng.uniform(lo, hi)) for lo, hi in self.config.q_limits)
            data.qpos[:] = model.qpos0
            for name, value in zip(ARM_JOINTS, q):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[jid]] = value
            mujoco.mj_forward(model, data)
            manual = forward_kinematics(q, self.config)
            np.testing.assert_allclose(manual.transform_base_tcp[:3, 3], data.site_xpos[tcp_id], atol=2e-12, rtol=0.0)
            engine_yaw = math.atan2(data.site_xmat[tcp_id].reshape(3, 3)[1, 0], data.site_xmat[tcp_id].reshape(3, 3)[0, 0])
            self.assertAlmostEqual(periodic_difference(manual.pose.yaw_rad, engine_yaw), 0.0, delta=2e-12)

    def test_all_named_input_and_sort_slots_are_kinematically_reachable(self) -> None:
        import yaml
        with DEFAULT_SSOT.open("r", encoding="utf-8-sig") as stream:
            p = yaml.safe_load(stream)["parameters"]
        values = {key: item["value"] for key, item in p.items()}
        input_center = values["cell.input_center_xy_m"]
        targets = []
        for dy in values["cell.input_row_y_offsets_m"]:
            for dx in values["cell.input_slot_x_offsets_m"]:
                targets.append((input_center[0] + dx, input_center[1] + dy, values["robot.object_pick_center_z_m"], 0.0))
        for color, center in values["cell.tray_centers_xy_m"].items():
            for dx in values["cell.tray_slot_x_offsets_m"]:
                targets.append((center[0] + dx, center[1] + values["cell.tray_slot_y_offset_m"], values["robot.object_place_center_z_m"], 0.0))
        self.assertEqual(len(targets), 15)
        for target in targets:
            result = inverse_kinematics(target, self.config)
            self.assertIsNotNone(result.selected, msg=f"unreachable configured slot: {target}; {result.status}: {result.reason}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
