from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

PROGRAM = Path(__file__).resolve().parents[2]
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from perception.types import Detection, DetectionBatch
from planning.context import load_context
from planning.planner import Planner
from planning.records import PlanCode, Phase
from kinematics.scara import forward_kinematics, inverse_kinematics


def detection(*, track_id=7, color="RED", xy=(0.0, -0.25), yaw=0.0,
              sigma_xy=0.001, sigma_yaw=math.radians(2), stamp=1.0, status="VALID"):
    return Detection(track_id=track_id, class_label=color, xy_base_m=xy,
                     yaw_base_rad=yaw, confidence=0.99, position_sigma_m=sigma_xy,
                     yaw_sigma_rad=sigma_yaw, status=status, reason=None,
                     frame_id=41, simulation_time_s=stamp, area_px=225.0,
                     bbox_xywh_px=(300, 200, 15, 15), center_uv_px=(307.5, 207.5),
                     frame_age_s=0.0)


def batch(item, *, stamp=1.0, status="OK"):
    return DetectionBatch(status=status, reason=None, frame_id=41,
                          simulation_time_s=stamp, detections=tuple(item))


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = load_context()
        cls.planner = Planner(cls.context)
        cls.open_gripper = float(cls.context.p("robot.finger_home_m"))
        cls.q_start = tuple(cls.context.arm.observation_q)

    def test_both_ik_branches_are_available_and_scored_without_branch_switch(self):
        q_source = (0.45, 0.9, 0.035, -0.25)
        pose = forward_kinematics(q_source, self.context.arm).pose
        candidates = inverse_kinematics(pose, self.context.arm, q_source,
                                       allow_object_symmetry=False).candidates
        self.assertEqual({candidate.branch_id for candidate in candidates},
                         {"ELBOW_POSITIVE", "ELBOW_NEGATIVE"})
        for candidate in candidates:
            selected, cost = self.planner._solve(pose, np.asarray(candidate.q),
                                                 candidate.branch_id)
            self.assertEqual("ELBOW_POSITIVE" if selected[1] > 0 else "ELBOW_NEGATIVE",
                             candidate.branch_id)
            self.assertGreaterEqual(cost, 0.0)
            self.assertTrue(np.allclose(selected, candidate.q, atol=1e-10, rtol=0.0))

    def test_unknown_target_is_rejected_without_a_motion_plan(self):
        result = self.planner.plan_cycle(
            batch((detection(color="UNKNOWN", status="UNKNOWN"),)), 7,
            self.q_start, self.open_gripper, ("RED:0",), 1.0,
        )
        self.assertEqual(result.code, PlanCode.UNSUPPORTED_CLASS)
        self.assertIsNone(result.plan)

    def test_stale_perception_is_rejected_fail_closed(self):
        result = self.planner.plan_cycle(batch((detection(stamp=0.0),), stamp=0.0), 7,
            self.q_start, self.open_gripper, ("RED:0",), 1.1)
        self.assertEqual(result.code, PlanCode.STALE_DETECTION)
        self.assertIsNone(result.plan)

    def test_no_explicit_free_slot_is_rejected_before_pick(self):
        result = self.planner.plan_cycle(batch((detection(),)), 7,
            self.q_start, self.open_gripper, (), 1.0)
        self.assertEqual(result.code, PlanCode.NO_AVAILABLE_SLOT)
        self.assertIsNone(result.plan)

    def test_uncertainty_over_budget_is_rejected_before_pick(self):
        result = self.planner.plan_cycle(batch((detection(sigma_xy=0.01),)), 7,
            self.q_start, self.open_gripper, ("RED:0",), 1.0)
        self.assertEqual(result.code, PlanCode.SCENE_UNCERTAIN)
        self.assertIsNone(result.plan)

    def test_invalid_obstacle_track_makes_the_scene_incomplete(self):
        target = detection()
        obstacle = detection(track_id=9, color="UNKNOWN", xy=(0.07, -0.25), status="OCCLUDED")
        result = self.planner.plan_cycle(batch((target, obstacle)), 7,
            self.q_start, self.open_gripper, ("RED:1",), 1.0)
        self.assertEqual(result.code, PlanCode.SCENE_UNCERTAIN)
        self.assertIsNone(result.plan)

    def test_unknown_color_with_valid_geometry_remains_a_collision_obstacle(self):
        target = detection()
        unknown = detection(track_id=9, color="UNKNOWN", xy=(0.13, -0.34), status="UNKNOWN")
        selected, tracks = self.planner._validate_inputs(
            batch((target, unknown)), 7, 1.0,
            np.asarray(self.q_start, dtype=float), self.open_gripper
        )
        self.assertEqual(selected.track_id, 7)
        self.assertEqual({item.track_id for item in tracks}, {7, 9})
        self.assertEqual(next(item for item in tracks if item.track_id == 9).class_label, "UNKNOWN")

    def test_unknown_target_remains_unselectable(self):
        unknown = detection(color="UNKNOWN", status="UNKNOWN")
        result = self.planner.plan_cycle(batch((unknown,)), 7,
            self.q_start, self.open_gripper, ("RED:0",), 1.0)
        self.assertEqual(result.code, PlanCode.UNSUPPORTED_CLASS)
        self.assertIsNone(result.plan)

    def test_full_cycle_is_preflighted_before_plan_success(self):
        item = detection()
        result = self.planner.plan_cycle(batch((item,)), 7, self.q_start,
            self.open_gripper, ("RED:0", "RED:1", "RED:2"), 1.0,
            sample_step_override_m=0.002)
        self.assertTrue(result.success, (result.code, result.reason, result.phase,
                                         len(result.diagnostics.get("attempts", [])),
                                         result.diagnostics.get("selected_diagnostics")))
        plan = result.plan
        self.assertEqual(plan.target_track_id, 7)
        self.assertEqual(plan.selected_ik_branch, self.planner._current_branch(np.asarray(self.q_start)))
        required = {phase.value for phase in Phase}
        self.assertTrue(required.issubset(set(plan.phases)))
        phase_order = {phase.value: index for index, phase in enumerate(Phase)}
        sample_phase_indexes = [phase_order[sample.phase] for sample in plan.samples]
        self.assertTrue(all(
            current <= following
            for current, following in zip(sample_phase_indexes, sample_phase_indexes[1:])
        ), "M5 trajectory samples must follow the ordered phase contract")
        self.assertEqual(plan.samples[-1].phase, Phase.SAFE_HOME.value)
        self.assertEqual([event.event for event in plan.events],
                         ["GRIPPER_CLOSE_COMPLETE", "OBJECT_ATTACHED", "RELEASE_BEGIN", "OBJECT_RELEASED"])
        self.assertLessEqual(plan.maximum_abs_velocity_ratio, 1.0 + 1e-10)
        self.assertLessEqual(plan.maximum_abs_acceleration_ratio, 1.0 + 1e-10)
        self.assertLessEqual(plan.vertical_xy_error_max_m,
                             float(self.context.p("planning.distance_numeric_tolerance_m")))
        self.assertLessEqual(plan.vertical_yaw_error_max_rad,
                             float(self.context.p("kinematics.yaw_acceptance_rad")))
        self.assertGreaterEqual(plan.clearance_lower_bound_m, 0.0)
        self.assertTrue(np.allclose(plan.samples[-1].q, self.q_start, atol=1e-10, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
