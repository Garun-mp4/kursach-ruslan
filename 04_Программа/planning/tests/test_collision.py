from __future__ import annotations

import math
import tempfile
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

PROGRAM = Path(__file__).resolve().parents[2]
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from perception.types import Detection
from planning.collision import CollisionScene
from planning.context import MODEL_PATH, load_context
from kinematics.scara import Pose, forward_kinematics, inverse_kinematics


def observation(track_id=1, xy=(0.0, -0.25), stamp=1.0):
    return Detection(track_id=track_id, class_label="RED", xy_base_m=xy, yaw_base_rad=0.0,
                     confidence=0.99, position_sigma_m=0.001, yaw_sigma_rad=math.radians(2),
                     status="VALID", reason=None, frame_id=1, simulation_time_s=stamp,
                     area_px=196.0, bbox_xywh_px=(100, 100, 14, 14),
                     center_uv_px=(107.0, 107.0), frame_age_s=0.0)


class CollisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = load_context()
        cls.track = observation()

    def test_thin_static_obstacle_is_missed_at_endpoints_but_caught_mid_sweep(self):
        ctx = self.context
        q0 = list(ctx.arm.observation_q)
        q1 = list(q0)
        q0[2], q1[2] = 0.015, 0.080
        qmid = [(a + b) / 2.0 for a, b in zip(q0, q1)]

        model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        for name, value in zip(("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist"), qmid):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = value
        for name in ("j5_finger_left", "j6_finger_right"):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = float(ctx.p("robot.finger_home_m"))
        mujoco.mj_forward(model, data)
        finger_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                        "finger_left_collision")
        center = data.geom_xpos[finger_geom]
        center = center.copy()
        # Keep the probe inside the finger's mid-sweep envelope while clearing
        # the wrist/palm's static margin at the upper endpoint.
        center[2] += 0.004

        tree = ET.parse(MODEL_PATH)
        worldbody = tree.getroot().find("worldbody")
        ET.SubElement(worldbody, "geom", {
            "name": "m5_thin_obstacle", "type": "box",
            "pos": " ".join(f"{v:.12g}" for v in center),
            "size": "0.002 0.002 0.0005", "contype": "1", "conaffinity": "1",
            "rgba": "0.9 0.15 0.10 1",
        })

        with tempfile.TemporaryDirectory(prefix="m5_collision_") as directory:
            model_path = Path(directory) / "thin_obstacle.xml"
            tree.write(model_path, encoding="utf-8", xml_declaration=True)
            scene = CollisionScene(ctx, model_path)
            def check(q):
                return scene.check(q, gripper_m=float(ctx.p("robot.finger_home_m")),
                                   detections=(self.track,), target_track_id=self.track.track_id,
                                   phase="TRANSFER", payload_mode="unheld",
                                   sample_step_bound_m=0.001)

            at_start, at_middle, at_end = check(q0), check(qmid), check(q1)

        self.assertTrue(at_start.valid, at_start.violations)
        self.assertFalse(at_middle.valid)
        self.assertTrue(any("m5_thin_obstacle" in (v.geom_a, v.geom_b)
                            for v in at_middle.violations))
        self.assertTrue(at_end.valid, at_end.violations)

    def test_signed_distance_report_keeps_a_conservative_global_lower_bound(self):
        scene = CollisionScene(self.context)
        q = self.context.arm.observation_q
        report = scene.check(q, gripper_m=float(self.context.p("robot.finger_home_m")),
                             detections=(self.track,), target_track_id=self.track.track_id,
                             phase="INITIAL_APPROACH", payload_mode="unheld")
        self.assertTrue(report.valid, report.violations)
        self.assertGreaterEqual(report.clearance_lower_bound_m, 0.0)
        self.assertLessEqual(report.clearance_lower_bound_m, report.minimum_clearance_m + 1e-12)
        self.assertGreater(report.checked_pairs, 0)

    def _pick_q(self, z_m=0.008):
        target = Pose(0.0, -0.25, float(z_m), 0.0)
        result = inverse_kinematics(target, self.context.arm, self.context.arm.observation_q,
                                    allow_object_symmetry=False)
        self.assertEqual(result.status.value, "SUCCESS", result.reason)
        candidate = next(c for c in result.candidates if c.branch_id == "ELBOW_POSITIVE")
        return tuple(candidate.q)

    def _pose_q(self, x_m, y_m, z_m):
        target = Pose(float(x_m), float(y_m), float(z_m), 0.0)
        result = inverse_kinematics(target, self.context.arm, self.context.arm.observation_q,
                                    allow_object_symmetry=False)
        self.assertEqual(result.status.value, "SUCCESS", result.reason)
        candidate = next(c for c in result.candidates if c.branch_id == "ELBOW_POSITIVE")
        return tuple(candidate.q)

    def test_centered_fingers_clear_the_table_only_during_target_grasp_phases(self):
        scene = CollisionScene(self.context)
        q = self._pick_q()
        open_width = float(self.context.p("robot.finger_home_m"))
        table_limit = (float(self.context.p("planning.grasp_finger_support_clearance_m"))
                       + float(self.context.p("planning.path_sweep_sample_step_m"))
                       + float(self.context.p("planning.distance_numeric_tolerance_m")))

        for phase, payload_mode in (("DESCENT", "unheld"),
                                    ("GRASP_CLOSE", "unheld"),
                                    ("LIFT", "carried")):
            with self.subTest(phase=phase):
                report = scene.check(q, gripper_m=open_width, detections=(self.track,),
                                     target_track_id=self.track.track_id, phase=phase,
                                     payload_mode=payload_mode, sample_step_bound_m=0.001)
                self.assertTrue(report.valid, report.violations)
                finger_table = [c for c in report.allowed_contacts
                                if {c["geom_a"], c["geom_b"]}
                                == {"finger_left_collision", "table_top_collision"}
                                or {c["geom_a"], c["geom_b"]}
                                == {"finger_right_collision", "table_top_collision"}]
                self.assertEqual(finger_table, [], "Positive clearance must not be mislabeled as contact")
                self.assertGreaterEqual(report.clearance_lower_bound_m, table_limit - 1e-12)

        for phase, detection in (
            ("TRANSFER", self.track),
            ("DESCENT", observation(xy=(0.01, -0.25))),
        ):
            with self.subTest(phase=phase, xy=detection.xy_base_m):
                report = scene.check(q, gripper_m=open_width, detections=(detection,),
                                     target_track_id=detection.track_id, phase=phase,
                                     payload_mode="unheld", sample_step_bound_m=0.001)
                finger_table = [v for v in report.violations
                                if {v.geom_a, v.geom_b}
                                in ({"finger_left_collision", "table_top_collision"},
                                    {"finger_right_collision", "table_top_collision"})]
                self.assertFalse(report.valid)
                self.assertEqual(len(finger_table), 2, report.violations)
                self.assertTrue(all(abs(v.required_m - (float(self.context.p("planning.static_clearance_m"))
                                                        + 0.001
                                                        + float(self.context.p("planning.distance_numeric_tolerance_m"))))
                                    < 1e-12 for v in finger_table))

    def test_grasp_clearance_does_not_allow_table_penetration_or_relax_other_obstacles(self):
        tree = ET.parse(MODEL_PATH)
        table = next(g for g in tree.getroot().iter("geom") if g.get("name") == "table_top_collision")
        table_position = [float(v) for v in table.get("pos").split()]
        table_position[2] += 0.004
        table.set("pos", " ".join(f"{v:.12g}" for v in table_position))
        with tempfile.TemporaryDirectory(prefix="m7_table_penetration_") as directory:
            model_path = Path(directory) / "penetrating_table.xml"
            tree.write(model_path, encoding="utf-8", xml_declaration=True)
            scene = CollisionScene(self.context, model_path)
            report = scene.check(self._pick_q(), gripper_m=float(self.context.p("robot.finger_home_m")),
                                 detections=(self.track,), target_track_id=self.track.track_id,
                                 phase="DESCENT", payload_mode="unheld")
        table_violations = [v for v in report.violations
                            if v.geom_b == "table_top_collision"
                            and v.geom_a in {"finger_left_collision", "finger_right_collision"}]
        self.assertFalse(report.valid)
        self.assertEqual(len(table_violations), 2, report.violations)
        self.assertTrue(all(v.distance_m < -float(self.context.p("planning.distance_numeric_tolerance_m"))
                            for v in table_violations))
        self.assertTrue(all(abs(v.required_m - (float(self.context.p("planning.grasp_finger_support_clearance_m"))
                                                 + float(self.context.p("planning.path_sweep_sample_step_m"))
                                                 + float(self.context.p("planning.distance_numeric_tolerance_m"))))
                            < 1e-12 for v in table_violations))

        tree = ET.parse(MODEL_PATH)
        worldbody = tree.getroot().find("worldbody")
        ET.SubElement(worldbody, "geom", {
            "name": "m7_clearance_probe", "type": "box", "pos": "0 -0.25 0.0195",
            "size": "0.002 0.002 0.001", "contype": "1", "conaffinity": "1",
            "rgba": "0.9 0.15 0.10 1",
        })
        with tempfile.TemporaryDirectory(prefix="m7_clearance_") as directory:
            model_path = Path(directory) / "clearance_probe.xml"
            tree.write(model_path, encoding="utf-8", xml_declaration=True)
            probe_scene = CollisionScene(self.context, model_path)
            probe_report = probe_scene.check(self._pick_q(),
                                             gripper_m=float(self.context.p("robot.finger_home_m")),
                                             detections=(self.track,), target_track_id=self.track.track_id,
                                             phase="DESCENT", payload_mode="unheld")

        probe_violations = [v for v in probe_report.violations
                            if "m7_clearance_probe" in {v.geom_a, v.geom_b}]
        expected_static = (float(self.context.p("planning.static_clearance_m"))
                           + float(self.context.p("planning.path_sweep_sample_step_m"))
                           + float(self.context.p("planning.distance_numeric_tolerance_m")))
        self.assertTrue(probe_violations, probe_report.violations)
        self.assertTrue(all(abs(v.required_m - expected_static) < 1e-12 for v in probe_violations))

    def test_finger_tray_floor_margin_is_limited_to_reserved_place_and_vertical_retreat(self):
        centers = self.context.p("cell.tray_centers_xy_m")
        offsets = tuple(map(float, self.context.p("cell.tray_slot_x_offsets_m")))
        y_offset = float(self.context.p("cell.tray_slot_y_offset_m"))
        slot_xy = (float(centers["RED"][0]) + offsets[len(offsets)//2],
                   float(centers["RED"][1]) + y_offset)
        place_z = float(self.context.p("robot.object_place_center_z_m"))
        q = self._pose_q(*slot_xy, place_z)
        size_x = float(self.context.p("object.size_xyz_m")[0])
        open_gap = 2.0 * (float(self.context.p("robot.finger_open_center_offset_m"))
                          - float(self.context.p("robot.finger_thickness_m")) / 2.0)
        held = (open_gap - size_x) / 2.0
        scene = CollisionScene(self.context)
        for phase, payload in (("PLACE_DESCENT", "carried"), ("RELEASE", "carried"),
                               ("RELEASE", "placed"), ("RETREAT", "placed")):
            with self.subTest(phase=phase):
                report = scene.check(q, gripper_m=held, detections=(self.track,),
                                     target_track_id=self.track.track_id, phase=phase,
                                     payload_mode=payload, slot_color="RED", slot_xy=slot_xy)
                self.assertTrue(report.valid, report.violations)

        rejected = scene.check(q, gripper_m=held, detections=(self.track,),
                               target_track_id=self.track.track_id, phase="TRANSFER",
                               payload_mode="carried", slot_color="RED", slot_xy=slot_xy)
        self.assertFalse(rejected.valid)
        floor_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM,
                                     "tray_red_floor_collision")
        finger_names = {"finger_left_collision", "finger_right_collision"}
        floor_violations = [v for v in rejected.violations
                            if (v.geom_a in finger_names and v.geom_b == "tray_red_floor_collision")
                            or (v.geom_b in finger_names and v.geom_a == "tray_red_floor_collision")]
        self.assertEqual(len(floor_violations), 2, rejected.violations)
        expected_static = (float(self.context.p("planning.static_clearance_m"))
                           + float(self.context.p("planning.path_sweep_sample_step_m"))
                           + float(self.context.p("planning.distance_numeric_tolerance_m")))
        self.assertTrue(all(abs(v.required_m - expected_static) < 1e-12 for v in floor_violations))
        self.assertGreaterEqual(floor_id, 0)

        wrong_slot = (slot_xy[0] + 0.01, slot_xy[1])
        misaligned = scene.check(q, gripper_m=held, detections=(self.track,),
                                 target_track_id=self.track.track_id, phase="PLACE_DESCENT",
                                 payload_mode="carried", slot_color="RED", slot_xy=wrong_slot,
                                 sample_step_bound_m=0.002)
        table_violations = [v for v in misaligned.violations
                            if v.geom_b == "table_top_collision" and v.geom_a in finger_names]
        self.assertFalse(misaligned.valid)
        self.assertEqual(len(table_violations), 2, misaligned.violations)
        expected_static_coarse = (float(self.context.p("planning.static_clearance_m"))
                                  + 0.002
                                  + float(self.context.p("planning.distance_numeric_tolerance_m")))
        self.assertTrue(all(abs(v.required_m - expected_static_coarse) < 1e-12 for v in table_violations),
                        table_violations)


if __name__ == "__main__":
    unittest.main()
