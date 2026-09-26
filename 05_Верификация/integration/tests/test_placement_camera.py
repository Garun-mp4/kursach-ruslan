from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04_Программа"))

from controller.config import ControllerConfig
from integration.config import RuntimeConfig, load_ssot, value
from integration.placement import PublicPlacementVerifier
from integration.placement_camera import CAMERA_NAME, load_integrated_model, placement_camera_values
from integration.sensors import PublicSensorPipeline


class PlacementCameraIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ssot = load_ssot()
        cls.model = load_integrated_model(cls.ssot)
        cls.runtime_config = RuntimeConfig.load()
        cls.controller_config = ControllerConfig.load()

    def _empty_trays_scene(self) -> mujoco.MjData:
        data = mujoco.MjData(self.model)
        home_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home_observation"
        )
        mujoco.mj_resetDataKeyframe(self.model, data, home_id)
        object_size = np.asarray(value(self.ssot, "object.size_xyz_m"), dtype=float)
        table_size = np.asarray(value(self.ssot, "cell.table_size_xy_m"), dtype=float)
        park_x = table_size[0] / 2.0 - object_size[0] / 2.0
        park_y = np.linspace(-0.30, 0.30, 6)
        names = [
            f"object_{color}_{index:02d}"
            for index in (1, 2)
            for color in ("red", "green", "blue")
        ]
        for name, y_m in zip(names, park_y, strict=True):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            joint_id = int(self.model.body_jntadr[body_id])
            qpos_adr = int(self.model.jnt_qposadr[joint_id])
            data.qpos[qpos_adr:qpos_adr + 7] = (
                park_x, float(y_m), object_size[2] / 2.0, 1.0, 0.0, 0.0, 0.0
            )
        mujoco.mj_forward(self.model, data)
        return data

    def test_integrated_model_declares_separate_fixed_placement_view(self) -> None:
        self.assertEqual(self.model.ncam, 2)
        self.assertGreaterEqual(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_rgb"), 0
        )
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        self.assertGreaterEqual(camera_id, 0)
        config = placement_camera_values(self.ssot)
        np.testing.assert_allclose(self.model.cam_pos[camera_id], config["position_world_m"])
        self.assertAlmostEqual(float(self.model.cam_fovy[camera_id]), config["fovy_deg"])

    def test_placement_calibration_is_self_consistent_on_held_out_grid(self) -> None:
        path = ROOT / "05_Верификация" / "integration" / "sensors" / "placement_camera_calibration.json"
        calibration = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(calibration["camera_config_id"], value(
            self.ssot, "camera.placement_config_id"
        ))
        self.assertEqual(calibration["object_top_plane_z_m"], value(
            self.ssot, "camera.placement_object_top_plane_z_m"
        ))
        self.assertLess(calibration["verification_metrics"]["max_m"], 1e-6)

    def test_release_pose_is_confirmed_from_public_placement_rgb(self) -> None:
        data = self._empty_trays_scene()
        pipeline = PublicSensorPipeline(self.model)
        try:
            verifier = PublicPlacementVerifier(self.ssot, self.runtime_config)
            baseline = pipeline.capture(data, simulation_time_s=0.0)
            self.assertEqual(baseline.frame.frame_id, baseline.placement_frame.frame_id)
            self.assertEqual(baseline.frame.camera_config_id, "overhead_rgb_M2_validated_v1")
            self.assertEqual(
                baseline.placement_frame.camera_config_id,
                value(self.ssot, "camera.placement_config_id"),
            )
            self.assertEqual(verifier.initial_occupied_slots(baseline.placement_batch), ())
            verifier.observe(baseline.placement_batch)

            # Captured arm posture from the M7 contact run that exposed the original
            # overhead-camera occlusion; the red payload is now resting in RED:0.
            for joint, coordinate in zip(
                ("j1", "j2", "j3", "j4"),
                (1.80102391, 1.41028185, 0.091296671, -1.64036814),
                strict=True,
            ):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                data.qpos[int(self.model.jnt_qposadr[joint_id])] = coordinate

            trays = value(self.ssot, "cell.tray_centers_xy_m")
            slot_offset = float(value(self.ssot, "cell.tray_slot_x_offsets_m")[0])
            object_size = value(self.ssot, "object.size_xyz_m")
            object_body = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "object_red_01"
            )
            object_joint = int(self.model.body_jntadr[object_body])
            object_qpos = int(self.model.jnt_qposadr[object_joint])
            data.qpos[object_qpos:object_qpos + 7] = (
                float(trays["RED"][0]) + slot_offset,
                float(trays["RED"][1]),
                float(value(self.ssot, "cell.tray_floor_thickness_m"))
                + float(object_size[2]) / 2.0,
                1.0, 0.0, 0.0, 0.0,
            )
            mujoco.mj_forward(self.model, data)

            placed = pipeline.capture(data, simulation_time_s=0.02)
            evidence = verifier.evidence(placed.placement_batch)
            minimum_confidence = float(
                self.controller_config.p("place_evidence_min_confidence")
            )
            self.assertEqual(placed.placement_frame.frame_id, 2)
            self.assertEqual(evidence.source, "PUBLIC_CAMERA_PERCEPTION")
            self.assertEqual(evidence.status.value, "CONFIRMED")
            self.assertEqual(evidence.observed_class, "RED")
            self.assertEqual(evidence.observed_zone_id, "RED")
            self.assertEqual(evidence.observed_slot_id, "RED:0")
            self.assertGreaterEqual(float(evidence.confidence or 0.0), minimum_confidence)
            estimated_xy = next(
                detection.xy_base_m
                for detection in placed.placement_batch.detections
                if detection.class_label == "RED"
            )
            target_xy = (float(trays["RED"][0]) + slot_offset, float(trays["RED"][1]))
            self.assertLess(float(np.linalg.norm(np.asarray(estimated_xy) - target_xy)), 0.005)
        finally:
            pipeline.close()

    def test_public_rgb_confirms_all_color_and_slot_combinations(self) -> None:
        """The public camera must verify each tray slot without consulting scene truth."""
        trays = value(self.ssot, "cell.tray_centers_xy_m")
        slot_offsets = tuple(map(float, value(self.ssot, "cell.tray_slot_x_offsets_m")))
        slot_y_offset = float(value(self.ssot, "cell.tray_slot_y_offset_m"))
        floor_z = float(value(self.ssot, "cell.tray_floor_thickness_m"))
        object_size = np.asarray(value(self.ssot, "object.size_xyz_m"), dtype=float)
        minimum_confidence = float(
            self.controller_config.p("place_evidence_min_confidence")
        )

        for color_index, color in enumerate(("RED", "GREEN", "BLUE")):
            for slot_index, slot_offset in enumerate(slot_offsets):
                with self.subTest(color=color, slot=slot_index):
                    data = self._empty_trays_scene()
                    pipeline = PublicSensorPipeline(self.model)
                    try:
                        verifier = PublicPlacementVerifier(self.ssot, self.runtime_config)
                        baseline = pipeline.capture(data, simulation_time_s=0.0)
                        self.assertEqual(verifier.initial_occupied_slots(
                            baseline.placement_batch
                        ), ())
                        verifier.observe(baseline.placement_batch)

                        object_name = f"object_{color.lower()}_01"
                        body_id = mujoco.mj_name2id(
                            self.model, mujoco.mjtObj.mjOBJ_BODY, object_name
                        )
                        self.assertGreaterEqual(body_id, 0)
                        joint_id = int(self.model.body_jntadr[body_id])
                        qpos = int(self.model.jnt_qposadr[joint_id])
                        yaw = (slot_index + color_index) * np.pi / 8.0
                        data.qpos[qpos:qpos + 7] = (
                            float(trays[color][0]) + slot_offset,
                            float(trays[color][1]) + slot_y_offset,
                            floor_z + object_size[2] / 2.0,
                            np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0),
                        )
                        mujoco.mj_forward(self.model, data)

                        placed = pipeline.capture(data, simulation_time_s=0.02)
                        evidence = verifier.evidence(placed.placement_batch)
                        self.assertEqual(evidence.source, "PUBLIC_CAMERA_PERCEPTION")
                        self.assertEqual(evidence.status.value, "CONFIRMED")
                        self.assertEqual(evidence.observed_class, color)
                        self.assertEqual(evidence.observed_zone_id, color)
                        self.assertEqual(evidence.observed_slot_id, f"{color}:{slot_index}")
                        self.assertGreaterEqual(
                            float(evidence.confidence or 0.0), minimum_confidence
                        )
                        detection = next(
                            item for item in placed.placement_batch.detections
                            if item.class_label == color
                        )
                        target_xy = np.asarray((
                            float(trays[color][0]) + slot_offset,
                            float(trays[color][1]) + slot_y_offset,
                        ))
                        self.assertLess(
                            float(np.linalg.norm(np.asarray(detection.xy_base_m) - target_xy)),
                            0.002,
                        )
                    finally:
                        pipeline.close()

    def test_camera_failure_invalidates_both_rgb_views(self) -> None:
        pipeline = PublicSensorPipeline(self.model)
        try:
            snapshot = pipeline.capture(self._empty_trays_scene(), simulation_time_s=0.0,
                                       valid=False)
            self.assertFalse(snapshot.frame.valid)
            self.assertFalse(snapshot.placement_frame.valid)
            self.assertEqual(snapshot.input_batch.status, "NO_FRAME")
            self.assertEqual(snapshot.placement_batch.status, "NO_FRAME")
        finally:
            pipeline.close()


if __name__ == "__main__":
    unittest.main()
