from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
PROGRAM_DIR = ROOT / "04_Программа"
sys.path.insert(0, str(PROGRAM_DIR))
sys.path.insert(0, str(ROOT / "05_Верификация" / "perception"))

from camera.frame import CameraFrame  # noqa: E402
from perception.detector import RGBObjectPerception, load_config  # noqa: E402
from perception.geometry import PlanarCalibration  # noqa: E402
from rig import GroundTruthObject, PerceptionRig  # noqa: E402
import run_m4_verification as m4_verification  # noqa: E402


class PerceptionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_path = PROGRAM_DIR / "perception" / "perception_config.yaml"
        cls.ssot_path = ROOT / "02_Спецификация" / "параметры_системы.yaml"
        cls.ssot = yaml.safe_load(cls.ssot_path.read_text(encoding="utf-8-sig"))
        cls.config = load_config(cls.config_path)
        cls.calibration = PlanarCalibration.from_json(str(
            ROOT / "05_Верификация" / "perception" / "calibration" / "camera_calibration.json"
        ))
        cls.rig = PerceptionRig(ROOT, cls.ssot)
        empty = cls.rig.render_empty()
        cls.background = empty.rgb.copy()
        cls.object = GroundTruthObject("unit-red", "RED", 0.0, -0.25, 0.3,
                                       (0.90, 0.06, 0.04, 1.0))
        cls.rig.set_objects([cls.object])
        cls.frame = cls.rig.capture(sim_time_s=2.0)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.rig.close()

    def make_detector(self) -> RGBObjectPerception:
        return RGBObjectPerception(self.config, self.calibration, self.background)

    def test_runtime_yaml_is_exact_ssot_export(self) -> None:
        profile = self.ssot["perception"]["runtime_config"]
        self.assertEqual(profile, self.config)
        self.assertEqual(profile["config_version"], "M4-v1.3")

    def test_camera_frame_exposes_rgb_metadata_without_truth(self) -> None:
        self.assertEqual(self.frame.resolution_px, (640, 480))
        self.assertAlmostEqual(self.frame.age_s(2.02), 0.02)
        self.assertEqual(self.frame.rgb.dtype, np.uint8)
        self.assertEqual(self.frame.rgb.shape, (480, 640, 3))
        self.assertFalse(hasattr(self.frame, "object_id"))
        self.assertFalse(hasattr(self.frame, "ground_truth"))

    def test_primary_rgb_order_and_cyclic_red_hue(self) -> None:
        rgb = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        self.assertEqual(hsv[0, :, 0].tolist(), [0, 60, 120])
        red_hsv = np.asarray([[[0, 255, 255], [179, 255, 255]]], dtype=np.uint8)
        red_mask = RGBObjectPerception._class_mask(
            red_hsv, self.config["color_classifier"]["hsv_ranges"]["RED"]
        )
        self.assertTrue(np.all(red_mask == 255))

    def test_rendered_red_candidate_returns_structured_valid_pose(self) -> None:
        batch = self.make_detector().detect(
            self.frame, now_simulation_time_s=self.frame.simulation_time_s + 0.02
        )
        self.assertEqual(batch.status, "OK")
        self.assertEqual(len(batch.detections), 1)
        detection = batch.detections[0]
        self.assertEqual(detection.status, "VALID")
        self.assertEqual(detection.class_label, "RED")
        self.assertLess(np.linalg.norm(np.asarray(detection.xy_base_m) -
                                       [self.object.x_m, self.object.y_m]), 0.004)
        self.assertIsNotNone(detection.track_id)
        self.assertAlmostEqual(detection.frame_age_s, 0.02)
        self.assertEqual(len(detection.center_uv_px), 2)

    def test_unsupported_saturated_hue_is_preserved_as_unknown(self) -> None:
        item = GroundTruthObject("unit-yellow", "UNKNOWN", 0.0, -0.25, 0.4,
                                 (0.90, 0.72, 0.04, 1.0))
        self.rig.set_objects([item])
        frame = self.rig.capture(sim_time_s=2.2)
        batch = self.make_detector().detect(frame)
        self.assertEqual(len(batch.detections), 1)
        self.assertEqual(batch.detections[0].class_label, "UNKNOWN")
        self.assertNotIn(batch.detections[0].class_label, {"RED", "GREEN", "BLUE"})

    def test_low_contrast_neutral_object_is_never_promoted_to_supported_color(self) -> None:
        item = GroundTruthObject("unit-pale-neutral", "UNKNOWN", 0.0, -0.25, 0.0,
                                 (0.55, 0.57, 0.58, 1.0))
        self.rig.set_objects([item])
        frame = self.rig.capture(sim_time_s=2.4)
        batch = self.make_detector().detect(frame)
        self.assertTrue(all(d.class_label == "UNKNOWN" for d in batch.detections))

    def test_invalid_camera_frames_fail_closed(self) -> None:
        detector = self.make_detector()
        expected = self.config["sensor_interface"]["camera_config_id"]
        missing = detector.detect(CameraFrame(None, 2, 2.0, False, expected, "NO_SENSOR_DATA"))
        malformed = detector.detect(CameraFrame(np.zeros((3, 3, 3), np.uint8), 3, 2.0, True, expected))
        wrong_camera = detector.detect(CameraFrame(self.frame.rgb, 4, 2.0, True, "other"))
        stale = detector.detect(CameraFrame(self.frame.rgb, 5, 2.0, True, expected),
                                now_simulation_time_s=2.1)
        future = detector.detect(CameraFrame(self.frame.rgb, 6, 2.1, True, expected),
                                 now_simulation_time_s=2.0)
        self.assertEqual([missing.status, malformed.status, wrong_camera.status,
                          stale.status, future.status],
                         ["NO_FRAME", "INVALID", "INVALID", "STALE", "INVALID"])
        self.assertTrue(all(not result.detections for result in
                            (missing, malformed, wrong_camera, stale, future)))

    def test_homography_pixel_base_round_trip(self) -> None:
        point = (-0.075, -0.265)
        uv = self.calibration.xy_to_pixel(*point)
        estimated = self.calibration.pixel_to_xy(*uv)
        np.testing.assert_allclose(estimated, point, rtol=0, atol=1e-10)

    def test_canonical_square_yaw_has_quarter_turn_symmetry(self) -> None:
        detections = self.make_detector().detect(self.frame).detections
        self.assertEqual(len(detections), 1)
        self.assertGreaterEqual(detections[0].yaw_base_rad, 0.0)
        self.assertLess(detections[0].yaw_base_rad, math.pi / 2)

    def test_oversized_connected_mask_splits_at_distinct_known_color_cores(self) -> None:
        seed = 20262009  # Retained diagnostic regression case; excluded from final validation.
        objects, lighting, noise = m4_verification._dataset_objects(self.ssot, seed, tuning=False)
        self.rig.set_objects(objects, lighting_scale=lighting)
        frame = self.rig.capture(sim_time_s=8.0, noise_sigma=noise, noise_seed=seed ^ 0x5A5A)
        batch = self.make_detector().detect(frame)
        matches, missed, false = m4_verification.match_detections(batch.detections, objects)
        split_detections = [d for d in batch.detections
                            if d.diagnostics.get("split_from_merged_candidate")]
        self.assertEqual(len(matches), len(objects))
        self.assertFalse(missed)
        self.assertFalse(false)
        self.assertGreaterEqual(len(split_detections), 2)
        self.assertTrue(all(d.status == "VALID" for d in split_detections))


if __name__ == "__main__":
    unittest.main()
