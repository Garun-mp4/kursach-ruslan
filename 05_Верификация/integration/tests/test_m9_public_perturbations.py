from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04_Программа"))

from camera.frame import CameraFrame  # noqa: E402
from integration.scene import SceneGenerator, load_scenario  # noqa: E402
from integration.sensors import apply_rgb_perturbation, offset_planar_calibration  # noqa: E402
from perception.geometry import PlanarCalibration  # noqa: E402


class M9PublicPerturbationTests(unittest.TestCase):
    def test_rgb_noise_is_seeded_and_does_not_mutate_the_source(self) -> None:
        source = np.full((8, 9, 3), 120, dtype=np.uint8)
        source.setflags(write=False)
        frame = CameraFrame(source, 7, 1.2, True, "test-camera")
        first = apply_rgb_perturbation(
            frame, gain=0.8, noise_sigma=4.0, rng=np.random.default_rng(1234)
        )
        second = apply_rgb_perturbation(
            frame, gain=0.8, noise_sigma=4.0, rng=np.random.default_rng(1234)
        )
        np.testing.assert_array_equal(first.rgb, second.rgb)
        np.testing.assert_array_equal(source, np.full((8, 9, 3), 120, dtype=np.uint8))
        self.assertFalse(first.rgb.flags.writeable)
        self.assertEqual(first.frame_id, frame.frame_id)

    def test_rgb_perturbation_rejects_nonphysical_parameters(self) -> None:
        frame = CameraFrame(np.zeros((1, 1, 3), dtype=np.uint8), 1, 0.0, True, "test")
        with self.assertRaises(ValueError):
            apply_rgb_perturbation(
                frame, gain=0.0, noise_sigma=0.0, rng=np.random.default_rng(1)
            )
        with self.assertRaises(ValueError):
            apply_rgb_perturbation(
                frame, gain=1.0, noise_sigma=-1.0, rng=np.random.default_rng(1)
            )

    def test_calibration_bias_is_an_xy_offset_in_world_coordinates(self) -> None:
        identity = np.eye(3, dtype=np.float64)
        base = PlanarCalibration(identity, identity, 0.016, "identity")
        biased = offset_planar_calibration(base, (0.002, -0.003))
        self.assertEqual(biased.pixel_to_xy(25.0, 40.0), (25.002, 39.997))
        self.assertEqual(biased.calibration_id, "identity+bias(0.002,-0.003)m")

    def test_seeded_red_subset_generation_is_reproducible(self) -> None:
        scenario = load_scenario(
            "m9_random_red_one", ROOT / "07_Испытания" / "scenarios"
        )
        generator = SceneGenerator()
        first = generator._randomized_objects(scenario)
        second = generator._randomized_objects(scenario)
        self.assertEqual(first, second)
        self.assertEqual([item["body_name"] for item in first], ["object_red_01"])

    def test_empty_input_is_a_valid_explicit_scene(self) -> None:
        scenario = load_scenario(
            "m9_empty_input", ROOT / "07_Испытания" / "scenarios"
        )
        self.assertEqual(scenario["active_objects"], [])


if __name__ == "__main__":
    unittest.main()
