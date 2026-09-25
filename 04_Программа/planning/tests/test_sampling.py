from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROGRAM = Path(__file__).resolve().parents[2]
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from planning.sampling import compare_sampling_runs


def run(case_id: str, *, success: bool = True, path: float = 2.0, clearance: float = 0.002):
    return {
        "case_id": case_id,
        "result": "SUCCESS" if success else "COLLISION_PATH",
        "success": success,
        "class_label_from_M4_contract": "RED",
        "estimated_xy_x_m": 0.0,
        "estimated_xy_y_m": -0.25,
        "estimated_yaw_rad": 0.0,
        "perceived_obstacle_count": 0,
        "selected_slot_id": "RED:1",
        "selected_ik_branch": "ELBOW_POSITIVE",
        "phase_sequence": "APPROACH;PICK;PLACE;RETURN",
        "event_sequence": "ATTACHED;RELEASED",
        "tcp_path_length_m": path,
        "duration_s": 10.0,
        "all_pair_clearance_lower_bound_m": clearance,
        "collision_sample_step_m": 0.001 if case_id == "coarse" else 0.0005,
    }


class SamplingRefinementTests(unittest.TestCase):
    def test_same_safe_route_passes_and_reports_grid_sensitive_clearance(self):
        result = compare_sampling_runs(
            run("coarse", clearance=0.0023), run("fine", path=2.000001, clearance=0.0011),
            refined_step_m=0.0005)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["clearance_estimate_status"], "GRID_SENSITIVE")
        self.assertAlmostEqual(result["clearance_lower_bound_difference_m"], 0.0012)
        self.assertFalse(result["new_collision_at_refined_resolution"])

    def test_refined_collision_fails_convergence(self):
        result = compare_sampling_runs(
            run("coarse"), run("fine", success=False), refined_step_m=0.0005)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["new_collision_at_refined_resolution"])

    def test_material_path_change_fails_convergence(self):
        result = compare_sampling_runs(
            run("coarse"), run("fine", path=2.001), refined_step_m=0.0005)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["tcp_path_length_stable"])

    def test_different_observation_fails_comparison(self):
        fine = run("fine")
        fine["estimated_xy_x_m"] = 0.001
        result = compare_sampling_runs(run("coarse"), fine, refined_step_m=0.0005)
        self.assertEqual(result["status"], "FAIL")
        self.assertFalse(result["same_m4_estimate"])


if __name__ == "__main__":
    unittest.main()
