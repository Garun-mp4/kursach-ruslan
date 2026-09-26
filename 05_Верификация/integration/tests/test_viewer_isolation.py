from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04_Программа"))

from integration.config import RuntimeConfig, load_ssot  # noqa: E402
from integration.display import copy_runtime_state_to_display  # noqa: E402
from integration.placement_camera import load_integrated_model  # noqa: E402
from integration.runner import _load_model  # noqa: E402
from integration.runtime import SorterRuntime  # noqa: E402
from integration.scene import load_scenario  # noqa: E402


class InteractiveDisplayIsolationTests(unittest.TestCase):
    def test_viewer_snapshot_is_separate_from_runtime_physics_state(self) -> None:
        scenario = load_scenario("single_red_center")
        runtime_model, runtime_data, _physics, _generated = _load_model(
            scenario, "nominal"
        )
        display_model = load_integrated_model(load_ssot())
        display_data = mujoco.MjData(display_model)

        copy_runtime_state_to_display(
            runtime_model, runtime_data, display_model, display_data
        )

        np.testing.assert_array_equal(display_data.qpos, runtime_data.qpos)
        np.testing.assert_array_equal(display_data.qvel, runtime_data.qvel)
        self.assertEqual(display_data.time, runtime_data.time)
        runtime_time = float(runtime_data.time)
        runtime_qpos = runtime_data.qpos.copy()
        display_data.time += 1.0
        display_data.qpos[0] += 0.01
        self.assertEqual(runtime_data.time, runtime_time)
        np.testing.assert_array_equal(runtime_data.qpos, runtime_qpos)

    def test_incompatible_display_layout_fails_before_viewer_launch(self) -> None:
        model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
        runtime_data = mujoco.MjData(model)
        other_model = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><body><freejoint/><geom type='sphere' size='.01'/></body></worldbody></mujoco>"
        )
        other_data = mujoco.MjData(other_model)
        with self.assertRaisesRegex(ValueError, "layout differs"):
            copy_runtime_state_to_display(model, runtime_data, other_model, other_data)

    def test_closed_viewer_detaches_without_stopping_the_runtime(self) -> None:
        class ClosedViewer:
            def is_running(self) -> bool:
                return False

            def sync(self) -> None:
                raise AssertionError("A closed viewer must never be synchronized")

        scenario = load_scenario("single_red_center")
        model, data, physics, _generated = _load_model(scenario, "nominal")
        output_dir = Path(self.enterContext(tempfile.TemporaryDirectory())) / "runtime"
        runtime = SorterRuntime(
            model, data, run_id="viewer-close-regression",
            runtime_config=RuntimeConfig.load(),
            output_dir=output_dir,
            viewer=ClosedViewer(), physics_expected=physics,
        )
        self.addCleanup(runtime.close)
        summary = runtime.run(max_simulation_time_s=1.0, max_control_ticks=1)
        self.assertTrue(summary["viewer_detached_early"])
        self.assertEqual(summary["control_ticks"], 1)
        self.assertGreaterEqual(summary["simulated_elapsed_s"], 0.01)


if __name__ == "__main__":
    unittest.main()
