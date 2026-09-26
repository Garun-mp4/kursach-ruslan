from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "04_Программа"))

from integration.config import RuntimeConfig  # noqa: E402
from integration.runner import _load_model  # noqa: E402
from integration.runtime import SorterRuntime  # noqa: E402
from integration.scene import load_scenario  # noqa: E402


class RuntimePerformanceDiagnosticTests(unittest.TestCase):
    def test_runtime_reports_wall_metrics_without_changing_physics_contract(self) -> None:
        scenario = load_scenario("single_red_center")
        model, data, physics, _generated = _load_model(scenario, "nominal")
        with tempfile.TemporaryDirectory(prefix="m7-performance-") as temporary:
            runtime = SorterRuntime(
                model,
                data,
                run_id="performance-diagnostic-regression",
                runtime_config=RuntimeConfig.load(),
                output_dir=Path(temporary) / "controller",
                physics_expected=physics,
            )
            try:
                summary = runtime.run(max_simulation_time_s=1.0, max_control_ticks=1)
            finally:
                runtime.close()

        performance = summary["performance"]
        self.assertEqual(performance["control_cycle_count"], 1)
        self.assertEqual(performance["physics_step_count"], 5)
        self.assertGreater(performance["simulation_to_wall_time_ratio"], 0.0)
        self.assertGreaterEqual(
            performance["control_cycle_wall_max_s"],
            performance["control_cycle_wall_average_s"],
        )
        self.assertEqual(performance["viewer_sync_count"], 0)
        self.assertEqual(performance["control_cycle_wall_warning_threshold_s"], 60.0)
        self.assertEqual(performance["control_cycle_wall_warning_count"], 0)
        sensor = performance["sensor_capture"]
        self.assertEqual(sensor["capture_count"], 0)
        self.assertEqual(performance["render_and_display_wall_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
