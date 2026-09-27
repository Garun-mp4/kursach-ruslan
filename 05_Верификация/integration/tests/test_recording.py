from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np

from integration.recording import RecordingConfig, RunStateRecorder, _source_log_hashes
from integration.config import ROOT


class RunStateRecorderTests(unittest.TestCase):
    @staticmethod
    def _model() -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body name="moving" pos="0 0 0.1">
              <freejoint name="moving_free"/>
              <geom name="moving_geom" type="sphere" size="0.02" mass="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """)

    def test_fixed_simulation_clock_sampling_and_integration_state_round_trip(self) -> None:
        model = self._model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        config = RecordingConfig()
        self.assertAlmostEqual(config.sample_period_s, 0.2)
        with tempfile.TemporaryDirectory(dir=ROOT / "99_Рабочие_материалы") as temporary:
            recorder = RunStateRecorder(
                model, run_id="recording-test", run_dir=Path(temporary), config=config,
            )
            recorder.observe(
                data, controller_state="INIT", track_records={}, force_sample=True,
            )
            for tick in range(1, 21):
                mujoco.mj_step(model, data)
                events = ()
                state = "OBSERVE"
                if tick == 10:
                    events = ({
                        "event_id": 1, "event_type": "TARGET_SELECTED",
                        "simulation_time_s": float(data.time), "state": "SELECT",
                        "track_id": 4,
                    },)
                    state = "SELECT"
                elif tick == 15:
                    events = ({
                        "event_id": 2, "event_type": "STATE_ENTERED",
                        "simulation_time_s": float(data.time), "state": "TRANSFER",
                    },)
                    state = "TRANSFER"
                recorder.observe(
                    data, controller_state=state, track_records={}, events=events,
                )
            summary = recorder.finalize(
                data, controller_state="DONE", track_records={},
            )
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["dropped_frames"], 0)
            with np.load(recorder.state_stream_path, allow_pickle=False) as stream:
                self.assertEqual(stream["timeline_states"].shape[0], 2)
                state = stream["timeline_states"][-1].copy()
                labels = set(stream["keyframe_labels"].tolist())
            self.assertTrue({"before_detection", "detection", "transfer", "final"} <= labels)
            restored = mujoco.MjData(model)
            mujoco.mj_setState(
                model, restored, state,
                int(mujoco.mjtState.mjSTATE_INTEGRATION),
            )
            np.testing.assert_allclose(restored.qpos, data.qpos)
            self.assertAlmostEqual(restored.time, data.time)
            document = json.loads(recorder.state_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(document["run_id"], "recording-test")
            self.assertNotIn("ground_truth_color", str(document))
            self.assertNotIn("evaluator", str(document))

    def test_source_log_hashes_cover_files_and_nested_plan_samples(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            controller = run_dir / "controller"
            plans = controller / "plans"
            plans.mkdir(parents=True)
            (controller / "telemetry.csv").write_text("tick,value\n0,1\n", encoding="utf-8")
            (plans / "plan.json").write_text('{"status":"OK"}\n', encoding="utf-8")

            logs = _source_log_hashes({
                "run_directory": str(run_dir),
                "relative_outputs": {
                    "controller_telemetry": "controller/telemetry.csv",
                    "m5_plan_samples": "controller/plans/",
                },
            })

            self.assertEqual(set(logs), {"controller_telemetry", "m5_plan_samples"})
            self.assertEqual(len(logs["controller_telemetry"]["sha256"]), 64)
            self.assertEqual(logs["m5_plan_samples"]["file_count"], 1)
            self.assertEqual(len(logs["m5_plan_samples"]["files"][0]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
