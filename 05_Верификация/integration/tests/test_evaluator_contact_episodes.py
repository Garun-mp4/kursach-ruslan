from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import mujoco

from integration.evaluator_telemetry import EvaluatorTelemetry


class _Object:
    body_name = "object_red_01"
    class_label = "RED"


class EvaluatorContactEpisodeTests(unittest.TestCase):
    @staticmethod
    def _model() -> mujoco.MjModel:
        return mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.002" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="table_collision" type="plane" size="1 1 0.1"/>
            <body name="object_red_01" pos="0 0 0.018">
              <freejoint name="object_free"/>
              <geom name="object_geom" type="sphere" size="0.02" mass="0.1"/>
            </body>
            <body name="wrist" pos="0.3 0 0.2">
              <geom name="wrist_geom" type="sphere" size="0.01" contype="0" conaffinity="0"/>
            </body>
            <body name="finger_left" pos="0.3 -0.02 0.2">
              <site name="touch_left_site" type="sphere" size="0.01"/>
            </body>
            <body name="finger_right" pos="0.3 0.02 0.2">
              <site name="touch_right_site" type="sphere" size="0.01"/>
            </body>
          </worldbody>
          <sensor>
            <touch name="touch_left" site="touch_left_site"/>
            <touch name="touch_right" site="touch_right_site"/>
          </sensor>
        </mujoco>
        """)

    def test_continuous_contact_is_one_physics_step_episode(self) -> None:
        model = self._model()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with tempfile.TemporaryDirectory() as temporary:
            telemetry = EvaluatorTelemetry(
                model, data, [_Object()], Path(temporary), run_id="contact-test",
                touch_threshold_N=0.001,
            )
            try:
                for _ in range(8):
                    mujoco.mj_step(model, data)
                    telemetry.sample_contact_step(float(data.time))
                qpos_address = int(model.jnt_qposadr[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")
                ])
                data.qpos[qpos_address + 2] = 0.5
                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)
                telemetry.sample_contact_step(float(data.time))
            finally:
                telemetry.close()
            lines = (Path(temporary) / "contact_events.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            events = [json.loads(line) for line in lines]
        object_contacts = [event for event in events
                           if {event["geom1"], event["geom2"]}
                           == {"object_geom", "table_collision"}]
        self.assertEqual(len(object_contacts), 1)
        event = object_contacts[0]
        self.assertEqual(event["run_id"], "contact-test")
        self.assertEqual(event["event_type"], "CONTACT_EPISODE")
        self.assertEqual(event["physics_sample_count"], 8)
        self.assertAlmostEqual(event["duration_s"], 8 * model.opt.timestep)
        self.assertAlmostEqual(event["start_time_s"], 0.0)
        self.assertAlmostEqual(event["end_time_s"], 8 * model.opt.timestep)
        self.assertAlmostEqual(event["detected_closed_time_s"], 9 * model.opt.timestep)
        self.assertLessEqual(event["minimum_distance_m"], 0.0)
        self.assertEqual(event["close_reason"], "PAIR_ABSENT_AT_PHYSICS_STEP")


if __name__ == "__main__":
    unittest.main()
