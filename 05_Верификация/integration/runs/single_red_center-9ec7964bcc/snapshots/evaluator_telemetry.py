from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import mujoco
import numpy as np


class EvaluatorTelemetry:
    """Ground-truth-only recorder; it has no return path into controller inputs."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 active_objects: Iterable[object], output_dir: Path,
                 *, touch_threshold_N: float):
        self.model, self.data = model, data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._objects: list[tuple[str, str, int, int, int]] = []
        for item in active_objects:
            body_name = str(item.body_name)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise ValueError(f"Evaluator telemetry cannot find body {body_name!r}")
            joint_id = int(model.body_jntadr[body_id])
            if joint_id < 0 or model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError(f"Evaluator telemetry requires a free object joint for {body_name!r}")
            self._objects.append((
                body_name, str(item.class_label), body_id,
                int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id]),
            ))
        wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "wrist")
        if wrist_id < 0:
            raise ValueError("Evaluator telemetry cannot find the wrist body")
        self._wrist_id = int(wrist_id)
        self._finger_body_ids = tuple(self._body_id(name) for name in ("finger_left", "finger_right"))
        self._touch_ids = tuple(self._sensor_id(name) for name in ("touch_left", "touch_right"))
        self._touch_threshold_N = float(touch_threshold_N)
        if not np.isfinite(self._touch_threshold_N) or self._touch_threshold_N <= 0:
            raise ValueError("Evaluator touch threshold must be finite and positive")
        self._touch_addresses = tuple(int(model.sensor_adr[sensor_id]) for sensor_id in self._touch_ids)
        self._relative_at_first_bilateral_contact: dict[str, np.ndarray] = {}
        self._max_slip_m = {body_name: 0.0 for body_name, *_ in self._objects}
        self._object_path = self.output_dir / "object_telemetry.csv"
        self._contact_path = self.output_dir / "contact_events.jsonl"
        self._object_stream = self._object_path.open("w", encoding="utf-8", newline="")
        self._contact_stream = self._contact_path.open("w", encoding="utf-8", newline="\n")
        self._writer = csv.writer(self._object_stream)
        self._writer.writerow((
            "simulation_time_s", "body_name", "ground_truth_color",
            "x_m", "y_m", "z_m", "vx_m_s", "vy_m_s", "vz_m_s",
            "wx_rad_s", "wy_rad_s", "wz_rad_s", "touch_left_N", "touch_right_N",
            "bilateral_finger_contact", "wrist_relative_x_m", "wrist_relative_y_m",
            "wrist_relative_z_m", "slip_from_first_bilateral_contact_m",
        ))

    def _sensor_id(self, name: str) -> int:
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id < 0 or int(self.model.sensor_dim[sensor_id]) != 1:
            raise ValueError(f"Evaluator telemetry requires scalar sensor {name!r}")
        return int(sensor_id)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Evaluator telemetry cannot find body {name!r}")
        return int(body_id)

    def sample(self, simulation_time_s: float) -> None:
        left, right = (max(0.0, float(self.data.sensordata[address]))
                       for address in self._touch_addresses)
        object_contacts: dict[int, set[int]] = {item[2]: set() for item in self._objects}
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            body_ids = {
                int(self.model.geom_bodyid[int(contact.geom1)]),
                int(self.model.geom_bodyid[int(contact.geom2)]),
            }
            for object_body_id in object_contacts:
                if object_body_id in body_ids:
                    object_contacts[object_body_id].update(body_ids - {object_body_id})
        wrist_position = np.asarray(self.data.xpos[self._wrist_id], dtype=float)
        wrist_rotation = np.asarray(self.data.xmat[self._wrist_id], dtype=float).reshape(3, 3)
        for body_name, color, body_id, qadr, dadr in self._objects:
            contacting_bodies = object_contacts[body_id]
            bilateral = all(finger_id in contacting_bodies for finger_id in self._finger_body_ids)
            position = np.asarray(self.data.qpos[qadr:qadr + 3], dtype=float)
            linear_velocity = np.asarray(self.data.qvel[dadr:dadr + 3], dtype=float)
            angular_velocity = np.asarray(self.data.qvel[dadr + 3:dadr + 6], dtype=float)
            relative = wrist_rotation.T @ (position - wrist_position)
            slip_value: float | str = ""
            if bilateral:
                reference = self._relative_at_first_bilateral_contact.setdefault(body_name, relative.copy())
                slip_value = float(np.linalg.norm(relative - reference))
                self._max_slip_m[body_name] = max(self._max_slip_m[body_name], slip_value)
            self._writer.writerow((
                f"{simulation_time_s:.9g}", body_name, color,
                *(f"{float(value):.9g}" for value in position),
                *(f"{float(value):.9g}" for value in linear_velocity),
                *(f"{float(value):.9g}" for value in angular_velocity),
                f"{left:.9g}", f"{right:.9g}", int(bilateral),
                *(f"{float(value):.9g}" for value in relative),
                "" if slip_value == "" else f"{slip_value:.9g}",
            ))

        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            force = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            geom1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      int(contact.geom1)) or f"geom_{contact.geom1}"
            geom2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                      int(contact.geom2)) or f"geom_{contact.geom2}"
            self._contact_stream.write(json.dumps({
                "simulation_time_s": simulation_time_s,
                "geom1": geom1,
                "geom2": geom2,
                "distance_m": float(contact.dist),
                "normal_force_N": float(force[0]),
                "tangent_force_1_N": float(force[1]),
                "tangent_force_2_N": float(force[2]),
            }, ensure_ascii=False, allow_nan=False) + "\n")

    def summary(self) -> dict[str, float]:
        return {body_name: float(value) for body_name, value in self._max_slip_m.items()}

    def close(self) -> None:
        self._object_stream.flush()
        self._contact_stream.flush()
        self._object_stream.close()
        self._contact_stream.close()
