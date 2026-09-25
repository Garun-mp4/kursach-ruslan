from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from camera.mujoco_adapter import MuJoCoRGBAdapter


@dataclass(frozen=True)
class GroundTruthObject:
    evaluator_key: str
    class_label: str
    x_m: float
    y_m: float
    yaw_rad: float
    rgba: tuple[float, float, float, float]


def _name(model: mujoco.MjModel, obj: mujoco.mjtObj, idx: int) -> str | None:
    return mujoco.mj_id2name(model, obj, idx)


class PerceptionRig:
    """M2-derived MuJoCo camera fixture; all truth stays in this evaluator-side module."""

    OBJECT_NAMES = (
        "object_red_01", "object_green_01", "object_blue_01",
        "object_red_02", "object_green_02", "object_blue_02",
    )

    def __init__(self, root: Path, ssot: dict[str, Any], *, include_occluder: bool = False,
                 include_arm_visual: bool = False) -> None:
        self.root = root
        self.ssot = ssot
        self.parameters = ssot["parameters"]
        self.width, self.height = map(int, self.p("camera.resolution_px"))
        self.camera_name = "overhead_rgb"
        xml = self._fixture_xml(include_occluder)
        xml = hide_diagnostic_sites(xml)
        if not include_arm_visual:
            xml = hide_robot_geometries(xml)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._hide_robot_and_objects(hide_robot=not include_arm_visual)
        self.renderer = mujoco.Renderer(self.model, width=self.width, height=self.height)
        self.camera = MuJoCoRGBAdapter(
            self.model, self.renderer, self.camera_name, self.width, self.height,
            "overhead_rgb_M2_validated_v1",
        )
        self.object_slots: list[dict[str, Any]] = []
        for name in self.OBJECT_NAMES:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            joint_id = int(self.model.body_jntadr[body_id])
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name + "_visual")
            self.object_slots.append({
                "name": name,
                "body_id": body_id,
                "joint_id": joint_id,
                "qpos_adr": int(self.model.jnt_qposadr[joint_id]),
                "geom_id": geom_id,
            })
        self.occluder_geom_id = -1
        if include_occluder:
            self.occluder_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, "m4_occlusion_proxy"
            )
        if include_arm_visual:
            self._set_observation_arm_pose()
        self.base_ambient = self.model.light_ambient.copy()
        self.base_diffuse = self.model.light_diffuse.copy()
        self.base_specular = self.model.light_specular.copy()
        self._hide_all_objects()
        self.set_lighting(1.0)
        mujoco.mj_forward(self.model, self.data)

    def p(self, key: str) -> Any:
        return self.parameters[key]["value"]

    def close(self) -> None:
        self.renderer.close()

    def capture(self, *, sim_time_s: float = 0.0, noise_sigma: float = 0.0,
                noise_seed: int | None = None, valid: bool = True):
        return self.camera.capture(self.data, simulation_time_s=sim_time_s, noise_sigma=noise_sigma,
                                   noise_seed=noise_seed, valid=valid)

    def set_lighting(self, scale: float) -> None:
        self.model.light_ambient[:] = np.clip(self.base_ambient * float(scale), 0, 1)
        self.model.light_diffuse[:] = np.clip(self.base_diffuse * float(scale), 0, 1)
        self.model.light_specular[:] = self.base_specular

    def set_objects(self, objects: list[GroundTruthObject], *, lighting_scale: float = 1.0,
                    occlusion: tuple[float, float] | None = None) -> None:
        if len(objects) > len(self.object_slots):
            raise ValueError("M2 fixture contains six object slots")
        self._hide_all_objects()
        self.set_lighting(lighting_scale)
        for slot, obj in zip(self.object_slots, objects):
            adr = slot["qpos_adr"]
            half = float(obj.yaw_rad) / 2.0
            self.data.qpos[adr:adr + 7] = [obj.x_m, obj.y_m, float(self.p("object.position_z_m")),
                                            math.cos(half), 0.0, 0.0, math.sin(half)]
            self.model.geom_rgba[slot["geom_id"], :] = obj.rgba
        if self.occluder_geom_id >= 0:
            self.model.geom_rgba[self.occluder_geom_id, 3] = 0.0
            if occlusion is not None:
                x, y = occlusion
                self.model.geom_pos[self.occluder_geom_id, :] = [x + 0.006, y, 0.055]
                self.model.geom_rgba[self.occluder_geom_id, :] = [0.34, 0.36, 0.38, 1.0]
        mujoco.mj_forward(self.model, self.data)

    def _hide_all_objects(self) -> None:
        for slot in self.object_slots:
            self.model.geom_rgba[slot["geom_id"], 3] = 0.0
        if self.occluder_geom_id >= 0:
            self.model.geom_rgba[self.occluder_geom_id, 3] = 0.0

    def _hide_robot_and_objects(self, *, hide_robot: bool) -> None:
        for geom_id in range(self.model.ngeom):
            name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if (hide_robot and name.startswith(("base_", "link", "elbow", "z_", "wrist", "finger"))) or name.startswith("object_"):
                self.model.geom_rgba[geom_id, 3] = 0.0

    def _set_observation_arm_pose(self) -> None:
        pose = self.p("camera.safe_observation_joint_pose")
        for name, value in pose.items():
            if name == "gripper":
                continue
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Missing observation joint {name}")
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = float(value)
        mujoco.mj_forward(self.model, self.data)

    def render_empty(self):
        self._hide_all_objects()
        self.set_lighting(1.0)
        mujoco.mj_forward(self.model, self.data)
        return self.capture()

    def _fixture_xml(self, include_occluder: bool) -> str:
        source = self.root / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
        root = ET.parse(source).getroot()
        world = root.find("worldbody")
        if world is None:
            raise ValueError("M2 MJCF has no worldbody")
        if include_occluder:
            ET.SubElement(world, "geom", {
                "name": "m4_occlusion_proxy", "type": "box", "pos": "0 0 0.055",
                "size": "0.006 0.025 0.004", "rgba": "0.34 0.36 0.38 0",
                "contype": "0", "conaffinity": "0", "group": "1", "mass": "0",
            })
        return ET.tostring(root, encoding="unicode")


def marker_board_xml(root: Path, points: list[tuple[float, float]], *, z_top_m: float) -> str:
    source = root / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
    xml_root = ET.parse(source).getroot()
    world = xml_root.find("worldbody")
    if world is None:
        raise ValueError("M2 MJCF has no worldbody")
    for idx, (x, y) in enumerate(points):
        ET.SubElement(world, "geom", {
            "name": f"m4_cal_{idx:03d}", "type": "cylinder",
            "pos": f"{x:.9f} {y:.9f} {z_top_m - 0.0002:.9f}",
            "size": "0.006 0.0002", "rgba": "1 0.82 0.02 1",
            "contype": "0", "conaffinity": "0", "group": "1", "mass": "0",
        })
    return ET.tostring(xml_root, encoding="unicode")


def hide_robot_geometries(xml: str) -> str:
    root = ET.fromstring(xml)
    names_to_hide = ("base_", "link", "elbow", "z_", "wrist", "finger")
    for parent in root.iter():
        for child in list(parent):
            if child.tag == "geom" and child.attrib.get("name", "").startswith(names_to_hide):
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


def hide_diagnostic_sites(xml: str) -> str:
    root = ET.fromstring(xml)
    for parent in root.iter():
        for child in list(parent):
            if child.tag in {"site", "sensor"}:
                parent.remove(child)
    return ET.tostring(root, encoding="unicode")


def calibration_grid() -> list[tuple[float, float]]:
    xs = np.linspace(-0.42, 0.42, 7)
    ys = np.linspace(-0.36, 0.36, 7)[::-1]
    return [(float(x), float(y)) for y in ys for x in xs]
