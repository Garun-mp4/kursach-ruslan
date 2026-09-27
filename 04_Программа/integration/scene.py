from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import random
from typing import Any

import mujoco
import numpy as np
import yaml

from .config import ROOT, RuntimeConfig, load_ssot, value

SCENARIO_DIR = Path(__file__).with_name("scenarios")
OBJECT_BODIES = tuple(
    f"object_{color.lower()}_{index:02d}"
    for index in (1, 2)
    for color in ("RED", "GREEN", "BLUE")
)


@dataclass(frozen=True)
class SceneObjectTruth:
    body_name: str
    class_label: str
    initial_xy_m: tuple[float, float]
    yaw_rad: float


@dataclass(frozen=True)
class GeneratedScene:
    scenario_id: str
    seed: int
    active_objects: tuple[SceneObjectTruth, ...]
    initial_slot_occupants: tuple[tuple[str, str], ...]
    inactive_bodies: tuple[str, ...]


def load_scenario(name: str, scenario_dir: str | Path = SCENARIO_DIR) -> dict[str, Any]:
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError("Scenario must be named by a simple file stem")
    path = Path(scenario_dir) / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("scenario_id") != name:
        raise ValueError("Scenario ID must match its filename")
    if not isinstance(raw.get("seed"), int) or isinstance(raw.get("seed"), bool):
        raise ValueError("Scenario seed must be an integer")
    return raw


class SceneGenerator:
    """Creates reproducible initial conditions; controller never receives its truth map."""

    def __init__(self, ssot: dict[str, Any] | None = None,
                 runtime_config: RuntimeConfig | None = None):
        self.ssot = ssot or load_ssot()
        self.runtime_config = runtime_config or RuntimeConfig.load()

    @staticmethod
    def _free_joint_qpos_address(model: mujoco.MjModel, body_name: str) -> int:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Unknown object body in scenario: {body_name}")
        joint_id = int(model.body_jntadr[body_id])
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise ValueError(f"Scenario body {body_name} does not have a free joint")
        return int(model.jnt_qposadr[joint_id])

    def _randomized_objects(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        randomized = config["randomized"]
        count = randomized.get("count_per_class")
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 2:
            raise ValueError("randomized.count_per_class must be 1 or 2")
        raw_colors = randomized.get("colors", ("RED", "GREEN", "BLUE"))
        if not isinstance(raw_colors, (list, tuple)) or not raw_colors:
            raise ValueError("randomized.colors must be a non-empty list of supported classes")
        colors = tuple(str(color).upper() for color in raw_colors)
        if (len(set(colors)) != len(colors)
                or any(color not in {"RED", "GREEN", "BLUE"} for color in colors)):
            raise ValueError("randomized.colors must contain unique RED/GREEN/BLUE classes")
        yaw_range = tuple(map(float, randomized.get("yaw_range_rad", (-math.pi / 4, math.pi / 4))))
        if len(yaw_range) != 2 or yaw_range[0] > yaw_range[1]:
            raise ValueError("yaw_range_rad must be an ordered pair")
        profile = self.ssot["perception"]["runtime_config"]
        bounds = profile["regions"]["input_xy_bounds_m"]
        (xmin, xmax), (ymin, ymax) = [tuple(map(float, axis)) for axis in bounds]
        half = np.asarray(value(self.ssot, "object.size_xyz_m")[:2], dtype=float) / 2.0
        xmin, xmax = xmin + half[0], xmax - half[0]
        ymin, ymax = ymin + half[1], ymax - half[1]
        min_spacing = self.runtime_config.randomized_scene_min_object_spacing_m
        rng = random.Random(int(config["seed"]))
        points: list[tuple[float, float]] = []
        max_attempts = self.runtime_config.randomized_scene_max_attempts
        target_count = len(colors) * count
        while len(points) < target_count and max_attempts:
            max_attempts -= 1
            candidate = (rng.uniform(xmin, xmax), rng.uniform(ymin, ymax))
            if all(math.dist(candidate, point) >= min_spacing for point in points):
                points.append(candidate)
        if len(points) != target_count:
            raise RuntimeError("Could not generate a non-overlapping randomized scene")
        objects: list[dict[str, Any]] = []
        cursor = 0
        for color in colors:
            for body_index in range(1, count + 1):
                objects.append({
                    "body_name": f"object_{color.lower()}_{body_index:02d}",
                    "xy_m": list(points[cursor]),
                    "yaw_rad": rng.uniform(*yaw_range),
                })
                cursor += 1
        shuffle_order = randomized.get("shuffle_order", False)
        if not isinstance(shuffle_order, bool):
            raise ValueError("randomized.shuffle_order must be a boolean")
        if shuffle_order:
            rng.shuffle(objects)
        return objects

    def generate(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        config: dict[str, Any],
    ) -> GeneratedScene:
        scenario_id, seed = str(config["scenario_id"]), int(config["seed"])
        active_specs = config.get("active_objects")
        if active_specs is None:
            if "randomized" not in config:
                raise ValueError("Scenario must define active_objects or randomized settings")
            active_specs = self._randomized_objects(config)
        if not isinstance(active_specs, list):
            raise ValueError("Scenario active_objects must be a list (which may be empty)")
        seen: set[str] = set()
        table_z = float(value(self.ssot, "cell.table_top_z_m"))
        object_height = float(value(self.ssot, "object.size_xyz_m")[2])
        object_size_xy = np.asarray(value(self.ssot, "object.size_xyz_m")[:2], dtype=float)
        profile = self.ssot["perception"]["runtime_config"]
        (xmin, xmax), (ymin, ymax) = [tuple(map(float, axis)) for axis in
                                      profile["regions"]["input_xy_bounds_m"]]
        truths: list[SceneObjectTruth] = []
        for spec in active_specs:
            body_name = str(spec["body_name"])
            if body_name not in OBJECT_BODIES or body_name in seen:
                raise ValueError(f"Invalid or duplicate scenario body: {body_name}")
            seen.add(body_name)
            xy = tuple(map(float, spec["xy_m"]))
            yaw = float(spec["yaw_rad"])
            if len(xy) != 2 or not all(math.isfinite(v) for v in (*xy, yaw)):
                raise ValueError(f"Non-finite scenario pose for {body_name}")
            if not (xmin + object_size_xy[0]/2 <= xy[0] <= xmax - object_size_xy[0]/2
                    and ymin + object_size_xy[1]/2 <= xy[1] <= ymax - object_size_xy[1]/2):
                raise ValueError(f"Scenario object {body_name} lies outside the input work region")
            qadr = self._free_joint_qpos_address(model, body_name)
            data.qpos[qadr:qadr + 7] = (
                xy[0], xy[1], table_z + object_height/2,
                math.cos(yaw/2), 0.0, 0.0, math.sin(yaw/2),
            )
            color = body_name.split("_")[1].upper()
            truths.append(SceneObjectTruth(body_name, color, xy, yaw))

        slot_offsets = tuple(map(float, value(self.ssot, "cell.tray_slot_x_offsets_m")))
        slot_y_offset = float(value(self.ssot, "cell.tray_slot_y_offset_m"))
        tray_centers = value(self.ssot, "cell.tray_centers_xy_m")
        slots = {
            f"{color}:{index}": (
                float(tray_centers[color][0]) + offset,
                float(tray_centers[color][1]) + slot_y_offset,
            )
            for color in ("RED", "GREEN", "BLUE")
            for index, offset in enumerate(slot_offsets)
        }
        occupant_specs = config.get("initial_slot_occupants", [])
        if not isinstance(occupant_specs, list):
            raise ValueError("initial_slot_occupants must be a list")
        initial_occupants: list[tuple[str, str]] = []
        occupied_slots: set[str] = set()
        for spec in occupant_specs:
            if not isinstance(spec, dict):
                raise ValueError("Each initial slot occupant must be a mapping")
            body_name = str(spec.get("body_name", ""))
            slot_id = str(spec.get("slot_id", ""))
            if body_name not in OBJECT_BODIES or body_name in seen:
                raise ValueError(f"Invalid or duplicate initial occupant body: {body_name}")
            if slot_id not in slots or slot_id in occupied_slots:
                raise ValueError(f"Invalid or duplicate initial occupied slot: {slot_id}")
            yaw = float(spec.get("yaw_rad", 0.0))
            if not math.isfinite(yaw):
                raise ValueError(f"Non-finite yaw for initial slot occupant {body_name}")
            xy = slots[slot_id]
            qadr = self._free_joint_qpos_address(model, body_name)
            data.qpos[qadr:qadr + 7] = (
                xy[0], xy[1], table_z + object_height / 2,
                math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2),
            )
            seen.add(body_name)
            occupied_slots.add(slot_id)
            initial_occupants.append((body_name, slot_id))

        # Keep unused model objects resting outside the reachable input area. They are
        # still ordinary physical bodies; they are neither deleted nor teleported later.
        inactive = tuple(body for body in OBJECT_BODIES if body not in seen)
        table_x_limit = float(value(self.ssot, "cell.table_size_xy_m")[0]) / 2
        park_x = table_x_limit - float(object_size_xy[0])
        park_y_values = np.linspace(-0.30, 0.30, max(1, len(inactive)))
        for index, body_name in enumerate(inactive):
            qadr = self._free_joint_qpos_address(model, body_name)
            data.qpos[qadr:qadr + 7] = (
                park_x, float(park_y_values[index]), table_z + object_height/2,
                1.0, 0.0, 0.0, 0.0,
            )
        if len(inactive) and (park_x + object_size_xy[0]/2 > table_x_limit + 1e-9):
            raise ValueError("Inactive object parking pose exceeds the table top")
        mujoco.mj_forward(model, data)
        return GeneratedScene(scenario_id, seed, tuple(truths), tuple(initial_occupants), inactive)
