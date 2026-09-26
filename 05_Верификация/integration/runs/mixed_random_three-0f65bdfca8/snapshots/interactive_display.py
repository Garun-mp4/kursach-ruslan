from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .config import load_ssot
from .placement_camera import load_integrated_model


def copy_runtime_state_to_display(
    runtime_model: mujoco.MjModel,
    runtime_data: mujoco.MjData,
    display_model: mujoco.MjModel,
    display_data: mujoco.MjData,
) -> None:
    """Publish one state snapshot into a model/data pair owned only by the viewer."""
    dimensions = ("nq", "nv", "na", "nu", "nbody", "ngeom", "ncam")
    if any(getattr(runtime_model, name) != getattr(display_model, name)
           for name in dimensions):
        raise ValueError("Interactive display model layout differs from the runtime model")
    display_model.geom_rgba[:] = runtime_model.geom_rgba
    mujoco.mj_copyData(display_data, display_model, runtime_data)


class IsolatedInteractiveDisplay:
    """Interactive rendering with private MuJoCo model/data state.

    The GUI may pause, scrub, reset, or otherwise change its own state. Runtime
    actuators, physics time, and controller state are never shared with it.
    """

    def __init__(
        self,
        runtime_model: mujoco.MjModel,
        runtime_data: mujoco.MjData,
        viewer_module: Any,
    ) -> None:
        self._runtime_model = runtime_model
        self._runtime_data = runtime_data
        self._display_model = load_integrated_model(load_ssot())
        self._display_data = mujoco.MjData(self._display_model)
        self._publish_state()
        self._handle = viewer_module.launch_passive(
            self._display_model, self._display_data
        )

    def _publish_state(self) -> None:
        copy_runtime_state_to_display(
            self._runtime_model,
            self._runtime_data,
            self._display_model,
            self._display_data,
        )

    def is_running(self) -> bool:
        return self._handle.is_running()

    def sync(self) -> None:
        if not self._handle.is_running():
            return
        with self._handle.lock():
            self._publish_state()
        self._handle.sync()

    def close(self) -> None:
        self._handle.close()
