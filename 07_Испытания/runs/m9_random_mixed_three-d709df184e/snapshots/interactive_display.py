from __future__ import annotations

import mujoco
import numpy as np
from mujoco.experimental.dear_imgui import dear_imgui as imgui
from mujoco.experimental.studio import launch_passive
from mujoco.experimental.studio import messages
from mujoco.experimental.studio import ux
from mujoco.experimental.studio import viewer_app
from mujoco.experimental.studio import viewer_protocol

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
    ) -> None:
        self._runtime_model = runtime_model
        self._runtime_data = runtime_data
        self._display_model = load_integrated_model(load_ssot())
        self._display_data = mujoco.MjData(self._display_model)
        self._publish_state()
        self._handle = launch_passive.launch_passive(
            viewer_protocol.ViewerConfig(
                title="Сортировка объектов — ЛКМ: выбор | ПКМ: вращение | WASD",
                width=1440,
                height=900,
            ),
            viewer_plugins=[_GameLikeViewerApp()],
        )
        self._handle.send_to_viewer(
            messages.ModelEvent(model=self._display_model)
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
        self._publish_state()
        self._display_model, self._display_data = self._handle.sync(
            self._display_model, self._display_data
        )

    def close(self) -> None:
        self._handle.close()


class _GameLikeViewerApp(viewer_app.ViewerApp):
    """Studio UI with conventional mouse and keyboard camera controls."""

    def __init__(self) -> None:
        super().__init__(
            viewer_app.ViewerAppConfig(
                show_options=True,
                show_inspector=True,
                show_status_bar=True,
            )
        )
        self.status = (
            "ЛКМ: выбрать объект | ПКМ: вращать | WASD: двигать камеру "
            "| колесо: приблизить/отдалить"
        )

    @messages.handler(priority=messages.Priority.USER)
    def _select_free_camera(self, event: messages.ModelEvent) -> bool:
        """Start each loaded model in a movable free-camera view."""
        del event
        self.ux_state.camera_index = ux.set_camera(
            self.model, self.viewer.camera, ux.FREE_CAMERA_IDX
        )
        return False

    def handle_keyboard_events(self) -> None:
        """Keep WASD navigation available after switching camera modes."""
        io = imgui.GetIO()
        if not io.WantCaptureKeyboard and any(
            imgui.IsKeyDown(key)
            for key in (imgui.Key.W, imgui.Key.A, imgui.Key.S, imgui.Key.D)
        ):
            if self.ux_state.camera_index != ux.FREE_CAMERA_IDX:
                self.ux_state.camera_index = ux.set_camera(
                    self.model, self.viewer.camera, ux.FREE_CAMERA_IDX
                )
        super().handle_keyboard_events()

    def handle_mouse_events(self) -> None:
        """Map left click to picking, right drag to orbit, and wheel to zoom."""
        io = imgui.GetIO()
        if (
            io.WantCaptureMouse
            or io.DisplaySize.x <= 0
            or io.DisplaySize.y <= 0
        ):
            return

        mouse_x = io.MousePos.x / io.DisplaySize.x
        mouse_y = io.MousePos.y / io.DisplaySize.y
        mouse_dx = io.MouseDelta.x / io.DisplaySize.x
        mouse_dy = io.MouseDelta.y / io.DisplaySize.y
        left_down = imgui.IsMouseDown(imgui.MouseButton.Left)
        right_down = imgui.IsMouseDown(imgui.MouseButton.Right)
        middle_down = imgui.IsMouseDown(imgui.MouseButton.Middle)
        any_mouse_down = left_down or right_down or middle_down
        dragging = (mouse_dx != 0.0 or mouse_dy != 0.0) and any_mouse_down
        perturb = self.viewer.perturb

        if not any_mouse_down:
            perturb.active = 0

        if dragging and io.KeyCtrl and perturb.select > 0:
            if left_down:
                action = int(
                    mujoco.mjtMouse.mjMOUSE_MOVE_H
                    if io.KeyShift
                    else mujoco.mjtMouse.mjMOUSE_MOVE_V
                )
            elif right_down:
                action = int(
                    mujoco.mjtMouse.mjMOUSE_ROTATE_H
                    if io.KeyShift
                    else mujoco.mjtMouse.mjMOUSE_ROTATE_V
                )
            else:
                action = int(mujoco.mjtMouse.mjMOUSE_ZOOM)

            active = int(
                mujoco.mjtPertBit.mjPERT_TRANSLATE
                if action
                in (
                    int(mujoco.mjtMouse.mjMOUSE_MOVE_H),
                    int(mujoco.mjtMouse.mjMOUSE_MOVE_V),
                )
                else mujoco.mjtPertBit.mjPERT_ROTATE
            )
            if active != perturb.active:
                ux.InitPerturb(
                    self.model, self.data, self.viewer.camera, perturb, active
                )
            ux.MovePerturb(
                self.model,
                self.data,
                self.viewer.camera,
                perturb,
                action,
                mouse_dx,
                mouse_dy,
            )
        elif dragging and right_down:
            motion = (
                ux.CameraMotion.PAN_TILT
                if self.ux_state.camera_index == ux.FREE_CAMERA_IDX
                else ux.CameraMotion.ORBIT
            )
            ux.MoveCamera(
                self.model,
                self.data,
                self.viewer.camera,
                motion,
                mouse_dx,
                mouse_dy,
            )
        elif dragging and middle_down:
            motion = (
                ux.CameraMotion.PLANAR_MOVE_H
                if io.KeyShift
                else ux.CameraMotion.PLANAR_MOVE_V
            )
            ux.MoveCamera(
                self.model,
                self.data,
                self.viewer.camera,
                motion,
                mouse_dx,
                mouse_dy,
            )

        # MuJoCo's default binding negates the wheel delta. Reverse that sign
        # so scrolling away from the user decreases camera distance.
        if io.MouseWheel != 0.0:
            ux.MoveCamera(
                self.model,
                self.data,
                self.viewer.camera,
                ux.CameraMotion.ZOOM,
                0.0,
                float(io.MouseWheel) / 50.0,
            )

        if imgui.IsMouseClicked(imgui.MouseButton.Left):
            picked = ux.Pick(
                self.model,
                self.data,
                self.viewer.camera,
                mouse_x,
                mouse_y,
                io.DisplaySize.x / io.DisplaySize.y,
                self.viewer.vis_options,
            )
            if picked.body >= 0:
                perturb.select = picked.body
                perturb.flexselect = picked.flex
                perturb.skinselect = picked.skin
                offset = np.asarray(picked.point, dtype=np.float64) - self.data.xpos[
                    picked.body
                ]
                rotation = np.asarray(
                    self.data.xmat[picked.body], dtype=np.float64
                ).reshape(3, 3)
                perturb.localpos = rotation.T @ offset
                body_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, picked.body
                )
                self.status = (
                    f"Выбрано: {body_name or f'body {picked.body}'} "
                    "| ПКМ: вращение | WASD: камера | колесо: масштаб"
                )
            else:
                perturb.select = 0
                perturb.flexselect = -1
                perturb.skinselect = -1
                self.status = (
                    "ЛКМ: выбрать объект | ПКМ: вращать | WASD: двигать камеру "
                    "| колесо: приблизить/отдалить"
                )

        self.handle_camera_tracking_mouse_events()
