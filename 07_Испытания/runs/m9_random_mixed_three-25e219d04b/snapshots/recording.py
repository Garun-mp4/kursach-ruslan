from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import cv2
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import ROOT


@dataclass(frozen=True)
class RecordingConfig:
    """Viewer-only recording settings; they never change the sensor cameras."""

    fps: int = 25
    playback_speed: float = 5.0
    width: int = 1280
    height: int = 720
    main_render_size: tuple[int, int] = (960, 720)
    detail_render_size: tuple[int, int] = (640, 628)

    def __post_init__(self) -> None:
        if self.fps <= 0 or not math.isfinite(self.playback_speed) or self.playback_speed <= 0:
            raise ValueError("Recording FPS and playback speed must be finite and positive")
        if (self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2
                or min(*self.main_render_size, *self.detail_render_size) <= 0):
            raise ValueError("MP4 dimensions must be positive even pixel values")
        if self.main_render_size[0] + self.detail_render_size[0] // 2 != self.width:
            raise ValueError("Main and detail render widths must compose to the output width")
        if self.main_render_size[1] != self.height:
            raise ValueError("Main view height must match the output height")

    @property
    def sample_period_s(self) -> float:
        return self.playback_speed / self.fps


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _source_log_hashes(run_manifest: dict[str, Any]) -> dict[str, Any]:
    run_directory = Path(run_manifest["run_directory"])
    if not run_directory.is_absolute():
        run_directory = ROOT / run_directory
    result: dict[str, Any] = {}
    for name, relative_path in run_manifest.get("relative_outputs", {}).items():
        source = (run_directory / relative_path).resolve()
        if source.is_file():
            result[name] = {"path": _relative(source), "sha256": _sha256(source)}
        elif source.is_dir():
            files = sorted(item for item in source.rglob("*") if item.is_file())
            result[name] = {
                "path": _relative(source),
                "file_count": len(files),
                "files": [
                    {"path": _relative(item), "sha256": _sha256(item)}
                    for item in files
                ],
            }
        else:
            raise FileNotFoundError(f"Run output listed in manifest does not exist: {source}")
    return result


class RunStateRecorder:
    """Captures integration states from the active M7 run for later viewer-only rendering."""

    def __init__(self, model: mujoco.MjModel, *, run_id: str, run_dir: Path,
                 config: RecordingConfig | None = None):
        if not run_id:
            raise ValueError("Recording requires a non-empty run_id")
        self.model = model
        self.run_id = run_id
        self.config = config or RecordingConfig()
        self.output_dir = Path(run_dir) / "recording"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self.state_size = int(mujoco.mj_stateSize(model, self.state_signature))
        self._timeline_states: list[np.ndarray] = []
        self._timeline_metadata: list[dict[str, Any]] = []
        self._keyframe_states: dict[str, np.ndarray] = {}
        self._keyframe_metadata: dict[str, dict[str, Any]] = {}
        self._next_sample_s: float | None = None
        self._pending_events: list[dict[str, Any]] = []
        self._previous_time_s: float | None = None
        self._display_track_id: int | None = None
        self._dropped_frames = 0
        self._saved = False
        self.state_stream_path = self.output_dir / "state_stream.npz"
        self.state_manifest_path = self.output_dir / "state_stream_manifest.json"

    @staticmethod
    def _event_document(event: Any) -> dict[str, Any]:
        if isinstance(event, dict):
            state = event.get("state")
            if hasattr(state, "value"):
                state = state.value
            return {
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "simulation_time_s": event.get("simulation_time_s"),
                "state": state,
                "reason_code": event.get("reason_code"),
                "track_id": event.get("track_id"),
            }
        state = getattr(event, "state", None)
        return {
            "event_id": getattr(event, "event_id", None),
            "event_type": getattr(event, "event_type", None),
            "simulation_time_s": getattr(event, "simulation_time_s", None),
            "state": getattr(state, "value", state),
            "reason_code": getattr(event, "reason_code", None),
            "track_id": getattr(event, "track_id", None),
        }

    def _snapshot(self, data: mujoco.MjData) -> np.ndarray:
        state = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(self.model, data, state, self.state_signature)
        return state

    @staticmethod
    def _records_summary(track_records: dict[int, Any]) -> dict[str, Any]:
        normalized: dict[int, Any] = {}
        for track_id, record in track_records.items():
            normalized[int(track_id)] = record
        reserved = [
            (track_id, record) for track_id, record in normalized.items()
            if getattr(getattr(record, "status", None), "value", None) == "RESERVED"
        ]
        confirmed = sum(
            getattr(getattr(record, "status", None), "value", None)
            == "PLACED_CONTROLLER_CONFIRMED"
            for record in normalized.values()
        )
        return {
            "active_track_id": min((item[0] for item in reserved), default=None),
            "estimated_color": (
                normalized[min(item[0] for item in reserved)].class_label if reserved else None
            ),
            "discovered_track_count": len(normalized),
            "controller_confirmed_placements": int(confirmed),
        }

    def _metadata(self, *, simulation_time_s: float, state: str,
                  track_records: dict[int, Any], events: Iterable[dict[str, Any]],
                  terminal: bool = False) -> dict[str, Any]:
        summary = self._records_summary(track_records)
        if summary["active_track_id"] is not None:
            self._display_track_id = summary["active_track_id"]
        elif events:
            selected = next((event for event in reversed(list(events))
                             if event.get("event_type") == "TARGET_SELECTED"), None)
            if selected and selected.get("track_id") is not None:
                self._display_track_id = int(selected["track_id"])
        color = summary["estimated_color"]
        if color is None and self._display_track_id in track_records:
            color = track_records[self._display_track_id].class_label
        return {
            "run_id": self.run_id,
            "simulation_time_s": float(simulation_time_s),
            "controller_state": str(state),
            "active_track_id": summary["active_track_id"],
            "display_track_id": self._display_track_id,
            "estimated_color": color,
            "discovered_track_count": summary["discovered_track_count"],
            "controller_confirmed_placements": summary["controller_confirmed_placements"],
            "events": list(events),
            "terminal": bool(terminal),
        }

    @staticmethod
    def _keyframe_labels(events: Iterable[dict[str, Any]]) -> list[str]:
        labels: list[str] = []
        for event in events:
            event_type = event.get("event_type")
            state = event.get("state")
            if event_type == "TARGET_SELECTED":
                labels.append("detection")
            if event_type == "STATE_ENTERED":
                if state in {"GRASP", "VERIFY_HOLD"}:
                    labels.append("grasp")
                elif state == "TRANSFER":
                    labels.append("transfer")
                elif state == "RELEASE":
                    labels.append("release")
                elif state in {"RECOVER", "SAFE_STOP"}:
                    labels.append("error_recovery")
                elif state == "DONE":
                    labels.append("final")
        return list(dict.fromkeys(labels))

    def observe(self, data: mujoco.MjData, *, controller_state: str,
                track_records: dict[int, Any], events: Iterable[Any] = (),
                force_sample: bool = False, terminal: bool = False) -> None:
        if self._saved:
            raise RuntimeError("Cannot append to a finalized recording")
        now_s = float(data.time)
        if not math.isfinite(now_s):
            raise ValueError("Simulation time in recording must be finite")
        if self._previous_time_s is not None and now_s < self._previous_time_s - 1e-12:
            raise ValueError("Recording timestamps must be monotonic")
        event_documents = [self._event_document(event) for event in events]
        event_documents = [item for item in event_documents if item["event_type"]]
        self._pending_events.extend(event_documents)
        metadata = self._metadata(
            simulation_time_s=now_s, state=controller_state,
            track_records=track_records, events=event_documents, terminal=terminal,
        )
        snapshot = self._snapshot(data)
        labels = self._keyframe_labels(event_documents)
        if not self._keyframe_states and not self._timeline_states:
            labels.insert(0, "before_detection")
        for label in labels:
            if label not in self._keyframe_states:
                self._keyframe_states[label] = snapshot.copy()
                self._keyframe_metadata[label] = dict(metadata)
        due = force_sample or self._next_sample_s is None or now_s + 1e-10 >= self._next_sample_s
        if due:
            sample_metadata = dict(metadata)
            sample_metadata["events"] = list(self._pending_events)
            self._pending_events.clear()
            self._timeline_states.append(snapshot.copy())
            self._timeline_metadata.append(sample_metadata)
            if self._next_sample_s is None:
                self._next_sample_s = now_s + self.config.sample_period_s
            else:
                missed = max(0, int(math.floor(
                    (now_s - self._next_sample_s + 1e-10) / self.config.sample_period_s
                )))
                self._dropped_frames += missed
                while self._next_sample_s <= now_s + 1e-10:
                    self._next_sample_s += self.config.sample_period_s
        self._previous_time_s = now_s

    def finalize(self, data: mujoco.MjData, *, controller_state: str,
                 track_records: dict[int, Any]) -> dict[str, Any]:
        now_s = float(data.time)
        final_metadata = self._metadata(
            simulation_time_s=now_s, state=controller_state,
            track_records=track_records, events=[],
            terminal=controller_state in {"DONE", "SAFE_STOP"},
        )
        if not self._timeline_metadata or abs(
            float(self._timeline_metadata[-1]["simulation_time_s"]) - now_s
        ) > 1e-10:
            final_metadata["events"] = list(self._pending_events)
            self._pending_events.clear()
            self._timeline_states.append(self._snapshot(data))
            self._timeline_metadata.append(final_metadata)
        else:
            self._timeline_metadata[-1]["terminal"] = True
            self._timeline_metadata[-1]["controller_state"] = str(controller_state)
            self._timeline_metadata[-1]["events"].extend(self._pending_events)
            self._pending_events.clear()
        final_label = (
            "error_recovery" if controller_state == "SAFE_STOP"
            else "final" if controller_state == "DONE" else "run_end"
        )
        if final_label in self._keyframe_metadata:
            final_metadata["events"] = list(
                self._keyframe_metadata[final_label].get("events", [])
            )
        self._keyframe_states[final_label] = self._snapshot(data)
        self._keyframe_metadata[final_label] = final_metadata
        self.save_state_stream()
        return {
            "schema_version": "M8-state-stream-v1",
            "run_id": self.run_id,
            "state_stream": _relative(self.state_stream_path),
            "state_stream_manifest": _relative(self.state_manifest_path),
            "frame_count": len(self._timeline_states),
            "keyframes": sorted(self._keyframe_states),
            "sampling_period_s": self.config.sample_period_s,
            "fps": self.config.fps,
            "playback_speed": self.config.playback_speed,
            "dropped_frames": self._dropped_frames,
        }

    def save_state_stream(self) -> None:
        if not self._timeline_states:
            return
        labels = sorted(self._keyframe_states)
        np.savez_compressed(
            self.state_stream_path,
            timeline_states=np.stack(self._timeline_states),
            keyframe_states=np.stack([self._keyframe_states[label] for label in labels]),
            keyframe_labels=np.asarray(labels, dtype="U32"),
        )
        manifest = {
            "schema_version": "M8-state-stream-v1",
            "run_id": self.run_id,
            "state_signature": "mjSTATE_INTEGRATION",
            "state_size": self.state_size,
            "sample_period_s": self.config.sample_period_s,
            "fps": self.config.fps,
            "playback_speed": self.config.playback_speed,
            "timeline": self._timeline_metadata,
            "keyframes": {
                label: self._keyframe_metadata[label] for label in labels
            },
            "dropped_frame_count": self._dropped_frames,
            "camera_separation": (
                "Viewer cameras are created only during post-run rendering; they do not replace "
                "overhead_rgb or placement_rgb sensor cameras."
            ),
        }
        self.state_manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        self._saved = True

    @staticmethod
    def _fonts(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        candidates = (
            Path(r"C:\Windows\Fonts\arial.ttf"),
            Path(r"C:\Windows\Fonts\segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return ImageFont.truetype(str(candidate), size=size)
        return ImageFont.load_default()

    @staticmethod
    def _free_camera(model: mujoco.MjModel, *, lookat: tuple[float, float, float],
                     distance: float, azimuth: float, elevation: float) -> mujoco.MjvCamera:
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, camera)
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = distance
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.trackbodyid = -1
        return camera

    def _render_views(self, model: mujoco.MjModel, data: mujoco.MjData,
                      main_renderer: mujoco.Renderer, detail_renderer: mujoco.Renderer,
                      cameras: dict[str, mujoco.MjvCamera]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        main_renderer.update_scene(data, camera=cameras["main"])
        detail_renderer.update_scene(data, camera=cameras["top"])
        main = np.asarray(main_renderer.render()).copy()
        top = np.asarray(detail_renderer.render()).copy()
        tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        if tcp_id < 0:
            raise RuntimeError("M8 viewer requires the M2 tcp site")
        cameras["close"].lookat[:] = data.site_xpos[tcp_id]
        cameras["close"].lookat[2] += 0.025
        detail_renderer.update_scene(data, camera=cameras["close"])
        close = np.asarray(detail_renderer.render()).copy()
        return main, top, close

    def _compose(self, views: tuple[np.ndarray, np.ndarray, np.ndarray],
                 metadata: dict[str, Any], frame_index: int, *, screenshot_label: str | None = None
                 ) -> np.ndarray:
        main_rgb, top_rgb, close_rgb = views
        canvas = Image.new("RGB", (self.config.width, self.config.height), (18, 26, 38))
        canvas.paste(Image.fromarray(main_rgb).resize(self.config.main_render_size), (0, 0))
        panel_width = self.config.width - self.config.main_render_size[0]
        side_x = self.config.main_render_size[0]
        panel_height = self.config.height // 2
        for panel, title, origin, available_height in (
            (top_rgb, "ВИД СВЕРХУ · зрительская камера", (side_x, 0), panel_height),
            (close_rgb, "ЗАХВАТ · крупный план", (side_x, panel_height), panel_height),
        ):
            panel_image = Image.fromarray(panel).resize((panel_width, available_height - 46))
            canvas.paste(panel_image, (origin[0], origin[1] + 34))
        draw = ImageDraw.Draw(canvas)
        font = self._fonts(22)
        small = self._fonts(17)
        title_font = self._fonts(20)
        draw.rectangle((0, 0, self.config.main_render_size[0], 112), fill=(16, 25, 37))
        draw.text((22, 12), "ПРОГРАММНАЯ СИМУЛЯЦИЯ", font=self._fonts(26), fill=(255, 255, 255))
        draw.text((23, 51), f"RUN {self.run_id}   ·   {float(metadata['simulation_time_s']):.2f} с   ·   x{self.config.playback_speed:g}",
                  font=small, fill=(198, 213, 226))
        target_id = metadata.get("display_track_id")
        target_label = f"#{target_id}" if target_id is not None else "—"
        color = metadata.get("estimated_color") or "—"
        state = str(metadata.get("controller_state", "UNKNOWN"))
        placed = int(metadata.get("controller_confirmed_placements", 0))
        seen = int(metadata.get("discovered_track_count", 0))
        draw.text((22, 78), f"FSM {state}   ·   объект {target_label}   ·   оценка цвета {color}   ·   учтено контроллером {placed}/{seen}",
                  font=small, fill=(236, 241, 246))
        draw.rectangle((0, self.config.height - 34, self.config.main_render_size[0], self.config.height),
                       fill=(16, 25, 37))
        draw.text((22, self.config.height - 29), "Свидетельная визуализация; камера зрителя отделена от RGB-камер управления.",
                  font=self._fonts(15), fill=(190, 205, 218))
        top_x = side_x
        for title, y in (("ВИД СВЕРХУ · обзор ячейки", 0), ("ЗАХВАТ · крупный план TCP", panel_height)):
            draw.rectangle((top_x, y, self.config.width, y + 34), fill=(16, 25, 37))
            draw.text((top_x + 10, y + 6), title, font=title_font, fill=(255, 255, 255))
        legend_y = panel_height - 34
        draw.rectangle((top_x, legend_y, self.config.width, panel_height), fill=(16, 25, 37))
        draw.text((top_x + 8, legend_y + 8), "ЗОНЫ:", font=small, fill=(240, 243, 247))
        x = top_x + 75
        for label, color_rgb in (("RED", (220, 67, 58)), ("GREEN", (60, 180, 90)),
                                 ("BLUE", (58, 103, 220))):
            draw.rounded_rectangle((x, legend_y + 9, x + 13, legend_y + 22), radius=3, fill=color_rgb)
            draw.text((x + 17, legend_y + 6), label, font=small, fill=(240, 243, 247))
            x += 78
        if screenshot_label:
            badge_x = self.config.main_render_size[0] - 220
            draw.rounded_rectangle((badge_x, 12, badge_x + 206, 42),
                                   radius=8, fill=(245, 190, 64))
            draw.text((badge_x + 10, 17), screenshot_label.upper(),
                      font=small, fill=(22, 28, 36))
        draw.text((self.config.width - 160, self.config.height - 26), f"Кадр {frame_index:04d}",
                  font=self._fonts(14), fill=(170, 188, 204))
        return np.asarray(canvas, dtype=np.uint8)[:, :, ::-1].copy()

    @staticmethod
    def _open_video_writer(path: Path, fps: int, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
        failures: list[str] = []
        for codec in ("avc1", "mp4v"):
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*codec), float(fps), size,
            )
            if writer.isOpened():
                return writer, codec
            writer.release()
            path.unlink(missing_ok=True)
            failures.append(codec)
        raise RuntimeError(f"No usable MP4 encoder; attempted codecs: {', '.join(failures)}")

    @staticmethod
    def _decode_video(path: Path) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open generated video {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        decoded = 0
        first_frame = None
        last_frame = None
        while True:
            success, frame = capture.read()
            if not success:
                break
            if first_frame is None:
                first_frame = frame
            last_frame = frame
            decoded += 1
        capture.release()
        if decoded == 0 or first_frame is None or last_frame is None:
            raise RuntimeError("Generated MP4 contains no decodable frames")
        return {"decoded_frame_count": decoded, "fps": fps, "width": width,
                "height": height, "first_frame_mean_bgr": first_frame.mean(axis=(0, 1)).tolist(),
                "last_frame_mean_bgr": last_frame.mean(axis=(0, 1)).tolist()}

    def render(self, model: mujoco.MjModel, *, media_dir: Path,
               run_manifest: dict[str, Any]) -> dict[str, Any]:
        if not self.state_stream_path.is_file() or not self.state_manifest_path.is_file():
            raise FileNotFoundError("The recorded M7 state stream is missing")
        media_dir = Path(media_dir)
        keyframes_dir = media_dir / "кадры"
        keyframes_dir.mkdir(parents=True, exist_ok=True)
        with np.load(self.state_stream_path, allow_pickle=False) as stream:
            timeline_states = np.asarray(stream["timeline_states"], dtype=np.float64)
            keyframe_states = np.asarray(stream["keyframe_states"], dtype=np.float64)
            keyframe_labels = [str(item) for item in stream["keyframe_labels"].tolist()]
        state_manifest = json.loads(self.state_manifest_path.read_text(encoding="utf-8"))
        timeline = state_manifest["timeline"]
        if len(timeline_states) != len(timeline) or len(timeline_states) < 2:
            raise RuntimeError("Recorded timeline and state array lengths disagree or are too short")
        out_path = media_dir / "симуляция.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        main_width, main_height = self.config.main_render_size
        detail_width, detail_height = self.config.detail_render_size
        main_renderer = mujoco.Renderer(model, height=main_height, width=main_width)
        detail_renderer = mujoco.Renderer(model, height=detail_height, width=detail_width)
        render_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, render_data)
        cameras = {
            "main": self._free_camera(
                model, lookat=(0.0, 0.045, 0.19), distance=1.27,
                azimuth=135.0, elevation=-39.0,
            ),
            "top": self._free_camera(
                model, lookat=(0.0, 0.045, 0.0), distance=1.22,
                azimuth=90.0, elevation=-89.5,
            ),
            "close": self._free_camera(
                model, lookat=(0.0, 0.0, 0.28), distance=0.45,
                azimuth=45.0, elevation=-75.0,
            ),
        }
        writer: cv2.VideoWriter | None = None
        codec: str | None = None
        try:
            first_meta = timeline[0]
            mujoco.mj_setState(model, render_data, timeline_states[0], self.state_signature)
            mujoco.mj_forward(model, render_data)
            first_frame = self._compose(
                self._render_views(model, render_data, main_renderer, detail_renderer, cameras),
                first_meta, 0,
            )
            writer, codec = self._open_video_writer(
                out_path, self.config.fps, (self.config.width, self.config.height)
            )
            writer.write(first_frame)
            for index in range(1, len(timeline_states)):
                mujoco.mj_setState(model, render_data, timeline_states[index], self.state_signature)
                mujoco.mj_forward(model, render_data)
                frame = self._compose(
                    self._render_views(model, render_data, main_renderer, detail_renderer, cameras),
                    timeline[index], index,
                )
                writer.write(frame)
            writer.release()
            writer = None

            timeline_times = np.asarray(
                [float(item["simulation_time_s"]) for item in timeline], dtype=float
            )
            keyframe_records: list[dict[str, Any]] = []
            for label, state in zip(keyframe_labels, keyframe_states, strict=True):
                metadata = state_manifest["keyframes"][label]
                mujoco.mj_setState(model, render_data, state, self.state_signature)
                mujoco.mj_forward(model, render_data)
                nearest = int(np.argmin(np.abs(
                    timeline_times - float(metadata["simulation_time_s"])
                )))
                screenshot = self._compose(
                    self._render_views(model, render_data, main_renderer, detail_renderer, cameras),
                    metadata, nearest, screenshot_label=label,
                )
                image_name = f"{label}.png"
                image_path = keyframes_dir / image_name
                Image.fromarray(screenshot[:, :, ::-1]).save(image_path, format="PNG", optimize=True)
                keyframe_records.append({
                    "label": label,
                    "image": _relative(image_path),
                    "simulation_time_s": float(metadata["simulation_time_s"]),
                    "controller_state": metadata["controller_state"],
                    "nearest_video_frame_index": nearest,
                    "nearest_frame_time_s": float(timeline_times[nearest]),
                    "nearest_frame_delta_s": float(abs(
                        timeline_times[nearest] - float(metadata["simulation_time_s"])
                    )),
                    "event_ids": [event.get("event_id") for event in metadata.get("events", [])],
                })
        finally:
            if writer is not None:
                writer.release()
            main_renderer.close()
            detail_renderer.close()

        decoded = self._decode_video(out_path)
        expected_frames = len(timeline)
        if decoded["decoded_frame_count"] != expected_frames:
            raise RuntimeError(
                f"MP4 decode count {decoded['decoded_frame_count']} differs from recorded frames {expected_frames}"
            )
        if (decoded["width"], decoded["height"]) != (self.config.width, self.config.height):
            raise RuntimeError("Decoded MP4 dimensions differ from the recording configuration")
        start_s = float(timeline_times[0])
        end_s = float(timeline_times[-1])
        video_duration_s = expected_frames / self.config.fps
        input_hashes = run_manifest.get("input_hashes_sha256", {})
        manifest_path = Path(media_dir) / "Запись_и_связанные_логи.json"
        result = {
            "schema_version": "M8-recording-manifest-v1",
            "run_id": self.run_id,
            "scenario_id": run_manifest.get("scenario_id"),
            "scenario_seed": run_manifest.get("scenario_seed"),
            "controller_final_state": run_manifest.get("controller_summary", {}).get("final_state"),
            "source_run_directory": run_manifest["run_directory"],
            "source_state_stream": _relative(self.state_stream_path),
            "source_state_stream_sha256": _sha256(self.state_stream_path),
            "source_run_manifest": {
                "path": run_manifest.get("pre_recording_manifest_path"),
                "sha256": run_manifest.get("pre_recording_manifest_sha256"),
            },
            "source_run_manifest_sha256": run_manifest.get("pre_recording_manifest_sha256"),
            "source_input_hashes_sha256": input_hashes,
            "source_logs": _source_log_hashes(run_manifest),
            "spectator_cameras": {
                "main": {"type": "free camera", "lookat_world_m": [0.0, 0.045, 0.19], "distance": 1.27,
                         "azimuth_deg": 135.0, "elevation_deg": -39.0},
                "top": {"type": "free camera", "lookat_world_m": [0.0, 0.045, 0.0], "distance": 1.22,
                        "azimuth_deg": 90.0, "elevation_deg": -89.5},
                "close": {"type": "free camera tracking the rendered TCP site", "distance": 0.45,
                          "azimuth_deg": 45.0, "elevation_deg": -75.0},
                "independent_of_sensor_views": True,
            },
            "video": {
                "path": _relative(out_path),
                "sha256": _sha256(out_path),
                "container": "MP4",
                "codec_fourcc": codec,
                "codec_note": (
                    "H.264 avc1 was attempted first but unavailable in the installed OpenCV build; "
                    "verified MPEG-4 Part 2 mp4v fallback was used."
                    if codec == "mp4v" else "H.264 avc1 encoder available."
                ),
                "fps": self.config.fps,
                "resolution_px": [self.config.width, self.config.height],
                "playback_speed_nominal": self.config.playback_speed,
                "playback_speed_effective": ((end_s - start_s) / video_duration_s
                                              if video_duration_s > 0 else 0.0),
                "source_simulation_time_start_s": start_s,
                "source_simulation_time_end_s": end_s,
                "source_simulation_duration_s": end_s - start_s,
                "encoded_video_duration_s": video_duration_s,
                "frame_count": expected_frames,
                "decoded_frame_count": decoded["decoded_frame_count"],
                "dropped_frames": 0,
                "frame_schedule": "Fixed 0.2 s simulation sampling at 25 fps / 5x; exact terminal state is appended if off-grid.",
                "decode_verification": decoded,
            },
            "keyframes": keyframe_records,
            "ground_truth_separation": (
                "HUD fields are derived only from public controller state/track records. "
                "Evaluator ground truth is not passed to the controller or HUD. The rendered "
                "viewer uses the recorded physical state solely to draw the same run."
            ),
            "created_from_recorded_run_states": True,
            "physical_prototype_or_physical_test": False,
        }
        result["video"]["dropped_frames"] = self._dropped_frames
        result["recording_manifest"] = _relative(manifest_path)
        manifest_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result
