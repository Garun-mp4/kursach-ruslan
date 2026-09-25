from __future__ import annotations

from dataclasses import replace

import numpy as np

from .types import Detection


class GreedyTrackManager:
    """Deterministic nearest-neighbour tracker; IDs are assigned locally."""

    def __init__(self, max_distance_m: float = 0.055, max_missed_frames: int = 2) -> None:
        self.max_distance_m = float(max_distance_m)
        self.max_missed_frames = int(max_missed_frames)
        self._next_id = 1
        self._tracks: dict[int, tuple[Detection, int]] = {}

    def update(self, detections: tuple[Detection, ...]) -> tuple[Detection, ...]:
        candidates: list[tuple[float, int, int]] = []
        for track_id, (prior, missed) in self._tracks.items():
            if missed > self.max_missed_frames:
                continue
            for idx, current in enumerate(detections):
                delta = np.asarray(prior.xy_base_m) - np.asarray(current.xy_base_m)
                distance = float(np.linalg.norm(delta))
                color_compatible = prior.class_label == current.class_label or "UNKNOWN" in {
                    prior.class_label, current.class_label
                }
                if color_compatible and distance <= self.max_distance_m:
                    candidates.append((distance, track_id, idx))
        used_tracks: set[int] = set()
        used_detections: set[int] = set()
        assigned: dict[int, int] = {}
        for _, track_id, idx in sorted(candidates):
            if track_id not in used_tracks and idx not in used_detections:
                used_tracks.add(track_id)
                used_detections.add(idx)
                assigned[idx] = track_id
        result: list[Detection] = []
        next_tracks: dict[int, tuple[Detection, int]] = {}
        for idx, detection in enumerate(detections):
            track_id = assigned.get(idx)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
            updated = replace(detection, track_id=track_id)
            result.append(updated)
            next_tracks[track_id] = (updated, 0)
        for track_id, (prior, missed) in self._tracks.items():
            if track_id not in used_tracks and track_id not in next_tracks and missed < self.max_missed_frames:
                next_tracks[track_id] = (prior, missed + 1)
        self._tracks = next_tracks
        return tuple(result)

