from __future__ import annotations

from dataclasses import dataclass
import math

from controller.types import EvidenceStatus, PlacementEvidence
from perception.types import Detection, DetectionBatch

from .config import RuntimeConfig


@dataclass(frozen=True)
class _Slot:
    slot_id: str
    color: str
    xy: tuple[float, float]


class PublicPlacementVerifier:
    """Confirms a newly visible placement from M4 RGB estimates, not MuJoCo truth."""

    def __init__(self, ssot: dict, config: RuntimeConfig):
        self.config = config
        parameters = ssot["parameters"]
        offsets = parameters["cell.tray_slot_x_offsets_m"]["value"]
        y_offset = float(parameters["cell.tray_slot_y_offset_m"]["value"])
        trays = parameters["cell.tray_centers_xy_m"]["value"]
        self.slots = tuple(
            _Slot(f"{color}:{index}", color,
                  (float(center[0]) + float(offsets[index]), float(center[1]) + y_offset))
            for color in ("RED", "GREEN", "BLUE")
            for index in range(len(offsets))
            for center in (trays[color],)
        )
        self._previous_tracks: dict[int, Detection] = {}

    def observe(self, batch: DetectionBatch) -> None:
        self._previous_tracks = {
            int(detection.track_id): detection
            for detection in batch.detections
            if detection.track_id is not None
        }

    def initial_occupied_slots(self, batch: DetectionBatch) -> tuple[str, ...]:
        """Infer initial bin occupancy from public RGB detections only."""
        if batch.status not in {"OK", "NO_CANDIDATES"}:
            raise ValueError(f"Cannot initialize slot occupancy from frame status {batch.status}")
        occupied: dict[str, int] = {}
        for detection in batch.detections:
            if detection.status != "VALID" or detection.class_label not in {"RED", "GREEN", "BLUE"}:
                continue
            distances = sorted(
                (math.dist(detection.xy_base_m, slot.xy), slot)
                for slot in self.slots
            )
            if not distances or distances[0][0] > self.config.placement_slot_acceptance_radius_m:
                continue
            if len(distances) > 1 and abs(distances[1][0] - distances[0][0]) < 0.001:
                continue
            slot = distances[0][1]
            if slot.slot_id in occupied:
                raise ValueError(f"Multiple public detections map to occupied slot {slot.slot_id}")
            occupied[slot.slot_id] = int(detection.track_id or -1)
        return tuple(sorted(occupied))

    def evidence(self, batch: DetectionBatch) -> PlacementEvidence:
        if batch.status not in {"OK", "NO_CANDIDATES"}:
            return PlacementEvidence(
                EvidenceStatus.AMBIGUOUS, batch.simulation_time_s,
                "PUBLIC_CAMERA_PERCEPTION", batch.frame_id,
                reason_code=f"PLACEMENT_FRAME_{batch.status}",
            )
        new_tracks = [
            detection for detection in batch.detections
            if detection.track_id is not None and int(detection.track_id) not in self._previous_tracks
        ]
        candidates: list[tuple[float, Detection, _Slot]] = []
        ambiguous = False
        for detection in new_tracks:
            distances = sorted(
                ((math.dist(detection.xy_base_m, slot.xy), slot) for slot in self.slots),
                key=lambda item: item[0],
            )
            if not distances or distances[0][0] > self.config.placement_slot_acceptance_radius_m:
                continue
            if len(distances) > 1 and abs(distances[1][0] - distances[0][0]) < 0.001:
                ambiguous = True
                continue
            if detection.status != "VALID" or detection.class_label not in {"RED", "GREEN", "BLUE"}:
                ambiguous = True
                continue
            candidates.append((distances[0][0], detection, distances[0][1]))
        self.observe(batch)
        if ambiguous or len(candidates) > 1:
            return PlacementEvidence(
                EvidenceStatus.AMBIGUOUS, batch.simulation_time_s,
                "PUBLIC_CAMERA_PERCEPTION", batch.frame_id,
                reason_code="NEW_SLOT_OBSERVATIONS_AMBIGUOUS",
            )
        if not candidates:
            return PlacementEvidence(
                EvidenceStatus.NOT_CONFIRMED, batch.simulation_time_s,
                "PUBLIC_CAMERA_PERCEPTION", batch.frame_id,
                reason_code="NO_NEW_OBJECT_DETECTED_IN_SORT_ZONES",
            )
        _, detection, slot = candidates[0]
        return PlacementEvidence(
            EvidenceStatus.CONFIRMED, batch.simulation_time_s,
            "PUBLIC_CAMERA_PERCEPTION", batch.frame_id,
            observed_zone_id=slot.color,
            observed_slot_id=slot.slot_id,
            observed_class=detection.class_label,
            confidence=float(detection.confidence),
            reason_code="NEW_RGB_TRACK_MATCHED_TO_SORT_SLOT",
        )
