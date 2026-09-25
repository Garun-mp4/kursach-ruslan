from __future__ import annotations

from dataclasses import dataclass

from .types import SlotSnapshot, SlotStatus


class ReservationError(RuntimeError):
    pass


@dataclass
class _SlotRecord:
    class_label: str
    status: SlotStatus = SlotStatus.FREE
    owner_track_id: int | None = None
    reason_code: str | None = None


class ReservationManager:
    """Controller-owned object and placement-slot reservations, with no truth access."""

    def __init__(self, slot_classes: dict[str, str], initial_occupied: tuple[str, ...] = ()):
        self._slots = {slot_id: _SlotRecord(color) for slot_id, color in slot_classes.items()}
        self._track_owner: int | None = None
        for slot_id in initial_occupied:
            if slot_id not in self._slots:
                raise ValueError(f"Unknown initial occupied slot {slot_id!r}")
            self._slots[slot_id].status = SlotStatus.OCCUPIED
            self._slots[slot_id].reason_code = "INITIAL_PUBLIC_OCCUPANCY"

    @property
    def reserved_track_id(self) -> int | None:
        return self._track_owner

    @property
    def reserved_slot_id(self) -> str | None:
        owned = [key for key, slot in self._slots.items() if slot.status == SlotStatus.RESERVED]
        if len(owned) > 1:
            raise ReservationError("More than one placement slot is reserved")
        return owned[0] if owned else None

    def free_slots(self, class_label: str) -> tuple[str, ...]:
        return tuple(sorted(
            slot_id for slot_id, slot in self._slots.items()
            if slot.class_label == class_label and slot.status == SlotStatus.FREE
        ))

    def reserve(self, track_id: int, slot_id: str) -> None:
        if self._track_owner is not None:
            raise ReservationError(f"Track {self._track_owner} is already reserved")
        if slot_id not in self._slots:
            raise ReservationError(f"Unknown slot {slot_id!r}")
        slot = self._slots[slot_id]
        if slot.status != SlotStatus.FREE:
            raise ReservationError(f"Slot {slot_id} is {slot.status.value}, not FREE")
        slot.status = SlotStatus.RESERVED
        slot.owner_track_id = track_id
        slot.reason_code = None
        self._track_owner = track_id
        self.assert_invariants()

    def release_track(self, track_id: int) -> None:
        if self._track_owner is not None and self._track_owner != track_id:
            raise ReservationError("Attempt to release a track owned by another cycle")
        self._track_owner = None

    def release_slot(self, slot_id: str, track_id: int, reason_code: str) -> None:
        slot = self._slots[slot_id]
        if slot.status != SlotStatus.RESERVED or slot.owner_track_id != track_id:
            raise ReservationError(f"Slot {slot_id} reservation owner mismatch")
        slot.status = SlotStatus.FREE
        slot.owner_track_id = None
        slot.reason_code = reason_code

    def occupy(self, slot_id: str, track_id: int, reason_code: str) -> None:
        slot = self._slots[slot_id]
        if slot.status != SlotStatus.RESERVED or slot.owner_track_id != track_id:
            raise ReservationError(f"Cannot occupy unreserved slot {slot_id}")
        slot.status = SlotStatus.OCCUPIED
        slot.owner_track_id = None
        slot.reason_code = reason_code

    def complete_cycle(self, track_id: int, slot_id: str, reason_code: str) -> None:
        self.occupy(slot_id, track_id, reason_code)
        self.release_track(track_id)
        self.assert_invariants()

    def quarantine(self, slot_id: str, track_id: int, reason_code: str) -> None:
        slot = self._slots[slot_id]
        if slot.status == SlotStatus.RESERVED and slot.owner_track_id != track_id:
            raise ReservationError(f"Cannot quarantine another cycle's slot {slot_id}")
        if slot.status not in {SlotStatus.RESERVED, SlotStatus.FREE}:
            raise ReservationError(f"Cannot quarantine slot {slot_id} from {slot.status.value}")
        slot.status = SlotStatus.QUARANTINED
        slot.owner_track_id = None
        slot.reason_code = reason_code

    def release_cycle(self, track_id: int, slot_id: str, reason_code: str) -> None:
        self.release_slot(slot_id, track_id, reason_code)
        self.release_track(track_id)

    def quarantine_cycle(self, track_id: int, slot_id: str, reason_code: str) -> None:
        self.quarantine(slot_id, track_id, reason_code)
        self.release_track(track_id)

    def reset_reserved(self, reason_code: str) -> None:
        for slot in self._slots.values():
            if slot.status == SlotStatus.RESERVED:
                slot.status = SlotStatus.QUARANTINED
                slot.owner_track_id = None
                slot.reason_code = reason_code
        self._track_owner = None

    def snapshots(self) -> tuple[SlotSnapshot, ...]:
        return tuple(SlotSnapshot(slot_id, slot.class_label, slot.status,
                                  slot.owner_track_id, slot.reason_code)
                     for slot_id, slot in sorted(self._slots.items()))

    def status(self, slot_id: str) -> SlotStatus:
        return self._slots[slot_id].status

    def assert_invariants(self) -> None:
        reserved = [(slot_id, slot) for slot_id, slot in self._slots.items()
                    if slot.status == SlotStatus.RESERVED]
        if len(reserved) > 1:
            raise ReservationError("At most one slot may be reserved by the single-arm controller")
        if bool(reserved) != (self._track_owner is not None):
            raise ReservationError("Track and slot reservations must be acquired/released together")
        if reserved and reserved[0][1].owner_track_id != self._track_owner:
            raise ReservationError("Reserved track and slot owners disagree")
        for slot in self._slots.values():
            if slot.status != SlotStatus.RESERVED and slot.owner_track_id is not None:
                raise ReservationError("Only RESERVED slots may name an owner")
