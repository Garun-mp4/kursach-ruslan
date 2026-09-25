from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROGRAM = Path(__file__).resolve().parents[2]
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from controller.config import ControllerConfig
from controller.controller import IllegalTransitionError, SortController
from controller.controller import ControllerContractError
from controller.fsm_spec import EXECUTION_PHASES, STATE_SPECS, TRANSITIONS
from controller.types import (
    ControllerInput,
    EvidenceStatus,
    ExecutionFeedback,
    ExecutionStatus,
    GraspEvidence,
    PlanResponse,
    PlacementEvidence,
    RecoveryFeedback,
    RecoveryStatus,
    State,
    SystemHealth,
    TrackStatus,
    SlotStatus,
)
from perception.types import Detection, DetectionBatch
from planning.records import (
    CartesianWaypoint,
    Phase,
    Plan,
    PlanCode,
    PlanEvent,
    PlanResult,
    TrajectorySample,
)


HEALTH = SystemHealth(dependencies_ready=True, arm_state_valid=True, gripper_state_known=True)
Q = (0.0, 0.9, 0.015, -0.25)
TCP = (0.0, -0.25)


def detection(track_id=7, color="RED", xy=(0.0, -0.25), *, status="VALID",
              confidence=0.95, stamp=0.01, frame_id=1, sigma_xy=0.001):
    return Detection(
        track_id=track_id,
        class_label=color,
        xy_base_m=xy,
        yaw_base_rad=0.0,
        confidence=confidence,
        position_sigma_m=sigma_xy,
        yaw_sigma_rad=math.radians(2),
        status=status,
        reason=None if status == "VALID" else "LOW_CLASS_SEPARATION",
        frame_id=frame_id,
        simulation_time_s=stamp,
        area_px=200.0,
        bbox_xywh_px=(100, 100, 15, 15),
        center_uv_px=(107.5, 107.5),
        frame_age_s=0.0,
    )


def batch(items=(), *, frame_id=1, stamp=0.01, status=None):
    items = tuple(items)
    if status is None:
        status = "OK" if items else "NO_CANDIDATES"
    return DetectionBatch(status, None, frame_id, stamp, items)


def inp(now, **kwargs):
    values = dict(
        simulation_time_s=now,
        health=HEALTH,
        current_q=Q,
        current_gripper_m=0.0,
        current_tcp_xy_m=TCP,
        grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
    )
    values.update(kwargs)
    return ControllerInput(**values)


def make_plan(config, source_batch, track_id=7, color="RED", slot_id="RED:0"):
    samples = []
    time_s = 0.0
    phase_names = tuple(phase.value for phase in Phase)
    for phase in Phase:
        samples.append(TrajectorySample(
            time_s=time_s, phase=phase.value, q=Q, dq=(0.0, 0.0, 0.0, 0.0),
            ddq=(0.0, 0.0, 0.0, 0.0), tcp_xyzyaw=(0.0, -0.25, 0.1, 0.0),
            branch_id="ELBOW_POSITIVE", gripper_m=0.0, payload_mode="unheld",
        ))
        time_s += 0.1
        samples.append(TrajectorySample(
            time_s=time_s, phase=phase.value, q=Q, dq=(0.0, 0.0, 0.0, 0.0),
            ddq=(0.0, 0.0, 0.0, 0.0), tcp_xyzyaw=(0.0, -0.25, 0.1, 0.0),
            branch_id="ELBOW_POSITIVE", gripper_m=0.0, payload_mode="unheld",
            event=("OBJECT_ATTACHED" if phase == Phase.LIFT else
                   "OBJECT_RELEASED" if phase == Phase.RELEASE else None),
        ))
        time_s += 0.1
    events = tuple(
        PlanEvent(sample.time_s, sample.event, sample.phase, track_id, slot_id, {})
        for sample in samples if sample.event is not None
    )
    return Plan(
        plan_id=f"controlled-{track_id}",
        ssot_version=config.config_version,
        ssot_sha256=config.ssot_sha256,
        model_sha256=config.model_sha256,
        perception_frame_id=source_batch.frame_id,
        perception_time_s=source_batch.simulation_time_s,
        target_track_id=track_id,
        target_class=color,
        slot_id=slot_id,
        selected_ik_branch="ELBOW_POSITIVE",
        resolved_tool_yaw_rad=0.0,
        phases=phase_names,
        waypoints=tuple(),
        samples=tuple(samples),
        events=events,
        duration_s=float(samples[-1].time_s),
        path_length_m=0.3,
        minimum_clearance_m=0.01,
        clearance_lower_bound_m=0.005,
        minimum_clearance_phase="TRANSFER",
        vertical_xy_error_max_m=0.0,
        vertical_yaw_error_max_rad=0.0,
        maximum_abs_velocity_ratio=0.5,
        maximum_abs_acceleration_ratio=0.5,
        planner_strategy="controlled test double; not a physical plan",
    )


def successful_result(config, source_batch, track_id=7, color="RED", slot_id="RED:0"):
    return PlanResult(
        PlanCode.SUCCESS, "Controlled complete plan fixture.", None,
        make_plan(config, source_batch, track_id, color, slot_id), {},
    )


def advance_to_plan(controller, *, detections=None, now=0.0):
    controller.step(inp(now))
    scene = batch(detections or (detection(),), frame_id=1, stamp=now + 0.01)
    controller.step(inp(now + 0.01, detection_batch=scene))
    output = controller.step(inp(now + 0.02))
    assert output.state == State.PLAN
    assert len(output.plan_requests) == 1
    return scene, output.plan_requests[0]


def drive_command(controller, current, *, holding, frame=None, placement=None):
    output = current
    now = max((event.simulation_time_s for event in current.events), default=0.0)
    if output.execution_commands:
        command_id = output.execution_commands[0].command_id
    else:
        command_id = controller.active_command_id
    if command_id is None:
        return output
    feedback = ExecutionFeedback(command_id, ExecutionStatus.COMPLETED, now + 0.05)
    output = controller.step(inp(
        now + 0.05,
        execution_feedback=feedback,
        grasp_evidence=GraspEvidence(holding, now + 0.05, "FINGER_CONTACT_SENSOR"),
        detection_batch=frame,
        placement_evidence=placement,
    ))
    return output


class ControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = ControllerConfig.load()

    def start_with_plan(self, *, run_id="test-run", detections=None):
        controller = SortController(self.config, run_id=run_id)
        scene, request = advance_to_plan(controller, detections=detections)
        response = PlanResponse(
            request.request_id,
            successful_result(self.config, scene, request.target_track_id,
                              request.target_class, request.reserved_slot_id),
        )
        output = controller.step(inp(0.03, plan_response=response))
        self.assertEqual(output.state, State.APPROACH)
        self.assertEqual(output.execution_commands[0].phase_names,
                         EXECUTION_PHASES[State.APPROACH])
        return controller, scene, request, output

    def run_through_to_transfer(self, controller, scene, output):
        self.assertEqual(output.state, State.APPROACH)
        for target_state, held in ((State.DESCEND, False), (State.GRASP, False),
                                   (State.VERIFY_HOLD, False)):
            output = drive_command(controller, output, holding=held)
            self.assertEqual(output.state, target_state)
        self.assertEqual(output.state, State.VERIFY_HOLD)
        # Complete the lift first. A fresh post-lift camera frame and sensor are separate evidence.
        output = drive_command(controller, output, holding=True)
        self.assertEqual(output.state, State.VERIFY_HOLD)
        post_lift = batch((), frame_id=controller._last_frame_id + 1,
                          stamp=output.events[-1].simulation_time_s)
        now = output.events[-1].simulation_time_s + 0.01
        output = controller.step(inp(
            now,
            grasp_evidence=GraspEvidence(True, now, "FINGER_CONTACT_SENSOR"),
            detection_batch=replace(post_lift, simulation_time_s=now),
        ))
        self.assertEqual(output.state, State.TRANSFER)
        return output

    def run_through_to_verify_place(self, controller, scene, output):
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s
        transfer_id = controller.active_command_id
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(transfer_id, ExecutionStatus.COMPLETED,
                                                 now + 0.05),
            grasp_evidence=GraspEvidence(True, now + 0.05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.PLACE)

        now = output.events[-1].simulation_time_s
        place_id = controller.active_command_id
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(place_id, ExecutionStatus.COMPLETED,
                                                 now + 0.05),
            grasp_evidence=GraspEvidence(True, now + 0.05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.RELEASE)

        now = output.events[-1].simulation_time_s
        release_id = controller.active_command_id
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(release_id, ExecutionStatus.COMPLETED,
                                                 now + 0.05),
            grasp_evidence=GraspEvidence(False, now + 0.05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.VERIFY_PLACE)
        return output

    def complete_nominal_cycle(self, controller, scene, output):
        track_id = controller._active_track_id
        class_label = controller._active_class
        slot_id = controller._active_slot_id
        self.assertIsNotNone(track_id)
        self.assertIsNotNone(class_label)
        self.assertIsNotNone(slot_id)
        output = self.run_through_to_verify_place(controller, scene, output)
        now = output.events[-1].simulation_time_s
        request = output.observation_requests[-1]
        now = output.events[-1].simulation_time_s + 0.01
        evidence = PlacementEvidence(
            EvidenceStatus.CONFIRMED, now, "PUBLIC_CAMERA_PERCEPTION",
            frame_id=request.minimum_frame_id,
            observed_zone_id=controller.config.class_to_zone[class_label],
            observed_slot_id=slot_id, observed_class=class_label, confidence=0.91,
        )
        output = controller.step(inp(now, placement_evidence=evidence))
        self.assertEqual(output.state, State.RETREAT)
        self.assertEqual(controller.reservations.status(slot_id), SlotStatus.OCCUPIED)
        self.assertEqual(controller.track_records[track_id].status,
                         TrackStatus.PLACED_CONTROLLER_CONFIRMED)
        retreat_id = controller.active_command_id
        now = output.events[-1].simulation_time_s
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(retreat_id, ExecutionStatus.COMPLETED, now + 0.05),
            grasp_evidence=GraspEvidence(False, now + 0.05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.OBSERVE)
        return output

    def test_nominal_cycle_requires_public_hold_and_place_evidence(self):
        controller, scene, request, output = self.start_with_plan()
        self.complete_nominal_cycle(controller, scene, output)
        event_types = [event.event_type for event in controller.event_log]
        self.assertIn("HOLD_CONFIRMED", event_types)
        self.assertIn("PLACEMENT_CONTROLLER_CONFIRMED", event_types)
        self.assertFalse(any(event.event_type == "SORT_SUCCESS" for event in controller.event_log))
        self.assertIsNone(controller._terminal_result)
        controller._assert_invariants()

    def test_init_fails_closed_without_fresh_empty_gripper_signal(self):
        controller = SortController(self.config)
        result = controller.step(inp(0.0, grasp_evidence=None))
        self.assertEqual(result.state, State.INIT)
        result = controller.step(inp(0.51, grasp_evidence=None))
        self.assertEqual(result.state, State.SAFE_STOP)
        self.assertTrue(result.safety_hold_requested)

    def test_unknown_is_never_mapped_to_a_sorting_zone(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        one = batch((detection(color="UNKNOWN", status="UNKNOWN"),), frame_id=1, stamp=0.01)
        self.assertEqual(controller.step(inp(0.01, detection_batch=one)).state, State.SELECT)
        self.assertEqual(controller.step(inp(0.02)).state, State.OBSERVE)
        two = batch((detection(color="UNKNOWN", status="UNKNOWN", frame_id=2, stamp=0.03),),
                    frame_id=2, stamp=0.03)
        self.assertEqual(controller.step(inp(0.03, detection_batch=two)).state, State.SELECT)
        out = controller.step(inp(0.04))
        self.assertEqual(out.state, State.DONE)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SKIPPED_UNKNOWN)
        self.assertIsNone(controller.reservations.reserved_slot_id)

    def test_valid_empty_scene_is_not_treated_as_camera_failure(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        for index in range(3):
            now = 0.01 + index * 0.02
            out = controller.step(inp(now, detection_batch=batch(
                (), frame_id=index + 1, stamp=now, status="NO_CANDIDATES"
            )))
            if index < 2:
                self.assertEqual(out.state, State.SELECT)
                self.assertEqual(controller._camera_failures, 0)
                controller.step(inp(now + 0.001))
            else:
                self.assertEqual(out.state, State.DONE)
        self.assertEqual(controller._terminal_result["status"], "DONE")

    def test_camera_failures_are_bounded_and_latch_safe_stop(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        for index in range(3):
            now = 0.01 + 0.02 * index
            bad = batch((), frame_id=index + 1, stamp=now, status="NO_FRAME")
            out = controller.step(inp(now, detection_batch=bad))
            if index < 2:
                self.assertEqual(out.state, State.OBSERVE)
            else:
                self.assertEqual(out.state, State.SAFE_STOP)
        self.assertEqual(controller._camera_failures, 3)

    def test_stale_frame_is_rejected_before_selection(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        stale = batch((detection(stamp=-1.0),), frame_id=1, stamp=-1.0)
        out = controller.step(inp(0.01, detection_batch=stale))
        self.assertEqual(out.state, State.OBSERVE)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertEqual(controller.event_log[-2].reason_code, "STALE_OR_FUTURE_FRAME")

    def test_same_m4_track_duplicates_collapse_before_reservation(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        item = detection()
        dup = replace(item, confidence=0.8)
        out = controller.step(inp(0.01, detection_batch=batch((item, dup))))
        self.assertEqual(out.state, State.SELECT)
        out = controller.step(inp(0.02))
        self.assertEqual(out.state, State.PLAN)
        self.assertEqual(len(out.plan_requests), 1)
        self.assertEqual(len(out.plan_requests[0].detections.detections), 1)
        self.assertTrue(any(event.event_type == "DUPLICATE_TRACK_COLLAPSED"
                            for event in controller.event_log))

    def test_conflicting_duplicate_track_fails_closed_instead_of_double_counting(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        red = detection(track_id=7, color="RED")
        blue = detection(track_id=7, color="BLUE")
        out = controller.step(inp(0.01, detection_batch=batch((red, blue))))
        self.assertEqual(out.state, State.OBSERVE)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertTrue(any(event.reason_code == "CONFLICTING_DUPLICATE_TRACK"
                            for event in controller.event_log))

    def test_permanent_planner_failure_releases_reservations_and_skips_track(self):
        controller = SortController(self.config)
        scene, request = advance_to_plan(controller)
        failed = PlanResult(PlanCode.IK_UNREACHABLE, "No valid IK.", "PREGRASP",
                            None, {"diagnostic": "controlled"})
        out = controller.step(inp(0.03, plan_response=PlanResponse(request.request_id, failed)))
        self.assertEqual(out.state, State.RECOVER)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SKIPPED_UNREACHABLE)
        self.assertEqual(controller.step(inp(0.04)).state, State.OBSERVE)

    def test_m5_plan_with_reordered_phases_is_rejected_before_motion(self):
        controller = SortController(self.config)
        scene, request = advance_to_plan(controller)
        result = successful_result(
            self.config, scene, request.target_track_id,
            request.target_class, request.reserved_slot_id,
        )
        invalid_plan = replace(result.plan, phases=tuple(reversed(result.plan.phases)))
        output = controller.step(inp(
            0.03,
            plan_response=PlanResponse(request.request_id, replace(result, plan=invalid_plan)),
        ))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"], "M5_PLAN_CONTRACT_MISMATCH")
        self.assertFalse(output.execution_commands)
        self.assertFalse(any(event.event_type == "PLAN_ACCEPTED" for event in controller.event_log))

    def test_m5_plan_without_release_marker_is_rejected_before_motion(self):
        controller = SortController(self.config)
        scene, request = advance_to_plan(controller)
        result = successful_result(
            self.config, scene, request.target_track_id,
            request.target_class, request.reserved_slot_id,
        )
        invalid_plan = replace(
            result.plan,
            events=tuple(event for event in result.plan.events
                         if event.event != "OBJECT_RELEASED"),
        )
        output = controller.step(inp(
            0.03,
            plan_response=PlanResponse(request.request_id, replace(result, plan=invalid_plan)),
        ))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"], "M5_PLAN_CONTRACT_MISMATCH")
        self.assertFalse(output.execution_commands)

    def test_transient_planner_failure_has_one_bounded_retry(self):
        controller = SortController(self.config)
        scene, request = advance_to_plan(controller)
        failed = PlanResult(PlanCode.TIMEOUT, "Planner timeout.", None, None, {})
        out = controller.step(inp(0.03, plan_response=PlanResponse(request.request_id, failed)))
        self.assertEqual(out.state, State.RECOVER)
        self.assertEqual(controller.track_records[7].plan_retries, 1)
        controller.step(inp(0.04))
        new = batch((detection(frame_id=2, stamp=0.05),), frame_id=2, stamp=0.05)
        controller.step(inp(0.05, detection_batch=new))
        out = controller.step(inp(0.06))
        self.assertEqual(out.state, State.PLAN)
        self.assertEqual(controller.track_records[7].attempts, 2)
        self.assertEqual(controller.reservations.reserved_slot_id, "RED:0")

    def test_all_slots_full_yields_safe_skip_without_planning_or_fake_placement(self):
        controller = SortController(
            self.config, initial_occupied_slots=("RED:0", "RED:1", "RED:2")
        )
        controller.step(inp(0.0))
        controller.step(inp(0.01, detection_batch=batch((detection(),))))
        out = controller.step(inp(0.02))
        self.assertEqual(out.state, State.DONE)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SKIPPED_FULL_ZONE)
        self.assertEqual(controller._terminal_result["controller_confirmed_placements"], 0)
        self.assertIsNone(controller.reservations.reserved_track_id)

    def test_unknown_obstacle_does_not_prevent_selecting_known_target(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        known = detection(track_id=7, color="RED", xy=(0.0, -0.25))
        unknown = detection(track_id=9, color="UNKNOWN", status="UNKNOWN",
                            xy=(0.1, -0.3))
        scene = batch((known, unknown), frame_id=1, stamp=0.01)
        controller.step(inp(0.01, detection_batch=scene))
        out = controller.step(inp(0.02))
        self.assertEqual(out.state, State.PLAN)
        self.assertEqual(out.plan_requests[0].target_track_id, 7)
        self.assertEqual(len(out.plan_requests[0].detections.detections), 2)

    def test_grasp_failure_uses_bounded_recovery_without_claiming_hold(self):
        controller, scene, request, output = self.start_with_plan()
        for _ in range(3):
            command_id = controller.active_command_id
            now = controller._event_clock_s + 0.05
            output = controller.step(inp(
                now,
                execution_feedback=ExecutionFeedback(
                    command_id, ExecutionStatus.COMPLETED, now
                ),
            ))
        self.assertEqual(output.state, State.VERIFY_HOLD)
        command_id = controller.active_command_id
        now = controller._event_clock_s + 0.05
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(command_id, ExecutionStatus.COMPLETED, now),
        ))
        self.assertEqual(output.state, State.VERIFY_HOLD)
        now += 0.01
        output = controller.step(inp(
            now, grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
            detection_batch=batch((detection(frame_id=2, stamp=now),),
                                  frame_id=2, stamp=now),
        ))
        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(output.recovery_requests[0].action, "OPEN_AND_RETREAT_NO_PAYLOAD")
        self.assertEqual(controller.track_records[7].status, TrackStatus.PENDING)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)
        self.assertIsNone(controller._terminal_result)

    def test_execution_failure_without_payload_releases_reservations(self):
        controller, scene, request, output = self.start_with_plan()
        command_id = controller.active_command_id
        now = controller._event_clock_s + 0.01
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(
                command_id, ExecutionStatus.FAILED, now, "CONTROLLED_EXECUTOR_FAULT"
            ),
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(controller.track_records[7].status, TrackStatus.PENDING)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)
        self.assertEqual(output.recovery_requests[0].action, "OPEN_AND_RETREAT_NO_PAYLOAD")

    def test_descend_execution_failure_with_fresh_no_hold_enters_recover(self):
        controller, scene, request, output = self.start_with_plan()
        output = drive_command(controller, output, holding=False)
        self.assertEqual(output.state, State.DESCEND)

        now = controller._event_clock_s + 0.01
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(
                controller.active_command_id, ExecutionStatus.FAILED,
                now, "CONTROLLED_DESCEND_FAULT",
            ),
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
        ))

        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(output.recovery_requests[0].action, "OPEN_AND_RETREAT_NO_PAYLOAD")
        self.assertEqual(controller.track_records[7].status, TrackStatus.PENDING)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)

    def test_grasp_execution_failure_with_fresh_no_hold_enters_recover(self):
        controller, scene, request, output = self.start_with_plan()
        output = drive_command(controller, output, holding=False)
        output = drive_command(controller, output, holding=False)
        self.assertEqual(output.state, State.GRASP)

        now = controller._event_clock_s + 0.01
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(
                controller.active_command_id, ExecutionStatus.FAILED,
                now, "CONTROLLED_GRASP_FAULT",
            ),
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
        ))

        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(output.recovery_requests[0].action, "OPEN_AND_RETREAT_NO_PAYLOAD")
        self.assertEqual(controller.track_records[7].status, TrackStatus.PENDING)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)

    def test_verify_place_cycle_timeout_enters_recover_without_claiming_sort(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_verify_place(controller, scene, output)
        self.assertIsNone(controller._terminal_result)

        timeout_at = controller._cycle_deadline_s + 0.01
        output = controller.step(inp(
            timeout_at,
            grasp_evidence=GraspEvidence(False, timeout_at, "FINGER_CONTACT_SENSOR"),
        ))

        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(output.recovery_requests[0].action, "OPEN_AND_RETREAT_NO_PAYLOAD")
        self.assertEqual(controller.track_records[7].status, TrackStatus.SAFE_FAILURE)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)
        self.assertFalse(any(event.event_type == "PLACEMENT_CONTROLLER_CONFIRMED"
                            for event in controller.event_log))

    def test_retreat_execution_failure_recovers_without_reclassifying_placed_track(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_verify_place(controller, scene, output)
        class_label = controller._active_class
        slot_id = controller._active_slot_id
        track_id = controller._active_track_id
        observation_request = output.observation_requests[-1]
        now = output.events[-1].simulation_time_s + 0.01
        evidence = PlacementEvidence(
            EvidenceStatus.CONFIRMED, now, "PUBLIC_CAMERA_PERCEPTION",
            frame_id=observation_request.minimum_frame_id,
            observed_zone_id=controller.config.class_to_zone[class_label],
            observed_slot_id=slot_id, observed_class=class_label, confidence=0.91,
        )
        output = controller.step(inp(now, placement_evidence=evidence))
        self.assertEqual(output.state, State.RETREAT)
        self.assertEqual(controller.track_records[track_id].status,
                         TrackStatus.PLACED_CONTROLLER_CONFIRMED)
        self.assertEqual(controller.reservations.status(slot_id), SlotStatus.OCCUPIED)

        retreat_id = controller.active_command_id
        failed_at = output.events[-1].simulation_time_s + 0.01
        output = controller.step(inp(
            failed_at,
            execution_feedback=ExecutionFeedback(
                retreat_id, ExecutionStatus.FAILED, failed_at, "CONTROLLED_RETREAT_FAULT"
            ),
            grasp_evidence=GraspEvidence(False, failed_at, "FINGER_CONTACT_SENSOR"),
        ))

        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(output.recovery_requests[0].action, "OPEN_AND_RETREAT_NO_PAYLOAD")
        self.assertEqual(controller.track_records[track_id].status,
                         TrackStatus.PLACED_CONTROLLER_CONFIRMED)
        self.assertEqual(controller.reservations.status(slot_id), SlotStatus.OCCUPIED)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)

    def test_grasp_execution_failure_with_unknown_hold_latches_safe_stop(self):
        controller, scene, request, output = self.start_with_plan()
        output = drive_command(controller, output, holding=False)  # approach
        output = drive_command(controller, output, holding=False)  # descent
        self.assertEqual(output.state, State.GRASP)
        self.assertIsNone(controller._holding_state)
        now = controller._event_clock_s + 0.01
        output = controller.step(inp(
            now,
            grasp_evidence=None,
            execution_feedback=ExecutionFeedback(
                controller.active_command_id, ExecutionStatus.FAILED, now,
                "CONTROLLED_GRASP_EXECUTOR_FAULT",
            ),
        ))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"],
                         "EXECUTION_FAILED_WITH_POSSIBLE_PAYLOAD")
        self.assertEqual(controller.reservations.reserved_track_id, 7)
        self.assertEqual(controller.reservations.reserved_slot_id, "RED:0")
        self.assertTrue(output.safety_hold_requested)

    def test_emergency_stop_latches_from_every_active_state(self):
        for active_state in set(State) - {State.SAFE_STOP, State.DONE}:
            with self.subTest(state=active_state.value):
                controller = SortController(self.config)
                controller.state = active_state
                output = controller.step(inp(0.01, emergency_stop=True))
                self.assertEqual(output.state, State.SAFE_STOP)
                self.assertEqual(output.terminal_result["reason_code"], "EMERGENCY_STOP")
                self.assertTrue(output.safety_hold_requested)
                self.assertFalse(output.execution_commands)
                self.assertFalse(output.plan_requests)

    def test_safe_stop_reset_requires_explicit_safe_disposition(self):
        controller = SortController(self.config)
        stopped = controller.step(inp(0.0, emergency_stop=True))
        self.assertEqual(stopped.state, State.SAFE_STOP)
        still_stopped = controller.step(inp(
            0.01, operator_reset=True, payload_safely_resolved=False,
            grasp_evidence=GraspEvidence(False, 0.01, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(still_stopped.state, State.SAFE_STOP)
        restarted = controller.step(inp(
            0.02, operator_reset=True, payload_safely_resolved=True,
            grasp_evidence=GraspEvidence(False, 0.02, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(restarted.state, State.INIT)
        self.assertTrue(any(event.reason_code == "OPERATOR_RESET_AFTER_SAFE_DISPOSITION"
                            for event in restarted.events))

    def test_cycle_timeout_without_payload_releases_reservations(self):
        controller, scene, request, output = self.start_with_plan()
        now = controller._cycle_deadline_s + 0.01
        output = controller.step(inp(
            now, grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR")
        ))
        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SAFE_FAILURE)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)

    def test_problematic_candidate_does_not_starve_next_candidate(self):
        controller = SortController(self.config)
        scene = batch((
            detection(track_id=7, color="RED", xy=(0.0, -0.25)),
            detection(track_id=9, color="BLUE", xy=(0.08, -0.25)),
        ), frame_id=1, stamp=0.01)
        controller.step(inp(0.0))
        controller.step(inp(0.01, detection_batch=scene))
        output = controller.step(inp(0.02))
        self.assertEqual(output.state, State.PLAN)
        self.assertEqual(output.plan_requests[0].target_track_id, 7)
        failed = PlanResult(PlanCode.IK_UNREACHABLE, "Controlled unreachable target.",
                            "PREGRASP", None, {})
        output = controller.step(inp(
            0.03, plan_response=PlanResponse(output.plan_requests[0].request_id, failed)
        ))
        self.assertEqual(output.state, State.RECOVER)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SKIPPED_UNREACHABLE)
        self.assertIsNone(controller.reservations.reserved_slot_id)
        output = controller.step(inp(0.04))  # plan failure has no physical recovery action
        self.assertEqual(output.state, State.OBSERVE)
        fresh = batch((
            detection(track_id=7, color="RED", xy=(0.0, -0.25), frame_id=2, stamp=0.05),
            detection(track_id=9, color="BLUE", xy=(0.08, -0.25), frame_id=2, stamp=0.05),
        ), frame_id=2, stamp=0.05)
        controller.step(inp(0.05, detection_batch=fresh))
        output = controller.step(inp(0.06))
        self.assertEqual(output.state, State.PLAN)
        self.assertEqual(output.plan_requests[0].target_track_id, 9)

    def test_multiple_objects_are_processed_once_across_batch_cycles(self):
        objects = (
            detection(track_id=7, color="RED", xy=(0.0, -0.25)),
            detection(track_id=8, color="GREEN", xy=(0.08, -0.25)),
        )
        controller, scene, request, output = self.start_with_plan(detections=objects)
        self.assertEqual(request.target_track_id, 7)
        self.complete_nominal_cycle(controller, scene, output)
        now = controller._event_clock_s + 0.01
        next_batch = batch((
            detection(track_id=7, color="RED", xy=(0.18, 0.24), frame_id=3, stamp=now),
            detection(track_id=8, color="GREEN", xy=(0.08, -0.25), frame_id=3, stamp=now),
        ), frame_id=3, stamp=now)
        controller.step(inp(now, detection_batch=next_batch))
        output = controller.step(inp(now + 0.01))
        self.assertEqual(output.state, State.PLAN)
        self.assertEqual(output.plan_requests[0].target_track_id, 8)
        self.assertEqual(output.plan_requests[0].reserved_slot_id, "GREEN:0")
        second_plan = successful_result(
            self.config, next_batch, 8, "GREEN", "GREEN:0"
        )
        output = controller.step(inp(
            now + 0.02,
            plan_response=PlanResponse(output.plan_requests[0].request_id, second_plan),
        ))
        self.assertEqual(output.state, State.APPROACH)
        self.complete_nominal_cycle(controller, next_batch, output)
        self.assertEqual(controller.reservations.status("RED:0"), SlotStatus.OCCUPIED)
        self.assertEqual(controller.reservations.status("GREEN:0"), SlotStatus.OCCUPIED)
        self.assertEqual(controller.track_records[7].status,
                         TrackStatus.PLACED_CONTROLLER_CONFIRMED)
        self.assertEqual(controller.track_records[8].status,
                         TrackStatus.PLACED_CONTROLLER_CONFIRMED)

    def test_selection_uses_composite_score_not_only_nearest_distance(self):
        controller = SortController(self.config)
        near_but_uncertain = replace(
            detection(track_id=5, xy=(0.0, -0.25), confidence=0.60, sigma_xy=0.005),
            yaw_sigma_rad=0.15,
        )
        farther_but_clear = replace(
            detection(track_id=9, xy=(0.05, -0.25), confidence=0.99, sigma_xy=0.0002),
            yaw_sigma_rad=0.001,
        )
        _, request = advance_to_plan(
            controller, detections=(near_but_uncertain, farther_but_clear)
        )
        self.assertEqual(request.target_track_id, 9)

    def test_equal_candidate_scores_break_tie_by_lower_track_id(self):
        controller = SortController(self.config)
        higher_id = detection(track_id=18, xy=(0.02, -0.25))
        lower_id = detection(track_id=7, xy=(-0.02, -0.25))
        _, request = advance_to_plan(controller, detections=(higher_id, lower_id))
        self.assertEqual(request.target_track_id, 7)

    def test_recovery_budget_resets_after_completed_safe_recovery(self):
        controller, scene, request, output = self.start_with_plan()
        for _ in range(3):
            output = drive_command(controller, output, holding=False)
        output = drive_command(controller, output, holding=False)  # lift completion
        now = output.events[-1].simulation_time_s + 0.01
        post_lift = batch((), frame_id=2, stamp=now)
        output = controller.step(inp(
            now,
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
            detection_batch=post_lift,
        ))
        self.assertEqual(output.state, State.RECOVER)
        request_id = output.recovery_requests[0].request_id
        now += 0.01
        output = controller.step(inp(
            now,
            recovery_feedback=RecoveryFeedback(
                request_id, RecoveryStatus.COMPLETED, now, True, True, False
            ),
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.OBSERVE)
        self.assertEqual(controller._recovery_attempts, 0)

    def test_grasp_attempt_budget_stops_after_one_bounded_retry(self):
        controller, scene, request, output = self.start_with_plan()
        for attempt in range(2):
            for _ in range(4):
                output = drive_command(controller, output, holding=False)
            self.assertEqual(output.state, State.VERIFY_HOLD)
            now = output.events[-1].simulation_time_s + 0.01
            post_lift = batch((), frame_id=controller._last_frame_id + 1, stamp=now)
            output = controller.step(inp(
                now,
                grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
                detection_batch=post_lift,
            ))
            self.assertEqual(output.state, State.RECOVER)
            record = controller.track_records[7]
            self.assertEqual(record.grasp_attempts, attempt + 1)
            if attempt == 0:
                self.assertEqual(record.status, TrackStatus.PENDING)
            else:
                self.assertEqual(record.status, TrackStatus.SAFE_FAILURE)
            self.assertIsNone(controller.reservations.reserved_track_id)
            recovery_id = output.recovery_requests[0].request_id
            now += 0.01
            output = controller.step(inp(
                now,
                recovery_feedback=RecoveryFeedback(
                    recovery_id, RecoveryStatus.COMPLETED, now, True, True, False
                ),
                grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
            ))
            self.assertEqual(output.state, State.OBSERVE)
            if attempt == 0:
                now += 0.01
                retry_scene = batch((detection(frame_id=controller._last_frame_id + 1,
                                               stamp=now),),
                                    frame_id=controller._last_frame_id + 1, stamp=now)
                controller.step(inp(now, detection_batch=retry_scene))
                output = controller.step(inp(now + 0.01))
                self.assertEqual(output.state, State.PLAN)
                retry_request = output.plan_requests[0]
                output = controller.step(inp(
                    now + 0.02,
                    plan_response=PlanResponse(
                        retry_request.request_id,
                        successful_result(self.config, retry_scene, 7, "RED", "RED:0"),
                    ),
                ))
                self.assertEqual(output.state, State.APPROACH)
        now = controller._event_clock_s + 0.01
        final_scene = batch((detection(frame_id=controller._last_frame_id + 1,
                                       stamp=now),),
                            frame_id=controller._last_frame_id + 1, stamp=now)
        controller.step(inp(now, detection_batch=final_scene))
        output = controller.step(inp(now + 0.01))
        self.assertEqual(output.state, State.DONE)
        self.assertEqual(controller.track_records[7].grasp_attempts, 2)
        self.assertEqual(controller._terminal_result["safe_failures"], 1)
        self.assertIsNone(controller.reservations.reserved_track_id)

    def test_lost_payload_during_transfer_quarantines_slot_and_stops(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s + 0.01
        result = controller.step(inp(
            now, grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR")
        ))
        self.assertEqual(result.state, State.SAFE_STOP)
        self.assertTrue(result.safety_hold_requested)
        self.assertEqual(controller.reservations.status("RED:0"), SlotStatus.QUARANTINED)
        self.assertNotIn(State.RELEASE, [command.state for command in result.execution_commands])
        self.assertTrue(result.terminal_result["no_automatic_release"])

    def test_stale_hold_signal_during_transfer_latches_safe_stop(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = controller._event_clock_s + self.config.p("hold_evidence_max_age_s") + 0.01
        feedback = ExecutionFeedback(
            controller.active_command_id, ExecutionStatus.RUNNING, now
        )
        result = controller.step(inp(now, grasp_evidence=None, execution_feedback=feedback))
        self.assertEqual(result.state, State.SAFE_STOP)
        self.assertEqual(result.terminal_result["reason_code"],
                         "PUBLIC_HOLD_SIGNAL_STALE_OR_UNKNOWN")
        self.assertEqual(controller.reservations.status("RED:0"), SlotStatus.QUARANTINED)
        self.assertTrue(result.safety_hold_requested)

    def test_hold_verification_timeout_with_no_fresh_sensor_is_safe_stop(self):
        controller, scene, request, output = self.start_with_plan()
        for _ in range(3):
            output = drive_command(controller, output, holding=False)
        self.assertEqual(output.state, State.VERIFY_HOLD)
        command_id = controller.active_command_id
        now = controller._event_clock_s + 0.01
        output = controller.step(inp(
            now,
            grasp_evidence=None,
            execution_feedback=ExecutionFeedback(command_id, ExecutionStatus.COMPLETED, now),
        ))
        self.assertIsNone(controller._holding_state)
        timeout_time = now + self.config.p("hold_verification_timeout_s") + 0.01
        output = controller.step(inp(timeout_time, grasp_evidence=None))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"],
                         "HOLD_EVIDENCE_TIMEOUT_UNKNOWN")
        self.assertEqual(controller.reservations.reserved_track_id, 7)
        self.assertEqual(controller.reservations.reserved_slot_id, "RED:0")
        self.assertTrue(output.safety_hold_requested)

    def test_release_confirmation_timeout_preserves_slot_and_stops(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s
        transfer_id = controller.active_command_id
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(transfer_id, ExecutionStatus.COMPLETED,
                                                 now + 0.05),
            grasp_evidence=GraspEvidence(True, now + 0.05, "FINGER_CONTACT_SENSOR"),
        ))
        now = output.events[-1].simulation_time_s
        place_id = controller.active_command_id
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(place_id, ExecutionStatus.COMPLETED,
                                                 now + 0.05),
            grasp_evidence=GraspEvidence(True, now + 0.05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.RELEASE)
        now = output.events[-1].simulation_time_s
        release_id = controller.active_command_id
        output = controller.step(inp(
            now + 0.05,
            execution_feedback=ExecutionFeedback(release_id, ExecutionStatus.COMPLETED,
                                                 now + 0.05),
            grasp_evidence=None,
        ))
        self.assertEqual(output.state, State.RELEASE)
        timeout_time = now + 0.05 + self.config.p(
            "gripper_release_confirmation_timeout_s"
        ) + 0.01
        output = controller.step(inp(timeout_time, grasp_evidence=None))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"], "RELEASE_STATE_UNCERTAIN")
        self.assertEqual(controller.reservations.reserved_slot_id, "RED:0")
        self.assertTrue(output.safety_hold_requested)

    def test_contact_loss_during_release_is_evaluated_at_release_completion(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s
        transfer_id = controller.active_command_id
        now += 0.05
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(transfer_id, ExecutionStatus.COMPLETED,
                                                 now),
            grasp_evidence=GraspEvidence(True, now, "FINGER_CONTACT_SENSOR"),
        ))
        place_id = controller.active_command_id
        now += 0.05
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(place_id, ExecutionStatus.COMPLETED,
                                                 now),
            grasp_evidence=GraspEvidence(True, now, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.RELEASE)
        release_id = controller.active_command_id
        now += 0.01
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(release_id, ExecutionStatus.RUNNING, now),
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.RELEASE)
        self.assertFalse(output.safety_hold_requested)
        now += 0.04
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(release_id, ExecutionStatus.COMPLETED, now),
            grasp_evidence=GraspEvidence(False, now, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.VERIFY_PLACE)
        self.assertTrue(output.observation_requests)

    def test_recovery_failure_and_timeout_escalate_to_safe_stop(self):
        controller, scene, request, output = self.start_with_plan()
        command_id = controller.active_command_id
        now = controller._event_clock_s + 0.01
        output = controller.step(inp(
            now,
            execution_feedback=ExecutionFeedback(command_id, ExecutionStatus.FAILED,
                                                 now, "CONTROLLED_FAULT"),
        ))
        self.assertEqual(output.state, State.RECOVER)
        request_id = output.recovery_requests[0].request_id
        now += 0.01
        output = controller.step(inp(
            now,
            recovery_feedback=RecoveryFeedback(
                request_id, RecoveryStatus.FAILED, now, True, True, False,
                "RECOVERY_ACTION_FAILED",
            ),
        ))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"],
                         "RECOVERY_FAILED_OR_STATE_UNKNOWN")
        self.assertEqual(controller.track_records[7].status, TrackStatus.SAFE_FAILURE)

        timed = SortController(self.config)
        scene, request = advance_to_plan(timed)
        response = successful_result(self.config, scene, request.target_track_id,
                                     request.target_class, request.reserved_slot_id)
        timed.step(inp(0.03, plan_response=PlanResponse(request.request_id, response)))
        active_command = timed.active_command_id
        failed_at = 0.04
        result = timed.step(inp(
            failed_at,
            execution_feedback=ExecutionFeedback(active_command, ExecutionStatus.FAILED,
                                                 failed_at, "CONTROLLED_FAULT"),
        ))
        self.assertEqual(result.state, State.RECOVER)
        timeout_at = failed_at + self.config.p("recovery_timeout_s") + 0.01
        result = timed.step(inp(timeout_at))
        self.assertEqual(result.state, State.SAFE_STOP)
        self.assertEqual(result.terminal_result["reason_code"], "RECOVERY_TIMEOUT")
        self.assertEqual(timed.track_records[7].status, TrackStatus.SAFE_FAILURE)

    def test_safe_stop_holding_payload_preserves_slot_and_never_opens_gripper(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s + 0.01
        stopped = controller.step(inp(
            now, emergency_stop=True,
            grasp_evidence=GraspEvidence(True, now, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(stopped.state, State.SAFE_STOP)
        self.assertEqual(controller.reservations.reserved_slot_id, "RED:0")
        self.assertEqual(controller.reservations.reserved_track_id, 7)
        self.assertFalse(stopped.execution_commands)
        self.assertTrue(stopped.safety_hold_requested)

    def test_safe_stop_without_payload_releases_reservations(self):
        controller, scene, request, _ = self.start_with_plan()
        stopped = controller.step(inp(
            0.04, emergency_stop=True,
            grasp_evidence=GraspEvidence(False, 0.04, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(stopped.state, State.SAFE_STOP)
        self.assertIsNone(controller.reservations.reserved_track_id)
        self.assertIsNone(controller.reservations.reserved_slot_id)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SAFE_FAILURE)

    def test_evaluator_only_placement_evidence_is_ignored(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s
        transfer_id = controller.active_command_id
        output = controller.step(inp(
            now + .05,
            execution_feedback=ExecutionFeedback(transfer_id, ExecutionStatus.COMPLETED, now + .05),
            grasp_evidence=GraspEvidence(True, now + .05, "FINGER_CONTACT_SENSOR"),
        ))
        # Get the valid controller to VERIFY_PLACE through its remaining logical stages.
        now = output.events[-1].simulation_time_s
        place_id = controller.active_command_id
        output = controller.step(inp(
            now + .05,
            execution_feedback=ExecutionFeedback(place_id, ExecutionStatus.COMPLETED, now + .05),
            grasp_evidence=GraspEvidence(True, now + .05, "FINGER_CONTACT_SENSOR"),
        ))
        now = output.events[-1].simulation_time_s
        release_id = controller.active_command_id
        output = controller.step(inp(
            now + .05,
            execution_feedback=ExecutionFeedback(release_id, ExecutionStatus.COMPLETED, now + .05),
            grasp_evidence=GraspEvidence(False, now + .05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.VERIFY_PLACE)
        req = output.observation_requests[-1]
        now += .06
        evidence = PlacementEvidence(
            EvidenceStatus.CONFIRMED, now, "EVALUATOR_GROUND_TRUTH",
            frame_id=req.minimum_frame_id, observed_zone_id="RED",
            observed_slot_id="RED:0", observed_class="RED", confidence=1.0,
        )
        output = controller.step(inp(now, placement_evidence=evidence))
        self.assertEqual(output.state, State.VERIFY_PLACE)
        self.assertEqual(controller.reservations.status("RED:0"), SlotStatus.RESERVED)
        self.assertFalse(any(event.event_type == "PLACEMENT_CONTROLLER_CONFIRMED"
                            for event in controller.event_log))

    def test_place_timeout_quarantines_instead_of_counting_success(self):
        controller, scene, request, output = self.start_with_plan()
        output = self.run_through_to_transfer(controller, scene, output)
        now = output.events[-1].simulation_time_s
        transfer_id = controller.active_command_id
        output = controller.step(inp(
            now + .05, execution_feedback=ExecutionFeedback(
                transfer_id, ExecutionStatus.COMPLETED, now + .05),
            grasp_evidence=GraspEvidence(True, now + .05, "FINGER_CONTACT_SENSOR"),
        ))
        now = output.events[-1].simulation_time_s
        place_id = controller.active_command_id
        output = controller.step(inp(
            now + .05, execution_feedback=ExecutionFeedback(
                place_id, ExecutionStatus.COMPLETED, now + .05),
            grasp_evidence=GraspEvidence(True, now + .05, "FINGER_CONTACT_SENSOR"),
        ))
        now = output.events[-1].simulation_time_s
        release_id = controller.active_command_id
        output = controller.step(inp(
            now + .05, execution_feedback=ExecutionFeedback(
                release_id, ExecutionStatus.COMPLETED, now + .05),
            grasp_evidence=GraspEvidence(False, now + .05, "FINGER_CONTACT_SENSOR"),
        ))
        self.assertEqual(output.state, State.VERIFY_PLACE)
        out = controller.step(inp(now + .6))
        self.assertEqual(out.state, State.RETREAT)
        self.assertEqual(controller.reservations.status("RED:0"), SlotStatus.QUARANTINED)
        self.assertEqual(controller.track_records[7].status, TrackStatus.SAFE_FAILURE)

    def test_simulation_clock_regression_latches_stop(self):
        controller = SortController(self.config)
        controller.step(inp(1.0))
        output = controller.step(inp(0.9))
        self.assertEqual(output.state, State.SAFE_STOP)
        self.assertEqual(output.terminal_result["reason_code"], "SIMULATION_TIME_REGRESSION")

    def test_forbidden_transition_is_detected(self):
        controller = SortController(self.config)
        with self.assertRaises(IllegalTransitionError):
            controller._transition(State.RELEASE, "TEST_ILLEGAL")

    def test_transition_spec_covers_every_state_and_is_explicit(self):
        self.assertEqual({spec.state for spec in STATE_SPECS}, set(State))
        self.assertEqual(set(TRANSITIONS), set(State))
        for source, targets in TRANSITIONS.items():
            for target in targets:
                self.assertNotEqual(source, target)
                self.assertTrue(target in TRANSITIONS[source])

    def test_transition_guard_covers_all_declared_and_forbidden_edges(self):
        controller = SortController(self.config)
        with patch.object(controller, "_enter_state", lambda _target: None):
            for source, allowed_targets in TRANSITIONS.items():
                for target in allowed_targets:
                    controller.state = source
                    controller._transition(target, "GUARD_COVERAGE")
                    self.assertEqual(controller.state, target)
                for target in set(State) - set(allowed_targets):
                    controller.state = source
                    with self.assertRaises(IllegalTransitionError):
                        controller._transition(target, "GUARD_NEGATIVE_COVERAGE")

    def test_transfer_invariant_requires_fresh_public_hold(self):
        controller = SortController(self.config)
        controller.state = State.TRANSFER
        controller._holding_state = False
        with self.assertRaises(ControllerContractError):
            controller._assert_invariants()

    def test_done_means_controller_terminated_not_all_items_sorted(self):
        controller = SortController(self.config)
        controller.step(inp(0.0))
        for frame_id in range(1, 4):
            now = frame_id * 0.02
            controller.step(inp(now, detection_batch=batch(
                (), frame_id=frame_id, stamp=now, status="NO_CANDIDATES"
            )))
            if frame_id < 3:
                controller.step(inp(now + 0.001))
        self.assertEqual(controller.state, State.DONE)
        self.assertEqual(controller._terminal_result["controller_confirmed_placements"], 0)
        self.assertIsNone(controller._terminal_result["evaluator_result"])


if __name__ == "__main__":
    unittest.main()
