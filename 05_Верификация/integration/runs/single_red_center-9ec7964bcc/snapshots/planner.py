from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import hashlib
import json
import math
import re
import time

import numpy as np

from .collision import CollisionScene
from .context import PlanningContext, load_context
from .trajectory import build_trajectory
from .records import (CartesianWaypoint, ClearanceSample, Phase, Plan, PlanCode,
                    PlanEvent, PlanResult)
from kinematics.scara import (
    IKStatus, Pose, check_joint_limits, forward_kinematics, inverse_kinematics,
    periodic_difference,
)


@dataclass(frozen=True)
class _Slot:
    slot_id: str
    color: str
    index: int
    xy: tuple[float, float]


class _Reject(Exception):
    def __init__(self, code: PlanCode, reason: str, phase: str | None = None, **diagnostics):
        super().__init__(reason)
        self.code, self.reason, self.phase = code, reason, phase
        self.diagnostics = diagnostics


class Planner:
    """Deterministic, fail-closed full-cycle planner using M3 IK and M4 detections."""

    def __init__(self, context: PlanningContext | None = None,
                 collision_scene: CollisionScene | None = None):
        self.context = context or load_context()
        self.collision = collision_scene or CollisionScene(self.context)

    @staticmethod
    def _reject_result(reject: _Reject, attempts=()) -> PlanResult:
        return PlanResult(reject.code, reject.reason, reject.phase, None,
                          {**reject.diagnostics, "attempts": list(attempts)})

    def _validate_inputs(self, batch, target_track_id: int, now: float,
                         q: np.ndarray, gripper: float):
        if batch is None or getattr(batch, "status", None) != "OK":
            raise _Reject(PlanCode.INVALID_BATCH, "M4 batch is not OK; no movement is planned.")
        if not isinstance(target_track_id, int) or isinstance(target_track_id, bool):
            raise _Reject(PlanCode.INVALID_INPUT, "target_track_id must be an integer M4 track ID.")
        if not math.isfinite(now) or not math.isfinite(float(batch.simulation_time_s)):
            raise _Reject(PlanCode.INVALID_INPUT, "Planning and M4 timestamps must be finite.")
        max_age = float(self.context.p("planning.maximum_detection_age_s"))
        age = now - float(batch.simulation_time_s)
        if age < -1e-9 or age > max_age:
            raise _Reject(PlanCode.STALE_DETECTION, "M4 batch is future-dated or older than the M4 frame-age limit.",
                          age_s=age, maximum_age_s=max_age)
        if q.shape != (4,) or not np.all(np.isfinite(q)) or not check_joint_limits(q, self.context.arm):
            raise _Reject(PlanCode.INVALID_INPUT, "current_q must be four finite, in-range M3 arm coordinates.")
        if not math.isfinite(gripper):
            raise _Reject(PlanCode.INVALID_INPUT, "Current gripper coordinate must be finite.")
        open_q = float(self.context.p("robot.finger_home_m"))
        if abs(gripper-open_q) > float(self.context.p("planning.distance_numeric_tolerance_m")):
            raise _Reject(PlanCode.INVALID_INPUT, "M5 starts only from a verified open-gripper, unheld configuration.",
                          current_gripper_m=gripper, required_open_m=open_q)
        tracks = tuple(batch.detections)
        if len(tracks) > len(self.collision._free_bodies):
            raise _Reject(PlanCode.SCENE_CAPACITY_EXCEEDED, "Perception tracks exceed M2 collision proxy capacity.",
                          count=len(tracks), capacity=len(self.collision._free_bodies))
        ids, target = set(), None
        for d in tracks:
            tid = getattr(d, "track_id", None)
            if not isinstance(tid, int) or isinstance(tid, bool) or tid in ids:
                raise _Reject(PlanCode.SCENE_UNCERTAIN, "All obstacle tracks must have unique integer M4 track IDs.")
            ids.add(tid)
            status = getattr(d, "status", None)
            class_label = getattr(d, "class_label", None)
            geometry_only_unknown = status == "UNKNOWN" and class_label == "UNKNOWN"
            if status != "VALID" and not geometry_only_unknown:
                raise _Reject(PlanCode.SCENE_UNCERTAIN, "An invalid/occluded M4 track makes the obstacle scene incomplete.",
                              track_id=tid, status=getattr(d, "status", None),
                              perception_reason=getattr(d, "reason", None))
            if status == "VALID" and class_label not in {"RED", "GREEN", "BLUE"}:
                raise _Reject(PlanCode.SCENE_UNCERTAIN,
                              "A color-VALID M4 track has an unsupported class label.",
                              track_id=tid, class_label=class_label)
            try:
                xy = np.asarray(d.xy_base_m, dtype=float)
                sigma_xy, sigma_yaw = float(d.position_sigma_m), float(d.yaw_sigma_rad)
                stamp, confidence = float(d.simulation_time_s), float(d.confidence)
            except (AttributeError, TypeError, ValueError):
                raise _Reject(PlanCode.SCENE_UNCERTAIN, "An M4 track has malformed geometry or quality data.", track_id=tid)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not all(
                math.isfinite(v) for v in (sigma_xy, sigma_yaw, stamp, confidence)
            ):
                raise _Reject(PlanCode.SCENE_UNCERTAIN, "An M4 track contains a non-finite field.", track_id=tid)
            if sigma_xy < 0 or sigma_yaw < 0 or not 0 <= confidence <= 1:
                raise _Reject(PlanCode.SCENE_UNCERTAIN, "An M4 track violates its uncertainty/confidence contract.", track_id=tid)
            track_age = now-stamp
            if track_age < -1e-9 or track_age > max_age:
                raise _Reject(PlanCode.STALE_DETECTION, "An individual M4 track is stale.", track_id=tid, age_s=track_age)
            yaw = getattr(d, "yaw_base_rad", None)
            if yaw is not None and not math.isfinite(float(yaw)):
                raise _Reject(PlanCode.SCENE_UNCERTAIN, "An obstacle track has non-finite yaw.", track_id=tid)
            if tid == target_track_id:
                target = d
        if target is None:
            raise _Reject(PlanCode.INVALID_INPUT, "Target track is not present in the M4 batch.", target_track_id=target_track_id)
        if target.class_label not in {"RED", "GREEN", "BLUE"}:
            raise _Reject(PlanCode.UNSUPPORTED_CLASS, "Only valid RED/GREEN/BLUE detections are pickable.",
                          target_class=target.class_label)
        if target.yaw_base_rad is None or not math.isfinite(float(target.yaw_base_rad)):
            raise _Reject(PlanCode.INVALID_INPUT, "Target detection has no finite estimated yaw.")
        if float(target.position_sigma_m) > float(self.context.p("planning.position_sigma_limit_m")):
            raise _Reject(PlanCode.SCENE_UNCERTAIN, "Target XY uncertainty exceeds the allocated grasp budget.",
                          sigma_m=target.position_sigma_m, limit_m=self.context.p("planning.position_sigma_limit_m"))
        if float(target.yaw_sigma_rad) > float(self.context.p("planning.yaw_sigma_limit_rad")):
            raise _Reject(PlanCode.SCENE_UNCERTAIN, "Target yaw uncertainty exceeds the M2 grasp tolerance.",
                          sigma_rad=target.yaw_sigma_rad, limit_rad=self.context.p("planning.yaw_sigma_limit_rad"))
        (xmin, xmax), (ymin, ymax) = self.collision._input_bounds
        x, y = map(float, target.xy_base_m)
        if not xmin <= x <= xmax or not ymin <= y <= ymax:
            raise _Reject(PlanCode.TARGET_OUT_OF_SCOPE, "Target pose is outside the M4 input region.", xy=[x, y])
        return target, tracks

    def _slots(self, available: Iterable[str] | None, color: str) -> tuple[_Slot, ...]:
        if available is None:
            raise _Reject(PlanCode.INVALID_INPUT, "An explicit M6-owned available-slot list is required.")
        offsets = tuple(map(float, self.context.p("cell.tray_slot_x_offsets_m")))
        yoff = float(self.context.p("cell.tray_slot_y_offset_m"))
        trays = self.context.p("cell.tray_centers_xy_m")
        slots = {}
        for value in available:
            if not isinstance(value, str):
                raise _Reject(PlanCode.SLOT_INVALID, "Slot IDs must use COLOR:index strings.")
            match = re.fullmatch(r"(RED|GREEN|BLUE):(\d+)", value)
            if not match:
                raise _Reject(PlanCode.SLOT_INVALID, f"Malformed slot ID {value!r}; expected COLOR:index.")
            slot_color, index_s = match.groups()
            index = int(index_s)
            if index >= len(offsets) or value in slots:
                raise _Reject(PlanCode.SLOT_INVALID, f"Unknown or duplicate slot ID {value!r}.")
            center = trays[slot_color]
            slots[value] = _Slot(value, slot_color, index,
                                 (float(center[0])+offsets[index], float(center[1])+yoff))
        result = [slot for slot in slots.values() if slot.color == color]
        if not result:
            raise _Reject(PlanCode.NO_AVAILABLE_SLOT, f"No free placement slot was supplied for {color}.")
        result.sort(key=lambda s: (abs(s.index-1), s.index))
        return tuple(result)

    def _current_branch(self, q: np.ndarray) -> str:
        threshold = float(self.context.p("planning.minimum_abs_sin_q2"))
        abs_sin = abs(math.sin(float(q[1])))
        if abs_sin <= threshold:
            raise _Reject(PlanCode.NEAR_SINGULARITY, "Current q is inside the singularity exclusion band.",
                          abs_sin_q2=abs_sin, threshold=threshold)
        return "ELBOW_POSITIVE" if q[1] > 0 else "ELBOW_NEGATIVE"

    def _solve(self, pose: Pose, q0: np.ndarray, branch: str):
        result = inverse_kinematics(pose, self.context.arm, q0, allow_object_symmetry=False)
        threshold = float(self.context.p("planning.minimum_abs_sin_q2"))
        candidates = [c for c in result.candidates if c.branch_id == branch and c.abs_sin_q2 > threshold]
        if not candidates:
            if result.candidates and branch not in {c.branch_id for c in result.candidates}:
                raise _Reject(PlanCode.BRANCH_DISCONTINUITY,
                              "Waypoint is solvable only on a different elbow branch; singular branch switching is prohibited.",
                              required_branch=branch, available_branches=sorted({c.branch_id for c in result.candidates}))
            if result.status == IKStatus.NEAR_SINGULARITY or (result.candidates and all(c.abs_sin_q2 <= threshold for c in result.candidates)):
                raise _Reject(PlanCode.NEAR_SINGULARITY, result.reason,
                              abs_sin_q2=[c.abs_sin_q2 for c in result.candidates])
            raise _Reject(PlanCode.IK_UNREACHABLE, result.reason or "No valid IK candidate.", status=result.status.value)
        spans = self.context.arm.characteristic_spans
        weight = float(self.context.p("planning.branch_singularity_weight"))
        scored = []
        for c in candidates:
            q = np.asarray(c.q)
            continuity = float(np.dot((q-q0)/spans, (q-q0)/spans))
            singularity = weight*(threshold/max(c.abs_sin_q2, threshold))**2
            scored.append((continuity+singularity, tuple(c.q), c))
        cost, _, selected = min(scored, key=lambda item: (item[0], item[1]))
        return np.asarray(selected.q, dtype=float), float(cost)

    @staticmethod
    def _interp(a: Pose, b: Pose, u: float) -> Pose:
        dyaw = periodic_difference(b.yaw_rad, a.yaw_rad)
        return Pose(a.x_m+(b.x_m-a.x_m)*u, a.y_m+(b.y_m-a.y_m)*u,
                    a.z_m+(b.z_m-a.z_m)*u, a.yaw_rad+dyaw*u)

    @staticmethod
    def _xyz_distance(a: Pose, b: Pose) -> float:
        return math.dist((a.x_m, a.y_m, a.z_m), (b.x_m, b.y_m, b.z_m))

    def _append_path(self, points: list[CartesianWaypoint], start: Pose, q_start: np.ndarray, end: Pose,
                     *, phase: str, branch: str, symmetry: int, gripper: float,
                     payload: str, deadline: float) -> np.ndarray:
        max_xyz = float(self.context.p("planning.maximum_cartesian_waypoint_step_m"))
        max_yaw = float(self.context.p("planning.transfer_arc_max_angle_rad"))
        max_joint = float(self.context.p("planning.max_normalized_joint_step"))
        spans = self.context.arm.characteristic_spans
        max_depth = int(self.context.p("planning.maximum_waypoint_refinements"))
        q_start = np.asarray(q_start, dtype=float)

        def solve_leg(a: Pose, qa: np.ndarray, b: Pose, depth: int):
            if time.perf_counter() > deadline:
                raise _Reject(PlanCode.TIMEOUT, "M5 planning wall-clock limit exceeded.", phase)
            qb, cost = self._solve(b, qa, branch)
            cart = self._xyz_distance(a, b)
            dyaw = abs(periodic_difference(b.yaw_rad, a.yaw_rad))
            dq_norm = float(np.linalg.norm((qb-qa)/spans))
            if cart <= max_xyz+1e-12 and dyaw <= max_yaw+1e-12 and dq_norm <= max_joint+1e-12:
                return [(b, qb, cost)]
            if depth >= max_depth:
                raise _Reject(PlanCode.BRANCH_DISCONTINUITY, "Adaptive Cartesian/joint subdivision exhausted.",
                              phase, cartesian_step_m=cart, yaw_step_rad=dyaw,
                              normalized_joint_step=dq_norm, allowed_joint_step=max_joint)
            mid = self._interp(a, b, 0.5)
            if self._xyz_distance(a, mid) < 1e-10 and abs(periodic_difference(mid.yaw_rad,a.yaw_rad)) < 1e-10:
                raise _Reject(PlanCode.BRANCH_DISCONTINUITY, "Waypoint subdivision stopped making progress.", phase)
            left = solve_leg(a, qa, mid, depth+1)
            pm, qm, _ = left[-1]
            return left + solve_leg(pm, qm, b, depth+1)

        xyz = self._xyz_distance(start, end)
        yaw = abs(periodic_difference(end.yaw_rad, start.yaw_rad))
        count = max(1, math.ceil(xyz/max_xyz), math.ceil(yaw/max_yaw))
        prev_pose, prev_q = start, q_start
        for i in range(1, count+1):
            goal = self._interp(start, end, i/count)
            solved = solve_leg(prev_pose, prev_q, goal, 0)
            for pose, q, _ in solved:
                points.append(CartesianWaypoint(phase, (pose.x_m,pose.y_m,pose.z_m,pose.yaw_rad),
                    tuple(map(float,q)), branch, symmetry, float(gripper), payload))
            prev_pose, prev_q, _ = solved[-1]
        return prev_q

    def _ring(self, a: tuple[float,float], b: tuple[float,float], z: float, yaw: float,
              direction: int = 0) -> tuple[Pose,...]:
        ra, rb = math.hypot(*a), math.hypot(*b)
        radius = max(ra, rb)+float(self.context.p("planning.transfer_ring_radial_clearance_m"))
        maximum = self.context.arm.l1_m+self.context.arm.l2_m-0.020
        if radius > maximum:
            raise _Reject(PlanCode.NO_DETERMINISTIC_ROUTE,
                          "Radial transfer ring exceeds the configured non-singular workspace.",
                          radius_m=radius, maximum_radius_m=maximum)
        ta, tb = math.atan2(a[1],a[0]), math.atan2(b[1],b[0])
        short_arc = (tb-ta+math.pi)%(2*math.pi)-math.pi
        if direction == 0:
            arc = short_arc
        elif direction == 1:
            arc = short_arc-math.copysign(2*math.pi, short_arc) if abs(short_arc)>1e-12 else 0.0
        else:
            raise ValueError("ring direction must be 0 (shortest) or 1 (opposite arc)")
        n = max(1, math.ceil(abs(arc)/float(self.context.p("planning.transfer_arc_max_angle_rad"))))
        xy=[a,(radius*math.cos(ta),radius*math.sin(ta))]
        xy.extend((radius*math.cos(ta+arc*i/n),radius*math.sin(ta+arc*i/n)) for i in range(1,n+1))
        xy.append(b)
        clean=[]
        for p in xy:
            if not clean or math.dist(clean[-1],p)>1e-9: clean.append(p)
        return tuple(Pose(x,y,z,yaw) for x,y in clean)

    def _route(self, points: list[CartesianWaypoint], route: Sequence[Pose], q: np.ndarray, *,
               final_phase: str, branch: str, symmetry: int, gripper: float,
               payload: str, deadline: float) -> np.ndarray:
        for i in range(1,len(route)):
            phase=final_phase if i==len(route)-1 else (
                Phase.PREGRASP_ROUTE.value if final_phase==Phase.PREGRASP.value else
                Phase.SAFE_RETURN.value if final_phase==Phase.SAFE_RETURN.value else Phase.TRANSFER.value)
            q=self._append_path(points,route[i-1],q,route[i],phase=phase,branch=branch,
                symmetry=symmetry,gripper=gripper,payload=payload,deadline=deadline)
        return q

    def _hold(self, yaw: float) -> float:
        sx,sy,_=map(float,self.context.p("object.size_xyz_m"))
        projected=sx*abs(math.cos(yaw))+sy*abs(math.sin(yaw))
        gap=2*(float(self.context.p("robot.finger_open_center_offset_m"))-
               float(self.context.p("robot.finger_thickness_m"))/2)
        qg=(gap-projected)/2
        lo,hi=map(float,self.context.p("robot.finger_range_m"))
        if not lo<=qg<=hi:
            raise _Reject(PlanCode.INVALID_INPUT,"Perceived object footprint does not fit the M2 gripper range.",
                          required_gripper_m=qg,range_m=[lo,hi])
        return qg

    def _candidate_waypoints(self, q0: np.ndarray, target, slot: _Slot, branch: str,
                             symmetry: int, yaw: float, route_directions: tuple[int,int,int],
                             deadline: float) -> list[CartesianWaypoint]:
        safe=float(self.context.p("planning.safe_transfer_tcp_z_m"))
        pick=float(self.context.p("robot.object_pick_center_z_m"))
        place=float(self.context.p("robot.object_place_center_z_m"))
        opened=float(self.context.p("robot.finger_home_m"))
        xy=tuple(map(float,target.xy_base_m))
        fk=forward_kinematics(q0,self.context.arm).pose
        start=Pose(fk.x_m,fk.y_m,fk.z_m,fk.yaw_rad)
        safe_start=Pose(start.x_m,start.y_m,safe,start.yaw_rad)
        pts=[CartesianWaypoint(Phase.INITIAL_APPROACH.value,(start.x_m,start.y_m,start.z_m,start.yaw_rad),
            tuple(map(float,q0)),branch,symmetry,opened,"unheld")]
        q=self._append_path(pts,start,q0,safe_start,phase=Phase.INITIAL_APPROACH.value,
            branch=branch,symmetry=symmetry,gripper=opened,payload="unheld",deadline=deadline)
        aligned=Pose(safe_start.x_m,safe_start.y_m,safe,yaw)
        q=self._append_path(pts,safe_start,q,aligned,phase=Phase.ORIENTATION_ALIGN.value,
            branch=branch,symmetry=symmetry,gripper=opened,payload="unheld",deadline=deadline)
        pre=Pose(xy[0],xy[1],safe,yaw)
        q=self._route(pts,self._ring((aligned.x_m,aligned.y_m),xy,safe,yaw,route_directions[0]),q,
            final_phase=Phase.PREGRASP.value,branch=branch,symmetry=symmetry,
            gripper=opened,payload="unheld",deadline=deadline)
        grasp=Pose(xy[0],xy[1],pick,yaw)
        q=self._append_path(pts,pre,q,grasp,phase=Phase.DESCENT.value,branch=branch,
            symmetry=symmetry,gripper=opened,payload="unheld",deadline=deadline)
        held=self._hold(yaw)
        pts.append(CartesianWaypoint(Phase.GRASP_CLOSE.value,(grasp.x_m,grasp.y_m,grasp.z_m,grasp.yaw_rad),
            tuple(map(float,q)),branch,symmetry,held,"unheld","GRIPPER_CLOSE_COMPLETE"))
        pts.append(CartesianWaypoint(Phase.LIFT.value,(grasp.x_m,grasp.y_m,grasp.z_m,grasp.yaw_rad),
            tuple(map(float,q)),branch,symmetry,held,"carried","OBJECT_ATTACHED"))
        lift=Pose(xy[0],xy[1],safe,yaw)
        q=self._append_path(pts,grasp,q,lift,phase=Phase.LIFT.value,branch=branch,
            symmetry=symmetry,gripper=held,payload="carried",deadline=deadline)
        place_xy=slot.xy
        preplace=Pose(place_xy[0],place_xy[1],safe,yaw)
        q=self._route(pts,self._ring(xy,place_xy,safe,yaw,route_directions[1]),q,final_phase=Phase.PREPLACE.value,
            branch=branch,symmetry=symmetry,gripper=held,payload="carried",deadline=deadline)
        place_pose=Pose(place_xy[0],place_xy[1],place,yaw)
        q=self._append_path(pts,preplace,q,place_pose,phase=Phase.PLACE_DESCENT.value,
            branch=branch,symmetry=symmetry,gripper=held,payload="carried",deadline=deadline)
        pts.append(CartesianWaypoint(Phase.RELEASE.value,(place_pose.x_m,place_pose.y_m,place_pose.z_m,place_pose.yaw_rad),
            tuple(map(float,q)),branch,symmetry,held,"carried","RELEASE_BEGIN"))
        pts.append(CartesianWaypoint(Phase.RELEASE.value,(place_pose.x_m,place_pose.y_m,place_pose.z_m,place_pose.yaw_rad),
            tuple(map(float,q)),branch,symmetry,opened,"placed","OBJECT_RELEASED"))
        retreat=Pose(place_xy[0],place_xy[1],safe,yaw)
        q=self._append_path(pts,place_pose,q,retreat,phase=Phase.RETREAT.value,branch=branch,
            symmetry=symmetry,gripper=opened,payload="placed",deadline=deadline)
        route=self._ring(place_xy,(start.x_m,start.y_m),safe,yaw,route_directions[2])
        q=self._route(pts,route,q,final_phase=Phase.SAFE_RETURN.value,branch=branch,
            symmetry=symmetry,gripper=opened,payload="placed",deadline=deadline)
        q=self._append_path(pts,Pose(start.x_m,start.y_m,safe,yaw),q,safe_start,
            phase=Phase.SAFE_RETURN.value,branch=branch,symmetry=symmetry,
            gripper=opened,payload="placed",deadline=deadline)
        self._append_path(pts,safe_start,q,start,phase=Phase.SAFE_HOME.value,branch=branch,
            symmetry=symmetry,gripper=opened,payload="placed",deadline=deadline)
        return pts

    def _vertical_errors(self, waypoints: Sequence[CartesianWaypoint]) -> tuple[float,float]:
        phases={Phase.DESCENT.value,Phase.LIFT.value,Phase.PLACE_DESCENT.value,
                Phase.RETREAT.value,Phase.SAFE_HOME.value}
        xy_error=yaw_error=0.0
        for a,b in zip(waypoints,waypoints[1:]):
            if b.phase not in phases: continue
            pa,pb=forward_kinematics(a.q,self.context.arm).pose,forward_kinematics(b.q,self.context.arm).pose
            xy_error=max(xy_error,math.hypot(pb.x_m-pa.x_m,pb.y_m-pa.y_m))
            yaw_error=max(yaw_error,abs(periodic_difference(pb.yaw_rad,pa.yaw_rad)))
        return xy_error,yaw_error

    def _verify(self, points, tracks, target_id, slot, deadline, step_override):
        xy_err,yaw_err=self._vertical_errors(points)
        xy_tol=float(self.context.p("planning.distance_numeric_tolerance_m"))
        yaw_tol=float(self.context.p("kinematics.yaw_acceptance_rad"))
        if xy_err>xy_tol or yaw_err>yaw_tol:
            raise _Reject(PlanCode.VERTICAL_PATH_ERROR,"Grasp/place/retreat is not vertical in TCP space.",
                          maximum_xy_error_m=xy_err,maximum_yaw_error_rad=yaw_err,
                          allowed_xy_error_m=xy_tol,allowed_yaw_error_rad=yaw_tol)
        samples,metrics=build_trajectory(points,self.context,sample_step_override_m=step_override)
        vratio,aratio=metrics["max_velocity_ratio"],metrics["max_acceleration_ratio"]
        if vratio>1+1e-10: raise _Reject(PlanCode.VELOCITY_LIMIT,"Sampled arm/gripper speed exceeds SSOT.",ratio=vratio)
        if aratio>1+1e-10: raise _Reject(PlanCode.ACCELERATION_LIMIT,"Sampled arm/gripper acceleration exceeds SSOT.",ratio=aratio)
        exact_min=lower_min=math.inf
        closest_phase=""
        clearance_profile=[]
        step=float(self.context.p("planning.path_sweep_sample_step_m") if step_override is None else step_override)
        for index,s in enumerate(samples):
            if time.perf_counter()>deadline:
                raise _Reject(PlanCode.TIMEOUT,"Full-path collision preflight exceeded M5 timeout.",
                              checked_samples=index,total_samples=len(samples))
            report=self.collision.check(s.q,gripper_m=s.gripper_m,detections=tracks,target_track_id=target_id,
                phase=s.phase,payload_mode=s.payload_mode,slot_color=slot.color,slot_xy=slot.xy,
                sample_step_bound_m=step)
            if report.minimum_clearance_m<exact_min:
                exact_min,closest_phase=report.minimum_clearance_m,s.phase
            lower_min=min(lower_min,report.clearance_lower_bound_m)
            clearance_profile.append(ClearanceSample(
                float(s.time_s),str(s.phase),str(s.payload_mode),
                float(report.minimum_clearance_m),float(report.clearance_lower_bound_m),
                int(report.checked_pairs),int(report.checked_near_pairs)))
            if not report.valid:
                v=report.violations[0]
                if index == 0: code=PlanCode.COLLISION_AT_START
                elif s.phase in {Phase.DESCENT.value,Phase.GRASP_CLOSE.value}: code=PlanCode.COLLISION_GRASP
                elif s.payload_mode in {"carried","placed"}: code=PlanCode.COLLISION_PAYLOAD
                else: code=PlanCode.COLLISION_PATH
                raise _Reject(code,"Full-cycle sampled collision check rejected a configuration.",s.phase,
                    sample_index=index,time_s=s.time_s,violation=v.__dict__,
                    violation_count=len(report.violations),checked_pairs=report.checked_pairs,
                    exact_near_pairs=report.checked_near_pairs)
        cap=float(self.context.p("planning.minimum_clearance_report_cap_m"))
        if not math.isfinite(exact_min): exact_min=cap
        if not math.isfinite(lower_min): lower_min=cap
        return samples,metrics,exact_min,lower_min,xy_err,yaw_err,closest_phase,tuple(clearance_profile)

    def plan_cycle(self, detections, target_track_id: int, current_q: Iterable[float],
                   current_gripper_m: float, available_slot_ids: Iterable[str] | None,
                   now_time_s: float, *, sample_step_override_m: float | None = None) -> PlanResult:
        deadline=time.perf_counter()+float(self.context.p("planning.plan_timeout_s"))
        attempts=[]
        try:
            q=np.asarray(tuple(current_q),dtype=float)
            target,tracks=self._validate_inputs(detections,target_track_id,float(now_time_s),q,float(current_gripper_m))
            slots=self._slots(available_slot_ids,str(target.class_label))
            branch=self._current_branch(q)
            pick=Pose(float(target.xy_base_m[0]),float(target.xy_base_m[1]),
                      float(self.context.p("robot.object_pick_center_z_m")),float(target.yaw_base_rad))
            all_ik=inverse_kinematics(pick,self.context.arm,q,allow_object_symmetry=True)
            threshold=float(self.context.p("planning.minimum_abs_sin_q2"))
            eligible=[]
            seen=set()
            for c in all_ik.candidates:
                if c.branch_id!=branch or c.abs_sin_q2<=threshold: continue
                key=(c.symmetry_index,float(c.resolved_target_yaw_rad))
                if key in seen: continue
                seen.add(key)
                d=(np.asarray(c.q)-q)/self.context.arm.characteristic_spans
                cost=float(np.dot(d,d))+float(self.context.p("planning.branch_singularity_weight"))*(threshold/c.abs_sin_q2)**2
                eligible.append((cost,c.symmetry_index,float(c.resolved_target_yaw_rad)))
            eligible.sort(key=lambda x:(x[0],x[1],x[2]))
            if not eligible:
                if all_ik.candidates:
                    raise _Reject(PlanCode.BRANCH_DISCONTINUITY,
                        "Reachable target solutions are on a disconnected elbow branch; planner will not cross a singularity.",
                        available_branches=sorted({c.branch_id for c in all_ik.candidates}),required_branch=branch)
                code=PlanCode.NEAR_SINGULARITY if all_ik.status==IKStatus.NEAR_SINGULARITY else PlanCode.IK_UNREACHABLE
                raise _Reject(code,all_ik.reason,status=all_ik.status.value)
            from itertools import product
            route_options=sorted(product((0,1),repeat=3),key=lambda dirs:(sum(dirs),dirs))
            for slot in slots:
                for _,symmetry,yaw in eligible:
                    for route_directions in route_options:
                        if time.perf_counter()>deadline:
                            raise _Reject(PlanCode.TIMEOUT,"M5 planning wall-clock limit exceeded.",
                                          attempts=len(attempts),timeout_s=self.context.p("planning.plan_timeout_s"))
                        try:
                            points=self._candidate_waypoints(q,target,slot,branch,symmetry,yaw,route_directions,deadline)
                            samples,metrics,clearance,lower,xyerr,yawerr,phase,clearance_profile=self._verify(
                                points,tracks,target_track_id,slot,deadline,sample_step_override_m)
                            events=tuple(PlanEvent(s.time_s,s.event,s.phase,target_track_id,slot.slot_id,
                                {"payload_mode":s.payload_mode,"gripper_m":s.gripper_m,"resolved_tool_yaw_rad":yaw})
                                for s in samples if s.event)
                            length=sum(math.dist(a.tcp_xyzyaw[:3],b.tcp_xyzyaw[:3]) for a,b in zip(samples,samples[1:]))
                            sample_step=float(self.context.p("planning.path_sweep_sample_step_m")
                                if sample_step_override_m is None else sample_step_override_m)
                            identity={"ssot":self.context.ssot_sha256,"model":self.context.model_sha256,
                                "frame":int(detections.frame_id),"target":target_track_id,"slot":slot.slot_id,
                                "branch":branch,"symmetry":symmetry,"yaw":yaw,"start_q":q.tolist(),
                                "target_xy":list(target.xy_base_m),"target_yaw":float(target.yaw_base_rad),
                                "detections":[{"track_id":int(d.track_id),"class":str(d.class_label),
                                    "status":str(d.status),"xy":list(map(float,d.xy_base_m)),
                                    "yaw":None if d.yaw_base_rad is None else float(d.yaw_base_rad),
                                    "sigma_xy":float(d.position_sigma_m),"sigma_yaw":float(d.yaw_sigma_rad),
                                    "stamp":float(d.simulation_time_s)} for d in sorted(tracks,key=lambda item:int(item.track_id))],
                                "ring_directions":route_directions,"sample_step_m":sample_step,
                                "method":"deterministic_ring_waypoints_v1"}
                            plan_id=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(",",":"),
                                allow_nan=False).encode()).hexdigest()[:20]
                            plan=Plan(plan_id,self.context.config_version,self.context.ssot_sha256,
                                self.context.model_sha256,int(detections.frame_id),float(detections.simulation_time_s),
                                target_track_id,str(target.class_label),slot.slot_id,branch,yaw,
                                tuple(dict.fromkeys(p.phase for p in points)),tuple(points),tuple(samples),events,
                                float(metrics["duration_s"]),float(length),float(clearance),float(lower),str(phase),
                                float(xyerr),float(yawerr),float(metrics["max_velocity_ratio"]),
                                float(metrics["max_acceleration_ratio"]),
                                "deterministic radial-ring transfer + vertical Cartesian contact segments",
                                float(sample_step),tuple(clearance_profile))
                            return PlanResult(PlanCode.SUCCESS,
                                "Full pick/place/return cycle passed IK, temporal and sampled collision preflight.",
                                None,plan,{"attempts":attempts,"sample_count":len(samples),
                                "maximum_swept_step_bound_m":metrics["max_swept_sample_bound_m"],
                                "collision_sample_step_m":sample_step,
                                "clearance_metric_note":"minimum_clearance_m is the nearest signed distance among narrow-phase pairs, capped by SSOT; clearance_lower_bound_m is the conservative bound over all checked pairs.",
                                "joint_metric":"Euclidean L2 over M3 characteristic joint spans",
                                "singularity_penalty_weight":self.context.p("planning.branch_singularity_weight"),
                                "ring_directions":route_directions,
                                "slot_reservation_owner":"M6; availability is caller supplied"})
                        except _Reject as reject:
                            attempts.append({"slot_id":slot.slot_id,"symmetry_index":symmetry,"yaw_rad":yaw,
                                "ring_directions":route_directions,"code":reject.code.value,"phase":reject.phase,
                                "reason":reject.reason,"diagnostics":reject.diagnostics})
                            if reject.code==PlanCode.TIMEOUT: raise
            if not attempts:
                raise _Reject(PlanCode.NO_AVAILABLE_SLOT,"No slot and yaw candidate could be evaluated.")
            preferred=(PlanCode.COLLISION_PAYLOAD,PlanCode.COLLISION_GRASP,PlanCode.COLLISION_PATH,
                PlanCode.NO_DETERMINISTIC_ROUTE,PlanCode.BRANCH_DISCONTINUITY,PlanCode.NEAR_SINGULARITY,
                PlanCode.IK_UNREACHABLE,PlanCode.SLOT_INVALID)
            failed=next((a for code in preferred for a in attempts if a["code"]==code.value),attempts[-1])
            raise _Reject(PlanCode(failed["code"]),failed["reason"],failed["phase"],
                          attempts=attempts,selected_diagnostics=failed["diagnostics"])
        except _Reject as reject:
            return self._reject_result(reject,attempts)
        except (ValueError,TypeError,OverflowError,KeyError) as exc:
            return self._reject_result(_Reject(PlanCode.INVALID_INPUT,str(exc)),attempts)

    def replan(self, previous_plan: Plan | None, detections, target_track_id: int,
               current_q: Iterable[float], current_gripper_m: float,
               available_slot_ids: Iterable[str] | None, now_time_s: float, **kwargs) -> PlanResult:
        """Discard and rebuild the complete cycle from fresh M4 tracks and current q."""
        available=tuple(available_slot_ids or ())
        preferred=([previous_plan.slot_id] if previous_plan and previous_plan.slot_id in available else [])
        preferred.extend(s for s in available if s not in preferred)
        return self.plan_cycle(detections,target_track_id,current_q,current_gripper_m,
                               preferred,now_time_s,**kwargs)


def plan_cycle(*args, **kwargs) -> PlanResult:
    return Planner().plan_cycle(*args, **kwargs)
