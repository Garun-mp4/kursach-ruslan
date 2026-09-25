from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import math

import mujoco
import numpy as np

from .context import PlanningContext, MODEL_PATH
from .records import Phase
from kinematics.scara import check_joint_limits, forward_kinematics


@dataclass(frozen=True)
class CollisionViolation:
    category: str
    geom_a: str
    geom_b: str
    phase: str
    distance_m: float
    required_m: float
    reason: str


@dataclass(frozen=True)
class CollisionReport:
    valid: bool
    minimum_clearance_m: float
    clearance_lower_bound_m: float
    nearest_phase: str | None
    checked_pairs: int
    checked_near_pairs: int
    allowed_contacts: tuple[dict, ...]
    violations: tuple[CollisionViolation, ...]
    sample_step_bound_m: float


class CollisionScene:
    """M2 collision geometry; free-body proxy poses come only from M4 observations."""

    ROBOT_BODY_NAMES = ("base_static", "link1", "link2", "z_slide", "wrist", "finger_left", "finger_right")
    JOINT_NAMES = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist", "j5_finger_left", "j6_finger_right")

    def __init__(self, context: PlanningContext, model_path=MODEL_PATH):
        self.context = context
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._robot_bodies = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.ROBOT_BODY_NAMES
        }
        if any(i < 0 for i in self._robot_bodies.values()):
            raise ValueError("M2 model is missing a required robot body")
        self._robot_body_names = {i: name for name, i in self._robot_bodies.items()}
        self._joint_addr = {}
        for name in self.JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"M2 model is missing joint {name}")
            self._joint_addr[name] = int(self.model.jnt_qposadr[jid])

        self._free_bodies = []
        for body_id in range(1, self.model.nbody):
            count = int(self.model.body_jntnum[body_id])
            first = int(self.model.body_jntadr[body_id])
            if count == 1 and int(self.model.jnt_type[first]) == int(mujoco.mjtJoint.mjJNT_FREE):
                self._free_bodies.append((body_id, int(self.model.jnt_qposadr[first])))
        self._free_bodies.sort(key=lambda p: p[0])
        if not self._free_bodies:
            raise ValueError("M2 model contains no free-body object collision proxies")
        self._proxy_geoms = {
            body: tuple(g for g in range(self.model.ngeom)
                        if int(self.model.geom_bodyid[g]) == body
                        and int(self.model.geom_contype[g]) != 0
                        and int(self.model.geom_conaffinity[g]) != 0)
            for body, _ in self._free_bodies
        }
        if any(not geoms for geoms in self._proxy_geoms.values()):
            raise ValueError("A free-body object has no active collision geom")
        self._proxy_geom_ids = {g for group in self._proxy_geoms.values() for g in group}
        active = {g for g in range(self.model.ngeom)
                  if int(self.model.geom_contype[g]) != 0 and int(self.model.geom_conaffinity[g]) != 0}
        self._robot_geoms = tuple(g for g in active if int(self.model.geom_bodyid[g]) in self._robot_body_names)
        self._static_geoms = tuple(g for g in active if g not in self._robot_geoms and g not in self._proxy_geom_ids)
        if not self._robot_geoms or not self._static_geoms:
            raise ValueError("M2 collision geometry is incomplete")
        self._geom_names = {
            g: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
            for g in range(self.model.ngeom)
        }
        self._geom_body = np.asarray(self.model.geom_bodyid, dtype=int)
        self._local_half = np.zeros((self.model.ngeom, 3), dtype=float)
        for g in active:
            size = self.model.geom_size[g]
            kind = int(self.model.geom_type[g])
            if kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                self._local_half[g] = size[:3]
            elif kind == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                self._local_half[g] = (size[0], size[0], size[1])
            elif kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                self._local_half[g] = (size[0], size[0], size[0]+size[1])
            else:
                raise ValueError(f"Unsupported M2 collision geom type {kind}; define its conservative bound.")
        exclusions = {tuple(sorted((str(a), str(b)))) for a, b in context.p("collision.allowed_pairs")}
        self._self_pairs = tuple(
            (a, b) for i, a in enumerate(self._robot_geoms) for b in self._robot_geoms[i+1:]
            if self._geom_body[a] != self._geom_body[b]
            and tuple(sorted((self._body_name(a), self._body_name(b)))) not in exclusions
        )
        self._robot_static_pairs = tuple((a, b) for a in self._robot_geoms for b in self._static_geoms)
        self._input_bounds = context.ssot["perception"]["runtime_config"]["regions"]["input_xy_bounds_m"]

    def _body_name(self, geom: int) -> str:
        bid = int(self._geom_body[geom])
        if bid in self._robot_body_names:
            return self._robot_body_names[bid]
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"

    def _set_pose(self, address: int, xyz: Sequence[float], yaw: float) -> None:
        self.data.qpos[address:address+3] = xyz
        half = float(yaw)/2
        self.data.qpos[address+3:address+7] = (math.cos(half), 0.0, 0.0, math.sin(half))

    def _set_state(self, q: Iterable[float], gripper: float, detections: Sequence,
                   target_id: int, payload: str, slot_xy: tuple[float,float] | None):
        q = np.asarray(tuple(q), dtype=float)
        if q.shape != (4,) or not check_joint_limits(q, self.context.arm):
            raise ValueError("Arm q is malformed or outside M3 joint limits")
        lo, hi = map(float, self.context.p("robot.finger_range_m"))
        if not math.isfinite(gripper) or not lo <= gripper <= hi:
            raise ValueError("Gripper coordinate violates its physical range")
        if payload not in {"unheld", "carried", "placed"}:
            raise ValueError(f"Unknown payload mode {payload}")
        mujoco.mj_resetData(self.model, self.data)
        for name, value in zip(self.JOINT_NAMES[:4], q):
            self.data.qpos[self._joint_addr[name]] = value
        for name in self.JOINT_NAMES[4:]:
            self.data.qpos[self._joint_addr[name]] = gripper
        mujoco.mj_forward(self.model, self.data)
        # Reset has no authority over object poses: overwrite every free proxy before use.
        for body, address in self._free_bodies:
            self._set_pose(address, (0.0, 0.0, 50.0+body), 0.0)
        tracks = sorted(detections, key=lambda d: int(d.track_id))
        if len(tracks) > len(self._free_bodies):
            raise OverflowError("M4 track count exceeds the configured object proxy capacity")
        object_z = float(self.context.p("object.position_z_m"))
        tcp = forward_kinematics(q, self.context.arm).pose
        proxies, target_proxy = {}, None
        for (body, address), detection in zip(self._free_bodies, tracks):
            track_id = int(detection.track_id)
            if payload == "carried" and track_id == target_id:
                xyz, yaw = (tcp.x_m, tcp.y_m, tcp.z_m), tcp.yaw_rad
            elif payload == "placed" and track_id == target_id:
                if slot_xy is None:
                    raise ValueError("Placed payload requires a reserved slot coordinate")
                xyz = (slot_xy[0], slot_xy[1], float(self.context.p("robot.object_place_center_z_m")))
                yaw = float(detection.yaw_base_rad or 0.0)
            else:
                xy = tuple(map(float, detection.xy_base_m))
                xyz = (xy[0], xy[1], object_z)
                yaw = float(detection.yaw_base_rad or 0.0)
            self._set_pose(address, xyz, yaw)
            proxies[track_id] = body
            if track_id == target_id:
                target_proxy = body
        if target_proxy is None:
            raise ValueError("Target has no M4-derived collision proxy")
        mujoco.mj_forward(self.model, self.data)
        return proxies, target_proxy

    def _aabbs(self):
        center = self.data.geom_xpos
        rotation = self.data.geom_xmat.reshape(self.model.ngeom, 3, 3)
        half = np.einsum("nij,nj->ni", np.abs(rotation), self._local_half)
        return center-half, center+half

    @staticmethod
    def _pair_aabb_gaps(lo: np.ndarray, hi: np.ndarray, pairs: np.ndarray) -> np.ndarray:
        if not len(pairs):
            return np.empty(0, dtype=float)
        a, b = pairs[:, 0], pairs[:, 1]
        gap = np.maximum(0.0, np.maximum(lo[a]-hi[b], lo[b]-hi[a]))
        return np.linalg.norm(gap, axis=1)

    def check(self, q: Iterable[float], *, gripper_m: float, detections: Sequence,
              target_track_id: int, phase: str, payload_mode: str = "unheld",
              slot_color: str | None = None, slot_xy: tuple[float,float] | None = None,
              sample_step_bound_m: float | None = None) -> CollisionReport:
        proxies,target_proxy=self._set_state(q,gripper_m,detections,target_track_id,payload_mode,slot_xy)
        step=float(sample_step_bound_m if sample_step_bound_m is not None
                   else self.context.p("planning.path_sweep_sample_step_m"))
        numeric=float(self.context.p("planning.distance_numeric_tolerance_m"))
        static_req=float(self.context.p("planning.static_clearance_m"))+step+numeric
        self_req=float(self.context.p("planning.self_collision_clearance_m"))+step+numeric
        obstacle_req=float(self.context.p("planning.perceived_object_clearance_m"))
        grasp_req=float(self.context.p("planning.grasp_entry_clearance_m"))
        grasp_support_req=(float(self.context.p("planning.grasp_finger_support_clearance_m"))
                           +step+numeric)
        penetration=numeric
        phase=str(phase)
        phase_contact=set(self.context.p("planning.allowed_gripper_object_contact_phases"))
        phase_support=set(self.context.p("planning.allowed_support_contact_phases"))
        placed_support_phases={Phase.PLACE_DESCENT.value,Phase.RELEASE.value,
                               Phase.RETREAT.value,Phase.SAFE_RETURN.value}
        table_geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"table_top_collision")
        base_geom=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,"base_foot_collision")
        tray_floor=(mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,
                    f"tray_{str(slot_color or '').lower()}_floor_collision")
                    if slot_color in {"RED","GREEN","BLUE"} else -1)
        allowed: dict[tuple[int,int],str]={}
        def mark(a,b,rule):
            allowed[(min(a,b),max(a,b))]=rule
        if base_geom>=0 and table_geom>=0:
            mark(base_geom,table_geom,"fixed_base_mount_to_table")
        # Only selected target fingers may contact the target, and only at the named phases.
        target_geoms=next((self._proxy_geoms[b] for t,b in proxies.items() if t==target_track_id),())
        finger_bodies={self._robot_bodies["finger_left"],self._robot_bodies["finger_right"]}
        if phase in phase_contact:
            for a in self._robot_geoms:
                if int(self._geom_body[a]) in finger_bodies:
                    for b in target_geoms: mark(a,b,"phase_scoped_finger_selected_object_contact")
        target_detection=next((d for d in detections if int(d.track_id)==target_track_id),None)
        tcp=forward_kinematics(q,self.context.arm).pose
        grasp_support_alignment = (
            ((payload_mode == "unheld"
              and phase in {Phase.DESCENT.value, Phase.GRASP_CLOSE.value})
             or (payload_mode == "carried" and phase == Phase.LIFT.value))
            and target_detection is not None
            and math.dist((tcp.x_m, tcp.y_m), tuple(map(float, target_detection.xy_base_m)))
                <= float(self.context.p("planning.position_sigma_limit_m"))
        )
        place_support_alignment = (
            (((payload_mode == "carried"
               and phase in {Phase.PLACE_DESCENT.value, Phase.RELEASE.value})
             or (payload_mode == "placed"
                 and phase in {Phase.RELEASE.value, Phase.RETREAT.value}))
             and target_detection is not None
             and str(target_detection.class_label) == str(slot_color)
             and slot_xy is not None
             and math.dist((tcp.x_m, tcp.y_m), tuple(map(float, slot_xy)))
                 <= float(self.context.p("planning.position_sigma_limit_m")))
        )
        if payload_mode=="carried" and phase==Phase.LIFT.value and table_geom>=0 and target_detection is not None:
            target_xy=tuple(map(float,target_detection.xy_base_m))
            if math.dist((tcp.x_m,tcp.y_m),target_xy)<=penetration:
                for a in target_geoms: mark(a,table_geom,"vertical_pick_lift_from_table_support")
        if (payload_mode=="carried" and phase in {Phase.PLACE_DESCENT.value,Phase.RELEASE.value}
                and tray_floor>=0 and slot_xy is not None
                and math.dist((tcp.x_m,tcp.y_m),slot_xy)<=penetration):
            for a in target_geoms: mark(a,tray_floor,"payload_supported_by_reserved_tray_floor")
            if table_geom>=0:
                for a in target_geoms: mark(a,table_geom,"payload_clearance_above_table_under_reserved_tray")
        if payload_mode=="placed" and table_geom>=0:
            for a in target_geoms: mark(a,table_geom,"payload_clearance_above_table_under_reserved_tray")
        if payload_mode=="placed" and tray_floor>=0:
            for a in target_geoms: mark(a,tray_floor,"selected_payload_supported_by_reserved_tray_floor")

        specs=[]  # (pairs, per-pair required distance, category)
        if self._self_pairs:
            specs.append((self._self_pairs,np.full(len(self._self_pairs),self_req),"self_collision"))
        if self._robot_static_pairs:
            required=np.full(len(self._robot_static_pairs),static_req)
            if grasp_support_alignment and table_geom >= 0:
                for index,(robot_geom,static_geom) in enumerate(self._robot_static_pairs):
                    if (int(self._geom_body[robot_geom]) in finger_bodies
                            and int(static_geom) == table_geom):
                        required[index]=grasp_support_req
            if place_support_alignment and tray_floor >= 0:
                for index,(robot_geom,static_geom) in enumerate(self._robot_static_pairs):
                    if int(self._geom_body[robot_geom]) in finger_bodies:
                        if int(static_geom) == tray_floor:
                            required[index]=grasp_support_req
                        elif int(static_geom) == table_geom:
                            # The selected tray floor is physically between these
                            # fingers and the tabletop; its own clearance remains
                            # checked above. Keep table penetration detection while
                            # avoiding a contradictory reserve through the tray.
                            required[index]=0.0
            specs.append((self._robot_static_pairs,required,"robot_environment"))

        robot_object=[]
        for track_id,body in proxies.items():
            for rg in self._robot_geoms:
                for og in self._proxy_geoms[body]:
                    required=obstacle_req
                    if track_id==target_track_id:
                        if payload_mode=="unheld":
                            required=grasp_req if int(self._geom_body[rg]) in finger_bodies and phase==Phase.DESCENT.value else static_req
                        elif payload_mode=="carried":
                            required=self_req
                        else:
                            required=static_req
                    robot_object.append((rg,og,required,"robot_object"))
        if robot_object:
            specs.append((tuple((a,b) for a,b,_,_ in robot_object),
                          np.asarray([c for _,_,c,_ in robot_object]),"robot_object"))

        if payload_mode in {"carried","placed"}:
            payload_pairs=tuple((og,sg) for og in target_geoms for sg in self._static_geoms)
            required=obstacle_req if payload_mode=="carried" else static_req
            if payload_pairs: specs.append((payload_pairs,np.full(len(payload_pairs),required),"payload_environment"))
            obstacle_tracks=[(t,b) for t,b in proxies.items() if t!=target_track_id]
            pair_list=[(a,b) for a in target_geoms for _,body in obstacle_tracks
                       for b in self._proxy_geoms[body]]
            if pair_list: specs.append((tuple(pair_list),np.full(len(pair_list),obstacle_req),"payload_obstacle"))
        obj_pairs=[]
        track_items=list(proxies.items())
        for i,(_,body_a) in enumerate(track_items):
            for _,body_b in track_items[i+1:]:
                obj_pairs.extend((a,b) for a in self._proxy_geoms[body_a] for b in self._proxy_geoms[body_b])
        if obj_pairs: specs.append((tuple(obj_pairs),np.full(len(obj_pairs),obstacle_req),"object_object"))

        lo,hi=self._aabbs()
        violations=[]; contacts=[]
        min_exact=math.inf; lower=math.inf; nearest=None; checked=near=0
        cap=float(self.context.p("planning.minimum_clearance_report_cap_m"))
        for pairs,required_array,category in specs:
            pairs=np.asarray(pairs,dtype=int).reshape((-1,2))
            gaps=self._pair_aabb_gaps(lo,hi,pairs)
            checked+=len(pairs)
            exact_indices=[]
            for i,(a,b) in enumerate(pairs):
                key=(min(int(a),int(b)),max(int(a),int(b)))
                rule=allowed.get(key)
                if rule is None and gaps[i] > required_array[i]:
                    # This pair is not sent to narrow phase. Its AABB gap is a
                    # conservative distance lower bound and must still feed the
                    # reported clearance bound.
                    lower=min(lower,float(gaps[i]))
                if rule is not None or gaps[i] <= required_array[i]:
                    exact_indices.append((i,rule))
            for index,rule in exact_indices:
                a,b=map(int,pairs[index])
                distance=float(mujoco.mj_geomDistance(self.model,self.data,a,b,cap,None))
                near+=1
                if rule is not None:
                    contacts.append({"geom_a":self._geom_names[a],"geom_b":self._geom_names[b],
                                     "rule":rule,"phase":phase,"distance_m":distance})
                    if distance < -penetration:
                        violations.append(CollisionViolation(category,self._geom_names[a],self._geom_names[b],
                            phase,distance,-penetration,"Permitted contact exceeds the signed-distance penetration tolerance."))
                    continue
                if distance<min_exact:
                    min_exact,nearest=distance,phase
                lower=min(lower,distance)
                req=float(required_array[index])
                if distance<req:
                    violations.append(CollisionViolation(category,self._geom_names[a],self._geom_names[b],
                        phase,distance,req,"Signed distance is below the required phase clearance."))
        if not math.isfinite(min_exact): min_exact=cap
        if not math.isfinite(lower): lower=cap
        return CollisionReport(not violations,float(min_exact),float(lower),nearest,checked,near,
                               tuple(contacts),tuple(violations),step)
