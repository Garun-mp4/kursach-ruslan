"""Validate the M2 SSOT, generated MJCF, geometry and model-loading invariants."""
from __future__ import annotations

import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import build_model

ROOT = build_model.ROOT
SSOT_PATH = build_model.SSOT_PATH
MODEL_PATH = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
RESULTS = ROOT / "99_Рабочие_материалы" / "m2_validation" / "results"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_ssot_contract(config: dict) -> None:
    require(config.get("schema_version") == "1.0", "Unsupported or missing SSOT schema_version")
    require(config.get("config_version") in {"M2-v1.0", "M3-v1.0", "M4-v1.1", "M4-v1.2", "M4-v1.3", "M5-v1.0", "M5-v1.1", "M5-v1.2", "M5-v1.3", "M7-v1.0", "M7-v1.1", "M7-v1.2", "M7-v1.3", "M7-v1.4", "M7-v1.5", "M7-v1.6", "M7-v1.7", "M7-v1.8"}, "Unexpected configuration version")
    require(isinstance(config.get("project"), dict), "Missing project metadata mapping")
    require(isinstance(config.get("assemblies"), dict) and config["assemblies"], "Missing assembly definitions")
    require(isinstance(config.get("parameters"), dict) and config["parameters"], "Missing parameter registry")
    p = lambda key: build_model.parameter(config, key)

    required = (
        "environment.engine", "environment.engine_version", "environment.python_version",
        "environment.timestep_s", "environment.gravity_m_s2", "environment.density_pla_kg_m3",
        "environment.solver_iterations", "environment.solver_tolerance",
        "environment.impratio", "environment.noslip_iterations",
        "coordinates.world_definition", "coordinates.base_definition", "coordinates.transform_semantics",
        "robot.type", "robot.arm_dof", "robot.gripper_actuated_dof", "robot.link1_length_m",
        "robot.link2_length_m", "robot.j3_range_m", "robot.finger_range_m", "robot.touch_site_radius_m", "robot.j1_range_rad",
        "robot.j2_range_rad", "robot.j4_range_rad", "robot.tcp_in_wrist_m", "robot.finger_center_in_wrist_m", "object.size_xyz_m",
        "planning.static_clearance_m", "planning.distance_numeric_tolerance_m",
        "planning.grasp_finger_support_clearance_m",
        "object.color_class_labels", "cell.table_size_xy_m", "cell.tray_centers_xy_m",
        "cell.tray_slot_x_offsets_m", "camera.position_world_m", "camera.fovy_deg",
        "camera.resolution_px", "camera.safe_observation_joint_pose", "collision.allowed_pairs",
    )
    for key in required:
        require(key in config["parameters"], f"Required SSOT parameter is missing: {key}")

    if str(config.get("config_version", "")).startswith("M7-"):
        placement_required = (
            "camera.placement_name", "camera.placement_config_id",
            "camera.placement_position_world_m", "camera.placement_target_world_m",
            "camera.placement_fovy_deg", "camera.placement_resolution_px",
            "camera.placement_object_top_plane_z_m", "camera.placement_max_age_s",
            "camera.placement_nominal_object_area_px", "camera.placement_anchor_xy_bounds_m",
        )
        for key in placement_required:
            require(key in config["parameters"], f"Required M7 camera parameter is missing: {key}")
        expected_top_z = (
            float(p("cell.table_top_z_m"))
            + float(p("cell.tray_floor_thickness_m"))
            + float(p("object.size_xyz_m")[2])
        )
        require(math.isclose(float(p("camera.placement_object_top_plane_z_m")),
                              expected_top_z, rel_tol=0.0, abs_tol=1e-12),
                "M7 placement camera plane must match the top face of an object in a tray")
        for key in ("camera.placement_position_world_m", "camera.placement_target_world_m"):
            vector = np.asarray(p(key), dtype=float)
            require(vector.shape == (3,) and np.all(np.isfinite(vector)),
                    f"M7 placement camera parameter {key} must be a finite 3D vector")
        require(not np.allclose(p("camera.placement_position_world_m"),
                                p("camera.placement_target_world_m"), rtol=0, atol=1e-9),
                "M7 placement camera position and target must be distinct")
        require(1.0 < float(p("camera.placement_fovy_deg")) < 179.0,
                "M7 placement camera field of view is invalid")
        resolution = np.asarray(p("camera.placement_resolution_px"), dtype=int)
        require(resolution.shape == (2,) and np.all(resolution > 0),
                "M7 placement camera resolution must be positive width/height")

    for key in ("environment.timestep_s", "environment.gravity_m_s2", "environment.density_pla_kg_m3",
                "environment.solver_iterations", "environment.solver_tolerance", "environment.impratio",
                "robot.link1_length_m", "robot.link2_length_m", "robot.shoulder_z_m",
                "robot.finger_length_m", "robot.finger_thickness_m", "robot.finger_height_m",
                "robot.j3_velocity_m_s", "robot.j3_acceleration_m_s2", "robot.j3_effort_N",
                "object.mass_kg", "camera.fovy_deg"):
        require(float(p(key)) > 0, f"Parameter {key} must be positive")
    require(int(p("environment.noslip_iterations")) >= 0,
            "NoSlip iteration count cannot be negative")

    for key in ("cell.table_size_xy_m", "cell.input_size_xy_m", "cell.tray_outer_size_xy_m",
                "cell.tray_inner_size_xy_m", "object.size_xyz_m", "robot.finger_xyz_m",
                "robot.base_foot_outer_xyz_m", "robot.base_column_outer_xyz_m"):
        values = np.asarray(p(key), dtype=float)
        require(values.ndim == 1 and values.size >= 2 and np.all(np.isfinite(values)) and np.all(values > 0),
                f"Dimensions in {key} must be finite and positive")

    finger_xyz = np.asarray(p("robot.finger_xyz_m"), dtype=float)
    finger_dimensions = np.asarray((p("robot.finger_length_m"), p("robot.finger_thickness_m"),
                                    p("robot.finger_height_m")), dtype=float)
    require(finger_xyz.shape == (3,) and np.allclose(finger_xyz, finger_dimensions, rtol=0, atol=1e-12),
            "Finger geometry vector must match its length, thickness, and height parameters")

    for key in ("robot.j1_range_rad", "robot.j2_range_rad", "robot.j3_range_m", "robot.j4_range_rad", "robot.finger_range_m"):
        low, high = [float(value) for value in p(key)]
        require(math.isfinite(low) and math.isfinite(high) and low < high, f"Invalid joint interval {key}: [{low}, {high}]")

    home = p("camera.safe_observation_joint_pose")
    for joint, interval_key in (("j1_shoulder", "robot.j1_range_rad"), ("j2_elbow", "robot.j2_range_rad"),
                                ("j3_lift", "robot.j3_range_m"), ("j4_wrist", "robot.j4_range_rad")):
        low, high = [float(value) for value in p(interval_key)]
        require(joint in home and low <= float(home[joint]) <= high, f"Home coordinate {joint} is outside {interval_key}")
    finger_low, finger_high = [float(value) for value in p("robot.finger_range_m")]
    finger_center = np.asarray(p("robot.finger_center_in_wrist_m"), dtype=float)
    tcp_offset = np.asarray(p("robot.tcp_in_wrist_m"), dtype=float)
    require(finger_center.shape == (3,) and tcp_offset.shape == (3,),
            "Finger center and TCP offsets must be 3D vectors")
    require(np.allclose(finger_center, tcp_offset, rtol=0, atol=1e-12),
            "Gripper contact-pad center must coincide with the modeled object-center TCP")
    touch_radius = float(p("robot.touch_site_radius_m"))
    object_contact_half_diagonal = math.hypot(
        float(p("object.size_xyz_m")[0]) / 2.0,
        float(p("object.size_xyz_m")[2]) / 2.0,
    )
    require(touch_radius > object_contact_half_diagonal,
            "Touch sensor zone does not cover the modeled object-finger contact face")
    grasp_table_clearance = float(p("planning.grasp_finger_support_clearance_m"))
    static_clearance = float(p("planning.static_clearance_m"))
    numeric_tolerance = float(p("planning.distance_numeric_tolerance_m"))
    require(math.isfinite(grasp_table_clearance) and grasp_table_clearance >= numeric_tolerance,
            "Grasp finger-table clearance must retain at least the distance-numeric reserve")
    require(grasp_table_clearance < static_clearance,
            "Grasp finger-table clearance must be below, and scoped separately from, static clearance")
    finger_home = float(p("robot.finger_home_m"))
    require(finger_low <= finger_home <= finger_high, "Gripper home coordinate is outside limits")

    object_x, object_y = [float(value) for value in p("object.size_xyz_m")[:2]]
    yaw_low, yaw_high = [float(value) for value in p("object.yaw_range_rad")]
    angles = [yaw_low, yaw_high]
    peak_index_low = math.ceil((yaw_low - math.pi / 4) / (math.pi / 2))
    peak_index_high = math.floor((yaw_high - math.pi / 4) / (math.pi / 2))
    angles.extend(math.pi / 4 + index * math.pi / 2 for index in range(peak_index_low, peak_index_high + 1))
    projected_span = max(object_x * abs(math.cos(angle)) + object_y * abs(math.sin(angle)) for angle in angles)
    require(abs(projected_span-float(p("robot.gripper_worst_yaw_object_span_m"))) < 1e-12,
            "Stored worst-yaw object span is inconsistent with the allowed yaw range")
    open_gap = 2.0 * (float(p("robot.finger_open_center_offset_m")) - float(p("robot.finger_thickness_m")) / 2.0)
    closed_gap = open_gap - 2.0 * finger_high
    require(open_gap > projected_span, "Open gripper is narrower than the object's worst allowed yaw projection")
    require(closed_gap < min(object_x, object_y), "Closed gripper cannot geometrically contact the minimum object width")
    require(float(p("robot.finger_length_m")) >= projected_span,
            "Finger contact length is shorter than the worst projected object span")
    require(0 < float(p("camera.fovy_deg")) < 180, "Camera vertical FOV must lie in (0, 180) degrees")
    resolution = np.asarray(p("camera.resolution_px"), dtype=float)
    require(resolution.shape == (2,) and np.all(resolution > 0) and np.all(resolution == np.floor(resolution)),
            "Camera resolution must be two positive integer pixel dimensions")
    zones = p("cell.tray_centers_xy_m")
    require(set(zones) == {"RED", "GREEN", "BLUE"}, "Exactly three sorting zones RED/GREEN/BLUE are required")
    require(len(p("cell.tray_slot_x_offsets_m")) >= 1, "At least one placement slot per tray is required")


def main() -> dict:
    config = build_model.load_ssot()
    validate_ssot_contract(config)
    p = lambda key: build_model.parameter(config, key)
    props = build_model.derived_properties(config)
    require(MODEL_PATH.is_file(), "Generated MJCF is missing")

    # Rebuild in memory and compare byte-for-byte to detect stale model files.
    expected_root = build_model.make_world(config, props)
    ET.indent(expected_root, space="  ")
    expected_bytes = ET.tostring(expected_root, encoding="utf-8", xml_declaration=True)
    actual_bytes = MODEL_PATH.read_bytes()
    require(expected_bytes == actual_bytes, "MJCF differs from the SSOT-driven generator output")

    # MjModel loader is the authoritative XML/model compiler check.
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    require(model.opt.timestep == float(p("environment.timestep_s")), "MuJoCo time step mismatch")
    require(int(model.opt.iterations) == int(p("environment.solver_iterations")),
            "MuJoCo solver iteration mismatch")
    require(int(model.opt.noslip_iterations) == int(p("environment.noslip_iterations")),
            "MuJoCo NoSlip iteration mismatch")
    require(abs(float(model.opt.impratio) - float(p("environment.impratio"))) <= 1e-12,
            "MuJoCo friction impedance ratio mismatch")
    np.testing.assert_allclose(model.opt.gravity, [0, 0, -float(p("environment.gravity_m_s2"))], rtol=0, atol=1e-12)

    expected_joints = {
        "j1_shoulder": (mujoco.mjtJoint.mjJNT_HINGE, p("robot.j1_range_rad"), [0,0,1]),
        "j2_elbow": (mujoco.mjtJoint.mjJNT_HINGE, p("robot.j2_range_rad"), [0,0,1]),
        "j3_lift": (mujoco.mjtJoint.mjJNT_SLIDE, p("robot.j3_range_m"), [0,0,-1]),
        "j4_wrist": (mujoco.mjtJoint.mjJNT_HINGE, p("robot.j4_range_rad"), [0,0,1]),
        "j5_finger_left": (mujoco.mjtJoint.mjJNT_SLIDE, p("robot.finger_range_m"), [0,-1,0]),
        "j6_finger_right": (mujoco.mjtJoint.mjJNT_SLIDE, p("robot.finger_range_m"), [0,1,0]),
    }
    joint_report = {}
    for name, (kind, limits, axis) in expected_joints.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        require(jid >= 0, f"Missing joint {name}")
        require(model.jnt_type[jid] == kind, f"Wrong joint type for {name}")
        np.testing.assert_allclose(model.jnt_range[jid], limits, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.jnt_axis[jid], axis, rtol=0, atol=1e-12)
        joint_report[name] = {"type": "hinge" if kind == mujoco.mjtJoint.mjJNT_HINGE else "slide",
                              "range": [float(v) for v in model.jnt_range[jid]],
                              "axis": [float(v) for v in model.jnt_axis[jid]],
                              "qpos_address": int(model.jnt_qposadr[jid])}

    require(model.njnt == 12, f"Expected 6 robot scalar joints plus 6 free objects; got njnt={model.njnt}")
    require(model.nq == 48 and model.nv == 42, f"Unexpected nq/nv {model.nq}/{model.nv}")
    require(model.nu == 5, f"Expected five controlled coordinates, got nu={model.nu}")
    require(model.neq == 1, f"Expected one finger coupling equality, got neq={model.neq}")
    require(model.nsensor == 2, f"Expected two local touch sensors, got nsensor={model.nsensor}")
    for site_name in ("touch_left_site", "touch_right_site"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        require(site_id >= 0, f"Missing touch sensor site {site_name}")
        require(model.site_type[site_id] == mujoco.mjtGeom.mjGEOM_SPHERE,
                f"Touch sensor site {site_name} must remain spherical")
        np.testing.assert_allclose(model.site_size[site_id, 0], p("robot.touch_site_radius_m"), rtol=0, atol=1e-12)
    allowed_pairs = {tuple(sorted(pair)) for pair in p("collision.allowed_pairs")}
    compiled_pairs = set()
    for signature in model.exclude_signature:
        body1_id = int(signature) >> 16
        body2_id = int(signature) & 0xFFFF
        body1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        body2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)
        compiled_pairs.add(tuple(sorted((body1, body2))))
    require(compiled_pairs == allowed_pairs,
            f"Compiled contact exclusions differ from explicit adjacent-pair allowlist: {compiled_pairs ^ allowed_pairs}")
    require(int(p("robot.scalar_joint_count")) == 6 and int(p("robot.effective_controlled_coordinates")) == 5,
            "SSOT joint/coordinate count mismatch")

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_static")
    require(base_id >= 0 and model.body_jntnum[base_id] == 0, "BASE must remain fixed to WORLD")
    require(model.nkey == 1, "Expected a single home observation keyframe")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home_observation")
    require(key_id >= 0, "Home observation keyframe is missing")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    home = p("camera.safe_observation_joint_pose")
    home_values = {"j1_shoulder": home["j1_shoulder"], "j2_elbow": home["j2_elbow"],
                   "j3_lift": home["j3_lift"], "j4_wrist": home["j4_wrist"],
                   "j5_finger_left": p("robot.finger_home_m"), "j6_finger_right": p("robot.finger_home_m")}
    for name, value in home_values.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actual = float(data.qpos[model.jnt_qposadr[jid]])
        require(abs(actual - float(value)) < 1e-10, f"Home keyframe mismatch for {name}: {actual} != {value}")
        low, high = model.jnt_range[jid]
        require(low-1e-12 <= actual <= high+1e-12, f"Home {name} outside limits")

    # Actuator command and effort ranges are checked against the SSOT, including the two-jaw effort convention.
    actuator_expected = {
        "act_j1_shoulder": (p("robot.j1_range_rad"), p("robot.j1_effort_Nm")),
        "act_j2_elbow": (p("robot.j2_range_rad"), p("robot.j2_effort_Nm")),
        "act_j3_lift": (p("robot.j3_range_m"), p("robot.j3_effort_N")),
        "act_j4_wrist": (p("robot.j4_range_rad"), p("robot.j4_effort_Nm")),
        "act_gripper_left": (p("robot.finger_range_m"), p("robot.gripper_actuator_total_force_N")),
    }
    actuator_report = {}
    for name, (control_range, effort) in actuator_expected.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        require(aid >= 0, f"Missing actuator {name}")
        np.testing.assert_allclose(model.actuator_ctrlrange[aid], control_range, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.actuator_forcerange[aid], [-float(effort), float(effort)], rtol=0, atol=1e-12)
        actuator_report[name] = {"ctrlrange": [float(v) for v in model.actuator_ctrlrange[aid]],
                                 "forcerange": [float(v) for v in model.actuator_forcerange[aid]]}

    # Compare inertial values generated from component calculations with compiled MuJoCo bodies.
    inertia_report = {}
    for name, expected in props.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        require(bid >= 0, f"Missing rigid body {name}")
        require(abs(float(model.body_mass[bid])-expected.mass_kg) < 1e-10, f"Body mass mismatch for {name}")
        np.testing.assert_allclose(model.body_ipos[bid], expected.com_m, rtol=0, atol=1e-9)
        # MJCF fullinertia is diagonalized by the compiler. Compare the
        # reconstructed tensor in the body frame, not the sorted principal
        # moments against the body-frame diagonal.
        inertial_rotation = np.zeros((3, 3), dtype=float)
        mujoco.mju_quat2Mat(inertial_rotation.ravel(), model.body_iquat[bid])
        reconstructed_inertia = inertial_rotation @ np.diag(model.body_inertia[bid]) @ inertial_rotation.T
        np.testing.assert_allclose(reconstructed_inertia, expected.inertia_kg_m2, rtol=1e-8, atol=1e-12)
        inertia_report[name] = {"mass_kg": float(model.body_mass[bid]),
                                "com_local_m": [float(v) for v in model.body_ipos[bid]],
                                "principal_inertia_kg_m2": [float(v) for v in model.body_inertia[bid]],
                                "inertial_quat_wxyz": [float(v) for v in model.body_iquat[bid]],
                                "reconstructed_body_frame_inertia_kg_m2": reconstructed_inertia.tolist()}

    # Visual/collision layers are distinct; collision boxes are hidden but active.
    geom_ids = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(model.ngeom)}
    visual_names = [name for name in geom_ids if name and name.endswith("_visual")]
    collision_names = [name for name in geom_ids if name and name.endswith("_collision")]
    # The four input-zone outlines are intentionally visual-only. Every other
    # physical shape must have a matching visual/collision pair.
    visual_bases = {name.removesuffix("_visual") for name in visual_names if not name.startswith("input_mark_")}
    collision_bases = {name.removesuffix("_collision") for name in collision_names}
    require(visual_names and collision_names and visual_bases == collision_bases,
            f"Visual/collision geom pairs are incomplete (unpaired visual={sorted(visual_bases-collision_bases)}, unpaired collision={sorted(collision_bases-visual_bases)})")
    for name in visual_names:
        gid=geom_ids[name]
        require(model.geom_contype[gid] == 0 and model.geom_conaffinity[gid] == 0, f"Visual geom collides: {name}")
    for name in collision_names:
        gid=geom_ids[name]
        require(model.geom_contype[gid] != 0 and model.geom_conaffinity[gid] != 0, f"Collision geom disabled: {name}")

    # Home should contain only intentional static support contacts (the six objects on the tabletop).
    contacts=[]
    unexpected=[]
    for index in range(data.ncon):
        contact=data.contact[index]
        body1=int(model.geom_bodyid[contact.geom1]); body2=int(model.geom_bodyid[contact.geom2])
        name1=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body1)
        name2=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,body2)
        pair={name1,name2}
        record={"body1":name1,"body2":name2,"distance_m":float(contact.dist)}
        contacts.append(record)
        if not ("table" in pair and any(name and name.startswith("object_") for name in pair)):
            unexpected.append(record)
        require(float(contact.dist) >= -1e-5, f"Initial collision penetration exceeds tolerance: {record}")
    require(not unexpected, f"Unexpected home contacts/penetrations: {unexpected}")
    support_pairs = {(item["body1"], item["body2"]) for item in contacts}
    require(len(support_pairs) == 6,
            f"Expected six unique intentional object-table support pairs, got {len(support_pairs)} across {len(contacts)} contact points")

    # Coarse radial target coverage only. No analytic/numerical IK is run here.
    rmin,rmax=[float(v) for v in p("robot.radial_reach_min_max_m")]
    points=[]
    input_center=p("cell.input_center_xy_m")
    for dy in p("cell.input_row_y_offsets_m"):
        for dx in p("cell.input_slot_x_offsets_m"):
            points.append((input_center[0]+dx,input_center[1]+dy,"INPUT"))
    for cls,center in p("cell.tray_centers_xy_m").items():
        for dx in p("cell.tray_slot_x_offsets_m"):
            points.append((center[0]+dx,center[1]+p("cell.tray_slot_y_offset_m"),cls))
    radius_report=[]
    for x,y,label in points:
        radius=math.hypot(x,y)
        require(rmin < radius < rmax, f"Target {label} center radial coordinate {radius} outside [{rmin},{rmax}]")
        radius_report.append({"zone":label,"xy_m":[x,y],"radius_m":radius,"radial_margin_m":min(radius-rmin,rmax-radius)})

    # Tray contents, source slots and fingers fit with explicit worst-yaw margins.
    inner=np.asarray(p("cell.tray_inner_size_xy_m"),dtype=float)
    half_diag=float(p("object.size_xyz_m")[0])*math.sqrt(2)/2
    finger_half=float(p("robot.finger_length_m"))/2
    slot_clearances=[]
    for cls,center in p("cell.tray_centers_xy_m").items():
        for dx in p("cell.tray_slot_x_offsets_m"):
            cx=center[0]+dx
            object_edge=abs(dx)+half_diag
            finger_edge=abs(dx)+finger_half
            require(object_edge < inner[0]/2, f"Object footprint does not fit in {cls} tray")
            require(finger_edge < inner[0]/2, f"Open finger envelope does not clear {cls} tray wall")
            slot_clearances.append({"class":cls,"x_offset_m":dx,"object_wall_clearance_m":inner[0]/2-object_edge,
                                    "finger_wall_clearance_m":inner[0]/2-finger_edge})
    input_size=p("cell.input_size_xy_m")
    for dy in p("cell.input_row_y_offsets_m"):
        for dx in p("cell.input_slot_x_offsets_m"):
            require(abs(dx)+half_diag < input_size[0]/2 and abs(dy)+half_diag < input_size[1]/2,
                    "Input object footprint exceeds source area")

    camera_h=float(p("camera.position_world_m")[2])-float(p("cell.table_top_z_m"))
    fovy=math.radians(float(p("camera.fovy_deg")))
    width,height=[float(v) for v in p("camera.resolution_px")]
    expected_h=2*camera_h*math.tan(fovy/2); expected_w=expected_h*width/height
    np.testing.assert_allclose(p("camera.coverage_width_height_m"),[expected_w,expected_h],rtol=0,atol=1e-12)
    table_width, table_height = [float(v) for v in p("cell.table_size_xy_m")]
    require(expected_w >= table_width and expected_h >= table_height,
            f"Overhead camera footprint {expected_w:.3f}x{expected_h:.3f} m does not cover table {table_width:.3f}x{table_height:.3f} m")
    fx,fy,cx,cy=[float(v) for v in p("camera.intrinsics_fx_fy_cx_cy_px")]
    require(abs(fx-fy)<1e-10 and abs(cx-(width-1)/2)<1e-10 and abs(cy-(height-1)/2)<1e-10,
            "Pinhole intrinsic matrix is inconsistent")

    # Force and grasp budgets are self-consistent and labelled as model assumptions.
    require(abs(float(p("robot.grasp_xy_error_budget_m"))-sum(float(p(k)) for k in (
        "robot.grasp_xy_budget_perception_m","robot.grasp_xy_budget_kinematics_m","robot.grasp_xy_budget_tracking_m","robot.grasp_xy_budget_reserve_m")))<1e-12,
        "Grasp XY error budget does not sum")
    require(float(p("robot.grasp_xy_error_budget_m")) < float(p("robot.gripper_worst_yaw_half_clearance_m")),
            "XY error budget exceeds worst-yaw geometric clearance")
    require(float(p("robot.gripper_contact_force_per_finger_N")) >= float(p("gripper.minimum_normal_force_per_finger_N")),
            "Gripper modeled force is below the safety-factored friction requirement")
    z_force=float(p("gripper.z_axis_moving_mass_including_payload_kg"))*(float(p("environment.gravity_m_s2"))+float(p("gripper.payload_vertical_acceleration_m_s2")))
    require(float(p("gripper.z_axis_force_limit_N")) > z_force, "Z actuator force limit does not cover payload load estimate")
    require(abs(float(p("robot.j3_pick_q_m"))-float(p("robot.j3_range_m")[1])) >= 0.001,
            "Pick pose lacks at least 1 mm Z-stroke margin")

    # Save actual MuJoCo renders for visual audit; no sorting/controller behavior is executed.
    RESULTS.mkdir(parents=True,exist_ok=True)
    w,h=[int(v) for v in p("camera.resolution_px")]
    # Documentation captures are rendered at 2x nominal sensor resolution;
    # the simulated RGB sensor remains at the SSOT's 640x480 resolution.
    render_w, render_h = 2*w, 2*h
    with mujoco.Renderer(model,width=render_w,height=render_h) as renderer:
        renderer.update_scene(data,camera="overhead_rgb")
        overhead=renderer.render()
        Image.fromarray(overhead).save(RESULTS/"m2_model_overhead.png")
        freecam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(freecam)
        freecam.lookat[:]=[0.0,0.0,0.08]; freecam.distance=1.55; freecam.azimuth=135; freecam.elevation=-28
        renderer.update_scene(data,camera=freecam)
        isometric=renderer.render()
        Image.fromarray(isometric).save(RESULTS/"m2_model_isometric.png")

    report={
        "status":"PASS",
        "config_version":config["config_version"],
        "model_sha256":hashlib.sha256(actual_bytes).hexdigest(),
        "mujoco_version":mujoco.__version__,
        "model_shape":{"nbody":int(model.nbody),"njnt":int(model.njnt),"nq":int(model.nq),"nv":int(model.nv),"nu":int(model.nu),"neq":int(model.neq),"nsensor":int(model.nsensor),"nkey":int(model.nkey)},
        "explicit_adjacent_pair_exclusions":sorted([list(pair) for pair in compiled_pairs]),
        "joint_report":joint_report,"actuator_report":actuator_report,"inertial_report":inertia_report,
        "home_qpos_joints":home_values,"initial_contact_point_count":len(contacts),
        "initial_unique_support_pair_count":len(support_pairs),"initial_expected_contacts":contacts,
        "radial_reach_m":[rmin,rmax],"target_radial_checks":radius_report,"tray_slot_clearances":slot_clearances,
        "camera_coverage_m":[expected_w,expected_h],"camera_intrinsics_fx_fy_cx_cy_px":[fx,fy,cx,cy],
        "object_mass_kg":float(p("object.mass_kg")),
        "grasp_force_min_per_finger_N":float(p("gripper.minimum_normal_force_per_finger_N")),
        "grasp_force_model_per_finger_N":float(p("robot.gripper_contact_force_per_finger_N")),
        "z_axis_moving_mass_with_payload_kg":float(p("gripper.z_axis_moving_mass_including_payload_kg")),
        "z_axis_load_estimate_N":z_force,
        "documentation_render_resolution_px":[render_w,render_h],
        "renders":["m2_model_overhead.png","m2_model_isometric.png"],
        "scope_note":"Geometry and model compilation only; no FK/IK, perception calibration, trajectory planning, FSM, full sorting run, or final tuning."
    }
    (RESULTS/"m2_validation.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    return report


if __name__=="__main__":
    try:
        print(json.dumps(main(),ensure_ascii=False,indent=2))
    except Exception as exc:
        print(f"M2 SSOT/model validation failed: {exc}",file=sys.stderr)
        raise
