"""Generate the M2 MuJoCo scene and derived engineering documents from the SSOT."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SSOT_PATH = ROOT / "02_Спецификация" / "параметры_системы.yaml"
SOURCE_DIR = ROOT / "03_Модель_и_схемы" / "source"
RESULT_DIR = ROOT / "99_Рабочие_материалы" / "m2_validation" / "results"
# ``inherited`` records that a downstream milestone carries a value forward
# without re-estimating it. It is provenance metadata, not a new physical
# parameter class, and must remain valid as SSOT grows beyond M2.
VALID_ORIGINS = {"source", "calculation", "design_assumption", "tuned_parameter", "inherited"}


def load_ssot() -> dict[str, Any]:
    with SSOT_PATH.open("r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not {"project", "parameters", "assemblies"}.issubset(config):
        raise ValueError("SSOT must define project, parameters, and assemblies mappings")
    if config.get("schema_version") != "1.0":
        raise ValueError(f"Unsupported schema_version: {config.get('schema_version')!r}")
    if config.get("config_version") not in {"M2-v1.0", "M3-v1.0", "M4-v1.1", "M4-v1.2", "M4-v1.3", "M5-v1.0", "M5-v1.1", "M5-v1.2", "M5-v1.3", "M7-v1.0", "M7-v1.1"}:
        raise ValueError("Unsupported config_version; record a controlled project revision")
    for key, item in config["parameters"].items():
        if not isinstance(item, dict):
            raise ValueError(f"Parameter {key} is not a metadata mapping")
        for required in ("value", "unit", "origin", "basis", "owner"):
            if required not in item:
                raise ValueError(f"Parameter {key} has no {required}")
        if item["origin"] not in VALID_ORIGINS:
            raise ValueError(f"Parameter {key} has unsupported origin {item['origin']}")
        if not item["unit"] or not item["basis"] or not item["owner"]:
            raise ValueError(f"Parameter {key} is missing unit, basis, or owner")
    return config


def parameter(config: dict[str, Any], key: str) -> Any:
    try:
        return config["parameters"][key]["value"]
    except KeyError as exc:
        raise KeyError(f"Missing SSOT parameter: {key}") from exc


def set_parameter(config: dict[str, Any], key: str, value: Any) -> None:
    if key not in config["parameters"]:
        raise KeyError(f"Missing derived parameter slot: {key}")
    config["parameters"][key]["value"] = value


def box_mass_inertia(dims: np.ndarray, density: float) -> tuple[float, np.ndarray]:
    if np.any(dims <= 0):
        raise ValueError(f"Non-positive box dimensions: {dims}")
    mass = density * float(np.prod(dims))
    x, y, z = dims
    inertia = mass / 12.0 * np.diag([y*y + z*z, x*x + z*z, x*x + y*y])
    return mass, inertia


def component_mass_inertia(config: dict[str, Any], component: dict[str, Any]) -> tuple[float, np.ndarray, np.ndarray]:
    density = float(parameter(config, "environment.density_pla_kg_m3"))
    shape = component["shape"]
    center = np.asarray(parameter(config, component["center_ref"]), dtype=float)
    if shape in {"solid_box", "closed_rect_shell", "open_rect_tube"}:
        outer = np.asarray(parameter(config, component["size_ref"]), dtype=float)
        outer_mass, outer_i = box_mass_inertia(outer, density)
        if shape == "solid_box":
            return outer_mass, center, outer_i
        wall = float(parameter(config, component["wall_ref"]))
        inner = outer.copy()
        if shape == "closed_rect_shell":
            inner -= 2.0 * wall
        else:
            axis = {"x": 0, "y": 1, "z": 2}[component["axis"]]
            for index in range(3):
                if index != axis:
                    inner[index] -= 2.0 * wall
        inner_mass, inner_i = box_mass_inertia(inner, density)
        return outer_mass - inner_mass, center, outer_i - inner_i
    if shape == "solid_cylinder_z":
        radius = float(parameter(config, component["radius_param"]))
        height = float(parameter(config, component["height_param"]))
        mass = density * math.pi * radius**2 * height
        inertia = np.diag([
            mass * (3.0 * radius**2 + height**2) / 12.0,
            mass * (3.0 * radius**2 + height**2) / 12.0,
            mass * radius**2 / 2.0,
        ])
        return mass, center, inertia
    raise ValueError(f"Unsupported inertial component shape {shape!r}")


@dataclass(frozen=True)
class BodyProperties:
    mass_kg: float
    com_m: np.ndarray
    inertia_kg_m2: np.ndarray


def combine_components(config: dict[str, Any], components: list[dict[str, Any]]) -> BodyProperties:
    pieces = [component_mass_inertia(config, item) for item in components]
    total_mass = sum(item[0] for item in pieces)
    if total_mass <= 0:
        raise ValueError("Body assembly has non-positive mass")
    com = sum((mass * center for mass, center, _ in pieces), np.zeros(3)) / total_mass
    inertia = np.zeros((3, 3), dtype=float)
    for mass, center, local_i in pieces:
        delta = center - com
        inertia += local_i + mass * ((delta @ delta) * np.eye(3) - np.outer(delta, delta))
    inertia = 0.5 * (inertia + inertia.T)
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.min(eigenvalues) <= 0:
        raise ValueError(f"Body inertia is not positive definite: {eigenvalues}")
    moments = np.diag(inertia)
    if np.max(moments) > np.sum(moments) - np.max(moments) + 1e-12:
        raise ValueError(f"Principal moment triangle inequality failed: {moments}")
    return BodyProperties(total_mass, com, inertia)


def derived_properties(config: dict[str, Any]) -> dict[str, BodyProperties]:
    props = {name: combine_components(config, spec["components"]) for name, spec in config["assemblies"].items()}
    set_parameter(config, "robot.mass_by_body_kg", {key: round(item.mass_kg, 12) for key, item in props.items()})
    set_parameter(config, "robot.com_by_body_m", {key: [round(float(v), 12) for v in item.com_m] for key, item in props.items()})
    set_parameter(config, "robot.inertia_by_body_kg_m2", {
        key: [round(float(item.inertia_kg_m2[i,j]), 15) for i,j in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))]
        for key,item in props.items()
    })

    l1, l2 = float(parameter(config, "robot.link1_length_m")), float(parameter(config, "robot.link2_length_m"))
    elbow_limit = max(abs(float(v)) for v in parameter(config, "robot.j2_range_rad"))
    rmax = l1 + l2
    rmin = math.sqrt(max(0.0, l1*l1 + l2*l2 + 2.0*l1*l2*math.cos(elbow_limit)))
    set_parameter(config, "robot.radial_reach_min_max_m", [rmin, rmax])

    tray_outer = np.asarray(parameter(config, "cell.tray_outer_size_xy_m"), dtype=float)
    tray_inner = tray_outer - 2.0 * float(parameter(config, "cell.tray_wall_thickness_m"))
    set_parameter(config, "cell.tray_inner_size_xy_m", [float(v) for v in tray_inner])

    width, height = [float(v) for v in parameter(config, "camera.resolution_px")]
    cam_z = float(parameter(config, "camera.position_world_m")[2])
    table_z = float(parameter(config, "cell.table_top_z_m"))
    fovy = math.radians(float(parameter(config, "camera.fovy_deg")))
    fy = height / (2.0 * math.tan(fovy / 2.0))
    fx, cx, cy = fy, (width - 1.0)/2.0, (height - 1.0)/2.0
    set_parameter(config, "camera.intrinsics_fx_fy_cx_cy_px", [fx, fy, cx, cy])
    vcover = 2.0 * (cam_z - table_z) * math.tan(fovy/2.0)
    set_parameter(config, "camera.coverage_width_height_m", [vcover*width/height, vcover])

    side = float(parameter(config, "object.size_xyz_m")[0])
    span = side*math.sqrt(2.0)
    finger_length = float(parameter(config, "robot.finger_length_m"))
    gap = 2.0*float(parameter(config, "robot.finger_open_center_offset_m"))-float(parameter(config, "robot.finger_thickness_m"))
    half_clearance = min(finger_length-span, gap-span)/2.0
    set_parameter(config, "robot.gripper_worst_yaw_object_span_m", span)
    set_parameter(config, "robot.gripper_worst_yaw_half_clearance_m", half_clearance)

    density = float(parameter(config, "environment.density_pla_kg_m3"))
    object_dims = np.asarray(parameter(config, "object.size_xyz_m"), dtype=float)
    object_mass = density*float(np.prod(object_dims))
    set_parameter(config, "object.mass_kg", object_mass)
    moving_mass = sum(props[name].mass_kg for name in ("z_slide","wrist","finger_left","finger_right"))+object_mass
    set_parameter(config, "gripper.z_axis_moving_mass_including_payload_kg", moving_mass)
    g, accel = float(parameter(config,"environment.gravity_m_s2")), float(parameter(config,"gripper.payload_vertical_acceleration_m_s2"))
    mu, sf = float(parameter(config,"gripper.required_pair_friction")), float(parameter(config,"gripper.safety_factor"))
    set_parameter(config,"gripper.minimum_normal_force_per_finger_N",sf*object_mass*(g+accel)/(2*mu))
    set_parameter(config,"robot.gripper_contact_force_per_finger_N",float(parameter(config,"robot.gripper_actuator_total_force_N"))/2)
    assigned=sum(float(parameter(config,k)) for k in ("robot.grasp_xy_budget_perception_m","robot.grasp_xy_budget_kinematics_m","robot.grasp_xy_budget_tracking_m"))
    set_parameter(config,"robot.grasp_xy_budget_reserve_m",float(parameter(config,"robot.grasp_xy_error_budget_m"))-assigned)
    return props


def fmt(values: Any) -> str:
    if isinstance(values, (list, tuple, np.ndarray)):
        return " ".join(fmt(v) for v in values)
    if isinstance(values, (float, np.floating)):
        # Keep a round-trippable IEEE-754 representation in generated MJCF.
        return f"{float(values):.17g}"
    return str(values)


def add(parent: ET.Element, tag: str, **attrs: Any) -> ET.Element:
    return ET.SubElement(parent, tag, {key: fmt(value) for key, value in attrs.items()})


def add_inertial(body: ET.Element, props: BodyProperties) -> None:
    inertia = props.inertia_kg_m2
    full = [inertia[0,0], inertia[1,1], inertia[2,2], inertia[0,1], inertia[0,2], inertia[1,2]]
    add(body, "inertial", pos=props.com_m, mass=props.mass_kg, fullinertia=full)


def geom_pair(parent: ET.Element, name: str, shape: str, full_size: list[float] | np.ndarray,
              pos: list[float] | np.ndarray, rgba: list[float], friction: list[float],
              config: dict[str, Any], *, collision: bool = True) -> None:
    size = np.asarray(full_size, dtype=float)
    center = np.asarray(pos, dtype=float)
    if shape == "box":
        geom_type, geom_size = "box", size/2.0
    elif shape == "cylinder_z":
        geom_type, geom_size = "cylinder", np.array([size[0], size[1]/2.0])
    else:
        raise ValueError(f"Unsupported geom shape {shape!r}")
    add(parent,"geom",name=f"{name}_visual",type=geom_type,size=geom_size,pos=center,rgba=rgba,
        contype=0,conaffinity=0,group=int(parameter(config,"collision.visual_geom_group")),mass=0)
    if collision:
        add(parent,"geom",name=f"{name}_collision",type=geom_type,size=geom_size,pos=center,rgba=[0,0,0,0],
            contype=1,conaffinity=1,group=int(parameter(config,"collision.collision_geom_group")),mass=0,
            friction=friction,condim=int(parameter(config,"contact.condim")),
            solref=parameter(config,"contact.solref"),solimp=parameter(config,"contact.solimp"))


def primitive_size(component: dict[str, Any], config: dict[str, Any]) -> tuple[str,list[float],np.ndarray]:
    shape=component["shape"]
    center=np.asarray(parameter(config,component["center_ref"]),dtype=float)
    if shape in {"solid_box","closed_rect_shell","open_rect_tube"}:
        return "box",list(parameter(config,component["size_ref"])),center
    if shape=="solid_cylinder_z":
        return "cylinder_z",[float(parameter(config,component["radius_param"])),float(parameter(config,component["height_param"]))],center
    raise ValueError(shape)


def add_assembly_geoms(body: ET.Element, assembly: dict[str,Any], rgba: list[float], friction: list[float], config: dict[str,Any]) -> None:
    for component in assembly["components"]:
        shape,size,center=primitive_size(component,config)
        geom_pair(body,f"{assembly['body_name']}_{component['name']}",shape,size,center,rgba,friction,config)


def make_world(config: dict[str,Any], props: dict[str,BodyProperties]) -> ET.Element:
    root=ET.Element("mujoco",{"model":"scara_color_sorter_m2"})
    add(root,"compiler",angle="radian",autolimits="true",fusestatic="false",inertiafromgeom="false",balanceinertia="false")
    add(root,"option",timestep=parameter(config,"environment.timestep_s"),
        gravity=[0,0,-float(parameter(config,"environment.gravity_m_s2"))],integrator="implicitfast",solver="Newton",
        iterations=parameter(config,"environment.solver_iterations"),tolerance=parameter(config,"environment.solver_tolerance"),cone="elliptic")
    add(root,"size",njmax=3000,nconmax=800)
    visual=add(root,"visual")
    camera_width, camera_height = [int(value) for value in parameter(config,"camera.resolution_px")]
    # Allocate a 2x offscreen framebuffer for documentation renders; this does
    # not change the nominal RGB sensor resolution stored in the SSOT.
    add(visual,"global",azimuth=135,elevation=-25,offwidth=2*camera_width,offheight=2*camera_height)
    default=add(root,"default")
    add(default,"joint",limited="true",armature=0.001)
    add(default,"geom",contype=1,conaffinity=1,condim=int(parameter(config,"contact.condim")),
        solref=parameter(config,"contact.solref"),solimp=parameter(config,"contact.solimp"))

    world=add(root,"worldbody")
    add(world,"light",name="key_light",pos=[0,0,1.35],dir=[0,0,-1],directional="true",
        diffuse=[0.82,0.82,0.82],ambient=[0.36,0.36,0.36],specular=[0.1,0.1,0.1])
    add(world,"camera",name="overhead_rgb",pos=parameter(config,"camera.position_world_m"),
        quat=parameter(config,"camera.mujoco_quaternion_wxyz"),fovy=parameter(config,"camera.fovy_deg"),mode="fixed")

    table_xy=parameter(config,"cell.table_size_xy_m")
    table_t=float(parameter(config,"cell.table_thickness_m"))
    table_z=float(parameter(config,"cell.table_top_z_m"))
    table=add(world,"body",name="table",pos=[0,0,0])
    table_friction=[parameter(config,"contact.friction_table_slide"),parameter(config,"contact.friction_table_spin"),parameter(config,"contact.friction_table_roll")]
    geom_pair(table,"table_top","box",[table_xy[0],table_xy[1],table_t],[0,0,table_z-table_t/2],
        [0.80,0.82,0.81,1],table_friction,config)

    # The coaxial axle overlaps the shoulder-link envelope intentionally; that one pair is excluded.
    base=add(world,"body",name="base_static",pos=[0,0,0])
    geom_pair(base,"base_foot","box",parameter(config,"robot.base_foot_outer_xyz_m"),parameter(config,"robot.base_foot_center_world_m"),
        [0.28,0.34,0.42,1],[0.5,0.005,0.001],config)
    geom_pair(base,"base_column","box",parameter(config,"robot.base_column_outer_xyz_m"),parameter(config,"robot.base_column_center_world_m"),
        [0.32,0.39,0.47,1],[0.5,0.005,0.001],config)
    geom_pair(base,"base_axle","cylinder_z",[parameter(config,"robot.base_axle_radius_m"),parameter(config,"robot.base_axle_height_m")],
        parameter(config,"robot.base_axle_center_world_m"),[0.38,0.44,0.52,1],[0.5,0.005,0.001],config)

    # Source-area markings are visual only and do not change contact or object detection.
    in_center=parameter(config,"cell.input_center_xy_m")
    in_size=parameter(config,"cell.input_size_xy_m")
    marker=add(world,"body",name="input_zone_visual",pos=[0,0,0])
    line,zmark=0.002,table_z+0.0005
    grey=parameter(config,"cell.input_mark_rgba")
    marks=[("xmin",[line,in_size[1],0.001],[in_center[0]-in_size[0]/2,in_center[1],zmark]),
           ("xmax",[line,in_size[1],0.001],[in_center[0]+in_size[0]/2,in_center[1],zmark]),
           ("ymin",[in_size[0],line,0.001],[in_center[0],in_center[1]-in_size[1]/2,zmark]),
           ("ymax",[in_size[0],line,0.001],[in_center[0],in_center[1]+in_size[1]/2,zmark])]
    for label,size,pos in marks:
        geom_pair(marker,f"input_mark_{label}","box",size,pos,grey,[0.5,0.005,0.001],config,collision=False)

    outer=parameter(config,"cell.tray_outer_size_xy_m")
    floor_t=float(parameter(config,"cell.tray_floor_thickness_m"))
    wall_t=float(parameter(config,"cell.tray_wall_thickness_m"))
    wall_h=float(parameter(config,"cell.tray_wall_height_m"))
    for label,center in parameter(config,"cell.tray_centers_xy_m").items():
        tray=add(world,"body",name=f"tray_{label.lower()}",pos=[center[0],center[1],0])
        color=np.asarray(parameter(config,"cell.bin_rgba")[label],dtype=float)
        floor_color=[float(0.72*color[i]+0.28) for i in range(3)]+[1.0]
        geom_pair(tray,f"tray_{label.lower()}_floor","box",[outer[0],outer[1],floor_t],[0,0,floor_t/2],floor_color,table_friction,config)
        z= floor_t+wall_h/2
        for side,sx in (("left",-1),("right",1)):
            geom_pair(tray,f"tray_{label.lower()}_{side}_wall","box",[wall_t,outer[1]-2*wall_t,wall_h],
                [sx*(outer[0]/2-wall_t/2),0,z],color.tolist(),table_friction,config)
        for side,sy in (("front",-1),("back",1)):
            geom_pair(tray,f"tray_{label.lower()}_{side}_wall","box",[outer[0],wall_t,wall_h],
                [0,sy*(outer[1]/2-wall_t/2),z],color.tolist(),table_friction,config)

    # R-R-P-R chain. The right finger is a second physical slide joint constrained to the left.
    arm_color=parameter(config,"robot.visual_rgba")
    grip_friction=[parameter(config,"gripper.friction_slide"),parameter(config,"gripper.friction_spin"),parameter(config,"gripper.friction_roll")]
    l1,l2=float(parameter(config,"robot.link1_length_m")),float(parameter(config,"robot.link2_length_m"))
    link1=add(world,"body",name="link1",pos=[0,0,parameter(config,"robot.shoulder_z_m")])
    add(link1,"joint",name="j1_shoulder",type="hinge",axis=[0,0,1],range=parameter(config,"robot.j1_range_rad"),damping=parameter(config,"robot.j1_damping_Nm_s_rad"))
    add_inertial(link1,props["link1"])
    add_assembly_geoms(link1,config["assemblies"]["link1"],arm_color,grip_friction,config)

    link2=add(link1,"body",name="link2",pos=[l1,0,0])
    add(link2,"joint",name="j2_elbow",type="hinge",axis=[0,0,1],range=parameter(config,"robot.j2_range_rad"),damping=parameter(config,"robot.j2_damping_Nm_s_rad"))
    add_inertial(link2,props["link2"])
    add_assembly_geoms(link2,config["assemblies"]["link2"],arm_color,grip_friction,config)

    zslide=add(link2,"body",name="z_slide",pos=[l2,0,0])
    add(zslide,"joint",name="j3_lift",type="slide",axis=[0,0,-1],range=parameter(config,"robot.j3_range_m"),damping=parameter(config,"robot.j3_damping_N_s_m"))
    add_inertial(zslide,props["z_slide"])
    add_assembly_geoms(zslide,config["assemblies"]["z_slide"],arm_color,grip_friction,config)

    wrist=add(zslide,"body",name="wrist",pos=parameter(config,"robot.wrist_center_in_zslide_m"))
    add(wrist,"joint",name="j4_wrist",type="hinge",axis=[0,0,1],range=parameter(config,"robot.j4_range_rad"),damping=parameter(config,"robot.j4_damping_Nm_s_rad"))
    add_inertial(wrist,props["wrist"])
    add_assembly_geoms(wrist,config["assemblies"]["wrist"],arm_color,grip_friction,config)
    add(wrist,"site",name="tcp",pos=parameter(config,"robot.tcp_in_wrist_m"),type="sphere",size=0.003,rgba=[1,1,0,1],group=2)

    finger_offset=float(parameter(config,"robot.finger_open_center_offset_m"))
    half_thickness=float(parameter(config,"robot.finger_thickness_m"))/2
    palm_z=float(parameter(config,"robot.finger_center_in_wrist_m")[2])
    finger_size=parameter(config,"robot.finger_xyz_m")
    touch_radius=float(parameter(config,"robot.touch_site_radius_m"))
    left=add(wrist,"body",name="finger_left",pos=[0,finger_offset,palm_z])
    add(left,"joint",name="j5_finger_left",type="slide",axis=[0,-1,0],range=parameter(config,"robot.finger_range_m"),damping=parameter(config,"robot.finger_damping_N_s_m"))
    add_inertial(left,props["finger_left"])
    geom_pair(left,"finger_left","box",finger_size,[0,0,0],arm_color,grip_friction,config)
    add(left,"site",name="touch_left_site",type="sphere",pos=[0,-half_thickness,0],size=touch_radius,rgba=[1,0.2,0.2,0.35],group=2)
    right=add(wrist,"body",name="finger_right",pos=[0,-finger_offset,palm_z])
    add(right,"joint",name="j6_finger_right",type="slide",axis=[0,1,0],range=parameter(config,"robot.finger_range_m"),damping=parameter(config,"robot.finger_damping_N_s_m"))
    add_inertial(right,props["finger_right"])
    geom_pair(right,"finger_right","box",finger_size,[0,0,0],arm_color,grip_friction,config)
    add(right,"site",name="touch_right_site",type="sphere",pos=[0,half_thickness,0],size=touch_radius,rgba=[0.2,0.2,1,0.35],group=2)

    source_center=parameter(config,"cell.input_center_xy_m")
    yaw=float(parameter(config,"object.initial_yaw_rad"))
    classes=("RED","GREEN","BLUE")
    object_dims=parameter(config,"object.size_xyz_m")
    class_rgba=parameter(config,"object.visual_rgba")
    table_friction=[parameter(config,"contact.friction_table_slide"),parameter(config,"contact.friction_table_spin"),parameter(config,"contact.friction_table_roll")]
    for row,dy in enumerate(parameter(config,"cell.input_row_y_offsets_m"),start=1):
        for col,dx in enumerate(parameter(config,"cell.input_slot_x_offsets_m")):
            label=classes[col]
            name=f"object_{label.lower()}_{row:02d}"
            quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
            body=add(world,"body",name=name,pos=[source_center[0]+dx,source_center[1]+dy,parameter(config,"object.position_z_m")],quat=quat)
            add(body,"freejoint",name=f"{name}_free")
            mass=float(parameter(config,"object.mass_kg"))
            ix=mass*(object_dims[1]**2+object_dims[2]**2)/12
            iy=mass*(object_dims[0]**2+object_dims[2]**2)/12
            iz=mass*(object_dims[0]**2+object_dims[1]**2)/12
            add(body,"inertial",pos=[0,0,0],mass=mass,fullinertia=[ix,iy,iz,0,0,0])
            geom_pair(body,name,"box",object_dims,[0,0,0],class_rgba[label],table_friction,config)

    contact=add(root,"contact")
    for body1,body2 in parameter(config,"collision.allowed_pairs"):
        add(contact,"exclude",body1=body1,body2=body2)
    equality=add(root,"equality")
    add(equality,"joint",name="finger_mimic",joint1="j6_finger_right",joint2="j5_finger_left",polycoef=[0,1,0,0,0])

    actuator=add(root,"actuator")
    gains=parameter(config,"robot.position_servo_gains")
    actuator_specs=[
        ("act_j1_shoulder","j1_shoulder","robot.j1_range_rad","robot.j1_effort_Nm",gains["j1_kp"],gains["j1_kv"]),
        ("act_j2_elbow","j2_elbow","robot.j2_range_rad","robot.j2_effort_Nm",gains["j2_kp"],gains["j2_kv"]),
        ("act_j3_lift","j3_lift","robot.j3_range_m","robot.j3_effort_N",gains["j3_kp"],gains["j3_kv"]),
        ("act_j4_wrist","j4_wrist","robot.j4_range_rad","robot.j4_effort_Nm",gains["j4_kp"],gains["j4_kv"]),
        ("act_gripper_left","j5_finger_left","robot.finger_range_m","robot.gripper_actuator_total_force_N",gains["finger_kp"],gains["finger_kv"]),
    ]
    for name,joint,range_key,effort_key,kp,kv in actuator_specs:
        effort=float(parameter(config,effort_key))
        add(actuator,"position",name=name,joint=joint,kp=kp,kv=kv,ctrllimited="true",ctrlrange=parameter(config,range_key),
            forcelimited="true",forcerange=[-effort,effort])
    sensor=add(root,"sensor")
    add(sensor,"touch",name="touch_left",site="touch_left_site")
    add(sensor,"touch",name="touch_right",site="touch_right_site")

    # Build keyframe qpos by named joint addresses, not a brittle hard-coded qpos index.
    model=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding="unicode"))
    home=parameter(config,"camera.safe_observation_joint_pose")
    home_values={"j1_shoulder":float(home["j1_shoulder"]),"j2_elbow":float(home["j2_elbow"]),
                 "j3_lift":float(home["j3_lift"]),"j4_wrist":float(home["j4_wrist"]),
                 "j5_finger_left":float(parameter(config,"robot.finger_home_m")),
                 "j6_finger_right":float(parameter(config,"robot.finger_home_m"))}
    qpos=model.qpos0.copy()
    for name,value in home_values.items():
        jid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,name)
        qpos[model.jnt_qposadr[jid]]=value
    ctrl=np.zeros(model.nu,dtype=float)
    targets={"act_j1_shoulder":home_values["j1_shoulder"],"act_j2_elbow":home_values["j2_elbow"],
             "act_j3_lift":home_values["j3_lift"],"act_j4_wrist":home_values["j4_wrist"],
             "act_gripper_left":home_values["j5_finger_left"]}
    for name,value in targets.items():
        aid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name)
        ctrl[aid]=value
    keyframe=add(root,"keyframe")
    add(keyframe,"key",name="home_observation",qpos=qpos,ctrl=ctrl)
    return root


def write_registry(config: dict[str,Any]) -> None:
    lines=["# Реестр параметров проекта","",f"**SSOT:** `параметры_системы.yaml`  ",
           f"**Версия конфигурации:** `{config['config_version']}`  ",
           f"**Геометрическая база:** `{config['project'].get('geometry_baseline_version', 'UNKNOWN')}`; YAML — единственный редактируемый источник чисел, этот реестр формируется генератором модели.","",
           "| Ключ | Значение | Единица | Происхождение | Основание / владелец |","|---|---|---|---|---|"]
    for key,item in config["parameters"].items():
        value=json.dumps(item["value"],ensure_ascii=False,separators=(",",":" )).replace("|","\\|")
        basis=str(item["basis"]).replace("|","\\|")
        lines.append(f"| `{key}` | `{value}` | {item['unit']} | `{item['origin']}` | {basis}; **{item['owner']}** |")
    lines += ["","Категории происхождения: `source` — документированный источник или M1; `calculation` — формула из SSOT; `design_assumption` — проектное допущение; `tuned_parameter` — подлежит настройке. Ограничения скорости и ускорения хранятся в конфигурации, но не являются жесткими лимитами MuJoCo joints; их должны обеспечивать M5–M7.",""]
    (ROOT/"02_Спецификация"/"Реестр_параметров.md").write_text("\n".join(lines),encoding="utf-8")


def write_nodes(config: dict[str,Any], props: dict[str,BodyProperties]) -> None:
    path=ROOT/"02_Спецификация"/"Спецификация_узлов.csv"
    fields=["node_id","node_name","parent","representation","dimensions_or_geometry","mass_kg","com_local_m","inertia_local_kg_m2_Ixx_Iyy_Izz_Ixy_Ixz_Iyz","joint_or_fixed","limits","collision_notes","origin_and_source"]
    joint={"link1":"j1_shoulder hinge","link2":"j2_elbow hinge","z_slide":"j3_lift slide","wrist":"j4_wrist hinge","finger_left":"j5_finger_left slide","finger_right":"j6_finger_right slide + equality"}
    limits={"link1":"J1 ±π rad","link2":"J2 ±150 deg","z_slide":"J3 0..0.097 m","wrist":"J4 ±π rad","finger_left":"q 0..0.027 m","finger_right":"q 0..0.027 m; q_right=q_left"}
    parents={"link1":"WORLD","link2":"link1","z_slide":"link2","wrist":"z_slide","finger_left":"wrist","finger_right":"wrist"}
    rows=[]
    for name in ("link1","link2","z_slide","wrist","finger_left","finger_right"):
        body=props[name]
        parts="; ".join(f"{c['name']}:{c['shape']}" for c in config["assemblies"][name]["components"])
        i=body.inertia_kg_m2
        full=[i[0,0],i[1,1],i[2,2],i[0,1],i[0,2],i[1,2]]
        rows.append({"node_id":name.upper(),"node_name":name,"parent":parents[name],"representation":"parameterized MJCF rigid body",
            "dimensions_or_geometry":parts,"mass_kg":f"{body.mass_kg:.9f}","com_local_m":json.dumps([round(float(v),9) for v in body.com_m]),
            "inertia_local_kg_m2_Ixx_Iyy_Izz_Ixy_Ixz_Iyz":json.dumps([round(float(v),12) for v in full]),"joint_or_fixed":joint[name],
            "limits":limits[name],"collision_notes":"Separate visual and collision geoms; adjacent mechanical pair exclusions explicit",
            "origin_and_source":"calculated from SSOT primitive geometry and PLA density; simplified virtual model"})
    density=float(parameter(config,"environment.density_pla_kg_m3")); dims=np.asarray(parameter(config,"object.size_xyz_m"),dtype=float)
    mass=density*float(np.prod(dims)); inertia=[mass*(dims[1]**2+dims[2]**2)/12,mass*(dims[0]**2+dims[2]**2)/12,mass*(dims[0]**2+dims[1]**2)/12,0,0,0]
    for label in ("RED","GREEN","BLUE"):
        for row in (1,2):
            rows.append({"node_id":f"OBJ-{label}-{row:02d}","node_name":f"object_{label.lower()}_{row:02d}","parent":"WORLD freejoint",
                "representation":"free rigid rectangular cuboid","dimensions_or_geometry":json.dumps([float(v) for v in dims]),"mass_kg":f"{mass:.9f}",
                "com_local_m":"[0,0,0]","inertia_local_kg_m2_Ixx_Iyy_Izz_Ixy_Ixz_Iyz":json.dumps([round(float(v),12) for v in inertia]),
                "joint_or_fixed":"freejoint","limits":"free; destination checked by evaluator","collision_notes":"Object/table, object/tray and object/finger contacts remain active",
                "origin_and_source":"calculated from solid PLA cuboid and TDS density; colors affect rendering only"})
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def write_sources() -> None:
    path=ROOT/"02_Спецификация"/"Источники.csv"
    rows=[
        ["SRC-M1-01","Архитектура робота-манипулятора","Project decision","Local project","02_Спецификация/Архитектура.md","2026-09-24","SCARA R-R-P-R, 4 arm coordinates + 1 gripper coordinate","M1 baseline"],
        ["SRC-MJ-01","MuJoCo XML Reference","Official documentation","DeepMind / MuJoCo","https://mujoco.readthedocs.io/en/latest/XMLreference.html","2026-09-24","MJCF inertial fullinertia order, equality joint, touch sensor, camera FOV, position actuator attributes","Software semantics; pinned engine version is M1"],
        ["SRC-MJ-02","MuJoCo Modeling","Official documentation","DeepMind / MuJoCo","https://mujoco.readthedocs.io/en/latest/modeling.html","2026-09-24","Kinematic tree, bodies, joints and model compilation",""],
        ["SRC-MAT-01","Prusament PLA Technical Data Sheet","Manufacturer data sheet","Prusa Polymers / Prusament","https://prusament.com/wp-content/uploads/2022/10/PLA_Prusament_TDS_2021_10_EN.pdf","2026-09-24","Density 1.24 g/cm^3 (ISO 1183), converted to 1240 kg/m^3","Used only as nominal solid-material density; printed infill is not modeled"],
        ["SRC-NIST-01","NIST Guide to the SI, Appendix B.9","National measurement institute","NIST","https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b9","2026-09-24","Standard acceleration of free fall 9.80665 m/s^2",""],
        ["SRC-CV-01","OpenCV 4.12 Perspective-n-Point pose computation","Official library documentation","OpenCV","https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html","2026-09-24","Camera axes convention X right, Y down, Z forward",""],
    ]
    with path.open("w",newline="",encoding="utf-8-sig") as stream:
        writer=csv.writer(stream); writer.writerow(["source_id","title","source_type","publisher_or_origin","url_or_project_path","accessed_or_baseline_date","supports","notes"]); writer.writerows(rows)


def write_calculations(config: dict[str,Any], props: dict[str,BodyProperties]) -> None:
    p=lambda key:parameter(config,key)
    l1,l2=float(p("robot.link1_length_m")),float(p("robot.link2_length_m"))
    rmin,rmax=p("robot.radial_reach_min_max_m")
    object_mass=float(p("object.mass_kg")); mu=float(p("gripper.required_pair_friction")); sf=float(p("gripper.safety_factor"))
    g=float(p("environment.gravity_m_s2")); az=float(p("gripper.payload_vertical_acceleration_m_s2"))
    fneed=float(p("gripper.minimum_normal_force_per_finger_N")); fcap=float(p("robot.gripper_contact_force_per_finger_N"))
    moving=float(p("gripper.z_axis_moving_mass_including_payload_kg")); zload=moving*(g+az); zcap=float(p("gripper.z_axis_force_limit_N"))
    span=float(p("robot.gripper_worst_yaw_object_span_m")); clearance=float(p("robot.gripper_worst_yaw_half_clearance_m"))
    fx,fy,cx,cy=p("camera.intrinsics_fx_fy_cx_cy_px"); covw,covh=p("camera.coverage_width_height_m")
    positions=[]; source=p("cell.input_center_xy_m")
    for dy in p("cell.input_row_y_offsets_m"):
        for dx in p("cell.input_slot_x_offsets_m"): positions.append((source[0]+dx,source[1]+dy,"input"))
    for color,center in p("cell.tray_centers_xy_m").items():
        for dx in p("cell.tray_slot_x_offsets_m"): positions.append((center[0]+dx,center[1]+p("cell.tray_slot_y_offset_m"),color))
    radii=[math.hypot(x,y) for x,y,_ in positions]
    budget=sum(float(p(k)) for k in ("robot.grasp_xy_budget_perception_m","robot.grasp_xy_budget_kinematics_m","robot.grasp_xy_budget_tracking_m","robot.grasp_xy_budget_reserve_m"))
    lines=[
        "# Геометрические и физические расчеты M2","", f"**Вход:** `параметры_системы.yaml`, конфигурация {config['config_version']}; геометрическая база {config['project'].get('geometry_baseline_version', 'M2-v1.0')}. Этот расчет — проектная виртуальная модель, не измерение изделия.","",
        "## Предварительная радиальная проверка","",
        "Для плоского двухзвенного SCARA используется только геометрическая граница `r_max=L1+L2`, `r_min=sqrt(L1²+L2²+2 L1 L2 cos(q2_max))`. Она не заменяет IK.",
        f"L1={l1:.3f} m, L2={l2:.3f} m, q2 limit={math.degrees(max(abs(float(v)) for v in p('robot.j2_range_rad'))):.1f}°; радиальная полоса [{rmin:.4f}, {rmax:.4f}] m. Центры входных и выходных slots: r={min(radii):.4f}…{max(radii):.4f} m.",
        "Эта проверка не оценивает IK branches, singularity, ориентацию TCP, self-collision или путь. Их владельцы — M3 и M5.","",
        "## Вертикальное положение","",
        "`z_TCP = z_J1 − |J4_z| − |TCP_z| − q3 = 0.190 − 0.075 − 0.012 − q3 = 0.103 − q3 m`.",
        f"Home q3={p('robot.j3_home_m'):.3f} m → zTCP={p('robot.tcp_home_z_m'):.3f} m. Transfer zTCP={p('robot.tcp_safe_transfer_z_m'):.3f} m. Pick zTCP={p('robot.object_pick_center_z_m'):.3f} m требует q3={p('robot.j3_pick_q_m'):.3f} m, верхний предел {p('robot.j3_range_m')[1]:.3f} m, запас {float(p('robot.j3_range_m')[1])-float(p('robot.j3_pick_q_m')):.3f} m. Place zTCP={p('robot.object_place_center_z_m'):.3f} m требует q3={p('robot.j3_place_q_m'):.3f} m.","",
        "## Захват и бюджет ошибки","",
        f"Для квадратного кубоида s={p('object.size_xyz_m')[0]*1000:.1f} mm худшая проекция на ось захвата при yaw=45° равна `s√2={span*1000:.2f} mm`. Пальцы 60 mm длиной и открытый зазор 60 mm дают минимальный геометрический полузапас `{clearance*1000:.2f} mm` по X/Y.",
        f"Общий радиальный XY-бюджет {budget*1000:.1f} mm: perception {float(p('robot.grasp_xy_budget_perception_m'))*1000:.1f} mm + кинематический/TCP {float(p('robot.grasp_xy_budget_kinematics_m'))*1000:.1f} mm + tracking {float(p('robot.grasp_xy_budget_tracking_m'))*1000:.1f} mm + резерв {float(p('robot.grasp_xy_budget_reserve_m'))*1000:.1f} mm. Остаток от геометрического худшего запаса {max(0,clearance-budget)*1000:.2f} mm.",
        "Эти допуски пока назначены, а не измерены. M3/M4/M7 должны подтвердить точность; M7 — реальный контактный подъем и удержание.","",
        "## Массы, центры масс и инерции","",
        "Массы получены из сплошной плотности PLA 1240 kg/m³. Открытая прямоугольная труба считается внешней призмой за вычетом внутренней призмы той же длины; закрытая оболочка — вычитанием внутреннего тела; соединители и ладонь/пальцы — упрощенными сплошными деталями. В пределах каждого тела детали соприкасаются гранями без объемного перекрытия.",
        "Для призмы `Ixx=m(b²+c²)/12`, `Iyy=m(a²+c²)/12`, `Izz=m(a²+b²)/12`. Сборка компонентов использует взвешенный COM и теорему параллельных осей. Генератор проверяет положительную определенность и неравенство треугольника главных моментов.","",
        "| Тело | Масса, kg | COM local, m | Ixx | Iyy | Izz |","|---|---:|---|---:|---:|---:|"
    ]
    for name in ("link1","link2","z_slide","wrist","finger_left","finger_right"):
        b=props[name]
        lines.append(f"| {name} | {b.mass_kg:.6f} | `{[round(float(v),6) for v in b.com_m]}` | {b.inertia_kg_m2[0,0]:.6e} | {b.inertia_kg_m2[1,1]:.6e} | {b.inertia_kg_m2[2,2]:.6e} |")
    object_dims=p("object.size_xyz_m")
    lines += [
        f"| payload | {object_mass:.6f} | `[0,0,0]` | {object_mass*(object_dims[1]**2+object_dims[2]**2)/12:.6e} | {object_mass*(object_dims[0]**2+object_dims[2]**2)/12:.6e} | {object_mass*(object_dims[0]**2+object_dims[1]**2)/12:.6e} |","",
        f"Объект 28×28×16 mm: `m=ρV=1240×0.028×0.028×0.016={object_mass:.8f} kg` ({object_mass*1000:.3f} g). Плотность материала не учитывает печатный infill, пористость и анизотропию.",
        f"Z-axis moving mass с payload: {moving:.6f} kg. При g={g:.5f} m/s² и aZ={az:.3f} m/s²: `F=m(g+a)={zload:.3f} N`. Виртуальный force cap {zcap:.2f} N, отношение {zcap/zload:.2f}×. Направляющее трение и контактные импульсы отдельно не включены.","",
        "## Сила захвата","",
        "Для двух равных боковых контактов `2 μ N ≥ SF·m(g+aZ)`, где N — нормальная сила на каждый палец.",
        f"μ={mu:.2f}, SF={sf:.1f}, m={object_mass:.6f} kg, aZ={az:.3f} m/s² → Nmin={fneed:.3f} N/палец. Обобщенный effort одной координаты равен 1.20 N, что соответствует до {fcap:.2f} N/палец при симметричной связи. Запас модели {fcap/fneed:.2f}×. Это идеализированная расчетная граница, не измеренная сила.","",
        "## Камера","",
        f"Параметры pinhole: {p('camera.resolution_px')[0]}×{p('camera.resolution_px')[1]} px, vertical FOV {p('camera.fovy_deg'):.1f}°, высота {p('camera.position_world_m')[2]:.3f} m. `fx=fy=H/(2tan(FOV/2))={fx:.2f} px`; центр `{cx:.1f},{cy:.1f} px`. Геометрическое покрытие стола на z=0: {covw:.3f}×{covh:.3f} m, размеры столешницы {p('cell.table_size_xy_m')[0]:.3f}×{p('cell.table_size_xy_m')[1]:.3f} m.",
        "Покрытие вычислено для идеальной камеры и не учитывает перекрытие рукой, фотометрию и calibration residual. M4 владеет калибровочной проверкой.","",
        "## Контакт и границы модели","",
        f"Начальные assumptions: μtable={p('contact.friction_table_slide')}, μfinger/object={p('gripper.friction_slide')}, condim={p('contact.condim')}, solref={p('contact.solref')}, solimp={p('contact.solimp')}. Отдельный измеренный restitution coefficient отсутствует: близко неупругое поведение является качеством выбранного damping/solver response, а не измеренным свойством PLA.",
        "Inertia упрощена: без крепежа, проводки, приводов и гибкости; расчет массы годится для учебной динамики и требует проверки чувствительности в M7. Физических измерений не проводилось.",""
    ]
    (ROOT/"02_Спецификация"/"Расчеты_геометрии_и_нагрузок.md").write_text("\n".join(lines),encoding="utf-8")


def generate() -> dict[str,Any]:
    SOURCE_DIR.mkdir(parents=True,exist_ok=True); RESULT_DIR.mkdir(parents=True,exist_ok=True)
    config=load_ssot(); props=derived_properties(config)
    SSOT_PATH.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False,default_flow_style=False),encoding="utf-8")
    root=make_world(config,props)
    ET.indent(root,space="  ")
    model_path=SOURCE_DIR/"scara_color_sorter_m2.xml"
    # Writing serialized bytes avoids Windows text-mode newline translation and keeps hashes reproducible.
    model_path.write_bytes(ET.tostring(root,encoding="utf-8",xml_declaration=True))
    write_registry(config); write_nodes(config,props); write_calculations(config,props); write_sources()
    return {"config_version":config["config_version"],"model_path":str(model_path),
            "model_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "body_mass_kg":{key:round(value.mass_kg,9) for key,value in props.items()}}


if __name__=="__main__":
    try:
        print(json.dumps(generate(),ensure_ascii=False,indent=2))
    except Exception as exc:
        print(f"M2 model generation failed: {exc}",file=sys.stderr)
        raise
