"""Reproducible M3 FK/IK, workspace, Jacobian and MuJoCo parity verification."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import platform
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from scara import (
    DEFAULT_SSOT,
    IKStatus,
    Pose,
    classify_singularity,
    forward_kinematics,
    inverse_kinematics,
    jacobian,
    load_config,
    periodic_difference,
    workspace_bounds,
    wrap_angle,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "05_Верификация" / "kinematics"
MODEL_PATH = ROOT / "03_Модель_и_схемы" / "source" / "scara_color_sorter_m2.xml"
SEED_ROUNDTRIP = 20260924
SEED_ENGINE = 71103
SEED_JACOBIAN = 52103
ROUNDTRIP_COUNT = 10_000
ENGINE_COUNT = 1_000
REGION_GRID_N = 31
WORKSPACE_Q1_N = 181
WORKSPACE_Q2_N = 121


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_parameter_values() -> dict:
    with DEFAULT_SSOT.open("r", encoding="utf-8-sig") as stream:
        document = yaml.safe_load(stream)
    return {key: item["value"] for key, item in document["parameters"].items()}


def read_geometry_baseline_version() -> str:
    with DEFAULT_SSOT.open("r", encoding="utf-8-sig") as stream:
        document = yaml.safe_load(stream)
    return str(document["project"]["geometry_baseline_version"])


def rectangle_radius_bounds(center_xy: list[float], size_xy: list[float]) -> tuple[float, float]:
    cx, cy = map(float, center_xy)
    hx, hy = (float(v) / 2.0 for v in size_xy)
    x_lo, x_hi = cx - hx, cx + hx
    y_lo, y_hi = cy - hy, cy + hy
    nearest_x = min(max(0.0, x_lo), x_hi)
    nearest_y = min(max(0.0, y_lo), y_hi)
    r_min = math.hypot(nearest_x, nearest_y)
    r_max = max(math.hypot(x, y) for x in (x_lo, x_hi) for y in (y_lo, y_hi))
    return r_min, r_max


def rotation_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    relative = a.T @ b
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    skew = np.asarray((relative[2, 1] - relative[1, 2],
                        relative[0, 2] - relative[2, 0],
                        relative[1, 0] - relative[0, 1]), dtype=float)
    sine = 0.5 * float(np.linalg.norm(skew))
    return math.atan2(sine, cosine)


def engine_parity(config, count: int, seed: int) -> tuple[list[dict], dict]:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    names = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")
    engine_body_by_frame = {"J1": "link1", "J2": "link2", "J3": "z_slide", "J4": "wrist"}
    body_ids = {frame: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                for frame, name in engine_body_by_frame.items()}
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    rows = []
    all_pos_errors: list[float] = []
    all_rot_errors: list[float] = []

    for sample_id in range(count):
        q = tuple(float(rng.uniform(lo, hi)) for lo, hi in config.q_limits)
        data.qpos[:] = model.qpos0
        for name, value in zip(names, q):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint_id]] = value
        mujoco.mj_forward(model, data)
        manual = forward_kinematics(q, config)
        per_frame_pos: dict[str, float] = {}
        per_frame_rot: dict[str, float] = {}
        for frame, body_id in body_ids.items():
            pos_error = float(np.linalg.norm(manual.frames_base[frame][:3, 3] - data.xpos[body_id]))
            rot_error = rotation_error_rad(manual.frames_base[frame][:3, :3], data.xmat[body_id].reshape(3, 3))
            per_frame_pos[frame] = pos_error
            per_frame_rot[frame] = rot_error
        tcp_pos_error = float(np.linalg.norm(manual.transform_base_tcp[:3, 3] - data.site_xpos[tcp_id]))
        tcp_rot_error = rotation_error_rad(manual.transform_base_tcp[:3, :3], data.site_xmat[tcp_id].reshape(3, 3))
        per_frame_pos["TCP"] = tcp_pos_error
        per_frame_rot["TCP"] = tcp_rot_error
        all_pos_errors.extend(per_frame_pos.values())
        all_rot_errors.extend(per_frame_rot.values())
        rows.append({
            "sample_id": sample_id,
            "q1_rad": q[0], "q2_rad": q[1], "q3_m": q[2], "q4_rad": q[3],
            "max_frame_position_error_m": max(per_frame_pos.values()),
            "max_frame_orientation_error_rad": max(per_frame_rot.values()),
            "tcp_position_error_m": tcp_pos_error,
            "tcp_orientation_error_rad": tcp_rot_error,
            **{f"{name.lower()}_position_error_m": value for name, value in per_frame_pos.items()},
        })
    summary = {
        "sample_count": count,
        "seed": seed,
        "compared_frames": ["J1", "J2", "J3", "J4", "TCP"],
        "max_position_error_m": max(all_pos_errors),
        "mean_position_error_m": float(np.mean(all_pos_errors)),
        "max_orientation_error_rad": max(all_rot_errors),
        "mean_orientation_error_rad": float(np.mean(all_rot_errors)),
    }
    return rows, summary


def task_area_verification(config, values: dict) -> tuple[list[dict], list[dict], dict]:
    bounds = workspace_bounds(config)
    inner, outer = bounds["joint_limited_r_min_m"], bounds["joint_limited_r_max_m"]
    regions = [("INPUT", values["cell.input_center_xy_m"], values["cell.input_size_xy_m"], values["robot.object_pick_center_z_m"])]
    for color, center in values["cell.tray_centers_xy_m"].items():
        regions.append((f"TRAY_{color}", center, values["cell.tray_outer_size_xy_m"], values["robot.object_place_center_z_m"]))

    region_rows = []
    sample_rows = []
    slot_rows = []
    max_position_residual = 0.0
    for region_id, center, size, z_m in regions:
        radial_min, radial_max = rectangle_radius_bounds(center, size)
        analytic_coverage = radial_min >= inner - 1e-12 and radial_max <= outer + 1e-12
        failures = 0
        checked = 0
        for ix, x in enumerate(np.linspace(center[0] - size[0] / 2, center[0] + size[0] / 2, REGION_GRID_N)):
            for iy, y in enumerate(np.linspace(center[1] - size[1] / 2, center[1] + size[1] / 2, REGION_GRID_N)):
                result = inverse_kinematics(Pose(float(x), float(y), float(z_m), 0.0), config)
                checked += 1
                success = result.selected is not None
                failures += int(not success)
                if success:
                    max_position_residual = max(max_position_residual, float(result.position_residual_m or 0.0))
                sample_rows.append({
                    "region_id": region_id, "grid_i": ix, "grid_j": iy,
                    "x_m": x, "y_m": y, "z_m": z_m, "yaw_rad": 0.0,
                    "ik_status": result.status.value, "selected_branch": result.selected.branch_id if success else "",
                    "q1_rad": result.selected.q[0] if success else "",
                    "q2_rad": result.selected.q[1] if success else "",
                    "q3_m": result.selected.q[2] if success else "",
                    "q4_rad": result.selected.q[3] if success else "",
                    "position_residual_m": result.position_residual_m if success else "",
                    "abs_sin_q2": result.selected.abs_sin_q2 if success else "",
                })
        region_rows.append({
            "region_id": region_id, "center_x_m": center[0], "center_y_m": center[1],
            "size_x_m": size[0], "size_y_m": size[1], "tcp_z_m": z_m,
            "radial_min_m": radial_min, "radial_max_m": radial_max,
            "joint_limited_r_min_m": inner, "joint_limited_r_max_m": outer,
            "analytic_rectangle_coverage": analytic_coverage,
            "grid_points_checked": checked, "grid_ik_failures": failures,
        })
        if not analytic_coverage or failures:
            raise AssertionError(f"Configured region {region_id} is not fully reachable: {region_rows[-1]}")

    input_center = values["cell.input_center_xy_m"]
    for row, dy in enumerate(values["cell.input_row_y_offsets_m"], start=1):
        for col, dx in enumerate(values["cell.input_slot_x_offsets_m"], start=1):
            color = ("RED", "GREEN", "BLUE")[col - 1]
            x, y = input_center[0] + dx, input_center[1] + dy
            slot_rows.append(_slot_result(f"INPUT_{color}_{row:02d}", "INPUT", color, x, y,
                                          values["robot.object_pick_center_z_m"], config))
    for color, center in values["cell.tray_centers_xy_m"].items():
        for slot, dx in enumerate(values["cell.tray_slot_x_offsets_m"], start=1):
            x, y = center[0] + dx, center[1] + values["cell.tray_slot_y_offset_m"]
            slot_rows.append(_slot_result(f"{color}_SLOT_{slot:02d}", f"TRAY_{color}", color, x, y,
                                          values["robot.object_place_center_z_m"], config))
    if any(row["ik_status"] not in {IKStatus.SUCCESS.value, IKStatus.NEAR_SINGULARITY.value} for row in slot_rows):
        raise AssertionError("At least one configured object or destination slot is unreachable")
    return region_rows, sample_rows, {"slots": slot_rows, "max_position_residual_m": max_position_residual}


def _slot_result(slot_id: str, region: str, color: str, x: float, y: float, z: float, config) -> dict:
    result = inverse_kinematics((x, y, z, 0.0), config)
    candidate = result.selected
    return {
        "slot_id": slot_id, "region_id": region, "class": color, "x_m": x, "y_m": y,
        "z_m": z, "target_yaw_rad": 0.0, "ik_status": result.status.value,
        "candidate_count": len(result.candidates), "branch_id": candidate.branch_id if candidate else "",
        "q1_rad": candidate.q[0] if candidate else "", "q2_rad": candidate.q[1] if candidate else "",
        "q3_m": candidate.q[2] if candidate else "", "q4_rad": candidate.q[3] if candidate else "",
        "position_residual_m": result.position_residual_m if candidate else "",
        "yaw_residual_rad": result.yaw_residual_rad if candidate else "",
        "abs_sin_q2": candidate.abs_sin_q2 if candidate else "",
    }


def make_manual_examples(config, values: dict) -> list[dict]:
    examples = [
        ("FK-01", "Нулевая плоская конфигурация", (0.0, 0.0, 0.015, 0.0)),
        ("FK-02", "Колено 90°, запястье компенсирует yaw", (0.0, math.pi / 2.0, 0.040, -math.pi / 2.0)),
        ("FK-03", "Обзорная поза из SSOT", config.observation_q),
        ("FK-04", "Поворот плеча на 90°", (math.pi / 2.0, 0.0, 0.020, 0.0)),
    ]
    rows = []
    for example_id, description, q in examples:
        fk = forward_kinematics(q, config)
        ik = inverse_kinematics(fk.pose, config, current_q=q)
        rows.append({
            "example_id": example_id, "description": description,
            "q1_rad": q[0], "q2_rad": q[1], "q3_m": q[2], "q4_rad": q[3],
            "target_x_m": fk.pose.x_m, "target_y_m": fk.pose.y_m, "target_z_m": fk.pose.z_m,
            "target_yaw_rad": fk.pose.yaw_rad,
            "ik_status": ik.status.value,
            "selected_branch": ik.selected.branch_id if ik.selected else "",
            "ik_q1_rad": ik.selected.q[0] if ik.selected else "",
            "ik_q2_rad": ik.selected.q[1] if ik.selected else "",
            "ik_q3_m": ik.selected.q[2] if ik.selected else "",
            "ik_q4_rad": ik.selected.q[3] if ik.selected else "",
            "position_residual_m": ik.position_residual_m,
            "yaw_residual_rad": ik.yaw_residual_rad,
        })
    return rows


def jacobian_checks(config, count: int, seed: int) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(seed)
    rows = []
    all_errors = []
    step = 1e-7
    for sample_id in range(count):
        q = tuple(float(rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)))
                  for lo, hi in config.q_limits)
        analytic = jacobian(q, config)
        numeric = np.zeros((4, 4), dtype=float)
        for axis in range(4):
            plus, minus = list(q), list(q)
            plus[axis] += step
            minus[axis] -= step
            p_plus = forward_kinematics(plus, config).pose
            p_minus = forward_kinematics(minus, config).pose
            delta = p_plus.as_array() - p_minus.as_array()
            delta[3] = periodic_difference(p_plus.yaw_rad, p_minus.yaw_rad)
            numeric[:, axis] = delta / (2.0 * step)
        difference = np.abs(analytic - numeric)
        maximum = float(np.max(difference))
        all_errors.append(maximum)
        rows.append({
            "sample_id": sample_id, "q1_rad": q[0], "q2_rad": q[1], "q3_m": q[2], "q4_rad": q[3],
            "finite_difference_step": step, "max_abs_jacobian_component_error": maximum,
        })
    summary = {"sample_count": count, "seed": seed, "finite_difference_step": step,
               "max_abs_component_error": max(all_errors), "mean_max_abs_component_error": float(np.mean(all_errors))}
    return rows, summary


def singularity_sensitivity(config) -> list[dict]:
    q2_values = (0.0, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2, math.pi / 2.0, abs(config.j2_limits_rad[1]))
    rows = []
    for q2 in q2_values:
        diag = classify_singularity((0.0, q2, config.observation_q[2], 0.0), config)
        sigma = float(diag["sigma_min_xy_m_per_rad"])
        rows.append({
            "q2_rad": q2, "q2_deg": math.degrees(q2), "abs_sin_q2": diag["abs_sin_q2"],
            "sigma_min_xy_m_per_rad": sigma,
            "local_inverse_gain_rad_per_m": math.inf if sigma <= np.finfo(float).eps else 1.0 / sigma,
            "condition_number_xy": diag["condition_number_xy"],
            "classification": diag["classification"],
        })
    return rows


def workspace_samples(config, count1: int, count2: int) -> tuple[list[dict], list[dict]]:
    q1_values = np.linspace(config.j1_limits_rad[0], config.j1_limits_rad[1], count1, endpoint=False)
    q2_values = np.linspace(config.j2_limits_rad[0], config.j2_limits_rad[1], count2)
    rows = []
    for i, q1 in enumerate(q1_values):
        for j, q2 in enumerate(q2_values):
            q = (float(q1), float(q2), config.observation_q[2], 0.0)
            fk = forward_kinematics(q, config)
            diag = classify_singularity(q, config)
            rows.append({"grid_i": i, "grid_j": j, "q1_rad": q1, "q2_rad": q2,
                         "x_m": fk.pose.x_m, "y_m": fk.pose.y_m,
                         "abs_sin_q2": diag["abs_sin_q2"],
                         "condition_number_xy": diag["condition_number_xy"],
                         "classification": diag["classification"]})
    bounds = workspace_bounds(config)
    boundary_rows = []
    theta_values = np.linspace(-math.pi, math.pi, 361)
    q2_inner = math.copysign(max(abs(config.j2_limits_rad[0]), abs(config.j2_limits_rad[1])), config.j2_limits_rad[1])
    inner_offset = math.atan2(config.l2_m * math.sin(q2_inner), config.l1_m + config.l2_m * math.cos(q2_inner))
    for index, theta in enumerate(theta_values):
        for name, radius, q2, q1_shift in (
            ("OUTER_STRAIGHT", bounds["joint_limited_r_max_m"], 0.0, 0.0),
            ("INNER_JOINT_LIMIT", bounds["joint_limited_r_min_m"], q2_inner, -inner_offset),
        ):
            q1 = wrap_angle(float(theta + q1_shift))
            q = (q1, q2, config.observation_q[2], 0.0)
            fk = forward_kinematics(q, config)
            boundary_rows.append({"sample_id": index, "boundary_id": name, "theta_rad": theta,
                                 "q1_rad": q1, "q2_rad": q2,
                                 "x_m": fk.pose.x_m, "y_m": fk.pose.y_m,
                                 "expected_radius_m": radius,
                                 "actual_radius_m": math.hypot(fk.pose.x_m, fk.pose.y_m)})
    return rows, boundary_rows


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf")
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_workspace_plot(path_png: Path, path_svg: Path, config, values: dict, samples: list[dict]) -> None:
    width, height = 1600, 1120
    bg, grid, ink = "#ffffff", "#d5dde4", "#173042"
    left, top, plot_size = 70, 125, 900
    extent = 0.52
    scale = plot_size / (2.0 * extent)
    center_x = left + plot_size / 2.0
    center_y = top + plot_size / 2.0

    def xy(x: float, y: float) -> tuple[int, int]:
        return int(round(center_x + x * scale)), int(round(center_y - y * scale))

    bounds = workspace_bounds(config)
    r_outer = bounds["joint_limited_r_max_m"]
    r_inner = bounds["joint_limited_r_min_m"]
    r_geometry_inner = bounds["geometric_r_min_m"]
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)
    title_font, label_font, small_font = _font(34), _font(22), _font(18)
    draw.text((70, 34), "SCARA: расчетная XY-рабочая область", font=title_font, fill=ink)
    draw.text((70, 78), "Точки получены FK-сеткой; границы — аналитически из длин звеньев и физического диапазона J2.", font=small_font, fill="#526a7a")
    plot_box = (left, top, left + plot_size, top + plot_size)
    draw.rectangle(plot_box, fill="#f7fafc", outline=grid, width=2)
    outer_box = (center_x - r_outer * scale, center_y - r_outer * scale,
                 center_x + r_outer * scale, center_y + r_outer * scale)
    inner_box = (center_x - r_inner * scale, center_y - r_inner * scale,
                 center_x + r_inner * scale, center_y + r_inner * scale)
    draw.ellipse(outer_box, fill="#d9eaf4", outline="#286487", width=4)
    draw.ellipse(inner_box, fill="#f7fafc", outline="#bd684a", width=4)
    geom_inner_box = (center_x - r_geometry_inner * scale, center_y - r_geometry_inner * scale,
                      center_x + r_geometry_inner * scale, center_y + r_geometry_inner * scale)
    draw.ellipse(geom_inner_box, outline="#7d8790", width=2)
    # Cartesian grid and axes are redrawn over the annular shading.
    tick_values = np.arange(-0.5, 0.51, 0.1)
    for tick in tick_values:
        sx0, sy0 = xy(float(tick), -extent)
        sx1, sy1 = xy(float(tick), extent)
        draw.line((sx0, sy0, sx1, sy1), fill="#e6ebef", width=1)
        sx0, sy0 = xy(-extent, float(tick))
        sx1, sy1 = xy(extent, float(tick))
        draw.line((sx0, sy0, sx1, sy1), fill="#e6ebef", width=1)
        tx, ty = xy(float(tick), -0.49)
        draw.text((tx - 18, top + plot_size + 5), f"{tick:.1f}", font=small_font, fill="#556b78")
        tx, ty = xy(-0.50, float(tick))
        draw.text((left - 54, ty - 10), f"{tick:.1f}", font=small_font, fill="#556b78")
    x0, y0 = xy(-extent, 0.0)
    x1, y1 = xy(extent, 0.0)
    draw.line((x0, y0, x1, y1), fill="#263c4a", width=3)
    x0, y0 = xy(0.0, -extent)
    x1, y1 = xy(0.0, extent)
    draw.line((x0, y0, x1, y1), fill="#263c4a", width=3)

    table_x, table_y = values["cell.table_size_xy_m"]
    a, b = xy(-table_x / 2.0, -table_y / 2.0), xy(table_x / 2.0, table_y / 2.0)
    draw.rectangle((a[0], b[1], b[0], a[1]), outline="#8d9ca5", width=2)
    # Task footprints: positions are visible separately from full rectangles.
    input_center, input_size = values["cell.input_center_xy_m"], values["cell.input_size_xy_m"]
    a, b = xy(input_center[0] - input_size[0] / 2, input_center[1] - input_size[1] / 2), xy(input_center[0] + input_size[0] / 2, input_center[1] + input_size[1] / 2)
    draw.rectangle((a[0], b[1], b[0], a[1]), fill="#e8edf0", outline="#607d8b", width=3)
    color_map = {"RED": "#d65b5b", "GREEN": "#399a69", "BLUE": "#477bd1"}
    for color, center in values["cell.tray_centers_xy_m"].items():
        size = values["cell.tray_outer_size_xy_m"]
        a, b = xy(center[0] - size[0] / 2, center[1] - size[1] / 2), xy(center[0] + size[0] / 2, center[1] + size[1] / 2)
        draw.rectangle((a[0], b[1], b[0], a[1]), fill=color_map[color], outline="#29404b", width=2)
        for dx in values["cell.tray_slot_x_offsets_m"]:
            px, py = xy(center[0] + dx, center[1] + values["cell.tray_slot_y_offset_m"])
            draw.ellipse((px - 6, py - 6, px + 6, py + 6), fill="#ffffff", outline="#152b38", width=2)
        draw.text((a[0], b[1] - 26), color, font=small_font, fill=ink)
    for dy in values["cell.input_row_y_offsets_m"]:
        for dx in values["cell.input_slot_x_offsets_m"]:
            px, py = xy(input_center[0] + dx, input_center[1] + dy)
            draw.ellipse((px - 5, py - 5, px + 5, py + 5), fill="#566e7a", outline="#ffffff", width=1)
    bx, by = xy(0.0, 0.0)
    draw.rectangle((bx - 14, by - 14, bx + 14, by + 14), fill="#173042", outline="#ffffff", width=2)
    draw.text((bx + 20, by + 12), "BASE", font=small_font, fill=ink)
    draw.text((left + 360, top + plot_size + 40), "X, m", font=label_font, fill=ink)
    draw.text((20, top + 420), "Y, m", font=label_font, fill=ink)

    panel_x = 1040
    draw.text((panel_x, 145), "Границы и чтение графика", font=label_font, fill=ink)
    legend = [
        ("#d9eaf4", f"Допустимая XY-область с лимитами J2: r={r_inner:.4f}…{r_outer:.3f} m"),
        ("#bd684a", "Внутренняя граница вызвана пределом J2 = ±150°"),
        ("#7d8790", f"Пунктир: геометрическая разность звеньев r={r_geometry_inner:.3f} m"),
        ("#607d8b", "Входная область; точки — центры объектов"),
        ("#477bd1", "Зоны сортировки; белые точки — центры слотов"),
    ]
    y = 205
    for swatch, text in legend:
        draw.rectangle((panel_x, y, panel_x + 28, y + 22), fill=swatch, outline="#637784")
        draw.text((panel_x + 42, y - 1), text, font=small_font, fill=ink)
        y += 56
    draw.text((panel_x, 520), "Вертикальный ход TCP", font=label_font, fill=ink)
    draw.text((panel_x, 559), f"z = {bounds['tcp_z_min_m']:.3f}…{bounds['tcp_z_max_m']:.3f} m", font=small_font, fill=ink)
    draw.text((panel_x, 600), f"Объект: z={values['robot.object_pick_center_z_m']:.3f} m", font=small_font, fill=ink)
    draw.text((panel_x, 635), f"Укладка: z={values['robot.object_place_center_z_m']:.3f} m", font=small_font, fill=ink)
    draw.text((panel_x, 700), f"FK-сетка: {len(samples):,} конфигураций", font=small_font, fill=ink)
    draw.text((panel_x, 735), "Ориентация yaw достижима по q4;", font=small_font, fill=ink)
    draw.text((panel_x, 765), "график не является collision-free картой.", font=small_font, fill=ink)
    image.save(path_png, format="PNG", optimize=True)

    def svg_rect(x: float, y: float, w: float, h: float, fill: str, stroke: str, sw: int = 2, extra: str = "") -> str:
        p1 = xy(x, y)
        p2 = xy(x + w, y - h)
        return f'<rect x="{p1[0]}" y="{p1[1]}" width="{p2[0]-p1[0]}" height="{p2[1]-p1[1]}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>'

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{bg}"/>',
        f'<text x="70" y="54" font-family="Arial,sans-serif" font-size="34" font-weight="700" fill="{ink}">SCARA: расчетная XY-рабочая область</text>',
        '<text x="70" y="94" font-family="Arial,sans-serif" font-size="18" fill="#526a7a">Точки получены FK-сеткой; границы — аналитически из длин звеньев и физического диапазона J2.</text>',
        f'<rect x="{left}" y="{top}" width="{plot_size}" height="{plot_size}" fill="#f7fafc" stroke="{grid}" stroke-width="2"/>',
        f'<circle cx="{center_x}" cy="{center_y}" r="{r_outer*scale}" fill="#d9eaf4" stroke="#286487" stroke-width="4"/>',
        f'<circle cx="{center_x}" cy="{center_y}" r="{r_inner*scale}" fill="#f7fafc" stroke="#bd684a" stroke-width="4"/>',
        f'<circle cx="{center_x}" cy="{center_y}" r="{r_geometry_inner*scale}" fill="none" stroke="#7d8790" stroke-width="2" stroke-dasharray="6 5"/>',
        svg_rect(-table_x/2, table_y/2, table_x, table_y, "none", "#8d9ca5", 2),
        svg_rect(input_center[0]-input_size[0]/2, input_center[1]+input_size[1]/2, input_size[0], input_size[1], "#e8edf0", "#607d8b", 3),
    ]
    for color, center in values["cell.tray_centers_xy_m"].items():
        size = values["cell.tray_outer_size_xy_m"]
        elements.append(svg_rect(center[0]-size[0]/2, center[1]+size[1]/2, size[0], size[1], color_map[color], "#29404b", 2))
        for dx in values["cell.tray_slot_x_offsets_m"]:
            px, py = xy(center[0]+dx, center[1]+values["cell.tray_slot_y_offset_m"])
            elements.append(f'<circle cx="{px}" cy="{py}" r="6" fill="#fff" stroke="#152b38" stroke-width="2"/>')
        a, b = xy(center[0]-size[0]/2, center[1]+size[1]/2)
        elements.append(f'<text x="{a}" y="{b-10}" font-family="Arial,sans-serif" font-size="18" fill="{ink}">{color}</text>')
    for dy in values["cell.input_row_y_offsets_m"]:
        for dx in values["cell.input_slot_x_offsets_m"]:
            px, py = xy(input_center[0]+dx,input_center[1]+dy)
            elements.append(f'<circle cx="{px}" cy="{py}" r="5" fill="#566e7a" stroke="#fff"/>')
    bx, by = xy(0.0,0.0)
    elements += [f'<rect x="{bx-14}" y="{by-14}" width="28" height="28" fill="#173042" stroke="#fff" stroke-width="2"/>',
                 f'<text x="{bx+20}" y="{by+30}" font-family="Arial,sans-serif" font-size="18" fill="{ink}">BASE</text>',
                 '<text x="405" y="1090" font-family="Arial,sans-serif" font-size="22" fill="#173042">X, m</text>',
                 '<text x="20" y="560" font-family="Arial,sans-serif" font-size="22" fill="#173042">Y, m</text>',
                 '<text x="1040" y="160" font-family="Arial,sans-serif" font-size="24" font-weight="700" fill="#173042">Границы и чтение графика</text>',
                 f'<text x="1040" y="215" font-family="Arial,sans-serif" font-size="19" fill="#173042">Допустимая область: r={r_inner:.4f}…{r_outer:.3f} m</text>',
                 '<text x="1040" y="260" font-family="Arial,sans-serif" font-size="19" fill="#173042">Внутренняя граница задается лимитом J2 ±150°</text>',
                 f'<text x="1040" y="305" font-family="Arial,sans-serif" font-size="19" fill="#173042">Теоретическая геометрия: r_min={r_geometry_inner:.3f} m</text>',
                 '<text x="1040" y="375" font-family="Arial,sans-serif" font-size="19" fill="#173042">Вход: серый прямоугольник; точки — объекты</text>',
                 '<text x="1040" y="420" font-family="Arial,sans-serif" font-size="19" fill="#173042">Лотки: цветные прямоугольники; точки — слоты</text>',
                 f'<text x="1040" y="540" font-family="Arial,sans-serif" font-size="22" fill="#173042">TCP z={bounds["tcp_z_min_m"]:.3f}…{bounds["tcp_z_max_m"]:.3f} m</text>',
                 f'<text x="1040" y="585" font-family="Arial,sans-serif" font-size="19" fill="#173042">Pick z={values["robot.object_pick_center_z_m"]:.3f} m; place z={values["robot.object_place_center_z_m"]:.3f} m</text>',
                 f'<text x="1040" y="670" font-family="Arial,sans-serif" font-size="19" fill="#173042">FK-сетка: {len(samples):,} конфигураций</text>',
                 '<text x="1040" y="715" font-family="Arial,sans-serif" font-size="19" fill="#173042">Ориентация достижима по q4; это не карта коллизий.</text>',
                 '</svg>']
    path_svg.write_text("\n".join(elements), encoding="utf-8")


def draw_error_plot(path_png: Path, path_svg: Path, position_errors: np.ndarray, yaw_errors: np.ndarray,
                    pos_limit: float, yaw_limit: float) -> None:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(image)
    title, label, small = _font(32), _font(22), _font(17)
    ink = "#173042"
    draw.text((70, 34), "Остатки FK → IK → FK, 10 000 воспроизводимых поз", font=title, fill=ink)
    panels = [(80, 150, 690, 650, position_errors, pos_limit, "Позиционный остаток, m"),
              (830, 150, 690, 650, yaw_errors, yaw_limit, "Остаток yaw, rad")]
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#fff"/>',
           '<text x="70" y="70" font-family="Arial,sans-serif" font-size="32" font-weight="700" fill="#173042">Остатки FK → IK → FK, 10 000 воспроизводимых поз</text>']
    for x, y, w, h, errors, limit, panel_title in panels:
        draw.rectangle((x, y, x+w, y+h), outline="#8095a2", width=2)
        draw.text((x+10, y-38), panel_title, font=label, fill=ink)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#8095a2" stroke-width="2"/>')
        logs = np.log10(np.maximum(errors, 1e-18))
        ymin, ymax = -18.0, -5.0
        for tick in range(-18, -4, 2):
            py = y+h-(tick-ymin)/(ymax-ymin)*h
            draw.line((x, py, x+w, py), fill="#e4e9ed", width=1)
            draw.text((x-48, py-9), f"1e{tick}", font=small, fill="#586e7b")
            svg.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x+w}" y2="{py:.1f}" stroke="#e4e9ed"/>')
        pts = []
        step = max(1, len(logs)//1800)
        for idx in range(0, len(logs), step):
            px = x + idx / max(1, len(logs)-1) * w
            clipped = float(np.clip(logs[idx], ymin, ymax))
            py = y+h-(clipped-ymin)/(ymax-ymin)*h
            pts.append((px, py))
        if len(pts) > 1:
            draw.line(pts, fill="#2c7da0", width=2)
            svg_points = " ".join(f"{px:.1f},{py:.1f}" for px,py in pts)
            svg.append(f'<polyline points="{svg_points}" fill="none" stroke="#2c7da0" stroke-width="2"/>')
        limit_log = math.log10(limit)
        limit_y = y+h-(limit_log-ymin)/(ymax-ymin)*h
        draw.line((x, limit_y, x+w, limit_y), fill="#c94b4b", width=3)
        draw.text((x+12, limit_y-27), f"допуск {limit:.0e}", font=small, fill="#a43838")
        svg.append(f'<line x1="{x}" y1="{limit_y:.1f}" x2="{x+w}" y2="{limit_y:.1f}" stroke="#c94b4b" stroke-width="3"/>')
        svg.append(f'<text x="{x+12}" y="{limit_y-10:.1f}" font-family="Arial,sans-serif" font-size="17" fill="#a43838">допуск {limit:.0e}</text>')
        draw.text((x+5, y+h+12), "1", font=small, fill="#586e7b")
        draw.text((x+w-85, y+h+12), f"{len(errors):,}", font=small, fill="#586e7b")
        svg.append(f'<text x="{x+w/2-30}" y="{y+h+45}" font-family="Arial,sans-serif" font-size="17" fill="#586e7b">номер выборки</text>')
    draw.text((80, 840), "Линия допуска зафиксирована в SSOT до формирования основной выборки; near-singularity отдельно размечается и не скрывается в остатках.", font=small, fill="#526a7a")
    svg += ['<text x="80" y="850" font-family="Arial,sans-serif" font-size="17" fill="#526a7a">Допуски зафиксированы в SSOT до основной выборки; near-singularity отдельно маркируется.</text>', '</svg>']
    image.save(path_png, format="PNG", optimize=True)
    path_svg.write_text("\n".join(svg), encoding="utf-8")


def run() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config(DEFAULT_SSOT)
    values = read_parameter_values()
    bounds = workspace_bounds(config)
    rng = np.random.default_rng(SEED_ROUNDTRIP)
    roundtrip_rows = []
    position_errors = []
    yaw_errors = []
    tool_yaw_errors = []
    branch_counts = {"ELBOW_POSITIVE": 0, "ELBOW_NEGATIVE": 0}
    near_singular_count = 0
    worst_index = -1
    worst_error = -1.0

    for sample_id in range(ROUNDTRIP_COUNT):
        q = tuple(float(rng.uniform(lo, hi)) for lo, hi in config.q_limits)
        fk = forward_kinematics(q, config)
        ik = inverse_kinematics(fk.pose, config, current_q=q, allow_object_symmetry=True)
        if ik.selected is None:
            raise AssertionError(f"Round-trip sample {sample_id} failed: {ik.status}: {ik.reason}; q={q}")
        fk2 = forward_kinematics(ik.selected.q, config)
        position_error = float(np.linalg.norm(fk2.transform_base_tcp[:3, 3] - fk.transform_base_tcp[:3, 3]))
        yaw_error = abs(periodic_difference(fk2.pose.yaw_rad, fk.pose.yaw_rad, config.symmetry_period_rad))
        tool_yaw_error = abs(periodic_difference(fk2.pose.yaw_rad, ik.selected.resolved_target_yaw_rad))
        if position_error > config.position_acceptance_m or yaw_error > config.yaw_acceptance_rad:
            raise AssertionError(f"Round-trip tolerance exceeded at {sample_id}: {position_error} m, {yaw_error} rad")
        position_errors.append(position_error)
        yaw_errors.append(yaw_error)
        tool_yaw_errors.append(tool_yaw_error)
        branch_counts[ik.selected.branch_id] = branch_counts.get(ik.selected.branch_id, 0) + 1
        singularity = classify_singularity(ik.selected.q, config)
        near_singular_count += int(bool(singularity["near_singular"]))
        if position_error > worst_error:
            worst_index, worst_error = sample_id, position_error
        roundtrip_rows.append({
            "sample_id": sample_id,
            "target_x_m": fk.pose.x_m, "target_y_m": fk.pose.y_m, "target_z_m": fk.pose.z_m,
            "target_yaw_rad": fk.pose.yaw_rad,
            "source_q1_rad": q[0], "source_q2_rad": q[1], "source_q3_m": q[2], "source_q4_rad": q[3],
            "selected_q1_rad": ik.selected.q[0], "selected_q2_rad": ik.selected.q[1],
            "selected_q3_m": ik.selected.q[2], "selected_q4_rad": ik.selected.q[3],
            "branch_id": ik.selected.branch_id, "symmetry_index": ik.selected.symmetry_index,
            "candidate_count": len(ik.candidates), "ik_status": ik.status.value,
            "position_error_m": position_error, "yaw_error_mod_object_symmetry_rad": yaw_error,
            "tool_yaw_error_rad": tool_yaw_error, "abs_sin_q2": ik.selected.abs_sin_q2,
            "condition_number_xy": ik.selected.condition_number_xy,
        })
    position_errors = np.asarray(position_errors)
    yaw_errors = np.asarray(yaw_errors)
    tool_yaw_errors = np.asarray(tool_yaw_errors)

    parity_rows, parity_summary = engine_parity(config, ENGINE_COUNT, SEED_ENGINE)
    if parity_summary["max_position_error_m"] > config.position_acceptance_m:
        raise AssertionError(f"Engine frame position parity exceeds tolerance: {parity_summary}")
    if parity_summary["max_orientation_error_rad"] > config.yaw_acceptance_rad:
        raise AssertionError(f"Engine frame orientation parity exceeds tolerance: {parity_summary}")

    jacobian_rows, jacobian_summary = jacobian_checks(config, 200, SEED_JACOBIAN)
    if jacobian_summary["max_abs_component_error"] > 2e-6:
        raise AssertionError(f"Jacobian finite-difference parity exceeds tolerance: {jacobian_summary}")

    region_rows, region_samples, slot_data = task_area_verification(config, values)
    manual_rows = make_manual_examples(config, values)
    sensitivity_rows = singularity_sensitivity(config)
    workspace_rows, boundary_rows = workspace_samples(config, WORKSPACE_Q1_N, WORKSPACE_Q2_N)

    write_csv(OUT / "round_trip_samples.csv", list(roundtrip_rows[0].keys()), roundtrip_rows)
    write_csv(OUT / "engine_parity.csv", list(parity_rows[0].keys()), parity_rows)
    write_csv(OUT / "task_slot_ik.csv", list(slot_data["slots"][0].keys()), slot_data["slots"])
    write_csv(OUT / "task_region_coverage.csv", list(region_rows[0].keys()), region_rows)
    write_csv(OUT / "task_region_grid_samples.csv", list(region_samples[0].keys()), region_samples)
    write_csv(OUT / "manual_examples.csv", list(manual_rows[0].keys()), manual_rows)
    write_csv(OUT / "jacobian_finite_difference.csv", list(jacobian_rows[0].keys()), jacobian_rows)
    write_csv(OUT / "singularity_sensitivity.csv", list(sensitivity_rows[0].keys()), sensitivity_rows)
    write_csv(OUT / "workspace_joint_samples.csv", list(workspace_rows[0].keys()), workspace_rows)
    write_csv(OUT / "workspace_boundary.csv", list(boundary_rows[0].keys()), boundary_rows)
    draw_workspace_plot(OUT / "workspace_xy.png", OUT / "workspace_xy.svg", config, values, workspace_rows)
    draw_error_plot(OUT / "kinematics_error_distribution.png", OUT / "kinematics_error_distribution.svg",
                    position_errors, yaw_errors, config.position_acceptance_m, config.yaw_acceptance_rad)

    singular_sensitivity = [row for row in sensitivity_rows if math.isclose(row["q2_rad"], 0.05, abs_tol=1e-12)][0]
    summary = {
        "status": "PASS",
        "configuration_version": config.ssot_version,
        "geometry_baseline_version": read_geometry_baseline_version(),
        "ssot_sha256": sha256(DEFAULT_SSOT),
        "mjcf_sha256": sha256(MODEL_PATH),
        "kinematics_source_sha256": sha256(Path(__file__).with_name("scara.py")),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "mujoco_version": mujoco.__version__,
        "joint_vector_order": ["q1_rad", "q2_rad", "q3_m", "q4_rad"],
        "fixed_tolerances": {
            "position_acceptance_m": config.position_acceptance_m,
            "yaw_acceptance_rad": config.yaw_acceptance_rad,
            "numeric_cos_tolerance": config.numeric_cos_tolerance,
            "near_singular_abs_sin_q2": config.near_singular_abs_sin_q2,
            "jacobian_finite_difference_max_abs_component": 2e-6,
        },
        "round_trip": {
            "sample_count": ROUNDTRIP_COUNT,
            "seed": SEED_ROUNDTRIP,
            "max_position_error_m": float(np.max(position_errors)),
            "mean_position_error_m": float(np.mean(position_errors)),
            "median_position_error_m": float(np.median(position_errors)),
            "worst_sample_id": worst_index,
            "max_yaw_error_mod_object_symmetry_rad": float(np.max(yaw_errors)),
            "mean_yaw_error_mod_object_symmetry_rad": float(np.mean(yaw_errors)),
            "max_tool_yaw_error_rad": float(np.max(tool_yaw_errors)),
            "branch_counts": branch_counts,
            "near_singular_selected_count": near_singular_count,
        },
        "engine_parity": parity_summary,
        "jacobian_finite_difference": jacobian_summary,
        "workspace": {
            **bounds,
            "q1_samples": WORKSPACE_Q1_N,
            "q2_samples": WORKSPACE_Q2_N,
            "joint_grid_sample_count": len(workspace_rows),
            "task_regions": region_rows,
            "task_region_grid_total": len(region_samples),
            "task_region_grid_failures": sum(row["grid_ik_failures"] for row in region_rows),
            "task_slot_count": len(slot_data["slots"]),
            "max_task_slot_position_residual_m": max(float(row["position_residual_m"]) for row in slot_data["slots"]),
            "collision_free": False,
        },
        "singularity_sensitivity_at_threshold": singular_sensitivity,
        "files": [
            "round_trip_samples.csv", "engine_parity.csv", "task_slot_ik.csv", "task_region_coverage.csv",
            "task_region_grid_samples.csv", "manual_examples.csv", "jacobian_finite_difference.csv",
            "singularity_sensitivity.csv", "workspace_joint_samples.csv", "workspace_boundary.csv",
            "workspace_xy.png", "workspace_xy.svg", "kinematics_error_distribution.png",
            "kinematics_error_distribution.svg",
        ],
        "note": "Kinematic reachability only. No collision checking, trajectory planning, camera calibration, dynamics, or sorting FSM is included in M3.",
    }
    (OUT / "m3_kinematics_verification.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
