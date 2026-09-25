"""Reproducible M5 path-planning and collision-preflight verification."""
from __future__ import annotations

import csv
import hashlib
import html
import importlib.metadata
import json
import math
import platform
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
PROGRAM = ROOT / "04_Программа"
if str(PROGRAM) not in sys.path:
    sys.path.insert(0, str(PROGRAM))

from perception.types import Detection, DetectionBatch
from planning.collision import CollisionScene
from planning.context import MODEL_PATH, SSOT_PATH, load_context
from planning.planner import Planner
from planning.records import PlanCode
from planning.sampling import compare_sampling_runs


OUT = ROOT / "05_Верификация" / "planning"
CASES = (
    {"case_id": "center_red_yaw0", "xy_estimate_m": [0.0, -0.25], "class_label": "RED",
     "yaw_estimate_rad": 0.0, "obstacles": [], "step_scale": 1.0,
     "expected": "SUCCESS", "purpose": "Central input, zero estimated yaw, red destination."},
    {"case_id": "left_green_yaw45", "xy_estimate_m": [-0.075, -0.285], "class_label": "GREEN",
     "yaw_estimate_rad": math.pi/4, "obstacles": [], "step_scale": 1.0,
     "expected": "SUCCESS", "purpose": "Left input region, square-object symmetry, green destination."},
    {"case_id": "right_blue_yaw30", "xy_estimate_m": [0.075, -0.215], "class_label": "BLUE",
     "yaw_estimate_rad": math.pi/6, "obstacles": [], "step_scale": 1.0,
     "expected": "SUCCESS", "purpose": "Right input region, nonzero yaw, blue destination."},
    {"case_id": "upper_edge_red_yaw90", "xy_estimate_m": [-0.10, -0.18], "class_label": "RED",
     "yaw_estimate_rad": math.pi/2, "obstacles": [], "step_scale": 1.0,
     "expected": "SUCCESS", "purpose": "Near upper-left boundary of configured input region."},
    {"case_id": "center_green_two_obstacles", "xy_estimate_m": [0.0, -0.25], "class_label": "GREEN",
     "yaw_estimate_rad": 0.35, "obstacles": [[-0.085, -0.25], [0.085, -0.25]],
     "step_scale": 1.0, "expected": "SUCCESS",
     "purpose": "Two additional perceived objects remain in the input field while planning a green sort."},
    {"case_id": "near_obstacle_safe_reject", "xy_estimate_m": [0.0, -0.25], "class_label": "GREEN",
     "yaw_estimate_rad": 0.35, "obstacles": [[-0.065, -0.25], [0.065, -0.25]],
     "step_scale": 1.0, "expected": "COLLISION_GRASP",
     "purpose": "Non-target object encroaches on the finger clearance envelope; planner must refuse before pickup."},
    {"case_id": "center_red_halfstep", "xy_estimate_m": [0.0, -0.25], "class_label": "RED",
     "yaw_estimate_rad": 0.0, "obstacles": [], "step_scale": 0.5,
     "expected": "SUCCESS", "purpose": "Half-baseline sampling for convergence comparison with center_red_yaw0."},
)

COLORS = {"q1": "#2b6f9c", "q2": "#d66a2c", "q3": "#4b8b5a", "q4": "#795ca8",
          "gripper": "#b34b68", "clearance": "#167c80", "robot": "#175d7a",
          "target": "#be3d32", "obstacle": "#667085", "slot": "#23875a"}
PLOT_MAX_POINTS_PER_SERIES = 1400


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes() -> dict[str, str]:
    sources = (
        PROGRAM / "planning" / "context.py",
        PROGRAM / "planning" / "records.py",
        PROGRAM / "planning" / "collision.py",
        PROGRAM / "planning" / "sampling.py",
        PROGRAM / "planning" / "trajectory.py",
        PROGRAM / "planning" / "planner.py",
        PROGRAM / "planning" / "run_m5_verification.py",
        PROGRAM / "kinematics" / "scara.py",
        PROGRAM / "perception" / "types.py",
    )
    return {str(path.relative_to(ROOT)): sha256(path) for path in sources}


def environment_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for distribution in ("mujoco", "numpy", "PyYAML", "Pillow"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "NOT_INSTALLED"
    return versions


def write_json(path: Path, document) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                default=lambda value: value.value if hasattr(value, "value") else float(value)),
                    encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def tcp_kinematic_metrics(sample, context) -> dict[str, float]:
    """Compute FK chain-rule TCP rates and envelopes derived from configured joint limits."""
    q1, q2, _, _ = sample.q
    dq1, dq2, dq3, dq4 = sample.dq
    ddq1, ddq2, ddq3, ddq4 = sample.ddq
    l1, l2 = float(context.p("robot.link1_length_m")), float(context.p("robot.link2_length_m"))
    a, b = q1, q1 + q2
    w2, alpha2 = dq1 + dq2, ddq1 + ddq2
    vx = -l1*math.sin(a)*dq1 - l2*math.sin(b)*w2
    vy = l1*math.cos(a)*dq1 + l2*math.cos(b)*w2
    vz = -dq3
    ax = (-l1*math.sin(a)*ddq1 - l1*math.cos(a)*dq1*dq1
          -l2*math.sin(b)*alpha2 - l2*math.cos(b)*w2*w2)
    ay = (l1*math.cos(a)*ddq1 - l1*math.sin(a)*dq1*dq1
          +l2*math.cos(b)*alpha2 - l2*math.sin(b)*w2*w2)
    az = -ddq3
    v1, v2, v3, v4 = context.motion_velocity_limits
    a1, a2, a3, a4 = context.motion_acceleration_limits
    return {
        "tcp_linear_speed_m_s": math.sqrt(vx*vx+vy*vy+vz*vz),
        "tcp_linear_acceleration_m_s2": math.sqrt(ax*ax+ay*ay+az*az),
        "tcp_yaw_rate_rad_s": dq1+dq2+dq4,
        "tcp_yaw_acceleration_rad_s2": ddq1+ddq2+ddq4,
        "tcp_linear_speed_joint_limit_bound_m_s": l1*v1+l2*(v1+v2)+v3,
        "tcp_linear_acceleration_joint_limit_bound_m_s2": l1*(a1+v1*v1)+l2*(a1+a2+(v1+v2)**2)+a3,
        "tcp_yaw_rate_joint_limit_bound_rad_s": v1+v2+v4,
        "tcp_yaw_acceleration_joint_limit_bound_rad_s2": a1+a2+a4,
    }


def make_detection(track_id: int, color: str, xy: tuple[float, float], yaw: float,
                   frame_id: int, stamp: float = 10.0) -> Detection:
    """Build an explicit M4-interface estimate; no simulation ground truth is passed."""
    # M4 uses UNKNOWN as a status when geometry is usable but color is not.
    # Marking such an obstacle VALID violates the planner's public-input contract.
    status = "UNKNOWN" if color == "UNKNOWN" else "VALID"
    return Detection(track_id=track_id, class_label=color, xy_base_m=xy, yaw_base_rad=yaw,
                     confidence=0.99, position_sigma_m=0.001,
                     yaw_sigma_rad=math.radians(2.0), status=status, reason=None,
                     frame_id=frame_id, simulation_time_s=stamp, area_px=225.0,
                     bbox_xywh_px=(300, 200, 15, 15), center_uv_px=(307.5, 207.5),
                     frame_age_s=0.0,
                     diagnostics={"source": "M5 deterministic DetectionBatch contract fixture"})


def _font(size: int):
    for candidate in (r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def _plot_indices(x_values, y_values, max_points: int = PLOT_MAX_POINTS_PER_SERIES) -> list[int]:
    """Downsample a plotted series while preserving each bin's extrema and endpoints."""
    count = min(len(x_values), len(y_values))
    if count <= 2 * max_points:
        return list(range(count))
    selected = {0, count - 1}
    bins = max(1, (max_points - 2) // 4)
    for bin_index in range(bins):
        start = bin_index * count // bins
        end = (bin_index + 1) * count // bins
        if end <= start:
            continue
        local_min = min(range(start, end), key=lambda i: y_values[i])
        local_max = max(range(start, end), key=lambda i: y_values[i])
        selected.update((start, end - 1, local_min, local_max))
    return sorted(selected)


def _legend_layout(series: list[dict], font, left: int, right: int,
                   canvas_height: int, row_height: int = 26) -> list[tuple[dict, int, int]]:
    """Wrap legend entries upward so additional rows stay clear of the x label."""
    rows: list[list[tuple[dict, int]]] = [[]]
    row_widths = [0]
    for item in series:
        if item.get("annotation"):
            continue
        label_box = font.getbbox(item["label"])
        label_width = label_box[2] - label_box[0]
        item_width = 32 + 40 + label_width + 20
        if rows[-1] and left + row_widths[-1] + item_width > right:
            rows.append([])
            row_widths.append(0)
        rows[-1].append((item, item_width))
        row_widths[-1] += item_width
    if not rows[0]:
        return []
    lowest_row_y = canvas_height - 61
    first_row_y = lowest_row_y - (len(rows) - 1) * row_height
    positioned: list[tuple[dict, int, int]] = []
    for row_index, row in enumerate(rows):
        x = left
        y = first_row_y + row_index * row_height
        for item, item_width in row:
            positioned.append((item, x, y))
            x += item_width
    return positioned


def _plot_bottom_for_legend(legend_entries: list[tuple[dict, int, int]],
                            nominal_bottom: int = 635, row_height: int = 26) -> int:
    """Reserve one plot-height increment for every wrapped legend row."""
    row_count = len({y for _, _, y in legend_entries})
    return nominal_bottom - max(0, row_count - 1) * row_height


def save_line_plot(path_base: Path, title: str, x_label: str, y_label: str,
                   series: list[dict], *, reference_lines=(), equal_aspect=False,
                   start_x_at_zero=False) -> None:
    """Render compact, dependency-light SVG and PNG technical plots."""
    width, height = 1280, 760
    left, top, right = 118, 92, 1215
    small_font = _font(16)
    legend_entries = _legend_layout(series, small_font, left, right, height)
    bottom = _plot_bottom_for_legend(legend_entries)
    all_x = [float(x) for item in series for x in item["x"]]
    all_y = [float(y) for item in series for y in item["y"]]
    all_y.extend(float(line["value"]) for line in reference_lines)
    if not all_x or not all_y:
        return
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    if equal_aspect:
        span = max(xmax-xmin, ymax-ymin, 1e-9)
        cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
        xmin, xmax, ymin, ymax = cx-span/2, cx+span/2, cy-span/2, cy+span/2
    padx = max((xmax-xmin)*0.04, 1e-6)
    pady = max((ymax-ymin)*0.08, 1e-6)
    if start_x_at_zero and xmin >= -1e-12:
        xmin, xmax = 0.0, xmax+padx
    else:
        xmin, xmax = xmin-padx, xmax+padx
    ymin, ymax = ymin-pady, ymax+pady
    if math.isclose(xmin, xmax): xmax = xmin+1
    if math.isclose(ymin, ymax): ymax = ymin+1

    def xy(x, y):
        return left+(float(x)-xmin)/(xmax-xmin)*(right-left), bottom-(float(y)-ymin)/(ymax-ymin)*(bottom-top)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="#ffffff"/>',
           f'<text x="{left}" y="48" font-family="Arial,sans-serif" font-size="27" font-weight="700" fill="#17313d">{html.escape(title)}</text>']
    im = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(im)
    title_font, axis_font = _font(26), _font(19)
    draw.text((left, 24), title, font=title_font, fill="#17313d")
    for fraction in range(6):
        val = ymin+(ymax-ymin)*fraction/5
        px = bottom-(bottom-top)*fraction/5
        svg.append(f'<line x1="{left}" y1="{px:.2f}" x2="{right}" y2="{px:.2f}" stroke="#dce5e8" stroke-width="1"/>')
        svg.append(f'<text x="{left-12}" y="{px+6:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="16" fill="#506672">{val:.4g}</text>')
        draw.line((left, px, right, px), fill="#dce5e8", width=1)
        draw.text((left-12, px-8), f"{val:.4g}", font=small_font,
                  fill="#506672", anchor="ra")
    for fraction in range(6):
        val = xmin+(xmax-xmin)*fraction/5
        px = left+(right-left)*fraction/5
        svg.append(f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" y2="{bottom}" stroke="#edf1f2" stroke-width="1"/>')
        svg.append(f'<text x="{px:.2f}" y="{bottom+28}" text-anchor="middle" font-family="Arial,sans-serif" font-size="16" fill="#506672">{val:.4g}</text>')
        draw.line((px, top, px, bottom), fill="#edf1f2", width=1)
        draw.text((px-22, bottom+10), f"{val:.4g}", font=small_font, fill="#506672")
    for line in reference_lines:
        y = xy(xmin, line["value"])[1]
        stroke = line.get("color", "#b63e3e")
        svg.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" stroke="{stroke}" stroke-width="2" stroke-dasharray="8 6"/>')
        svg.append(f'<text x="{right-5}" y="{y-7:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="15" fill="{stroke}">{html.escape(line["label"])}</text>')
        draw.line((left, y, right, y), fill=stroke, width=2)
        draw.text((right-240, y-24), line["label"], font=small_font, fill=stroke)
    for item in series:
        indices = _plot_indices(item["x"], item["y"])
        points = [xy(item["x"][i], item["y"][i]) for i in indices]
        if len(points) > 1:
            svg_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            svg.append(f'<polyline points="{svg_points}" fill="none" stroke="{item["color"]}" stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"/>')
            draw.line(points, fill=item["color"], width=3)
        if item.get("markers"):
            for px, py in points:
                svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="6" fill="{item["color"]}"/>')
                draw.ellipse((px-5, py-5, px+5, py+5), fill=item["color"])
                if item.get("annotation"):
                    dx, dy = item.get("annotation_offset", (8, -16))
                    label = html.escape(item["annotation"])
                    svg.append(f'<text x="{px+dx:.2f}" y="{py+dy:.2f}" font-family="Arial,sans-serif" font-size="14" fill="{item["color"]}">{label}</text>')
                    draw.text((px+dx, py+dy-14), item["annotation"], font=small_font, fill=item["color"])
    svg.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#526a75" stroke-width="2"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#526a75" stroke-width="2"/>')
    svg.append(f'<text x="{(left+right)/2}" y="{height-22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="18" fill="#304c59">{html.escape(x_label)}</text>')
    svg.append(f'<text x="26" y="{(top+bottom)/2}" text-anchor="middle" transform="rotate(-90 26 {(top+bottom)/2})" font-family="Arial,sans-serif" font-size="18" fill="#304c59">{html.escape(y_label)}</text>')
    draw.text((int((left+right)/2)-150, height-34), x_label, font=axis_font, fill="#304c59")
    label_box = axis_font.getbbox(y_label)
    label_width = label_box[2] - label_box[0]
    label_height = label_box[3] - label_box[1]
    label_image = Image.new("RGBA", (label_width + 4, label_height + 4), (255, 255, 255, 0))
    ImageDraw.Draw(label_image).text((2-label_box[0], 2-label_box[1]), y_label,
                                     font=axis_font, fill="#304c59")
    rotated_label = label_image.rotate(90, expand=True)
    im.paste(rotated_label, (16, int((top+bottom-rotated_label.height)/2)), rotated_label)
    for item, legend_x, legend_y in legend_entries:
        svg.append(f'<line x1="{legend_x}" y1="{legend_y-5}" x2="{legend_x+32}" y2="{legend_y-5}" stroke="{item["color"]}" stroke-width="4"/>')
        svg.append(f'<text x="{legend_x+40}" y="{legend_y}" font-family="Arial,sans-serif" font-size="15" fill="#304c59">{html.escape(item["label"])}</text>')
        draw.line((legend_x, legend_y-5, legend_x+32, legend_y-5), fill=item["color"], width=4)
        draw.text((legend_x+40, legend_y-18), item["label"], font=small_font, fill="#304c59")
    svg.append('</svg>')
    path_base.parent.mkdir(parents=True, exist_ok=True)
    path_base.with_suffix(".svg").write_text("\n".join(svg), encoding="utf-8")
    im.save(path_base.with_suffix(".png"), format="PNG", optimize=True)


def write_trajectory(path: Path, plan, context) -> None:
    limits_v = (*context.motion_velocity_limits, float(context.p("robot.finger_velocity_m_s")))
    limits_a = (*context.motion_acceleration_limits, float(context.p("robot.finger_acceleration_m_s2")))
    rows = []
    for sample, clearance in zip(plan.samples, plan.clearance_profile):
        tcp_metrics = tcp_kinematic_metrics(sample, context)
        row = {"time_s": sample.time_s, "phase": sample.phase,
               "payload_mode": sample.payload_mode, "branch_id": sample.branch_id,
               "gripper_m": sample.gripper_m,
               "gripper_velocity_m_s": sample.gripper_velocity_m_s,
               "gripper_acceleration_m_s2": sample.gripper_acceleration_m_s2,
               "event": sample.event or "",
               "tcp_x_m": sample.tcp_xyzyaw[0], "tcp_y_m": sample.tcp_xyzyaw[1],
               "tcp_z_m": sample.tcp_xyzyaw[2], "tcp_yaw_rad": sample.tcp_xyzyaw[3],
               "narrowphase_minimum_m_capped": clearance.narrowphase_minimum_m,
               "conservative_clearance_lower_bound_m": clearance.conservative_lower_bound_m,
               "checked_pair_count": clearance.checked_pair_count,
               "narrowphase_pair_count": clearance.narrowphase_pair_count}
        row.update(tcp_metrics)
        for i, name in enumerate(("q1", "q2", "q3", "q4")):
            row[f"{name}_position"] = sample.q[i]
            row[f"{name}_velocity"] = sample.dq[i]
            row[f"{name}_acceleration"] = sample.ddq[i]
            row[f"{name}_velocity_ratio"] = sample.dq[i]/limits_v[i]
            row[f"{name}_acceleration_ratio"] = sample.ddq[i]/limits_a[i]
        row["gripper_velocity_ratio"] = sample.gripper_velocity_m_s/limits_v[4]
        row["gripper_acceleration_ratio"] = sample.gripper_acceleration_m_s2/limits_a[4]
        rows.append(row)
    write_csv(path, rows)


def create_plots(case_id: str, plan, context, detections) -> list[str]:
    root = OUT / "plots" / case_id
    t = [sample.time_s for sample in plan.samples]
    spans = context.arm.characteristic_spans
    q_series = []
    for i, name in enumerate(("q1", "q2", "q3", "q4")):
        q_series.append({"label": f"{name} / M3 range", "x": t,
                         "y": [s.q[i]/spans[i] for s in plan.samples], "color": COLORS[name]})
    files = []
    save_line_plot(root / "joint_position", "Нормированные координаты суставов во времени",
                   "Время t, с", "q / диапазон M3", q_series, start_x_at_zero=True)
    files.extend([str((root/"joint_position").with_suffix(ext).relative_to(OUT)) for ext in (".svg", ".png")])

    vlim = (*context.motion_velocity_limits, float(context.p("robot.finger_velocity_m_s")))
    alim = (*context.motion_acceleration_limits, float(context.p("robot.finger_acceleration_m_s2")))
    names = ("q1", "q2", "q3", "q4", "gripper")
    vseries = [{"label": name, "x": t,
                "y": [s.dq[i]/vlim[i] for s in plan.samples] if i < 4 else
                     [s.gripper_velocity_m_s/vlim[i] for s in plan.samples],
                "color": COLORS[name]} for i, name in enumerate(names)]
    save_line_plot(root / "velocity_ratio", "Относительная скорость суставов и захвата",
                   "Время t, с", "Скорость / допустимая скорость", vseries,
                   reference_lines=({"value": 1.0, "label": "верхний предел +1", "color": "#a83d3d"},
                                   {"value": -1.0, "label": "нижний предел -1", "color": "#a83d3d"}),
                   start_x_at_zero=True)
    files.extend([str((root/"velocity_ratio").with_suffix(ext).relative_to(OUT)) for ext in (".svg", ".png")])
    aseries = [{"label": name, "x": t,
                "y": [s.ddq[i]/alim[i] for s in plan.samples] if i < 4 else
                     [s.gripper_acceleration_m_s2/alim[i] for s in plan.samples],
                "color": COLORS[name]} for i, name in enumerate(names)]
    save_line_plot(root / "acceleration_ratio", "Относительное ускорение суставов и захвата",
                   "Время t, с", "Ускорение / допустимое ускорение", aseries,
                   reference_lines=({"value": 1.0, "label": "верхний предел +1", "color": "#a83d3d"},
                                   {"value": -1.0, "label": "нижний предел -1", "color": "#a83d3d"}),
                   start_x_at_zero=True)
    files.extend([str((root/"acceleration_ratio").with_suffix(ext).relative_to(OUT)) for ext in (".svg", ".png")])
    cseries = [{"label": "all-pair conservative lower bound", "x": t,
                "y": [c.conservative_lower_bound_m for c in plan.clearance_profile],
                "color": COLORS["clearance"]}]
    save_line_plot(root / "clearance_lower_bound", "Консервативная нижняя оценка зазора по траектории",
                   "Время t, с", "Нижняя оценка зазора, м", cseries,
                   reference_lines=({"value": 0.0, "label": "касание / проникновение", "color": "#a83d3d"},),
                   start_x_at_zero=True)
    files.extend([str((root/"clearance_lower_bound").with_suffix(ext).relative_to(OUT)) for ext in (".svg", ".png")])
    path_series = [{"label": "TCP path", "x": [s.tcp_xyzyaw[0] for s in plan.samples],
                    "y": [s.tcp_xyzyaw[1] for s in plan.samples], "color": COLORS["robot"]}]
    path_series.append({"label": "start", "x": [plan.samples[0].tcp_xyzyaw[0]],
                        "y": [plan.samples[0].tcp_xyzyaw[1]], "color": "#23875a", "markers": True})
    path_series.append({"label": "end", "x": [plan.samples[-1].tcp_xyzyaw[0]],
                        "y": [plan.samples[-1].tcp_xyzyaw[1]], "color": "#795ca8", "markers": True})
    phase_markers = {
        "PREGRASP": ("approach", "#2e7d32", (12, -30)),
        "DESCENT": ("grasp", "#c62828", (-48, 22)),
        "LIFT": ("lift", "#ef6c00", (-42, -42)),
        "TRANSFER": ("transfer", "#6a1b9a", (34, 22)),
        "PLACE_DESCENT": ("place", "#ad1457", (-48, -22)),
        "RETREAT": ("retreat", "#00838f", (14, 24)),
    }
    for phase, (label, color, offset) in phase_markers.items():
        sample = next((s for s in plan.samples if s.phase == phase), None)
        if sample is not None:
            path_series.append({"label": label, "x": [sample.tcp_xyzyaw[0]],
                                "y": [sample.tcp_xyzyaw[1]], "color": color, "markers": True,
                                "annotation": label, "annotation_offset": offset})
    path_series.append({"label": "target estimate", "x": [detections[0].xy_base_m[0]],
                        "y": [detections[0].xy_base_m[1]], "color": COLORS["target"], "markers": True})
    for item in detections[1:]:
        path_series.append({"label": f"perceived obstacle {item.track_id}", "x": [item.xy_base_m[0]],
                            "y": [item.xy_base_m[1]], "color": COLORS["obstacle"], "markers": True})
    slot = next(slot for slot in plan_clearance_slot(context, plan.target_class, plan.slot_id))
    path_series.append({"label": f"slot {plan.slot_id}", "x": [slot[0]], "y": [slot[1]],
                        "color": COLORS["slot"], "markers": True})
    save_line_plot(root / "tcp_path_xy", "Путь TCP с фазами цикла сортировки",
                   "X базы, м", "Y базы, м", path_series, equal_aspect=True)
    files.extend([str((root/"tcp_path_xy").with_suffix(ext).relative_to(OUT)) for ext in (".svg", ".png")])
    return files


def plan_clearance_slot(context, color: str, slot_id: str):
    index = int(slot_id.split(":")[1])
    center = context.p("cell.tray_centers_xy_m")[color]
    offset = context.p("cell.tray_slot_x_offsets_m")[index]
    yoff = context.p("cell.tray_slot_y_offset_m")
    return ((float(center[0])+float(offset), float(center[1])+float(yoff)),)


def _case_detections(case: dict, frame_id: int):
    timestamp = 10.0
    target = make_detection(1000+frame_id, case["class_label"],
                            tuple(case["xy_estimate_m"]), float(case["yaw_estimate_rad"]),
                            frame_id, timestamp)
    others = tuple(make_detection(2000+frame_id+i, "UNKNOWN", tuple(xy), 0.0, frame_id, timestamp)
                   for i, xy in enumerate(case["obstacles"]))
    return (target, *others)


def run_case(planner: Planner, context, case: dict, frame_id: int) -> dict:
    baseline_step = float(context.p("planning.path_sweep_sample_step_m"))
    step_m = baseline_step * float(case["step_scale"])
    resolved_case = {**case, "resolved_sample_step_m": step_m,
                     "baseline_sample_step_m": baseline_step}
    detections = _case_detections(case, frame_id)
    batch = DetectionBatch("OK", None, frame_id, 10.0, detections)
    start_q = tuple(context.arm.observation_q)
    available_slots = (f"{case['class_label']}:1",)
    start = time.perf_counter()
    result = planner.plan_cycle(batch, detections[0].track_id, start_q,
        float(context.p("robot.finger_home_m")), available_slots, 10.0,
        sample_step_override_m=step_m)
    elapsed = time.perf_counter()-start
    row = {"case_id": case["case_id"], "frame_id": frame_id,
           "expected": case["expected"], "result": result.code.value,
           "passed": result.code.value == case["expected"],
           "success": result.success, "failure_phase": result.phase or "",
           "wall_time_s": elapsed, "seed": "deterministic/no-randomness",
           "perception_frame_time_s": batch.simulation_time_s,
           "target_track_id": detections[0].track_id,
           "class_label_from_M4_contract": detections[0].class_label,
           "estimated_xy_x_m": detections[0].xy_base_m[0],
           "estimated_xy_y_m": detections[0].xy_base_m[1],
           "estimated_yaw_rad": detections[0].yaw_base_rad,
           "perceived_obstacle_count": len(detections)-1,
           "collision_sample_step_m": step_m,
           "plan_id": result.plan.plan_id if result.plan else "",
           "selected_slot_id": result.plan.slot_id if result.plan else "",
           "selected_ik_branch": result.plan.selected_ik_branch if result.plan else "",
           "waypoint_count": len(result.plan.waypoints) if result.plan else 0,
           "trajectory_sample_count": len(result.plan.samples) if result.plan else 0,
           "duration_s": result.plan.duration_s if result.plan else "",
           "tcp_path_length_m": result.plan.path_length_m if result.plan else "",
           "narrowphase_minimum_m_capped": result.plan.minimum_clearance_m if result.plan else "",
           "all_pair_clearance_lower_bound_m": result.plan.clearance_lower_bound_m if result.plan else "",
           "vertical_xy_error_max_m": result.plan.vertical_xy_error_max_m if result.plan else "",
           "vertical_yaw_error_max_rad": result.plan.vertical_yaw_error_max_rad if result.plan else "",
           "max_velocity_ratio": result.plan.maximum_abs_velocity_ratio if result.plan else "",
           "max_acceleration_ratio": result.plan.maximum_abs_acceleration_ratio if result.plan else "",
           "tcp_linear_speed_max_m_s": "",
           "tcp_linear_acceleration_max_m_s2": "",
           "tcp_yaw_rate_max_abs_rad_s": "",
           "tcp_yaw_acceleration_max_abs_rad_s2": "",
           "tcp_linear_speed_joint_limit_bound_m_s": "",
           "tcp_linear_acceleration_joint_limit_bound_m_s2": "",
           "reason": result.reason}
    case_dir = OUT / "cases" / case["case_id"]
    case_dir.mkdir(parents=True, exist_ok=True)
    result_payload = result.to_dict()
    if result_payload.get("plan") is not None:
        plan = result.plan
        plan_payload = result_payload["plan"]
        plan_payload["trajectory_sample_count"] = len(plan.samples)
        plan_payload["clearance_sample_count"] = len(plan.clearance_profile)
        # Full sampled data is preserved once, in the case's machine-readable CSV.
        # The JSON retains the plan definition and metadata without duplicating tens of MB.
        plan_payload.pop("samples", None)
        plan_payload.pop("clearance_profile", None)
    result_document = {
        "case": resolved_case, "environment": environment_versions(),
        "configuration_version": context.config_version, "ssot_sha256": context.ssot_sha256,
        "model_sha256": context.model_sha256, "planner_source_sha256": sha256(Path(__file__).with_name("planner.py")),
        "source_sha256": source_hashes(),
        "batch": {"status": batch.status, "frame_id": batch.frame_id,
                  "simulation_time_s": batch.simulation_time_s,
                  "detections": [d.__dict__ for d in detections]},
        "current_q": start_q, "current_gripper_m": context.p("robot.finger_home_m"),
        "available_slot_ids": available_slots, "collision_sample_step_m": step_m,
        "complete_sampled_trajectory_csv": str((OUT / "trajectories" / f"{case['case_id']}.csv").relative_to(ROOT)) if result.success else None,
        "wall_time_s": elapsed, "result": result_payload}
    if result.success:
        plan = result.plan
        tcp_metrics = [tcp_kinematic_metrics(sample, context) for sample in plan.samples]
        tcp_summary = {
            "max_tcp_linear_speed_m_s": max(item["tcp_linear_speed_m_s"] for item in tcp_metrics),
            "max_tcp_linear_acceleration_m_s2": max(item["tcp_linear_acceleration_m_s2"] for item in tcp_metrics),
            "max_abs_tcp_yaw_rate_rad_s": max(abs(item["tcp_yaw_rate_rad_s"]) for item in tcp_metrics),
            "max_abs_tcp_yaw_acceleration_rad_s2": max(abs(item["tcp_yaw_acceleration_rad_s2"]) for item in tcp_metrics),
            "tcp_linear_speed_joint_limit_bound_m_s": tcp_metrics[0]["tcp_linear_speed_joint_limit_bound_m_s"],
            "tcp_linear_acceleration_joint_limit_bound_m_s2": tcp_metrics[0]["tcp_linear_acceleration_joint_limit_bound_m_s2"],
            "tcp_yaw_rate_joint_limit_bound_rad_s": tcp_metrics[0]["tcp_yaw_rate_joint_limit_bound_rad_s"],
            "tcp_yaw_acceleration_joint_limit_bound_rad_s2": tcp_metrics[0]["tcp_yaw_acceleration_joint_limit_bound_rad_s2"],
            "method": "Analytic FK chain-rule derivatives; conservative envelopes derived from configured M2 joint velocity/acceleration limits and link lengths. These are kinematic bounds, not independent actuator caps.",
        }
        numeric_tolerance = 1e-9
        if any(item["tcp_linear_speed_m_s"] > item["tcp_linear_speed_joint_limit_bound_m_s"]+numeric_tolerance
               or item["tcp_linear_acceleration_m_s2"] > item["tcp_linear_acceleration_joint_limit_bound_m_s2"]+numeric_tolerance
               or abs(item["tcp_yaw_rate_rad_s"]) > item["tcp_yaw_rate_joint_limit_bound_rad_s"]+numeric_tolerance
               or abs(item["tcp_yaw_acceleration_rad_s2"]) > item["tcp_yaw_acceleration_joint_limit_bound_rad_s2"]+numeric_tolerance
               for item in tcp_metrics):
            raise AssertionError("FK-derived TCP velocity/acceleration exceeded the conservative M2 joint-limit envelope")
        row.update({
            "tcp_linear_speed_max_m_s": tcp_summary["max_tcp_linear_speed_m_s"],
            "tcp_linear_acceleration_max_m_s2": tcp_summary["max_tcp_linear_acceleration_m_s2"],
            "tcp_yaw_rate_max_abs_rad_s": tcp_summary["max_abs_tcp_yaw_rate_rad_s"],
            "tcp_yaw_acceleration_max_abs_rad_s2": tcp_summary["max_abs_tcp_yaw_acceleration_rad_s2"],
            "tcp_linear_speed_joint_limit_bound_m_s": tcp_summary["tcp_linear_speed_joint_limit_bound_m_s"],
            "tcp_linear_acceleration_joint_limit_bound_m_s2": tcp_summary["tcp_linear_acceleration_joint_limit_bound_m_s2"],
        })
        result_payload["tcp_motion_summary"] = tcp_summary
        write_trajectory(OUT / "trajectories" / f"{case['case_id']}.csv", plan, context)
        row["event_sequence"] = ";".join(event.event for event in plan.events)
        row["phase_sequence"] = ";".join(plan.phases)
        row["sample_sweep_bound_max_m"] = result.diagnostics["maximum_swept_step_bound_m"]
        row["attempt_count"] = len(result.diagnostics.get("attempts", []))
        row["attempt_failure_codes"] = ";".join(a["code"] for a in result.diagnostics.get("attempts", []))
        row["plot_files"] = ";".join(create_plots(case["case_id"], plan, context, detections))
    else:
        row["event_sequence"] = ""
        row["phase_sequence"] = ""
        row["sample_sweep_bound_max_m"] = ""
        row["attempt_count"] = len(result.diagnostics.get("attempts", []))
        row["attempt_failure_codes"] = ";".join(a.get("code", "") for a in result.diagnostics.get("attempts", []))
        row["plot_files"] = ""
    write_json(case_dir / "plan_result.json", result_document)
    return row


def verify_thin_obstacle(context) -> dict:
    """Prove that a thin static obstacle between free endpoint poses is detected at the sweep midpoint."""
    q_start = list(context.arm.observation_q)
    q_end = list(q_start)
    q_start[2], q_end[2] = 0.015, 0.080
    q_mid = [(a + b) / 2.0 for a, b in zip(q_start, q_end)]
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    joint_names = ("j1_shoulder", "j2_elbow", "j3_lift", "j4_wrist")
    for name, value in zip(joint_names, q_mid):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value
    for name in ("j5_finger_left", "j6_finger_right"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = float(context.p("robot.finger_home_m"))
    mujoco.mj_forward(model, data)
    finger_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "finger_left_collision")
    center = data.geom_xpos[finger_geom]

    xml_tree = ET.parse(MODEL_PATH)
    worldbody = xml_tree.getroot().find("worldbody")
    if worldbody is None:
        raise RuntimeError("M2 model has no worldbody for controlled thin-obstacle test")
    obstacle = ET.SubElement(worldbody, "geom", {
        "name": "m5_thin_obstacle", "type": "box",
        "pos": " ".join(f"{value:.12g}" for value in center),
        "size": "0.002 0.002 0.0005", "contype": "1", "conaffinity": "1",
        "rgba": "0.9 0.15 0.10 1",
    })
    obstacle_description = {"name": obstacle.get("name"), "center_m": [float(v) for v in center],
                            "half_extents_m": [0.002, 0.002, 0.0005]}
    batch = _case_detections({"xy_estimate_m": [0.0, -0.25], "class_label": "RED",
                              "yaw_estimate_rad": 0.0, "obstacles": []}, frame_id=51999)
    target_id = batch[0].track_id
    with tempfile.TemporaryDirectory(prefix="m5_thin_obstacle_") as temp_dir:
        modified_model = Path(temp_dir) / "thin_obstacle.xml"
        xml_tree.write(modified_model, encoding="utf-8", xml_declaration=True)
        scene = CollisionScene(context, modified_model)

        def check(q):
            return scene.check(q, gripper_m=float(context.p("robot.finger_home_m")),
                               detections=batch, target_track_id=target_id,
                               phase="TRANSFER", payload_mode="unheld",
                               sample_step_bound_m=0.001)

        at_start, at_midpoint, at_end = check(q_start), check(q_mid), check(q_end)
    midpoint_hits_obstacle = any("m5_thin_obstacle" in (v.geom_a, v.geom_b)
                                 for v in at_midpoint.violations)
    passed = at_start.valid and not at_midpoint.valid and at_end.valid and midpoint_hits_obstacle
    artifact = {
        "status": "PASS" if passed else "FAIL",
        "test": "thin_static_obstacle_between_collision_free_endpoint_poses",
        "configuration_version": context.config_version,
        "ssot_sha256": context.ssot_sha256,
        "model_sha256": context.model_sha256,
        "modified_model_obstacle": obstacle_description,
        "q_start": q_start, "q_midpoint": q_mid, "q_end": q_end,
        "sweep_sample_step_bound_m": 0.001,
        "observations": {
            "start_valid": at_start.valid,
            "midpoint_valid": at_midpoint.valid,
            "midpoint_reports_named_obstacle": midpoint_hits_obstacle,
            "end_valid": at_end.valid,
            "midpoint_violations": [v.__dict__ for v in at_midpoint.violations],
        },
        "interpretation": "This controlled static obstacle test confirms midpoint collision detection when both endpoint configurations are collision-free; it does not prove continuous collision freedom for arbitrary motion.",
    }
    write_json(OUT / "collision" / "thin_obstacle_test.json", artifact)
    if not passed:
        raise AssertionError("Thin-obstacle collision test failed; inspect collision/thin_obstacle_test.json")
    return artifact


def run() -> dict:
    for name in ("cases", "trajectories", "plots", "collision", "negative", "reports"):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    context = load_context()
    planner = Planner(context)
    write_json(OUT / "cases" / "systematic_cases.json", {
        "configuration_version": context.config_version,
        "ssot_sha256": context.ssot_sha256,
        "baseline_collision_sample_step_m": context.p("planning.path_sweep_sample_step_m"),
        "input_contract": "M4 DetectionBatch estimated fields; no simulator object qpos, true color, object identity, or true pose is passed to Planner.",
        "randomness": "None; all cases and detector records are deterministic.",
        "cases": list(CASES)})
    rows=[]
    for index,case in enumerate(CASES, start=1):
        row=run_case(planner,context,case,frame_id=51000+index)
        rows.append(row)
        print(f"{case['case_id']}: {row['result']} ({row['wall_time_s']:.2f}s, {row['trajectory_sample_count']} samples)",flush=True)
    thin_obstacle = verify_thin_obstacle(context)
    write_csv(OUT / "tables" / "systematic_planning_results.csv",rows)
    write_csv(OUT / "cases" / "systematic_planning_runs.csv",rows)
    negative_rows=[row for row in rows if row["expected"] != PlanCode.SUCCESS.value]
    write_csv(OUT / "negative" / "negative_planning_cases.csv",negative_rows)
    write_json(OUT / "negative" / "negative_planning_cases.json",{
        "status": "PASS" if all(row["passed"] and not row["success"] for row in negative_rows) else "FAIL",
        "cases": negative_rows,
        "interpretation": "A safe refusal is expected for a perceived non-target obstacle inside the configured grasp-clearance envelope; this is not counted as a successful pick/place plan."})

    by_id={row["case_id"]:row for row in rows}
    coarse=by_id["center_red_yaw0"]
    fine=by_id["center_red_halfstep"]
    if coarse["result"] != PlanCode.SUCCESS.value or fine["result"] != PlanCode.SUCCESS.value:
        raise AssertionError("The baseline / half-baseline convergence plans must both pass before comparison.")
    convergence=compare_sampling_runs(
        coarse, fine, refined_step_m=float(context.p("planning.path_sweep_sample_step_m"))*0.5)
    write_json(OUT / "collision" / "sampling_convergence.json",convergence)
    report={"status": "PASS" if all(row["passed"] for row in rows) and convergence["status"]=="PASS" and thin_obstacle["status"]=="PASS" else "FAIL",
        "milestone": "M5", "created_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_version": context.config_version, "ssot_sha256": context.ssot_sha256,
        "model_sha256": context.model_sha256, "planner_source_sha256": sha256(Path(__file__).with_name("planner.py")),
        "trajectory_source_sha256": sha256(Path(__file__).with_name("trajectory.py")),
        "collision_source_sha256": sha256(Path(__file__).with_name("collision.py")),
        "source_sha256": source_hashes(),
        "environment": environment_versions(),
        "randomness": "No random source; each case has an explicit deterministic detection batch.",
        "input_authority": "Planner receives only M4 interface records and explicit M6-style available slot IDs; evaluator truth is not passed.",
        "scenario_count": len(rows), "scenario_pass_count": sum(row["passed"] for row in rows),
        "results": rows, "sampling_convergence": convergence, "thin_obstacle_test": thin_obstacle,
        "limitations": ["Kinematic/time-parameterized plan and sampled collision preflight only; not dynamic execution or physical validation.",
            "Configured inputs are deterministic synthetic M4 contract records, not evidence of a physical camera or a full sorting experiment.",
            "MuJoCo distance queries are capped by SSOT; all-pair clearance is a conservative bound using AABB broad phase and signed-distance narrow phase.",
            "Deterministic radial-ring strategy has a bounded scene scope and fails closed when no checked candidate is valid."]}
    write_json(OUT / "reports" / "m5_validation.json",report)
    if report["status"] != "PASS":
        raise AssertionError("M5 systematic route verification or sampling convergence did not pass; inspect saved reports.")
    return report


if __name__ == "__main__":
    summary=run()
    print(json.dumps({"status":summary["status"],"scenario_count":summary["scenario_count"],
                      "scenario_pass_count":summary["scenario_pass_count"],
                      "sampling_convergence":summary["sampling_convergence"]},ensure_ascii=False,indent=2))
