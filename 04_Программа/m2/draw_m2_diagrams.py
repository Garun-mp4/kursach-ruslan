"""Generate editable M2 engineering diagrams from the project's SSOT YAML."""
from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import build_model

ROOT = build_model.ROOT
OUT = ROOT / "03_Модель_и_схемы" / "source"


class Diagram:
    def __init__(self, name: str, width: int, height: int):
        self.name, self.width, self.height = name, width, height
        self.cells: list[ET.Element] = []
        self.next_id = 2

    def _id(self) -> str:
        value = str(self.next_id)
        self.next_id += 1
        return value

    def box(self, x: float, y: float, w: float, h: float, text: str = "", *,
            fill: str = "#FFFFFF", stroke: str = "#526170", color: str = "#172B3A",
            size: int = 14, bold: bool = False, rounded: bool = False,
            dashed: bool = False, opacity: int = 100, align: str = "center",
            valign: str = "middle", shape: str = "rectangle", extra: str = "") -> str:
        style = (f"shape={shape};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
                 f"fontColor={color};fontSize={size};fontFamily=Arial;align={align};"
                 f"verticalAlign={valign};rounded={1 if rounded else 0};opacity={opacity};")
        if bold:
            style += "fontStyle=1;"
        if dashed:
            style += "dashed=1;dashPattern=8 5;"
        style += extra
        cell = ET.Element("mxCell", {"id": self._id(), "value": text, "style": style,
                                      "vertex": "1", "parent": "1"})
        ET.SubElement(cell, "mxGeometry", {"x": f"{x:.3f}", "y": f"{y:.3f}",
                      "width": f"{w:.3f}", "height": f"{h:.3f}", "as": "geometry"})
        self.cells.append(cell)
        return cell.attrib["id"]

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             color: str = "#526170", width: float = 2, dashed: bool = False,
             start_arrow: str = "none", end_arrow: str = "none", rounded: bool = False) -> None:
        style = (f"edgeStyle=none;html=1;rounded={1 if rounded else 0};strokeColor={color};"
                 f"strokeWidth={width};startArrow={start_arrow};endArrow={end_arrow};")
        if dashed:
            style += "dashed=1;dashPattern=8 5;"
        edge = ET.Element("mxCell", {"id": self._id(), "style": style,
                                     "edge": "1", "parent": "1"})
        geom = ET.SubElement(edge, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geom, "mxPoint", {"x": f"{x1:.3f}", "y": f"{y1:.3f}", "as": "sourcePoint"})
        ET.SubElement(geom, "mxPoint", {"x": f"{x2:.3f}", "y": f"{y2:.3f}", "as": "targetPoint"})
        self.cells.append(edge)

    def text(self, x: float, y: float, w: float, h: float, value: str, **kwargs) -> None:
        kwargs.setdefault("align", "left")
        self.box(x, y, w, h, value, fill="none", stroke="none", **kwargs)

    def save(self, path: Path) -> None:
        mxfile = ET.Element("mxfile", {"host": "app.diagrams.net", "modified": "2026-09-24T12:00:00.000Z",
                                       "agent": "Codex", "version": "26.2.2", "type": "device"})
        diagram = ET.SubElement(mxfile, "diagram", {"id": f"m2-{self.name}", "name": self.name})
        model = ET.SubElement(diagram, "mxGraphModel", {"dx": str(self.width), "dy": str(self.height),
            "grid": "1", "gridSize": "10", "guides": "1", "tooltips": "1", "connect": "1",
            "arrows": "1", "fold": "1", "page": "1", "pageScale": "1", "pageWidth": str(self.width),
            "pageHeight": str(self.height), "math": "0", "shadow": "0", "background": "#FFFFFF"})
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", {"id": "0"})
        ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
        root.extend(self.cells)
        ET.indent(mxfile, space="  ")
        path.write_bytes(ET.tostring(mxfile, encoding="utf-8", xml_declaration=True))


def fmt_mm(m: float) -> str:
    return f"{m * 1000:.0f} mm"


def map_xy(x: float, y: float, cx: float, cy: float, scale: float) -> tuple[float, float]:
    return cx + x * scale, cy - y * scale


def add_plan(d: Diagram, p, *, cx: float, cy: float, scale: float,
             show_camera: bool, show_reach: bool, robot_home: bool = True) -> None:
    table_w, table_h = p("cell.table_size_xy_m")
    left, top = cx - table_w * scale / 2, cy - table_h * scale / 2
    d.box(left, top, table_w * scale, table_h * scale, "РАБОЧИЙ СТОЛ · Z = 0",
          fill="#F7F9FB", stroke="#667788", color="#667788", size=11, align="left", valign="top",
          extra="spacingLeft=8;spacingTop=6;")

    if show_camera:
        cov_w, cov_h = p("camera.coverage_width_height_m")
        d.box(cx - cov_w * scale / 2, cy - cov_h * scale / 2, cov_w * scale, cov_h * scale,
              "", fill="none", stroke="#7856A8", color="#7856A8", dashed=True, shape="rectangle",
              extra="strokeWidth=2;")

    if show_reach:
        rmin, rmax = p("robot.radial_reach_min_max_m")
        d.box(cx - rmax * scale, cy - rmax * scale, 2 * rmax * scale, 2 * rmax * scale,
              "", fill="none", stroke="#2C8391", color="#2C8391", dashed=True,
              shape="ellipse", extra="strokeWidth=2;")
        d.box(cx - rmin * scale, cy - rmin * scale, 2 * rmin * scale, 2 * rmin * scale,
              "", fill="#FFFFFF", stroke="#2C8391", color="#2C8391", opacity=35,
              shape="ellipse", extra="strokeWidth=1;dashed=1;")

    # Source area, coordinate placement, and object footprints all come from SSOT.
    sx, sy = p("cell.input_center_xy_m")
    sw, sh = p("cell.input_size_xy_m")
    px, py = map_xy(sx, sy, cx, cy, scale)
    d.box(px - sw * scale / 2, py - sh * scale / 2, sw * scale, sh * scale,
          "", fill="#FFFFFF", stroke="#6E7D8C", color="#34495E", size=12, dashed=True)
    d.text(px-sw*scale/2+4, py-sh*scale/2-22, sw*scale-8, 20,
           "ВХОДНАЯ ЗОНА", size=9, bold=True, color="#526170")
    obj = p("object.size_xyz_m")
    for row_y in p("cell.input_row_y_offsets_m"):
        for slot_x in p("cell.input_slot_x_offsets_m"):
            ox, oy = map_xy(sx + slot_x, sy + row_y, cx, cy, scale)
            d.box(ox - obj[0] * scale / 2, oy - obj[1] * scale / 2,
                  obj[0] * scale, obj[1] * scale, "", fill="#9AA8B4", stroke="#44515C", size=8)

    slot_offsets = p("cell.tray_slot_x_offsets_m")
    outer_w, outer_h = p("cell.tray_outer_size_xy_m")
    inner_w, inner_h = p("cell.tray_inner_size_xy_m")
    tray_colors = {"RED": ("#FCE9E8", "#B73535"), "GREEN": ("#E7F5EA", "#36854A"),
                   "BLUE": ("#E8EEFC", "#355DA8")}
    for cls, (tx, ty) in p("cell.tray_centers_xy_m").items():
        qx, qy = map_xy(tx, ty, cx, cy, scale)
        pale, strong = tray_colors[cls]
        d.box(qx - outer_w * scale / 2, qy - outer_h * scale / 2,
              outer_w * scale, outer_h * scale, "",
              fill=pale, stroke=strong, color=strong, size=12, bold=True, rounded=False)
        d.text(qx-outer_w*scale/2, qy-outer_h*scale/2-21, outer_w*scale, 20,
               cls, size=10, bold=True, color=strong, align="center")
        d.box(qx - inner_w * scale / 2, qy - inner_h * scale / 2,
              inner_w * scale, inner_h * scale, "", fill="none", stroke=strong, color=strong,
              dashed=True, size=8)
        for dx in slot_offsets:
            ox, oy = map_xy(tx + dx, ty + p("cell.tray_slot_y_offset_m"), cx, cy, scale)
            d.box(ox - obj[0] * scale / 2, oy - obj[1] * scale / 2,
                  obj[0] * scale, obj[1] * scale, "", fill=strong, stroke=strong, size=8)

    # Robot plan pose is explicitly the M2 safe observation keyframe.
    bx, by = map_xy(*p("robot.base_world_position_m")[:2], cx, cy, scale)
    d.box(bx - 0.07 * scale, by - 0.07 * scale, 0.14 * scale, 0.14 * scale,
          "", fill="#24384A", stroke="#24384A", color="#FFFFFF", size=10, bold=True,
          rounded=True)
    d.text(bx+38, by+30, 120, 20, "BASE / WORLD", size=9, bold=True, color="#24384A")
    if robot_home:
        q = p("camera.safe_observation_joint_pose")
        q1, q2 = float(q["j1_shoulder"]), float(q["j2_elbow"])
        l1, l2 = float(p("robot.link1_length_m")), float(p("robot.link2_length_m"))
        exw, eyw = l1 * math.cos(q1), l1 * math.sin(q1)
        txw, tyw = exw + l2 * math.cos(q1 + q2), eyw + l2 * math.sin(q1 + q2)
        ex, ey = map_xy(exw, eyw, cx, cy, scale)
        tx, ty = map_xy(txw, tyw, cx, cy, scale)
        d.line(bx, by, ex, ey, color="#4479A8", width=11, rounded=True)
        d.line(ex, ey, tx, ty, color="#68A1C5", width=10, rounded=True)
        d.box(bx - 7, by - 7, 14, 14, "J1", fill="#1B4567", stroke="#FFFFFF", color="#FFFFFF", size=8, shape="ellipse")
        d.box(ex - 7, ey - 7, 14, 14, "J2", fill="#1B4567", stroke="#FFFFFF", color="#FFFFFF", size=8, shape="ellipse")
        d.box(tx - 9, ty - 9, 18, 18, "TCP", fill="#E9A23B", stroke="#FFFFFF", color="#172B3A", size=7, shape="ellipse")

    # World axes remain legible in every top view.
    ax, ay = left + 46, top + table_h * scale - 38
    d.line(ax, ay, ax + 45, ay, color="#C14B43", width=2, end_arrow="classic")
    d.line(ax, ay, ax, ay - 45, color="#3E8B62", width=2, end_arrow="classic")
    d.text(ax + 48, ay - 12, 35, 22, "+X", color="#C14B43", size=12, bold=True)
    d.text(ax - 12, ay - 68, 45, 22, "+Y", color="#3E8B62", size=12, bold=True)


def create_layout(p) -> None:
    d = Diagram("M2 layout", 1100, 720)
    d.text(35, 20, 1030, 34, "Компоновка рабочей ячейки — вид сверху", size=22, bold=True)
    d.text(35, 54, 1030, 26, "Модельная конфигурация M2 · все координаты и размеры взяты из SSOT", size=13, color="#5D6B78")
    add_plan(d, p, cx=330, cy=370, scale=500, show_camera=True, show_reach=False)
    d.text(690, 112, 370, 27, "Геометрия и ориентация", size=16, bold=True)
    d.text(690, 150, 350, 148,
           f"Стол: {fmt_mm(p('cell.table_size_xy_m')[0])} × {fmt_mm(p('cell.table_size_xy_m')[1])}\n"
           f"Плечо L1: {fmt_mm(p('robot.link1_length_m'))}\n"
           f"Локоть L2: {fmt_mm(p('robot.link2_length_m'))}\n"
           f"Поза руки: безопасная обзорная из SSOT\n"
           f"Сплошная схема руки: плановая проекция в home",
           size=13, color="#34495E")
    d.text(690, 317, 350, 112,
           f"Вход: {fmt_mm(p('cell.input_size_xy_m')[0])} × {fmt_mm(p('cell.input_size_xy_m')[1])}\n"
           f"Лотки: {fmt_mm(p('cell.tray_outer_size_xy_m')[0])} × {fmt_mm(p('cell.tray_outer_size_xy_m')[1])}\n"
           f"Объекты: {fmt_mm(p('object.size_xyz_m')[0])} × {fmt_mm(p('object.size_xyz_m')[1])} × {fmt_mm(p('object.size_xyz_m')[2])}\n"
           f"На входе: {int(p('object.count_per_class'))} объекта каждого класса",
           size=13, color="#34495E")
    d.box(688, 455, 18, 18, "", fill="#F7F9FB", stroke="#667788")
    d.text(714, 450, 340, 26, "Стол; тонкая пунктирная граница — камера", size=12, color="#34495E")
    d.line(688, 496, 706, 496, color="#4479A8", width=8)
    d.text(714, 484, 340, 26, "Звенья в home-позе", size=12, color="#34495E")
    d.box(688, 525, 18, 18, "", fill="#9AA8B4", stroke="#44515C")
    d.text(714, 520, 340, 26, "Однослойные сортируемые кубоиды", size=12, color="#34495E")
    d.text(688, 585, 360, 75,
           "Примечание: это схема компоновки и целевых зон, а не результат проверки достижимости. Точную рабочую область, ориентационные ограничения и траектории проверяет M3–M5.",
           size=11, color="#7A4B17")
    d.save(OUT / "Компоновка_ячейки_M2.drawio")


def create_dimensioned(p) -> None:
    d = Diagram("M2 dimensions", 1320, 850)
    d.text(35, 18, 1250, 34, "Размерная схема модели манипулятора и ячейки", size=22, bold=True)
    d.text(35, 52, 1250, 25, "Плановый масштаб для ячейки; боковой разрез схематический. Значения — параметры учебной модели, не паспорт реального изделия.", size=12, color="#5D6B78")
    # Scaled top layout and physical dimensions.
    add_plan(d, p, cx=350, cy=345, scale=470, show_camera=False, show_reach=False)
    table_w, table_h = p("cell.table_size_xy_m")
    left, top = 350-table_w*470/2, 345-table_h*470/2
    d.line(left, top-22, left+table_w*470, top-22, color="#3B5268", width=1.5,
           start_arrow="classic", end_arrow="classic")
    d.text(left+table_w*470/2-75, top-48, 150, 23, f"{fmt_mm(table_w)}", size=12, bold=True)
    d.line(left-23, top, left-23, top+table_h*470, color="#3B5268", width=1.5,
           start_arrow="classic", end_arrow="classic")
    d.text(left-72, top+table_h*470/2-12, 48, 25, f"{fmt_mm(table_h)}", size=11, bold=True)

    # Compact, independent physical callouts below the plan.
    d.text(55, 590, 265, 24, "ОБЪЕКТ", size=12, bold=True, color="#34495E")
    d.text(55, 616, 265, 65,
           f"{fmt_mm(p('object.size_xyz_m')[0])} × {fmt_mm(p('object.size_xyz_m')[1])} × {fmt_mm(p('object.size_xyz_m')[2])}; центр покоя z = {fmt_mm(p('object.position_z_m'))}.",
           size=11, color="#34495E")
    d.text(350, 590, 300, 24, "ЗАХВАТ", size=12, bold=True, color="#34495E")
    d.text(350, 616, 300, 105,
           f"Палец: {fmt_mm(p('robot.finger_length_m'))} × {fmt_mm(p('robot.finger_thickness_m'))} × {fmt_mm(p('robot.finger_height_m'))}.\n"
           f"Открытый зазор: {fmt_mm(2*(p('robot.finger_open_center_offset_m')-p('robot.finger_thickness_m')/2))}.\n"
           f"Ход координаты: {fmt_mm(p('robot.finger_range_m')[1])} на палец.",
           size=11, color="#34495E")

    # Side section. Vertical scale is physical; horizontal arm depiction is schematic.
    shoulder = float(p("robot.shoulder_z_m"))
    camera_h = float(p("camera.position_world_m")[2])
    q_min, q_max = p("robot.j3_range_m")
    tcp_high = 0.103-float(q_min)
    tcp_low = 0.103-float(q_max)
    d.text(720, 104, 570, 28, "Боковой разрез: высоты и вертикальный ход", size=16, bold=True)
    side_left, side_right, z0, zscale = 720, 1285, 690, 300
    d.line(side_left, z0, side_right, z0, color="#71808C", width=3)
    d.text(side_left, z0+7, 250, 23, "Плоскость стола · z = 0", size=10, color="#526170")
    base_x = 800
    base_height = float(p("robot.base_foot_center_world_m")[2])*2
    d.box(base_x-24, z0-base_height*zscale, 48, base_height*zscale,
          "", fill="#DCE3EA", stroke="#506477")
    d.box(base_x-7, z0-shoulder*zscale, 14, shoulder*zscale,
          "", fill="#607D96", stroke="#40576D")
    joint_y = z0-shoulder*zscale
    d.box(base_x+7, joint_y-7, 213, 14, "", fill="#D7E6F2", stroke="#4479A8")
    d.box(base_x+214, joint_y-7, 157, 14, "", fill="#E2EDF4", stroke="#68A1C5")
    d.box(base_x+360, z0-tcp_high*zscale-5, 18,
          max(8,(tcp_high-tcp_low)*zscale+10), "", fill="#E8B44D", stroke="#9B6A15")
    d.box(base_x+355, z0-tcp_high*zscale-9, 28, 10, "", fill="#E9A23B", stroke="#9B6A15")
    d.line(base_x+369, z0-tcp_high*zscale, base_x+369, z0-tcp_low*zscale,
           color="#9B6A15", width=2, start_arrow="classic", end_arrow="classic")
    camera_x = 1250
    d.box(camera_x-16, z0-camera_h*zscale-12, 32, 24, "CAM", fill="#73569A", stroke="#543C75", color="#FFFFFF", size=9, bold=True, rounded=True)
    d.line(camera_x, z0-camera_h*zscale+13, camera_x, z0-0.02*zscale,
           color="#73569A", width=2, dashed=True, end_arrow="classic")
    d.text(camera_x-104, z0-camera_h*zscale-39, 208, 22,
           f"CAMERA z = {fmt_mm(camera_h)}", size=10, color="#543C75", align="center")
    # Independent dimension arrows and non-overlapping labels.
    d.line(770, z0, 770, joint_y, color="#4479A8", width=1.5,
           start_arrow="classic", end_arrow="classic")
    d.text(730, 520, 150, 22, f"Высота оси J1: {fmt_mm(shoulder)}", size=10, color="#4479A8")
    d.line(1300, z0, 1300, z0-camera_h*zscale, color="#73569A", width=1.5,
           start_arrow="classic", end_arrow="classic")
    d.text(1130, z0-camera_h*zscale/2-14, 100, 22, f"{fmt_mm(camera_h)}", size=10, color="#73569A")
    d.text(985, 697, 305, 34,
           f"TCP: {fmt_mm(tcp_high)} … {fmt_mm(tcp_low)}; ход J3: {fmt_mm(q_max-q_min)}",
           size=9, color="#6B4A13")
    d.text(720, 744, 585, 48,
           "Боковой вид задаёт только высоты и вертикальный диапазон. Точные положения вдоль XY и достижимость цели проверяются по кинематике в M3.",
           size=11, color="#7A4B17")
    d.save(OUT / "Размерный_вид_M2.drawio")


def create_frames(p) -> None:
    d = Diagram("M2 frames", 1160, 760)
    d.text(35, 18, 1090, 36, "Системы координат и кинематические соглашения", size=22, bold=True)
    d.text(35, 56, 1090, 25, "Схема показывает нулевую конфигурацию звеньев; координатные оси и нули заданы в SSOT.", size=12, color="#5D6B78")
    # Zero-pose planar geometry.
    scale, x0, y0 = 700, 160, 500
    l1, l2 = float(p("robot.link1_length_m")), float(p("robot.link2_length_m"))
    j1=(x0,y0); j2=(x0+l1*scale,y0); j3=(x0+(l1+l2)*scale,y0)
    d.line(j1[0],j1[1],j2[0],j2[1],color="#4479A8",width=16,rounded=True)
    d.line(j2[0],j2[1],j3[0],j3[1],color="#68A1C5",width=14,rounded=True)
    for name,(x,y),col in [("J1",j1,"#1B4567"),("J2",j2,"#1B4567"),("J3",j3,"#E9A23B")]:
        d.box(x-11,y-11,22,22,name,fill=col,stroke="#FFFFFF",color="#FFFFFF",size=9,shape="ellipse")
    d.line(j3[0],j3[1],j3[0],j3[1]+105,color="#9B6A15",width=5,rounded=True)
    d.box(j3[0]-13,j3[1]+95,26,20,"J4",fill="#8A5B19",stroke="#FFFFFF",color="#FFFFFF",size=9,shape="ellipse")
    d.box(j3[0]-18,j3[1]+135,36,22,"TCP",fill="#E9A23B",stroke="#FFFFFF",color="#172B3A",size=9,shape="ellipse")
    d.line(j1[0],j1[1],j1[0]+92,j1[1],color="#C14B43",width=2,end_arrow="classic")
    d.line(j1[0],j1[1],j1[0],j1[1]-92,color="#3E8B62",width=2,end_arrow="classic")
    d.text(j1[0]+96,j1[1]-14,42,24,"+X",color="#C14B43",size=12,bold=True)
    d.text(j1[0]-14,j1[1]-116,42,24,"+Y",color="#3E8B62",size=12,bold=True)
    d.text(j1[0]-8,j1[1]+28,120,26,"WORLD = BASE XY",size=12,bold=True)
    d.line(j1[0]+10,j1[1]-40,j2[0]-10,j2[1]-40,color="#365E7F",width=1.4,start_arrow="classic",end_arrow="classic")
    d.text((j1[0]+j2[0])/2-42,j1[1]-70,84,23,f"L1 = {fmt_mm(l1)}",size=11,bold=True)
    d.line(j2[0]+10,j2[1]-40,j3[0]-10,j3[1]-40,color="#365E7F",width=1.4,start_arrow="classic",end_arrow="classic")
    d.text((j2[0]+j3[0])/2-42,j2[1]-70,84,23,f"L2 = {fmt_mm(l2)}",size=11,bold=True)

    # Vertical stack and camera frame conventions.
    right=770
    d.text(right,112,350,27,"Оси и преобразования",size=16,bold=True)
    d.text(right,150,350,115,
           "WORLD: правая система; начало на центре стола; +Z вверх.\nBASE совпадает с WORLD и закреплена.\nJ1/J2/J4 вращаются вокруг локальной +Z.\nJ3 перемещается вдоль локальной −Z.\nПальцы движутся симметрично по ±Y.",
           size=11,color="#34495E")
    d.text(right,282,350,82,
           "T_A_B переводит координаты из B в A: p_A = T_A_B · p_B. Столбцовые векторы; длина — m, угол — rad.",
           size=11,color="#34495E")
    d.box(right,395,330,122,
          f"TCP в J4: {p('robot.tcp_in_wrist_m')} m\nJ4 в каретке: {p('robot.wrist_center_in_zslide_m')} m\nJ1 в BASE: [0, 0, {p('robot.shoulder_z_m')}] m",
          fill="#F2F5F7",stroke="#AAB5BF",color="#34495E",size=11,align="left",extra="spacingLeft=10;")
    d.text(right,540,340,26,"Камера: два разных соглашения",size=15,bold=True)
    d.box(right,570,168,106,"CAM_MJ\nMuJoCo-камера\nлокальная −Z направлена вниз\nquaternion wxyz: [1,0,0,0]",
          fill="#EFEAF7",stroke="#73569A",color="#49386A",size=10,rounded=True)
    d.box(right+180,570,168,106,"CAM_CV\nоптический кадр OpenCV\nX→+X, Y→−Y, Z→−Z\nR = diag(1,−1,−1)",
          fill="#F5EFFA",stroke="#73569A",color="#49386A",size=10,rounded=True)
    d.text(40,700,650,38,
           "Положительное вращение J1/J2/J4 — правило правой руки вокруг +Z. Уравнения FK/IK и проверка знаков принадлежат M3.",
           size=11,color="#7A4B17")
    d.save(OUT / "Системы_координат_M2.drawio")


def create_camera(p) -> None:
    d = Diagram("M2 camera and workspace", 1080, 720)
    d.text(35,18,1010,36,"Камера и предварительная геометрия рабочей зоны",size=22,bold=True)
    d.text(35,56,1010,25,"Вид сверху · границы радиального кольца являются грубой проверкой M2, не полной достижимой областью.",size=12,color="#5D6B78")
    add_plan(d,p,cx=390,cy=380,scale=430,show_camera=True,show_reach=True,robot_home=False)
    cov=p("camera.coverage_width_height_m")
    ppx=p("camera.intrinsics_fx_fy_cx_cy_px")
    rmin,rmax=p("robot.radial_reach_min_max_m")
    d.text(750,132,290,26,"Верхняя RGB-камера",size=16,bold=True)
    d.text(750,170,290,126,
           f"Поза WORLD: {p('camera.position_world_m')} m\n"
           f"Изображение: {p('camera.resolution_px')} px\n"
           f"Поле обзора: {p('camera.fovy_deg')}°\n"
           f"Номинальный охват: {cov[0]:.3f} × {cov[1]:.3f} m\n"
           f"fx = fy ≈ {ppx[0]:.2f} px; центр ({ppx[2]:.1f}, {ppx[3]:.1f})",
           size=12,color="#34495E")
    d.box(750,322,22,22,"",fill="none",stroke="#7856A8",dashed=True)
    d.text(780,317,250,28,"Номинальный footprint камеры",size=11,color="#543C75")
    d.box(750,352,22,22,"",fill="none",stroke="#2C8391",dashed=True,shape="ellipse")
    d.text(780,347,250,36,f"r = {rmin:.3f} … {rmax:.3f} m",size=11,color="#286D77")
    d.text(750,405,285,145,
           "Все входные точки и слоты размещения проходят грубую проверку расстояния до BASE. Проверка joint limits, ориентации TCP, self-collision и obstacle clearance будет выполнена в M3–M5.",
           size=12,color="#34495E")
    d.text(750,580,280,55,
           "Камера покрывает весь стол по идеальной pinhole-геометрии. Перекрытие звеньями и точность цветовой локализации проверяет M4.",
           size=11,color="#7A4B17")
    d.save(OUT / "Камера_и_рабочая_зона_M2.drawio")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config = build_model.load_ssot()
    p = lambda key: build_model.parameter(config, key)
    create_layout(p)
    create_dimensioned(p)
    create_frames(p)
    create_camera(p)
    print("Created editable M2 draw.io sources:")
    for path in sorted(OUT.glob("*_M2.drawio")):
        print(path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"M2 diagram generation failed: {exc}", file=sys.stderr)
        raise
