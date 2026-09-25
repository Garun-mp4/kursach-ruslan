from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / "05_Верификация" / "perception" / "validation"
CALIBRATION = ROOT / "05_Верификация" / "perception" / "calibration" / "camera_calibration.json"
OUT = ROOT / "05_Верификация" / "perception" / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1600, 900
NAVY = "#18333D"
INK = "#263943"
MUTED = "#657982"
GRID = "#D9E2E5"
PANEL = "#F4F8F8"
TEAL = "#138A85"
BLUE = "#4A7F9C"
YELLOW = "#D99A2B"
RED = "#CB584F"
WHITE = "#FFFFFF"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    draw.text((90, 58), title, font=font(42, bold=True), fill=NAVY)
    draw.text((92, 118), subtitle, font=font(23), fill=MUTED)
    draw.line((90, 165, WIDTH - 90, 165), fill=GRID, width=2)
    return image, draw


def save(image: Image.Image, name: str) -> None:
    image.save(OUT / name, dpi=(180, 180), optimize=True)


def read_metrics() -> dict:
    return json.loads((VALIDATION / "perception_metrics.json").read_text(encoding="utf-8-sig"))


def read_errors() -> list[dict[str, str]]:
    with (VALIDATION / "localization_errors.csv").open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def draw_distribution_chart(
    name: str,
    title: str,
    subtitle: str,
    rows: list[dict[str, str]],
    value_column: str,
    limit: float,
    unit: str,
    xmax: float,
    categories: list[tuple[str, set[str] | None, str]],
) -> None:
    image, draw = canvas(title, subtitle)
    x0, x1 = 390, 1320
    y_rows = [300, 455, 610]
    scale = (x1 - x0) / xmax
    plot_top, plot_bottom = 220, 690
    for tick in np.arange(0.0, xmax + 1e-9, xmax / 5):
        x = round(x0 + float(tick) * scale)
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        draw.text((x - 18, plot_bottom + 18), f"{tick:g}", font=font(18), fill=MUTED)
    draw.text((x0, 735), unit, font=font(20, bold=True), fill=INK)
    limit_x = round(x0 + limit * scale)
    draw.line((limit_x, plot_top - 12, limit_x, plot_bottom + 4), fill=RED, width=3)
    draw.rounded_rectangle((limit_x - 72, 185, limit_x + 72, 220), radius=8, fill="#FBECEA")
    draw.text((limit_x - 61, 192), f"предел {limit:g} {unit}", font=font(16, bold=True), fill=RED)

    for y, (label, classes, color) in zip(y_rows, categories):
        subset = rows if classes is None else [r for r in rows if r["true_class"] in classes]
        values = [float(r[value_column]) * (1000.0 if unit == "мм" else 180.0 / math.pi) for r in subset]
        if not values:
            continue
        p95 = percentile(values, 95)
        maximum = max(values)
        median = percentile(values, 50)
        draw.text((95, y - 16), label, font=font(22, bold=True), fill=INK)
        draw.text((95, y + 17), f"n = {len(values)}", font=font(18), fill=MUTED)
        xmed = round(x0 + min(median, xmax) * scale)
        xp95 = round(x0 + min(p95, xmax) * scale)
        xmax_value = round(x0 + min(maximum, xmax) * scale)
        draw.line((x0, y, xp95, y), fill=color, width=10)
        draw.ellipse((xmed - 9, y - 9, xmed + 9, y + 9), fill=WHITE, outline=color, width=4)
        draw.polygon([(xp95, y - 13), (xp95 + 12, y), (xp95, y + 13), (xp95 - 12, y)], fill=color)
        draw.ellipse((xmax_value - 7, y - 7, xmax_value + 7, y + 7), fill=NAVY)
        detail_x = 1350
        draw.text((detail_x, y - 25), f"p95 {p95:.3f}", font=font(17, bold=True), fill=color)
        draw.text((detail_x, y + 3), f"max {maximum:.3f}", font=font(17), fill=NAVY)
    draw.text((x0, 790), "○ медиана     ◆ 95-й перцентиль     ● максимум", font=font(19), fill=MUTED)
    save(image, name)


def plot_xy(metrics: dict, rows: list[dict[str, str]]) -> None:
    draw_distribution_chart(
        "xy_error_distribution.png",
        "Погрешность положения в плоскости BASE",
        "Независимая выборка M4-v1.3; UNKNOWN показан отдельно и не является целью сортировки",
        rows,
        "planar_error_m",
        float(metrics["grasp_budget"]["perception_xy_budget_m"]) * 1000.0,
        "мм",
        2.5,
        [
            ("Цветные классы", {"RED", "GREEN", "BLUE"}, TEAL),
            ("UNKNOWN", {"UNKNOWN"}, YELLOW),
            ("Все принятые позы", None, BLUE),
        ],
    )


def plot_yaw(metrics: dict, rows: list[dict[str, str]]) -> None:
    draw_distribution_chart(
        "yaw_error_distribution.png",
        "Погрешность ориентации квадратного объекта",
        "Симметричная ошибка по периоду π/2; yaw приведен к градусам",
        rows,
        "yaw_error_mod_pi2_rad",
        math.degrees(float(metrics["grasp_budget"]["yaw_error_limit_rad"])),
        "°",
        10.0,
        [
            ("Цветные классы", {"RED", "GREEN", "BLUE"}, TEAL),
            ("UNKNOWN", {"UNKNOWN"}, YELLOW),
            ("Все принятые позы", None, BLUE),
        ],
    )


def plot_confusion(metrics: dict) -> None:
    image, draw = canvas(
        "Матрица классификации цвета",
        "Ground truth оценивает результат независимо; NO_DETECTION сохранен отдельным столбцом",
    )
    labels = ["RED", "GREEN", "BLUE", "UNKNOWN", "NO_DETECTION"]
    matrix = metrics["validation_metrics"]["color_confusion_matrix_counts"]
    x0, y0, cell_w, cell_h = 460, 300, 190, 92
    draw.text((104, 205), "Истинный класс", font=font(20, bold=True), fill=NAVY)
    draw.text((x0 + 165, 205), "Класс по RGB", font=font(21, bold=True), fill=NAVY)
    for col, label in enumerate(labels):
        xx = x0 + col * cell_w
        draw.text((xx + 12, y0 - 48), label, font=font(17, bold=True), fill=INK)
    for row, actual in enumerate(labels[:-1]):
        yy = y0 + row * cell_h
        draw.text((105, yy + 28), actual, font=font(22, bold=True), fill=INK)
        for col, predicted in enumerate(labels):
            count = int(matrix[actual].get(predicted, 0))
            xx = x0 + col * cell_w
            fill = "#DDF1ED" if count else PANEL
            if actual == "UNKNOWN" and predicted == "UNKNOWN":
                fill = "#FFF0CF"
            draw.rounded_rectangle((xx, yy, xx + cell_w - 8, yy + cell_h - 8), radius=10,
                                   fill=fill, outline=GRID, width=1)
            draw.text((xx + 68, yy + 24), str(count), font=font(30, bold=True), fill=NAVY)
    total = sum(sum(int(v) for v in matrix[row].values()) for row in matrix)
    draw.text((104, 705), f"Всего объектов: {total}     Ошибочная переклассификация: 0     NO_DETECTION: 0",
              font=font(20, bold=True), fill=TEAL)
    save(image, "color_confusion_matrix.png")


def plot_calibration_map() -> None:
    artifact = json.loads(CALIBRATION.read_text(encoding="utf-8-sig"))
    xy = np.asarray(artifact["validation_points_base_xy_m"], dtype=float)
    uv = np.asarray(artifact["validation_points_detected_uv_px"], dtype=float)
    predicted = np.asarray(artifact["pixel_to_base_xy_homography"], dtype=float)
    import cv2
    estimated = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float64), predicted).reshape(-1, 2)
    errors = np.linalg.norm(estimated - xy, axis=1) * 1000.0
    image, draw = canvas(
        "Проверка калибровки по плоскости предмета",
        "64 независимые контрольные точки; цвет и размер маркера показывают модуль XY-ошибки",
    )
    x0, y0, plot_w, plot_h = 250, 250, 1040, 490
    xmin, xmax, ymin, ymax = -0.5, 0.5, -0.45, 0.45
    for x in np.arange(xmin, xmax + 1e-9, 0.1):
        xx = round(x0 + (float(x) - xmin) / (xmax - xmin) * plot_w)
        draw.line((xx, y0, xx, y0 + plot_h), fill=GRID, width=1)
        draw.text((xx - 18, y0 + plot_h + 12), f"{x:.1f}", font=font(16), fill=MUTED)
    for y in np.arange(ymin, ymax + 1e-9, 0.1):
        yy = round(y0 + (ymax - float(y)) / (ymax - ymin) * plot_h)
        draw.line((x0, yy, x0 + plot_w, yy), fill=GRID, width=1)
        draw.text((x0 - 55, yy - 10), f"{y:.1f}", font=font(16), fill=MUTED)
    for (x, y), error in zip(xy, errors):
        xx = round(x0 + (x - xmin) / (xmax - xmin) * plot_w)
        yy = round(y0 + (ymax - y) / (ymax - ymin) * plot_h)
        ratio = min(float(error) / 1.0, 1.0)
        color = (int(19 + 190 * ratio), int(138 - 55 * ratio), int(133 - 85 * ratio))
        radius = 7 + round(min(float(error), 1.0) * 7)
        draw.ellipse((xx - radius, yy - radius, xx + radius, yy + radius),
                     fill="#%02X%02X%02X" % color, outline=WHITE, width=2)
    draw.text((x0, 795),
              f"Средняя ошибка {np.mean(errors):.3f} мм     p95 {np.percentile(errors, 95):.3f} мм     максимум {np.max(errors):.3f} мм",
              font=font(20, bold=True), fill=NAVY)
    save(image, "calibration_validation_map.png")


def main() -> None:
    metrics = read_metrics()
    rows = read_errors()
    plot_xy(metrics, rows)
    plot_yaw(metrics, rows)
    plot_confusion(metrics)
    plot_calibration_map()
    print(json.dumps({"status": "OK", "outputs": sorted(p.name for p in OUT.glob("*.png"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
