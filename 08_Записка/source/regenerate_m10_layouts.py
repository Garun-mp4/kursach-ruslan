from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M2_MODULES = ROOT / "04_Программа" / "m2"
SOURCE_DIR = ROOT / "08_Записка" / "source"
FIGURE_DIR = ROOT / "08_Записка" / "figures"
RENDERED_M7_MODEL = ROOT / "99_Рабочие_материалы" / "m2_validation" / "results" / "m2_model_isometric.png"
M7_VALIDATION = ROOT / "99_Рабочие_материалы" / "m2_validation" / "results" / "m2_validation.json"

sys.path.insert(0, str(M2_MODULES))
import build_model  # noqa: E402
import draw_m2_diagrams  # noqa: E402


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    draw_m2_diagrams.OUT = SOURCE_DIR
    config = build_model.load_ssot()
    p = lambda key: build_model.parameter(config, key)
    draw_m2_diagrams.create_layout(p)
    draw_m2_diagrams.create_dimensioned(p)

    layout_tmp = SOURCE_DIR / "Компоновка_ячейки_M2.drawio"
    dims_tmp = SOURCE_DIR / "Размерный_вид_M2.drawio"
    layout_src = SOURCE_DIR / "robot_layout_m7.drawio"
    dims_src = SOURCE_DIR / "robot_dimensions_m7.drawio"
    layout_xml = layout_tmp.read_text(encoding="utf-8")
    dimensions_xml = dims_tmp.read_text(encoding="utf-8")
    layout_xml = layout_xml.replace("M2 layout", "M7 final layout")
    layout_xml = layout_xml.replace("Компоновка рабочей ячейки — вид сверху", "Компоновка рабочей ячейки — финальная геометрия M7-v1.8")
    layout_xml = layout_xml.replace("Модельная конфигурация M2 · все координаты и размеры взяты из SSOT", "Финальная конфигурация M7-v1.8 · координаты и размеры из SSOT")
    layout_xml = layout_xml.replace("Плечо L1:", "Звено L1:").replace("Локоть L2:", "Звено L2:")
    layout_xml = layout_xml.replace(
        "Примечание: это схема компоновки и целевых зон, а не результат проверки достижимости. Точную рабочую область, ориентационные ограничения и траектории проверяет M3–M5.",
        "Примечание: схема показывает компоновку и целевые зоны, а не карту достижимости. Достижимость проверена кинематически и интеграционно в M3–M5 и M7/M9.",
    )
    dimensions_xml = dimensions_xml.replace("M2 dimensions", "M7 final dimensions")
    dimensions_xml = dimensions_xml.replace("Размерная схема модели манипулятора и ячейки", "Размерная схема финальной модели M7-v1.8 и ячейки")
    dimensions_xml = dimensions_xml.replace("Плечо L1:", "Звено L1:").replace("Локоть L2:", "Звено L2:")
    dimensions_xml = dimensions_xml.replace(
        "TCP: 103 mm … 6 mm; ход J3: 97 mm",
        "Диапазон z_TCP: 6–103 mm; ход J3: 97 mm",
    )
    dimensions_xml = dimensions_xml.replace(
        "Боковой вид задаёт только высоты и вертикальный диапазон. Точные положения вдоль XY и достижимость цели проверяются по кинематике в M3.",
        "Боковой вид показывает высоты и вертикальный ход. Достижимость целевых точек проверена кинематически; столкновения пути оценивались отдельно в M5–M9.",
    )
    if "Диапазон z_TCP: 6–103 mm; ход J3: 97 mm" not in dimensions_xml:
        raise ValueError("The expected final M7 vertical TCP range label was not updated")
    layout_src.write_text(layout_xml, encoding="utf-8")
    dims_src.write_text(dimensions_xml, encoding="utf-8")
    layout_tmp.unlink()
    dims_tmp.unlink()

    if not RENDERED_M7_MODEL.is_file() or not M7_VALIDATION.is_file():
        raise FileNotFoundError("M7-v1.8 model validation render is missing")
    validation = json.loads(M7_VALIDATION.read_text(encoding="utf-8"))
    if validation.get("status") != "PASS" or validation.get("config_version") != "M7-v1.8":
        raise ValueError("The isometric render is not tied to the passing M7-v1.8 model validation")
    iso_dest = FIGURE_DIR / "02_SCARA_изометрия_M7.png"
    shutil.copy2(RENDERED_M7_MODEL, iso_dest)

    manifest = {
        "schema_version": "M10-figure-provenance-v1",
        "ssot_version": "M7-v1.8",
        "layout_source": str(layout_src.relative_to(ROOT)),
        "dimension_source": str(dims_src.relative_to(ROOT)),
        "drawio_generator": "04_Программа/m2/draw_m2_diagrams.py, regenerated from final SSOT",
        "isometric_image": str(iso_dest.relative_to(ROOT)),
        "isometric_model_sha256": validation["model_sha256"],
        "isometric_engine_version": validation["mujoco_version"],
        "isometric_validation_status": validation["status"],
    }
    (ROOT / "08_Записка" / "review" / "Иллюстрации_M7.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
