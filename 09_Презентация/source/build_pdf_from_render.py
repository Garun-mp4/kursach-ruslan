"""Assemble the final slide renders into a fixed-size, 16:9 PDF handout."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


PAGE_WIDTH = 960.0
PAGE_HEIGHT = 540.0
EXPECTED_SLIDES = 15


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("render_dir", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()

    renders = [args.render_dir / f"slide-{number}.png" for number in range(1, EXPECTED_SLIDES + 1)]
    missing = [path for path in renders if not path.is_file()]
    if missing:
        raise SystemExit("Missing slide render(s): " + ", ".join(str(path) for path in missing))

    dimensions = set()
    for path in renders:
        with Image.open(path) as image:
            dimensions.add(image.size)
    if dimensions != {(1920, 1080)}:
        raise SystemExit(f"Expected 1920x1080 slide renders, found: {sorted(dimensions)}")

    args.output_pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(args.output_pdf), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), pageCompression=1)
    pdf.setTitle("Курсовая работа — сортировка объектов манипулятором по цвету")
    pdf.setAuthor("Шатуев Руслан Саматович")
    pdf.setSubject("Слайды для защиты курсовой работы; результаты виртуального моделирования")
    for path in renders:
        pdf.drawImage(ImageReader(str(path)), 0, 0, width=PAGE_WIDTH, height=PAGE_HEIGHT, mask="auto")
        pdf.showPage()
    pdf.save()

    from pypdf import PdfReader

    reader = PdfReader(str(args.output_pdf))
    if len(reader.pages) != EXPECTED_SLIDES:
        raise SystemExit(f"Expected {EXPECTED_SLIDES} PDF pages, got {len(reader.pages)}")
    print(f"Created {args.output_pdf} ({len(reader.pages)} pages, 16:9, 1920x1080 source renders)")


if __name__ == "__main__":
    main()
