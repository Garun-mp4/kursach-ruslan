from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "06_Медиа" / "M9" / "single_red_center-2d771e58cc" / "кадры" / "final.png"
DEFAULT_OUTPUT = ROOT / "08_Записка" / "figures" / "12_Кадр_успешного_виртуального_цикла.png"
EXPECTED_SOURCE_SHA256 = "8339e335cc8a6b805e35fd07717cbaef8a87fe8564d3015e7b7b004e919f33b6"
BACKGROUND = (16, 25, 37)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove only the two redundant figure-level headings from the M9 success frame."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_bytes = SOURCE.read_bytes()
    digest = hashlib.sha256(source_bytes).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected M9 source frame SHA-256: {digest}")

    with Image.open(SOURCE) as source:
        image = source.convert("RGB")
    if image.size != (1280, 720):
        raise ValueError(f"Unexpected M9 source frame dimensions: {image.size}")

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 519, 47), fill=BACKGROUND)
    draw.rectangle((963, 0, 1279, 35), fill=BACKGROUND)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output, format="PNG")
    print(f"Wrote titleless figure: {args.output}")


if __name__ == "__main__":
    main()
