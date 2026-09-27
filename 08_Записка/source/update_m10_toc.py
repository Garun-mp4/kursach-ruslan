from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

from pypdf import PdfReader

from build_m10_docx import toc_headings


INTRO_SIGNATURE = "Автоматизированная сортировка деталей требует согласованной работы сенсора"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?<=[0-9a-zа-яё])[-‐‑]\s+(?=[0-9a-zа-яё])", "", text)
    text = re.sub(r"[^0-9a-zа-яё]+", " ", text)
    return " ".join(text.split())


def page_map(pdf_path: Path, markdown_path: Path) -> tuple[int, dict[str, int]]:
    pages = [(page.extract_text() or "") for page in PdfReader(pdf_path).pages]
    signature = normalize(INTRO_SIGNATURE)
    body_index = next((i for i, page in enumerate(pages) if signature in normalize(page)), None)
    if body_index is None:
        raise ValueError("Could not locate the body start using the introduction signature")

    headings = toc_headings(markdown_path.read_text(encoding="utf-8"))
    result: dict[str, int] = {}
    for heading in headings:
        needle = normalize(heading)
        matches = [i for i in range(body_index, len(pages)) if needle in normalize(pages[i])]
        if not matches:
            raise ValueError(f"Heading not found in rendered PDF: {heading}")
        result[heading] = matches[0] + 1
    return body_index + 1, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a static, page-accurate M10 contents map from the rendered PDF.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    body_page, mapping = page_map(args.pdf, args.markdown)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Body begins on printed page {body_page}; mapped {len(mapping)} contents entries")
    for heading, page in mapping.items():
        print(f"{page:>3}  {heading}")


if __name__ == "__main__":
    main()
