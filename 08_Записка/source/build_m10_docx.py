from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

from prepare_m10_front_matter import sanitize_package_metadata


TITLE = "Моделирование и симуляция робота-манипулятора, выполняющего задачу сортировки объектов по цвету"
STUDENT = "Шатуев Руслан Саматович"
GROUP = "ДИНРб-31"
SUPERVISOR = "Кузнецова В.Ю."
INSTITUTE = "Институт информационных технологий и коммуникаций"
DEPARTMENT = "Кафедра автоматизированных систем обработки информации и управления"
UNIVERSITY = "Астраханский государственный технический университет"
DISCIPLINE = "Моделирование роботов"
DOCUMENT_YEAR = "2026"

INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
FIGURE_RE = re.compile(r"^\{\{figure:\s*(.*?)\s*\|\s*caption:\s*(.*?)\}\}\s*$")
TABLE_CAPTION_RE = re.compile(r"^Таблица\s+\d+(?:\.\d+)?\s*-")
APPENDIX_HEADING_RE = re.compile(r"^ПРИЛОЖЕНИЕ\s+([А-ЯЁ])\.?\s+(.+)$", re.IGNORECASE)


def set_font(run, name: str = "Times New Roman", size: float = 12, bold: bool | None = None,
             italic: bool | None = None, color: str | None = None) -> None:
    run.font.name = name
    run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color:
        run.font.color.rgb = __import__("docx").shared.RGBColor.from_string(color)
    rpr = run._r.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for attr in ("ascii", "hAnsi", "eastAsia", "cs"):
        rfonts.set(qn(f"w:{attr}"), name)
    lang = rpr.find(qn("w:lang"))
    if lang is None:
        lang = OxmlElement("w:lang")
        rpr.append(lang)
    lang.set(qn("w:val"), "ru-RU")


def add_inline(paragraph, text: str, size: float = 12) -> None:
    pos = 0
    for match in INLINE_RE.finditer(text):
        if match.start() > pos:
            run = paragraph.add_run(text[pos:match.start()])
            set_font(run, size=size)
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            set_font(run, size=size, bold=True)
        elif token.startswith("`"):
            run = paragraph.add_run(token[1:-1])
            set_font(run, name="Consolas", size=size)
        else:
            run = paragraph.add_run(token[1:-1])
            set_font(run, size=size, italic=True)
        pos = match.end()
    if pos < len(text):
        run = paragraph.add_run(text[pos:])
        set_font(run, size=size)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    shd.set(qn("w:val"), "clear")
    tc_pr.append(shd)


def set_cell_margins(cell, top: int = 70, start: int = 85, bottom: int = 70, end: int = 85) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color: str = "000000", size: str = "5") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def remove_table_borders(table) -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = qn(f"w:{edge}")
        element = borders.find(tag)
        if element is None:
            element = OxmlElement(f"w:{edge}")
            borders.append(element)
        element.set(qn("w:val"), "nil")


def add_table(doc: Document, rows: list[list[str]], section) -> None:
    if not rows:
        return
    col_count = len(rows[0])
    normalized = [r + [""] * (col_count - len(r)) for r in rows]
    table = doc.add_table(rows=len(normalized), cols=col_count)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    set_table_borders(table)
    total_cm = (section.page_width - section.left_margin - section.right_margin) / 360000
    maxima = []
    for col in range(col_count):
        longest = max((len(row[col]) for row in normalized), default=8)
        maxima.append(min(max(longest, 7), 34) ** 0.65)
    total_weight = sum(maxima)
    widths = [Cm(total_cm * weight / total_weight) for weight in maxima]
    for row_idx, rowdata in enumerate(normalized):
        row = table.rows[row_idx]
        prevent_row_split(row)
        for col_idx, value in enumerate(rowdata):
            cell = row.cells[col_idx]
            cell.width = widths[col_idx]
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_margins(cell)
            shade_cell(cell, "FFFFFF")
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if row_idx == 0 else WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.first_line_indent = Cm(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            add_inline(p, value, size=12)
            if row_idx == 0:
                for run in p.runs:
                    run.bold = True
        if row_idx == 0:
            set_repeat_table_header(row)


def parse_table(lines: list[str], start: int) -> tuple[list[list[str]], int]:
    rows: list[list[str]] = []
    i = start
    while i < len(lines) and "|" in lines[i]:
        cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
        if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells):
            rows.append(cells)
        i += 1
    return rows, i


def add_page_number(paragraph) -> None:
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    fonts = OxmlElement("w:rFonts")
    fonts.set(qn("w:ascii"), "Times New Roman")
    fonts.set(qn("w:hAnsi"), "Times New Roman")
    rpr.append(fonts)
    size = OxmlElement("w:sz")
    size.set(qn("w:val"), "24")
    rpr.append(size)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = "1"
    r.append(t)
    fld.append(r)
    paragraph._p.append(fld)


def body_heading_level(markdown_level: int) -> int:
    """Map Markdown headings (which include a discarded annotation title) to Word outline levels."""
    return max(1, min(markdown_level - 1, 3))


def toc_headings(markdown: str) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    body_started = False
    for line in markdown.splitlines():
        match = HEADING_RE.match(line.strip())
        if not match:
            continue
        level, title = len(match.group(1)), match.group(2)
        if title.upper() == "ВВЕДЕНИЕ":
            body_started = True
        if body_started and level <= 4:
            entries.append((body_heading_level(level), title))
    titles = [title for _, title in entries]
    duplicates = sorted({title for title in titles if titles.count(title) > 1})
    if duplicates:
        raise ValueError(f"Duplicate heading text makes the TOC page map ambiguous: {duplicates}")
    return entries


def _field_run(field_type: str) -> object:
    run = OxmlElement("w:r")
    field = OxmlElement("w:fldChar")
    field.set(qn("w:fldCharType"), field_type)
    run.append(field)
    return run


def _text_run(text: str) -> object:
    run = OxmlElement("w:r")
    node = OxmlElement("w:t")
    if text[:1].isspace() or text[-1:].isspace():
        node.set(qn("xml:space"), "preserve")
    node.text = text
    run.append(node)
    return run


def _toc_entry_paragraph(doc: Document, level: int, title: str, page: str,
                         *, field_start: bool = False, field_end: bool = False):
    p = OxmlElement("w:p")
    ppr = OxmlElement("w:pPr")
    pstyle = OxmlElement("w:pStyle")
    pstyle.set(qn("w:val"), doc.styles[f"toc {level}"].style_id)
    ppr.append(pstyle)
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "right")
    tab.set(qn("w:leader"), "dot")
    tab.set(qn("w:pos"), "9975")
    tabs.append(tab)
    ppr.append(tabs)
    p.append(ppr)
    if field_start:
        p.append(_field_run("begin"))
        instruction = OxmlElement("w:r")
        instr_text = OxmlElement("w:instrText")
        instr_text.set(qn("xml:space"), "preserve")
        instr_text.text = ' TOC \\o "1-3" \\h \\z \\u '
        instruction.append(instr_text)
        p.append(instruction)
        p.append(_field_run("separate"))
    p.append(_text_run(title))
    tab_run = OxmlElement("w:r")
    tab_run.append(OxmlElement("w:tab"))
    p.append(tab_run)
    p.append(_text_run(page))
    if field_end:
        p.append(_field_run("end"))
    return p


def populate_reference_toc(doc: Document, entries: list[tuple[int, str]],
                           pages: dict[str, str]) -> None:
    controls = [element for element in doc.element.body if element.tag == qn("w:sdt")]
    control = next(
        (
            item for item in controls
            if item.find(qn("w:sdtPr")) is not None
            and item.find(qn("w:sdtPr")).find(qn("w:docPartObj")) is not None
            and item.find(qn("w:sdtPr")).find(qn("w:docPartObj")).find(qn("w:docPartGallery")) is not None
            and item.find(qn("w:sdtPr")).find(qn("w:docPartObj")).find(qn("w:docPartGallery")).get(qn("w:val")) == "Table of Contents"
        ),
        None,
    )
    if control is None:
        raise ValueError("AGTU template has no Table of Contents content control")
    content = control.find(qn("w:sdtContent"))
    if content is None:
        raise ValueError("Table of Contents content control has no content container")
    if not entries:
        raise ValueError("Markdown body has no headings to include in the table of contents")

    titles = [title for _, title in entries]
    missing = [title for title in titles if title not in pages]
    if missing and pages:
        raise ValueError(f"No rendered page number was supplied for TOC headings: {missing}")
    for child in list(content):
        content.remove(child)
    for index, (level, title) in enumerate(entries):
        if level not in (1, 2, 3):
            raise ValueError(f"Unsupported TOC heading level: {level}")
        content.append(
            _toc_entry_paragraph(
                doc,
                level,
                title,
                str(pages.get(title, "?")),
                field_start=index == 0,
                field_end=index == len(entries) - 1,
            )
        )

    title_paragraph = next(
        (p for p in doc.paragraphs if p.text.strip().casefold() == "содержание"),
        None,
    )
    if title_paragraph is None:
        raise ValueError("AGTU template is missing the contents heading")
    if title_paragraph.runs:
        title_paragraph.runs[0].text = "СОДЕРЖАНИЕ"
        for run in title_paragraph.runs[1:]:
            run.text = ""
    else:
        title_paragraph.add_run("СОДЕРЖАНИЕ")
    for run in title_paragraph.runs:
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
        run.bold = True


def add_bottom_rule(paragraph) -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        p_pr.append(borders)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), "5")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "000000")
    borders.append(bottom)


def add_running_header(section) -> None:
    section.header_distance = Cm(1.0)
    section.footer_distance = Cm(0.8)
    section.header.is_linked_to_previous = False
    header = section.header
    # The imported template has its own running header in this section. Remove
    # every old paragraph/table before writing the project-specific header.
    for child in list(header._element):
        header._element.remove(child)
    first = header.add_paragraph()
    first.clear()
    first.alignment = WD_ALIGN_PARAGRAPH.CENTER
    first.paragraph_format.first_line_indent = Cm(0)
    first.paragraph_format.space_after = Pt(0)
    add_page_number(first)
    for run in first.runs:
        set_font(run, size=12)

    second = header.add_paragraph()
    second.alignment = WD_ALIGN_PARAGRAPH.LEFT
    second.paragraph_format.first_line_indent = Cm(0)
    second.paragraph_format.space_before = Pt(0)
    second.paragraph_format.space_after = Pt(0)
    second.paragraph_format.tab_stops.add_tab_stop(Cm(17.5), WD_TAB_ALIGNMENT.RIGHT)
    left = second.add_run("Кафедра АСОИУ")
    set_font(left, size=12, italic=True)
    right = second.add_run("\tМоделирование роботов  |  Курсовой проект")
    set_font(right, size=12, italic=True)
    add_bottom_rule(second)


def configure_style(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(12)
    for style_name, size, align, before, after in (
        ("Heading 1", 12, WD_ALIGN_PARAGRAPH.CENTER, 12, 8),
        ("Heading 2", 12, WD_ALIGN_PARAGRAPH.LEFT, 10, 5),
        ("Heading 3", 12, WD_ALIGN_PARAGRAPH.LEFT, 7, 4),
    ):
        style = doc.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.italic = False
        style.paragraph_format.alignment = align
        style.paragraph_format.first_line_indent = Cm(0)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True

    for style in doc.styles:
        if style.name.casefold().startswith("toc "):
            style.font.name = "Times New Roman"
            style.font.size = Pt(12)
            style.paragraph_format.line_spacing = 1.0
            style.paragraph_format.space_before = Pt(0)
            style.paragraph_format.space_after = Pt(0)


def add_cover(doc: Document, project_root: Path) -> None:
    section = doc.sections[0]
    width_cm = (section.page_width - section.left_margin - section.right_margin) / 360000
    head = doc.add_table(rows=1, cols=2)
    head.alignment = WD_TABLE_ALIGNMENT.CENTER
    head.autofit = False
    remove_table_borders(head)
    head.columns[0].width = Cm(3.3)
    head.columns[1].width = Cm(width_cm - 3.3)
    for cell, width in zip(head.rows[0].cells, (3.3, width_cm - 3.3)):
        cell.width = Cm(width)
        set_cell_margins(cell, 0, 0, 0, 0)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
    logo = project_root / "08_Записка" / "figures" / "00_Эмблема_АГТУ_из_референса.png"
    p = head.cell(0, 0).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Cm(0)
    p.add_run().add_picture(str(logo), width=Cm(3.0))
    p = head.cell(0, 1).paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1.0
    institutional_lines = (
        "Федеральное агентство Российской Федерации по рыболовству",
        "Федеральное государственное бюджетное образовательное учреждение",
        "высшего образования",
        f"«{UNIVERSITY}»",
    )
    for index, line in enumerate(institutional_lines):
        run = p.add_run(line + ("\n" if index < len(institutional_lines) - 1 else ""))
        set_font(run, size=10.5, bold=True, italic=True)

    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.left_indent = Cm(0)
    p.paragraph_format.space_before = Pt(30)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.15
    for line in (INSTITUTE, DEPARTMENT):
        run = p.add_run(line + "\n")
        set_font(run, size=12)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(110)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run("КУРСОВОЙ ПРОЕКТ")
    set_font(run, size=16, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(TITLE)
    set_font(run, size=20, bold=True)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run(f"по дисциплине «{DISCIPLINE}»")
    set_font(run, size=12)

    blocks = doc.add_table(rows=1, cols=2)
    blocks.alignment = WD_TABLE_ALIGNMENT.CENTER
    blocks.autofit = False
    remove_table_borders(blocks)
    blocks.columns[0].width = Cm(width_cm / 2)
    blocks.columns[1].width = Cm(width_cm / 2)
    for cell in blocks.rows[0].cells:
        cell.width = Cm(width_cm / 2)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.TOP
        set_cell_margins(cell, 0, 0, 0, 0)
    blocks.rows[0].cells[0].paragraphs[0].paragraph_format.space_before = Pt(30)
    blocks.rows[0].cells[1].paragraphs[0].paragraph_format.space_before = Pt(30)
    cover_blocks = (
        ("Допущен к защите\n«____» ______________ 20___ г.\n\n"
         "Руководитель\nКузнецова В.Ю. __________________\n\n"
         "Оценка на защите: «____________»\nДата защиты: __________________", WD_ALIGN_PARAGRAPH.LEFT),
        ("Проект выполнен\nобучающимся группы " + GROUP + "\n" + STUDENT +
         "\n______________________________\n\nРуководитель курсового проекта\n" + SUPERVISOR +
         " __________________", WD_ALIGN_PARAGRAPH.LEFT),
    )
    for cell, (text, alignment) in zip(blocks.rows[0].cells, cover_blocks):
        p = cell.paragraphs[0]
        p.alignment = alignment
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.line_spacing = 1.12
        p.paragraph_format.space_after = Pt(0)
        for line_index, line in enumerate(text.split("\n")):
            if line_index:
                p.add_run().add_break()
            run = p.add_run(line)
            set_font(run, size=11.5)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(115)
    p.paragraph_format.keep_together = True
    run = p.add_run(f"Астрахань — {DOCUMENT_YEAR}")
    set_font(run, size=12, bold=True)


def parse_annotation(markdown: str) -> tuple[str, str]:
    match = re.search(r"(?ms)^# АННОТАЦИЯ\s*\n(.*?)(?=^# ВВЕДЕНИЕ\s*$)", markdown)
    if match is None:
        raise ValueError("Markdown must contain an annotation before # ВВЕДЕНИЕ")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", match.group(1)) if block.strip()]
    annotation = next((block for block in blocks if not block.startswith("**Ключевые слова:**")), None)
    keywords = next((block for block in blocks if block.startswith("**Ключевые слова:**")), None)
    if annotation is None or keywords is None:
        raise ValueError("Markdown annotation must include a summary and keywords")
    return " ".join(annotation.split()), " ".join(keywords.split())


def add_front_pages(doc: Document, annotation: str, keywords: str, project_root: Path,
                    toc_entries: list[str], toc_pages: dict[str, str]) -> None:
    add_cover(doc, project_root)
    doc.add_page_break()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run("АННОТАЦИЯ")
    set_font(r, size=12, bold=True)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(1.25)
    p.paragraph_format.line_spacing = 1.5
    add_inline(p, annotation)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.space_before = Pt(10)
    add_inline(p, keywords, size=12)
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Cm(0)
    r = p.add_run("Автор: ")
    set_font(r, bold=True)
    r = p.add_run(STUDENT)
    set_font(r)
    doc.add_page_break()
    add_toc(doc, toc_entries, toc_pages)


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph(style=f"Heading {min(level, 3)}")
    p.paragraph_format.keep_with_next = True
    p_pr = p._p.get_or_add_pPr()
    suppress_hyphens = OxmlElement("w:suppressAutoHyphens")
    suppress_hyphens.set(qn("w:val"), "true")
    p_pr.append(suppress_hyphens)
    if level == 1 and text != "ВВЕДЕНИЕ":
        p.paragraph_format.page_break_before = True
    appendix = APPENDIX_HEADING_RE.match(text) if level == 1 else None
    if appendix:
        label, subtitle = appendix.groups()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.first_line_indent = Cm(0)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(8.75), WD_TAB_ALIGNMENT.CENTER)
        p.paragraph_format.tab_stops.add_tab_stop(Cm(17.5), WD_TAB_ALIGNMENT.RIGHT)
        prefix = p.add_run("\t\t")
        set_font(prefix, size=12, bold=True)
        label_run = p.add_run(f"ПРИЛОЖЕНИЕ {label.upper()}")
        set_font(label_run, size=12, bold=True)
        label_run.add_break()
        subtitle_prefix = p.add_run("\t")
        set_font(subtitle_prefix, size=12, bold=True)
        subtitle_run = p.add_run(subtitle)
        set_font(subtitle_run, size=12, bold=True)
        return
    p.paragraph_format.first_line_indent = Cm(0)
    add_inline(p, text, size=12)
    for run in p.runs:
        run.bold = True
        run.italic = False


def add_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(1.25)
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.widow_control = True
    add_inline(p, text)


def add_caption(doc: Document, text: str, center: bool = False) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = not center
    add_inline(p, text, size=12)


def add_figure(doc: Document, image_path: Path, caption: str, section) -> None:
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.keep_together = True
    run = p.add_run()
    available_width = section.page_width - section.left_margin - section.right_margin
    picture = run.add_picture(str(image_path), width=available_width)
    max_height = Cm(13.5)
    if picture.height > max_height:
        scale = max_height / picture.height
        picture.width = int(picture.width * scale)
        picture.height = max_height
    add_caption(doc, caption, center=True)


def add_equation(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_together = True
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run(text.strip())
    set_font(r, name="Cambria Math", size=12)


def set_page_number_start(section, start: int = 1) -> None:
    existing = section._sectPr.find(qn("w:pgNumType"))
    if existing is not None:
        section._sectPr.remove(existing)
    page_type = OxmlElement("w:pgNumType")
    page_type.set(qn("w:start"), str(start))
    section._sectPr.append(page_type)


def normalize_document_font_sizes(doc: Document) -> None:
    """Apply 12 pt to the thesis body, preserving the reference front matter."""
    body_start = next(
        (index for index, paragraph in enumerate(doc.paragraphs)
         if paragraph.text.strip() == "ВВЕДЕНИЕ"),
        None,
    )
    if body_start is None:
        raise ValueError("Cannot normalize fonts: body heading 'ВВЕДЕНИЕ' is missing")

    for paragraph in doc.paragraphs[body_start:]:
        for run in paragraph.runs:
            run.font.size = Pt(12)

    # The template's first six tables belong to its reference-formatted cover,
    # assignment, and calendar pages; body tables are added after them.
    for table in doc.tables[6:]:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(12)

    body_section = doc.sections[-1]
    for part in (body_section.header, body_section.footer,
                 body_section.first_page_header, body_section.first_page_footer):
        for paragraph in part.paragraphs:
            for run in paragraph.runs:
                run.font.size = Pt(12)
        for table in part.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        for run in paragraph.runs:
                            run.font.size = Pt(12)

    for style in doc.styles:
        if hasattr(style, "font"):
            style.font.name = "Times New Roman"
            style.font.size = Pt(12)


def build(input_path: Path, output_path: Path, project_root: Path, template_path: Path,
          toc_pages: dict[str, str], body_page_start: int) -> None:
    text = input_path.read_text(encoding="utf-8")
    doc = Document(template_path)
    configure_style(doc)
    populate_reference_toc(doc, toc_headings(text), toc_pages)
    body_section = doc.sections[-1]
    body_section.page_width = Cm(21)
    body_section.page_height = Cm(29.7)
    body_section.left_margin = Cm(2)
    body_section.right_margin = Cm(1.5)
    body_section.top_margin = Cm(2.5)
    body_section.bottom_margin = Cm(1.5)
    body_section.header_distance = Cm(1.0)
    body_section.footer_distance = Cm(0.8)
    body_section.different_first_page_header_footer = False
    add_running_header(body_section)
    set_page_number_start(body_section, body_page_start)

    lines = text.splitlines()
    body_start = next((i for i, line in enumerate(lines) if line.strip() == "# ВВЕДЕНИЕ"), None)
    if body_start is None:
        raise ValueError("Markdown does not contain the required # ВВЕДЕНИЕ body heading")
    i = body_start
    paragraph: list[str] = []
    in_code = False
    code: list[str] = []
    in_equation = False
    equation: list[str] = []

    def flush() -> None:
        nonlocal paragraph
        if paragraph:
            add_body(doc, " ".join(part.strip() for part in paragraph))
            paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            if in_code:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Cm(0.8)
                p.paragraph_format.right_indent = Cm(0.4)
                p.paragraph_format.first_line_indent = Cm(0)
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(8)
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.keep_together = True
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                r = p.add_run("\n".join(code))
                set_font(r, name="Consolas", size=12)
                code = []
                in_code = False
            else:
                in_code = True
            i += 1
            continue
        if in_code:
            code.append(line)
            i += 1
            continue
        if stripped == "$$":
            flush()
            if in_equation:
                add_equation(doc, " ".join(equation))
                equation = []
                in_equation = False
            else:
                in_equation = True
            i += 1
            continue
        if in_equation:
            equation.append(stripped)
            i += 1
            continue
        if not stripped:
            flush()
            i += 1
            continue
        heading = HEADING_RE.match(stripped)
        if heading:
            flush()
            title = heading.group(2)
            markdown_level = len(heading.group(1))
            word_level = body_heading_level(markdown_level)
            add_heading(doc, title, word_level)
            i += 1
            continue
        figure = FIGURE_RE.match(stripped)
        if figure:
            flush()
            add_figure(doc, (input_path.parent.parent / figure.group(1)).resolve(), figure.group(2), body_section)
            i += 1
            continue
        if TABLE_CAPTION_RE.match(stripped):
            flush()
            add_caption(doc, stripped)
            i += 1
            continue
        if i + 1 < len(lines) and "|" in stripped and re.match(r"^\s*\|?\s*:?-{3,}", lines[i + 1]):
            flush()
            rows, i = parse_table(lines, i)
            add_table(doc, rows, body_section)
            continue
        bullet = re.match(r"^\s*-\s+(.+)$", line)
        if bullet:
            flush()
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8)
            p.paragraph_format.first_line_indent = Cm(-0.45)
            p.paragraph_format.line_spacing = 1.25
            add_inline(p, "— " + bullet.group(1), size=12)
            i += 1
            continue
        paragraph.append(line)
        i += 1

    flush()
    settings = doc.settings._element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")
    normalize_document_font_sizes(doc)
    doc.core_properties.title = TITLE
    doc.core_properties.subject = "Курсовая работа по моделированию и симуляции робота-манипулятора"
    doc.core_properties.author = STUDENT
    doc.core_properties.keywords = "SCARA, сортировка по цвету, симуляция, MuJoCo"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    sanitize_package_metadata(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).with_name("M10_шаблон_стиля_АГТУ.docx"))
    parser.add_argument("--toc-pages", type=Path,
                        help="JSON object mapping major heading text to final printed page number")
    parser.add_argument("--body-page-start", type=int, default=1,
                        help="printed page number of the first body page")
    args = parser.parse_args()
    toc_pages = json.loads(args.toc_pages.read_text(encoding="utf-8")) if args.toc_pages else {}
    build(args.input.resolve(), args.output.resolve(), args.project_root.resolve(),
          args.template.resolve(), toc_pages, args.body_page_start)
    print(f"Created: {args.output.resolve()}")


if __name__ == "__main__":
    main()
