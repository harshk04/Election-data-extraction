from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

import fitz
from PIL import Image

# Update these paths before running the script.
INPUT_PDF_PATH = Path("data/pdfs/2026-EROLLGEN-S24-46-SIR-FinalRoll-Revision1-HIN-1-WI (1).pdf")
INPUT_EXCEL_PATH = Path("outputs/File-1.xlsx")

# Leave as None to update INPUT_EXCEL_PATH in place.
OUTPUT_EXCEL_PATH: Path | None = None

# Render scale for PDF pages before inserting them into Excel.
PAGE_RENDER_SCALE = 2.0


XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
APP_NS = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT_NS = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"
XDR_NS = "http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"

NS = {
    "main": XML_NS,
    "r": REL_NS,
    "pr": PKG_REL_NS,
    "ct": CT_NS,
    "app": APP_NS,
    "vt": VT_NS,
}

ET.register_namespace("", XML_NS)
ET.register_namespace("r", REL_NS)


@dataclass(frozen=True)
class ImageSheetSpec:
    name: str
    image_path: Path
    image_width_px: int
    image_height_px: int


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")


def sanitize_sheet_name(name: str, fallback: str) -> str:
    candidate = (name or "").strip() or fallback
    for char in '[]:*?/\\':
        candidate = candidate.replace(char, "-")
    return candidate[:31] or fallback


def unique_sheet_name(name: str, used_names: Iterable[str]) -> str:
    seen = {item.lower() for item in used_names}
    if name.lower() not in seen:
        return name

    base = name[:28] or "Sheet"
    counter = 2
    while True:
        candidate = f"{base}_{counter}"[:31]
        if candidate.lower() not in seen:
            return candidate
        counter += 1


def render_page_to_image(document: fitz.Document, page_index: int, output_path: Path) -> Path:
    log(f"Rendering PDF page {page_index + 1} to image: {output_path}")
    page = document.load_page(page_index)
    pixmap = page.get_pixmap(matrix=fitz.Matrix(PAGE_RENDER_SCALE, PAGE_RENDER_SCALE), alpha=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(output_path)
    return output_path


def get_image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.width, image.height


def build_image_sheet_spec(sheet_name: str, image_path: Path) -> ImageSheetSpec:
    width_px, height_px = get_image_size(image_path)
    return ImageSheetSpec(
        name=sheet_name,
        image_path=image_path,
        image_width_px=width_px,
        image_height_px=height_px,
    )


def pixels_to_emu(pixels: int) -> int:
    return int(pixels * 9525)


def build_image_worksheet_xml() -> bytes:
    row_count = 120
    rows = []
    for row_number in range(1, row_count + 1):
        rows.append(f'<row r="{row_number}" ht="20" customHeight="1"/>')

    cols = []
    for column_index in range(1, 21):
        cols.append(f'<col min="{column_index}" max="{column_index}" width="14" customWidth="1"/>')

    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="{XML_NS}" xmlns:r="{REL_NS}">
  <dimension ref="A1:T120"/>
  <sheetViews>
    <sheetView workbookViewId="0" showGridLines="0"/>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="20"/>
  <cols>
    {''.join(cols)}
  </cols>
  <sheetData>
    {''.join(rows)}
  </sheetData>
  <pageMargins left="0.25" right="0.25" top="0.25" bottom="0.25" header="0.1" footer="0.1"/>
  <drawing r:id="rId1"/>
</worksheet>"""
    return xml.encode("utf-8")


def build_worksheet_drawing_rels_xml(drawing_filename: str) -> bytes:
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing" Target="../drawings/{drawing_filename}"/>
</Relationships>"""
    return xml.encode("utf-8")


def build_drawing_xml(image_width_px: int, image_height_px: int) -> bytes:
    width_emu = pixels_to_emu(image_width_px)
    height_emu = pixels_to_emu(image_height_px)
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<xdr:wsDr xmlns:xdr="{XDR_NS}" xmlns:a="{A_NS}" xmlns:r="{REL_NS}">
  <xdr:oneCellAnchor>
    <xdr:from>
      <xdr:col>0</xdr:col>
      <xdr:colOff>0</xdr:colOff>
      <xdr:row>0</xdr:row>
      <xdr:rowOff>0</xdr:rowOff>
    </xdr:from>
    <xdr:ext cx="{width_emu}" cy="{height_emu}"/>
    <xdr:pic>
      <xdr:nvPicPr>
        <xdr:cNvPr id="1" name="PagePreview"/>
        <xdr:cNvPicPr/>
      </xdr:nvPicPr>
      <xdr:blipFill>
        <a:blip r:embed="rId1"/>
        <a:stretch>
          <a:fillRect/>
        </a:stretch>
      </xdr:blipFill>
      <xdr:spPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="{width_emu}" cy="{height_emu}"/>
        </a:xfrm>
        <a:prstGeom prst="rect">
          <a:avLst/>
        </a:prstGeom>
      </xdr:spPr>
    </xdr:pic>
    <xdr:clientData/>
  </xdr:oneCellAnchor>
</xdr:wsDr>"""
    return xml.encode("utf-8")


def build_drawing_rels_xml(media_filename: str) -> bytes:
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{PKG_REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_filename}"/>
</Relationships>"""
    return xml.encode("utf-8")


def _next_numeric_suffix(names: Iterable[str], prefix: str, suffix: str, default: int = 1) -> int:
    values: list[int] = []
    for name in names:
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        middle = name[len(prefix) : -len(suffix)]
        if middle.isdigit():
            values.append(int(middle))
    return (max(values) + 1) if values else default


def _next_rid(existing_ids: Iterable[str]) -> str:
    numbers = []
    for value in existing_ids:
        if value.startswith("rId") and value[3:].isdigit():
            numbers.append(int(value[3:]))
    return f"rId{(max(numbers) + 1) if numbers else 1}"


def add_png_content_type_if_missing(content_types_root: ET.Element) -> None:
    for default in content_types_root.findall("ct:Default", NS):
        if default.attrib.get("Extension") == "png":
            return

    default = ET.Element(f"{{{CT_NS}}}Default")
    default.set("Extension", "png")
    default.set("ContentType", "image/png")
    content_types_root.append(default)


def add_override_if_missing(content_types_root: ET.Element, part_name: str, content_type: str) -> None:
    for override in content_types_root.findall("ct:Override", NS):
        if override.attrib.get("PartName") == part_name:
            return

    override = ET.Element(f"{{{CT_NS}}}Override")
    override.set("PartName", part_name)
    override.set("ContentType", content_type)
    content_types_root.append(override)


def update_workbook(input_path: Path, output_path: Path, first_sheet: ImageSheetSpec, last_sheet: ImageSheetSpec) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Excel file not found: {input_path}")

    log(f"Opening existing Excel workbook: {input_path}")
    with ZipFile(input_path) as source_zip:
        files = {info.filename: source_zip.read(info.filename) for info in source_zip.infolist()}

    workbook_root = ET.fromstring(files["xl/workbook.xml"])
    workbook_rels_root = ET.fromstring(files["xl/_rels/workbook.xml.rels"])
    content_types_root = ET.fromstring(files["[Content_Types].xml"])

    sheets_parent = workbook_root.find("main:sheets", NS)
    if sheets_parent is None:
        raise ValueError("Workbook does not contain any sheets.")

    existing_sheets = list(sheets_parent.findall("main:sheet", NS))
    existing_names = [sheet.attrib.get("name", "") for sheet in existing_sheets]

    first_sheet = ImageSheetSpec(
        name=unique_sheet_name(first_sheet.name, existing_names),
        image_path=first_sheet.image_path,
        image_width_px=first_sheet.image_width_px,
        image_height_px=first_sheet.image_height_px,
    )
    last_sheet = ImageSheetSpec(
        name=unique_sheet_name(last_sheet.name, [first_sheet.name, *existing_names]),
        image_path=last_sheet.image_path,
        image_width_px=last_sheet.image_width_px,
        image_height_px=last_sheet.image_height_px,
    )

    next_sheet_id = max((int(sheet.attrib.get("sheetId", "0")) for sheet in existing_sheets), default=0) + 1
    next_sheet_xml_index = _next_numeric_suffix(
        [name.rsplit("/", 1)[-1] for name in files if name.startswith("xl/worksheets/sheet")],
        "sheet",
        ".xml",
        default=1,
    )
    next_drawing_index = _next_numeric_suffix(
        [name.rsplit("/", 1)[-1] for name in files if name.startswith("xl/drawings/drawing")],
        "drawing",
        ".xml",
        default=1,
    )
    next_media_index = _next_numeric_suffix(
        [name.rsplit("/", 1)[-1] for name in files if name.startswith("xl/media/image")],
        "image",
        ".png",
        default=1,
    )
    existing_rel_ids = [rel.attrib.get("Id", "") for rel in workbook_rels_root.findall("pr:Relationship", NS)]

    sheet_entries = [
        (
            first_sheet,
            f"sheet{next_sheet_xml_index}.xml",
            f"drawing{next_drawing_index}.xml",
            f"image{next_media_index}.png",
            _next_rid(existing_rel_ids),
            str(next_sheet_id),
        ),
        (
            last_sheet,
            f"sheet{next_sheet_xml_index + 1}.xml",
            f"drawing{next_drawing_index + 1}.xml",
            f"image{next_media_index + 1}.png",
            "",
            str(next_sheet_id + 1),
        ),
    ]

    existing_rel_ids.append(sheet_entries[0][4])
    sheet_entries[1] = (
        sheet_entries[1][0],
        sheet_entries[1][1],
        sheet_entries[1][2],
        sheet_entries[1][3],
        _next_rid(existing_rel_ids),
        sheet_entries[1][5],
    )

    new_sheet_elements = []
    for spec, sheet_filename, _, _, rid, sheet_id in sheet_entries:
        sheet_element = ET.Element(f"{{{XML_NS}}}sheet")
        sheet_element.set("name", spec.name)
        sheet_element.set("sheetId", sheet_id)
        sheet_element.set(f"{{{REL_NS}}}id", rid)
        new_sheet_elements.append(sheet_element)

        relationship = ET.Element(f"{{{PKG_REL_NS}}}Relationship")
        relationship.set("Id", rid)
        relationship.set(
            "Type",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
        )
        relationship.set("Target", f"worksheets/{sheet_filename}")
        workbook_rels_root.append(relationship)

    for sheet in existing_sheets:
        sheets_parent.remove(sheet)

    log("Reordering workbook sheets so the new first page becomes sheet 1 and last page becomes sheet 3")
    sheets_parent.append(new_sheet_elements[0])
    if existing_sheets:
        sheets_parent.append(existing_sheets[0])
    sheets_parent.append(new_sheet_elements[1])
    for sheet in existing_sheets[1:]:
        sheets_parent.append(sheet)

    add_png_content_type_if_missing(content_types_root)

    for spec, sheet_filename, drawing_filename, media_filename, _, _ in sheet_entries:
        log(f"Embedding image into worksheet {spec.name}: {spec.image_path}")
        files[f"xl/worksheets/{sheet_filename}"] = build_image_worksheet_xml()
        files[f"xl/worksheets/_rels/{sheet_filename}.rels"] = build_worksheet_drawing_rels_xml(
            drawing_filename
        )
        files[f"xl/drawings/{drawing_filename}"] = build_drawing_xml(
            spec.image_width_px,
            spec.image_height_px,
        )
        files[f"xl/drawings/_rels/{drawing_filename}.rels"] = build_drawing_rels_xml(media_filename)
        files[f"xl/media/{media_filename}"] = spec.image_path.read_bytes()

        add_override_if_missing(
            content_types_root,
            f"/xl/worksheets/{sheet_filename}",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml",
        )
        add_override_if_missing(
            content_types_root,
            f"/xl/drawings/{drawing_filename}",
            "application/vnd.openxmlformats-officedocument.drawing+xml",
        )

    _update_app_properties(
        files,
        [sheet.attrib.get("name", "") for sheet in sheets_parent.findall("main:sheet", NS)],
    )

    files["xl/workbook.xml"] = ET.tostring(workbook_root, encoding="utf-8", xml_declaration=True)
    files["xl/_rels/workbook.xml.rels"] = ET.tostring(
        workbook_rels_root,
        encoding="utf-8",
        xml_declaration=True,
    )
    files["[Content_Types].xml"] = ET.tostring(
        content_types_root,
        encoding="utf-8",
        xml_declaration=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing updated workbook: {output_path}")
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as target_zip:
        for filename, content in files.items():
            target_zip.writestr(filename, content)


def _update_app_properties(files: dict[str, bytes], sheet_names: list[str]) -> None:
    app_xml_path = "docProps/app.xml"
    if app_xml_path not in files:
        return

    try:
        root = ET.fromstring(files[app_xml_path])
    except ET.ParseError:
        return

    heading_pairs = root.find("app:HeadingPairs", NS)
    titles_of_parts = root.find("app:TitlesOfParts", NS)
    if heading_pairs is None or titles_of_parts is None:
        return

    vector = heading_pairs.find("vt:vector", NS)
    if vector is not None:
        variants = vector.findall("vt:variant", NS)
        if len(variants) >= 2:
            i4 = variants[1].find("vt:i4", NS)
            if i4 is not None:
                i4.text = str(len(sheet_names))

    parts_vector = titles_of_parts.find("vt:vector", NS)
    if parts_vector is not None:
        parts_vector.attrib["size"] = str(len(sheet_names))
        for child in list(parts_vector):
            parts_vector.remove(child)
        for sheet_name in sheet_names:
            lpstr = ET.SubElement(parts_vector, f"{{{VT_NS}}}lpstr")
            lpstr.text = sheet_name

    files[app_xml_path] = ET.tostring(root, encoding="utf-8", xml_declaration=True)


def run() -> Path:
    pdf_path = INPUT_PDF_PATH
    excel_path = INPUT_EXCEL_PATH
    output_path = OUTPUT_EXCEL_PATH or excel_path

    log(f"Starting first/last page image export for PDF: {pdf_path}")
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    document = fitz.open(pdf_path)
    if document.page_count < 2:
        raise ValueError("PDF must have at least two pages to extract first and last pages.")

    image_dir = Path("outputs/page_sheet_images")
    pdf_stem = pdf_path.stem

    first_image_path = render_page_to_image(document, 0, image_dir / f"{pdf_stem}_first_page.png")
    last_image_path = render_page_to_image(
        document,
        document.page_count - 1,
        image_dir / f"{pdf_stem}_last_page.png",
    )

    first_sheet = build_image_sheet_spec("First Page", first_image_path)
    last_sheet = build_image_sheet_spec("Last Page", last_image_path)

    update_workbook(excel_path, output_path, first_sheet, last_sheet)
    log("Completed successfully")
    return output_path


if __name__ == "__main__":
    updated_workbook = run()
    print(f"Workbook updated: {updated_workbook}")
