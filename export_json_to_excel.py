from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

HEADER_LABELS = {
    "serial_number": "Serial No",
    "epic_number": "Epic No",
    "elector_name": "Elector Name",
    "relation_type": "Relation Type",
    "relation_name": "Relation Name",
    "house_number": "House No",
    "age": "Age",
    "gender": "Gender",
    "deleted": "Deleted",
    "raw_text": "Raw Text",
}

GENDER_LABELS = {
    "male": "पुरुष",
    "female": "महिला",
}

RELATION_TYPE_LABELS = {
    "father": "पिता",
    "husband": "पति",
}


def column_name(index: int) -> str:
    name = ""
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def normalize_rows(data: object) -> list[dict[str, object]]:
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = [data]
    else:
        raise ValueError("JSON root must be an object or array of objects.")

    normalized: list[dict[str, object]] = []
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Each entry must be a JSON object.")
        normalized.append(item)
    return normalized


def collect_headers(rows: list[dict[str, object]]) -> list[str]:
    headers: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                headers.append(key)
    return headers


def display_header(header: str) -> str:
    if header in HEADER_LABELS:
        return HEADER_LABELS[header]
    return header.replace("_", " ").title()


def transform_value(header: str, value: object) -> object:
    if header == "deleted":
        return True if value is True else None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if header == "gender":
            return GENDER_LABELS.get(normalized, value)
        if header == "relation_type":
            return RELATION_TYPE_LABELS.get(normalized, value)
    return value


def serialize_cell(value: object) -> tuple[str, str]:
    if value is None:
        return "empty", ""
    if isinstance(value, bool):
        return "bool", "1" if value else "0"
    if isinstance(value, int | float) and not isinstance(value, bool):
        return "number", str(value)
    if isinstance(value, (dict, list)):
        return "string", json.dumps(value, ensure_ascii=False)
    return "string", str(value)


def build_sheet_xml(headers: list[str], rows: list[dict[str, object]]) -> str:
    parts = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "<sheetData>",
    ]

    all_rows = [
        [display_header(header) for header in headers]
    ] + [[transform_value(header, row.get(header)) for header in headers] for row in rows]

    for row_index, values in enumerate(all_rows, start=1):
        parts.append(f'<row r="{row_index}">')
        for col_index, value in enumerate(values, start=1):
            cell_ref = f"{column_name(col_index)}{row_index}"
            kind, serialized = serialize_cell(value)
            if kind == "empty":
                parts.append(f'<c r="{cell_ref}"/>')
            elif kind == "number":
                parts.append(f'<c r="{cell_ref}"><v>{escape(serialized)}</v></c>')
            elif kind == "bool":
                parts.append(f'<c r="{cell_ref}" t="b"><v>{serialized}</v></c>')
            else:
                parts.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t>{escape(serialized)}</t></is></c>'
                )
        parts.append("</row>")

    parts.extend(["</sheetData>", "</worksheet>"])
    return "".join(parts)


def build_content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def build_root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def build_workbook_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Sheet1" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>"""


def build_workbook_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""


def build_styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf/></cellStyleXfs>
  <cellXfs count="1"><xf xfId="0"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>"""


def build_core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
</cp:coreProperties>"""


def build_app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Excel</Application>
</Properties>"""


def write_xlsx(output_path: Path, headers: list[str], rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", build_content_types_xml())
        archive.writestr("_rels/.rels", build_root_rels_xml())
        archive.writestr("xl/workbook.xml", build_workbook_xml())
        archive.writestr("xl/_rels/workbook.xml.rels", build_workbook_rels_xml())
        archive.writestr("xl/styles.xml", build_styles_xml())
        archive.writestr("xl/worksheets/sheet1.xml", build_sheet_xml(headers, rows))
        archive.writestr("docProps/core.xml", build_core_xml())
        archive.writestr("docProps/app.xml", build_app_xml())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a JSON file into an Excel workbook.")
    parser.add_argument("input_json", type=Path, help="Path to the JSON file.")
    parser.add_argument("output_xlsx", type=Path, nargs="?", help="Path to the output .xlsx file.")
    args = parser.parse_args()

    input_path = args.input_json
    output_path = args.output_xlsx or input_path.with_suffix(".xlsx")

    data = json.loads(input_path.read_text(encoding="utf-8"))
    rows = normalize_rows(data)
    headers = collect_headers(rows)
    write_xlsx(output_path, headers, rows)
    print(f"Wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
