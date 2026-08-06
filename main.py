"""Single-command entrypoint for the Electoral Roll OCR pipeline."""

from __future__ import annotations

from pathlib import Path

import fitz

from app.config.settings import get_settings
from app.main import run_pipeline
from app.utils.logging import configure_logging, get_logger
from export_json_to_excel import collect_headers, normalize_rows, write_xlsx
from first_last_page_extractor import (
    build_image_sheet_spec,
    render_page_to_image,
    update_workbook,
)


# Update this path when you want to process a different PDF.
INPUT_PDF_PATH = Path("data/pdfs/UP Handia 2/2026-EROLLGEN-S24-258-SIR-FinalRoll-Revision1-HIN-2-WI.pdf")

# Leave as None to save into outputs/<pdf-name>.json automatically.
OUTPUT_JSON_PATH: Path | None = None

logger = get_logger(__name__)


def build_pipeline_input_pdf(pdf_path: Path, output_dir: Path) -> Path:
    """Create a temporary PDF for OCR by excluding page 1, page 2, and the last page."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_pdf_path = output_dir / f"{pdf_path.stem}_pipeline_input.pdf"

    with fitz.open(pdf_path) as source_document:
        page_count = source_document.page_count
        if page_count <= 3:
            raise ValueError(
                "The PDF must have at least 4 pages to exclude page 1, page 2, and the last page "
                "for the OCR pipeline."
            )

        filtered_document = fitz.open()
        # Keep only the middle content pages required by the crop/extraction flow.
        for page_index in range(2, page_count - 1):
            filtered_document.insert_pdf(
                source_document,
                from_page=page_index,
                to_page=page_index,
            )

        filtered_document.save(pipeline_pdf_path)
        filtered_document.close()

    return pipeline_pdf_path


def export_pipeline_json_to_excel(json_path: Path, excel_path: Path) -> Path:
    """Convert the pipeline JSON output into an Excel workbook."""
    import json

    data = json.loads(json_path.read_text(encoding="utf-8"))
    rows = normalize_rows(data)
    headers = collect_headers(rows)
    write_xlsx(excel_path, headers, rows)
    return excel_path


def attach_first_and_last_pdf_pages(pdf_path: Path, excel_path: Path) -> Path:
    """Embed rendered first and last PDF pages into the Excel workbook."""
    image_dir = excel_path.parent / "page_sheet_images"
    pdf_stem = pdf_path.stem

    with fitz.open(pdf_path) as document:
        first_image_path = render_page_to_image(
            document,
            0,
            image_dir / f"{pdf_stem}_first_page.png",
        )
        last_image_path = render_page_to_image(
            document,
            document.page_count - 1,
            image_dir / f"{pdf_stem}_last_page.png",
        )

    first_sheet = build_image_sheet_spec("First Page", first_image_path)
    last_sheet = build_image_sheet_spec("Last Page", last_image_path)
    update_workbook(excel_path, excel_path, first_sheet, last_sheet)
    return excel_path


def main() -> int:
    """Run the OCR pipeline using the PDF path defined in this file."""
    configure_logging()
    settings = get_settings()

    pdf_path = INPUT_PDF_PATH
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    output_path = OUTPUT_JSON_PATH or settings.outputs_dir / f"{pdf_path.stem}.json"
    final_excel_path = settings.outputs_dir / f"{pdf_path.stem}.xlsx"
    pipeline_pdf_path = build_pipeline_input_pdf(
        pdf_path,
        settings.outputs_dir / "pipeline_input_pdfs",
    )

    logger.info(
        "Starting Electoral Roll OCR pipeline",
        extra={
            "app_name": settings.app_name,
            "environment": settings.app_env,
            "pdf_path": str(pdf_path),
            "pipeline_pdf_path": str(pipeline_pdf_path),
            "output_path": str(output_path),
            "final_excel_path": str(final_excel_path),
        },
    )

    results = run_pipeline(pdf_path=pipeline_pdf_path, output_path=output_path)
    export_pipeline_json_to_excel(output_path, final_excel_path)
    attach_first_and_last_pdf_pages(pdf_path, final_excel_path)

    logger.info(
        "Pipeline completed",
        extra={
            "record_count": len(results),
            "output_path": str(output_path),
            "final_excel_path": str(final_excel_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
