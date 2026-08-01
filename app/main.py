"""Script-style entrypoint for the Electoral Roll OCR pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.config.settings import get_settings
from app.services.deleted_entry_service import DeletedEntryDetectionService
from app.services.extraction_service import ExtractionService
from app.services.grid_service import GridDetectionService
from app.services.image_service import ImageService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.pipeline_service import ElectoralRollPipelineService
from app.services.validation_service import ValidationService
from app.utils.file_utils import ensure_directory
from app.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)


class IncrementalJsonListWriter:
    """Persist records as a growing JSON list during pipeline execution."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.records: list[dict[str, Any]] = []
        ensure_directory(output_path.parent)
        self._flush()

    def append(self, record: dict[str, Any]) -> None:
        """Append one record and rewrite the JSON list on disk."""
        self.records.append(record)
        self._flush()

    def _flush(self) -> None:
        self.output_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_pipeline() -> ElectoralRollPipelineService:
    """Build the OCR pipeline with the default service graph."""
    pdf_service = PDFService()
    image_service = ImageService()
    grid_service = GridDetectionService()
    ocr_service = OCRService()
    extraction_service = ExtractionService()
    deleted_entry_service = DeletedEntryDetectionService(ocr_service=ocr_service)
    validation_service = ValidationService()

    return ElectoralRollPipelineService(
        pdf_service=pdf_service,
        image_service=image_service,
        grid_service=grid_service,
        ocr_service=ocr_service,
        extraction_service=extraction_service,
        deleted_entry_service=deleted_entry_service,
        validation_service=validation_service,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline runner."""
    parser = argparse.ArgumentParser(description="Run the Electoral Roll OCR pipeline on a PDF.")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the input electoral roll PDF.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save extracted voter records as JSON.",
    )
    return parser.parse_args()


def run_pipeline(pdf_path: Path, output_path: Path | None = None) -> list[dict[str, Any]]:
    """Run the OCR pipeline and optionally save the output JSON."""
    pipeline = build_pipeline()
    incremental_writer: IncrementalJsonListWriter | None = None
    if output_path is not None:
        incremental_writer = IncrementalJsonListWriter(output_path)

    def on_record_processed(record: Any) -> None:
        if incremental_writer is None:
            return
        incremental_writer.append(record.model_dump())

    records = pipeline.process_pdf(pdf_path, on_record_processed=on_record_processed)
    serialized_records = [record.model_dump() for record in records]

    if output_path is not None:
        logger.info("Saved pipeline output", extra={"output_path": str(output_path)})

    return serialized_records


def main() -> int:
    """Run the CLI entrypoint."""
    configure_logging()
    settings = get_settings()
    args = parse_args()

    logger.info(
        "Starting Electoral Roll OCR pipeline",
        extra={
            "app_name": settings.app_name,
            "environment": settings.app_env,
            "pdf_path": str(args.pdf),
        },
    )

    output_path = args.output
    if output_path is None:
        output_path = settings.outputs_dir / f"{args.pdf.stem}.json"

    results = run_pipeline(args.pdf, output_path)
    logger.info(
        "Pipeline completed",
        extra={"record_count": len(results), "output_path": str(output_path)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
