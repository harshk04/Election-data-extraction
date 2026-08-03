"""Library entrypoint for the end-to-end electoral roll pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config.settings import get_settings
from app.services.classified_crop_extraction_service import ClassifiedCropExtractionService
from app.services.deleted_entry_service import DeletedEntryDetectionService
from app.services.grid_service import GridDetectionService
from app.services.image_service import ImageService
from app.services.llm_extraction_service import LLMExtractionService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.pipeline_service import ElectoralRollPipelineService
from app.utils.file_utils import ensure_directory
from app.utils.logging import get_logger

logger = get_logger(__name__)


class IncrementalJsonListWriter:
    """Persist records as a growing JSON list during pipeline execution."""

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.records: list[dict[str, Any]] = []
        ensure_directory(output_path.parent)
        self._flush()

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self._flush()

    def _flush(self) -> None:
        self.output_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def build_classification_pipeline() -> ElectoralRollPipelineService:
    """Build the PDF-to-classified-crops pipeline."""
    ocr_service = OCRService()
    return ElectoralRollPipelineService(
        pdf_service=PDFService(),
        image_service=ImageService(),
        grid_service=GridDetectionService(),
        ocr_service=ocr_service,
        deleted_entry_service=DeletedEntryDetectionService(ocr_service=ocr_service),
    )


def build_classified_crop_extractor() -> ClassifiedCropExtractionService:
    """Build the ordered LLM extractor for classified crops."""
    settings = get_settings()
    return ClassifiedCropExtractionService(
        llm_extraction_service=LLMExtractionService(settings=settings)
    )


def run_pipeline(pdf_path: Path, output_path: Path) -> list[dict[str, Any]]:
    """Run the full pipeline from PDF to ordered JSON."""
    pipeline = build_classification_pipeline()
    extractor = build_classified_crop_extractor()
    incremental_writer = IncrementalJsonListWriter(output_path)

    classified_crops = pipeline.classify_pdf_entries(pdf_path)
    logger.info(
        "Finished crop classification",
        extra={"pdf_path": str(pdf_path), "classified_crop_count": len(classified_crops)},
    )

    records = extractor.extract_pdf_records(
        pdf_path.stem,
        on_record_extracted=lambda record: incremental_writer.append(record.model_dump()),
    )
    serialized_records: list[dict[str, Any]] = []
    for record in records:
        record_payload = record.model_dump()
        serialized_records.append(record_payload)

    logger.info(
        "Finished ordered LLM extraction",
        extra={"pdf_path": str(pdf_path), "record_count": len(serialized_records)},
    )
    return serialized_records
