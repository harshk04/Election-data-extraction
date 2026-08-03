"""Ordered LLM extraction for crops already classified as normal or deleted."""

from __future__ import annotations

import re
from pathlib import Path

from app.config.settings import get_settings
from app.models.image import ClassifiedEntryImage
from app.models.voter import VoterRecord
from app.services.llm_extraction_service import LLMExtractionService
from app.utils.file_utils import ensure_directory
from app.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


class ClassifiedCropExtractionService:
    """Extract ordered voter records from classified crop folders."""

    _ENTRY_PATTERN = re.compile(r"page_(\d+)_entry_(\d+)\.png$", re.IGNORECASE)

    def __init__(
        self,
        llm_extraction_service: LLMExtractionService | None = None,
        normal_entries_dir: Path | None = None,
        deleted_entries_dir: Path | None = None,
        failure_logs_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.normal_entries_dir = normal_entries_dir or settings.normal_entries_dir
        self.deleted_entries_dir = deleted_entries_dir or settings.deleted_entries_dir
        self.failure_logs_dir = failure_logs_dir or settings.failed_cases_dir
        self.llm_extraction_service = llm_extraction_service or LLMExtractionService(settings)

    def extract_pdf_records(self, pdf_name: str) -> list[VoterRecord]:
        """Extract ordered records from the classified crop directories for one PDF."""
        failure_log_path = self._failure_log_path(pdf_name)
        ensure_directory(failure_log_path.parent)
        failure_log_path.write_text("", encoding="utf-8")

        records: list[VoterRecord] = []
        for classified_crop in self.collect_classified_crops(pdf_name):
            try:
                record = self.llm_extraction_service.extract_voter_record(
                    image_path=classified_crop.image_path,
                    deleted=classified_crop.deleted,
                )
                records.append(record)
            except Exception as error:
                self._append_failure(
                    failure_log_path=failure_log_path,
                    classified_crop=classified_crop,
                    reason=str(error),
                )
                log_exception(
                    logger,
                    "LLM extraction failed for classified crop",
                    image_path=classified_crop.image_path,
                    deleted=classified_crop.deleted,
                    page=classified_crop.page,
                    entry_index=classified_crop.entry_index,
                )

        return records

    def collect_classified_crops(self, pdf_name: str) -> list[ClassifiedEntryImage]:
        """Collect and sort all classified crops across normal and deleted folders."""
        collected: list[ClassifiedEntryImage] = []
        for deleted, root_dir in (
            (False, self.normal_entries_dir / pdf_name),
            (True, self.deleted_entries_dir / pdf_name),
        ):
            if not root_dir.exists():
                continue

            for image_path in sorted(root_dir.glob("*.png")):
                page, entry_index = self._parse_page_and_entry(image_path)
                collected.append(
                    ClassifiedEntryImage(
                        page=page,
                        entry_index=entry_index,
                        image_path=image_path,
                        deleted=deleted,
                    )
                )

        return sorted(
            collected,
            key=lambda item: (item.page, item.entry_index, item.image_path.name.lower()),
        )

    def _failure_log_path(self, pdf_name: str) -> Path:
        return self.failure_logs_dir / f"{pdf_name}.txt"

    def _append_failure(
        self,
        failure_log_path: Path,
        classified_crop: ClassifiedEntryImage,
        reason: str,
    ) -> None:
        with failure_log_path.open("a", encoding="utf-8") as file_handle:
            file_handle.write(
                f"{classified_crop.image_path.name}\tdeleted={classified_crop.deleted}\t"
                f"page={classified_crop.page}\tentry={classified_crop.entry_index}\t"
                f"reason={reason}\n"
            )

    def _parse_page_and_entry(self, image_path: Path) -> tuple[int, int]:
        match = self._ENTRY_PATTERN.search(image_path.name)
        if match is None:
            raise ValueError(
                "Classified crop filename does not match expected pattern "
                f"'page_###_entry_###.png': {image_path.name}"
            )
        return int(match.group(1)), int(match.group(2))
