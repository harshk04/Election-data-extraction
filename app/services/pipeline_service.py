"""Pipeline orchestration service for electoral roll OCR processing."""

import signal
import shutil
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config.settings import get_settings
from app.models.image import CropMetadata
from app.models.voter import VoterRecord
from app.services.deleted_entry_service import DeletedEntryDetectionService
from app.services.extraction_service import ExtractionService
from app.services.grid_service import GridDetectionService
from app.services.image_service import ImageService
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.validation_service import ValidationService
from app.utils.file_utils import ensure_directory, load_image
from app.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


class EntryProcessingTimeoutError(TimeoutError):
    """Raised when a single crop exceeds the configured processing limit."""


class ElectoralRollPipelineService:
    """Coordinate PDF ingestion, OCR, and structured export steps."""

    def __init__(
        self,
        pdf_service: PDFService,
        image_service: ImageService,
        grid_service: GridDetectionService,
        ocr_service: OCRService,
        extraction_service: ExtractionService,
        deleted_entry_service: DeletedEntryDetectionService | None = None,
        validation_service: ValidationService | None = None,
        crops_root_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.pdf_service = pdf_service
        self.image_service = image_service
        self.grid_service = grid_service
        self.ocr_service = ocr_service
        self.extraction_service = extraction_service
        self.deleted_entry_service = deleted_entry_service or DeletedEntryDetectionService(
            ocr_service=ocr_service,
        )
        self.validation_service = validation_service or ValidationService()
        self.crops_root_dir = crops_root_dir
        self.deleted_entries_dir = settings.deleted_entries_dir
        self.normal_entries_dir = settings.normal_entries_dir
        self.entry_timeout_seconds = settings.entry_timeout_seconds
        self.timed_out_entries_dir = settings.timed_out_entries_dir

    def process_pdf(
        self,
        pdf_path: Path,
        on_record_processed: Callable[[VoterRecord], None] | None = None,
    ) -> list[VoterRecord]:
        """Process a scanned electoral roll PDF into structured records."""
        logger.info("Processing PDF through pipeline", extra={"pdf_path": str(pdf_path)})
        records: list[VoterRecord] = []

        try:
            page_metadata = self.pdf_service.extract_pages(pdf_path)
        except Exception:
            log_exception(logger, "Failed to render PDF pages", pdf_path=pdf_path)
            return records

        try:
            self.ocr_service.initialize_engine()
        except RuntimeError:
            log_exception(logger, "OCR engine initialization failed", pdf_path=pdf_path)
            return records

        pdf_output_name = pdf_path.stem
        self._reset_classified_output_dirs(pdf_output_name)
        self._reset_timeout_log(pdf_output_name)
        crops_root_dir = ensure_directory(
            self.crops_root_dir or self.pdf_service.pages_root_dir.parent / "crops" / pdf_output_name
        )

        for page_item in page_metadata:
            try:
                page_image = load_image(page_item.image_path)
                preprocessed_page, _ = self.image_service.preprocess_page(page_image)
                grid_result = self.grid_service.detect_entries(
                    image=preprocessed_page,
                    page=page_item.page_number,
                    image_name=page_item.image_path.stem,
                )
                crop_metadata = self.image_service.extract_record_crops(
                    image=page_image,
                    bounding_boxes=grid_result.entries,
                    output_dir=crops_root_dir,
                )
            except Exception:
                log_exception(
                    logger,
                    "Failed to process page",
                    pdf_path=pdf_path,
                    page_number=page_item.page_number,
                )
                continue

            records.extend(self._process_crops(pdf_path, crop_metadata, on_record_processed))

        return records

    def _process_crops(
        self,
        pdf_path: Path,
        crop_metadata: list[CropMetadata],
        on_record_processed: Callable[[VoterRecord], None] | None = None,
    ) -> list[VoterRecord]:
        """Process cropped voter entries while isolating failures per entry."""
        records: list[VoterRecord] = []
        pdf_output_name = pdf_path.stem
        for crop_item in crop_metadata:
            try:
                with self._entry_timeout(crop_item.image_path):
                    ocr_result = self.ocr_service.run_ocr(crop_item.image_path)
                    deleted_result = self.deleted_entry_service.detect_deleted_entry_from_path(crop_item.image_path)
                    self._archive_classified_crop(
                        pdf_name=pdf_output_name,
                        crop_path=crop_item.image_path,
                        is_deleted=deleted_result.deleted,
                    )
                    record = self.extraction_service.parse_voter_record(
                        ocr_result,
                        image_path=crop_item.image_path,
                        deleted=deleted_result.deleted,
                    )
                    self.validation_service.validate_record(record, ocr_result, deleted_result)
                    records.append(record)
                    if on_record_processed is not None:
                        on_record_processed(record)
            except EntryProcessingTimeoutError:
                self._record_timed_out_entry(pdf_output_name, crop_item.image_path)
                logger.warning(
                    "Entry processing timed out and was skipped",
                    extra={
                        "image_path": str(crop_item.image_path),
                        "page": crop_item.page,
                        "entry_index": crop_item.entry_index,
                        "timeout_seconds": self.entry_timeout_seconds,
                    },
                )
            except Exception:
                log_exception(
                    logger,
                    "Failed to process entry crop",
                    image_path=crop_item.image_path,
                    page=crop_item.page,
                    entry_index=crop_item.entry_index,
                )
                continue

        return records

    def _archive_classified_crop(self, pdf_name: str, crop_path: Path, is_deleted: bool) -> None:
        """Copy the processed crop into the deleted or normal archive folder."""
        root_dir = self.deleted_entries_dir if is_deleted else self.normal_entries_dir
        destination_dir = ensure_directory(root_dir / pdf_name)
        shutil.copy2(crop_path, destination_dir / crop_path.name)

    def _reset_classified_output_dirs(self, pdf_name: str) -> None:
        """Clear previous deleted/normal classifications for the current PDF run."""
        for root_dir in (self.deleted_entries_dir, self.normal_entries_dir):
            target_dir = root_dir / pdf_name
            if target_dir.exists():
                shutil.rmtree(target_dir)

    def _reset_timeout_log(self, pdf_name: str) -> None:
        """Clear previous timeout records for the current PDF run."""
        timeout_log = self._timeout_log_path(pdf_name)
        ensure_directory(timeout_log.parent)
        timeout_log.write_text("", encoding="utf-8")

    def _record_timed_out_entry(self, pdf_name: str, image_path: Path) -> None:
        """Append a timed-out crop name so the run can be reviewed later."""
        timeout_log = self._timeout_log_path(pdf_name)
        ensure_directory(timeout_log.parent)
        with timeout_log.open("a", encoding="utf-8") as file_handle:
            file_handle.write(f"{image_path.name}\n")

    def _timeout_log_path(self, pdf_name: str) -> Path:
        """Return the timeout log file path for a PDF run."""
        return self.timed_out_entries_dir / f"{pdf_name}.txt"

    @contextmanager
    def _entry_timeout(self, image_path: Path) -> Iterator[None]:
        """Abort a crop if processing exceeds the configured time budget."""
        if self.entry_timeout_seconds <= 0 or not hasattr(signal, "setitimer"):
            yield
            return

        previous_handler = signal.getsignal(signal.SIGALRM)

        def _handle_timeout(signum: int, frame: object | None) -> None:
            del signum, frame
            raise EntryProcessingTimeoutError(f"Timed out while processing {image_path}")

        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, float(self.entry_timeout_seconds))
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
