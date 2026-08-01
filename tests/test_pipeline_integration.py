"""Integration tests for resilient pipeline orchestration."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np

import app.services.pipeline_service as pipeline_module
from app.models.image import CropMetadata, DeletedEntryDetectionResult, EntryBoundingBox, OCRResult
from app.models.pdf import PageRenderMetadata
from app.models.voter import ValidationReport, VoterRecord
from app.services.pipeline_service import ElectoralRollPipelineService, EntryProcessingTimeoutError


class FakePDFService:
    """Stub PDF service for pipeline integration tests."""

    def __init__(self, pages_root_dir: Path, page_metadata: list[PageRenderMetadata]) -> None:
        self.pages_root_dir = pages_root_dir
        self.page_metadata = page_metadata

    def extract_pages(self, pdf_path: Path) -> list[PageRenderMetadata]:
        del pdf_path
        return self.page_metadata


class FakeImageService:
    """Stub image service for pipeline integration tests."""

    def preprocess_page(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        return image, {}

    def extract_record_crops(
        self,
        image: np.ndarray,
        bounding_boxes: list[EntryBoundingBox],
        output_dir: Path,
    ) -> list[CropMetadata]:
        del image, output_dir
        return [
            CropMetadata(
                page=box.page,
                entry_index=box.entry_index,
                x=box.x,
                y=box.y,
                width=box.width,
                height=box.height,
                image_path=Path(f"/tmp/page_{box.page:03d}_entry_{box.entry_index:03d}.png"),
            )
            for box in bounding_boxes
        ]


class FakeGridService:
    """Stub grid detector for pipeline integration tests."""

    def detect_entries(self, image: np.ndarray, page: int, image_name: str = "page"):  # type: ignore[no-untyped-def]
        del image, image_name
        if page == 1:
            return type(
                "GridResult",
                (),
                {
                    "entries": [
                        EntryBoundingBox(page=1, entry_index=1, x=0, y=0, width=100, height=50),
                        EntryBoundingBox(page=1, entry_index=2, x=110, y=0, width=100, height=50),
                    ]
                },
            )()
        raise RuntimeError("synthetic page failure")


class FakeOCRService:
    """Stub OCR service for pipeline integration tests."""

    def run_ocr(self, image_path: Path) -> OCRResult:
        if image_path.name.endswith("002.png"):
            raise RuntimeError("synthetic entry OCR failure")
        return OCRResult(lines=[])


class FakeExtractionService:
    """Stub extraction service for pipeline integration tests."""

    def parse_voter_record(self, ocr_payload: OCRResult) -> VoterRecord:
        del ocr_payload
        return VoterRecord(serial_number="1", epic_number="ABC1234567")


class FakeDeletedEntryService:
    """Stub deleted detector for pipeline integration tests."""

    def detect_deleted_entry_from_path(self, image_path: Path) -> DeletedEntryDetectionResult:
        del image_path
        return DeletedEntryDetectionResult(deleted=False, confidence=0.9)


class FakeValidationService:
    """Stub validation service for pipeline integration tests."""

    def validate_record(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        deleted_result: DeletedEntryDetectionResult | None = None,
    ) -> ValidationReport:
        del record, ocr_result, deleted_result
        return ValidationReport(is_valid=True)


def test_process_pdf_continues_when_entry_and_page_fail(tmp_path: Path) -> None:
    page_metadata = [
        PageRenderMetadata(page_number=1, width=100, height=50, image_path=tmp_path / "page_0001.png"),
        PageRenderMetadata(page_number=2, width=100, height=50, image_path=tmp_path / "page_0002.png"),
    ]
    pipeline = ElectoralRollPipelineService(
        pdf_service=FakePDFService(tmp_path, page_metadata),
        image_service=FakeImageService(),
        grid_service=FakeGridService(),
        ocr_service=FakeOCRService(),
        extraction_service=FakeExtractionService(),
        deleted_entry_service=FakeDeletedEntryService(),
        validation_service=FakeValidationService(),
        crops_root_dir=tmp_path / "crops",
    )

    original_load_image = pipeline_module.load_image
    pipeline_module.load_image = lambda image_path: np.zeros((20, 20, 3), dtype=np.uint8)
    try:
        records = pipeline.process_pdf(tmp_path / "sample.pdf")
    finally:
        pipeline_module.load_image = original_load_image

    assert len(records) == 1
    assert records[0].epic_number == "ABC1234567"
    assert records[0].deleted is False


def test_process_crops_records_timed_out_entries_and_continues(tmp_path: Path) -> None:
    pipeline = ElectoralRollPipelineService(
        pdf_service=FakePDFService(tmp_path, []),
        image_service=FakeImageService(),
        grid_service=FakeGridService(),
        ocr_service=FakeOCRService(),
        extraction_service=FakeExtractionService(),
        deleted_entry_service=FakeDeletedEntryService(),
        validation_service=FakeValidationService(),
        crops_root_dir=tmp_path / "crops",
    )
    pipeline.timed_out_entries_dir = tmp_path / "timed_out_entries"
    pipeline._reset_timeout_log("sample")
    pipeline._archive_classified_crop = lambda pdf_name, crop_path, is_deleted: None

    crop_metadata = [
        CropMetadata(
            page=1,
            entry_index=1,
            x=0,
            y=0,
            width=100,
            height=50,
            image_path=tmp_path / "page_001_entry_001.png",
        ),
        CropMetadata(
            page=1,
            entry_index=2,
            x=110,
            y=0,
            width=100,
            height=50,
            image_path=tmp_path / "page_001_entry_002.png",
        ),
    ]

    timeout_names = {"page_001_entry_002.png"}

    @contextmanager
    def fake_entry_timeout(image_path: Path):
        if image_path.name in timeout_names:
            raise EntryProcessingTimeoutError(f"Timed out while processing {image_path}")
        yield

    pipeline._entry_timeout = fake_entry_timeout  # type: ignore[method-assign]

    records = pipeline._process_crops(tmp_path / "sample.pdf", crop_metadata)

    assert len(records) == 1
    assert records[0].serial_number == "1"
    timeout_log = pipeline.timed_out_entries_dir / "sample.txt"
    assert timeout_log.read_text(encoding="utf-8").splitlines() == ["page_001_entry_002.png"]
