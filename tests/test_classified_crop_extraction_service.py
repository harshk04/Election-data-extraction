"""Tests for ordered LLM extraction from classified crop folders."""

from pathlib import Path

from app.models.voter import VoterRecord
from app.services.classified_crop_extraction_service import ClassifiedCropExtractionService


class FakeLLMExtractionService:
    """Stub LLM service for ordered extraction tests."""

    def __init__(self, failing_names: set[str] | None = None) -> None:
        self.failing_names = failing_names or set()
        self.calls: list[tuple[str, bool | None]] = []

    def extract_voter_record(self, image_path: Path, deleted: bool | None = None) -> VoterRecord:
        self.calls.append((image_path.name, deleted))
        if image_path.name in self.failing_names:
            raise RuntimeError("synthetic llm failure")
        return VoterRecord(
            serial_number=image_path.stem,
            epic_number="ABC1234567",
            deleted=deleted,
        )


def test_collect_classified_crops_preserves_page_entry_sequence(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    deleted_dir = tmp_path / "deleted"
    failure_dir = tmp_path / "failed"
    pdf_name = "Updated2"

    for image_path in (
        normal_dir / pdf_name / "page_001_entry_001.png",
        normal_dir / pdf_name / "page_001_entry_005.png",
        deleted_dir / pdf_name / "page_001_entry_006.png",
        normal_dir / pdf_name / "page_002_entry_001.png",
    ):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"test")

    service = ClassifiedCropExtractionService(
        llm_extraction_service=FakeLLMExtractionService(),
        normal_entries_dir=normal_dir,
        deleted_entries_dir=deleted_dir,
        failure_logs_dir=failure_dir,
    )

    crops = service.collect_classified_crops(pdf_name)

    assert [(crop.page, crop.entry_index, crop.deleted) for crop in crops] == [
        (1, 1, False),
        (1, 5, False),
        (1, 6, True),
        (2, 1, False),
    ]


def test_extract_pdf_records_logs_failures_and_continues_in_order(tmp_path: Path) -> None:
    normal_dir = tmp_path / "normal"
    deleted_dir = tmp_path / "deleted"
    failure_dir = tmp_path / "failed"
    pdf_name = "Updated2"

    paths = [
        normal_dir / pdf_name / "page_001_entry_001.png",
        normal_dir / pdf_name / "page_001_entry_005.png",
        deleted_dir / pdf_name / "page_001_entry_006.png",
    ]
    for image_path in paths:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"test")

    fake_llm = FakeLLMExtractionService(failing_names={"page_001_entry_005.png"})
    service = ClassifiedCropExtractionService(
        llm_extraction_service=fake_llm,
        normal_entries_dir=normal_dir,
        deleted_entries_dir=deleted_dir,
        failure_logs_dir=failure_dir,
    )

    records = service.extract_pdf_records(pdf_name)

    assert [record.serial_number for record in records] == [
        "page_001_entry_001",
        "page_001_entry_006",
    ]
    assert [record.deleted for record in records] == [False, True]
    assert fake_llm.calls == [
        ("page_001_entry_001.png", False),
        ("page_001_entry_005.png", False),
        ("page_001_entry_006.png", True),
    ]

    failure_log = failure_dir / f"{pdf_name}.txt"
    failure_text = failure_log.read_text(encoding="utf-8")
    assert "page_001_entry_005.png" in failure_text
    assert "reason=synthetic llm failure" in failure_text
