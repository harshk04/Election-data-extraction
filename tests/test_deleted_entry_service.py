"""Unit tests for deleted-entry detection."""

from pathlib import Path

import cv2
import numpy as np

from app.services.deleted_entry_service import DeletedEntryDetectionService
from app.services.ocr_service import OCRService


class FakeOCREngine:
    """Test double for OCR responses."""

    def __init__(self, responses: list[list[list[object]]]) -> None:
        self.responses = responses
        self.call_count = 0

    def ocr(self, image: np.ndarray, cls: bool = True) -> list[list[object]]:
        del image, cls
        index = min(self.call_count, len(self.responses) - 1)
        self.call_count += 1
        return self.responses[index]


def _make_card(include_watermark: bool = False, include_q: bool = False) -> np.ndarray:
    image = np.full((260, 420, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (8, 8), (190, 52), (40, 40, 40), 2)
    cv2.rectangle(image, (318, 58), (402, 238), (40, 40, 40), 2)
    cv2.putText(image, "33" if include_q else "31", (146, 41), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)
    if include_q:
        cv2.putText(image, "Q", (18, 41), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (40, 40, 40), 2)
    cv2.putText(image, "NAME : SAMPLE", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    cv2.putText(image, "FATHER : PERSON", (18, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    cv2.putText(image, "HOUSE : 8", (18, 164), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    cv2.putText(image, "AGE : 45", (18, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)

    if include_watermark:
        overlay = image.copy()
        cv2.putText(
            overlay,
            "DELETED",
            (60, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.9,
            (150, 150, 150),
            4,
            cv2.LINE_AA,
        )
        center = (overlay.shape[1] // 2, overlay.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, -33.0, 1.0)
        rotated = cv2.warpAffine(
            overlay,
            matrix,
            (overlay.shape[1], overlay.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
        image = cv2.addWeighted(rotated, 0.45, image, 0.55, 0.0)

    return image


def test_detect_deleted_entry_from_q_and_geometry(tmp_path: Path) -> None:
    engine = FakeOCREngine(
        responses=[
            [[
                [[[5, 5], [25, 5], [25, 25], [5, 25]], ("Q", 0.98)],
                [[[35, 5], [85, 5], [85, 25], [35, 25]], ("33", 0.99)],
            ]],
            [],
        ]
    )
    service = DeletedEntryDetectionService(
        ocr_service=OCRService(engine=engine),
        debug_output_dir=tmp_path / "debug",
    )

    result = service.detect_deleted_entry(_make_card(include_watermark=True, include_q=True), image_name="q_case")

    assert result.deleted is True
    assert result.confidence >= 0.9
    assert result.watermark_confidence > 0.0
    assert result.debug_image_path is not None


def test_detect_deleted_entry_from_watermark_ocr_and_geometry(tmp_path: Path) -> None:
    engine = FakeOCREngine(
        responses=[
            [[[[[5, 5], [80, 5], [80, 25], [5, 25]], ("31", 0.99)]]],
            [[[[[5, 5], [140, 5], [140, 25], [5, 25]], ("DELETFD", 0.88)]]],
        ]
    )
    service = DeletedEntryDetectionService(
        ocr_service=OCRService(engine=engine),
        debug_output_dir=tmp_path / "debug",
    )

    result = service.detect_deleted_entry(_make_card(include_watermark=True, include_q=False), image_name="ocr_case")

    assert result.deleted is True
    assert result.ocr_confidence > 0.0
    assert result.watermark_confidence > 0.0


def test_detect_deleted_entry_returns_false_for_q_without_watermark(tmp_path: Path) -> None:
    engine = FakeOCREngine(
        responses=[
            [[
                [[[5, 5], [25, 5], [25, 25], [5, 25]], ("Q", 0.96)],
                [[[35, 5], [85, 5], [85, 25], [35, 25]], ("59", 0.98)],
            ]],
        ]
    )
    service = DeletedEntryDetectionService(
        ocr_service=OCRService(engine=engine),
        debug_output_dir=tmp_path / "debug",
    )

    result = service.detect_deleted_entry(_make_card(include_watermark=False, include_q=True), image_name="q_only")

    assert result.deleted is False
    assert result.confidence <= 0.3


def test_debug_artifacts_are_saved_for_every_crop(tmp_path: Path) -> None:
    engine = FakeOCREngine(
        responses=[
            [[
                [[[5, 5], [25, 5], [25, 25], [5, 25]], ("Q", 0.98)],
                [[[35, 5], [85, 5], [85, 25], [35, 25]], ("33", 0.99)],
            ]],
            [[[[[5, 5], [140, 5], [140, 25], [5, 25]], ("DELETED", 0.93)]]],
        ]
    )
    debug_dir = tmp_path / "debug"
    service = DeletedEntryDetectionService(
        ocr_service=OCRService(engine=engine),
        debug_output_dir=debug_dir,
    )

    service.detect_deleted_entry(_make_card(include_watermark=True, include_q=True), image_name="artifact_case")

    artifact_dir = debug_dir / "artifact_case"
    expected_files = {
        "original.png",
        "serial_region.png",
        "serial_ocr.txt",
        "watermark_mask.png",
        "largest_component.png",
        "rotated_band.png",
        "watermark_ocr.txt",
        "decision.json",
        "deleted_detection.png",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})


def test_detect_deleted_entry_from_path_raises_for_missing_file(tmp_path: Path) -> None:
    engine = FakeOCREngine(responses=[[]])
    service = DeletedEntryDetectionService(
        ocr_service=OCRService(engine=engine),
        debug_output_dir=tmp_path / "debug",
    )

    try:
        service.detect_deleted_entry_from_path(tmp_path / "missing.png")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing image")
