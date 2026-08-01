"""Unit tests for the OCR service."""

from pathlib import Path

import cv2
import numpy as np

from app.services.ocr_service import OCRService


class FakeOCREngine:
    """Test double for PaddleOCR responses."""

    def __init__(self, response: list[list[object]]) -> None:
        self.response = response
        self.call_count = 0

    def ocr(self, image: np.ndarray, cls: bool = True) -> list[list[object]]:
        del image, cls
        self.call_count += 1
        return self.response


def test_run_ocr_on_image_returns_structured_result() -> None:
    engine = FakeOCREngine(
        response=[
            [
                [[10, 20], [110, 20], [110, 45], [10, 45]],
                ("SERIAL 123", 0.91),
            ],
            [
                [[15, 60], [130, 60], [130, 90], [15, 90]],
                ("NAME BLOCK", 0.87),
            ],
        ]
    )
    service = OCRService(engine=engine, language="en")
    image = np.full((120, 200, 3), 255, dtype=np.uint8)

    result = service.run_ocr_on_image(image)

    assert result.image_path is None
    assert len(result.lines) == 2
    assert result.lines[0].text == "SERIAL 123"
    assert result.lines[0].confidence == 0.91
    assert result.lines[0].language == "en"
    assert len(result.lines[0].bounding_box) == 4
    assert result.lines[0].bounding_box[0].x == 10.0
    assert result.lines[0].bounding_box[0].y == 20.0


def test_run_ocr_from_path_returns_image_path(tmp_path: Path) -> None:
    engine = FakeOCREngine(
        response=[
            [
                [[5, 5], [50, 5], [50, 20], [5, 20]],
                ("ENTRY TEXT", 0.8),
            ]
        ]
    )
    service = OCRService(engine=engine, language="en")
    image_path = tmp_path / "entry.png"
    image = np.full((50, 80, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    result = service.run_ocr(image_path)

    assert result.image_path == image_path
    assert len(result.lines) == 1
    assert result.lines[0].text == "ENTRY TEXT"


def test_run_ocr_raises_for_missing_file(tmp_path: Path) -> None:
    service = OCRService(engine=FakeOCREngine(response=[]))

    try:
        service.run_ocr(tmp_path / "missing.png")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing image")


def test_run_ocr_handles_empty_engine_result() -> None:
    service = OCRService(engine=FakeOCREngine(response=[]))
    image = np.full((20, 20, 3), 255, dtype=np.uint8)

    result = service.run_ocr_on_image(image)

    assert result.lines == []
