"""Unit tests for the image preprocessing pipeline."""

from pathlib import Path

import cv2
import numpy as np

from app.models.image import EntryBoundingBox, PreprocessingConfig
from app.services.image_service import ImageService


def _create_test_image() -> np.ndarray:
    image = np.full((120, 160, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (0, 0), (159, 119), (0, 0, 0), 4)
    cv2.putText(image, "VOTER", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    return image


def test_preprocess_image_file_runs_pipeline_and_saves_debug_images(tmp_path: Path) -> None:
    input_path = tmp_path / "page_0001.png"
    output_path = tmp_path / "clean" / "page_0001_clean.png"
    debug_dir = tmp_path / "debug"
    cv2.imwrite(str(input_path), _create_test_image())

    service = ImageService(debug_enabled=True, debug_output_dir=debug_dir)

    result = service.preprocess_image_file(input_path, output_path)

    assert result.image_path == output_path
    assert result.image_path.exists()
    assert result.width == 160
    assert result.height == 120
    assert len(result.debug_image_paths) == 6
    assert all(path.exists() for path in result.debug_image_paths)


def test_preprocess_page_respects_optional_configuration() -> None:
    image = _create_test_image()
    config = PreprocessingConfig(
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        enable_threshold=False,
        enable_border_cleanup=False,
    )
    service = ImageService(preprocessing_config=config, debug_enabled=False)

    processed, debug_images = service.preprocess_page(image)

    assert processed.ndim == 2
    assert processed.shape == image.shape[:2]
    assert list(debug_images.keys()) == ["original"]


def test_load_image_raises_for_missing_file(tmp_path: Path) -> None:
    service = ImageService(debug_enabled=False)

    try:
        service.load_image(tmp_path / "missing.png")
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing image")


def test_border_cleanup_whitens_outer_margin() -> None:
    grayscale = np.full((40, 40), 255, dtype=np.uint8)
    grayscale[:, 0] = 0
    grayscale[:, -1] = 0
    grayscale[0, :] = 0
    grayscale[-1, :] = 0

    config = PreprocessingConfig(
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        enable_threshold=False,
        enable_border_cleanup=True,
        border_margin=2,
    )
    service = ImageService(preprocessing_config=config, debug_enabled=False)

    processed, _ = service.preprocess_page(grayscale)

    assert np.all(processed[:2, :] == 255)
    assert np.all(processed[-2:, :] == 255)
    assert np.all(processed[:, :2] == 255)
    assert np.all(processed[:, -2:] == 255)


def test_extract_record_crops_saves_ordered_entry_images(tmp_path: Path) -> None:
    image = np.full((240, 360, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (140, 100), (0, 0, 0), -1)
    cv2.rectangle(image, (180, 20), (300, 100), (60, 60, 60), -1)
    cv2.rectangle(image, (20, 130), (140, 210), (120, 120, 120), -1)

    service = ImageService(debug_enabled=False)
    bounding_boxes = [
        EntryBoundingBox(page=1, entry_index=1, x=20, y=20, width=120, height=80),
        EntryBoundingBox(page=1, entry_index=2, x=180, y=20, width=120, height=80),
        EntryBoundingBox(page=1, entry_index=3, x=20, y=130, width=120, height=80),
    ]

    result = service.extract_record_crops(image=image, bounding_boxes=bounding_boxes, output_dir=tmp_path)

    assert [item.entry_index for item in result] == [1, 2, 3]
    assert [item.image_path.name for item in result] == [
        "page_001_entry_001.png",
        "page_001_entry_002.png",
        "page_001_entry_003.png",
    ]
    assert all(item.image_path.exists() for item in result)
    assert result[0].width == 120
    assert result[0].height == 80


def test_extract_record_crops_ignores_tiny_rectangles(tmp_path: Path) -> None:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    service = ImageService(debug_enabled=False)
    bounding_boxes = [
        EntryBoundingBox(page=1, entry_index=1, x=10, y=10, width=120, height=80),
        EntryBoundingBox(page=1, entry_index=2, x=20, y=20, width=20, height=20),
    ]

    result = service.extract_record_crops(image=image, bounding_boxes=bounding_boxes, output_dir=tmp_path)

    assert len(result) == 1
    assert result[0].entry_index == 1
    assert result[0].image_path.name == "page_001_entry_001.png"


def test_extract_record_crops_clamps_boxes_to_image_bounds(tmp_path: Path) -> None:
    image = np.full((150, 150, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 100), (149, 149), (0, 0, 0), -1)
    service = ImageService(debug_enabled=False)
    bounding_boxes = [
        EntryBoundingBox(page=2, entry_index=1, x=100, y=100, width=90, height=90),
    ]

    result = service.extract_record_crops(image=image, bounding_boxes=bounding_boxes, output_dir=tmp_path)

    assert len(result) == 1
    assert result[0].page == 2
    assert result[0].width == 50
    assert result[0].height == 50
    assert result[0].image_path.name == "page_002_entry_001.png"
