"""Unit tests for electoral roll grid detection."""

from pathlib import Path

import cv2
import numpy as np

from app.services.grid_service import GridDetectionService


def _create_grid_page(rows: int = 4, columns: int = 3) -> np.ndarray:
    width = 900
    height = 1200
    margin_x = 60
    margin_top = 80
    margin_bottom = 80
    row_height = (height - margin_top - margin_bottom) // rows
    column_width = (width - (2 * margin_x)) // columns

    image = np.full((height, width, 3), 255, dtype=np.uint8)

    for row_index in range(rows + 1):
        y_coord = margin_top + (row_index * row_height)
        cv2.line(image, (margin_x, y_coord), (margin_x + (columns * column_width), y_coord), (0, 0, 0), 4)

    for column_index in range(columns + 1):
        x_coord = margin_x + (column_index * column_width)
        cv2.line(
            image,
            (x_coord, margin_top),
            (x_coord, margin_top + (rows * row_height)),
            (0, 0, 0),
            4,
        )

    for row_index in range(rows):
        for column_index in range(columns):
            text_x = margin_x + (column_index * column_width) + 25
            text_y = margin_top + (row_index * row_height) + 65
            cv2.putText(
                image,
                f"{row_index}-{column_index}",
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    return image


def test_detect_entries_finds_and_sorts_grid_cells(tmp_path: Path) -> None:
    service = GridDetectionService(debug_enabled=True, debug_output_dir=tmp_path / "debug")
    image = _create_grid_page(rows=4, columns=3)

    result = service.detect_entries(image=image, page=2, image_name="sample_page")

    assert len(result.entries) == 12
    assert result.visualization_path is not None
    assert result.visualization_path.exists()
    assert [entry.entry_index for entry in result.entries] == list(range(1, 13))

    first = result.entries[0]
    second = result.entries[1]
    fourth = result.entries[3]

    assert first.page == 2
    assert first.x < second.x
    assert first.y == second.y or abs(first.y - second.y) <= 10
    assert fourth.y > first.y


def test_detect_entries_from_path_raises_for_missing_file(tmp_path: Path) -> None:
    service = GridDetectionService(debug_enabled=False)

    try:
        service.detect_entries_from_path(tmp_path / "missing.png", page=1)
    except FileNotFoundError as exc:
        assert "does not exist" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError for missing image")


def test_detect_entries_filters_out_invalid_rectangles(tmp_path: Path) -> None:
    service = GridDetectionService(debug_enabled=False, debug_output_dir=tmp_path / "debug")
    image = _create_grid_page(rows=2, columns=3)
    cv2.rectangle(image, (10, 10), (880, 1140), (0, 0, 0), 5)

    result = service.detect_entries(image=image, page=1, image_name="with_border_noise")

    assert len(result.entries) == 6
    assert all(entry.width < 400 for entry in result.entries)


def test_deduplicate_overlapping_rectangles_keeps_one_box_per_entry() -> None:
    rectangles = [
        (112, 228, 1561, 634),
        (122, 235, 1540, 617),
        (1695, 228, 1561, 634),
        (1705, 235, 1541, 617),
        (3282, 228, 1558, 634),
        (3289, 235, 1541, 617),
    ]

    result = GridDetectionService._deduplicate_overlapping_rectangles(rectangles)

    assert len(result) == 3
    assert (112, 228, 1561, 634) in result
    assert (1695, 228, 1561, 634) in result
    assert (3282, 228, 1558, 634) in result
