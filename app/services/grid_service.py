"""Service implementation for automatic electoral-roll grid detection."""

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import get_settings
from app.models.image import EntryBoundingBox, GridDetectionConfig, GridDetectionResult
from app.utils.file_utils import ensure_directory, load_image, save_image
from app.utils.logging import get_logger

logger = get_logger(__name__)


class GridDetectionService:
    """Detect voter entry rectangles from an electoral roll page image."""

    def __init__(
        self,
        grid_config: GridDetectionConfig | None = None,
        debug_enabled: bool | None = None,
        debug_output_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.grid_config = grid_config or GridDetectionConfig(
            min_width_ratio=settings.grid_min_width_ratio,
            min_height_ratio=settings.grid_min_height_ratio,
            max_width_ratio=settings.grid_max_width_ratio,
            max_height_ratio=settings.grid_max_height_ratio,
            horizontal_kernel_ratio=settings.grid_horizontal_kernel_ratio,
            vertical_kernel_ratio=settings.grid_vertical_kernel_ratio,
        )
        self.debug_enabled = settings.app_debug if debug_enabled is None else debug_enabled
        self.debug_output_dir = debug_output_dir or settings.grid_debug_dir

    def detect_entries(
        self,
        image: np.ndarray,
        page: int,
        image_name: str = "page",
    ) -> GridDetectionResult:
        """Detect entry rectangles by extracting horizontal and vertical grid lines."""
        logger.info("Detecting grid entries", extra={"page": page, "image_name": image_name})

        grayscale = self._to_grayscale(image)
        binary = self._to_binary(grayscale)
        horizontal_lines = self._detect_horizontal_lines(binary)
        vertical_lines = self._detect_vertical_lines(binary)
        grid_mask = self._build_grid_mask(horizontal_lines, vertical_lines)
        rectangles = self._find_entry_rectangles(grid_mask, page=page)
        visualization_path = self._save_visualization(
            source_image=image,
            entries=rectangles,
            image_name=image_name,
            page=page,
        )

        return GridDetectionResult(entries=rectangles, visualization_path=visualization_path)

    def detect_entries_from_path(self, image_path: Path, page: int) -> GridDetectionResult:
        """Load an image from disk and run entry detection."""
        image = load_image(image_path)
        return self.detect_entries(image=image, page=page, image_name=image_path.stem)

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale when required."""
        if image.ndim == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _to_binary(image: np.ndarray) -> np.ndarray:
        """Create an inverted binary image suitable for line extraction."""
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            15,
        )

    @staticmethod
    def _build_grid_mask(horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> np.ndarray:
        """Combine line masks and close small gaps in detected rectangles."""
        grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(grid_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    def _detect_horizontal_lines(self, binary_image: np.ndarray) -> np.ndarray:
        """Extract horizontal grid lines from the binary page image."""
        kernel_width = max(10, binary_image.shape[1] // max(1, self.grid_config.horizontal_kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 1))
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(opened, dilate_kernel, iterations=1)

    def _detect_vertical_lines(self, binary_image: np.ndarray) -> np.ndarray:
        """Extract vertical grid lines from the binary page image."""
        kernel_height = max(10, binary_image.shape[0] // max(1, self.grid_config.vertical_kernel_ratio))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.dilate(opened, dilate_kernel, iterations=1)

    def _find_entry_rectangles(self, grid_mask: np.ndarray, page: int) -> list[EntryBoundingBox]:
        """Find, filter, and sort entry rectangles from the combined grid mask."""
        contours, _ = cv2.findContours(grid_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        image_height, image_width = grid_mask.shape[:2]
        min_width = int(image_width * self.grid_config.min_width_ratio)
        max_width = int(image_width * self.grid_config.max_width_ratio)
        min_height = int(image_height * self.grid_config.min_height_ratio)
        max_height = int(image_height * self.grid_config.max_height_ratio)

        candidates: list[tuple[int, int, int, int]] = []
        for contour in contours:
            x_coord, y_coord, width, height = cv2.boundingRect(contour)
            if not self._is_valid_rectangle(
                x=x_coord,
                y=y_coord,
                width=width,
                height=height,
                image_width=image_width,
                image_height=image_height,
                min_width=min_width,
                max_width=max_width,
                min_height=min_height,
                max_height=max_height,
            ):
                continue

            if self._contains_similar_rectangle(candidates, x_coord, y_coord, width, height):
                continue

            candidates.append((x_coord, y_coord, width, height))

        deduplicated_candidates = self._deduplicate_overlapping_rectangles(candidates)
        filtered_candidates = self._filter_outliers(deduplicated_candidates)
        sorted_candidates = self._sort_rectangles(filtered_candidates)
        return [
            EntryBoundingBox(
                page=page,
                entry_index=index,
                x=x_coord,
                y=y_coord,
                width=width,
                height=height,
            )
            for index, (x_coord, y_coord, width, height) in enumerate(sorted_candidates, start=1)
        ]

    @staticmethod
    def _is_valid_rectangle(
        x: int,
        y: int,
        width: int,
        height: int,
        image_width: int,
        image_height: int,
        min_width: int,
        max_width: int,
        min_height: int,
        max_height: int,
    ) -> bool:
        """Filter out contours that do not look like entry cells."""
        if width < min_width or width > max_width:
            return False
        if height < min_height or height > max_height:
            return False
        if x <= 0 or y <= 0:
            return False
        if x + width >= image_width or y + height >= image_height:
            return False
        aspect_ratio = width / max(1, height)
        return 1.2 <= aspect_ratio <= 4.5

    @staticmethod
    def _contains_similar_rectangle(
        rectangles: list[tuple[int, int, int, int]],
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> bool:
        """Prevent near-duplicate rectangles produced by contour hierarchy."""
        for existing_x, existing_y, existing_width, existing_height in rectangles:
            if (
                abs(existing_x - x) <= 4
                and abs(existing_y - y) <= 4
                and abs(existing_width - width) <= 6
                and abs(existing_height - height) <= 6
            ):
                return True
        return False

    @staticmethod
    def _deduplicate_overlapping_rectangles(
        rectangles: list[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        """Collapse near-identical rectangles that overlap the same entry cell."""
        deduplicated: list[tuple[int, int, int, int]] = []
        for rectangle in sorted(rectangles, key=lambda rect: (rect[1], rect[0], -(rect[2] * rect[3]))):
            matched_index: int | None = None
            for index, existing in enumerate(deduplicated):
                if GridDetectionService._intersection_over_union(existing, rectangle) >= 0.9:
                    matched_index = index
                    break

            if matched_index is None:
                deduplicated.append(rectangle)
                continue

            deduplicated[matched_index] = GridDetectionService._prefer_larger_rectangle(
                deduplicated[matched_index],
                rectangle,
            )

        return deduplicated

    @staticmethod
    def _intersection_over_union(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        """Measure rectangle overlap to identify duplicate entry detections."""
        first_x, first_y, first_width, first_height = first
        second_x, second_y, second_width, second_height = second

        left = max(first_x, second_x)
        top = max(first_y, second_y)
        right = min(first_x + first_width, second_x + second_width)
        bottom = min(first_y + first_height, second_y + second_height)

        if right <= left or bottom <= top:
            return 0.0

        intersection = float((right - left) * (bottom - top))
        first_area = float(first_width * first_height)
        second_area = float(second_width * second_height)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _prefer_larger_rectangle(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        """Keep the larger of two duplicate rectangles to preserve full crop coverage."""
        first_area = first[2] * first[3]
        second_area = second[2] * second[3]
        return first if first_area >= second_area else second

    @staticmethod
    def _sort_rectangles(rectangles: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Sort rectangles row-wise from top to bottom and left to right."""
        if not rectangles:
            return []

        median_height = int(np.median([height for _, _, _, height in rectangles]))
        row_tolerance = max(10, median_height // 2)
        sorted_by_position = sorted(rectangles, key=lambda rect: (rect[1], rect[0]))

        rows: list[list[tuple[int, int, int, int]]] = []
        for rectangle in sorted_by_position:
            if not rows or abs(rectangle[1] - rows[-1][0][1]) > row_tolerance:
                rows.append([rectangle])
                continue
            rows[-1].append(rectangle)

        ordered_rectangles: list[tuple[int, int, int, int]] = []
        for row in rows:
            ordered_rectangles.extend(sorted(row, key=lambda rect: rect[0]))
        return ordered_rectangles

    @staticmethod
    def _filter_outliers(rectangles: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
        """Remove rectangle outliers using data-derived size bounds."""
        if len(rectangles) < 4:
            return rectangles

        widths = np.array([width for _, _, width, _ in rectangles], dtype=np.float32)
        heights = np.array([height for _, _, _, height in rectangles], dtype=np.float32)
        median_width = float(np.median(widths))
        median_height = float(np.median(heights))

        filtered = [
            rectangle
            for rectangle in rectangles
            if 0.55 * median_width <= rectangle[2] <= 1.65 * median_width
            and 0.55 * median_height <= rectangle[3] <= 1.65 * median_height
        ]
        return filtered or rectangles

    def _save_visualization(
        self,
        source_image: np.ndarray,
        entries: list[EntryBoundingBox],
        image_name: str,
        page: int,
    ) -> Path | None:
        """Draw entry bounding boxes and persist the visualization when debugging is enabled."""
        if not self.debug_enabled:
            return None

        visualization_dir = ensure_directory(self.debug_output_dir / image_name)
        visualization_path = visualization_dir / f"page_{page:04d}_entries.png"
        canvas = source_image.copy() if source_image.ndim == 3 else cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)

        for entry in entries:
            top_left = (entry.x, entry.y)
            bottom_right = (entry.x + entry.width, entry.y + entry.height)
            cv2.rectangle(canvas, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(
                canvas,
                str(entry.entry_index),
                (entry.x + 6, entry.y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return save_image(visualization_path, canvas)
