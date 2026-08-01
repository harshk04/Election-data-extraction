"""Service implementation for page preprocessing and crop generation."""

from pathlib import Path

import cv2
import numpy as np

from app.config.settings import get_settings
from app.models.image import CropMetadata, EntryBoundingBox, PreprocessingConfig, PreprocessingResult
from app.utils.file_utils import ensure_directory, load_image, save_image
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ImageService:
    """Handle page preprocessing and record crop extraction."""

    def __init__(
        self,
        preprocessing_config: PreprocessingConfig | None = None,
        debug_enabled: bool | None = None,
        debug_output_dir: Path | None = None,
    ) -> None:
        settings = get_settings()
        self.preprocessing_config = preprocessing_config or PreprocessingConfig(
            enable_deskew=settings.preprocessing_enable_deskew,
            enable_denoise=settings.preprocessing_enable_denoise,
            enable_contrast=settings.preprocessing_enable_contrast,
            enable_threshold=settings.preprocessing_enable_threshold,
            enable_border_cleanup=settings.preprocessing_enable_border_cleanup,
            clahe_clip_limit=settings.preprocessing_clahe_clip_limit,
            clahe_tile_grid_size=settings.preprocessing_clahe_tile_grid_size,
            threshold_block_size=settings.preprocessing_threshold_block_size,
            threshold_c=settings.preprocessing_threshold_c,
            denoise_kernel_size=settings.preprocessing_denoise_kernel_size,
            border_margin=settings.preprocessing_border_margin,
        )
        self.debug_enabled = settings.app_debug if debug_enabled is None else debug_enabled
        self.debug_output_dir = debug_output_dir or settings.preprocessing_debug_dir
        self.crop_min_width = settings.crop_min_width
        self.crop_min_height = settings.crop_min_height

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load an image from disk."""
        logger.info("Loading image", extra={"image_path": str(image_path)})
        return load_image(image_path)

    def preprocess_image_file(
        self,
        image_path: Path,
        output_path: Path | None = None,
    ) -> PreprocessingResult:
        """Load, preprocess, and save a cleaned page image."""
        image = self.load_image(image_path)
        preprocessed_image, debug_images = self.preprocess_page(image)

        target_path = output_path or image_path.with_name(f"{image_path.stem}_clean.png")
        save_image(target_path, preprocessed_image)

        debug_image_paths = self._save_debug_images(image_path.stem, debug_images)

        return PreprocessingResult(
            image_path=target_path,
            width=int(preprocessed_image.shape[1]),
            height=int(preprocessed_image.shape[0]),
            debug_image_paths=debug_image_paths,
        )

    def preprocess_page(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Preprocess a page image before OCR."""
        logger.info("Preprocessing page image")

        processed = self._to_grayscale(image)
        debug_images: dict[str, np.ndarray] = {"original": processed.copy()}

        if self.preprocessing_config.enable_deskew:
            processed = self._deskew(processed)
            debug_images["deskew"] = processed.copy()

        if self.preprocessing_config.enable_denoise:
            processed = self._denoise(processed)
            debug_images["denoise"] = processed.copy()

        if self.preprocessing_config.enable_contrast:
            processed = self._enhance_contrast(processed)
            debug_images["contrast"] = processed.copy()

        if self.preprocessing_config.enable_threshold:
            processed = self._adaptive_threshold(processed)
            debug_images["threshold"] = processed.copy()

        if self.preprocessing_config.enable_border_cleanup:
            processed = self._cleanup_border(processed)
            debug_images["border_cleanup"] = processed.copy()

        return processed, debug_images

    def extract_record_crops(
        self,
        image: np.ndarray,
        bounding_boxes: list[EntryBoundingBox],
        output_dir: Path,
    ) -> list[CropMetadata]:
        """Extract individual voter record crops from a page image."""
        logger.info(
            "Extracting record crops",
            extra={"output_dir": str(output_dir), "requested_boxes": len(bounding_boxes)},
        )

        ensure_directory(output_dir)
        image_height, image_width = image.shape[:2]
        crop_metadata: list[CropMetadata] = []

        for bounding_box in sorted(bounding_boxes, key=lambda box: (box.page, box.entry_index)):
            normalized_box = self._normalize_bounding_box(bounding_box, image_width, image_height)
            if normalized_box is None:
                logger.info(
                    "Skipping invalid or tiny bounding box",
                    extra={"page": bounding_box.page, "entry_index": bounding_box.entry_index},
                )
                continue

            x_coord, y_coord, width, height = normalized_box
            crop = image[y_coord : y_coord + height, x_coord : x_coord + width]
            image_path = output_dir / (
                f"page_{bounding_box.page:03d}_entry_{bounding_box.entry_index:03d}.png"
            )
            cv2.imwrite(str(image_path), crop)

            crop_metadata.append(
                CropMetadata(
                    page=bounding_box.page,
                    entry_index=bounding_box.entry_index,
                    x=x_coord,
                    y=y_coord,
                    width=width,
                    height=height,
                    image_path=image_path,
                )
            )

        return crop_metadata

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert an image to grayscale if needed."""
        if image.ndim == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Estimate skew angle and rotate the image back to horizontal."""
        inverted = cv2.bitwise_not(image)
        _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        coordinates = cv2.findNonZero(binary)
        if coordinates is None:
            return image

        angle = cv2.minAreaRect(coordinates)[-1]
        if angle < -45:
            angle = 90 + angle
        else:
            angle = angle

        if abs(angle) < 0.1:
            return image

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Reduce image noise while preserving text edges."""
        kernel_size = self._normalized_odd_value(self.preprocessing_config.denoise_kernel_size)
        return cv2.medianBlur(image, kernel_size)

    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement to improve OCR readability."""
        tile_size = max(1, self.preprocessing_config.clahe_tile_grid_size)
        clahe = cv2.createCLAHE(
            clipLimit=self.preprocessing_config.clahe_clip_limit,
            tileGridSize=(tile_size, tile_size),
        )
        return clahe.apply(image)

    def _adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Binarize the image using adaptive Gaussian thresholding."""
        block_size = self._normalized_odd_value(self.preprocessing_config.threshold_block_size)
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            self.preprocessing_config.threshold_c,
        )

    def _cleanup_border(self, image: np.ndarray) -> np.ndarray:
        """Remove dark artifacts touching the border and whiten the page margin."""
        cleaned = image.copy()
        margin = max(0, self.preprocessing_config.border_margin)
        if margin > 0 and margin < min(cleaned.shape[:2]):
            cleaned[:margin, :] = 255
            cleaned[-margin:, :] = 255
            cleaned[:, :margin] = 255
            cleaned[:, -margin:] = 255

        if cleaned.ndim != 2:
            return cleaned

        working = cleaned.copy()
        flood_mask = np.zeros((working.shape[0] + 2, working.shape[1] + 2), dtype=np.uint8)
        border_points = [
            (0, 0),
            (working.shape[1] - 1, 0),
            (0, working.shape[0] - 1),
            (working.shape[1] - 1, working.shape[0] - 1),
        ]

        for x_coord, y_coord in border_points:
            if working[y_coord, x_coord] < 250:
                cv2.floodFill(working, flood_mask, (x_coord, y_coord), 255)

        return working

    def _save_debug_images(
        self,
        image_stem: str,
        debug_images: dict[str, np.ndarray],
    ) -> list[Path]:
        """Persist intermediate pipeline images when debug mode is enabled."""
        if not self.debug_enabled:
            return []

        debug_dir = ensure_directory(self.debug_output_dir / image_stem)
        debug_paths: list[Path] = []

        for step_name, step_image in debug_images.items():
            step_path = debug_dir / f"{step_name}.png"
            debug_paths.append(save_image(step_path, step_image))

        return debug_paths

    @staticmethod
    def _normalized_odd_value(value: int) -> int:
        """Normalize values that OpenCV expects to be positive odd integers."""
        normalized = max(1, value)
        if normalized % 2 == 0:
            normalized += 1
        return normalized

    def _normalize_bounding_box(
        self,
        bounding_box: EntryBoundingBox,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        """Clamp bounding boxes to image bounds and reject tiny invalid rectangles."""
        x_coord = max(0, bounding_box.x)
        y_coord = max(0, bounding_box.y)
        if x_coord >= image_width or y_coord >= image_height:
            return None

        max_width = max(0, image_width - x_coord)
        max_height = max(0, image_height - y_coord)
        width = min(bounding_box.width, max_width)
        height = min(bounding_box.height, max_height)

        if width < self.crop_min_width or height < self.crop_min_height:
            return None

        if width <= 0 or height <= 0:
            return None

        return x_coord, y_coord, width, height
