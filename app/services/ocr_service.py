"""Reusable OCR service for cropped voter-entry images."""

import os
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.config.settings import get_settings
from app.models.image import OCRBoundingBoxPoint, OCRResult, OCRTextLine
from app.utils.file_utils import load_image
from app.utils.logging import get_logger

logger = get_logger(__name__)


class OCREngineProtocol(Protocol):
    """Minimal OCR engine contract used by the application."""

    def ocr(self, image: np.ndarray, cls: bool = True) -> Any:
        """Run OCR on an image array."""


class OCRService:
    """Run OCR on extracted crops or page images."""

    def __init__(self, engine: OCREngineProtocol | None = None, language: str | None = None) -> None:
        settings = get_settings()
        self._engine = engine
        self._initialization_error: RuntimeError | None = None
        self.language = language or settings.ocr_language
        self.model_base_dir = settings.ocr_model_base_dir

    def initialize_engine(self) -> OCREngineProtocol:
        """Initialize and return the OCR engine lazily."""
        if self._engine is not None:
            logger.debug("Reusing cached OCR engine")
            return self._engine
        if self._initialization_error is not None:
            raise self._initialization_error

        logger.info("Initializing OCR engine")
        model_base_dir = self._prepare_model_base_dir()
        os.environ["PADDLE_OCR_BASE_DIR"] = str(model_base_dir)
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            self._initialization_error = RuntimeError(self._build_dependency_error_message(exc))
            raise self._initialization_error from exc

        self._engine = PaddleOCR(use_angle_cls=True, lang=self.language, show_log=False)
        return self._engine

    def _prepare_model_base_dir(self) -> Path:
        """Ensure PaddleOCR model downloads stay inside the project workspace."""
        self.model_base_dir.mkdir(parents=True, exist_ok=True)
        return self.model_base_dir

    @staticmethod
    def _build_dependency_error_message(exc: ImportError) -> str:
        """Explain common PaddleOCR dependency issues with actionable guidance."""
        installed_versions: dict[str, str] = {}
        for package_name in ("paddleocr", "paddlepaddle", "paddle"):
            try:
                installed_versions[package_name] = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                continue

        message = (
            "Unable to initialize PaddleOCR. "
            "This project expects the PaddlePaddle runtime package `paddlepaddle`, not the unrelated `paddle` package."
        )

        if "paddlepaddle" not in installed_versions:
            message += " `paddlepaddle` is not installed in the current environment."

        if "paddle" in installed_versions and "paddlepaddle" not in installed_versions:
            message += f" Detected `paddle=={installed_versions['paddle']}`, which is the wrong package for OCR."

        message += (
            " Use a Python 3.12 virtualenv for this project, uninstall `paddle`, then install a compatible "
            "`paddlepaddle` build before running `python main.py` again."
        )

        if str(exc):
            message += f" Original import error: {exc}"

        return message

    def run_ocr(self, image_path: Path) -> OCRResult:
        """Run OCR on a given image path and return structured OCR output."""
        logger.info("Running OCR", extra={"image_path": str(image_path)})
        image = load_image(image_path)
        result = self.run_ocr_on_image(image)
        return OCRResult(image_path=image_path, lines=result.lines)

    def run_ocr_on_image(self, image: np.ndarray) -> OCRResult:
        """Run OCR on an image array and normalize the raw engine output."""
        engine = self.initialize_engine()
        prepared_image = self._prepare_image(image)
        raw_result = engine.ocr(prepared_image, cls=True)
        return self._normalize_result(raw_result)

    def _normalize_result(self, raw_result: Any) -> OCRResult:
        """Convert PaddleOCR-style responses into structured OCR objects."""
        normalized_lines: list[OCRTextLine] = []
        if not raw_result:
            return OCRResult(lines=normalized_lines)

        result_pages = raw_result if isinstance(raw_result, list) else [raw_result]
        for page_result in result_pages:
            if not page_result:
                continue
            for line in page_result:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                box, content = line[0], line[1]
                if not isinstance(content, (list, tuple)) or len(content) < 2:
                    continue
                text, confidence = content[0], content[1]
                bounding_box = self._normalize_bounding_box(box)
                normalized_lines.append(
                    OCRTextLine(
                        text=str(text),
                        confidence=float(confidence),
                        language=self.language,
                        bounding_box=bounding_box,
                    )
                )

        return OCRResult(lines=normalized_lines)

    @staticmethod
    def _normalize_bounding_box(box: Any) -> list[OCRBoundingBoxPoint]:
        """Convert PaddleOCR polygon data into typed point objects."""
        if not isinstance(box, (list, tuple)):
            return []

        points: list[OCRBoundingBoxPoint] = []
        for point in box:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            points.append(OCRBoundingBoxPoint(x=float(point[0]), y=float(point[1])))
        return points

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        """Apply lightweight OCR-friendly normalization without changing geometry."""
        if image.ndim == 2:
            grayscale = image
        else:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if OCRService._is_binary_image(grayscale):
            return grayscale

        return cv2.fastNlMeansDenoising(grayscale, h=7, templateWindowSize=7, searchWindowSize=21)

    @staticmethod
    def _is_binary_image(image: np.ndarray) -> bool:
        """Detect whether an image is already near-binary to avoid redundant preprocessing."""
        unique_values = np.unique(image)
        return len(unique_values) <= 4
