"""File-system and image helper utilities."""

from pathlib import Path

import cv2
import numpy as np

from app.utils.logging import get_logger

logger = get_logger(__name__)


def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists."""
    logger.debug("Ensuring directory exists", extra={"path": str(path)})
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(image_path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Load an image from disk with consistent validation."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    image = cv2.imread(str(image_path), flags)
    if image is None:
        raise ValueError(f"Unable to load image file: {image_path}")

    return image


def save_image(image_path: Path, image: np.ndarray) -> Path:
    """Save an image to disk and ensure the destination directory exists."""
    ensure_directory(image_path.parent)
    if not cv2.imwrite(str(image_path), image):
        raise ValueError(f"Unable to write image file: {image_path}")
    return image_path
