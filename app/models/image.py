"""Image preprocessing models."""

from pathlib import Path

from pydantic import BaseModel, Field


class PreprocessingConfig(BaseModel):
    """Configuration for the image preprocessing pipeline."""

    enable_deskew: bool = True
    enable_denoise: bool = True
    enable_contrast: bool = True
    enable_threshold: bool = True
    enable_border_cleanup: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    threshold_block_size: int = 31
    threshold_c: int = 15
    denoise_kernel_size: int = 3
    border_margin: int = 12


class PreprocessingResult(BaseModel):
    """Metadata for a preprocessed page image."""

    image_path: Path
    width: int
    height: int
    debug_image_paths: list[Path] = Field(default_factory=list)


class GridDetectionConfig(BaseModel):
    """Configuration for electoral roll entry grid detection."""

    min_width_ratio: float = 0.15
    min_height_ratio: float = 0.05
    max_width_ratio: float = 0.4
    max_height_ratio: float = 0.2
    horizontal_kernel_ratio: int = 30
    vertical_kernel_ratio: int = 30


class EntryBoundingBox(BaseModel):
    """Detected entry rectangle metadata."""

    page: int
    entry_index: int
    x: int
    y: int
    width: int
    height: int


class GridDetectionResult(BaseModel):
    """Detected entry bounding boxes with optional debug visualization."""

    entries: list[EntryBoundingBox]
    visualization_path: Path | None = None


class CropMetadata(BaseModel):
    """Metadata for a saved voter-entry crop."""

    page: int
    entry_index: int
    x: int
    y: int
    width: int
    height: int
    image_path: Path


class DeletedEntryDetectionResult(BaseModel):
    """Classification result for deleted-entry detection."""

    deleted: bool
    confidence: float
    ocr_confidence: float = 0.0
    watermark_confidence: float = 0.0
    debug_image_path: Path | None = None


class OCRBoundingBoxPoint(BaseModel):
    """Single OCR polygon point."""

    x: float
    y: float


class OCRTextLine(BaseModel):
    """Structured OCR text line."""

    text: str
    confidence: float
    language: str
    bounding_box: list[OCRBoundingBoxPoint]


class OCRResult(BaseModel):
    """Structured OCR response for a single voter-entry crop."""

    image_path: Path | None = None
    lines: list[OCRTextLine] = Field(default_factory=list)
