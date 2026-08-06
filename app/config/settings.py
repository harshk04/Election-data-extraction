"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application settings."""

    app_name: str = Field(default="Electoral Roll OCR", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_debug: bool = Field(default=True, alias="APP_DEBUG")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(default=Path("logs"), alias="LOG_DIR")
    log_file_name: str = Field(default="electoral_roll_ocr", alias="LOG_FILE_NAME")
    log_to_file: bool = Field(default=True, alias="LOG_TO_FILE")
    log_max_bytes: int = Field(default=5_242_880, alias="LOG_MAX_BYTES")
    log_backup_count: int = Field(default=5, alias="LOG_BACKUP_COUNT")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    pdfs_dir: Path = Field(default=Path("data/pdfs"), alias="PDFS_DIR")
    pages_dir: Path = Field(default=Path("data/pages"), alias="PAGES_DIR")
    crops_dir: Path = Field(default=Path("data/crops"), alias="CROPS_DIR")
    outputs_dir: Path = Field(default=Path("outputs"), alias="OUTPUTS_DIR")
    preprocessing_debug_dir: Path = Field(
        default=Path("outputs/preprocessing_debug"),
        alias="PREPROCESSING_DEBUG_DIR",
    )
    preprocessing_enable_deskew: bool = Field(default=True, alias="PREPROCESSING_ENABLE_DESKEW")
    preprocessing_enable_denoise: bool = Field(default=True, alias="PREPROCESSING_ENABLE_DENOISE")
    preprocessing_enable_contrast: bool = Field(
        default=True,
        alias="PREPROCESSING_ENABLE_CONTRAST",
    )
    preprocessing_enable_threshold: bool = Field(
        default=True,
        alias="PREPROCESSING_ENABLE_THRESHOLD",
    )
    preprocessing_enable_border_cleanup: bool = Field(
        default=True,
        alias="PREPROCESSING_ENABLE_BORDER_CLEANUP",
    )
    preprocessing_clahe_clip_limit: float = Field(
        default=2.0,
        alias="PREPROCESSING_CLAHE_CLIP_LIMIT",
    )
    preprocessing_clahe_tile_grid_size: int = Field(
        default=8,
        alias="PREPROCESSING_CLAHE_TILE_GRID_SIZE",
    )
    preprocessing_threshold_block_size: int = Field(
        default=31,
        alias="PREPROCESSING_THRESHOLD_BLOCK_SIZE",
    )
    preprocessing_threshold_c: int = Field(default=15, alias="PREPROCESSING_THRESHOLD_C")
    preprocessing_denoise_kernel_size: int = Field(
        default=3,
        alias="PREPROCESSING_DENOISE_KERNEL_SIZE",
    )
    preprocessing_border_margin: int = Field(default=12, alias="PREPROCESSING_BORDER_MARGIN")
    grid_debug_dir: Path = Field(default=Path("outputs/grid_debug"), alias="GRID_DEBUG_DIR")
    grid_min_width_ratio: float = Field(default=0.15, alias="GRID_MIN_WIDTH_RATIO")
    grid_min_height_ratio: float = Field(default=0.05, alias="GRID_MIN_HEIGHT_RATIO")
    grid_max_width_ratio: float = Field(default=0.4, alias="GRID_MAX_WIDTH_RATIO")
    grid_max_height_ratio: float = Field(default=0.2, alias="GRID_MAX_HEIGHT_RATIO")
    grid_horizontal_kernel_ratio: int = Field(default=30, alias="GRID_HORIZONTAL_KERNEL_RATIO")
    grid_vertical_kernel_ratio: int = Field(default=30, alias="GRID_VERTICAL_KERNEL_RATIO")
    crop_min_width: int = Field(default=40, alias="CROP_MIN_WIDTH")
    crop_min_height: int = Field(default=40, alias="CROP_MIN_HEIGHT")
    deleted_debug_dir: Path = Field(default=Path("outputs/deleted_debug"), alias="DELETED_DEBUG_DIR")
    deleted_confidence_threshold: float = Field(
        default=0.65,
        alias="DELETED_CONFIDENCE_THRESHOLD",
    )
    deleted_watermark_min_score: float = Field(
        default=0.2,
        alias="DELETED_WATERMARK_MIN_SCORE",
    )
    deleted_entries_dir: Path = Field(
        default=Path("outputs/classified_crops/deleted"),
        alias="DELETED_ENTRIES_DIR",
    )
    normal_entries_dir: Path = Field(
        default=Path("outputs/classified_crops/normal"),
        alias="NORMAL_ENTRIES_DIR",
    )
    entry_timeout_seconds: float = Field(
        default=20.0,
        alias="ENTRY_TIMEOUT_SECONDS",
    )
    timed_out_entries_dir: Path = Field(
        default=Path("outputs/timed_out_entries"),
        alias="TIMED_OUT_ENTRIES_DIR",
    )
    failed_cases_dir: Path = Field(
        default=Path("outputs/failed_cases"),
        alias="FAILED_CASES_DIR",
    )
    extraction_backend: str = Field(default="auto", alias="EXTRACTION_BACKEND")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model_id: str | None = Field(default=None, alias="GROQ_MODEL_ID")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1", alias="GROQ_BASE_URL")
    groq_request_timeout_seconds: int = Field(
        default=120,
        alias="GROQ_REQUEST_TIMEOUT_SECONDS",
    )
    groq_max_retries: int = Field(default=3, alias="GROQ_MAX_RETRIES")
    groq_quality_retries: int = Field(default=3, alias="GROQ_QUALITY_RETRIES")
    groq_temperature: float = Field(default=0.0, alias="GROQ_TEMPERATURE")
    groq_max_tokens: int = Field(default=256, alias="GROQ_MAX_TOKENS")
    openai_bedrock_base_url: str = Field(
        default="https://bedrock-mantle.ap-south-1.api.aws/v1",
        alias="OPENAI_BEDROCK_BASE_URL",
    )
    openai_bedrock_api_key: str | None = Field(default=None, alias="OPENAI_BEDROCK_API_KEY")
    openai_bedrock_model_id: str | None = Field(default=None, alias="OPENAI_BEDROCK_MODEL_ID")
    openai_bedrock_fallback_model_id: str | None = Field(
        default="google.gemma-3-27b-it",
        alias="OPENAI_BEDROCK_FALLBACK_MODEL_ID",
    )
    openai_bedrock_project: str = Field(default="default", alias="OPENAI_BEDROCK_PROJECT")
    validation_ocr_confidence_threshold: float = Field(
        default=0.8,
        alias="VALIDATION_OCR_CONFIDENCE_THRESHOLD",
    )
    ocr_language: str = Field(default="hi", alias="OCR_LANGUAGE")
    ocr_model_base_dir: Path = Field(
        default=Path("outputs/paddleocr"),
        alias="OCR_MODEL_BASE_DIR",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()
