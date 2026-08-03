"""Single-command entrypoint for the Electoral Roll OCR pipeline."""

from __future__ import annotations

from pathlib import Path

from app.config.settings import get_settings
from app.main import run_pipeline
from app.utils.logging import configure_logging, get_logger


# Update this path when you want to process a different PDF.
INPUT_PDF_PATH = Path("data/pdfs/FIle-1.pdf")

# Leave as None to save into outputs/<pdf-name>.json automatically.
OUTPUT_JSON_PATH: Path | None = None

logger = get_logger(__name__)


def main() -> int:
    """Run the OCR pipeline using the PDF path defined in this file."""
    configure_logging()
    settings = get_settings()

    pdf_path = INPUT_PDF_PATH
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")

    output_path = OUTPUT_JSON_PATH or settings.outputs_dir / f"{pdf_path.stem}.json"

    logger.info(
        "Starting Electoral Roll OCR pipeline",
        extra={
            "app_name": settings.app_name,
            "environment": settings.app_env,
            "pdf_path": str(pdf_path),
            "output_path": str(output_path),
        },
    )

    results = run_pipeline(pdf_path=pdf_path, output_path=output_path)

    logger.info(
        "Pipeline completed",
        extra={"record_count": len(results), "output_path": str(output_path)},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
