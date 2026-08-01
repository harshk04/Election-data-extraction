"""Service implementation for PDF ingestion and 600 DPI page rendering."""

import re
from pathlib import Path

import fitz

from app.config.settings import get_settings
from app.models.pdf import PageRenderMetadata
from app.utils.file_utils import ensure_directory
from app.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


class PDFService:
    """Handle scanned PDF loading and page extraction workflows."""

    def __init__(self, pages_root_dir: Path | None = None) -> None:
        settings = get_settings()
        self.pages_root_dir = pages_root_dir or settings.pages_dir

    def open_document(self, pdf_path: Path) -> fitz.Document:
        """Open a PDF document and validate that it can be processed."""
        logger.info("Opening PDF document", extra={"pdf_path": str(pdf_path)})

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {pdf_path}")

        if not pdf_path.is_file():
            raise FileNotFoundError(f"PDF path is not a file: {pdf_path}")

        try:
            document = fitz.open(pdf_path)
        except (fitz.FileDataError, fitz.EmptyFileError, RuntimeError, ValueError) as exc:
            log_exception(logger, "Failed to open PDF document", pdf_path=pdf_path)
            raise ValueError(f"Invalid PDF file: {pdf_path}") from exc

        if document.needs_pass:
            document.close()
            raise PermissionError(f"Encrypted PDF is not supported: {pdf_path}")

        return document

    def extract_pages(self, pdf_path: Path) -> list[PageRenderMetadata]:
        """Render each PDF page to a 600 DPI PNG while preserving page order."""
        logger.info("Extracting pages from PDF", extra={"pdf_path": str(pdf_path)})

        output_dir = ensure_directory(self.pages_root_dir / self._normalize_pdf_name(pdf_path.stem))
        zoom_factor = 600 / 72
        matrix = fitz.Matrix(zoom_factor, zoom_factor)
        rendered_pages: list[PageRenderMetadata] = []

        document = self.open_document(pdf_path)
        try:
            for page_index, page in enumerate(document, start=1):
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                image_path = output_dir / f"page_{page_index:04d}.png"
                width = pixmap.width
                height = pixmap.height
                pixmap.save(image_path)

                rendered_pages.append(
                    PageRenderMetadata(
                        page_number=page_index,
                        width=width,
                        height=height,
                        image_path=image_path,
                    )
                )
        finally:
            document.close()

        return rendered_pages

    @staticmethod
    def _normalize_pdf_name(pdf_name: str) -> str:
        """Convert a PDF stem into a stable directory name."""
        normalized_name = re.sub(r"[^A-Za-z0-9._-]+", "_", pdf_name).strip("._")
        return normalized_name or "document"
