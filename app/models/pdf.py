"""PDF rendering models."""

from pathlib import Path

from pydantic import BaseModel


class PageRenderMetadata(BaseModel):
    """Metadata returned for each rendered PDF page."""

    page_number: int
    width: int
    height: int
    image_path: Path
