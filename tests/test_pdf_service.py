"""Unit tests for PDF rendering."""

from pathlib import Path

import fitz
import pytest

from app.services.pdf_service import PDFService


def _create_sample_pdf(pdf_path: Path, page_count: int = 2) -> None:
    document = fitz.open()
    try:
        for index in range(page_count):
            page = document.new_page(width=300, height=200)
            page.insert_text((72, 72), f"Page {index + 1}")
        document.save(pdf_path)
    finally:
        document.close()


def _create_encrypted_pdf(pdf_path: Path) -> None:
    source_path = pdf_path.with_name("source.pdf")
    document = fitz.open()
    try:
        page = document.new_page(width=300, height=200)
        page.insert_text((72, 72), "Encrypted")
        document.save(
            source_path,
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner-secret",
            user_pw="user-secret",
        )
    finally:
        document.close()

    source_path.rename(pdf_path)


def test_extract_pages_renders_pngs_in_order(tmp_path: Path) -> None:
    pdf_path = tmp_path / "Sample Roll.pdf"
    pages_root_dir = tmp_path / "pages"
    service = PDFService(pages_root_dir=pages_root_dir)
    _create_sample_pdf(pdf_path, page_count=2)

    result = service.extract_pages(pdf_path)

    assert len(result) == 2
    assert [page.page_number for page in result] == [1, 2]
    assert [page.image_path.name for page in result] == ["page_0001.png", "page_0002.png"]
    assert result[0].image_path.parent == pages_root_dir / "Sample_Roll"
    assert result[1].image_path.parent == pages_root_dir / "Sample_Roll"
    assert all(page.image_path.exists() for page in result)
    assert all(page.width > 0 and page.height > 0 for page in result)
    assert result[0].width == 2500
    assert result[0].height == 1667


def test_extract_pages_raises_for_missing_file(tmp_path: Path) -> None:
    service = PDFService(pages_root_dir=tmp_path / "pages")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        service.extract_pages(tmp_path / "missing.pdf")


def test_extract_pages_raises_for_invalid_pdf(tmp_path: Path) -> None:
    invalid_pdf_path = tmp_path / "invalid.pdf"
    invalid_pdf_path.write_text("not a valid pdf", encoding="utf-8")
    service = PDFService(pages_root_dir=tmp_path / "pages")

    with pytest.raises(ValueError, match="Invalid PDF file"):
        service.extract_pages(invalid_pdf_path)


def test_extract_pages_raises_for_encrypted_pdf(tmp_path: Path) -> None:
    encrypted_pdf_path = tmp_path / "encrypted.pdf"
    _create_encrypted_pdf(encrypted_pdf_path)
    service = PDFService(pages_root_dir=tmp_path / "pages")

    with pytest.raises(PermissionError, match="Encrypted PDF is not supported"):
        service.extract_pages(encrypted_pdf_path)
