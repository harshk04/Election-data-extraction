"""Unit tests for OCR-to-JSON voter record extraction."""

from app.models.image import OCRBoundingBoxPoint, OCRResult, OCRTextLine
from app.services.extraction_service import ExtractionService


def _line(text: str, x_coord: float, y_coord: float) -> OCRTextLine:
    return OCRTextLine(
        text=text,
        confidence=0.9,
        language="en",
        bounding_box=[
            OCRBoundingBoxPoint(x=x_coord, y=y_coord),
            OCRBoundingBoxPoint(x=x_coord + 100, y=y_coord),
            OCRBoundingBoxPoint(x=x_coord + 100, y=y_coord + 20),
            OCRBoundingBoxPoint(x=x_coord, y=y_coord + 20),
        ],
    )


def test_parse_voter_record_extracts_expected_fields() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(
        lines=[
            _line("123 ABC1234567", 10, 10),
            _line("Name: Suresh Kumar", 10, 40),
            _line("पिता का नाम: Ramesh Kumar", 10, 70),
            _line("House No: 42/A", 10, 100),
            _line("Age: 34 Gender: Male", 10, 130),
        ]
    )

    result = service.parse_voter_record(ocr_result)

    assert result.serial_number == "123"
    assert result.epic_number == "ABC1234567"
    assert result.elector_name == "Suresh Kumar"
    assert result.relation_type == "father"
    assert result.relation_name == "Ramesh Kumar"
    assert result.house_number == "42/A"
    assert result.age == 34
    assert result.gender == "male"


def test_parse_voter_record_supports_husband_label_and_ocr_noise() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(
        lines=[
            _line("Q 87 XYZ7654321", 10, 10),
            _line("Elector Name Sunita Devi", 10, 40),
            _line("पति का नाभ: Mohan Lal", 10, 70),
            _line("मकान संख्या : 18B", 10, 100),
            _line("आयु: 29 लिंग: Femaie", 10, 130),
        ]
    )

    result = service.parse_voter_record(ocr_result)

    assert result.serial_number == "87"
    assert result.epic_number == "XYZ7654321"
    assert result.elector_name == "Sunita Devi"
    assert result.relation_type == "husband"
    assert result.relation_name == "Mohan Lal"
    assert result.house_number == "18B"
    assert result.age == 29
    assert result.gender == "female"


def test_parse_voter_record_uses_next_line_for_values_when_label_is_isolated() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(
        lines=[
            _line("Serial 56", 10, 10),
            _line("EPIC: DEF2345678", 10, 40),
            _line("निर्वाचक का नाम", 10, 70),
            _line("Geeta Bai", 10, 95),
            _line("पति का नाम", 10, 120),
            _line("Raju", 10, 145),
            _line("House Number", 10, 170),
            _line("17-C", 10, 195),
        ]
    )

    result = service.parse_voter_record(ocr_result)

    assert result.serial_number == "56"
    assert result.elector_name == "Geeta Bai"
    assert result.relation_type == "husband"
    assert result.relation_name == "Raju"
    assert result.house_number == "17-C"


def test_parse_voter_record_handles_label_suffix_variations() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(
        lines=[
            _line("124 ABC1234567", 10, 10),
            _line("निर्वाचक का नाम - Meena", 10, 40),
            _line("पति का नाम : Sohan", 10, 70),
            _line("House No- 9C", 10, 100),
            _line("Age- 31", 10, 130),
            _line("Gender- Male", 10, 160),
        ]
    )

    result = service.parse_voter_record(ocr_result)

    assert result.elector_name == "Meena"
    assert result.relation_type == "husband"
    assert result.relation_name == "Sohan"
    assert result.house_number == "9C"
    assert result.age == 31
    assert result.gender == "male"


def test_parse_voter_record_handles_missing_fields_gracefully() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(lines=[_line("Unclear entry text only", 10, 10)])

    result = service.parse_voter_record(ocr_result)

    assert result.serial_number is None
    assert result.epic_number is None
    assert result.relation_type is None
    assert result.relation_name is None
    assert result.age is None
    assert result.gender is None
    assert result.raw_text == "Unclear entry text only"
