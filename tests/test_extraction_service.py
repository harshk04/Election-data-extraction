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


def test_parse_voter_record_reconstructs_fragmented_hindi_rows() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(
        lines=[
            _line("३१", 30, 10),
            _line("HDR2OTOB4E", 420, 12),
            _line("निवाचक का नाम", 10, 60),
            _line("गीता देवी", 180, 62),
            _line("पति का नाम", 10, 98),
            _line("कैलाश", 165, 100),
            _line("गृह संख्या", 10, 136),
            _line("8", 160, 138),
            _line("उम्र", 10, 174),
            _line("56", 80, 174),
            _line("लिंग", 130, 174),
            _line("महिला", 190, 174),
        ]
    )

    result = service.parse_voter_record(ocr_result)

    assert result.serial_number == "31"
    assert result.epic_number == "RDR2070845"
    assert result.elector_name == "गीता देवी"
    assert result.relation_type == "husband"
    assert result.relation_name == "कैलाश"
    assert result.house_number == "8"
    assert result.age == 56
    assert result.gender == "female"


def test_parse_voter_record_handles_deleted_q_serial_and_noisy_epic() -> None:
    service = ExtractionService()
    ocr_result = OCRResult(
        lines=[
            _line("Q", 25, 10),
            _line("३३", 170, 10),
            _line("ADR2417BO?", 420, 10),
            _line("निवाचक का नाम", 10, 60),
            _line("रोहिताश भट्टेल", 185, 60),
            _line("पिता का नाम", 10, 96),
            _line("बंशीधर", 170, 96),
            _line("गृह संख्या", 10, 132),
            _line("8", 160, 132),
            _line("उम्र", 10, 168),
            _line("45", 75, 168),
            _line("लिंग", 130, 168),
            _line("पुरुष", 190, 168),
        ]
    )

    result = service.parse_voter_record(ocr_result)

    assert result.serial_number == "33"
    assert result.epic_number == "RDR2417806"
    assert result.elector_name == "रोहिताश भट्टेल"
    assert result.relation_type == "father"
    assert result.relation_name == "बंशीधर"
    assert result.house_number == "8"
    assert result.age == 45
    assert result.gender == "male"
