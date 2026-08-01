"""Unit tests for voter record validation."""

from app.models.image import DeletedEntryDetectionResult, OCRBoundingBoxPoint, OCRResult, OCRTextLine
from app.models.voter import VoterRecord
from app.services.validation_service import ValidationService


def _line(text: str, confidence: float, x_coord: float, y_coord: float) -> OCRTextLine:
    return OCRTextLine(
        text=text,
        confidence=confidence,
        language="en",
        bounding_box=[
            OCRBoundingBoxPoint(x=x_coord, y=y_coord),
            OCRBoundingBoxPoint(x=x_coord + 50, y=y_coord),
            OCRBoundingBoxPoint(x=x_coord + 50, y=y_coord + 15),
            OCRBoundingBoxPoint(x=x_coord, y=y_coord + 15),
        ],
    )


def test_validate_record_reports_invalid_fields_and_retry_recommendations() -> None:
    service = ValidationService(ocr_confidence_threshold=0.8)
    record = VoterRecord(
        serial_number="12",
        epic_number="AB1234567",
        elector_name="Rita",
        relation_type="father",
        relation_name="Mohan",
        house_number="",
        age=17,
        gender="female",
    )
    ocr_result = OCRResult(
        lines=[
            _line("AB1234567", 0.63, 10, 10),
            _line("Age: 17", 0.71, 10, 30),
            _line("Gender: female", 0.69, 10, 50),
            _line("Father Name: Mohan", 0.74, 10, 70),
        ]
    )

    report = service.validate_record(record, ocr_result)

    assert report.is_valid is False
    assert {issue.field_name for issue in report.issues} >= {
        "epic_number",
        "age",
        "gender",
        "house_number",
        "relation_type",
    }
    assert {item.field_name for item in report.retry_fields} >= {
        "epic_number",
        "age",
        "gender",
        "relation_name",
    }


def test_validate_record_passes_for_valid_values() -> None:
    service = ValidationService(ocr_confidence_threshold=0.7)
    record = VoterRecord(
        serial_number="123",
        epic_number="ABC1234567",
        elector_name="Sita Devi",
        relation_type="Father",
        relation_name="Mohan Lal",
        house_number="12A",
        age=34,
        gender="महिला",
    )
    ocr_result = OCRResult(
        lines=[
            _line("ABC1234567", 0.95, 10, 10),
            _line("महिला", 0.93, 10, 30),
            _line("Age: 34", 0.92, 10, 50),
            _line("Father: Mohan Lal", 0.9, 10, 70),
            _line("House No: 12A", 0.91, 10, 90),
        ]
    )

    report = service.validate_record(record, ocr_result)

    assert report.is_valid is True
    assert report.issues == []
    assert report.retry_fields == []


def test_validate_record_never_modifies_values() -> None:
    service = ValidationService()
    record = VoterRecord(
        epic_number="abc1234567",
        relation_type="husband",
        house_number="  ",
        age=130,
        gender="male",
    )
    ocr_result = OCRResult(lines=[])

    report = service.validate_record(record, ocr_result)

    assert record.epic_number == "abc1234567"
    assert record.relation_type == "husband"
    assert record.gender == "male"
    assert report.is_valid is False


def test_validate_record_accepts_normalized_parser_values() -> None:
    service = ValidationService()
    record = VoterRecord(
        epic_number="ABC1234567",
        relation_type="father",
        relation_name="Ramlal",
        house_number="12",
        age=45,
        gender="male",
    )
    ocr_result = OCRResult(
        lines=[
            _line("ABC1234567", 0.92, 10, 10),
            _line("Father Name: Ramlal", 0.88, 10, 30),
            _line("Male", 0.9, 10, 50),
            _line("Age 45", 0.94, 10, 70),
            _line("12", 0.91, 10, 90),
        ]
    )

    report = service.validate_record(record, ocr_result)

    assert report.is_valid is True


def test_validate_record_reports_deleted_detection_issue() -> None:
    service = ValidationService()
    record = VoterRecord(
        epic_number="ABC1234567",
        relation_type="Father",
        relation_name="Ramlal",
        house_number="12",
        age=45,
        gender="महिला",
    )
    ocr_result = OCRResult(lines=[_line("ABC1234567", 0.95, 10, 10)])
    deleted_result = DeletedEntryDetectionResult(deleted=True, confidence=0.4)

    report = service.validate_record(record, ocr_result, deleted_result)

    assert any(issue.field_name == "deleted_detection" for issue in report.issues)
