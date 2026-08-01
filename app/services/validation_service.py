"""Validation service for extracted voter records."""

import re

from app.config.settings import get_settings
from app.models.image import DeletedEntryDetectionResult, OCRResult
from app.models.voter import (
    FieldRetryRecommendation,
    FieldValidationIssue,
    ValidationReport,
    VoterRecord,
)
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ValidationService:
    """Validate extracted voter records without mutating their values."""

    def __init__(self, ocr_confidence_threshold: float | None = None) -> None:
        settings = get_settings()
        self.ocr_confidence_threshold = (
            settings.validation_ocr_confidence_threshold
            if ocr_confidence_threshold is None
            else ocr_confidence_threshold
        )

    _VALID_GENDER_VALUES = {"पुरुष", "महिला", "male", "female"}
    _VALID_RELATION_VALUES = {"Father", "Husband", "father", "husband"}

    def validate_record(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        deleted_result: DeletedEntryDetectionResult | None = None,
    ) -> ValidationReport:
        """Validate a parsed voter record and recommend retries for weak OCR fields."""
        issues: list[FieldValidationIssue] = []
        retry_fields: list[FieldRetryRecommendation] = []

        self._validate_epic_number(record, ocr_result, issues, retry_fields)
        self._validate_age(record, ocr_result, issues, retry_fields)
        self._validate_gender(record, ocr_result, issues, retry_fields)
        self._validate_house_number(record, ocr_result, issues, retry_fields)
        self._validate_relation(record, ocr_result, issues, retry_fields)
        self._validate_deleted_detection(deleted_result, issues)

        return ValidationReport(
            is_valid=not issues,
            issues=issues,
            retry_fields=retry_fields,
        )

    def _validate_epic_number(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        issues: list[FieldValidationIssue],
        retry_fields: list[FieldRetryRecommendation],
    ) -> None:
        confidence = self._field_confidence(record.epic_number, ocr_result)
        if record.epic_number is None:
            self._append_issue(
                issues,
                field_name="epic_number",
                issue_code="missing_epic_number",
                message="EPIC number is missing.",
                value=None,
                confidence=confidence,
            )
        elif not re.fullmatch(r"[A-Z]{3}[0-9]{7}", str(record.epic_number).upper()):
            self._append_issue(
                issues,
                field_name="epic_number",
                issue_code="invalid_epic_format",
                message="EPIC number must match [A-Z]{3}[0-9]{7}.",
                value=record.epic_number,
                confidence=confidence,
            )

        self._append_retry_if_low_confidence(
            retry_fields=retry_fields,
            field_name="epic_number",
            confidence=confidence,
        )

    def _validate_age(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        issues: list[FieldValidationIssue],
        retry_fields: list[FieldRetryRecommendation],
    ) -> None:
        confidence = self._field_confidence(record.age, ocr_result)
        if record.age is None:
            self._append_issue(
                issues,
                field_name="age",
                issue_code="missing_age",
                message="Age is missing.",
                value=None,
                confidence=confidence,
            )
        elif not 18 <= record.age <= 120:
            self._append_issue(
                issues,
                field_name="age",
                issue_code="invalid_age_range",
                message="Age must be between 18 and 120.",
                value=record.age,
                confidence=confidence,
            )

        self._append_retry_if_low_confidence(
            retry_fields=retry_fields,
            field_name="age",
            confidence=confidence,
        )

    def _validate_gender(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        issues: list[FieldValidationIssue],
        retry_fields: list[FieldRetryRecommendation],
    ) -> None:
        confidence = self._field_confidence(record.gender, ocr_result)
        if record.gender is None:
            self._append_issue(
                issues,
                field_name="gender",
                issue_code="missing_gender",
                message="Gender is missing.",
                value=None,
                confidence=confidence,
            )
        elif record.gender not in self._VALID_GENDER_VALUES:
            self._append_issue(
                issues,
                field_name="gender",
                issue_code="invalid_gender_value",
                message="Gender must map to one of: पुरुष, महिला.",
                value=record.gender,
                confidence=confidence,
            )

        self._append_retry_if_low_confidence(
            retry_fields=retry_fields,
            field_name="gender",
            confidence=confidence,
        )

    def _validate_house_number(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        issues: list[FieldValidationIssue],
        retry_fields: list[FieldRetryRecommendation],
    ) -> None:
        confidence = self._field_confidence(record.house_number, ocr_result)
        if record.house_number is None or not str(record.house_number).strip():
            self._append_issue(
                issues,
                field_name="house_number",
                issue_code="empty_house_number",
                message="House number must not be empty.",
                value=record.house_number,
                confidence=confidence,
            )

        self._append_retry_if_low_confidence(
            retry_fields=retry_fields,
            field_name="house_number",
            confidence=confidence,
        )

    def _validate_relation(
        self,
        record: VoterRecord,
        ocr_result: OCRResult,
        issues: list[FieldValidationIssue],
        retry_fields: list[FieldRetryRecommendation],
    ) -> None:
        type_confidence = self._field_confidence(record.relation_type, ocr_result)
        name_confidence = self._field_confidence(record.relation_name, ocr_result)
        if record.relation_type is None:
            self._append_issue(
                issues,
                field_name="relation_type",
                issue_code="missing_relation_type",
                message="Relation type is missing.",
                value=None,
                confidence=type_confidence,
            )
        elif record.relation_type not in self._VALID_RELATION_VALUES:
            self._append_issue(
                issues,
                field_name="relation_type",
                issue_code="invalid_relation_type",
                message="Relation type must map to Father or Husband.",
                value=record.relation_type,
                confidence=type_confidence,
            )

        if record.relation_name is None or not str(record.relation_name).strip():
            self._append_issue(
                issues,
                field_name="relation_name",
                issue_code="missing_relation_name",
                message="Relation name is missing.",
                value=record.relation_name,
                confidence=name_confidence,
            )

        self._append_retry_if_low_confidence(
            retry_fields=retry_fields,
            field_name="relation_type",
            confidence=type_confidence,
        )
        self._append_retry_if_low_confidence(
            retry_fields=retry_fields,
            field_name="relation_name",
            confidence=name_confidence,
        )

    def _validate_deleted_detection(
        self,
        deleted_result: DeletedEntryDetectionResult | None,
        issues: list[FieldValidationIssue],
    ) -> None:
        """Validate deleted-entry classifier output when it is available."""
        if deleted_result is None:
            return

        if not 0.0 <= deleted_result.confidence <= 1.0:
            self._append_issue(
                issues,
                field_name="deleted_detection",
                issue_code="invalid_deleted_confidence",
                message="Deleted-entry confidence must be between 0.0 and 1.0.",
                value=deleted_result.confidence,
                confidence=deleted_result.confidence,
            )
        elif deleted_result.deleted and deleted_result.confidence < 0.5:
            self._append_issue(
                issues,
                field_name="deleted_detection",
                issue_code="weak_deleted_confidence",
                message="Deleted-entry flag is set with weak confidence.",
                value=deleted_result.deleted,
                confidence=deleted_result.confidence,
            )

    def _append_issue(
        self,
        issues: list[FieldValidationIssue],
        field_name: str,
        issue_code: str,
        message: str,
        value: str | int | None,
        confidence: float | None,
    ) -> None:
        """Create and log a validation issue."""
        issue = FieldValidationIssue(
            field_name=field_name,
            issue_code=issue_code,
            message=message,
            value=value,
            confidence=confidence,
        )
        logger.warning(
            "Validation issue detected",
            extra={
                "field_name": field_name,
                "issue_code": issue_code,
                "value": value,
                "confidence": confidence,
            },
        )
        issues.append(issue)

    def _append_retry_if_low_confidence(
        self,
        retry_fields: list[FieldRetryRecommendation],
        field_name: str,
        confidence: float | None,
    ) -> None:
        """Append retry recommendation for low-confidence fields."""
        if confidence is None or confidence >= self.ocr_confidence_threshold:
            return

        if any(item.field_name == field_name for item in retry_fields):
            return

        retry_fields.append(
            FieldRetryRecommendation(
                field_name=field_name,
                confidence=confidence,
                reason=(
                    f"OCR confidence {confidence:.2f} is below threshold "
                    f"{self.ocr_confidence_threshold:.2f}."
                ),
            )
        )

    @staticmethod
    def _field_confidence(field_value: str | int | None, ocr_result: OCRResult) -> float | None:
        """Estimate field confidence by matching extracted value against OCR lines."""
        if field_value is None:
            return None

        target = str(field_value).strip()
        if not target:
            return None

        normalized_target = ValidationService._normalize_text(target)
        best_confidence: float | None = None

        for line in ocr_result.lines:
            normalized_line = ValidationService._normalize_text(line.text)
            if normalized_target in normalized_line or normalized_line in normalized_target:
                if best_confidence is None or line.confidence > best_confidence:
                    best_confidence = line.confidence

        return best_confidence

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for matching without changing stored field values."""
        return re.sub(r"\s+", " ", text).strip().upper()
