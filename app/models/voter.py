"""Structured voter data models."""

from pydantic import BaseModel, Field


class VoterRecord(BaseModel):
    """Structured representation of a voter record."""

    serial_number: str | None = None
    epic_number: str | None = None
    elector_name: str | None = None
    relation_type: str | None = None
    relation_name: str | None = None
    house_number: str | None = None
    age: int | None = None
    gender: str | None = None
    deleted: bool | None = None
    raw_text: str | None = None


class FieldValidationIssue(BaseModel):
    """Single validation issue for a parsed field."""

    field_name: str
    issue_code: str
    message: str
    value: str | int | None = None
    confidence: float | None = None


class FieldRetryRecommendation(BaseModel):
    """Recommendation to retry a field due to low OCR confidence."""

    field_name: str
    confidence: float
    reason: str


class ValidationReport(BaseModel):
    """Validation result for a parsed voter record."""

    is_valid: bool
    issues: list[FieldValidationIssue] = Field(default_factory=list)
    retry_fields: list[FieldRetryRecommendation] = Field(default_factory=list)
