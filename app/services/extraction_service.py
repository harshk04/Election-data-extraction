"""Service implementation for OCR-to-JSON voter record parsing."""

import json
import re
from pathlib import Path
from typing import Iterable

from app.config.settings import get_settings
from app.models.image import OCRResult, OCRTextLine
from app.models.voter import VoterRecord
from app.services.llm_extraction_service import LLMExtractionService
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ExtractionService:
    """Transform raw OCR output into structured voter records."""

    _SEPARATOR_PATTERN = r"(?:\s*[:\-–—]\s*|\s+)"
    _DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
    _RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
        "father": (
            "father",
            "fathers name",
            "father name",
            "पिता का नाम",
            "पिता का नाभ",
            "पिता का नाय",
            "पिता का नाम :",
            "पिता का नाम-",
        ),
        "husband": (
            "husband",
            "husbands name",
            "husband name",
            "पति का नाम",
            "पति का नाभ",
            "पति का नाय",
            "पति का नाम :",
            "पति का नाम-",
        ),
    }
    _NAME_PATTERNS: tuple[str, ...] = (
        "elector name",
        "निर्वाचक का नाम",
        "निवाचक का नाम",
        "नवाचक का नाम",
        "मतदाता का नाम",
        "name",
    )
    _HOUSE_PATTERNS: tuple[str, ...] = (
        "house no",
        "house number",
        "house no.",
        "मकान संख्या",
        "मकान सं",
        "घर संख्या",
        "गृह संख्या",
        "गृह संखया",
        "गृह संक्या",
    )
    _AGE_PATTERNS: tuple[str, ...] = (
        "age",
        "उम्र",
        "आयु",
    )
    _GENDER_PATTERNS: tuple[str, ...] = (
        "gender",
        "sex",
        "लिंग",
        "िलंग",
    )

    def __init__(
        self,
        extraction_backend: str | None = None,
        llm_extraction_service: LLMExtractionService | None = None,
    ) -> None:
        settings = get_settings()
        self.extraction_backend = (extraction_backend or settings.extraction_backend).strip().lower()
        self._llm_extraction_service = llm_extraction_service

        if self.extraction_backend not in {"auto", "ocr", "llm"}:
            raise ValueError("EXTRACTION_BACKEND must be one of: auto, ocr, llm.")

        if self._llm_extraction_service is None and self.extraction_backend in {"auto", "llm"}:
            if settings.groq_api_key and settings.groq_model_id:
                try:
                    self._llm_extraction_service = LLMExtractionService(settings)
                except Exception:
                    if self.extraction_backend == "llm":
                        raise
                    logger.warning(
                        "LLM extraction could not be initialized; defaulting to OCR parsing",
                        exc_info=True,
                    )
            elif self.extraction_backend == "llm":
                raise ValueError(
                    "LLM extraction is enabled but GROQ_API_KEY or GROQ_MODEL_ID is missing."
                )

    def parse_voter_record(
        self,
        ocr_payload: OCRResult,
        image_path: Path | None = None,
        deleted: bool | None = None,
    ) -> VoterRecord:
        """Parse a voter record using LLM extraction or OCR fallback."""
        if image_path is not None and self._llm_extraction_service is not None:
            try:
                record = self._llm_extraction_service.extract_voter_record(
                    image_path=image_path,
                    deleted=deleted,
                )
                if record.raw_text is None:
                    record.raw_text = self._build_raw_text_from_ocr(ocr_payload)
                return record
            except Exception:
                if self.extraction_backend == "llm":
                    raise
                logger.warning(
                    "LLM extraction failed; falling back to OCR parsing",
                    extra={"image_path": str(image_path)},
                    exc_info=True,
                )

        record = self._parse_voter_record_from_ocr(ocr_payload)
        record.deleted = deleted
        return record

    def _parse_voter_record_from_ocr(self, ocr_payload: OCRResult) -> VoterRecord:
        """Parse OCR output into a structured voter record."""
        logger.info("Parsing OCR payload into voter record")

        ordered_lines = self._sort_lines(ocr_payload.lines)
        line_texts = self._clean_line_texts(ordered_lines)
        row_texts = self._group_lines_into_rows(ordered_lines)
        text_candidates = row_texts or line_texts
        raw_text = "\n".join(text_candidates) if text_candidates else None

        serial_number = self._extract_serial_number(row_texts, line_texts)
        epic_number = self._extract_epic_number(row_texts, line_texts)
        elector_name = self._extract_labeled_value(row_texts, line_texts, self._NAME_PATTERNS)
        relation_type, relation_name = self._extract_relation(row_texts, line_texts)
        house_number = self._extract_labeled_value(row_texts, line_texts, self._HOUSE_PATTERNS)
        age = self._extract_age(row_texts, line_texts)
        gender = self._extract_gender(row_texts, line_texts)

        if elector_name is None:
            elector_name = self._extract_name_fallback(text_candidates, relation_name=relation_name)

        return VoterRecord(
            serial_number=serial_number,
            epic_number=epic_number,
            elector_name=elector_name,
            relation_type=relation_type,
            relation_name=relation_name,
            house_number=house_number,
            age=age,
            gender=gender,
            raw_text=raw_text,
        )

    def _build_raw_text_from_ocr(self, ocr_payload: OCRResult) -> str | None:
        """Build a compact OCR text payload for diagnostics."""
        ordered_lines = self._sort_lines(ocr_payload.lines)
        line_texts = self._clean_line_texts(ordered_lines)
        row_texts = self._group_lines_into_rows(ordered_lines)
        text_candidates = row_texts or line_texts
        if not text_candidates:
            return None
        return json.dumps({"ocr_text": text_candidates}, ensure_ascii=False)

    def _extract_serial_number(self, row_texts: list[str], line_texts: list[str]) -> str | None:
        """Extract the voter serial number from OCR text."""
        for text in row_texts + line_texts:
            normalized = self._normalize_for_matching(text, keep_digits=True)
            label_match = re.search(
                r"(?:serial|क्रमांक|क्रमांक संख्या|मतदाता क्रमांक)\s*[:\-–—]?\s*([A-Z0-9]{1,6})",
                normalized,
            )
            if label_match:
                return label_match.group(1).lstrip("0") or "0"

            q_match = re.search(r"\bQ\s*([0-9]{1,4})\b", normalized)
            if q_match:
                return q_match.group(1).lstrip("0") or "0"

            leading_match = re.match(r"^\D{0,3}([0-9]{1,4})\b", normalized)
            if leading_match:
                return leading_match.group(1).lstrip("0") or "0"
        return None

    def _extract_epic_number(self, row_texts: list[str], line_texts: list[str]) -> str | None:
        """Extract the EPIC number from OCR text."""
        for text in row_texts + line_texts:
            normalized = self._normalize_for_matching(text, keep_digits=True)
            epic_match = re.search(r"\b([A-Z]{3}[0-9]{7})\b", normalized)
            if epic_match:
                return epic_match.group(1)

            label_match = re.search(
                r"(?:epic|photo identity card number|पहचान पत्र संख्या)\s*[:\-–—]?\s*([A-Z0-9?]+)",
                normalized,
            )
            if label_match:
                candidate = self._repair_epic_candidate(label_match.group(1))
                if candidate:
                    return candidate

            for token in re.findall(r"[A-Z0-9?]{8,12}", normalized):
                candidate = self._repair_epic_candidate(token)
                if candidate:
                    return candidate
        return None

    def _extract_relation(self, row_texts: list[str], line_texts: list[str]) -> tuple[str | None, str | None]:
        """Extract relation type and relation name from OCR text."""
        for relation_type, labels in self._RELATION_PATTERNS.items():
            value = self._extract_labeled_value(row_texts, line_texts, labels)
            if value:
                return relation_type, value
        return None, None

    def _extract_age(self, row_texts: list[str], line_texts: list[str]) -> int | None:
        """Extract voter age while being tolerant to OCR noise."""
        for text in row_texts + line_texts:
            normalized = self._normalize_for_matching(text, keep_digits=True)
            label_match = re.search(r"(?:age|उम्र|आयु)\s*[:\-–—]?\s*([0-9]{1,3})", normalized)
            if label_match:
                return self._safe_int(label_match.group(1))

            fallback_match = re.search(r"\b([1-9][0-9])\b", normalized)
            if fallback_match and any(pattern in normalized for pattern in self._AGE_PATTERNS):
                return self._safe_int(fallback_match.group(1))
        return None

    def _extract_gender(self, row_texts: list[str], line_texts: list[str]) -> str | None:
        """Extract and normalize gender from OCR text."""
        gender_aliases = {
            "male": ("male", "पुरुष", "mle", "maie", "पुरष", "पुरूष", "पुंरंष"),
            "female": ("female", "महिला", "मिहला", "femaie", "femaje", "fe male"),
            "other": ("other", "अन्य"),
        }

        for text in row_texts + line_texts:
            normalized = self._normalize_for_matching(text)
            for canonical, aliases in gender_aliases.items():
                if any(alias in normalized for alias in aliases):
                    return canonical
        return None

    def _extract_labeled_value(
        self,
        row_texts: list[str],
        line_texts: list[str],
        labels: tuple[str, ...],
    ) -> str | None:
        """Extract a field value using label-based matching."""
        for texts in (row_texts, line_texts):
            for index, text in enumerate(texts):
                normalized = self._normalize_for_matching(text, keep_digits=True)
                for label in labels:
                    normalized_label = self._normalize_for_matching(label, keep_digits=True)
                    if normalized_label not in normalized:
                        continue

                    value = self._extract_value_after_label(text, label)
                    if value:
                        return value

                    next_line = texts[index + 1] if index + 1 < len(texts) else None
                    if next_line and not self._looks_like_label(next_line):
                        return self._clean_field_value(next_line)
        return None

    def _extract_name_fallback(self, line_texts: list[str], relation_name: str | None) -> str | None:
        """Fallback name extraction for OCR output missing clear labels."""
        banned_tokens = {"MALE", "FEMALE", "OTHER", "उम्र", "लिंग", "गृह", "EPIC", "PHOTO"}
        for text in line_texts:
            candidate = self._clean_field_value(text)
            if not candidate:
                continue
            normalized = self._normalize_for_matching(candidate, keep_digits=True)
            if relation_name and candidate == relation_name:
                continue
            if any(token in normalized for token in banned_tokens):
                continue
            if self._repair_epic_candidate(candidate):
                continue
            if re.search(r"\b\d{1,4}\b", normalized) and len(candidate.split()) <= 2:
                continue
            if self._looks_like_label(candidate):
                continue
            if len(candidate) <= 2:
                continue
            return candidate
        return None

    def _extract_value_after_label(self, text: str, label: str) -> str | None:
        """Extract the substring after a known label."""
        pattern = re.compile(
            rf"{re.escape(label)}{self._SEPARATOR_PATTERN}(.+)$",
            flags=re.IGNORECASE,
        )
        match = pattern.search(text)
        if not match:
            compact_pattern = re.compile(
                rf"{re.escape(label)}\s*(.+)$",
                flags=re.IGNORECASE,
            )
            match = compact_pattern.search(text)
        if not match:
            return None
        return self._clean_field_value(match.group(1))

    def _looks_like_label(self, text: str) -> bool:
        """Detect whether a text line looks like a field label rather than a value."""
        normalized = self._normalize_for_matching(text, keep_digits=True)
        all_labels = (
            list(self._NAME_PATTERNS)
            + list(self._HOUSE_PATTERNS)
            + list(self._AGE_PATTERNS)
            + list(self._GENDER_PATTERNS)
            + [item for values in self._RELATION_PATTERNS.values() for item in values]
        )
        return any(self._normalize_for_matching(label, keep_digits=True) in normalized for label in all_labels)

    def _clean_line_texts(self, lines: Iterable[OCRTextLine]) -> list[str]:
        """Normalize OCR line text once and drop empty results."""
        cleaned_lines: list[str] = []
        for line in lines:
            cleaned = self._clean_text(line.text)
            if cleaned:
                cleaned_lines.append(cleaned)
        return cleaned_lines

    def _group_lines_into_rows(self, lines: list[OCRTextLine]) -> list[str]:
        """Join OCR fragments that belong to the same visual text row."""
        if not lines:
            return []

        grouped_rows: list[list[OCRTextLine]] = []
        current_row: list[OCRTextLine] = []
        current_center_y = 0.0
        current_height = 0.0

        for line in lines:
            if not self._clean_text(line.text):
                continue

            center_y = self._line_center_y(line)
            height = self._line_height(line)
            if not current_row:
                current_row = [line]
                current_center_y = center_y
                current_height = height
                continue

            tolerance = max(current_height * 0.8, height * 0.8, 18.0)
            if abs(center_y - current_center_y) <= tolerance:
                current_row.append(line)
                row_size = len(current_row)
                current_center_y = (current_center_y * (row_size - 1) + center_y) / row_size
                current_height = max(current_height, height)
                continue

            grouped_rows.append(current_row)
            current_row = [line]
            current_center_y = center_y
            current_height = height

        if current_row:
            grouped_rows.append(current_row)

        row_texts: list[str] = []
        for row in grouped_rows:
            ordered_row = sorted(row, key=self._line_min_x)
            text = self._clean_text(" ".join(self._clean_text(item.text) for item in ordered_row))
            if text:
                row_texts.append(text)
        return row_texts

    def _repair_epic_candidate(self, candidate: str) -> str | None:
        """Repair OCR-noisy EPIC tokens into the canonical AAA9999999 pattern."""
        cleaned = re.sub(r"[^A-Za-z0-9?]", "", self._to_ascii_digits(candidate).upper())
        if len(cleaned) < 8:
            return None

        for window_size in range(min(len(cleaned), 12), 9, -1):
            for start in range(0, len(cleaned) - window_size + 1):
                repaired = self._coerce_epic_window(cleaned[start : start + window_size])
                if repaired:
                    return repaired
        return None

    def _coerce_epic_window(self, token: str) -> str | None:
        compact = token.replace("?", "")
        if len(compact) < 10:
            return None
        compact = compact[:10]

        prefix = "".join(self._coerce_epic_prefix_char(char) for char in compact[:3])
        suffix = "".join(self._coerce_epic_digit_char(char) for char in compact[3:])
        if len(prefix) != 3 or len(suffix) != 7:
            return None

        epic_number = prefix + suffix
        if re.fullmatch(r"[A-Z]{3}[0-9]{7}", epic_number):
            return epic_number
        return None

    @staticmethod
    def _coerce_epic_prefix_char(char: str) -> str:
        replacements = {
            "0": "O",
            "1": "I",
            "2": "Z",
            "4": "A",
            "5": "S",
            "6": "G",
            "8": "B",
            "A": "R",
            "H": "R",
        }
        value = replacements.get(char.upper(), char.upper())
        return value if "A" <= value <= "Z" else ""

    @staticmethod
    def _coerce_epic_digit_char(char: str) -> str:
        replacements = {
            "A": "4",
            "B": "8",
            "D": "0",
            "E": "5",
            "G": "6",
            "I": "1",
            "L": "1",
            "O": "0",
            "Q": "0",
            "S": "5",
            "T": "7",
            "U": "0",
            "Z": "2",
        }
        value = replacements.get(char.upper(), char.upper())
        return value if value.isdigit() else ""

    @classmethod
    def _to_ascii_digits(cls, text: str) -> str:
        """Convert Devanagari digits into ASCII digits."""
        return text.translate(cls._DEVANAGARI_DIGITS)

    @staticmethod
    def _sort_lines(lines: list[OCRTextLine]) -> list[OCRTextLine]:
        """Sort OCR lines top-to-bottom then left-to-right."""

        def sort_key(line: OCRTextLine) -> tuple[float, float]:
            if not line.bounding_box:
                return 0.0, 0.0
            min_y = min(point.y for point in line.bounding_box)
            min_x = min(point.x for point in line.bounding_box)
            return min_y, min_x

        return sorted(lines, key=sort_key)

    @staticmethod
    def _line_center_y(line: OCRTextLine) -> float:
        if not line.bounding_box:
            return 0.0
        values = [point.y for point in line.bounding_box]
        return (min(values) + max(values)) / 2

    @staticmethod
    def _line_height(line: OCRTextLine) -> float:
        if not line.bounding_box:
            return 0.0
        values = [point.y for point in line.bounding_box]
        return max(values) - min(values)

    @staticmethod
    def _line_min_x(line: OCRTextLine) -> float:
        if not line.bounding_box:
            return 0.0
        return min(point.x for point in line.bounding_box)

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalize OCR text spacing and punctuation noise."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = cleaned.replace("|", "I")
        return cleaned

    @staticmethod
    def _normalize_for_matching(text: str, keep_digits: bool = False) -> str:
        """Normalize text for resilient label and regex matching."""
        normalized = re.sub(r"\s+", " ", text).strip()
        normalized = normalized.translate(ExtractionService._DEVANAGARI_DIGITS).upper()
        if not keep_digits:
            normalized = normalized.replace("0", "O")
        return normalized.strip()

    @staticmethod
    def _clean_field_value(value: str) -> str | None:
        """Normalize extracted field values."""
        cleaned = re.sub(r"\s+", " ", value).strip(" :-")
        cleaned = cleaned.strip(" ?!.,;/")
        cleaned = re.sub(
            r"\b(?:name|age|gender|sex|house no|house number|elector name|photo identity card number|epic)\b\s*[:\-–—]?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return cleaned or None

    @staticmethod
    def _safe_int(value: str) -> int | None:
        """Convert a string to integer safely."""
        try:
            return int(value)
        except ValueError:
            return None
