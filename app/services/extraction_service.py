"""Service implementation for OCR-to-JSON voter record parsing."""

import re
from typing import Iterable

from app.models.image import OCRResult, OCRTextLine
from app.models.voter import VoterRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)


class ExtractionService:
    """Transform raw OCR output into structured voter records."""

    _SEPARATOR_PATTERN = r"(?:\s*[:\-–—]\s*|\s+)"
    _RELATION_PATTERNS: dict[str, tuple[str, ...]] = {
        "father": (
            "father",
            "fathers name",
            "father name",
            "पिता का नाम",
            "पिता का नाभ",
            "पिता का नाम :",
            "पिता का नाम-",
        ),
        "husband": (
            "husband",
            "husbands name",
            "husband name",
            "पति का नाम",
            "पति का नाभ",
            "पति का नाम :",
            "पति का नाम-",
        ),
    }
    _NAME_PATTERNS: tuple[str, ...] = (
        "elector name",
        "निर्वाचक का नाम",
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
    )

    def parse_voter_record(self, ocr_payload: OCRResult) -> VoterRecord:
        """Parse OCR output into a structured voter record."""
        logger.info("Parsing OCR payload into voter record")

        ordered_lines = self._sort_lines(ocr_payload.lines)
        line_texts = self._clean_line_texts(ordered_lines)
        raw_text = "\n".join(line_texts) if line_texts else None

        serial_number = self._extract_serial_number(line_texts)
        epic_number = self._extract_epic_number(line_texts)
        elector_name = self._extract_labeled_value(line_texts, self._NAME_PATTERNS)
        relation_type, relation_name = self._extract_relation(line_texts)
        house_number = self._extract_labeled_value(line_texts, self._HOUSE_PATTERNS)
        age = self._extract_age(line_texts)
        gender = self._extract_gender(line_texts)

        if elector_name is None:
            elector_name = self._extract_name_fallback(line_texts, relation_name=relation_name)

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

    def _extract_serial_number(self, line_texts: list[str]) -> str | None:
        """Extract the voter serial number from OCR text."""
        for text in line_texts:
            normalized = self._normalize_for_matching(text, keep_digits=True)
            label_match = re.search(
                r"(?:serial|क्रमांक|क्रमांक संख्या|मतदाता क्रमांक)\s*[:\-–—]?\s*([A-Z0-9]{1,6})",
                normalized,
            )
            if label_match:
                return label_match.group(1)

            q_match = re.search(r"\bQ\s*([0-9]{1,4})\b", normalized)
            if q_match:
                return q_match.group(1)

            leading_match = re.match(r"^\D{0,3}([0-9]{1,4})\b", normalized)
            if leading_match:
                return leading_match.group(1)
        return None

    def _extract_epic_number(self, line_texts: list[str]) -> str | None:
        """Extract the EPIC number from OCR text."""
        for text in line_texts:
            normalized = self._normalize_for_matching(text, keep_digits=True)
            epic_match = re.search(r"\b([A-Z]{3}[0-9]{7})\b", normalized)
            if epic_match:
                return epic_match.group(1)

            label_match = re.search(
                r"(?:epic|photo identity card number|पहचान पत्र संख्या)\s*[:\-–—]?\s*([A-Z0-9]+)",
                normalized,
            )
            if label_match:
                return label_match.group(1)
        return None

    def _extract_relation(self, line_texts: list[str]) -> tuple[str | None, str | None]:
        """Extract relation type and relation name from OCR text."""
        for relation_type, labels in self._RELATION_PATTERNS.items():
            value = self._extract_labeled_value(line_texts, labels)
            if value:
                return relation_type, value
        return None, None

    def _extract_age(self, line_texts: list[str]) -> int | None:
        """Extract voter age while being tolerant to OCR noise."""
        for text in line_texts:
            normalized = self._normalize_for_matching(text, keep_digits=True)
            label_match = re.search(r"(?:age|उम्र|आयु)\s*[:\-–—]?\s*([0-9]{1,3})", normalized)
            if label_match:
                return self._safe_int(label_match.group(1))

            fallback_match = re.search(r"\b([1-9][0-9])\b", normalized)
            if fallback_match and any(pattern in normalized for pattern in self._AGE_PATTERNS):
                return self._safe_int(fallback_match.group(1))
        return None

    def _extract_gender(self, line_texts: list[str]) -> str | None:
        """Extract and normalize gender from OCR text."""
        gender_aliases = {
            "male": ("male", "पुरुष", "mle", "maie"),
            "female": ("female", "महिला", "femaie", "femaje", "fe male"),
            "other": ("other", "अन्य"),
        }

        for text in line_texts:
            normalized = self._normalize_for_matching(text)
            for canonical, aliases in gender_aliases.items():
                if any(alias in normalized for alias in aliases):
                    return canonical
        return None

    def _extract_labeled_value(self, line_texts: list[str], labels: tuple[str, ...]) -> str | None:
        """Extract a field value using label-based matching."""
        for index, text in enumerate(line_texts):
            normalized = self._normalize_for_matching(text, keep_digits=True)
            for label in labels:
                normalized_label = self._normalize_for_matching(label, keep_digits=True)
                if normalized_label not in normalized:
                    continue

                value = self._extract_value_after_label(text, label)
                if value:
                    return value

                next_line = line_texts[index + 1] if index + 1 < len(line_texts) else None
                if next_line and not self._looks_like_label(next_line):
                    return self._clean_field_value(next_line)
        return None

    def _extract_name_fallback(self, line_texts: list[str], relation_name: str | None) -> str | None:
        """Fallback name extraction for OCR output missing clear labels."""
        banned_tokens = {"male", "female", "other"}
        for text in line_texts:
            candidate = self._clean_field_value(text)
            if not candidate:
                continue
            normalized = self._normalize_for_matching(candidate, keep_digits=True)
            if relation_name and candidate == relation_name:
                continue
            if any(token in normalized for token in banned_tokens):
                continue
            if re.search(r"\b[A-Z]{3}[0-9]{7}\b", normalized):
                continue
            if re.search(r"\b\d{1,4}\b", normalized) and len(candidate.split()) <= 2:
                continue
            if self._looks_like_label(candidate):
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
    def _clean_text(text: str) -> str:
        """Normalize OCR text spacing and punctuation noise."""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = cleaned.replace("|", "I")
        return cleaned

    @staticmethod
    def _normalize_for_matching(text: str, keep_digits: bool = False) -> str:
        """Normalize text for resilient label and regex matching."""
        normalized = re.sub(r"\s+", " ", text).strip().upper()
        if not keep_digits:
            normalized = normalized.replace("0", "O")
        return normalized.strip()

    @staticmethod
    def _clean_field_value(value: str) -> str | None:
        """Normalize extracted field values."""
        cleaned = re.sub(r"\s+", " ", value).strip(" :-")
        cleaned = re.sub(
            r"\b(?:name|age|gender|sex|house no|house number|elector name)\b\s*[:\-–—]?",
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
