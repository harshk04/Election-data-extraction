"""LLM-backed voter record extraction from cropped entry images."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import time
from pathlib import Path

from app.config.settings import Settings
from app.models.voter import VoterRecord
from app.utils.logging import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """
You are extracting structured data from a single cropped Indian voter-card image.

Rules:
1. Read only what is visible in the image.
2. Never translate Hindi text.
3. Never infer or guess missing values.
4. Return exactly what is written.
5. Return only valid JSON.
6. If a field is unreadable, return null.
7. serial_number must preserve prefixes such as Q exactly.
8. epic_number must preserve letters and numbers exactly without correction.
9. elector_name and relation_name must stay exactly in Hindi if shown in Hindi.
10. relation_type must be exactly "father", "husband", or null.
11. house_number must be returned exactly as written, including values like 8, 50 ए, 2/F1, 03.
12. age must be an integer or null.
13. gender must be exactly "male", "female", or null.
14. confidence must be an integer from 0 to 100 representing extraction confidence.
15. Do not include labels like "निर्वाचक का नाम", "पति का नाम", "पिता का नाम", "गृह", "उम्र", or "लिंग" inside field values.
16. elector_name must contain only the voter's name, not the full labeled line.
17. relation_name must contain only the related person's name, not the full labeled line.
18. If the top-left area has two small number boxes, serial_number must preserve both exactly as shown. For example, return 642 1 if the image shows 642 in the first box and 1 in the second box.

Return JSON with exactly these keys:
{
  "serial_number": string | null,
  "epic_number": string | null,
  "elector_name": string | null,
  "relation_type": "father" | "husband" | null,
  "relation_name": string | null,
  "house_number": string | null,
  "age": integer | null,
  "gender": "male" | "female" | null,
  "confidence": integer
}
""".strip()

SERIAL_NUMBER_PROMPT_TEMPLATE = """
You are extracting only the serial number from a single cropped Indian voter-card image.

Rules:
1. Focus on the top-left serial-number boxes only.
2. Return only valid JSON.
3. Never guess missing digits or text.
4. If the serial number is unreadable, return null.
5. If the top-left area has two small number boxes, preserve both exactly as shown in one string separated by a single space.
6. Preserve prefixes such as Q exactly.

Return JSON with exactly these keys:
{
  "serial_number": string | null,
  "confidence": integer
}
""".strip()


class LLMExtractionService:
    """Use an OpenAI-compatible multimodal model to extract structured fields from entry crops."""

    _LABEL_FRAGMENTS = (
        "निर्वाचक",
        "निवाचक",
        "मतदाता",
        "elector",
        "name",
        "पिता",
        "पति",
        "गृह",
        "house",
        "उम्र",
        "आयु",
        "लिंग",
        "gender",
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = self._build_client()

    def extract_voter_record(self, image_path: Path, deleted: bool | None = None) -> VoterRecord:
        """Extract a voter record directly from a crop image."""
        last_error: Exception | None = None
        total_attempts = max(
            self._settings.groq_quality_retries,
            self._settings.groq_max_retries,
            1,
        )
        response_text: str | None = None
        model_used: str | None = None

        for attempt in range(1, total_attempts + 1):
            try:
                payload, response_text, model_used = self._extract_best_payload(image_path)
                quality_issue = self._get_quality_issue(payload)
                if quality_issue is not None:
                    raise ValueError(f"Low-quality extraction: {quality_issue}")
                break
            except Exception as error:
                last_error = error
                retryable = self._is_retryable_exception(error)
                retry_delay_seconds = self._get_retry_delay_seconds(error, attempt)
                logger.warning(
                    "LLM extraction validation failed",
                    extra={
                        "image_path": str(image_path),
                        "attempt": attempt,
                        "max_attempts": total_attempts,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "retryable": retryable,
                        "retry_delay_seconds": retry_delay_seconds,
                        "model": model_used,
                        "response_preview": self._preview_text(response_text),
                    },
                )
                if attempt >= total_attempts:
                    raise RuntimeError(
                        f"LLM extraction failed for {image_path.name}: {error}"
                    ) from error
                logger.warning(
                    "LLM extraction failed; retrying crop",
                    extra={
                        "image_path": str(image_path),
                        "attempt": attempt,
                        "max_attempts": total_attempts,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "retryable": retryable,
                        "retry_delay_seconds": retry_delay_seconds,
                        "model": model_used,
                    },
                )
                time.sleep(retry_delay_seconds)

        if last_error is not None and 'payload' not in locals():
            raise RuntimeError(f"LLM extraction failed for {image_path.name}: {last_error}") from last_error

        return VoterRecord(
            serial_number=self._normalize_serial_number(payload.get("serial_number")),
            epic_number=self._clean_optional_string(payload.get("epic_number")),
            elector_name=self._clean_optional_string(payload.get("elector_name")),
            relation_type=self._normalize_relation_type(payload.get("relation_type")),
            relation_name=self._clean_optional_string(payload.get("relation_name")),
            house_number=self._clean_optional_string(payload.get("house_number")),
            age=self._normalize_age(payload.get("age")),
            gender=self._normalize_gender(payload.get("gender")),
            deleted=deleted,
            raw_text=json.dumps(payload, ensure_ascii=False),
        )

    def _extract_best_payload(self, image_path: Path) -> tuple[dict[str, object], str, str]:
        """Run primary extraction and selectively escalate when quality is weak."""
        primary_model = self._settings.openai_bedrock_model_id
        if primary_model is None:
            raise ValueError("OPENAI_BEDROCK_MODEL_ID is required for LLM extraction")

        primary_response = self._invoke_model(
            image_path=image_path,
            model_id=primary_model,
            prompt_text=PROMPT_TEMPLATE,
        )
        primary_payload = self._parse_json_payload(primary_response)
        repaired_primary_payload = self._recover_missing_serial_number(
            image_path=image_path,
            payload=primary_payload,
            model_id=primary_model,
        )
        primary_quality = self._get_quality_issue(repaired_primary_payload)
        if primary_quality is None:
            return repaired_primary_payload, json.dumps(repaired_primary_payload, ensure_ascii=False), primary_model

        fallback_model = self._settings.openai_bedrock_fallback_model_id
        if fallback_model and fallback_model != primary_model:
            fallback_response = self._invoke_model(
                image_path=image_path,
                model_id=fallback_model,
                prompt_text=PROMPT_TEMPLATE,
            )
            fallback_payload = self._parse_json_payload(fallback_response)
            repaired_fallback_payload = self._recover_missing_serial_number(
                image_path=image_path,
                payload=fallback_payload,
                model_id=fallback_model,
            )
            fallback_quality = self._get_quality_issue(repaired_fallback_payload)
            if fallback_quality is None:
                logger.info(
                    "LLM extraction succeeded with fallback model",
                    extra={"image_path": str(image_path), "fallback_model": fallback_model},
                )
                return (
                    repaired_fallback_payload,
                    json.dumps(repaired_fallback_payload, ensure_ascii=False),
                    fallback_model,
                )

            better_payload, better_model = self._choose_better_payload(
                primary_payload=repaired_primary_payload,
                fallback_payload=repaired_fallback_payload,
                primary_model=primary_model,
                fallback_model=fallback_model,
            )
            return better_payload, json.dumps(better_payload, ensure_ascii=False), better_model

        return repaired_primary_payload, json.dumps(repaired_primary_payload, ensure_ascii=False), primary_model

    def _build_client(self):  # type: ignore[no-untyped-def]
        if not self._settings.openai_bedrock_api_key or not self._settings.openai_bedrock_model_id:
            raise ValueError(
                "OPENAI_BEDROCK_API_KEY and OPENAI_BEDROCK_MODEL_ID are required for LLM extraction"
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError(
                "The 'openai' package is required for LLM extraction. Install project dependencies first."
            ) from error

        return OpenAI(
            api_key=self._settings.openai_bedrock_api_key,
            base_url=self._normalize_base_url(self._settings.openai_bedrock_base_url),
            default_headers={"OpenAI-Project": self._settings.openai_bedrock_project},
            timeout=self._settings.groq_request_timeout_seconds,
            max_retries=0,
        )

    def _invoke_model(self, image_path: Path, model_id: str, prompt_text: str) -> str:
        image_url = self._build_data_url(image_path)
        last_error: Exception | None = None

        for attempt in range(1, self._settings.groq_max_retries + 1):
            try:
                started_at = time.perf_counter()
                response = self._client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise information extraction engine. Return JSON only.",
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_text},
                                {"type": "image_url", "image_url": {"url": image_url}},
                            ],
                        },
                    ],
                    temperature=self._settings.groq_temperature,
                    max_tokens=self._settings.groq_max_tokens,
                )
                logger.info(
                    "LLM extraction completed",
                    extra={
                        "image_path": str(image_path),
                        "model": model_id,
                        "latency_seconds": round(time.perf_counter() - started_at, 3),
                    },
                )
                return self._extract_response_text(response)
            except Exception as error:  # noqa: BLE001
                last_error = error
                retryable = self._is_retryable_exception(error)
                retry_delay_seconds = self._get_retry_delay_seconds(error, attempt)
                logger.warning(
                    "LLM extraction attempt failed",
                    extra={
                        "image_path": str(image_path),
                        "attempt": attempt,
                        "max_retries": self._settings.groq_max_retries,
                        "error": str(error),
                        "model": model_id,
                        "retryable": retryable,
                        "retry_delay_seconds": retry_delay_seconds,
                    },
                )
                if not retryable or attempt >= self._settings.groq_max_retries:
                    break
                time.sleep(retry_delay_seconds)

        assert last_error is not None
        raise RuntimeError(f"LLM invocation failed for {image_path.name}: {last_error}") from last_error

    def _build_data_url(self, image_path: Path) -> str:
        mime_type, _ = mimetypes.guess_type(str(image_path))
        resolved_mime_type = mime_type or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{resolved_mime_type};base64,{encoded}"

    @staticmethod
    def _normalize_base_url(base_url: str) -> str:
        """Normalize base URL so the client does not duplicate API path segments."""
        normalized = base_url.strip().rstrip("/")
        if normalized.endswith("/v1"):
            return normalized
        return f"{normalized}/v1" if normalized else "https://bedrock-mantle.ap-south-1.api.aws/v1"

    @staticmethod
    def _is_retryable_exception(error: Exception) -> bool:
        """Return whether the request is worth retrying."""
        message = str(error).lower()
        retryable_markers = (
            "429",
            "rate limit",
            "rate_limit",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection",
            "server error",
            "service unavailable",
            "too many requests",
        )
        return any(marker in message for marker in retryable_markers)

    @staticmethod
    def _get_retry_delay_seconds(error: Exception, attempt: int) -> float:
        """Choose a retry delay, preferring provider hints when available."""
        message = str(error)
        match = re.search(r"try again in\s*([0-9]+(?:\.[0-9]+)?)s", message, re.IGNORECASE)
        if match:
            return max(float(match.group(1)) + 0.5, 1.0)

        backoff_seconds = min(2 ** max(attempt - 1, 0), 30)
        return float(backoff_seconds)

    def _extract_response_text(self, response: object) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise ValueError("Model response did not contain any choices")

        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(str(item.get("text", "")))
                else:
                    text_value = getattr(item, "text", None)
                    if text_value:
                        text_parts.append(str(text_value))
            if text_parts:
                return "\n".join(text_parts)

        raise ValueError("Model response did not contain text content")

    @staticmethod
    def _preview_text(raw_text: str | None, limit: int = 300) -> str | None:
        if raw_text is None:
            return None
        compact = " ".join(raw_text.split())
        if len(compact) <= limit:
            return compact
        return f"{compact[:limit]}..."

    def _parse_json_payload(self, raw_text: str) -> dict[str, object]:
        json_text = self._extract_json_object(raw_text)
        payload = json.loads(json_text)
        if not isinstance(payload, dict):
            raise ValueError("Model response JSON must be an object")
        return payload

    def _get_quality_issue(self, payload: dict[str, object]) -> str | None:
        """Return a reason when the extraction is too weak to trust on the first pass."""
        serial_number = self._normalize_serial_number(payload.get("serial_number"))
        elector_name = self._clean_optional_string(payload.get("elector_name"))
        relation_name = self._clean_optional_string(payload.get("relation_name"))
        house_number = self._clean_optional_string(payload.get("house_number"))
        epic_number = self._clean_optional_string(payload.get("epic_number"))
        age = payload.get("age")
        gender = self._clean_optional_string(payload.get("gender"))

        populated_secondary_fields = sum(
            value is not None and value != ""
            for value in (epic_number, relation_name, house_number, gender)
        )
        if age is not None:
            populated_secondary_fields += 1

        if serial_number is None and populated_secondary_fields < 3:
            return "serial_number is missing"
        if elector_name is None:
            return "elector_name is missing"
        if self._looks_like_labeled_line(elector_name):
            return "elector_name still contains labels or non-name text"
        if relation_name is not None and self._looks_like_labeled_line(relation_name):
            return "relation_name still contains labels or non-name text"

        if populated_secondary_fields == 0:
            return "all secondary fields are missing"
        if epic_number is None and relation_name is None and house_number is None and age is None and gender is None:
            return "epic, relation, house, age, and gender are all missing"

        return None

    def _recover_missing_serial_number(
        self,
        image_path: Path,
        payload: dict[str, object],
        model_id: str,
    ) -> dict[str, object]:
        """Run a focused serial-number pass when the main extraction missed it."""
        if self._normalize_serial_number(payload.get("serial_number")) is not None:
            return payload

        serial_response = self._invoke_model(
            image_path=image_path,
            model_id=model_id,
            prompt_text=SERIAL_NUMBER_PROMPT_TEMPLATE,
        )
        serial_payload = self._parse_json_payload(serial_response)
        serial_number = self._normalize_serial_number(serial_payload.get("serial_number"))
        if serial_number is None:
            return payload

        merged_payload = dict(payload)
        merged_payload["serial_number"] = serial_number
        logger.info(
            "Recovered serial number with targeted extraction",
            extra={"image_path": str(image_path), "model": model_id, "serial_number": serial_number},
        )
        return merged_payload

    def _choose_better_payload(
        self,
        primary_payload: dict[str, object],
        fallback_payload: dict[str, object],
        primary_model: str,
        fallback_model: str,
    ) -> tuple[dict[str, object], str]:
        """Prefer the payload with more populated useful fields when both are imperfect."""
        primary_score = self._payload_completeness_score(primary_payload)
        fallback_score = self._payload_completeness_score(fallback_payload)

        if fallback_score > primary_score:
            return fallback_payload, fallback_model
        return primary_payload, primary_model

    def _payload_completeness_score(self, payload: dict[str, object]) -> int:
        fields = (
            self._normalize_serial_number(payload.get("serial_number")),
            self._clean_optional_string(payload.get("epic_number")),
            self._clean_optional_string(payload.get("elector_name")),
            self._clean_optional_string(payload.get("relation_name")),
            self._clean_optional_string(payload.get("house_number")),
            self._clean_optional_string(payload.get("gender")),
        )
        score = sum(value is not None and value != "" for value in fields)
        if payload.get("age") is not None:
            score += 1
        return score

    def _looks_like_labeled_line(self, value: str) -> bool:
        """Detect values that still include field labels instead of just the extracted name."""
        lowered = value.strip().lower()
        if ":" in lowered:
            return True
        if len(lowered.split()) >= 4 and any(fragment in lowered for fragment in self._LABEL_FRAGMENTS):
            return True
        return any(fragment in lowered for fragment in ("निर्वाचक का", "निवाचक का", "पिता का", "पति का"))

    @staticmethod
    def _extract_json_object(raw_text: str) -> str:
        stripped = raw_text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        markdown_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if markdown_match:
            return markdown_match.group(1)

        json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if json_match:
            return json_match.group(1)

        raise ValueError("No JSON object found in model response")

    @staticmethod
    def _clean_optional_string(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @classmethod
    def _normalize_serial_number(cls, value: object) -> str | None:
        return cls._clean_optional_string(value)

    @staticmethod
    def _normalize_age(value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        raise ValueError(f"Invalid age value returned by LLM: {value!r}")

    @staticmethod
    def _normalize_gender(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"male", "female"}:
            return normalized
        if normalized in {"पुरुष", "male"}:
            return "male"
        if normalized in {"महिला", "female"}:
            return "female"
        if normalized == "":
            return None
        raise ValueError(f"Invalid gender value returned by LLM: {value!r}")

    @staticmethod
    def _normalize_relation_type(value: object) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized in {"father", "husband"}:
            return normalized
        if normalized == "":
            return None
        raise ValueError(f"Invalid relation_type returned by LLM: {value!r}")
