"""Service implementation for deleted-entry detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config.settings import get_settings
from app.models.image import DeletedEntryDetectionResult, OCRResult
from app.services.ocr_service import OCRService
from app.utils.file_utils import ensure_directory, load_image, save_image
from app.utils.logging import get_logger, log_exception

logger = get_logger(__name__)


@dataclass
class SerialStageResult:
    """OCR result for the fixed serial-number box."""

    image: np.ndarray
    text_lines: list[str]
    raw_text: str
    level: str
    score: float
    q_detected: bool
    note: str
    left_text: str
    right_text: str
    visual_q_detected: bool


@dataclass
class WatermarkGeometryResult:
    """Geometry result for the diagonal watermark band."""

    mask: np.ndarray
    largest_component_visualization: np.ndarray
    rotated_band: np.ndarray
    angle: float | None
    box: tuple[int, int, int, int] | None
    level: str
    score: float
    component_count: int
    crosses_center: bool
    width_ratio: float
    height_ratio: float
    mean_intensity: float
    projected_polygon: list[list[int]]


@dataclass
class WatermarkOCRResult:
    """OCR result for the isolated watermark band."""

    text_lines: list[str]
    raw_text: str
    best_match: str
    similarity: float
    level: str
    score: float
    note: str


@dataclass
class DecisionResult:
    """Final rule-based deleted-entry decision."""

    deleted: bool
    confidence: float
    rule: str
    reason: str


class DeletedEntryDetectionService:
    """Detect deleted voter entries using serial-box OCR and watermark verification."""

    _SERIAL_BOX = (0.015, 0.02, 0.34, 0.18)
    _PHOTO_BOX = (0.74, 0.17, 0.97, 0.93)
    _WATERMARK_ANGLES = (-45.0, -40.0, -35.0, -30.0, -25.0, 25.0, 30.0, 35.0, 40.0, 45.0)

    def __init__(
        self,
        ocr_service: OCRService | None = None,
        debug_enabled: bool | None = None,
        debug_output_dir: Path | None = None,
        confidence_threshold: float | None = None,
        watermark_min_score: float | None = None,
    ) -> None:
        del debug_enabled, confidence_threshold, watermark_min_score
        settings = get_settings()
        self.ocr_service = ocr_service or OCRService()
        self.debug_output_dir = debug_output_dir or settings.deleted_debug_dir

    def detect_deleted_entry(
        self,
        image: np.ndarray,
        image_name: str = "entry",
    ) -> DeletedEntryDetectionResult:
        """Classify whether an entry crop is deleted."""
        logger.info("Detecting deleted entry", extra={"image_name": image_name})

        serial_stage = self._run_serial_stage(image)
        geometry_stage = self._run_watermark_geometry_stage(image)
        watermark_ocr_stage = self._run_watermark_ocr_stage(geometry_stage)
        decision = self._decide_deleted(serial_stage, geometry_stage, watermark_ocr_stage)
        debug_image_path = self._save_debug_artifacts(
            image_name=image_name,
            image=image,
            serial_stage=serial_stage,
            geometry_stage=geometry_stage,
            watermark_ocr_stage=watermark_ocr_stage,
            decision=decision,
        )

        ocr_confidence = max(serial_stage.score, watermark_ocr_stage.score)
        return DeletedEntryDetectionResult(
            deleted=decision.deleted,
            confidence=round(decision.confidence, 4),
            ocr_confidence=round(ocr_confidence, 4),
            watermark_confidence=round(geometry_stage.score, 4),
            debug_image_path=debug_image_path,
        )

    def detect_deleted_entry_from_path(self, image_path: Path) -> DeletedEntryDetectionResult:
        """Load a crop image from disk and run deleted-entry detection."""
        image = load_image(image_path)
        return self.detect_deleted_entry(image=image, image_name=image_path.stem)

    def _run_serial_stage(self, image: np.ndarray) -> SerialStageResult:
        """Run OCR only on the fixed serial-number region."""
        serial_region = self._extract_layout_region(image, self._SERIAL_BOX)
        text_lines = self._ocr_lines_or_empty(serial_region, "Serial-region OCR failed during deleted detection")
        raw_text = "\n".join(text_lines)
        compact_lines = [self._normalize_text(line) for line in text_lines if line.strip()]

        height, width = serial_region.shape[:2]
        left_region = serial_region[:, : max(1, int(width * 0.34))]
        right_region = serial_region[:, int(width * 0.55) :]
        left_lines = self._ocr_lines_or_empty(left_region, "Left serial OCR failed during deleted detection")
        right_lines = self._ocr_lines_or_empty(right_region, "Right serial OCR failed during deleted detection")
        normalized_left = [self._normalize_text(line) for line in left_lines if line.strip()]
        normalized_right = [self._normalize_text(line) for line in right_lines if line.strip()]
        visual_q_detected = self._detect_visual_q_marker(left_region)

        has_q_number = any(re.fullmatch(r"Q\d{1,4}", line) for line in compact_lines)
        has_q_only = any(line == "Q" for line in compact_lines) or any(line == "Q" for line in normalized_left)
        right_has_number = any(re.fullmatch(r"\d{1,4}", self._map_digits_to_ascii(line)) for line in normalized_right)
        compact_has_number = any(re.fullmatch(r"\d{1,4}", self._map_digits_to_ascii(line)) for line in compact_lines)
        left_has_q_like = any(self._is_q_like_token(line) for line in normalized_left)
        split_q_number = (left_has_q_like or visual_q_detected) and right_has_number
        inline_q_number = has_q_only and compact_has_number

        if has_q_number or split_q_number or inline_q_number:
            return SerialStageResult(
                image=serial_region,
                text_lines=text_lines + left_lines + right_lines,
                raw_text=raw_text,
                level="strong",
                score=0.95,
                q_detected=True,
                note="serial_q_detected",
                left_text="\n".join(left_lines),
                right_text="\n".join(right_lines),
                visual_q_detected=visual_q_detected,
            )

        if has_q_only or left_has_q_like or visual_q_detected:
            return SerialStageResult(
                image=serial_region,
                text_lines=text_lines + left_lines + right_lines,
                raw_text=raw_text,
                level="medium",
                score=0.65,
                q_detected=True,
                note="serial_q_only_detected",
                left_text="\n".join(left_lines),
                right_text="\n".join(right_lines),
                visual_q_detected=visual_q_detected,
            )

        return SerialStageResult(
            image=serial_region,
            text_lines=text_lines + left_lines + right_lines,
            raw_text=raw_text,
            level="none",
            score=0.0,
            q_detected=False,
            note="serial_normal",
            left_text="\n".join(left_lines),
            right_text="\n".join(right_lines),
            visual_q_detected=visual_q_detected,
        )

    def _run_watermark_geometry_stage(self, image: np.ndarray) -> WatermarkGeometryResult:
        """Detect a large diagonal watermark band by geometry, not by OCR."""
        grayscale = self._to_grayscale(image)
        prepared = self._prepare_for_watermark_detection(grayscale)

        best_result = WatermarkGeometryResult(
            mask=np.zeros_like(grayscale),
            largest_component_visualization=cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR),
            rotated_band=np.full((32, 32), 255, dtype=np.uint8),
            angle=None,
            box=None,
            level="none",
            score=0.0,
            component_count=0,
            crosses_center=False,
            width_ratio=0.0,
            height_ratio=0.0,
            mean_intensity=255.0,
            projected_polygon=[],
        )

        for angle in self._WATERMARK_ANGLES:
            rotated_gray = self._rotate_image(prepared, angle, border_value=255)
            letter_mask, watermark_mask = self._build_watermark_candidate_masks(rotated_gray)
            candidate = self._select_watermark_component(rotated_gray, letter_mask, watermark_mask, angle)
            if candidate.score > best_result.score:
                best_result = candidate

        return best_result

    def _run_watermark_ocr_stage(self, geometry_stage: WatermarkGeometryResult) -> WatermarkOCRResult:
        """Run OCR only on the rotated watermark band."""
        if geometry_stage.level == "none" or geometry_stage.box is None:
            return WatermarkOCRResult(
                text_lines=[],
                raw_text="",
                best_match="",
                similarity=0.0,
                level="none",
                score=0.0,
                note="no_watermark_band",
            )

        band_image = geometry_stage.rotated_band
        try:
            ocr_result = self.ocr_service.run_ocr_on_image(band_image)
        except RuntimeError:
            logger.warning("OCR engine unavailable for watermark-band deleted detection")
            return WatermarkOCRResult(
                text_lines=[],
                raw_text="",
                best_match="",
                similarity=0.0,
                level="none",
                score=0.0,
                note="watermark_ocr_unavailable",
            )
        except Exception:
            log_exception(logger, "Watermark-band OCR failed during deleted detection")
            return WatermarkOCRResult(
                text_lines=[],
                raw_text="",
                best_match="",
                similarity=0.0,
                level="none",
                score=0.0,
                note="watermark_ocr_failed",
            )

        text_lines = self._extract_ocr_lines(ocr_result)
        raw_text = "\n".join(text_lines)
        normalized_candidates = [self._normalize_text(line) for line in text_lines if line.strip()]
        normalized_candidates.extend(
            token
            for line in normalized_candidates
            for token in re.findall(r"[A-Z]+", line)
        )

        best_match = ""
        best_similarity = 0.0
        for candidate in normalized_candidates:
            similarity = SequenceMatcher(a=candidate, b="DELETED").ratio()
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate

        if "DELETED" in normalized_candidates or best_similarity >= 0.82:
            return WatermarkOCRResult(
                text_lines=text_lines,
                raw_text=raw_text,
                best_match=best_match,
                similarity=best_similarity,
                level="strong",
                score=0.95,
                note="watermark_deleted_text_detected",
            )

        if best_similarity >= 0.65:
            return WatermarkOCRResult(
                text_lines=text_lines,
                raw_text=raw_text,
                best_match=best_match,
                similarity=best_similarity,
                level="medium",
                score=0.7,
                note="watermark_deleted_text_fuzzy_match",
            )

        return WatermarkOCRResult(
            text_lines=text_lines,
            raw_text=raw_text,
            best_match=best_match,
            similarity=best_similarity,
            level="none",
            score=0.0,
            note="watermark_text_not_detected",
        )

    def _decide_deleted(
        self,
        serial_stage: SerialStageResult,
        geometry_stage: WatermarkGeometryResult,
        watermark_ocr_stage: WatermarkOCRResult,
    ) -> DecisionResult:
        """Apply explicit multi-stage decision rules instead of score averaging."""
        geometry_medium_or_strong = geometry_stage.level in {"medium", "strong"}
        watermark_medium_or_strong = watermark_ocr_stage.level in {"medium", "strong"}

        if serial_stage.q_detected and geometry_medium_or_strong:
            return DecisionResult(
                deleted=True,
                confidence=0.96 if serial_stage.level == "strong" else 0.9,
                rule="rule_1_q_and_geometry",
                reason="Serial box contains a Q-number and a large diagonal watermark band crosses the card center.",
            )

        if watermark_medium_or_strong and geometry_medium_or_strong:
            return DecisionResult(
                deleted=True,
                confidence=0.97 if watermark_ocr_stage.level == "strong" else 0.9,
                rule="rule_2_watermark_ocr_and_geometry",
                reason="Watermark-band OCR matches DELETED and the diagonal band geometry is valid.",
            )

        if geometry_stage.level == "strong" and (
            watermark_ocr_stage.level == "medium" or serial_stage.level == "medium"
        ):
            return DecisionResult(
                deleted=True,
                confidence=0.9,
                rule="rule_3_strong_geometry_plus_medium_ocr",
                reason="A very strong diagonal watermark band is present and OCR provides medium-strength deleted evidence.",
            )

        if serial_stage.q_detected and geometry_stage.level == "none":
            return DecisionResult(
                deleted=False,
                confidence=0.3,
                rule="rule_5_q_without_watermark",
                reason="Serial OCR saw a Q-like prefix but no diagonal watermark band was verified.",
            )

        if geometry_stage.level != "none" and watermark_ocr_stage.level == "none" and not serial_stage.q_detected:
            return DecisionResult(
                deleted=False,
                confidence=0.2,
                rule="rule_4_geometry_without_ocr",
                reason="Gray blobs were found, but they were not supported by serial OCR or watermark OCR.",
            )

        return DecisionResult(
            deleted=False,
            confidence=0.0,
            rule="normal",
            reason="No valid deleted-entry evidence chain was found.",
        )

    def _save_debug_artifacts(
        self,
        image_name: str,
        image: np.ndarray,
        serial_stage: SerialStageResult,
        geometry_stage: WatermarkGeometryResult,
        watermark_ocr_stage: WatermarkOCRResult,
        decision: DecisionResult,
    ) -> Path:
        """Persist all mandatory debug artifacts for a crop."""
        output_dir = ensure_directory(self.debug_output_dir / image_name)

        original_path = save_image(output_dir / "original.png", image)
        save_image(output_dir / "serial_region.png", serial_stage.image)
        self._write_text(output_dir / "serial_ocr.txt", serial_stage.raw_text or serial_stage.note)
        save_image(output_dir / "watermark_mask.png", geometry_stage.mask)
        save_image(output_dir / "largest_component.png", geometry_stage.largest_component_visualization)
        save_image(output_dir / "rotated_band.png", geometry_stage.rotated_band)
        self._write_text(output_dir / "watermark_ocr.txt", watermark_ocr_stage.raw_text or watermark_ocr_stage.note)
        self._write_json(
            output_dir / "decision.json",
            {
                "deleted": decision.deleted,
                "confidence": decision.confidence,
                "rule": decision.rule,
                "reason": decision.reason,
                "serial_stage": {
                    "level": serial_stage.level,
                    "score": serial_stage.score,
                    "q_detected": serial_stage.q_detected,
                    "visual_q_detected": serial_stage.visual_q_detected,
                    "text_lines": serial_stage.text_lines,
                    "note": serial_stage.note,
                    "left_text": serial_stage.left_text,
                    "right_text": serial_stage.right_text,
                },
                "watermark_geometry_stage": {
                    "level": geometry_stage.level,
                    "score": geometry_stage.score,
                    "angle": geometry_stage.angle,
                    "box": geometry_stage.box,
                    "component_count": geometry_stage.component_count,
                    "crosses_center": geometry_stage.crosses_center,
                    "width_ratio": geometry_stage.width_ratio,
                    "height_ratio": geometry_stage.height_ratio,
                    "mean_intensity": geometry_stage.mean_intensity,
                },
                "watermark_ocr_stage": {
                    "level": watermark_ocr_stage.level,
                    "score": watermark_ocr_stage.score,
                    "best_match": watermark_ocr_stage.best_match,
                    "similarity": watermark_ocr_stage.similarity,
                    "text_lines": watermark_ocr_stage.text_lines,
                    "note": watermark_ocr_stage.note,
                },
            },
        )
        save_image(output_dir / "deleted_detection.png", geometry_stage.largest_component_visualization)
        return original_path

    def _extract_layout_region(self, image: np.ndarray, relative_box: tuple[float, float, float, float]) -> np.ndarray:
        """Crop a fixed-layout region using normalized coordinates."""
        height, width = image.shape[:2]
        left = max(0, int(width * relative_box[0]))
        top = max(0, int(height * relative_box[1]))
        right = min(width, int(width * relative_box[2]))
        bottom = min(height, int(height * relative_box[3]))
        return image[top:bottom, left:right].copy()

    def _prepare_for_watermark_detection(self, grayscale: np.ndarray) -> np.ndarray:
        """Mask static layout regions that should not contribute to watermark detection."""
        prepared = grayscale.copy()
        height, width = prepared.shape[:2]
        margin = max(8, min(height, width) // 40)
        prepared[:margin, :] = 255
        prepared[-margin:, :] = 255
        prepared[:, :margin] = 255
        prepared[:, -margin:] = 255

        for relative_box in (self._SERIAL_BOX, self._PHOTO_BOX):
            left = int(width * relative_box[0])
            top = int(height * relative_box[1])
            right = int(width * relative_box[2])
            bottom = int(height * relative_box[3])
            prepared[top:bottom, left:right] = 255

        return prepared

    @staticmethod
    def _build_watermark_candidate_masks(rotated_grayscale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build masks for candidate watermark letters and the connected watermark band."""
        blurred = cv2.GaussianBlur(rotated_grayscale, (5, 5), 0)
        letter_mask = cv2.inRange(blurred, 100, 240)
        horizontal_open = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5))
        horizontal_close = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 7))
        opened = cv2.morphologyEx(letter_mask, cv2.MORPH_OPEN, horizontal_open)
        connected_band = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, horizontal_close)
        connected_band = cv2.dilate(connected_band, np.ones((3, 3), dtype=np.uint8), iterations=1)
        return opened, connected_band

    def _select_watermark_component(
        self,
        rotated_gray: np.ndarray,
        letter_mask: np.ndarray,
        watermark_mask: np.ndarray,
        angle: float,
    ) -> WatermarkGeometryResult:
        """Select the best diagonal watermark band for one rotation angle."""
        contours, _ = cv2.findContours(watermark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_height, image_width = rotated_gray.shape[:2]
        best_score = 0.0
        best_box: tuple[int, int, int, int] | None = None
        best_component_count = 0
        best_level = "none"
        best_width_ratio = 0.0
        best_height_ratio = 0.0
        best_mean_intensity = 255.0
        best_projected_polygon: list[list[int]] = []
        candidate_boxes: list[tuple[int, int, int, int]] = []
        for contour in contours:
            candidate_box = cv2.boundingRect(contour)
            if self._is_geometry_candidate_valid(
                image_shape=rotated_gray.shape[:2],
                box=candidate_box,
            ):
                candidate_boxes.append(candidate_box)

        grouped_boxes = self._group_watermark_boxes(candidate_boxes, image_height)
        for grouped_box in grouped_boxes:
            x_coord, y_coord, width, height = grouped_box
            component_count = self._count_letter_components(letter_mask, grouped_box)
            width_ratio = width / image_width
            height_ratio = height / image_height
            projected_polygon = self._project_rotated_box_to_original(
                image_shape=rotated_gray.shape[:2],
                angle=angle,
                box=grouped_box,
            )
            crosses_center = self._polygon_crosses_original_center(projected_polygon, rotated_gray.shape[:2])
            if not crosses_center:
                continue

            masked_pixels = rotated_gray[y_coord : y_coord + height, x_coord : x_coord + width][
                watermark_mask[y_coord : y_coord + height, x_coord : x_coord + width] > 0
            ]
            mean_intensity = float(masked_pixels.mean()) if masked_pixels.size else 255.0
            if not 105.0 <= mean_intensity <= 240.0:
                continue

            level = self._geometry_level(width_ratio, height_ratio, component_count)
            if level == "none":
                continue

            score = 0.8 if level == "strong" else 0.58
            if score > best_score:
                best_score = score
                best_box = grouped_box
                best_component_count = component_count
                best_level = level
                best_width_ratio = width_ratio
                best_height_ratio = height_ratio
                best_mean_intensity = mean_intensity
                best_projected_polygon = projected_polygon

        visualization = cv2.cvtColor(rotated_gray, cv2.COLOR_GRAY2BGR)
        band_image = np.full((32, 32), 255, dtype=np.uint8)
        if best_box is not None:
            x_coord, y_coord, width, height = best_box
            cv2.rectangle(
                visualization,
                (x_coord, y_coord),
                (x_coord + width, y_coord + height),
                (0, 255, 0),
                2,
            )
            cv2.putText(
                visualization,
                f"angle:{int(angle)} level:{best_level}",
                (12, max(24, y_coord - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            band_image = self._crop_band(rotated_gray, best_box)

        return WatermarkGeometryResult(
            mask=watermark_mask,
            largest_component_visualization=visualization,
            rotated_band=band_image,
            angle=angle if best_box is not None else None,
            box=best_box,
            level=best_level,
            score=best_score,
            component_count=best_component_count,
            crosses_center=best_box is not None,
            width_ratio=best_width_ratio,
            height_ratio=best_height_ratio,
            mean_intensity=best_mean_intensity,
            projected_polygon=best_projected_polygon,
        )

    @staticmethod
    def _group_watermark_boxes(
        boxes: list[tuple[int, int, int, int]],
        image_height: int,
    ) -> list[tuple[int, int, int, int]]:
        """Merge aligned watermark components into larger diagonal band candidates."""
        if not boxes:
            return []

        grouped: list[list[tuple[int, int, int, int]]] = []
        tolerance = max(24, int(image_height * 0.08))
        for box in sorted(boxes, key=lambda item: (item[1], item[0])):
            box_center_y = box[1] + (box[3] / 2)
            placed = False
            for group in grouped:
                group_centers = [candidate[1] + (candidate[3] / 2) for candidate in group]
                if abs(box_center_y - (sum(group_centers) / len(group_centers))) <= tolerance:
                    group.append(box)
                    placed = True
                    break
            if not placed:
                grouped.append([box])

        merged_boxes: list[tuple[int, int, int, int]] = []
        for group in grouped:
            for candidate in group:
                merged_boxes.append(candidate)
            if len(group) < 2:
                continue
            min_x = min(candidate[0] for candidate in group)
            min_y = min(candidate[1] for candidate in group)
            max_x = max(candidate[0] + candidate[2] for candidate in group)
            max_y = max(candidate[1] + candidate[3] for candidate in group)
            merged_boxes.append((min_x, min_y, max_x - min_x, max_y - min_y))

        # remove exact duplicates while preserving order
        unique_boxes: list[tuple[int, int, int, int]] = []
        for box in merged_boxes:
            if box not in unique_boxes:
                unique_boxes.append(box)
        return unique_boxes

    @staticmethod
    def _is_geometry_candidate_valid(
        image_shape: tuple[int, int],
        box: tuple[int, int, int, int],
    ) -> bool:
        """Reject border artifacts, tiny blobs, and full-image components."""
        image_height, image_width = image_shape
        x_coord, y_coord, width, height = box
        edge_margin = max(8, min(image_width, image_height) // 45)
        if (
            x_coord <= edge_margin
            or y_coord <= edge_margin
            or x_coord + width >= image_width - edge_margin
            or y_coord + height >= image_height - edge_margin
        ):
            return False

        width_ratio = width / image_width
        height_ratio = height / image_height
        area_ratio = (width * height) / float(image_width * image_height)
        if area_ratio >= 0.14:
            return False
        if width_ratio < 0.1 or height_ratio < 0.02:
            return False
        return True

    @staticmethod
    def _count_letter_components(mask: np.ndarray, box: tuple[int, int, int, int]) -> int:
        """Count separate letter-like components inside a candidate watermark band."""
        x_coord, y_coord, width, height = box
        roi = mask[y_coord : y_coord + height, x_coord : x_coord + width]
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(roi, connectivity=8)
        del labels
        valid_count = 0
        roi_area = float(max(1, roi.shape[0] * roi.shape[1]))
        for component_index in range(1, component_count):
            component_area = float(stats[component_index, cv2.CC_STAT_AREA])
            if component_area / roi_area < 0.002:
                continue
            valid_count += 1
        return valid_count

    @staticmethod
    def _geometry_level(width_ratio: float, height_ratio: float, component_count: int) -> str:
        """Categorize watermark geometry into none, medium, or strong."""
        if width_ratio >= 0.18 and height_ratio >= 0.03 and component_count >= 3:
            return "strong"
        if width_ratio >= 0.12 and height_ratio >= 0.025 and component_count >= 2:
            return "medium"
        return "none"

    @staticmethod
    def _crop_band(rotated_gray: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        """Crop the detected watermark band with a small safety padding."""
        x_coord, y_coord, width, height = box
        pad_x = max(10, width // 18)
        pad_y = max(8, height // 3)
        image_height, image_width = rotated_gray.shape[:2]
        left = max(0, x_coord - pad_x)
        top = max(0, y_coord - pad_y)
        right = min(image_width, x_coord + width + pad_x)
        bottom = min(image_height, y_coord + height + pad_y)
        return rotated_gray[top:bottom, left:right].copy()

    def _ocr_lines_or_empty(self, image: np.ndarray, error_message: str) -> list[str]:
        """Run OCR on an image and return cleaned lines, swallowing runtime failures."""
        try:
            return self._extract_ocr_lines(self.ocr_service.run_ocr_on_image(image))
        except RuntimeError:
            logger.warning(error_message.replace("failed", "unavailable"))
            return []
        except Exception:
            log_exception(logger, error_message)
            return []

    @staticmethod
    def _extract_ocr_lines(ocr_result: OCRResult) -> list[str]:
        """Return cleaned OCR line strings from a structured OCR result."""
        return [re.sub(r"\s+", " ", line.text).strip() for line in ocr_result.lines if line.text.strip()]

    @staticmethod
    def _map_digits_to_ascii(text: str) -> str:
        """Convert Devanagari digits to ASCII digits for serial matching."""
        devanagari_digits = str.maketrans("०१२३४५६७८९", "0123456789")
        return text.translate(devanagari_digits)

    def _is_q_like_token(self, token: str) -> bool:
        """Treat OCR outputs resembling a leading Q as deleted markers in the serial box."""
        mapped = self._map_digits_to_ascii(token)
        if mapped == "Q":
            return True
        return token in {"Q", "९", "०"} or mapped in {"9", "0"}

    @staticmethod
    def _detect_visual_q_marker(left_region: np.ndarray) -> bool:
        """Detect the fixed-layout Q marker visually when OCR drops it."""
        grayscale = left_region if left_region.ndim == 2 else cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(thresholded, connectivity=8)

        region_height, region_width = thresholded.shape[:2]
        for component_index in range(1, component_count):
            x_coord, y_coord, width, height, area = stats[component_index]
            if x_coord == 0 and width >= region_width - 2:
                continue
            if area < 180:
                continue
            if x_coord > int(region_width * 0.55):
                continue
            if y_coord < int(region_height * 0.15) or y_coord + height > int(region_height * 0.9):
                continue
            if width < int(region_width * 0.12) or height < int(region_height * 0.25):
                continue
            return True
        return False

    @staticmethod
    def _project_rotated_box_to_original(
        image_shape: tuple[int, int],
        angle: float,
        box: tuple[int, int, int, int],
    ) -> list[list[int]]:
        """Project a rotated-space box back into original image coordinates."""
        image_height, image_width = image_shape
        x_coord, y_coord, width, height = box
        corners = np.array(
            [
                [x_coord, y_coord],
                [x_coord + width, y_coord],
                [x_coord + width, y_coord + height],
                [x_coord, y_coord + height],
            ],
            dtype=np.float32,
        ).reshape((-1, 1, 2))
        center = (image_width / 2, image_height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        inverse_matrix = cv2.invertAffineTransform(rotation_matrix)
        projected = cv2.transform(corners, inverse_matrix).reshape((-1, 2))
        projected[:, 0] = np.clip(projected[:, 0], 0, image_width - 1)
        projected[:, 1] = np.clip(projected[:, 1], 0, image_height - 1)
        return projected.astype(int).tolist()

    @staticmethod
    def _polygon_crosses_original_center(polygon: list[list[int]], image_shape: tuple[int, int]) -> bool:
        """Check whether a projected watermark polygon crosses the central card area."""
        if not polygon:
            return False
        image_height, image_width = image_shape
        polygon_array = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
        x_coord, y_coord, width, height = cv2.boundingRect(polygon_array)

        center_left = int(image_width * 0.35)
        center_top = int(image_height * 0.35)
        center_right = int(image_width * 0.65)
        center_bottom = int(image_height * 0.65)

        overlap_left = max(x_coord, center_left)
        overlap_top = max(y_coord, center_top)
        overlap_right = min(x_coord + width, center_right)
        overlap_bottom = min(y_coord + height, center_bottom)
        return overlap_right > overlap_left and overlap_bottom > overlap_top

    @staticmethod
    def _write_text(path: Path, content: str) -> Path:
        """Write a UTF-8 text file."""
        path.write_text(content, encoding="utf-8")
        return path

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> Path:
        """Write a JSON file with indentation."""
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if image.ndim == 2:
            return image.copy()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize OCR text for resilient DELETED and Q matching."""
        return re.sub(r"[^A-Z0-9]", "", text.upper())

    @staticmethod
    def _rotate_image(image: np.ndarray, angle: float, border_value: int | tuple[int, int, int]) -> np.ndarray:
        """Rotate an image around its center."""
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
