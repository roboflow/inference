"""Shared label text building and adaptive text sizing for visualization blocks."""

from typing import Any, List

import supervision as sv

from inference.core.workflows.execution_engine.constants import (
    AREA_CONVERTED_KEY_IN_SV_DETECTIONS,
    AREA_KEY_IN_SV_DETECTIONS,
)

REFERENCE_MIN_DIMENSION_PX = 1080
REFERENCE_RICH_FONT_SIZE_PT = 14
REFERENCE_LABEL_TEXT_SCALE = 0.7

MIN_RICH_FONT_SIZE_PT = 8
MAX_RICH_FONT_SIZE_PT = 128
MIN_LABEL_TEXT_SCALE = 0.3
MAX_LABEL_TEXT_SCALE = 5.0

TEXT_SIZE_MODE_MANUAL = "Manual"
TEXT_SIZE_MODE_AUTOMATIC = "Automatic"


def build_detection_labels(predictions: sv.Detections, text: str) -> Any:
    """Build per-detection label strings from predictions and a text option."""
    if text == "Class":
        return predictions["class_name"]
    if text == "Tracker Id":
        if predictions.tracker_id is not None:
            return [
                str(t) if t is not None else "No Tracker ID"
                for t in predictions.tracker_id
            ]
        return ["No Tracker ID"] * len(predictions)
    if text == "Time In Zone":
        if "time_in_zone" in predictions.data:
            return [
                f"In zone: {round(t, 2)}s" if t else "In zone: N/A"
                for t in predictions.data["time_in_zone"]
            ]
        return ["In zone: N/A"] * len(predictions)
    if text == "Confidence":
        return [f"{confidence:.2f}" for confidence in predictions.confidence]
    if text == "Class and Confidence":
        class_names = predictions["class_name"]
        if class_names is None:
            return [f"{confidence:.2f}" for confidence in predictions.confidence]
        return [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence in zip(class_names, predictions.confidence)
        ]
    if text == "Index":
        return [str(i) for i in range(len(predictions))]
    if text == "Dimensions":
        labels = []
        for i in range(len(predictions)):
            x1, y1, x2, y2 = predictions.xyxy[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            labels.append(f"{int(cx)}, {int(cy)} {int(w)}x{int(h)}")
        return labels
    if text == "Area":
        return [str(int(area)) for area in predictions.area]
    if text == "Area (mask)":
        if AREA_KEY_IN_SV_DETECTIONS in predictions.data:
            return [
                f"Area (mask): {a:.2f}" if a is not None else "Area (mask): N/A"
                for a in predictions.data[AREA_KEY_IN_SV_DETECTIONS]
            ]
        return ["Area (mask): N/A"] * len(predictions)
    if text == "Area (converted)":
        if AREA_CONVERTED_KEY_IN_SV_DETECTIONS in predictions.data:
            return [
                f"Area (conv): {a:.2f}" if a is not None else "Area (conv): N/A"
                for a in predictions.data[AREA_CONVERTED_KEY_IN_SV_DETECTIONS]
            ]
        return ["Area (conv): N/A"] * len(predictions)
    try:
        return [str(d) if d else "" for d in predictions[text]]
    except Exception as error:
        raise ValueError(f"Invalid text type: {text}") from error


def _resolution_scale(height: int, width: int) -> float:
    return min(height, width) / REFERENCE_MIN_DIMENSION_PX


def compute_adaptive_rich_font_size(
    height: int,
    width: int,
    manual_font_size: int,
    text_size_mode: str,
) -> int:
    """Resolve effective Rich Label font size in points."""
    if text_size_mode != TEXT_SIZE_MODE_AUTOMATIC:
        return manual_font_size

    base_size = REFERENCE_RICH_FONT_SIZE_PT * _resolution_scale(height, width)
    multiplier = manual_font_size / REFERENCE_RICH_FONT_SIZE_PT
    effective_size = round(base_size * multiplier)
    return max(MIN_RICH_FONT_SIZE_PT, min(MAX_RICH_FONT_SIZE_PT, effective_size))


def compute_adaptive_label_text_scale(
    height: int,
    width: int,
    manual_text_scale: float,
    text_size_mode: str,
) -> float:
    """Resolve effective Label Visualization text scale."""
    if text_size_mode != TEXT_SIZE_MODE_AUTOMATIC:
        return manual_text_scale

    base_scale = REFERENCE_LABEL_TEXT_SCALE * _resolution_scale(height, width)
    multiplier = manual_text_scale / REFERENCE_LABEL_TEXT_SCALE
    effective_scale = base_scale * multiplier
    return max(MIN_LABEL_TEXT_SCALE, min(MAX_LABEL_TEXT_SCALE, effective_scale))
