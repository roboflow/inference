import json
from typing import List, Optional, Union

import supervision as sv

from inference.core.workflows.core_steps.formatters.vlm_as_detector.qwen_detection_parsing import (
    QWEN_BOX_COORDINATE_SCALE,
    extract_qwen_detection_entries,
    get_qwen_detection_box,
    parse_qwen_object_detection_response,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

# The two Z.ai detection contracts, keyed by the box field each prompt pins:
# GLM 5V Turbo prompts for "box_2d" xyxy integers normalized to 0-1000 (the
# Qwen contract), while GLM 5.3 Flash prompts for "bbox_2d" in absolute
# pixels of the original image (the format that scored best for it in the
# vlm-exam benchmarks; it scores ~0 mAP with normalized prompts). Rather
# than duplicating the Qwen assembly, absolute-pixel entries are rescaled
# into the 0-1000 space and handed to the Qwen parser, which scales them
# back onto the image exactly.
ZAI_ABSOLUTE_PIXEL_BOX_KEY = "bbox_2d"


def extract_zai_json_array(raw: str) -> Optional[list]:
    """Recover the outermost JSON array from prose-wrapped Z.ai output.

    GLM models occasionally wrap the detection list in extra text that
    breaks whole-string JSON parsing. Mirrors the vlm-exam fallback: take
    the substring between the first ``[`` and the last ``]`` and parse it.

    Args:
        raw: Raw VLM output that failed regular JSON parsing.

    Returns:
        The recovered list, or ``None`` when nothing recoverable.
    """
    start = raw.find("[")
    stop = raw.rfind("]")
    if start == -1 or stop <= start:
        return None
    try:
        recovered = json.loads(raw[start : stop + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(recovered, list):
        return None
    return recovered


def zai_entries_use_absolute_pixels(entries: List[dict]) -> bool:
    """Tell whether Z.ai detection entries carry absolute-pixel boxes.

    The GLM 5.3 Flash prompt pins the ``bbox_2d`` key for absolute-pixel
    boxes; the GLM 5V Turbo prompt pins ``box_2d`` for 0-1000 normalized
    boxes. The models echo the prompted key, so its presence selects the
    coordinate space.

    Args:
        entries: Raw detection entry dicts.

    Returns:
        True when any dict entry contains the ``bbox_2d`` key.
    """
    return any(
        isinstance(entry, dict) and ZAI_ABSOLUTE_PIXEL_BOX_KEY in entry
        for entry in entries
    )


def convert_zai_pixel_entries_to_normalized(
    entries: List[dict],
    image_height: int,
    image_width: int,
) -> List[dict]:
    """Rescale absolute-pixel ``bbox_2d`` entries into the 0-1000 space.

    Coordinates are absolute pixels of the image the model saw. The block
    only downscales uploads above the OpenRouter payload cap (extremely
    large images), so the coordinates map onto the original image directly.
    Entries without a well-formed box pass through untouched and are
    skipped downstream.

    Args:
        entries: Raw detection entry dicts.
        image_height: Original image height in pixels.
        image_width: Original image width in pixels.

    Returns:
        Entries with ``box_2d`` normalized to 0-1000, consumable by the
        Qwen parser.
    """
    scale = QWEN_BOX_COORDINATE_SCALE
    normalized: List[dict] = []
    for entry in entries:
        box = (
            get_qwen_detection_box(detection=entry) if isinstance(entry, dict) else None
        )
        if box is None:
            normalized.append(entry)
            continue
        x_min, y_min, x_max, y_max = box
        converted = {k: v for k, v in entry.items() if k != ZAI_ABSOLUTE_PIXEL_BOX_KEY}
        converted["box_2d"] = [
            x_min / image_width * scale,
            y_min / image_height * scale,
            x_max / image_width * scale,
            y_max / image_height * scale,
        ]
        normalized.append(converted)
    return normalized


def parse_zai_object_detection_response(
    image: WorkflowImageData,
    parsed_data: Union[dict, list],
    classes: List[str],
    inference_id: str,
) -> sv.Detections:
    """Parse Z.ai block object-detection output into detections.

    Dispatches on the box key the response carries: ``bbox_2d`` entries
    (GLM 5.3 Flash) are rescaled from absolute pixels into the 0-1000
    space first; either way the Qwen parser does the parsing.

    Args:
        image: Workflow image the detections refer to.
        parsed_data: JSON payload extracted from the VLM output.
        classes: Class names used to map labels onto class ids.
        inference_id: Identifier attached to every parsed detection.

    Returns:
        Parsed detections in the original image's coordinate space.

    Raises:
        ValueError: If the response is neither a JSON list nor a
            ``{"detections": [...]}`` object.
    """
    entries = extract_qwen_detection_entries(parsed_data=parsed_data)
    if zai_entries_use_absolute_pixels(entries):
        image_height, image_width = image.numpy_image.shape[:2]
        parsed_data = convert_zai_pixel_entries_to_normalized(
            entries=entries,
            image_height=image_height,
            image_width=image_width,
        )
    return parse_qwen_object_detection_response(
        image=image,
        parsed_data=parsed_data,
        classes=classes,
        inference_id=inference_id,
    )
