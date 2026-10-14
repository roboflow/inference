"""Decoding helpers used by the VLM parser blocks: box formats and Florence-2.

``vlm_as_classifier/v2.py`` and ``vlm_as_detector/v2.py`` own their Params and
``run``; their identities are retired from V2 discovery and public exports.
They read an answer produced by another block and decode it with
``decoding``, the decoder VLM blocks also use. Florence-2 still uses
``decode_florence_2_detections`` below.

``MODEL_BOX_FORMATS`` picks the box format from the model that answered; the
detector's ``ModelType`` lists exactly its keys plus ``florence-2``::

    model_type                  box format      upload size of absolute pixels
    openai                      xyxy_absolute   longest edge <= 2048
    anthropic-claude            xyxy_absolute   Claude's own resize
    google-gemini               yxyx_0_1000     -
    spacexai                    xyxy_percent    -
    qwen, zai, zai-flash        xyxy_0_1000     -
    muse                        named_0_1000    -
    florence-2                  Florence-2 task output, absolute pixels

``xyxy_absolute`` and ``yxyx_0_1000`` also read the V1 OpenAI, Claude and
Gemini answers with 0-1 ``x_min``.. fields. Each block returns
``error_status`` ``True`` and ``predictions`` ``None`` when the answer cannot
be read; downstream steps that need ``predictions`` are then skipped.

Differences from V1, besides those of ``decoding``:

* One decoder per box format replaces the per-vendor parsers. An entry
  without a well-formed box is skipped instead of failing the answer; an
  answer in which no entry has one still sets ``error_status``. Boxes are
  clamped to their format's range and corners are reordered.
* Florence-2 ``open-vocabulary-object-detection`` needs ``classes``; V1
  accepted the step and then failed at run time.
* ``classes`` must be unique strings; a repeated name fails the step. V1's
  classifier silently gave the earlier name the later id.
* Florence-2 caption, OCR and region-proposal rows keep class id ``0`` and
  their own ``"class"`` text, but ``class_names`` is empty. V1 tensor mode
  mapped ``0`` to the last row's text, which misnamed every other row.
"""

import hashlib
import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
from inference_models.models.base.object_detection import Detections
from roboflow_workflows.v2.blocks.language.decoding import (
    DETECTION_MAX_EDGE_PIXELS,
    DETECTION_PREDICTION_TYPE,
    SizeWH,
    anthropic_upload_size,
    checked_classes,
    extract_json,
    max_edge_upload_size,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import with_row_identity

MODEL_BOX_FORMATS: Dict[str, Tuple[str, Optional[Callable[[SizeWH], SizeWH]]]] = {
    "openai": (
        "xyxy_absolute",
        lambda size_wh: max_edge_upload_size(
            size_wh, max_edge=DETECTION_MAX_EDGE_PIXELS
        ),
    ),
    "anthropic-claude": ("xyxy_absolute", anthropic_upload_size),
    "google-gemini": ("yxyx_0_1000", None),
    "spacexai": ("xyxy_percent", None),
    "qwen": ("xyxy_0_1000", None),
    "zai": ("xyxy_0_1000", None),
    "zai-flash": ("xyxy_0_1000", None),
    "muse": ("named_0_1000", None),
}

FLORENCE_2_TASKS = {
    "object-detection": "<OD>",
    "open-vocabulary-object-detection": "<OPEN_VOCABULARY_DETECTION>",
    "object-detection-and-caption": "<DENSE_REGION_CAPTION>",
    "phrase-grounded-object-detection": "<CAPTION_TO_PHRASE_GROUNDING>",
    "region-proposal": "<REGION_PROPOSAL>",
    "ocr-with-text-detection": "<OCR_WITH_REGION>",
}

REGION_PROPOSAL_LABEL = "roi"


def decode_florence_2_detections(
    raw_output: Any,
    *,
    image: ImageData,
    task_type: str,
    classes: Optional[List[str]],
    inference_id: str,
) -> Tuple[bool, Optional[Detections]]:
    """Decode a Florence-2 task answer into boxes, as V1 does.

    The answer holds image-pixel ``bboxes`` (``quad_boxes`` for OCR, whose
    enclosing box is kept) and their labels. Class ids follow V1: a stable
    0-9999 id derived from the label for object detection and phrase
    grounding, the ``classes`` index (else ``-1``) for open vocabulary, and
    ``0`` otherwise. Region proposals are labelled ``roi``. Confidence is 1.
    Boxes are kept as answered: no clamping or rounding.

    Every row's text is its ``"class"``. ``class_names`` maps ids to names
    only where an id stands for one name. Caption, OCR and region-proposal
    rows all share id ``0`` whatever their text, so their ``class_names`` is
    empty: the text is not a category, and consumers read each row's own
    ``"class"`` first.

    Args:
        raw_output: The Florence-2 task output as JSON text or a dict.
        image: The image Florence-2 saw.
        task_type: A ``FLORENCE_2_TASKS`` key.
        classes: Class names; required for open-vocabulary detection.
        inference_id: Namespace of the result's row identity.

    Returns:
        ``(error_status, predictions)``: ``Detections`` on the image's device,
        or ``None`` when the answer cannot be read.

    Raises:
        ValueError: On an unknown task or invalid ``classes``.
    """
    if task_type not in FLORENCE_2_TASKS:
        raise ValueError(f"Florence-2 does not run task_type {task_type!r}")
    open_vocabulary = task_type == "open-vocabulary-object-detection"
    classes = checked_classes(classes) if open_vocabulary or classes else []

    error_status, payload = extract_json(raw_output)
    if error_status or not isinstance(payload, Mapping):
        return True, None

    try:
        boxes, labels = _florence_2_rows(payload, task=FLORENCE_2_TASKS[task_type])
    except (KeyError, TypeError, ValueError):
        return True, None

    if open_vocabulary:
        ids_by_name = {name: class_id for class_id, name in enumerate(classes)}
        class_ids = [ids_by_name.get(label, -1) for label in labels]
        class_names = dict(enumerate(classes))
    elif task_type in ("object-detection", "phrase-grounded-object-detection"):
        class_ids = [_label_id(label) for label in labels]
        class_names = dict(zip(class_ids, labels))
    else:
        # Every text shares id 0, so no single name stands for the id.
        class_ids = [0] * len(labels)
        class_names = {}

    device = image.device
    detections = Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).to(device),
        class_id=torch.tensor(class_ids, dtype=torch.int64, device=device),
        confidence=torch.ones(len(boxes), dtype=torch.float32, device=device),
        image_metadata={
            **image.prediction_metadata(),
            "class_names": class_names,
            "prediction_type": DETECTION_PREDICTION_TYPE,
        },
        bboxes_metadata=[{"class": label} for label in labels],
    )
    identified = with_row_identity(detections, namespace=inference_id)

    return False, identified


def _florence_2_rows(
    payload: Mapping, *, task: str
) -> Tuple[List[List[float]], List[str]]:
    if task == "<OCR_WITH_REGION>":
        boxes = [_enclosing_box(quad) for quad in payload["quad_boxes"]]
    else:
        boxes = [_number_list(box, length=4) for box in payload["bboxes"]]

    if task == "<REGION_PROPOSAL>":
        labels = [REGION_PROPOSAL_LABEL] * len(boxes)
    else:
        key = "bboxes_labels" if task == "<OPEN_VOCABULARY_DETECTION>" else "labels"
        labels = [str(label) for label in payload[key]]

    if len(labels) != len(boxes):
        raise ValueError(f"{len(boxes)} boxes but {len(labels)} labels")

    return boxes, labels


def _number_list(values: Any, *, length: int) -> List[float]:
    numbers = [float(value) for value in values]
    malformed = isinstance(values, (str, bytes)) or len(numbers) != length
    if malformed or not all(math.isfinite(number) for number in numbers):
        raise ValueError(f"expected {length} finite numbers, got {values!r}")

    return numbers


def _enclosing_box(quad: Any) -> List[float]:
    numbers = _number_list(quad, length=8)
    xs, ys = numbers[0::2], numbers[1::2]
    box = [min(xs), min(ys), max(xs), max(ys)]

    return box


def _label_id(label: str) -> int:
    # V1 get_4digit_from_md5: stable across runs and processes.
    digest = hashlib.md5(label.encode("utf-8")).hexdigest()
    label_id = int(digest[:9], 16) % 10000

    return label_id


__all__ = ["FLORENCE_2_TASKS", "MODEL_BOX_FORMATS", "decode_florence_2_detections"]
