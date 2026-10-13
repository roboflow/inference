"""Turn a VLM answer into a prediction: ports of the V1 VLM parser blocks.

Ports of ``roboflow_core/vlm_as_classifier@v2`` and
``roboflow_core/vlm_as_detector@v2``. They read an answer produced by another
block and decode it with ``decoding``, the decoder VLM blocks also use. The
detector picks the box format from the model that answered::

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
import uuid
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import torch
from pydantic import Field, model_validator
from roboflow_workflows.v2.blocks.language.decoding import (
    DETECTION_MAX_EDGE_PIXELS,
    DETECTION_PREDICTION_TYPE,
    SizeWH,
    anthropic_upload_size,
    checked_classes,
    decode_classification,
    decode_object_detections,
    extract_json,
    max_edge_upload_size,
    vlm_prediction_outputs,
)
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import LIST_OF_VALUES_KIND, STRING_KIND
from roboflow_workflows.v2.data.language import LANGUAGE_MODEL_OUTPUT_KIND
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Ref

from inference_models.models.base.object_detection import Detections

ModelType = Literal[
    "openai",
    "google-gemini",
    "anthropic-claude",
    "spacexai",
    "qwen",
    "muse",
    "zai",
    "zai-flash",
    "florence-2",
]
DetectorTaskType = Literal[
    "object-detection",
    "object-detection-and-caption",
    "open-vocabulary-object-detection",
    "phrase-grounded-object-detection",
    "region-proposal",
    "ocr-with-text-detection",
]

# model_type -> (box format, upload size of absolute formats from image size).
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
# task_type -> Florence-2 task token, as V1 maps them.
FLORENCE_2_TASKS = {
    "object-detection": "<OD>",
    "open-vocabulary-object-detection": "<OPEN_VOCABULARY_DETECTION>",
    "object-detection-and-caption": "<DENSE_REGION_CAPTION>",
    "phrase-grounded-object-detection": "<CAPTION_TO_PHRASE_GROUNDING>",
    "region-proposal": "<REGION_PROPOSAL>",
    "ocr-with-text-detection": "<OCR_WITH_REGION>",
}
REGION_PROPOSAL_LABEL = "roi"

_VLM_OUTPUT_DESCRIPTION = (
    "Answer of a VLM or LLM step: text with JSON, raw or inside a Markdown "
    "fence (the first ```json fence is read), or an already decoded dict."
)


class _ParserParams(BlockParams):
    image: Ref(IMAGE_KIND) = Field(
        description=(
            "The image the VLM answered about; sets the prediction's size, "
            "frame and device."
        ),
        examples=["$inputs.image", "$steps.crop.crops"],
    )
    vlm_output: str | Ref(LANGUAGE_MODEL_OUTPUT_KIND, STRING_KIND) = Field(
        description=_VLM_OUTPUT_DESCRIPTION,
        examples=["$steps.vlm.output"],
    )


class VLMAsClassifierBlock(Block):
    """Decode a VLM classification answer into a classification prediction.

    ``{"class_name": "cat", "confidence": 0.9}`` gives a single-label
    prediction with the full confidence vector over ``classes``;
    ``{"predicted_classes": [{"class": ..., "confidence": ...}]}`` or a bare
    list of such entries gives a multi-label prediction. A label outside
    ``classes`` is appended after them. Stateless and prunable.
    """

    type = "v2/vlm_as_classifier@v2"
    prunable = True
    metadata = {
        "name": "VLM As Classifier",
        "port_of": "roboflow_core/vlm_as_classifier@v2",
        "section": "advanced",
    }
    outputs = vlm_prediction_outputs("classification")

    class Params(_ParserParams):
        classes: List[str] | Ref(LIST_OF_VALUES_KIND) = Field(
            description=(
                "Class names the VLM was asked to choose from; classes[i] has "
                "class id i. Names must be unique."
            ),
            examples=[["dog", "cat"], "$inputs.classes"],
        )

    def run(
        self, *, image: ImageData, vlm_output: Any, classes: List[str]
    ) -> Dict[str, Any]:
        """Decode ``vlm_output`` for ``image``.

        Args:
            image: The classified image.
            vlm_output: Answer text or decoded dict.
            classes: Unique class names.

        Returns:
            ``predictions``, ``error_status`` and a fresh ``inference_id``.

        Raises:
            ValueError: When ``classes`` is not a list of unique strings.
        """
        error_status, predictions = decode_classification(
            vlm_output, image=image, classes=classes
        )
        result = {
            "predictions": predictions,
            "error_status": error_status,
            "inference_id": uuid.uuid4().hex,
        }

        return result


class VLMAsDetectorBlock(Block):
    """Decode a VLM detection answer into boxes on the image.

    ``model_type`` names the model that answered, which fixes the coordinate
    format to read (see the module table). Labels outside ``classes`` keep
    their name with class id ``-1``. Stateless and prunable.
    """

    type = "v2/vlm_as_detector@v2"
    prunable = True
    metadata = {
        "name": "VLM As Detector",
        "port_of": "roboflow_core/vlm_as_detector@v2",
        "section": "advanced",
    }
    outputs = vlm_prediction_outputs("object-detection")

    class Params(_ParserParams):
        classes: Optional[List[str]] | Ref(LIST_OF_VALUES_KIND) = Field(
            default=None,
            description=(
                "Class names the VLM was asked to detect; classes[i] has class "
                "id i. Names must be unique. Required except for Florence-2 "
                "tasks other than open-vocabulary-object-detection."
            ),
            examples=[["dog", "cat"], "$inputs.classes"],
        )
        model_type: ModelType = Field(
            description=(
                "Model that produced the answer; selects the box coordinate "
                "format to read."
            ),
            examples=["google-gemini", "qwen"],
        )
        task_type: DetectorTaskType = Field(
            default="object-detection",
            description=(
                "Task the model ran. Every model supports object-detection; "
                "the other tasks are Florence-2 tasks."
            ),
            examples=["object-detection", "ocr-with-text-detection"],
        )

        @model_validator(mode="after")
        def _task_and_classes_fit_the_model(self) -> "VLMAsDetectorBlock.Params":
            florence = self.model_type == "florence-2"
            if not florence and self.task_type != "object-detection":
                raise ValueError(
                    f"model_type {self.model_type!r} supports only task_type "
                    f"'object-detection', got {self.task_type!r}"
                )
            needs_classes = (
                not florence or self.task_type == "open-vocabulary-object-detection"
            )
            if needs_classes and self.classes is None:
                raise ValueError(
                    f"classes is required for model_type {self.model_type!r} "
                    f"and task_type {self.task_type!r}"
                )

            return self

    def run(
        self,
        *,
        image: ImageData,
        vlm_output: Any,
        model_type: str,
        task_type: str = "object-detection",
        classes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Decode ``vlm_output`` into ``Detections`` on ``image``.

        Args:
            image: The image the model saw.
            vlm_output: Answer text or decoded dict.
            model_type: Model that answered.
            task_type: Task the model ran.
            classes: Unique class names, or ``None`` where allowed.

        Returns:
            ``predictions``, ``error_status`` and a fresh ``inference_id``,
            also the namespace of the rows' identity.

        Raises:
            ValueError: When ``classes`` is not a list of unique strings.
        """
        inference_id = uuid.uuid4().hex
        if model_type == "florence-2":
            error_status, predictions = decode_florence_2_detections(
                vlm_output,
                image=image,
                task_type=task_type,
                classes=classes,
                inference_id=inference_id,
            )
        else:
            box_format, upload_size = MODEL_BOX_FORMATS[model_type]
            height, width = image.size_hw
            error_status, predictions = decode_object_detections(
                vlm_output,
                image=image,
                classes=classes,
                box_format=box_format,
                inference_id=inference_id,
                upload_size_wh=upload_size((width, height)) if upload_size else None,
            )
        result = {
            "predictions": predictions,
            "error_status": error_status,
            "inference_id": inference_id,
        }

        return result


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


__all__ = [
    "FLORENCE_2_TASKS",
    "MODEL_BOX_FORMATS",
    "VLMAsClassifierBlock",
    "VLMAsDetectorBlock",
    "decode_florence_2_detections",
]
