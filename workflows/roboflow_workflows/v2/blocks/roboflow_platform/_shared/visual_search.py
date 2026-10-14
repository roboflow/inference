"""Query encoding, candidate parsing and classification for the search blocks.

The blocks port V1 ``roboflow_core/visual_search@v1`` and
``roboflow_core/visual_search_classifier@v1`` (latest tensor sibling). Both
call the platform's project search; neither runs a model, locally or
remotely::

    image ─▶ JPEG q95 (classifier: downscaled to max_image_size first)
          ─▶ port.search_project_images ─▶ candidates, best first
                                            ├─ none ─▶ candidate_found=False
                                            └─ best ─▶ outputs
    classifier: best candidate's classification / labels / annotations
          ─▶ ClassificationPrediction or MultiLabelClassificationPrediction
    best_candidate_image: port.load_candidate_image(image_url), only when
          some reader wants it
    any failure ─▶ error_status=True and V1's "... failed: <error>" message,
          except a missing port or API key, which fails the step

Each block owns its Params, outputs, constructor, range checks and result
shape. This module keeps what both share: query encoding, candidate
formatting, the candidate image download, score rules and classification
parsing.

Kept from V1: parameters, outputs, candidate fields, messages, score and
confidence rules, the classification parsing rules and the class id layout of
the tensor sibling (ids below the matched one are filled with ``str(id)``
names and confidence 0.0).

Differences from V1:

* The query is always re-encoded as JPEG quality 95; V1 sent a base64 input
  image's original bytes unchanged.
* ``best_candidate_image`` is downloaded in the step, and only when demanded.
  V1 built a lazy image that downloaded when a later step read its pixels.
  A failed download fails the search result, like any other failure.
* The classifier searches the images of a batch one after another. V1 sent up
  to eight requests at once.
* ``target_project`` is a ``string``; V2 has no ``roboflow_project`` kind.
* A ``classification`` with a class name but no usable ``class_id`` gets class
  id 0. V1 reported -1, and its tensor sibling failed on it.
* Classification tensors live on the input image's device; their metadata
  holds the image's frame record, ``class_names``, ``prediction_type`` and
  ``inference_id``. V1's serializer-only keys are not written: V2 serializes
  native predictions as they are.
"""

import base64
import math
from typing import Any, Dict, List, Optional, Sequence, Union
import cv2
import numpy as np
import torch
from roboflow_workflows.v2.blocks.roboflow_platform.ports import VisualSearchPort
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.row_identity import with_classification_parent
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)

JPEG_QUALITY = 95

CLASSIFIER_FIELDS = (
    "id",
    "name",
    "filename",
    "url",
    "user_metadata",
    "tags",
    "width",
    "height",
    "aspectRatio",
    "score",
    "labels",
    "annotations",
)

__all__ = []


def _candidate_image(
    port: VisualSearchPort, candidate: Dict[str, Any], *, fallback_id: str
) -> Optional[ImageData]:
    # Downloads only on demand; the caller checks that a reader wants it.
    image_url = candidate.get("image_url")
    if not image_url:
        return None

    pixels = port.load_candidate_image(image_url)
    image_id = candidate.get("image_id") or candidate.get("filename") or fallback_id
    image = ImageData.from_numpy_rgb(pixels, image_id=str(image_id))

    return image


def _query_base64(image: ImageData, *, max_side: Optional[int]) -> str:
    # V1 bytes: BGR host pixels, cv2 bilinear downscale, JPEG quality 95.
    pixels = image.tensor_image.permute(1, 2, 0).cpu().numpy()
    if image.channels == 3:
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    else:
        pixels = pixels[:, :, 0]
    if max_side is not None:
        pixels = _downscale(pixels, max_side=max_side)

    encoded, jpeg = cv2.imencode(
        ".jpg", pixels, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    )
    if not encoded:
        raise ValueError("the query image could not be encoded as JPEG")

    query = base64.b64encode(jpeg.tobytes()).decode("ascii")

    return query


def _downscale(pixels: np.ndarray, *, max_side: int) -> np.ndarray:
    # V1 downscale_image_keeping_aspect_ratio with a square (max_side, max_side).
    height, width = pixels.shape[:2]
    if height <= max_side and width <= max_side:
        return pixels

    if width / height >= 1:
        new_width, new_height = max_side, int(max_side / (width / height))
    else:
        new_width, new_height = int(max_side * (width / height)), max_side
    resized = cv2.resize(pixels, (new_width, new_height))

    return resized


def _format_candidate(
    candidate: Dict[str, Any], extra_fields: Sequence[str] = ()
) -> Dict[str, Any]:
    filename = candidate.get("filename") or candidate.get("name")
    formatted = {
        "image_id": candidate.get("id"),
        "image_url": candidate.get("url"),
        "name": candidate.get("name") or filename,
        "filename": filename,
        "metadata": candidate.get("user_metadata") or {},
        "tags": candidate.get("tags") or [],
        "width": candidate.get("width"),
        "height": candidate.get("height"),
        "aspect_ratio": candidate.get("aspectRatio"),
    }
    for field in extra_fields:
        formatted[field] = candidate.get(field)

    return formatted


def _format_classifier_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    extra_fields = ["score", "labels", "annotations"]
    if "classification" in candidate:
        extra_fields.append("classification")

    formatted = _format_candidate(candidate, extra_fields=extra_fields)
    formatted["score"] = _finite_score(candidate.get("score"))

    return formatted


def _finite_score(score: Any) -> Optional[float]:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def _confidence(score: Any) -> float:
    # V1: project search scores map to [0, 1] as score / 2, clipped.
    value = _finite_score(score)
    if value is None:
        return 0.0

    return max(0.0, min(1.0, value / 2.0))


def _classification_prediction(
    image: ImageData,
    *,
    classification: Dict[str, Any],
    inference_id: str,
    confidence: float,
) -> Union[ClassificationPrediction, MultiLabelClassificationPrediction]:
    # Dense vectors indexed by class id, as the V1 tensor sibling: ids below the
    # largest matched one are named str(id) with confidence 0.0, because the
    # project's full class list is not part of a search result. float64 keeps
    # the API-provided confidence exact.
    classes = classification["classes"]
    class_count = max(int(entry["class_id"]) for entry in classes) + 1
    class_names = {class_id: str(class_id) for class_id in range(class_count)}
    confidences = [0.0] * class_count
    for entry in classes:
        class_names[int(entry["class_id"])] = str(entry["class"])
        confidences[int(entry["class_id"])] = confidence

    metadata = {
        **image.prediction_metadata(),
        "class_names": class_names,
        "prediction_type": "classification",
        "inference_id": inference_id,
    }
    device = image.device
    if classification["type"] == "multi_label":
        prediction = MultiLabelClassificationPrediction(
            class_ids=torch.tensor(
                [int(entry["class_id"]) for entry in classes],
                dtype=torch.long,
                device=device,
            ),
            confidence=torch.tensor(confidences, dtype=torch.float64, device=device),
            image_metadata=metadata,
        )
    else:
        prediction = ClassificationPrediction(
            class_id=torch.tensor(
                [int(classes[0]["class_id"])], dtype=torch.long, device=device
            ),
            confidence=torch.tensor([confidences], dtype=torch.float64, device=device),
            images_metadata=[metadata],
        )

    parented = with_classification_parent(prediction, image.parent_row)

    return parented


def parse_visual_search_classification(
    candidate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Read a candidate's classification, as V1's annotation parser.

    Sources in order: ``classification`` (``class`` or ``predicted_classes``),
    then ``labels``, then the ``classes`` of ``annotations``. Repeated names
    are dropped. In ``labels``, ``annotations`` and ``predicted_classes``, a
    missing, negative, non-integer or repeated class id is replaced by the
    entry's position, or the next unused id after it. A single ``class``
    without a usable id gets id 0.

    Args:
        candidate: A formatted search candidate.

    Returns:
        ``{"type": "single_label" | "multi_label", "classes": [{"class":
        name, "class_id": id}, ...]}``, or ``None`` when no source names a
        class.
    """
    classification = candidate.get("classification")
    if isinstance(classification, dict):
        if isinstance(classification.get("class"), str) and classification["class"]:
            # V1 kept a missing id as -1, which its tensor sibling could not
            # index; class 0 is the one id every dense vector has.
            class_id = max(0, _class_id(classification.get("class_id")))
            single = {
                "type": "single_label",
                "classes": [{"class": classification["class"], "class_id": class_id}],
            }
            return single

        parsed = _multi_label_classification(classification)
        if parsed is not None:
            return parsed

    labels = candidate.get("labels")
    if isinstance(labels, list):
        entries = [
            {"class": label} if isinstance(label, str) else label
            for label in labels
            if isinstance(label, (str, dict))
        ]
        parsed = _classes_from_entries(entries)
        if parsed is not None:
            return parsed

    annotations = candidate.get("annotations")
    if not isinstance(annotations, list):
        return None

    names = []
    for annotation in annotations:
        if isinstance(annotation, dict) and isinstance(annotation.get("classes"), list):
            names.extend(annotation["classes"])
    parsed = _classes_from_entries([{"class": name} for name in names])

    return parsed


def _multi_label_classification(
    classification: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    predicted_classes = classification.get("predicted_classes")
    if not isinstance(predicted_classes, list) or not predicted_classes:
        return None

    predictions = classification.get("predictions")
    if not isinstance(predictions, dict):
        predictions = {}
    entries = []
    for name in predicted_classes:
        prediction = predictions.get(name) if isinstance(name, str) else None
        class_id = prediction.get("class_id") if isinstance(prediction, dict) else None
        entries.append({"class": name, "class_id": class_id})
    classes = _unique_classes(entries)
    if not classes:
        return None

    # V1: predicted_classes are multi-label even when one class survives.
    parsed = {"type": "multi_label", "classes": classes}

    return parsed


def _classes_from_entries(entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    classes = _unique_classes(entries)
    if not classes:
        return None

    label_type = "single_label" if len(classes) == 1 else "multi_label"
    parsed = {"type": label_type, "classes": classes}

    return parsed


def _unique_classes(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    classes = []
    seen_names = set()
    used_ids = set()
    for position, entry in enumerate(entries):
        name = entry.get("class")
        if not isinstance(name, str) or not name or name in seen_names:
            continue

        class_id = _class_id(entry.get("class_id"))
        if class_id < 0 or class_id in used_ids:
            class_id = position
            while class_id in used_ids:
                class_id += 1
        classes.append({"class": name, "class_id": class_id})
        seen_names.add(name)
        used_ids.add(class_id)

    return classes


def _class_id(value: Any) -> int:
    if value is None:
        return -1

    try:
        class_id = int(value)
    except (TypeError, ValueError):
        return -1

    return class_id
