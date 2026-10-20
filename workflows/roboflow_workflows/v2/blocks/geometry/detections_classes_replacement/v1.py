"""Replace detection classes with per-crop classifications.

``roboflow_core/detections_classes_replacement@v1`` ports V1
``roboflow_core/detections_classes_replacement@v1``. The usual workflow::

    detect -> roboflow_core/dynamic_crop@v1 -> classify each crop -> this block
              (crop k = row k)                            (row k <- child k)

Results join rows by row key, never by position (decision 015)::

    row key of row k        = (namespace, id) from the predictions' row record
    row key of a result     = classification metadata parent row, set by the
                              classifier from crop.parent_row; for text
                              results, crops[same index].parent_row
    result for row k        = the result whose key equals row k's key

So filtering or sorting the detections before cropping still labels the right
rows, and repeated rows share one result. Errors instead of guesses: a result
without a key, a key of other predictions (foreign namespace), two results
for one key, and results for predictions without row identity. A result whose
row is no longer among the targets is ignored.

V1 matched by ``parent_id`` strings; V2 row keys replace them. Strings and
lists (e.g. OCR text) carry no key: bind ``crops`` to the crops they were read
from, so each text result takes its crop's ``parent_row``.

One result per row::

    ClassificationPrediction       top class: id, name, confidence
    MultiLabelClassification       most confident class of the whole
                                   distribution, if any class is predicted
    str                            (text, class id 0, confidence 1.0)
    list                           its first string; empty -> no result
    no child / no result           fallback (name, id, confidence 0.0) when
                                   fallback_class_name is set, else row dropped

Kept rows keep every aligned field (box, mask, keypoints, row metadata) and
get the new ``class_id`` and ``confidence``, a per-row ``"class"`` name and a
new ``detection_id``. ``image_metadata["class_names"]`` maps each new id to its
first name; the per-row name decides when two names share an id (all strings
share id 0).

Empty and absent inputs (decision 016): absent targets give an absent output;
present targets with no results (empty or absent group) take the fallback, or
lose every row without one. Keypoints without boxes, ``(KeyPoints, None)``,
have no classes to replace and are a ``ContractError`` (decision 018).

Difference from V1: the fallback applies to every row without a result. V1
dropped such rows when the inputs were classification predictions, and returned
no rows at all when every classification was empty.

CPU boundary: one host read of the target row ids, and one of every leading
class id and confidence per device; no read per result.
"""

import sys
import uuid
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from pydantic import Field, StrictInt
from roboflow_workflows.execution_engine.constants import DETECTION_ID_KEY
from roboflow_workflows.v2.blocks.geometry._rows import without_host_mirrors
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import (
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
)
from roboflow_workflows.v2.data.prediction_ops import select_predictions
from roboflow_workflows.v2.data.predictions import (
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import (
    RowRef,
    classification_parent,
    require_row_record,
)
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Group,
    Output,
    Ref,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


PREDICTION_KINDS = (
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

CLASS_NAMES_KEY = "class_names"

ROW_CLASS_NAME_KEY = "class"

STRING_CLASS_ID = 0

ClassResult = Tuple[str, int, float]

Child = Tuple[Tuple[int, ...], Any]

class DetectionsClassesReplacementBlock(Block):
    """Give each detection the class its crop was classified as.

    Example: detections ``[A, B, C]`` are filtered to ``[A, C]`` and cropped;
    crop 0 (of A) is classified ``"cat"``, crop 1 (of C) has no result, and the
    fallback is ``"unknown"``. Against ``[A, B, C]``: A becomes ``"cat"``, B and
    C ``"unknown"`` with confidence 0. Without a fallback only A remains.
    """

    type = "roboflow_core/detections_classes_replacement@v1"
    accepts_empty = True
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="object_detection_predictions",
            description="Rows with a classification or fallback, classes replaced.",
        ),
    }

    class Params(BlockParams):
        object_detection_predictions: Ref(*PREDICTION_KINDS) = Field(
            description="Predictions whose rows were cropped and classified."
        )
        classification_predictions: Group(
            CLASSIFICATION_PREDICTION_KIND, STRING_KIND, LIST_OF_VALUES_KIND
        ) = Field(
            description=(
                "One result per crop: a classification prediction, a string or "
                "a list of strings (e.g. OCR text). Results join rows by the "
                "row key of the crop they were made from."
            )
        )
        crops: Optional[Group(IMAGE_KIND)] = Field(
            default=None,
            description=(
                "The crops the results were made from, same nesting. Needed "
                "only for text results, which carry no row key themselves."
            ),
        )
        fallback_class_name: Optional[str] | Ref(STRING_KIND) = Field(
            default=None,
            description=(
                "Class for rows without a classification; null or empty drops "
                "those rows."
            ),
        )
        fallback_class_id: Optional[StrictInt] | Ref(INTEGER_KIND) = Field(
            default=None,
            description=(
                "Class id of the fallback; null or negative means sys.maxsize."
            ),
        )

    def run(
        self,
        *,
        object_detection_predictions: Any,
        classification_predictions: Optional[Batch],
        crops: Optional[Batch],
        fallback_class_name: Optional[str],
        fallback_class_id: Optional[int],
    ) -> Dict[str, Any]:
        """Replace the classes of ``object_detection_predictions``.

        Args:
            object_detection_predictions: ``Detections``,
                ``InstanceDetections`` or a ``(KeyPoints, Detections)`` tuple;
                ``None`` when the step had no predictions.
            classification_predictions: Results of the crops of these rows;
                ``None`` when the group is absent.
            crops: The crops the results were made from, or ``None``.
            fallback_class_name: Class for rows without a result, or ``None``.
            fallback_class_id: Id of the fallback class, or ``None``.

        Returns:
            ``predictions``: the kept rows with new classes; ``None`` when
            ``object_detection_predictions`` is ``None``.

        Raises:
            ContractError: When a result has no row key, a key of other
                predictions or the same key as another result, the
                predictions have no row identity, or keypoints come without
                boxes.
        """
        if object_detection_predictions is None:
            return {"predictions": None}

        if _boxes_of(object_detection_predictions) is None:
            key_points = object_detection_predictions[0]
            raise ContractError(
                f"{self.type}: the keypoint prediction has no bounding boxes "
                f"({len(key_points)} skeletons, Detections None), so it has no "
                "classes to replace; use a keypoint model that also returns boxes"
            )

        children = [
            (index, value)
            for index, value in (
                classification_predictions.iter_with_indices()
                if classification_predictions is not None
                else ()
            )
            if value is not None
        ]
        results = _row_results(
            object_detection_predictions,
            children=children,
            crops=crops,
            consumer=self.type,
        )

        fallback = _fallback(fallback_class_name, class_id=fallback_class_id)
        kept_rows, kept_results = [], []
        for row, row_result in enumerate(results):
            row_result = row_result or fallback
            if row_result is not None:
                kept_rows.append(row)
                kept_results.append(row_result)

        selected = select_predictions(object_detection_predictions, indices=kept_rows)
        result = {"predictions": _with_classes(selected, results=kept_results)}

        return result

def leading_class(prediction: Any) -> Optional[ClassResult]:
    """Return the ``(name, class_id, confidence)`` a classification stands for.

    Args:
        prediction: ``ClassificationPrediction``,
            ``MultiLabelClassificationPrediction``, ``str`` or ``list``.

    Returns:
        The leading class, or ``None`` when the prediction names none: an
        empty distribution, no predicted label, zero top multi-label
        confidence or a list without a leading string.

    Raises:
        TypeError: On any other value.
    """
    (result,) = leading_classes([prediction])

    return result

def leading_classes(predictions: Sequence[Any]) -> List[Optional[ClassResult]]:
    """Return the leading class of every prediction, as ``leading_class``.

    Class ids and confidences stay on their device until one host read per
    device for all predictions together.

    Args:
        predictions: Values accepted by ``leading_class``.

    Returns:
        One leading class or ``None`` per prediction, in order.

    Raises:
        TypeError: On an unsupported value.
    """
    results: List[Optional[ClassResult]] = [None] * len(predictions)
    pending: Dict[torch.device, List[Tuple[int, torch.Tensor, Any, bool]]] = {}
    for position, prediction in enumerate(predictions):
        if isinstance(prediction, str):
            results[position] = (prediction, STRING_CLASS_ID, 1.0)
            continue
        if isinstance(prediction, list):
            first = prediction[0] if prediction else None
            if isinstance(first, str):
                results[position] = (first, STRING_CLASS_ID, 1.0)
            continue

        pair = _leading_pair_on_device(prediction)
        if pair is not None:
            pending.setdefault(pair[0].device, []).append((position, *pair))

    for entries in pending.values():
        # Explicit host boundary: ids and confidences of every prediction on
        # this device, read together. float64 holds both exactly.
        values = torch.stack([pair for _, pair, _, _ in entries]).cpu().tolist()
        for (position, _, metadata, is_multi_label), (class_id, confidence) in zip(
            entries, values
        ):
            if is_multi_label and confidence == 0:
                # V1: zero confidence means no label.
                continue
            class_id = int(class_id)
            name = _names(metadata).get(class_id, f"class_{class_id}")
            results[position] = (name, class_id, confidence)

    return results

def _leading_pair_on_device(
    prediction: Any,
) -> Optional[Tuple[torch.Tensor, Any, bool]]:
    """``([class_id, confidence]`` float64 on the device, metadata, multi-label)."""
    if isinstance(prediction, ClassificationPrediction):
        if prediction.confidence.shape[-1] == 0:
            return None
        class_id = prediction.class_id[:1].to(torch.int64)
        confidence = prediction.confidence[0].gather(0, class_id)
        metadata = prediction.images_metadata[0] if prediction.images_metadata else None
        is_multi_label = False
    elif isinstance(prediction, MultiLabelClassificationPrediction):
        if len(prediction.class_ids) == 0 or prediction.confidence.numel() == 0:
            return None
        # V1: once any label is predicted, the most confident class of the
        # whole distribution wins (first on ties).
        scores = prediction.confidence.reshape(-1)
        class_id = torch.argmax(scores).reshape(1)
        confidence = scores.gather(0, class_id)
        metadata = prediction.image_metadata
        is_multi_label = True
    else:
        raise TypeError(
            "classification_predictions must hold classification predictions, "
            f"strings or lists, got {type(prediction).__name__}"
        )

    pair = torch.cat([class_id.to(torch.float64), confidence.to(torch.float64)])

    return pair, metadata, is_multi_label

def _row_results(
    targets: Any, *, children: List[Child], crops: Optional[Batch], consumer: str
) -> List[Optional[ClassResult]]:
    """The joined result of every target row; ``None`` where there is none."""
    row_count = len(_boxes_of(targets).xyxy)
    if row_count == 0 or not children:
        return [None] * row_count

    # Targets first: without their identity no result can join, whatever it is.
    record = require_row_record(targets, consumer=consumer)
    keys = _result_keys(children, crops=crops)
    _check_keys(children, keys=keys, namespace=record.namespace)
    leading = leading_classes([value for _, value in children])
    result_by_id = {key.id: result for key, result in zip(keys, leading)}
    results = [result_by_id.get(row_id) for row_id in record.host_ids()]

    return results

def _result_keys(children: List[Child], *, crops: Optional[Batch]) -> List[RowRef]:
    crop_by_index = dict(crops.iter_with_indices()) if crops is not None else {}
    keys = []
    for index, value in children:
        crop = crop_by_index.get(index)
        crop_key = crop.parent_row if crop is not None else None
        is_classification = isinstance(
            value, (ClassificationPrediction, MultiLabelClassificationPrediction)
        )
        own_key = classification_parent(value) if is_classification else None
        if own_key is not None and crop_key is not None and own_key != crop_key:
            raise ContractError(
                f"classification at index {index} was made for row {own_key}, but "
                f"the crop bound at that index was cut for row {crop_key}; bind "
                "the crops the classifications were made from"
            )

        key = own_key or crop_key
        if key is None:
            raise ContractError(_missing_key_message(index, value, crop=crop))
        keys.append(key)

    return keys

def _missing_key_message(index: Tuple[int, ...], value: Any, *, crop: Any) -> str:
    if crop is not None:
        message = (
            f"the crop at index {index} has no parent_row: the predictions it was "
            "cut from carry no row identity, so its result cannot be joined"
        )
    elif isinstance(value, (str, list)):
        message = (
            f"the text result at index {index} carries no detection row; bind "
            "'crops' to the crops it was read from"
        )
    else:
        message = (
            f"classification at index {index} carries no detection row; classify "
            "crops made by roboflow_core/dynamic_crop@v1 from these predictions, or bind "
            "'crops'. A classifier block records the row with "
            "with_classification_parent(prediction, image.parent_row)"
        )

    return message

def _check_keys(children: List[Child], *, keys: List[RowRef], namespace: str) -> None:
    seen: Dict[RowRef, Tuple[int, ...]] = {}
    for (index, _), key in zip(children, keys):
        if key.namespace != namespace:
            raise ContractError(
                f"result at index {index} belongs to rows of other predictions "
                f"({key.namespace!r}, not {namespace!r}); bind results of crops "
                "made from these predictions"
            )
        if key in seen:
            raise ContractError(
                f"results at indices {seen[key]} and {index} are both for row "
                f"{key.id}; each row takes at most one result"
            )
        seen[key] = index

def _names(metadata: Optional[Dict[str, Any]]) -> Dict[int, str]:
    names = (metadata or {}).get(CLASS_NAMES_KEY) or {}
    if not isinstance(names, dict):
        names = dict(enumerate(names))

    return names

def _fallback(name: Optional[str], *, class_id: Optional[int]) -> Optional[ClassResult]:
    if not name:
        return None
    if class_id is None or class_id < 0:
        class_id = sys.maxsize

    return name, class_id, 0.0

def _boxes_of(prediction: Any) -> Any:
    boxes = prediction[1] if isinstance(prediction, tuple) else prediction

    return boxes

def _with_classes(prediction: Any, *, results: List[ClassResult]) -> Any:
    detections = _boxes_of(prediction)
    names = [name for name, _, _ in results]
    class_ids = [class_id for _, class_id, _ in results]
    class_names: Dict[int, str] = {}
    for name, class_id in zip(names, class_ids):
        class_names.setdefault(class_id, name)

    rows = detections.bboxes_metadata or [{} for _ in results]
    renamed = replace(
        detections,
        class_id=torch.tensor(
            class_ids, dtype=torch.int64, device=detections.class_id.device
        ),
        confidence=torch.tensor(
            [confidence for _, _, confidence in results],
            dtype=detections.confidence.dtype,
            device=detections.confidence.device,
        ),
        image_metadata={
            **(detections.image_metadata or {}),
            CLASS_NAMES_KEY: class_names,
        },
        bboxes_metadata=[
            {
                **without_host_mirrors(row),
                ROW_CLASS_NAME_KEY: name,
                DETECTION_ID_KEY: str(uuid.uuid4()),
            }
            for row, name in zip(rows, names)
        ],
    )
    if isinstance(prediction, tuple):
        return prediction[0], renamed

    return renamed
