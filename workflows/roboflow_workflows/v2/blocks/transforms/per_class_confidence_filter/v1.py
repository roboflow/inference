"""Keep detections above a per-class confidence: ``roboflow_core/per_class_confidence_filter@v1``.

Port of V1 ``roboflow_core/per_class_confidence_filter@v1``::

    name      = row metadata "class" if it is a string (row_class_names)
                else  image_metadata["class_names"][class_id]  (dict or list)
    threshold = class_thresholds[name]  else  default_threshold
    keep row  if  confidence >= threshold

Host work per call: ``class_id`` is read once (class names are host
metadata), the per-row thresholds go to the device once, and the comparison
and row gather run on the device. Rows keep their masks, keypoints, row
metadata and row record; a ``SegmentationView`` is never upscaled.
"""

import math
from typing import Any, Dict, Mapping
import torch
from pydantic import Field, StrictFloat, StrictInt, field_validator
from roboflow_workflows.v2.blocks.query.detections import (
    DetectionTable,
    boxes_of,
    row_count,
    select_rows,
)
from roboflow_workflows.v2.blocks.query.errors import OperationFailure
from roboflow_workflows.v2.blocks.transforms.carriers import PREDICTION_KINDS
from roboflow_workflows.v2.data.kinds import DICTIONARY_KIND, FLOAT_KIND
from roboflow_workflows.v2.data.prediction_ops import (
    RowSelection,
    class_names_by_id,
    row_class_names,
)
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref


class PerClassConfidenceFilterBlock(Block):
    """Keep each detection whose confidence reaches its class's threshold.

    Example: ``class_thresholds={"person": 0.9}``, ``default_threshold=0.5``
    keeps a person at 0.95 and a car at 0.5, and drops a person at 0.8.
    """

    type = "roboflow_core/per_class_confidence_filter@v1"
    metadata = {
        "name": "Per-Class Confidence Filter",
        "port_of": "roboflow_core/per_class_confidence_filter@v1",
        "section": "flow_control",
    }
    outputs = {
        "predictions": Output(
            *PREDICTION_KINDS,
            source="predictions",
            description="The rows that pass, in input order; same carrier type.",
        ),
    }

    class Params(BlockParams):
        predictions: Ref(*PREDICTION_KINDS) = Field(
            description=(
                "Detections, instance segmentation (dense, RLE or view), keypoint "
                "predictions with boxes, or prediction rows."
            ),
            examples=["$steps.object_detection_model.predictions"],
        )
        class_thresholds: Dict[str, StrictFloat | StrictInt] | Ref(DICTIONARY_KIND) = (
            Field(
                description=(
                    "Minimum confidence per class name, each in [0, 1]. Other "
                    "classes use default_threshold."
                ),
                examples=[{"person": 0.9, "car": 0.5}, "$inputs.class_thresholds"],
            )
        )
        default_threshold: StrictFloat | StrictInt | Ref(FLOAT_KIND) = Field(
            default=0.3,
            description=(
                "Minimum confidence, in [0, 1], for classes not listed in "
                "class_thresholds and for rows without a class name."
            ),
            examples=[0.3, "$inputs.default_threshold"],
        )

        @field_validator("class_thresholds")
        @classmethod
        def _literal_thresholds_in_range(cls, value: Any) -> Any:
            if isinstance(value, dict):
                checked_thresholds(value)

            return value

        @field_validator("default_threshold")
        @classmethod
        def _literal_default_in_range(cls, value: Any) -> Any:
            if isinstance(value, (int, float)):
                checked_threshold(value, name="default_threshold")

            return value

    def run(
        self,
        *,
        predictions: Any,
        class_thresholds: Mapping[str, float],
        default_threshold: float,
    ) -> Dict[str, Any]:
        """Filter one prediction by per-class confidence.

        Args:
            predictions: A carrier listed in ``transforms.carriers``.
            class_thresholds: Threshold by class name.
            default_threshold: Threshold for other rows.

        Returns:
            ``predictions``: kept rows in input order; the input object
            itself when every row is kept or there are no rows.

        Raises:
            ValueError: On a threshold that is not a number in [0, 1], or
                keypoints without boxes.
        """
        filtered = keep_confident_rows(
            predictions,
            class_thresholds=checked_thresholds(class_thresholds),
            default_threshold=checked_threshold(
                default_threshold, name="default_threshold"
            ),
        )

        return {"predictions": filtered}

def keep_confident_rows(
    predictions: Any, *, class_thresholds: Mapping[str, float], default_threshold: float
) -> Any:
    """Return the rows whose confidence reaches their class's threshold.

    Args:
        predictions: A carrier listed in ``transforms.carriers``.
        class_thresholds: Validated threshold by class name.
        default_threshold: Validated threshold for other rows.

    Returns:
        The kept rows, same carrier type; ``predictions`` itself when all
        rows are kept or there are none.

    Raises:
        ValueError: For keypoints without boxes.
    """
    try:
        boxes_of(predictions)
    except OperationFailure as error:
        raise ValueError(f"roboflow_core/per_class_confidence_filter@v1: {error}") from None

    count = row_count(predictions)
    if count == 0:
        return predictions

    table = DetectionTable(predictions)
    names = row_class_names(
        table.column("class_id"),
        row_metadata=table.row_metadata,
        names_by_id=class_names_by_id(table.image_metadata),
    )
    limits = []
    for name in names:
        limit = default_threshold if name is None else class_thresholds.get(str(name))
        limits.append(default_threshold if limit is None else limit)

    confidence = table.boxes.confidence
    keep = confidence >= torch.tensor(
        limits, dtype=confidence.dtype, device=confidence.device
    )
    if bool(keep.all()):
        return predictions

    kept = select_rows(predictions, RowSelection.of(count, mask=keep))

    return kept

def checked_thresholds(value: Any) -> Dict[str, float]:
    """Validate a class-name-to-threshold mapping.

    Args:
        value: The mapping from parameters or a selector.

    Returns:
        A new dict of ``float`` thresholds.

    Raises:
        ValueError: When ``value`` is not a dict with string keys and
            numbers in [0, 1].
    """
    if not isinstance(value, Mapping):
        raise ValueError(
            "class_thresholds must be a dictionary of class name to threshold, "
            f"got {type(value).__name__}"
        )

    thresholds = {}
    for name, threshold in value.items():
        if not isinstance(name, str):
            raise ValueError(f"class_thresholds keys must be class names, got {name!r}")
        thresholds[name] = checked_threshold(
            threshold, name=f"class_thresholds[{name!r}]"
        )

    return thresholds

def checked_threshold(value: Any, *, name: str) -> float:
    """Return ``value`` as a float after checking it is a number in [0, 1].

    Args:
        value: Candidate threshold.
        name: Parameter name for the error.

    Returns:
        The threshold as ``float``.

    Raises:
        ValueError: For booleans, non-numbers, NaN and values outside [0, 1].
    """
    is_number = isinstance(value, (int, float)) and not isinstance(value, bool)
    if not is_number or math.isnan(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")

    threshold = float(value)

    return threshold
