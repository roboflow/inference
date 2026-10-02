"""Strict row selection for native V2 detection and keypoint payloads."""

from numbers import Integral
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from roboflow_workflows.core_steps.common.tensor_native import (
    take_prediction_by_indices,
    take_prediction_by_mask,
)
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections

Selection = Union[torch.Tensor, np.ndarray, Sequence]


def select_predictions(
    prediction: Any,
    *,
    mask: Optional[Selection] = None,
    indices: Optional[Selection] = None,
) -> Any:
    """Select aligned native rows with exactly one mask or ordered index list.

    Args:
        prediction: Detections, InstanceDetections, KeyPoints, or a tuple of
            KeyPoints and optional Detections. Fields are validated first.
        mask: One-dimensional boolean selection with exactly one entry per row.
        indices: One-dimensional integer positions; may reorder or repeat rows.
            Negative and out-of-range positions are rejected.

    Returns:
        A carrier of the original family with all tensor fields, masks, row IDs
        and metadata selected together. Identity selections may share tensors;
        selected row dictionaries are copied and image metadata is shared.
        Tensor fields remain on their device. Selecting Python metadata or RLE
        rows transfers only selected indices to the host; the existing gather
        helper also synchronizes its all-true mask check.

    Raises:
        ContractError: If the payload or selection is malformed, unsupported,
            or both/neither selection forms are supplied.
    """
    if (mask is None) == (indices is None):
        raise ContractError("Provide exactly one of mask or indices")

    rows = _validated_row_count(prediction)
    if mask is not None:
        _check_mask(mask, rows=rows)
        selected = take_prediction_by_mask(prediction, mask)
    else:
        positions = _checked_indices(indices, rows=rows)
        selected = take_prediction_by_indices(prediction, positions)

    return selected


def _validated_row_count(prediction: Any) -> int:
    if isinstance(prediction, Detections):
        OBJECT_DETECTION_PREDICTION_KIND.check(prediction)
        return len(prediction)

    if isinstance(prediction, InstanceDetections):
        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(prediction)
        return len(prediction)

    if isinstance(prediction, KeyPoints):
        KEYPOINT_DETECTION_PREDICTION_KIND.check((prediction, None))
        return len(prediction)

    if isinstance(prediction, tuple):
        KEYPOINT_DETECTION_PREDICTION_KIND.check(prediction)
        return len(prediction[0])

    raise ContractError(
        "select_predictions supports Detections, InstanceDetections, KeyPoints "
        f"or (KeyPoints, optional Detections), got {type(prediction).__name__}"
    )


def _check_mask(mask: Selection, *, rows: int) -> None:
    if isinstance(mask, (torch.Tensor, np.ndarray)):
        expected_dtype = torch.bool if isinstance(mask, torch.Tensor) else np.bool_
        if mask.ndim != 1 or mask.dtype != expected_dtype:
            raise ContractError(
                "mask must be one-dimensional with boolean dtype, "
                f"got shape {tuple(mask.shape)} and dtype {mask.dtype}"
            )
    elif not isinstance(mask, (list, tuple)) or not all(
        isinstance(value, (bool, np.bool_)) for value in mask
    ):
        raise ContractError("mask must be a one-dimensional boolean sequence")

    if len(mask) != rows:
        raise ContractError(
            f"mask has {len(mask)} entries; expected {rows} prediction rows"
        )


def _checked_indices(indices: Selection, *, rows: int) -> list:
    if isinstance(indices, (torch.Tensor, np.ndarray)):
        if indices.ndim != 1:
            raise ContractError("indices must be one-dimensional")

        if isinstance(indices, torch.Tensor):
            integer_dtypes = (
                torch.uint8,
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            )
            if indices.dtype not in integer_dtypes:
                raise ContractError(
                    f"indices require integer dtype, got {indices.dtype}"
                )

            indices = indices.detach().cpu().tolist()
        else:
            if indices.dtype.kind not in "iu":
                raise ContractError(
                    f"indices require integer dtype, got {indices.dtype}"
                )

            indices = indices.tolist()

    if not isinstance(indices, (list, tuple)) or not all(
        isinstance(index, Integral) and not isinstance(index, (bool, np.bool_))
        for index in indices
    ):
        raise ContractError("indices must be a one-dimensional integer sequence")

    positions = [int(index) for index in indices]
    for index in positions:
        if not 0 <= index < rows:
            raise ContractError(f"prediction index {index} is outside [0, {rows})")

    return positions
