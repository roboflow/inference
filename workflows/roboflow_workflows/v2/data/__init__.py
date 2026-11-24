"""Shared V2 data contracts: batches, kinds, images, predictions and masks.

Blocks, the engine and hosts exchange these types. This package imports no
engine runtime and no block family::

    from roboflow_workflows.v2.data import Batch, ContractError, ImageData, Kind

Batch structure, kinds and ``ContractError`` load with the package. Image,
frame, prediction, row-selection and mask names load on first use, because
their modules import image and model-output libraries.
"""

import importlib
from typing import Any

from roboflow_workflows.v2.data.batch import (
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_TIME,
    AXIS_KINDS,
    SOURCE_TYPE_STATIC,
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    Index,
    InputValue,
    SampleContext,
    TemporalContext,
    TimeCoverage,
    TimeSpan,
    Timestamp,
    WorkflowsBuffer,
    validate_entry,
)
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.kinds import (
    BOOLEAN_KIND,
    BUILTIN_KINDS,
    DICTIONARY_KIND,
    FLOAT_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    WILDCARD_KIND,
    Kind,
    kinds_compatible,
    normalize_kinds,
)

_IMAGE = "roboflow_workflows.v2.data.image"
_IMAGE_KINDS = "roboflow_workflows.v2.data.image_kinds"
_PREDICTIONS = "roboflow_workflows.v2.data.predictions"
_MASKS = "roboflow_workflows.v2.data.masks.grid"
_BOUNDARIES = "roboflow_workflows.v2.data.boundaries"
_FRAMES = "roboflow_workflows.v2.data.frames"
_PREDICTION_OPS = "roboflow_workflows.v2.data.prediction_ops"
_ROWS = "roboflow_workflows.v2.data.rows"
_LAZY = {
    "FrameMapping": _IMAGE,
    "ImageData": _IMAGE,
    "FRAME_RECORD_KEY": _FRAMES,
    "FrameRef": _FRAMES,
    "frame_from_metadata": _FRAMES,
    "frame_of": _FRAMES,
    "CROP_SUMMARY_KIND": _IMAGE_KINDS,
    "IMAGE_KIND": _IMAGE_KINDS,
    "BAR_CODE_DETECTION_KIND": _PREDICTIONS,
    "CLASSIFICATION_PREDICTION_KIND": _PREDICTIONS,
    "DETECTION_KIND": _PREDICTIONS,
    "EMBEDDING_KIND": _PREDICTIONS,
    "INSTANCE_SEGMENTATION_PREDICTION_KIND": _PREDICTIONS,
    "KEYPOINT_DETECTION_PREDICTION_KIND": _PREDICTIONS,
    "NATIVE_KINDS": _PREDICTIONS,
    "NATIVE_SEMANTIC_SEGMENTATION_KIND": _PREDICTIONS,
    "OBJECT_DETECTION_PREDICTION_KIND": _PREDICTIONS,
    "QR_CODE_DETECTION_KIND": _PREDICTIONS,
    "RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND": _PREDICTIONS,
    "SEMANTIC_SEGMENTATION_PREDICTION_KIND": _PREDICTIONS,
    "TENSOR_KIND": _PREDICTIONS,
    "INSTANCE_SEGMENTATION_VIEW_KIND": _MASKS,
    "MaskGridGeometry": _MASKS,
    "SegmentationView": _MASKS,
    "MEDIA_WILDCARD_KIND": _BOUNDARIES,
    "NUMPY_ARRAY_KIND": _BOUNDARIES,
    "RowSelection": _PREDICTION_OPS,
    "select_predictions": _PREDICTION_OPS,
    "take_rows": _PREDICTION_OPS,
    "PREDICTION_ROWS_KIND": _ROWS,
    "PredictionRows": _ROWS,
}

__all__ = [
    "AXIS_KINDS",
    "AXIS_KIND_DYNAMIC_NESTING",
    "AXIS_KIND_SAMPLE",
    "AXIS_KIND_STATIC_NESTING",
    "AXIS_KIND_TIME",
    "Axis",
    "BOOLEAN_KIND",
    "BUILTIN_KINDS",
    "Batch",
    "ContractError",
    "DICTIONARY_KIND",
    "EntryLayout",
    "EntryMetadata",
    "FLOAT_KIND",
    "INTEGER_KIND",
    "Index",
    "InputValue",
    "Kind",
    "LIST_OF_VALUES_KIND",
    "SOURCE_TYPE_STATIC",
    "STRING_KIND",
    "SampleContext",
    "TemporalContext",
    "TimeCoverage",
    "TimeSpan",
    "Timestamp",
    "WILDCARD_KIND",
    "WorkflowsBuffer",
    "kinds_compatible",
    "normalize_kinds",
    "validate_entry",
    *_LAZY,
]


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(importlib.import_module(module), name)

    return value
