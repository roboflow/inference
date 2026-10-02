"""Explicit media boundaries for wildcard and legacy ``numpy_array`` ports.

The wildcard leaves native values and ordinary containers untouched at ingress.
At serialization, known native families use their typed Kind codec inside a
``{"__workflows_v2_media__": 1, "kind": ..., "value": ...}``
envelope. Only that reserved tag triggers decoding; dictionaries resembling V1
predictions, images or model configuration remain ordinary dictionaries.

Coordinate output conversion delegates to the same typed hooks before walking
ordinary containers. In particular, prediction tuples and their metadata stay
one native value. ``own`` preserves identity; ``parent`` and ``root`` use the
typed prediction hook's workflow-root semantics and composite-root rejection.

``numpy_array`` is a compatibility label for V1 tensor-native depth outputs.
Numeric NumPy ingress becomes a tensor once. Internal values, validation and
wire encoding then follow ``TENSOR_KIND``; new declarations should use ``tensor``.
"""

import math
from numbers import Integral, Real
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    BAR_CODE_DETECTION_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    NATIVE_KINDS,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    TENSOR_KIND,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.kinds import Kind

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

_MEDIA_WIRE_TAG = "__workflows_v2_media__"
_MEDIA_WIRE_VERSION = 1
_MEDIA_WIRE_FIELDS = {_MEDIA_WIRE_TAG, "kind", "value"}
_NATIVE_KINDS_BY_NAME = {kind.name: kind for kind in (IMAGE_KIND, *NATIVE_KINDS)}


def _native_kind(value: Any) -> Optional[Kind]:
    if isinstance(value, ImageData):
        return IMAGE_KIND
    if isinstance(value, torch.Tensor):
        return TENSOR_KIND
    if isinstance(
        value, (ClassificationPrediction, MultiLabelClassificationPrediction)
    ):
        return CLASSIFICATION_PREDICTION_KIND
    if isinstance(value, (InstanceDetections, Detections)):
        metadata = value.image_metadata
        prediction_type = (
            metadata.get("prediction_type") if isinstance(metadata, Mapping) else None
        )
        if isinstance(value, InstanceDetections):
            if prediction_type == "semantic-segmentation":
                return SEMANTIC_SEGMENTATION_PREDICTION_KIND
            if isinstance(value.mask, InstancesRLEMasks):
                return RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND

            return INSTANCE_SEGMENTATION_PREDICTION_KIND
        if prediction_type == "qrcode-detection":
            return QR_CODE_DETECTION_KIND
        if prediction_type == "barcode-detection":
            return BAR_CODE_DETECTION_KIND

        return OBJECT_DETECTION_PREDICTION_KIND
    if isinstance(value, tuple):
        if len(value) == 2 and isinstance(value[0], KeyPoints):
            return KEYPOINT_DETECTION_PREDICTION_KIND
        if (
            len(value) == 7
            and isinstance(value[0], torch.Tensor)
            and isinstance(value[2], (torch.Tensor, Integral))
            and isinstance(value[3], (torch.Tensor, Real))
            and isinstance(value[5], Mapping)
            and isinstance(value[6], Mapping)
        ):
            return DETECTION_KIND

    return None


def _map_container(value: Any, *, transform: Callable[[Any], Any]) -> Any:
    """Transform container children while preserving unchanged object identity."""
    if isinstance(value, Mapping):
        mapped = {key: transform(item) for key, item in value.items()}
        if all(mapped[key] is item for key, item in value.items()):
            return value

        return mapped
    if isinstance(value, (list, tuple)):
        mapped = [transform(item) for item in value]
        if all(result is item for result, item in zip(mapped, value)):
            return value
        if isinstance(value, tuple):
            mapped = tuple(mapped)

        return mapped

    return value


def _media_from_value(value: Any) -> Any:
    if _native_kind(value) is not None:
        return value
    if isinstance(value, Mapping) and _MEDIA_WIRE_TAG in value:
        if set(value) != _MEDIA_WIRE_FIELDS:
            raise ContractError(
                "A media envelope needs exactly __workflows_v2_media__, kind and value fields"
            )
        if (
            type(value[_MEDIA_WIRE_TAG]) is not int
            or value[_MEDIA_WIRE_TAG] != _MEDIA_WIRE_VERSION
        ):
            raise ContractError(
                f"Unsupported media envelope version: {value[_MEDIA_WIRE_TAG]!r}"
            )
        name = value["kind"]
        if not isinstance(name, str) or name not in _NATIVE_KINDS_BY_NAME:
            raise ContractError(f"Unsupported native media kind in envelope: {name!r}")

        kind = _NATIVE_KINDS_BY_NAME[name]
        payload = kind.to_payload(value["value"])
        kind.check(payload)

        return payload

    decoded = _map_container(value, transform=_media_from_value)

    return decoded


def _media_to_serialized(value: Any) -> Any:
    kind = _native_kind(value)
    if kind is not None:
        kind.check(value)
        serialized = {
            _MEDIA_WIRE_TAG: _MEDIA_WIRE_VERSION,
            "kind": kind.name,
            "value": kind.to_serialized(value),
        }

        return serialized
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContractError("A JSON scalar must be finite at the media boundary")

        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ContractError("Media wildcard JSON mappings require string keys")

        serialized = {key: _media_to_serialized(item) for key, item in value.items()}

        return serialized
    if isinstance(value, (list, tuple)):
        serialized = [_media_to_serialized(item) for item in value]

        return serialized

    raise ContractError(
        f"Cannot serialize {type(value).__name__} through the media wildcard; "
        "use a declared Kind with a serializer for this payload"
    )


def _media_to_output(value: Any, options: Mapping[str, Any]) -> Any:
    kind = _native_kind(value)
    if kind is not None:
        converted = kind.to_output(value, options=options)

        return converted

    converted = _map_container(
        value, transform=lambda item: _media_to_output(item, options)
    )

    return converted


def _numpy_array_from_value(value: Any) -> torch.Tensor:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ContractError(
                f"numpy_array requires a real numeric array, got dtype {value.dtype}"
            )

        array = value
        if not array.dtype.isnative:
            array = array.astype(array.dtype.newbyteorder("="))
        if not array.flags.c_contiguous or not array.flags.writeable:
            array = array.copy(order="C")

        try:
            value = torch.from_numpy(array)
        except (TypeError, ValueError) as error:
            raise ContractError(
                f"Cannot convert numpy_array dtype {value.dtype} to a tensor: {error}"
            ) from error

    tensor = TENSOR_KIND.to_payload(value)
    TENSOR_KIND.check(tensor)

    return tensor


MEDIA_WILDCARD_KIND = Kind(
    name="*",
    description="Native media and ordinary JSON values at explicit workflow boundaries.",
    deserialize=_media_from_value,
    serialize=_media_to_serialized,
    convert_output=_media_to_output,
)
NUMPY_ARRAY_KIND = Kind(
    name="numpy_array",
    description=(
        "Compatibility label for V1 tensor-native depth: real NumPy input becomes "
        "a torch.Tensor; native tensors and tensor wire encoding are preserved. "
        "Use tensor for new V2 declarations."
    ),
    validate=TENSOR_KIND.validate,
    deserialize=_numpy_array_from_value,
    serialize=TENSOR_KIND.serialize,
)

__all__ = ["MEDIA_WILDCARD_KIND", "NUMPY_ARRAY_KIND"]
