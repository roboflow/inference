"""Representation boundary for custom Python (dynamic) blocks.

Dynamic blocks are compiled per-workflow from user code the engine cannot
rewrite, so the sibling-file swap pattern of the tensor pivot cannot apply to
them. Instead, blocks declaring ``tensor_compatibility=legacy_compatibility``
(the default) get a conversion boundary around their ``run()``: native
``inference_models`` objects are converted into the documented legacy
``sv.Detections`` / numpy representations at the input boundary, and the
returned legacy objects are converted back at the output boundary.

This module hosts the boundary. Step 2 of the plan ships the IN direction
(``convert_kwargs_to_legacy`` and the per-kind native -> legacy converters)
plus the shared batch/value walking and sniffing machinery; the OUT direction
(``convert_block_result_to_native``, legacy -> native) lands in Step 3 and
slots into the same dispatch structure.

The whole boundary is an IDENTITY when ``ENABLE_TENSOR_DATA_REPRESENTATION``
is off — resolved once at import time into ``_TENSOR_REPRESENTATION_ACTIVE``
(tests patch that constant) — which is the flag-off byte-parity guarantee of
the tensor pivot.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import numpy as np
import supervision as sv
import torch

from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
)
from inference.core.workflows.errors import DynamicBlockError
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    ManifestDescription,
    TensorCompatibility,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

# Resolved once at import time, mirroring the pivot's load-time-swap philosophy.
# The boundary is a strict identity when the flag is off.
_TENSOR_REPRESENTATION_ACTIVE: bool = ENABLE_TENSOR_DATA_REPRESENTATION

# The sv.Detections.data column carrying the class-name string (supervision's own
# convention; the numpy serializer reads data["class_name"] literally).
CLASS_NAME_DATA_COLUMN = "class_name"

_BOUNDARY_CONTEXT = (
    "workflow_execution | step_execution | dynamic_block_representation_boundary"
)

# Kind names whose runtime representation differs between the numpy and tensor
# paths — exactly the kinds whose serializers/deserializers are flag-swapped in
# `core_steps/loader.py`. The kind NAME is the stable conversion key (same-name
# kinds by design). All other kinds are representation-invariant.
_DETECTIONS_FAMILY_KIND_NAMES = {
    "object_detection_prediction",
    "instance_segmentation_prediction",
    "rle_instance_segmentation_prediction",
    "semantic_segmentation_prediction",
    "qr_code_detection",
    "bar_code_detection",
}
_KEYPOINT_KIND_NAME = "keypoint_detection_prediction"
_CLASSIFICATION_KIND_NAME = "classification_prediction"
_EMBEDDING_KIND_NAME = "embedding"
_TENSOR_KIND_NAME = "tensor"
_IMAGE_KIND_NAME = "image"
CONVERTIBLE_KIND_NAMES = _DETECTIONS_FAMILY_KIND_NAMES | {
    _KEYPOINT_KIND_NAME,
    _CLASSIFICATION_KIND_NAME,
    _EMBEDDING_KIND_NAME,
    _TENSOR_KIND_NAME,
    _IMAGE_KIND_NAME,
}

_NativeDetections = (Detections, InstanceDetections)


class RepresentationBoundaryError(DynamicBlockError):
    """Raised when a value cannot cross the dynamic-block representation
    boundary — carries the block name, the input/output name, the offending
    type and a remediation hint, all baked into the public message."""

    def __init__(
        self,
        public_message: str,
        context: str,
        inner_error: Optional[Exception] = None,
        block_name: Optional[str] = None,
        value_name: Optional[str] = None,
        offending_type: Optional[type] = None,
    ):
        super().__init__(
            public_message=public_message,
            context=context,
            inner_error=inner_error,
        )
        self.block_name = block_name
        self.value_name = value_name
        self.offending_type = offending_type


def _raise_boundary_error(
    block_name: str,
    value_name: str,
    value: Any,
    problem: str,
    remediation: str,
) -> None:
    offending_type = type(value)
    raise RepresentationBoundaryError(
        public_message=(
            f"Dynamic block `{block_name}`, input/output `{value_name}`: {problem} "
            f"(offending type: `{offending_type.__module__}.{offending_type.__qualname__}`). "
            f"{remediation}"
        ),
        context=_BOUNDARY_CONTEXT,
        block_name=block_name,
        value_name=value_name,
        offending_type=offending_type,
    )


def convert_kwargs_to_legacy(
    kwargs: Dict[str, Any],
    manifest_description: Optional[ManifestDescription],
    block_name: str,
) -> Dict[str, Any]:
    """IN boundary: convert native tensor objects in a dynamic block's assembled
    kwargs into the documented legacy representations, driven by declared kinds
    with best-effort sniffing for wildcard inputs.

    Identity function (the very same ``kwargs`` object) when the tensor
    representation is off or the block declared ``tensor_native``.
    """
    if not _TENSOR_REPRESENTATION_ACTIVE:
        return kwargs
    if (
        manifest_description is not None
        and manifest_description.tensor_compatibility
        is TensorCompatibility.TENSOR_NATIVE
    ):
        return kwargs
    declared_inputs = manifest_description.inputs if manifest_description else {}
    converted = {}
    for name, value in kwargs.items():
        declared_kinds = _collect_declared_kind_names(declared_inputs.get(name))
        converted[name] = _walk_value_to_legacy(
            value=value,
            declared_kinds=declared_kinds,
            block_name=block_name,
            value_name=name,
        )
    return converted


def _collect_declared_kind_names(input_definition: Optional[Any]) -> set:
    """Union of kind names across the input's selector types. Empty set means
    wildcard (no declaration ⇒ sniffing)."""
    if input_definition is None:
        return set()
    selector_data_kind = getattr(input_definition, "selector_data_kind", None) or {}
    declared = set()
    for kind_names in selector_data_kind.values():
        declared.update(kind_names)
    declared.discard("*")
    return declared


def _walk_value_to_legacy(
    value: Any,
    declared_kinds: set,
    block_name: str,
    value_name: str,
) -> Any:
    """Walk containers (Batch preserving indices, dict/list/plain tuple) down to
    leaf values and convert each leaf. The keypoint-detection tuple is a LEAF,
    not a container — checked before generic tuple recursion."""
    if value is None:
        return None
    if isinstance(value, Batch):
        converted_content = [
            _walk_value_to_legacy(
                value=element,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
            for element in value
        ]
        return Batch.init(content=converted_content, indices=value.indices)
    if _is_key_point_prediction_tuple(value):
        return _convert_leaf_to_legacy(
            value=value,
            declared_kinds=declared_kinds,
            block_name=block_name,
            value_name=value_name,
        )
    if isinstance(value, dict):
        return {
            key: _walk_value_to_legacy(
                value=element,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
            for key, element in value.items()
        }
    if isinstance(value, (list, tuple)):
        converted = [
            _walk_value_to_legacy(
                value=element,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
            for element in value
        ]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return _convert_leaf_to_legacy(
        value=value,
        declared_kinds=declared_kinds,
        block_name=block_name,
        value_name=value_name,
    )


def _convert_leaf_to_legacy(
    value: Any,
    declared_kinds: set,
    block_name: str,
    value_name: str,
) -> Any:
    convertible_declared = declared_kinds & CONVERTIBLE_KIND_NAMES
    if convertible_declared:
        return _convert_by_declared_kinds(
            value=value,
            declared_kinds=convertible_declared,
            block_name=block_name,
            value_name=value_name,
        )
    return _sniff_convert_to_legacy(
        value=value,
        block_name=block_name,
        value_name=value_name,
    )


def _convert_by_declared_kinds(
    value: Any,
    declared_kinds: set,
    block_name: str,
    value_name: str,
) -> Any:
    """Kind-driven conversion: the declared kind names pick the converter; a value
    whose type does not match any declared kind's native representation fails
    loudly (the producer/consumer contract is broken upstream)."""
    if _IMAGE_KIND_NAME in declared_kinds and isinstance(value, WorkflowImageData):
        # Images need no conversion in either direction: WorkflowImageData is
        # lazily dual-representation.
        return value
    if declared_kinds & _DETECTIONS_FAMILY_KIND_NAMES and isinstance(
        value, _NativeDetections
    ):
        return native_detections_to_sv(detections=value)
    if _KEYPOINT_KIND_NAME in declared_kinds and _is_key_point_prediction_tuple(value):
        return native_key_point_prediction_to_sv(
            prediction=value, block_name=block_name, value_name=value_name
        )
    if _CLASSIFICATION_KIND_NAME in declared_kinds and isinstance(
        value, (ClassificationPrediction, MultiLabelClassificationPrediction)
    ):
        return serialise_native_classification(prediction=value)
    if _EMBEDDING_KIND_NAME in declared_kinds and isinstance(value, torch.Tensor):
        # numpy embedding blocks (clip / perception_encoder) emit List[float]
        # in-memory (`predictions.embeddings[0]`).
        return value.detach().cpu().reshape(-1).tolist()
    if _TENSOR_KIND_NAME in declared_kinds and isinstance(value, torch.Tensor):
        # The `tensor` kind has no numpy-path producer; ndarray is the legacy
        # in-memory equivalent numpy user code can operate on.
        return value.detach().cpu().numpy()
    if _is_representation_invariant(value):
        # Declared kinds also admit static values / already-legacy payloads
        # (e.g. a plain list fed to an embedding input) — those pass through.
        return value
    _raise_boundary_error(
        block_name=block_name,
        value_name=value_name,
        value=value,
        problem=(
            f"value does not match the native representation of the declared "
            f"kind(s) {sorted(declared_kinds)}"
        ),
        remediation=(
            "Check the upstream step's output kind, or remove the kind "
            "declaration to enable best-effort conversion."
        ),
    )


def _sniff_convert_to_legacy(
    value: Any,
    block_name: str,
    value_name: str,
) -> Any:
    """Wildcard best-effort: convert the known native prediction types, pass
    representation-invariant values through, and fail loudly on tensor-only
    values with no legacy equivalent."""
    if isinstance(value, _NativeDetections):
        return native_detections_to_sv(detections=value)
    if _is_key_point_prediction_tuple(value):
        return native_key_point_prediction_to_sv(
            prediction=value, block_name=block_name, value_name=value_name
        )
    if isinstance(
        value, (ClassificationPrediction, MultiLabelClassificationPrediction)
    ):
        return serialise_native_classification(prediction=value)
    if isinstance(value, KeyPoints):
        _raise_boundary_error(
            block_name=block_name,
            value_name=value_name,
            value=value,
            problem="a bare `KeyPoints` prediction (without its bounding-box "
            "component) has no legacy sv.Detections equivalent",
            remediation="Pass the full keypoint-detection prediction, or switch "
            "the block to `tensor_compatibility=tensor_native`.",
        )
    if isinstance(value, torch.Tensor):
        _raise_boundary_error(
            block_name=block_name,
            value_name=value_name,
            value=value,
            problem="a bare torch.Tensor has no unambiguous legacy equivalent",
            remediation="Declare the input's kind (e.g. `embedding` or `tensor`) "
            "or switch the block to `tensor_compatibility=tensor_native`.",
        )
    if _is_representation_invariant(value):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        _raise_boundary_error(
            block_name=block_name,
            value_name=value_name,
            value=value,
            problem="unrecognised tensor-native dataclass cannot be converted "
            "to a legacy representation",
            remediation="Declare the input's kind or switch the block to "
            "`tensor_compatibility=tensor_native`.",
        )
    # D3 fallthrough — INTENTIONAL representation-invariant passthrough. Anything
    # not caught above (arbitrary objects, pydantic models, NamedTuples, ...) is
    # assumed representation-agnostic and shipped as-is; the loud-error net covers
    # only the tensor-only shapes we can positively identify (native dataclasses,
    # bare tensors, bare KeyPoints).
    return value


def _is_representation_invariant(value: Any) -> bool:
    return isinstance(
        value, (WorkflowImageData, str, bytes, bool, int, float, np.ndarray)
    )


def _is_key_point_prediction_tuple(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], KeyPoints)
        and (value[1] is None or isinstance(value[1], _NativeDetections))
    )


def native_key_point_prediction_to_sv(
    prediction: Tuple[KeyPoints, Optional[Detections]],
    block_name: str,
    value_name: str,
) -> sv.Detections:
    """Convert the native keypoint-detection tuple via its bounding-box component
    (per-instance keypoint payloads already ride in ``bboxes_metadata`` under the
    sv-data key names)."""
    _, detections = prediction
    if detections is None:
        _raise_boundary_error(
            block_name=block_name,
            value_name=value_name,
            value=prediction,
            problem="keypoint prediction is missing the bounding-box component "
            "required to build the legacy sv.Detections",
            remediation="Use a keypoint model output that carries bounding boxes, "
            "or switch the block to `tensor_compatibility=tensor_native`.",
        )
    return native_detections_to_sv(detections=detections)


def native_detections_to_sv(
    detections: Union[Detections, InstanceDetections],
) -> sv.Detections:
    """Materialise a native ``Detections`` / ``InstanceDetections`` into the full
    legacy ``sv.Detections`` the numpy blocks operate on.

    Extends ``to_supervision_for_annotation`` semantics (visualizations
    ``base_tensor.py``) with what legacy USER CODE additionally relies on:

    * ``data['class_name']`` resolved effective-name / override-first — per-box
      ``bboxes_metadata['class']`` when present, else the
      ``image_metadata['class_names']`` map (the serializer's C1 rule),
    * per-box ``detection_id`` minted (uuid4) when missing — mirroring the numpy
      deserializer, never an empty string,
    * lineage broadcast into ``.data`` exactly as
      ``attach_parents_coordinates_to_sv_detections`` shapes it, plus
      ``prediction_type`` / ``image_dimensions`` / ``inference_id``,
    * masks ALWAYS dense ``(N, H, W)`` bool ndarray — RLE carriers are decoded
      (legacy user code indexes ``detections.mask``; the device->host + densify
      cost is inherent to running numpy user code),
    * per-box keypoint payloads padded into proper N-d arrays following the
      numpy ``add_inference_keypoints_to_sv_detections`` convention (ragged
      object arrays break supervision's ``is_data_equal``).
    """
    image_metadata = detections.image_metadata or {}
    detections_number = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is None:
        bboxes_metadata = [{} for _ in range(detections_number)]
    class_names_mapping = image_metadata.get(CLASS_NAMES_KEY) or {}
    # Single device→host sync for the three per-box tensors (class_id round-trips
    # through float32, which is exact for any realistic class count).
    packed = (
        torch.cat(
            [
                detections.xyxy.reshape(detections_number, 4).to(torch.float32),
                detections.confidence.reshape(detections_number, 1).to(torch.float32),
                detections.class_id.reshape(detections_number, 1).to(torch.float32),
            ],
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
    )
    xyxy = packed[:, :4].astype(np.float32)
    confidence = packed[:, 4].astype(np.float32)
    class_id = packed[:, 5].astype(int)
    mask = _materialise_dense_mask(detections=detections)
    tracker_id = _materialise_tracker_id(bboxes_metadata=bboxes_metadata)
    data: Dict[str, np.ndarray] = {}
    class_names = [
        _resolve_effective_class_name(
            per_box=bboxes_metadata[index],
            class_id=int(class_id[index]),
            class_names_mapping=class_names_mapping,
        )
        for index in range(detections_number)
    ]
    data[CLASS_NAME_DATA_COLUMN] = np.asarray(class_names, dtype=object)
    detection_ids = [
        str(per_box.get(DETECTION_ID_KEY) or uuid4()) for per_box in bboxes_metadata
    ]
    data[DETECTION_ID_KEY] = np.array(detection_ids)
    _broadcast_image_level_columns(
        data=data,
        image_metadata=image_metadata,
        detections_number=detections_number,
    )
    _attach_per_box_columns(
        data=data,
        bboxes_metadata=bboxes_metadata,
        detections_number=detections_number,
    )
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=mask,
        tracker_id=tracker_id,
        data=data,
    )


def _resolve_effective_class_name(
    per_box: dict,
    class_id: int,
    class_names_mapping: dict,
) -> str:
    if CLASS_NAME_KEY in per_box:
        return str(per_box[CLASS_NAME_KEY])
    class_name = class_names_mapping.get(class_id)
    if class_name is None:
        return f"class_{class_id}"
    return str(class_name)


def _materialise_dense_mask(
    detections: Union[Detections, InstanceDetections],
) -> Optional[np.ndarray]:
    if not isinstance(detections, InstanceDetections):
        return None
    mask = detections.mask
    if mask is None:
        return None
    if isinstance(mask, InstancesRLEMasks):
        return coco_rle_masks_to_numpy_mask(mask)
    # single bulk device->host transfer for the whole stack
    return mask.detach().cpu().numpy().astype(bool)


def _materialise_tracker_id(bboxes_metadata: List[dict]) -> Optional[np.ndarray]:
    tracker_ids = [per_box.get(TRACKER_ID_KEY) for per_box in bboxes_metadata]
    if not tracker_ids or any(tracker_id is None for tracker_id in tracker_ids):
        return None
    return np.asarray([int(tracker_id) for tracker_id in tracker_ids])


_IMAGE_LEVEL_ID_KEYS = (PARENT_ID_KEY, ROOT_PARENT_ID_KEY)
_IMAGE_LEVEL_PAIR_KEYS = (
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    IMAGE_DIMENSIONS_KEY,
)
_IMAGE_LEVEL_SCALAR_KEYS = (PREDICTION_TYPE_KEY, INFERENCE_ID_KEY)


def _broadcast_image_level_columns(
    data: Dict[str, np.ndarray],
    image_metadata: dict,
    detections_number: int,
) -> None:
    """Broadcast per-image metadata into per-row ``.data`` columns with the exact
    key names and shapes the numpy path produces: id/scalar keys as string arrays,
    coordinate/dimension keys as ``(n, 2)`` arrays (``[x, y]`` for coordinates,
    ``[h, w]`` for dimensions — the native metadata already stores them in those
    orders)."""
    for key in _IMAGE_LEVEL_ID_KEYS + _IMAGE_LEVEL_SCALAR_KEYS:
        value = image_metadata.get(key)
        if value is not None:
            data[key] = np.array([value] * detections_number)
    for key in _IMAGE_LEVEL_PAIR_KEYS:
        value = image_metadata.get(key)
        if value is not None:
            data[key] = np.array([list(value)] * detections_number).reshape(
                detections_number, 2
            )


_KEYPOINT_PAYLOAD_KEYS = (
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
)


def _attach_per_box_columns(
    data: Dict[str, np.ndarray],
    bboxes_metadata: List[dict],
    detections_number: int,
) -> None:
    extra_keys = set()
    for per_box in bboxes_metadata:
        extra_keys.update(per_box.keys())
    extra_keys.discard(DETECTION_ID_KEY)
    extra_keys.discard(TRACKER_ID_KEY)
    extra_keys.discard(CLASS_NAME_KEY)  # resolved into data['class_name'] already
    keypoint_keys_present = extra_keys & set(_KEYPOINT_PAYLOAD_KEYS)
    if keypoint_keys_present == set(_KEYPOINT_PAYLOAD_KEYS):
        _attach_padded_keypoint_columns(
            data=data,
            bboxes_metadata=bboxes_metadata,
            detections_number=detections_number,
        )
        extra_keys -= keypoint_keys_present
    for key in extra_keys:
        data[key] = np.asarray(
            [per_box.get(key) for per_box in bboxes_metadata], dtype=object
        )


def _attach_padded_keypoint_columns(
    data: Dict[str, np.ndarray],
    bboxes_metadata: List[dict],
    detections_number: int,
) -> None:
    """Pad per-box keypoint payloads into proper N-d arrays — the exact
    convention of ``add_inference_keypoints_to_sv_detections`` (utils.py): ragged
    object arrays break supervision's ``is_data_equal``."""
    keypoints_xy = [
        list(per_box.get(KEYPOINTS_XY_KEY_IN_SV_DETECTIONS) or [])
        for per_box in bboxes_metadata
    ]
    keypoints_confidence = [
        list(per_box.get(KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS) or [])
        for per_box in bboxes_metadata
    ]
    keypoints_class_id = [
        list(per_box.get(KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS) or [])
        for per_box in bboxes_metadata
    ]
    keypoints_class_name = [
        list(per_box.get(KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS) or [])
        for per_box in bboxes_metadata
    ]
    max_kps = max((len(kp) for kp in keypoints_xy), default=0)
    padded_xy = np.zeros((detections_number, max_kps, 2), dtype=np.float32)
    padded_conf = np.zeros((detections_number, max_kps), dtype=np.float32)
    padded_class_id = np.zeros((detections_number, max_kps), dtype=int)
    padded_class_name = np.full((detections_number, max_kps), "", dtype=object)
    for index in range(detections_number):
        kps_in_instance = len(keypoints_xy[index])
        if kps_in_instance > 0:
            padded_xy[index, :kps_in_instance] = keypoints_xy[index]
            padded_conf[index, :kps_in_instance] = keypoints_confidence[index]
            padded_class_id[index, :kps_in_instance] = keypoints_class_id[index]
            padded_class_name[index, :kps_in_instance] = keypoints_class_name[index]
    data[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = padded_xy
    data[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = padded_conf
    data[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = padded_class_id
    data[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = padded_class_name
