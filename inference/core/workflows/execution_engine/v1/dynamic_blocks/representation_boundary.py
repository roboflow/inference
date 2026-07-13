"""Representation boundary for custom Python (dynamic) blocks.

Dynamic blocks are compiled per-workflow from user code the engine cannot
rewrite, so the sibling-file swap pattern of the tensor pivot cannot apply to
them. Instead, blocks declaring ``tensor_compatibility=legacy_compatibility``
(the default) get a conversion boundary around their ``run()``: native
``inference_models`` objects are converted into the documented legacy
``sv.Detections`` / numpy representations at the input boundary, and the
returned legacy objects are converted back at the output boundary.

This module hosts both directions of the boundary: the IN direction
(``convert_kwargs_to_legacy`` — native -> legacy at the block's input) and the
OUT direction (``convert_block_result_to_native`` — legacy -> native over the
returned ``BlockResult``, ``FlowControl`` passing through untouched). Both are
kind-driven first (declared kinds pick the converter) with best-effort
sniffing for wildcard values.

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

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_IMAGE_TENSOR_DEVICE,
)
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    build_native_key_points,
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
    DynamicOutputDefinition,
    ManifestDescription,
    TensorCompatibility,
)
from inference.core.workflows.execution_engine.v1.entities import FlowControl
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

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
    declared_input_kinds: Optional[Dict[str, frozenset]] = None,
) -> Dict[str, Any]:
    """IN boundary: convert native tensor objects in a dynamic block's assembled
    kwargs into the documented legacy representations, driven by declared kinds
    with best-effort sniffing for wildcard inputs.

    Identity function (the very same ``kwargs`` object) when the tensor
    representation is off or the block declared ``tensor_native``.

    ``declared_input_kinds`` is the optional assembly-time precomputed lookup
    (see ``collect_declared_input_kind_names``); when omitted it is derived from
    the manifest here — same behavior, one extra computation per call.
    """
    if not _TENSOR_REPRESENTATION_ACTIVE:
        return kwargs
    if (
        manifest_description is not None
        and manifest_description.tensor_compatibility
        is TensorCompatibility.TENSOR_NATIVE
    ):
        return kwargs
    if declared_input_kinds is None:
        declared_input_kinds = (
            collect_declared_input_kind_names(manifest_description) or {}
        )
    converted = {}
    for name, value in kwargs.items():
        converted[name] = _walk_value_to_legacy(
            value=value,
            declared_kinds=declared_input_kinds.get(name, frozenset()),
            block_name=block_name,
            value_name=name,
        )
    return converted


def collect_declared_input_kind_names(
    manifest_description: Optional[ManifestDescription],
) -> Optional[Dict[str, frozenset]]:
    """Precompute the per-input declared-kind lookup once per block — the
    manifest is static, so callers (block assembly) hoist this out of run()."""
    if manifest_description is None:
        return None
    return {
        name: frozenset(_collect_declared_kind_names(definition))
        for name, definition in manifest_description.inputs.items()
    }


def collect_declared_output_kind_names(
    manifest_description: Optional[ManifestDescription],
) -> Optional[Dict[str, frozenset]]:
    """Per-output twin of ``collect_declared_input_kind_names``."""
    if manifest_description is None:
        return None
    return {
        name: frozenset(_collect_declared_output_kind_names(definition))
        for name, definition in manifest_description.outputs.items()
    }


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
        if isinstance(value, tuple):
            # Namedtuples take one positional argument PER FIELD - splat the
            # converted elements; a plain tuple takes the iterable whole.
            if hasattr(value, "_fields"):
                return type(value)(*converted)
            return type(value)(converted)
        return converted
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
        # Iterative densification: decode one instance at a time into a single
        # preallocated (N, H, W) output. The bulk pycocotools decode holds up to
        # three full-stack transients at peak ((H, W, N) uint8 + bool astype +
        # contiguous copy); per-instance decoding caps the transient at one mask.
        height, width = (int(value) for value in mask.image_size)
        dense = np.zeros((len(mask.masks), height, width), dtype=bool)
        for index in range(len(mask.masks)):
            dense[index] = coco_rle_masks_to_numpy_mask(
                InstancesRLEMasks(
                    image_size=mask.image_size,
                    masks=[mask.masks[index]],
                )
            )[0]
        return dense
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


# --------------------------------------------------------------------------- #
# OUT boundary: legacy -> native over the returned BlockResult (Step 3)       #
# --------------------------------------------------------------------------- #

# Same literal the classification tensor blocks attach; the tensor serializer
# reads it from image metadata to filter sub-threshold classes.
CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY = "classification_confidence_threshold"

RLE_MASK_DATA_COLUMN = "rle_mask"

# Kinds whose native instance-segmentation carrier should be RLE rather than a
# dense torch stack (the tensor path's compact-mask convention).
_RLE_CARRIER_KIND_NAMES = {
    "rle_instance_segmentation_prediction",
    "semantic_segmentation_prediction",
}

# Declared kinds whose native representation is `InstanceDetections` (mask
# carrier required) — an empty output under these must stay instance-shaped.
_MASK_CARRIER_KIND_NAMES = {
    "instance_segmentation_prediction"
} | _RLE_CARRIER_KIND_NAMES

_NATIVE_CLASSIFICATION_TYPES = (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)


def convert_block_result_to_native(
    result: Any,
    manifest_description: Optional[ManifestDescription],
    block_name: str,
    declared_output_kinds: Optional[Dict[str, frozenset]] = None,
) -> Any:
    """OUT boundary: convert legacy objects in a dynamic block's ``BlockResult``
    back into native tensor representations, driven by declared output kinds
    with best-effort sniffing for wildcard outputs.

    ``BlockResult`` is ``dict | FlowControl | List[...] | List[List[...]]``;
    ``FlowControl`` elements pass through untouched (they carry no
    representation-dependent payload). Identity function (the very same object)
    when the tensor representation is off or the block declared
    ``tensor_native``.

    ``declared_output_kinds`` is the optional assembly-time precomputed lookup
    (see ``collect_declared_output_kind_names``); when omitted it is derived
    from the manifest here — same behavior, one extra computation per call.
    """
    if not _TENSOR_REPRESENTATION_ACTIVE:
        return result
    if (
        manifest_description is not None
        and manifest_description.tensor_compatibility
        is TensorCompatibility.TENSOR_NATIVE
    ):
        return result
    if declared_output_kinds is None:
        declared_output_kinds = (
            collect_declared_output_kind_names(manifest_description) or {}
        )
    return _walk_result_to_native(
        result=result,
        declared_output_kinds=declared_output_kinds,
        block_name=block_name,
    )


def _walk_result_to_native(
    result: Any,
    declared_output_kinds: Dict[str, frozenset],
    block_name: str,
) -> Any:
    if isinstance(result, FlowControl):
        return result
    if isinstance(result, list):
        return [
            _walk_result_to_native(
                result=element,
                declared_output_kinds=declared_output_kinds,
                block_name=block_name,
            )
            for element in result
        ]
    if isinstance(result, dict):
        return {
            output_name: _walk_output_value_to_native(
                value=value,
                declared_kinds=declared_output_kinds.get(output_name, frozenset()),
                block_name=block_name,
                value_name=output_name,
            )
            for output_name, value in result.items()
        }
    # Anything else is not a valid BlockResult shape — the engine rejects it
    # downstream with its own diagnostics; representation is not the problem.
    return result


def _collect_declared_output_kind_names(
    output_definition: Optional[DynamicOutputDefinition],
) -> set:
    if output_definition is None:
        return set()
    declared = set(getattr(output_definition, "kind", None) or [])
    declared.discard("*")
    return declared


def _walk_output_value_to_native(
    value: Any,
    declared_kinds: set,
    block_name: str,
    value_name: str,
) -> Any:
    """Walk containers down to leaf values (mirroring the IN walker; results
    never carry ``Batch``). Legacy predictions (sv.Detections, classification
    dicts under a declared kind) are leaves."""
    if value is None:
        return None
    if _is_key_point_prediction_tuple(value):
        # Already-native keypoint tuple returned under legacy mode: idempotent
        # passthrough (it is the target representation already).
        return value
    if isinstance(value, dict):
        if declared_kinds & {_CLASSIFICATION_KIND_NAME}:
            return _convert_output_leaf_to_native(
                value=value,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
        return {
            key: _walk_output_value_to_native(
                value=element,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
            for key, element in value.items()
        }
    if isinstance(value, (list, tuple)):
        if declared_kinds & {_EMBEDDING_KIND_NAME, _TENSOR_KIND_NAME}:
            # A declared embedding/tensor output returning a plain list IS the
            # legacy leaf value - convert it, don't recurse into the numbers.
            return _convert_output_leaf_to_native(
                value=value,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
        converted = [
            _walk_output_value_to_native(
                value=element,
                declared_kinds=declared_kinds,
                block_name=block_name,
                value_name=value_name,
            )
            for element in value
        ]
        if isinstance(value, tuple):
            # Namedtuples take one positional argument PER FIELD - splat the
            # converted elements; a plain tuple takes the iterable whole.
            if hasattr(value, "_fields"):
                return type(value)(*converted)
            return type(value)(converted)
        return converted
    return _convert_output_leaf_to_native(
        value=value,
        declared_kinds=declared_kinds,
        block_name=block_name,
        value_name=value_name,
    )


def _convert_output_leaf_to_native(
    value: Any,
    declared_kinds: set,
    block_name: str,
    value_name: str,
) -> Any:
    convertible_declared = declared_kinds & CONVERTIBLE_KIND_NAMES
    if convertible_declared:
        return _convert_output_by_declared_kinds(
            value=value,
            declared_kinds=convertible_declared,
            block_name=block_name,
            value_name=value_name,
        )
    return _sniff_convert_to_native(
        value=value, block_name=block_name, value_name=value_name
    )


def _convert_output_by_declared_kinds(
    value: Any,
    declared_kinds: set,
    block_name: str,
    value_name: str,
) -> Any:
    if _IMAGE_KIND_NAME in declared_kinds and isinstance(value, WorkflowImageData):
        return value
    if _KEYPOINT_KIND_NAME in declared_kinds:
        if _is_key_point_prediction_tuple(value):
            return value
        # Kind-UNION disambiguation: outputs commonly declare
        # [object_detection, instance_segmentation, keypoint_detection] together
        # (mirroring model-agnostic inputs). Rebuilding the KP tuple from an sv
        # that carries no keypoint payload would fabricate a keypoint prediction
        # out of a plain detection - so within a union the choice is data-driven
        # (payload columns present -> tuple), and only a SOLE keypoint
        # declaration keeps the strict always-tuple contract.
        if isinstance(value, sv.Detections) and (
            _sv_detections_carry_keypoint_payload(value)
            or not declared_kinds & _DETECTIONS_FAMILY_KIND_NAMES
        ):
            return sv_detections_to_native_key_point_prediction(sv_detections=value)
    if declared_kinds & _DETECTIONS_FAMILY_KIND_NAMES:
        if isinstance(value, _NativeDetections):
            return value
        if isinstance(value, sv.Detections):
            prefer_rle = bool(declared_kinds & _RLE_CARRIER_KIND_NAMES)
            converted = sv_detections_to_native(
                sv_detections=value,
                prefer_rle=prefer_rle,
            )
            # An empty sv carries no mask, so the converter degrades to plain
            # `Detections`; under a declared mask-carrying kind the native
            # convention is an EMPTY `InstanceDetections` (mirrors
            # `_empty_instance_detections` in sam3_video / the module's own
            # empty prefer_rle branch). Image dims are unrecoverable from zero
            # rows, hence the (0, 0)-sized empty carrier. Non-empty box-only sv
            # stays plain `Detections` — masks cannot be invented, and the
            # numpy path serializes box-only rows the same way.
            if (
                len(value) == 0
                and declared_kinds & _MASK_CARRIER_KIND_NAMES
                and not isinstance(converted, InstanceDetections)
            ):
                converted = _as_empty_instance_detections(
                    empty_detections=converted, prefer_rle=prefer_rle
                )
            return converted
    if _CLASSIFICATION_KIND_NAME in declared_kinds:
        if isinstance(value, _NATIVE_CLASSIFICATION_TYPES):
            return value
        if isinstance(value, dict):
            return classification_dict_to_native(
                prediction=value,
                block_name=block_name,
                value_name=value_name,
            )
    if declared_kinds & {_EMBEDDING_KIND_NAME, _TENSOR_KIND_NAME}:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple, np.ndarray)):
            return _legacy_array_to_tensor(value=value, declared_kinds=declared_kinds)
    if _is_representation_invariant(value):
        return value
    _raise_boundary_error(
        block_name=block_name,
        value_name=value_name,
        value=value,
        problem=(
            f"returned value does not match the legacy representation of the "
            f"declared output kind(s) {sorted(declared_kinds)}"
        ),
        remediation=(
            "Return the documented legacy type for the declared kind, or remove "
            "the kind declaration to enable best-effort conversion."
        ),
    )


def _sniff_convert_to_native(value: Any, block_name: str, value_name: str) -> Any:
    """Wildcard best-effort for outputs: ``sv.Detections`` converts data-driven
    (keypoint payload columns -> the KP tuple; masks -> ``InstanceDetections``;
    else ``Detections``); already-native objects pass through; a bare
    ``sv.KeyPoints`` fails loudly (mirror of the IN-side bare-``KeyPoints``
    rule); EVERYTHING else - dicts included (a classification dict through a
    wildcard output stays a dict, the plan's documented limitation) - is
    representation-invariant."""
    if isinstance(value, sv.Detections):
        if _sv_detections_carry_keypoint_payload(value):
            return sv_detections_to_native_key_point_prediction(sv_detections=value)
        return sv_detections_to_native(sv_detections=value, prefer_rle=False)
    if isinstance(value, sv.KeyPoints):
        _raise_boundary_error(
            block_name=block_name,
            value_name=value_name,
            value=value,
            problem="a bare `sv.KeyPoints` (without its detections component) "
            "has no native keypoint-prediction equivalent",
            remediation="Return the full keypoint payload on an sv.Detections "
            "(the numpy keypoint blocks' shape), declare the output kind, or "
            "switch the block to `tensor_compatibility=tensor_native`.",
        )
    return value


def _sv_detections_carry_keypoint_payload(sv_detections: sv.Detections) -> bool:
    return all(key in sv_detections.data for key in _KEYPOINT_PAYLOAD_KEYS)


def _legacy_array_to_tensor(value: Any, declared_kinds: set) -> torch.Tensor:
    if _EMBEDDING_KIND_NAME in declared_kinds and not isinstance(value, np.ndarray):
        # IN emits embeddings as List[float]; the exact inverse.
        return torch.as_tensor(
            np.asarray(value, dtype=np.float32), device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
    array = np.asarray(value)
    if _EMBEDDING_KIND_NAME in declared_kinds and array.dtype != np.float32:
        array = array.astype(np.float32)
    return torch.as_tensor(array.copy(), device=WORKFLOWS_IMAGE_TENSOR_DEVICE)


def sv_detections_to_native_key_point_prediction(
    sv_detections: sv.Detections,
) -> Tuple[KeyPoints, Union[Detections, InstanceDetections]]:
    """Rebuild the native ``(KeyPoints, Detections)`` keypoint prediction from an
    ``sv.Detections`` whose rows carry the keypoint payload columns (the shape
    the IN converter / numpy keypoint blocks produce). The bbox component keeps
    the payload in ``bboxes_metadata`` (the serializer's convention); the
    ``KeyPoints`` component is rebuilt via the shared
    ``build_native_key_points`` helper."""
    bbox_component = sv_detections_to_native(sv_detections=sv_detections)
    detections_number = int(len(sv_detections))
    per_instance_xy: List[List[List[float]]] = []
    per_instance_confidence: List[List[float]] = []
    xy_column = sv_detections.data.get(KEYPOINTS_XY_KEY_IN_SV_DETECTIONS)
    confidence_column = sv_detections.data.get(
        KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS
    )
    for index in range(detections_number):
        xy_row = xy_column[index] if xy_column is not None else []
        confidence_row = (
            confidence_column[index] if confidence_column is not None else []
        )
        per_instance_xy.append(np.asarray(xy_row).reshape(-1, 2).tolist())
        per_instance_confidence.append(np.asarray(confidence_row).reshape(-1).tolist())
    class_ids = (
        bbox_component.class_id.detach().cpu().tolist() if detections_number > 0 else []
    )
    key_points = build_native_key_points(
        per_instance_xy=per_instance_xy,
        per_instance_confidence=per_instance_confidence,
        object_class_ids=class_ids,
        image_metadata=bbox_component.image_metadata or {},
    )
    return key_points, bbox_component


def sv_detections_to_native(
    sv_detections: sv.Detections,
    *,
    prefer_rle: bool = False,
) -> Union[Detections, InstanceDetections]:
    """Direct ``sv.Detections`` -> native converter (the plan's main new
    primitive; no dict round-trip, dense masks preserved).

    Inverts ``native_detections_to_sv`` exactly:

    * xyxy/class_id/confidence become torch tensors on
      ``WORKFLOWS_IMAGE_TENSOR_DEVICE``;
    * ``image_metadata`` is rebuilt from the broadcast ``.data`` lineage columns
      (row 0 of each; rows share them by construction) — parent/root ids,
      coordinates and dimensions (including ``ROOT_PARENT_DIMENSIONS_KEY``, the
      root-coordinates precondition for masked instances), prediction_type,
      image_dimensions, inference_id;
    * ``CLASS_NAMES_KEY`` maps class_id -> first-seen ``data['class_name']``
      (fallback ``f"class_{id}"``); rows whose name differs from the map entry
      (id-sharing overrides) keep a per-box ``class`` override — the exact
      inverse of the IN converter's effective-name resolution;
    * ``bboxes_metadata`` carries per-box ``detection_id`` (minted uuid4 when
      missing — the tensor serializer hard-requires it), ``tracker_id`` from the
      sv field, keypoint payload rows and every remaining ``.data`` column;
    * mask carrier: a ``data['rle_mask']`` column wins (COCO dicts ->
      ``InstancesRLEMasks``, no re-encode); otherwise a dense sv mask becomes a
      torch bool stack, or — under ``prefer_rle=True`` (declared
      RLE/semantic-seg output kinds) — is encoded to ``InstancesRLEMasks``.
    """
    detections_number = int(len(sv_detections))
    data = sv_detections.data or {}
    image_metadata = _rebuild_image_metadata_from_sv(
        data=data, detections_number=detections_number
    )
    class_names_column = data.get(CLASS_NAME_DATA_COLUMN)
    class_id_values = (
        [int(value) for value in sv_detections.class_id]
        if sv_detections.class_id is not None
        else [0] * detections_number
    )
    class_names_mapping: Dict[int, str] = {}
    per_box_class_overrides: Dict[int, str] = {}
    for index in range(detections_number):
        class_id = class_id_values[index]
        class_name = (
            str(class_names_column[index])
            if class_names_column is not None
            else f"class_{class_id}"
        )
        if class_id not in class_names_mapping:
            # First-seen name wins the map entry. KNOWN DRIFT: if row 0 of a
            # shared class_id was itself a per-box override, the override
            # string lands in CLASS_NAMES_KEY and the base name becomes the
            # per-box override — the base-map/override split is NOT recoverable
            # from sv.Detections. Effective names and serialized output are
            # preserved (the collision round-trip test proves it); only
            # consumers reading image_metadata[CLASS_NAMES_KEY][id] directly
            # may observe the swapped roles.
            class_names_mapping[class_id] = class_name
        elif class_names_mapping[class_id] != class_name:
            per_box_class_overrides[index] = class_name
    image_metadata[CLASS_NAMES_KEY] = class_names_mapping
    bboxes_metadata = _rebuild_bboxes_metadata_from_sv(
        sv_detections=sv_detections,
        data=data,
        detections_number=detections_number,
        per_box_class_overrides=per_box_class_overrides,
    )
    xyxy = torch.as_tensor(
        np.asarray(sv_detections.xyxy, dtype=np.float32).reshape(-1, 4),
        device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
    )
    class_id_tensor = torch.as_tensor(
        class_id_values, dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
    ).reshape(-1)
    confidence_values = (
        np.asarray(sv_detections.confidence, dtype=np.float32).reshape(-1)
        if sv_detections.confidence is not None
        else np.zeros(detections_number, dtype=np.float32)
    )
    confidence_tensor = torch.as_tensor(
        confidence_values, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
    )
    mask_carrier = _rebuild_mask_carrier_from_sv(
        sv_detections=sv_detections,
        data=data,
        detections_number=detections_number,
        prefer_rle=prefer_rle,
    )
    if mask_carrier is not None:
        return InstanceDetections(
            xyxy=xyxy,
            class_id=class_id_tensor,
            confidence=confidence_tensor,
            mask=mask_carrier,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=xyxy,
        class_id=class_id_tensor,
        confidence=confidence_tensor,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _rebuild_image_metadata_from_sv(
    data: Dict[str, Any],
    detections_number: int,
) -> dict:
    image_metadata: dict = {}
    if detections_number == 0:
        return image_metadata
    for key in _IMAGE_LEVEL_ID_KEYS + _IMAGE_LEVEL_SCALAR_KEYS:
        column = data.get(key)
        if column is not None and len(column) > 0:
            image_metadata[key] = str(column[0])
    for key in _IMAGE_LEVEL_PAIR_KEYS:
        column = data.get(key)
        if column is not None and len(column) > 0:
            image_metadata[key] = [int(value) for value in np.asarray(column[0])]
    return image_metadata


_IMAGE_LEVEL_COLUMN_KEYS = frozenset(
    _IMAGE_LEVEL_ID_KEYS + _IMAGE_LEVEL_PAIR_KEYS + _IMAGE_LEVEL_SCALAR_KEYS
)


def _as_empty_instance_detections(
    empty_detections: Detections,
    prefer_rle: bool,
) -> InstanceDetections:
    """Re-shape an empty plain ``Detections`` into the empty
    ``InstanceDetections`` convention for declared mask-carrying kinds."""
    if prefer_rle:
        mask_carrier: Union[torch.Tensor, InstancesRLEMasks] = InstancesRLEMasks(
            image_size=(0, 0), masks=[]
        )
    else:
        mask_carrier = torch.zeros(
            (0, 0, 0), dtype=torch.bool, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
    return InstanceDetections(
        xyxy=empty_detections.xyxy,
        class_id=empty_detections.class_id,
        confidence=empty_detections.confidence,
        mask=mask_carrier,
        image_metadata=empty_detections.image_metadata,
        bboxes_metadata=empty_detections.bboxes_metadata,
    )


def _rebuild_bboxes_metadata_from_sv(
    sv_detections: sv.Detections,
    data: Dict[str, Any],
    detections_number: int,
    per_box_class_overrides: Dict[int, str],
) -> Optional[List[dict]]:
    # Zero rows -> None, matching the native convention
    # (`attach_native_detection_metadata` / the fusion empty builders); the
    # root-shift helper and the serializer both special-case None already.
    if detections_number == 0:
        return None
    detection_id_column = data.get(DETECTION_ID_KEY)
    per_box_keys = [
        key
        for key in data.keys()
        if key
        not in _IMAGE_LEVEL_COLUMN_KEYS
        | {CLASS_NAME_DATA_COLUMN, DETECTION_ID_KEY, RLE_MASK_DATA_COLUMN}
    ]
    keypoint_keys = set(_KEYPOINT_PAYLOAD_KEYS)
    bboxes_metadata: List[dict] = []
    for index in range(detections_number):
        per_box: dict = {}
        carried_id = (
            detection_id_column[index] if detection_id_column is not None else None
        )
        per_box[DETECTION_ID_KEY] = str(carried_id) if carried_id else str(uuid4())
        if sv_detections.tracker_id is not None:
            per_box[TRACKER_ID_KEY] = int(sv_detections.tracker_id[index])
        if index in per_box_class_overrides:
            per_box[CLASS_NAME_KEY] = per_box_class_overrides[index]
        for key in per_box_keys:
            row_value = data[key][index]
            if key in keypoint_keys:
                row_value = np.asarray(row_value).tolist()
            per_box[key] = row_value
        bboxes_metadata.append(per_box)
    return bboxes_metadata


def _rebuild_mask_carrier_from_sv(
    sv_detections: sv.Detections,
    data: Dict[str, Any],
    detections_number: int,
    prefer_rle: bool,
) -> Optional[Union[torch.Tensor, InstancesRLEMasks]]:
    rle_column = data.get(RLE_MASK_DATA_COLUMN)
    if rle_column is not None and len(rle_column) == detections_number > 0:
        coco_masks = []
        for index, entry in enumerate(rle_column):
            if (
                not isinstance(entry, dict)
                or "size" not in entry
                or "counts" not in entry
            ):
                raise ValueError(
                    f"sv.Detections carries a `{RLE_MASK_DATA_COLUMN}` column whose "
                    f"entry at index {index} is not a COCO-RLE dict with `size` and "
                    f"`counts` keys (got `{type(entry).__name__}`) - cannot rebuild "
                    f"the native RLE mask carrier from it."
                )
            coco_masks.append(dict(entry))
        image_size = tuple(int(value) for value in coco_masks[0]["size"])
        return InstancesRLEMasks.from_coco_rle_masks(
            image_size=image_size, masks=coco_masks
        )
    if sv_detections.mask is None:
        return None
    dense = np.asarray(sv_detections.mask)
    if dense.dtype != np.bool_:
        # Both carriers require bool; copy ONLY when the dtype actually differs
        # (sv masks are typically bool already — the old unconditional astype
        # duplicated the full (N, H, W) stack every call).
        dense = dense.astype(bool)
    if prefer_rle:
        if detections_number == 0:
            return InstancesRLEMasks(image_size=(0, 0), masks=[])
        # Iterative encode straight off the existing rows — rows of a contiguous
        # stack share memory with torch.as_tensor, so the only per-instance
        # transient is the fortran-order flatten inside torch_mask_to_coco_rle.
        coco_masks = [
            torch_mask_to_coco_rle(torch.as_tensor(np.ascontiguousarray(instance_mask)))
            for instance_mask in dense
        ]
        image_size = tuple(int(value) for value in coco_masks[0]["size"])
        return InstancesRLEMasks.from_coco_rle_masks(
            image_size=image_size, masks=coco_masks
        )
    return torch.as_tensor(dense, device=WORKFLOWS_IMAGE_TENSOR_DEVICE)


def classification_dict_to_native(
    prediction: dict,
    block_name: str,
    value_name: str,
) -> Union[ClassificationPrediction, MultiLabelClassificationPrediction]:
    """Rebuild a native classification prediction from the legacy in-memory dict.

    The confidence vector is reconstructed SPARSE — zeros for classes absent
    from the (already thresholded) dict — the plan's explicitly documented
    approximation. Single-label vs multi-label is discriminated by the legacy
    dict shape (``predictions`` list + ``top`` vs ``predictions`` dict +
    ``predicted_classes``)."""
    predictions_field = prediction.get("predictions")
    if "predicted_classes" in prediction and isinstance(predictions_field, dict):
        return _multi_label_dict_to_native(prediction=prediction)
    if isinstance(predictions_field, list):
        return _single_label_dict_to_native(prediction=prediction)
    _raise_boundary_error(
        block_name=block_name,
        value_name=value_name,
        value=prediction,
        problem="dict is not a recognised legacy classification prediction "
        "(expected a `predictions` list with `top`, or a `predictions` dict "
        "with `predicted_classes`)",
        remediation="Return the documented classification dict shape, or remove "
        "the kind declaration.",
    )


def _classification_metadata_from_dict(prediction: dict, class_names: dict) -> dict:
    metadata: dict = {
        CLASS_NAMES_KEY: class_names,
        PREDICTION_TYPE_KEY: str(
            prediction.get(PREDICTION_TYPE_KEY) or "classification"
        ),
    }
    image_field = prediction.get("image") or {}
    height, width = image_field.get("height"), image_field.get("width")
    if height is not None and width is not None:
        metadata[IMAGE_DIMENSIONS_KEY] = [int(height), int(width)]
    for key in (INFERENCE_ID_KEY, PARENT_ID_KEY, ROOT_PARENT_ID_KEY):
        value = prediction.get(key)
        if value is not None:
            metadata[key] = str(value)
    # `time` round-trips when the dict carries it (the tensor serializer emits
    # it between inference_id and image — see the amended plan).
    if prediction.get("time") is not None:
        metadata["time"] = float(prediction["time"])
    return metadata


def _single_label_dict_to_native(prediction: dict) -> ClassificationPrediction:
    entries = prediction.get("predictions") or []
    entry_names: Dict[int, str] = {}
    entry_confidences: Dict[int, float] = {}
    for entry in entries:
        class_id = int(entry["class_id"])
        entry_names[class_id] = str(entry.get("class", class_id))
        entry_confidences[class_id] = float(entry.get("confidence", 0.0))
    # Sparse-reconstruction approximation, part 2: the vector length is
    # max(SURVIVING class_id) + 1 — when the highest-id classes were thresholded
    # out of the legacy dict, the rebuilt vector is SHORTER than the model's
    # original class count (serialization-stable: the surviving set round-trips
    # exactly; only the trailing all-zero tail is unrecoverable).
    num_classes = (max(entry_names.keys()) + 1) if entry_names else 0
    class_names = {
        class_id: entry_names.get(class_id, str(class_id))
        for class_id in range(num_classes)
    }
    confidence_vector = [entry_confidences.get(i, 0.0) for i in range(num_classes)]
    top_class = prediction.get("top")
    top_class_id = next(
        (cid for cid, name in entry_names.items() if name == top_class),
        (
            int(max(entry_confidences, key=entry_confidences.get))
            if entry_confidences
            else 0
        ),
    )
    metadata = _classification_metadata_from_dict(
        prediction=prediction, class_names=class_names
    )
    # Re-attaching the smallest listed confidence as the serializer threshold
    # keeps the gap-filled zero classes out of re-serialization, so the
    # round-tripped dict matches the input (edge: a listed 0.0-confidence entry
    # means no threshold can be attached and gap classes reappear — the same
    # documented P2 edge as the visual-search classifier).
    positive_confidences = [c for c in entry_confidences.values() if c > 0.0]
    if entry_confidences and len(positive_confidences) == len(entry_confidences):
        metadata[CLASSIFICATION_CONFIDENCE_THRESHOLD_KEY] = min(positive_confidences)
    return ClassificationPrediction(
        class_id=torch.tensor(
            [top_class_id], dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.tensor(
            [confidence_vector],
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        images_metadata=[metadata],
    )


def _multi_label_dict_to_native(
    prediction: dict,
) -> MultiLabelClassificationPrediction:
    entries = prediction.get("predictions") or {}
    name_to_id = {
        str(name): int(payload["class_id"]) for name, payload in entries.items()
    }
    num_classes = (max(name_to_id.values()) + 1) if name_to_id else 0
    class_names = {class_id: str(class_id) for class_id in range(num_classes)}
    confidence_vector = [0.0] * num_classes
    for name, payload in entries.items():
        class_id = int(payload["class_id"])
        class_names[class_id] = str(name)
        confidence_vector[class_id] = float(payload.get("confidence", 0.0))
    predicted_class_ids = [
        name_to_id[str(name)]
        for name in prediction.get("predicted_classes") or []
        if str(name) in name_to_id
    ]
    metadata = _classification_metadata_from_dict(
        prediction=prediction, class_names=class_names
    )
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor(
            predicted_class_ids,
            dtype=torch.long,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        confidence=torch.tensor(
            confidence_vector,
            dtype=torch.float32,
            device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
        ),
        # MultiLabel carries SINGULAR image_metadata (dict).
        image_metadata=metadata,
    )
