"""Native prediction kinds of the V2 block catalogue.

Payloads are the ``inference_models`` dataclasses and ``torch.Tensor`` values
that V1 TENSOR_NATIVE blocks produce. Kind names match
``execution_engine/entities/tensor_native_types.py``. Scalars and configuration
keep the engine's built-in kinds; images live in the image kind module.

=====================================  =========================================
kind name                              native payload
=====================================  =========================================
embedding                              1-D floating ``torch.Tensor``
tensor                                 ``torch.Tensor`` of any shape
classification_prediction              ``ClassificationPrediction`` of one image,
                                       or ``MultiLabelClassificationPrediction``
detection                              7-tuple yielded by iterating
                                       ``Detections`` / ``InstanceDetections``
object_detection_prediction            ``Detections``
qr_code_detection, bar_code_detection  ``Detections``; decoded text per box
                                       under ``bboxes_metadata[i]["data"]``
instance_segmentation_prediction       ``InstanceDetections``, dense or RLE mask
rle_instance_segmentation_prediction   ``InstanceDetections``, RLE mask
semantic_segmentation_prediction       ``InstanceDetections``, RLE mask, one row
                                       per class
keypoint_detection_prediction          ``(KeyPoints, Detections | None)``
=====================================  =========================================

Every kind's hooks follow one policy::

    validate        Reads tensor shape, dtype and device only: no copy, no
                    device sync. Raises ContractError naming the mismatch.
    deserialize     A native payload of the kind's family is returned as is
                    (same object, storage and device). A native envelope
                    (payload_codec) decodes to CPU tensors and is validated.
                    Some kinds also import legacy V1 JSON (lossy, see
                    _import_legacy). Anything else raises, so a kind union
                    tries the next kind.
    serialize       Validates, then writes the faithful native envelope. This
                    is the explicit host boundary.
    convert_output  Geometry kinds and classification use
                    coordinates.convert_prediction_output: own or workflow-root
                    coordinates, returning new objects only when converting.
"""

from collections.abc import Mapping
from functools import partial
from numbers import Integral, Real
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
from roboflow_workflows.execution_engine.v2.blocks.coordinates import (
    convert_prediction_output,
)
from roboflow_workflows.execution_engine.v2.blocks.payload_codec import (
    decode_native,
    encode_native,
    is_native_envelope,
    tensor_dtype_problem,
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

KeyPointPrediction = Tuple[KeyPoints, Optional[Detections]]

DETECTION_TUPLE_FIELDS = (
    "xyxy",
    "mask",
    "class_id",
    "confidence",
    "tracker_id",
    "data",
    "image_metadata",
)

# ---------------------------------------------------------------------------
# Structural checks. They read tensor metadata only, never tensor values.
# ---------------------------------------------------------------------------


def _dtype_matches(dtype: torch.dtype, *, family: str) -> bool:
    if family == "floating":
        return dtype.is_floating_point
    if family == "binary":
        return dtype in (torch.bool, torch.uint8)

    is_real_number = not dtype.is_complex and dtype != torch.bool
    if family == "integer":
        return is_real_number and not dtype.is_floating_point

    return is_real_number


def _check_tensor(
    value: Any,
    *,
    name: str,
    shape: Tuple[Optional[int], ...],
    dtype: str,
    device: Optional[torch.device] = None,
) -> None:
    """``shape`` lists expected sizes; ``None`` accepts any size on that axis.

    ``dtype`` is a family: ``integer``, ``floating``, ``real`` (integer or
    floating) or ``binary`` (bool or uint8).
    """
    if not isinstance(value, torch.Tensor):
        raise ContractError(
            f"{name} must be a torch.Tensor, got {type(value).__name__}"
        )

    shape_matches = value.ndim == len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(value.shape, shape)
    )
    if not shape_matches:
        expected_text = ", ".join("*" if size is None else str(size) for size in shape)
        raise ContractError(
            f"{name} must have shape ({expected_text}), got {tuple(value.shape)}"
        )
    if not _dtype_matches(value.dtype, family=dtype):
        raise ContractError(f"{name} dtype must be {dtype}, got {value.dtype}")
    dtype_problem = tensor_dtype_problem(value.dtype)
    if dtype_problem is not None:
        raise ContractError(f"{name}: {dtype_problem}")
    if device is not None and value.device != device:
        raise ContractError(
            f"{name} is on {value.device}, but the rest of the prediction is on "
            f"{device}"
        )


def _check_mapping(value: Any, *, name: str) -> None:
    if value is not None and not isinstance(value, Mapping):
        raise ContractError(
            f"{name} must be a dict or None, got {type(value).__name__}"
        )


def _check_row_metadata(value: Any, *, name: str, rows: int) -> None:
    if value is None:
        return

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ContractError(
            f"{name} must be a list of {rows} dicts or None, got {type(value).__name__}"
        )
    if len(value) != rows:
        raise ContractError(f"{name} has {len(value)} entries for {rows} rows")
    for position, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ContractError(
                f"{name}[{position}] must be a dict, got {type(entry).__name__}"
            )


def _require_instance(payload: Any, family: type) -> None:
    if not isinstance(payload, family):
        raise ContractError(
            f"Expected inference_models.{family.__name__}, got {type(payload).__name__}"
        )


def _check_boxes(payload: Any, *, device: Optional[torch.device] = None) -> int:
    """Check the fields ``Detections`` and ``InstanceDetections`` share.

    Returns the number of boxes.
    """
    family = type(payload).__name__
    _check_tensor(
        payload.xyxy,
        name=f"{family}.xyxy",
        shape=(None, 4),
        dtype="real",
        device=device,
    )
    rows = payload.xyxy.shape[0]
    device = payload.xyxy.device

    _check_tensor(
        payload.class_id,
        name=f"{family}.class_id",
        shape=(rows,),
        dtype="integer",
        device=device,
    )
    _check_tensor(
        payload.confidence,
        name=f"{family}.confidence",
        shape=(rows,),
        dtype="floating",
        device=device,
    )
    _check_mapping(payload.image_metadata, name=f"{family}.image_metadata")
    _check_row_metadata(
        payload.bboxes_metadata, name=f"{family}.bboxes_metadata", rows=rows
    )

    return rows


def _check_rle_masks(masks: InstancesRLEMasks, *, rows: int) -> None:
    image_size = masks.image_size
    size_is_valid = (
        isinstance(image_size, (tuple, list))
        and len(image_size) == 2
        and all(isinstance(size, Integral) and size >= 0 for size in image_size)
    )
    if not size_is_valid:
        raise ContractError(
            "InstancesRLEMasks.image_size must be (height, width) integers, "
            f"got {image_size!r}"
        )
    if len(masks.masks) != rows:
        raise ContractError(
            f"InstancesRLEMasks holds {len(masks.masks)} masks for {rows} rows"
        )
    for position, counts in enumerate(masks.masks):
        if not isinstance(counts, (bytes, str)):
            raise ContractError(
                f"InstancesRLEMasks.masks[{position}] must be COCO RLE counts "
                f"(bytes or str), got {type(counts).__name__}"
            )


def _validate_instance_detections(payload: Any, *, require_rle: bool) -> bool:
    _require_instance(payload, InstanceDetections)
    rows = _check_boxes(payload)

    if isinstance(payload.mask, InstancesRLEMasks):
        _check_rle_masks(payload.mask, rows=rows)
        return True

    if require_rle:
        raise ContractError(
            "InstanceDetections.mask must be InstancesRLEMasks for this kind, got "
            f"{type(payload.mask).__name__}"
        )
    _check_tensor(
        payload.mask,
        name="InstanceDetections.mask",
        shape=(rows, None, None),
        dtype="binary",
        device=payload.xyxy.device,
    )

    return True


def _validate_detections(payload: Any) -> bool:
    _require_instance(payload, Detections)
    _check_boxes(payload)

    return True


def _validate_key_point_prediction(payload: Any) -> bool:
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ContractError(
            "Expected a (KeyPoints, Detections | None) tuple, got "
            f"{type(payload).__name__}"
        )

    key_points, detections = payload
    _require_instance(key_points, KeyPoints)
    _check_tensor(
        key_points.xy, name="KeyPoints.xy", shape=(None, None, 2), dtype="real"
    )
    rows, slots = key_points.xy.shape[:2]
    device = key_points.xy.device

    _check_tensor(
        key_points.class_id,
        name="KeyPoints.class_id",
        shape=(rows,),
        dtype="integer",
        device=device,
    )
    _check_tensor(
        key_points.confidence,
        name="KeyPoints.confidence",
        shape=(rows, slots),
        dtype="floating",
        device=device,
    )
    if key_points.covariance is not None:
        _check_tensor(
            key_points.covariance,
            name="KeyPoints.covariance",
            shape=(rows, slots, 2, 2),
            dtype="floating",
            device=device,
        )
    if key_points.detection_confidence is not None:
        _check_tensor(
            key_points.detection_confidence,
            name="KeyPoints.detection_confidence",
            shape=(rows,),
            dtype="floating",
            device=device,
        )
    _check_mapping(key_points.image_metadata, name="KeyPoints.image_metadata")
    _check_row_metadata(
        key_points.key_points_metadata, name="KeyPoints.key_points_metadata", rows=rows
    )

    if detections is None:
        return True

    _require_instance(detections, Detections)
    box_rows = _check_boxes(detections, device=device)
    if box_rows != rows:
        raise ContractError(
            f"KeyPoints has {rows} instances but its Detections has {box_rows} boxes"
        )

    return True


def _validate_classification(payload: Any) -> bool:
    if isinstance(payload, ClassificationPrediction):
        # One image per payload: the V1 serializer reads row 0 only.
        _check_tensor(
            payload.class_id,
            name="ClassificationPrediction.class_id",
            shape=(1,),
            dtype="integer",
        )
        _check_tensor(
            payload.confidence,
            name="ClassificationPrediction.confidence",
            shape=(1, None),
            dtype="floating",
            device=payload.class_id.device,
        )
        _check_row_metadata(
            payload.images_metadata,
            name="ClassificationPrediction.images_metadata",
            rows=1,
        )
        return True

    if isinstance(payload, MultiLabelClassificationPrediction):
        _check_tensor(
            payload.class_ids,
            name="MultiLabelClassificationPrediction.class_ids",
            shape=(None,),
            dtype="integer",
        )
        _check_tensor(
            payload.confidence,
            name="MultiLabelClassificationPrediction.confidence",
            shape=(None,),
            dtype="floating",
            device=payload.class_ids.device,
        )
        _check_mapping(
            payload.image_metadata,
            name="MultiLabelClassificationPrediction.image_metadata",
        )
        return True

    raise ContractError(
        "Expected inference_models.ClassificationPrediction or "
        f"MultiLabelClassificationPrediction, got {type(payload).__name__}"
    )


def _check_detection_scalar(
    value: Any, *, name: str, dtype: str, device: torch.device
) -> None:
    python_type = Integral if dtype == "integer" else Real
    if isinstance(value, python_type) and not isinstance(value, bool):
        return

    _check_tensor(value, name=name, shape=(), dtype=dtype, device=device)


def _validate_detection(payload: Any) -> bool:
    if not isinstance(payload, tuple) or len(payload) != len(DETECTION_TUPLE_FIELDS):
        raise ContractError(
            f"Expected the tuple ({', '.join(DETECTION_TUPLE_FIELDS)}) yielded by "
            f"iterating Detections, got {type(payload).__name__}"
        )

    xyxy, mask, class_id, confidence, tracker_id, data, image_metadata = payload
    _check_tensor(xyxy, name="detection xyxy", shape=(4,), dtype="real")
    device = xyxy.device

    if isinstance(mask, Mapping):
        if "size" not in mask or "counts" not in mask:
            raise ContractError(
                "detection mask must be a COCO RLE dict with 'size' and 'counts'"
            )
    elif mask is not None:
        _check_tensor(
            mask,
            name="detection mask",
            shape=(None, None),
            dtype="binary",
            device=device,
        )
    _check_detection_scalar(
        class_id, name="detection class_id", dtype="integer", device=device
    )
    _check_detection_scalar(
        confidence, name="detection confidence", dtype="floating", device=device
    )
    if tracker_id is not None and (
        not isinstance(tracker_id, Integral) or isinstance(tracker_id, bool)
    ):
        raise ContractError(
            f"detection tracker_id must be an integer or None, got {tracker_id!r}"
        )
    for name, value in (("data", data), ("image_metadata", image_metadata)):
        if not isinstance(value, Mapping):
            raise ContractError(
                f"detection {name} must be a dict, got {type(value).__name__}"
            )

    return True


def _validate_embedding(payload: Any) -> bool:
    _check_tensor(payload, name="embedding", shape=(None,), dtype="floating")

    return True


def _validate_tensor(payload: Any) -> bool:
    if not isinstance(payload, torch.Tensor):
        raise ContractError(f"Expected a torch.Tensor, got {type(payload).__name__}")
    dtype_problem = tensor_dtype_problem(payload.dtype)
    if dtype_problem is not None:
        raise ContractError(f"tensor: {dtype_problem}")

    return True


# ---------------------------------------------------------------------------
# Boundary hooks
# ---------------------------------------------------------------------------


def _serialize(payload: Any, *, validate: Callable[[Any], bool]) -> dict:
    """Write the faithful envelope; only payloads of the kind's own family.

    Checking the family first lets a kind union pick the matching kind.
    """
    validate(payload)
    envelope = encode_native(payload)

    return envelope


def _deserialize(
    value: Any,
    *,
    native_types: Tuple[type, ...],
    validate: Callable[[Any], bool],
    import_legacy: Optional[Callable[[Any], Any]] = None,
) -> Any:
    if isinstance(value, native_types):
        return value
    if is_native_envelope(value):
        payload = decode_native(value)
        validate(payload)
        return payload
    if import_legacy is not None:
        payload = import_legacy(value)
        return payload

    raise ContractError(
        f"Expected {' or '.join(kind.__name__ for kind in native_types)} or a "
        f"native envelope, got {type(value).__name__}"
    )


def _import_legacy_tensor(value: Any) -> torch.Tensor:
    """V1 wire of the tensor kinds: a nested list, read as float32 (V1 parity)."""
    if not isinstance(value, list):
        raise ContractError(
            "Expected a torch.Tensor, a native envelope or a legacy nested list of "
            f"numbers, got {type(value).__name__}"
        )

    tensor = torch.as_tensor(value, dtype=torch.float32)

    return tensor


def _import_legacy(value: Any, *, v1_deserializer: str, parameter: str) -> Any:
    """Import legacy V1 JSON through the V1 tensor-native deserializer.

    Explicit compatibility boundary with V1's own losses: it rebuilds lineage
    from per-box keys only, drops per-box extras such as ``tracker_id`` and
    decoded code text, and places tensors on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``.
    Imported on use, because the V1 module loads V1 executor modules.
    """
    from roboflow_workflows.core_steps.common import deserializers_tensor

    deserializer = getattr(deserializers_tensor, v1_deserializer)
    try:
        payload = deserializer(parameter, value)
    except Exception as error:
        raise ContractError(
            f"Not a native payload, native envelope or importable legacy V1 "
            f"{parameter} ({type(error).__name__}: {error})"
        ) from error

    return payload


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


def _codec_hooks(
    validate: Callable[[Any], bool],
    *,
    native_types: Tuple[type, ...],
    import_legacy: Optional[Callable[[Any], Any]] = None,
) -> dict:
    """The ``validate``, ``deserialize`` and ``serialize`` hooks of one kind."""
    hooks = {
        "validate": validate,
        "deserialize": partial(
            _deserialize,
            native_types=native_types,
            validate=validate,
            import_legacy=import_legacy,
        ),
        "serialize": partial(_serialize, validate=validate),
    }

    return hooks


def _v1_importer(v1_deserializer: str, *, kind_name: str) -> Callable[[Any], Any]:
    importer = partial(
        _import_legacy, v1_deserializer=v1_deserializer, parameter=kind_name
    )

    return importer


_validate_any_mask = partial(_validate_instance_detections, require_rle=False)
_validate_rle_mask = partial(_validate_instance_detections, require_rle=True)

EMBEDDING_KIND = Kind(
    name="embedding",
    description="Vector embedding: 1-D floating torch.Tensor.",
    **_codec_hooks(
        _validate_embedding,
        native_types=(torch.Tensor,),
        import_legacy=_import_legacy_tensor,
    ),
)
TENSOR_KIND = Kind(
    name="tensor",
    description="Raw torch.Tensor of any shape and a supported real dtype.",
    **_codec_hooks(
        _validate_tensor,
        native_types=(torch.Tensor,),
        import_legacy=_import_legacy_tensor,
    ),
)
CLASSIFICATION_PREDICTION_KIND = Kind(
    name="classification_prediction",
    description=(
        "inference_models.ClassificationPrediction of one image or "
        "MultiLabelClassificationPrediction."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_classification,
        native_types=(ClassificationPrediction, MultiLabelClassificationPrediction),
        import_legacy=_v1_importer(
            "deserialize_native_classification_prediction_kind",
            kind_name="classification_prediction",
        ),
    ),
)
DETECTION_KIND = Kind(
    name="detection",
    description=(
        "One box as yielded by iterating inference_models.Detections: "
        f"({', '.join(DETECTION_TUPLE_FIELDS)})."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(_validate_detection, native_types=(tuple,)),
)
OBJECT_DETECTION_PREDICTION_KIND = Kind(
    name="object_detection_prediction",
    description=(
        "inference_models.Detections with xyxy (N, 4), class_id (N,) and "
        "confidence (N,) on one device."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_detections,
        native_types=(Detections,),
        import_legacy=_v1_importer(
            "deserialize_detections_kind", kind_name="object_detection_prediction"
        ),
    ),
)
QR_CODE_DETECTION_KIND = Kind(
    name="qr_code_detection",
    description=(
        "inference_models.Detections of QR codes; decoded text per box under "
        "bboxes_metadata[i]['data']."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_detections,
        native_types=(Detections,),
        import_legacy=_v1_importer(
            "deserialize_detections_kind", kind_name="qr_code_detection"
        ),
    ),
)
BAR_CODE_DETECTION_KIND = Kind(
    name="bar_code_detection",
    description=(
        "inference_models.Detections of barcodes; decoded text per box under "
        "bboxes_metadata[i]['data']."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_detections,
        native_types=(Detections,),
        import_legacy=_v1_importer(
            "deserialize_detections_kind", kind_name="bar_code_detection"
        ),
    ),
)
INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="instance_segmentation_prediction",
    description=(
        "inference_models.InstanceDetections with a dense bool/uint8 (N, H, W) "
        "mask or InstancesRLEMasks."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_any_mask,
        native_types=(InstanceDetections,),
        import_legacy=_v1_importer(
            "deserialize_rle_detections_kind",
            kind_name="instance_segmentation_prediction",
        ),
    ),
)
RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND = Kind(
    name="rle_instance_segmentation_prediction",
    description="inference_models.InstanceDetections with InstancesRLEMasks.",
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_rle_mask,
        native_types=(InstanceDetections,),
        import_legacy=_v1_importer(
            "deserialize_rle_detections_kind",
            kind_name="rle_instance_segmentation_prediction",
        ),
    ),
)
SEMANTIC_SEGMENTATION_PREDICTION_KIND = Kind(
    name="semantic_segmentation_prediction",
    description=(
        "inference_models.InstanceDetections with one row and one InstancesRLEMasks "
        "mask per predicted class; a mask may hold disconnected regions."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(
        _validate_rle_mask,
        native_types=(InstanceDetections,),
        import_legacy=_v1_importer(
            "deserialize_rle_detections_kind",
            kind_name="semantic_segmentation_prediction",
        ),
    ),
)
KEYPOINT_DETECTION_PREDICTION_KIND = Kind(
    name="keypoint_detection_prediction",
    description=(
        "(inference_models.KeyPoints, Detections | None) with one box per "
        "skeleton when boxes are present."
    ),
    convert_output=convert_prediction_output,
    **_codec_hooks(_validate_key_point_prediction, native_types=(tuple,)),
)

NATIVE_KINDS: Tuple[Kind, ...] = (
    EMBEDDING_KIND,
    TENSOR_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    DETECTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    QR_CODE_DETECTION_KIND,
    BAR_CODE_DETECTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
)

__all__ = [
    "BAR_CODE_DETECTION_KIND",
    "CLASSIFICATION_PREDICTION_KIND",
    "DETECTION_KIND",
    "DETECTION_TUPLE_FIELDS",
    "EMBEDDING_KIND",
    "INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "KEYPOINT_DETECTION_PREDICTION_KIND",
    "KeyPointPrediction",
    "NATIVE_KINDS",
    "OBJECT_DETECTION_PREDICTION_KIND",
    "QR_CODE_DETECTION_KIND",
    "RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND",
    "SEMANTIC_SEGMENTATION_PREDICTION_KIND",
    "TENSOR_KIND",
]
