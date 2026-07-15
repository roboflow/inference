"""
Tensor-native sibling of `common/deserializers.py`. The loader selects this module
when `ENABLE_TENSOR_DATA_REPRESENTATION` is enabled. Functions here add tensor-aware
code paths and delegate other inputs to the NumPy implementations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.core_steps.common.deserializers import (
    _parse_optional_parent_metadata,
)
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_image_kind as _deserialize_image_kind_numpy,
)
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_video_metadata_kind,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    native_detections_from_inference_predictions,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAME_KEY,
    CLASS_NAMES_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PARENT_ORIGIN_KEY,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_INFERENCE_RESPONSE,
    ROOT_PARENT_ID_KEY,
    ROOT_PARENT_ORIGIN_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)
from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

DEFAULT_OBJECT_DETECTION_PREDICTION_TYPE = "object-detection"
DEFAULT_INSTANCE_SEGMENTATION_PREDICTION_TYPE = "instance-segmentation"


#: Channel counts treated as a "channels" axis when sniffing tensor layout.
_CHANNEL_AXIS_SIZES = (1, 3, 4)


def _ensure_chw_layout(image: torch.Tensor) -> torch.Tensor:
    """Normalise a single-image tensor to the `WorkflowImageData` CHW contract.

    `WorkflowImageData.tensor_image` uses CHW RGB. Producers may provide a
    channels-last HWC tensor, which is normalised before constructing the workflow
    image container.

    Detect channels-last with the same heuristic the model preprocessing uses
    (`pre_processing.py`: channels not at the front but present at the back) and
    permute HWC -> CHW. CHW input (front axis is the channels axis) is returned
    untouched. Non-3D tensors are left alone.
    """
    if image.ndim == 2:
        # Single-channel (H, W) tensors are normalised to the (1, H, W) CHW
        # contract WorkflowImageData.tensor_image documents - grayscale carries
        # no channel semantics, so no reversal is involved.
        return image.unsqueeze(0).contiguous()
    if image.ndim != 3:
        return image
    channels_first = image.shape[0] in _CHANNEL_AXIS_SIZES
    channels_last = image.shape[2] in _CHANNEL_AXIS_SIZES
    if channels_last and not channels_first:
        return image.permute(2, 0, 1).contiguous()
    return image


def deserialize_image_kind(
    parameter: str,
    image: Any,
    prevent_local_images_loading: bool = False,
) -> WorkflowImageData:
    if isinstance(image, WorkflowImageData):
        return image
    if isinstance(image, torch.Tensor):
        parent_metadata, workflow_root_ancestor_metadata, video_metadata = (
            _parse_image_metadata_fields(parameter=parameter, image=None)
        )
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            tensor_image=_ensure_chw_layout(image),
            video_metadata=video_metadata,
        )
    if isinstance(image, dict) and image.get("type") == "tensor":
        value = image.get("value")
        if not isinstance(value, torch.Tensor):
            raise RuntimeInputError(
                public_message=(
                    f"Detected runtime parameter `{parameter}` declared with "
                    f"type='tensor' but its value is of type {type(value)}; "
                    "expected torch.Tensor."
                ),
                context="workflow_execution | runtime_input_validation",
            )
        parent_metadata, workflow_root_ancestor_metadata, video_metadata = (
            _parse_image_metadata_fields(parameter=parameter, image=image)
        )
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            tensor_image=_ensure_chw_layout(value),
            video_metadata=video_metadata,
        )
    return _deserialize_image_kind_numpy(
        parameter=parameter,
        image=image,
        prevent_local_images_loading=prevent_local_images_loading,
    )


def _parse_image_metadata_fields(parameter: str, image: Any):
    """Parse the parent/root-parent/video metadata block shared by the
    image-kind deserializer paths. Mirrors the prefix of
    `deserialize_image_kind` in the numpy file."""
    is_image_dict = isinstance(image, dict)
    parent_id = image.get(PARENT_ID_KEY, parameter) if is_image_dict else parameter
    parent_origin = image.get(PARENT_ORIGIN_KEY) if is_image_dict else None
    parent_metadata = _parse_optional_parent_metadata(
        parameter=parameter,
        parent_id=parent_id,
        parent_origin=parent_origin,
    )
    root_parent_id = image.get(ROOT_PARENT_ID_KEY) if is_image_dict else None
    root_parent_origin = image.get(ROOT_PARENT_ORIGIN_KEY) if is_image_dict else None
    workflow_root_ancestor_metadata = _parse_optional_parent_metadata(
        parameter=parameter,
        parent_id=root_parent_id,
        parent_origin=root_parent_origin,
    )
    video_metadata = None
    if is_image_dict and "video_metadata" in image:
        video_metadata = deserialize_video_metadata_kind(
            parameter=parameter, video_metadata=image["video_metadata"]
        )
    return parent_metadata, workflow_root_ancestor_metadata, video_metadata


def deserialize_detections_kind(
    parameter: str,
    detections: Any,
) -> Detections:
    """Tensor-native sibling of the numpy ``deserialize_detections_kind``.

    The numpy path returns ``sv.Detections``; on the tensor branch every consumer
    block expects a native ``inference_models.Detections`` (xyxy/class_id/confidence
    tensors on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``, ``image_metadata[CLASS_NAMES_KEY]``
    + per-box ``detection_id``). Builds it by re-using
    ``native_detections_from_inference_predictions`` from the serialised inference
    prediction dicts (center ``x``/``y``/``width``/``height``). Lineage carried by the
    serialised predictions is reconstructed onto the placeholder image so crop-aware
    coordinate recovery downstream keeps working.
    """
    if isinstance(detections, (Detections, InstanceDetections)):
        return detections
    raw_predictions = _validate_serialized_detections(
        parameter=parameter, detections=detections
    )
    return _native_detections_from_serialized(
        parameter=parameter,
        detections=detections,
        raw_predictions=raw_predictions,
        prediction_type=DEFAULT_OBJECT_DETECTION_PREDICTION_TYPE,
    )


def deserialize_rle_detections_kind(
    parameter: str,
    detections: Any,
) -> InstanceDetections:
    """Tensor-native sibling of the numpy ``deserialize_rle_detections_kind``.

    Builds the base ``Detections`` exactly as ``deserialize_detections_kind`` does,
    then rebuilds ``InstancesRLEMasks`` from each serialised prediction's COCO RLE
    (``RLE_MASK_KEY_IN_INFERENCE_RESPONSE``) and returns a native
    ``InstanceDetections``. Used for the rle-instance-seg and semantic-seg kinds.
    """
    if isinstance(detections, (Detections, InstanceDetections)):
        return detections
    raw_predictions = _validate_serialized_detections(
        parameter=parameter, detections=detections
    )
    base_detections = _native_detections_from_serialized(
        parameter=parameter,
        detections=detections,
        raw_predictions=raw_predictions,
        prediction_type=DEFAULT_INSTANCE_SEGMENTATION_PREDICTION_TYPE,
    )
    mask = _rebuild_instances_rle_masks(
        detections=detections, raw_predictions=raw_predictions
    )
    return InstanceDetections(
        xyxy=base_detections.xyxy,
        class_id=base_detections.class_id,
        confidence=base_detections.confidence,
        mask=mask,
        image_metadata=base_detections.image_metadata,
        bboxes_metadata=base_detections.bboxes_metadata,
    )


def _validate_serialized_detections(parameter: str, detections: Any) -> List[dict]:
    """Validate the serialised-detections dict shape (mirrors the numpy
    deserialiser's checks) and return the list of per-box prediction dicts."""
    if not isinstance(detections, dict):
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` declared to hold "
            f"detections, but invalid type of data found.",
            context="workflow_execution | runtime_input_validation",
        )
    if "predictions" not in detections or "image" not in detections:
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` declared to hold "
            f"detections, but dictionary misses required keys.",
            context="workflow_execution | runtime_input_validation",
        )
    raw_predictions = detections["predictions"]
    if not isinstance(raw_predictions, list):
        raise RuntimeInputError(
            public_message=f"Detected runtime parameter `{parameter}` declared to hold "
            f"detections, but `predictions` is not a list.",
            context="workflow_execution | runtime_input_validation",
        )
    return raw_predictions


def _native_detections_from_serialized(
    parameter: str,
    detections: dict,
    raw_predictions: List[dict],
    prediction_type: str,
) -> Detections:
    placeholder_image = _build_placeholder_image(
        parameter=parameter,
        detections=detections,
        raw_predictions=raw_predictions,
    )
    return native_detections_from_inference_predictions(
        image=placeholder_image,
        predictions=raw_predictions,
        prediction_type=prediction_type,
        device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
    )


def _rebuild_instances_rle_masks(
    detections: dict,
    raw_predictions: List[dict],
) -> InstancesRLEMasks:
    """Rebuild an ``InstancesRLEMasks`` from the serialised per-box COCO RLE entries.

    Each prediction carries ``rle_mask`` as ``{"size": [h, w], "counts": ...}``.
    The image size is taken from the RLE entries (falling back to the serialised
    ``image`` dimensions) so the rebuilt masks decode against the right canvas.
    """
    image_height, image_width = _read_serialized_image_shape(detections)
    coco_rle_masks: List[dict] = []
    for prediction in raw_predictions:
        rle = prediction.get(RLE_MASK_KEY_IN_INFERENCE_RESPONSE)
        if rle is None:
            raise RuntimeInputError(
                public_message=(
                    "Detected runtime parameter declared to hold RLE instance "
                    "segmentation, but a prediction is missing the "
                    f"`{RLE_MASK_KEY_IN_INFERENCE_RESPONSE}` entry."
                ),
                context="workflow_execution | runtime_input_validation",
            )
        coco_rle_masks.append(rle)
    image_size: Tuple[int, int]
    if coco_rle_masks and coco_rle_masks[0].get("size") is not None:
        size = coco_rle_masks[0]["size"]
        image_size = (int(size[0]), int(size[1]))
    else:
        image_size = (int(image_height), int(image_width))
    return InstancesRLEMasks.from_coco_rle_masks(
        image_size=image_size,
        masks=coco_rle_masks,
    )


def _build_placeholder_image(
    parameter: str,
    detections: dict,
    raw_predictions: List[dict],
) -> WorkflowImageData:
    """Build a placeholder ``WorkflowImageData`` carrying the serialised image
    dimensions and the parent/root lineage embedded in the predictions.

    ``native_detections_from_inference_predictions`` only reads the image to derive
    ``image_metadata`` (dimensions + parent/root lineage), so a zero CHW tensor on
    ``WORKFLOWS_IMAGE_TENSOR_DEVICE`` is sufficient and avoids materialising any
    real pixels. Lineage is shared across detections of a single image, so it is
    read from the first prediction that carries it.
    """
    image_height, image_width = _read_serialized_image_shape(detections)
    tensor_image = torch.zeros(
        (3, int(image_height), int(image_width)),
        dtype=torch.uint8,
        device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
    )
    parent_metadata, root_parent_metadata = _reconstruct_lineage(
        parameter=parameter,
        raw_predictions=raw_predictions,
        image_height=int(image_height),
        image_width=int(image_width),
    )
    return WorkflowImageData(
        parent_metadata=parent_metadata,
        workflow_root_ancestor_metadata=root_parent_metadata,
        tensor_image=tensor_image,
    )


def _read_serialized_image_shape(detections: dict) -> Tuple[int, int]:
    image = detections.get("image") or {}
    height = image.get("height")
    width = image.get("width")
    if height is None or width is None:
        # numpy parity: a serialised detections payload always carries image dims;
        # fall back to a 1x1 canvas so metadata can still be built for empty inputs.
        return 1, 1
    return int(height), int(width)


def _reconstruct_lineage(
    parameter: str,
    raw_predictions: List[dict],
    image_height: int,
    image_width: int,
) -> Tuple[ImageParentMetadata, ImageParentMetadata]:
    parent_id = parameter
    root_parent_id = parameter
    parent_origin = None
    root_parent_origin = None
    for prediction in raw_predictions:
        if not isinstance(prediction, dict):
            continue
        parent_id = prediction.get(PARENT_ID_KEY, parent_id)
        root_parent_id = prediction.get(ROOT_PARENT_ID_KEY, root_parent_id)
        parent_origin = prediction.get(PARENT_ORIGIN_KEY, parent_origin)
        root_parent_origin = prediction.get(ROOT_PARENT_ORIGIN_KEY, root_parent_origin)
        if parent_origin is not None or root_parent_origin is not None:
            break
    parent_metadata = ImageParentMetadata(
        parent_id=parent_id,
        origin_coordinates=_origin_coordinates_from_serialized(
            origin=parent_origin,
            image_height=image_height,
            image_width=image_width,
        ),
    )
    root_parent_metadata = ImageParentMetadata(
        parent_id=root_parent_id,
        origin_coordinates=_origin_coordinates_from_serialized(
            origin=root_parent_origin,
            image_height=image_height,
            image_width=image_width,
        ),
    )
    return parent_metadata, root_parent_metadata


def _origin_coordinates_from_serialized(
    origin: Optional[dict],
    image_height: int,
    image_width: int,
) -> OriginCoordinatesSystem:
    """Map a serialised ``ParentOrigin`` dict back to an ``OriginCoordinatesSystem``.

    When the prediction carried no origin (top-level image, not a crop), the origin
    defaults to the image's own dimensions at offset (0, 0) - matching how the
    producer-side ``build_native_image_metadata`` reads a non-crop image's lineage.
    """
    if isinstance(origin, dict):
        return OriginCoordinatesSystem(
            left_top_x=int(origin.get("offset_x", 0)),
            left_top_y=int(origin.get("offset_y", 0)),
            origin_width=int(origin.get("width", image_width)),
            origin_height=int(origin.get("height", image_height)),
        )
    return OriginCoordinatesSystem(
        left_top_x=0,
        left_top_y=0,
        origin_width=int(image_width),
        origin_height=int(image_height),
    )


def deserialize_native_embedding_kind(parameter: str, value: Any) -> torch.Tensor:
    """Tensor-native deserialiser for the embedding kind.

    The numpy branch registers no deserialiser for embeddings, so a serialised
    embedding (a JSON ``List[float]`` — the inverse of ``serialise_native_embedding``,
    which emits ``value.detach().cpu().tolist()``) would reach a tensor consumer as a
    plain ``list`` and break on ``.shape`` / ``torch.dot`` (e.g. ``cosine_similarity``).
    This rebuilds a 1-D ``torch.Tensor`` on ``WORKFLOWS_IMAGE_TENSOR_DEVICE``.
    """
    return _tensor_from_serialized(parameter=parameter, value=value)


def deserialize_native_tensor_kind(parameter: str, value: Any) -> torch.Tensor:
    """Tensor-native deserialiser for the tensor kind.

    Inverse of ``serialise_native_tensor`` (``value.detach().cpu().tolist()``); rebuilds
    a (possibly N-D) ``torch.Tensor`` from the serialised nested-list JSON so tensor
    consumers receive a real tensor rather than a nested ``list``. Same tensor device as
    the producers (``WORKFLOWS_IMAGE_TENSOR_DEVICE``).
    """
    return _tensor_from_serialized(parameter=parameter, value=value)


def _tensor_from_serialized(parameter: str, value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(WORKFLOWS_IMAGE_TENSOR_DEVICE)
    if not isinstance(value, (list, tuple)):
        raise RuntimeInputError(
            public_message=(
                f"Detected runtime parameter `{parameter}` declared to hold an "
                f"embedding / tensor value, but found {type(value)}; expected a JSON "
                f"list of numbers (optionally nested) or a torch.Tensor."
            ),
            context="workflow_execution | runtime_input_validation",
        )
    try:
        return torch.as_tensor(
            value, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        )
    except (ValueError, TypeError) as error:
        raise RuntimeInputError(
            public_message=(
                f"Detected runtime parameter `{parameter}` declared to hold an "
                f"embedding / tensor value, but its list could not be converted to a "
                f"torch.Tensor: {error}"
            ),
            context="workflow_execution | runtime_input_validation",
        ) from error


def deserialize_native_classification_prediction_kind(
    parameter: str,
    value: Any,
) -> Union[ClassificationPrediction, MultiLabelClassificationPrediction]:
    """Tensor-native sibling of the numpy ``deserialize_classification_prediction_kind``.

    The numpy path returns the classification dict unchanged; on the tensor branch every
    consumer (e.g. the UQL ``*_tensor_native`` classification extractors) expects a native
    ``inference_models.ClassificationPrediction`` (single-label) or
    ``MultiLabelClassificationPrediction`` (multi-label). This rebuilds the native object
    from the serialised classification dict — the same shape
    ``serialise_native_classification`` emits — so a round-trip is byte-faithful for a
    canonical model-emitted prediction (see the module-level caveat below).

    Single-label (``top``/``confidence`` present): a dense ``confidence`` vector
    ``(1, num_classes)`` indexed by ``class_id`` is rebuilt from ``predictions`` and the
    top-1 ``class_id`` is recovered from ``top``. Multi-label (``predicted_classes``
    present): a dense ``confidence`` vector ``(num_classes,)`` plus the predicted
    ``class_ids`` are rebuilt from the ``predictions`` map. Both carry the
    ``class_id -> name`` map and image lineage on the metadata (``CLASS_NAMES_KEY`` etc.)
    that the serializer and consumers read.

    BYTE-PARITY CAVEAT: perfect ``serialise(deserialize(x)) == x`` identity only holds for
    a canonical model-emitted classification prediction — the full class distribution with
    contiguous 0-based ``class_id`` values, already sorted desc and rounded. Owner-2's
    ``serialise_native_classification`` re-enumerates the whole confidence vector, re-sorts
    desc and re-rounds; a sparse / top-K / out-of-list (``class_id == -1``) / non-contiguous
    input cannot be represented losslessly in the native (dense, index==class_id) structure
    and will normalise rather than preserve. Flagged in the owner report.
    """
    if isinstance(value, (ClassificationPrediction, MultiLabelClassificationPrediction)):
        return value
    value = _validate_classification_dict(parameter=parameter, value=value)
    if "predicted_classes" in value:
        return _build_multi_label_prediction(value=value)
    return _build_single_label_prediction(value=value)


def _validate_classification_dict(parameter: str, value: Any) -> dict:
    if not isinstance(value, dict):
        raise RuntimeInputError(
            public_message=(
                f"Detected runtime parameter `{parameter}` declared to hold a "
                f"classification prediction, but found {type(value)}; expected a dict."
            ),
            context="workflow_execution | runtime_input_validation",
        )
    if "image" not in value or "predictions" not in value:
        raise RuntimeInputError(
            public_message=(
                f"Detected runtime parameter `{parameter}` declared to hold a "
                f"classification prediction, but the dict misses required keys "
                f"('image', 'predictions')."
            ),
            context="workflow_execution | runtime_input_validation",
        )
    if "predicted_classes" not in value and (
        "top" not in value or "confidence" not in value
    ):
        raise RuntimeInputError(
            public_message=(
                f"Detected runtime parameter `{parameter}` declared to hold a "
                f"classification prediction, but the value misses prediction details "
                f"(neither 'predicted_classes' nor 'top'/'confidence' present)."
            ),
            context="workflow_execution | runtime_input_validation",
        )
    return value


def _dense_confidence_vector(
    class_id_to_confidence: Dict[int, float],
) -> List[float]:
    """Build a dense confidence list indexed by ``class_id``.

    Length is ``max(class_id) + 1`` for the canonical contiguous 0-based case; class ids
    outside ``[0, length)`` (e.g. an out-of-list ``-1``) cannot be positioned and are
    dropped from the dense vector (their name still lives in ``CLASS_NAMES_KEY``).
    """
    if not class_id_to_confidence:
        return []
    highest_class_id = max(class_id_to_confidence)
    length = highest_class_id + 1 if highest_class_id >= 0 else 0
    vector = [0.0] * length
    for class_id, confidence in class_id_to_confidence.items():
        if 0 <= class_id < length:
            vector[class_id] = confidence
    return vector


def _build_classification_image_metadata(
    value: dict,
    class_names_mapping: Dict[int, str],
) -> dict:
    """Assemble the per-image ``image_metadata`` that the serializer and the UQL
    ``*_tensor_native`` classification extractors read back."""
    metadata: dict = {CLASS_NAMES_KEY: class_names_mapping}
    image = value.get("image") or {}
    height, width = image.get("height"), image.get("width")
    if height is not None and width is not None:
        metadata[IMAGE_DIMENSIONS_KEY] = [int(height), int(width)]
    for key in (
        INFERENCE_ID_KEY,
        "time",
        PREDICTION_TYPE_KEY,
        PARENT_ID_KEY,
        ROOT_PARENT_ID_KEY,
    ):
        if value.get(key) is not None:
            metadata[key] = value[key]
    return metadata


def _build_single_label_prediction(value: dict) -> ClassificationPrediction:
    predictions = value["predictions"]
    if not isinstance(predictions, list):
        raise RuntimeInputError(
            public_message=(
                "Detected a single-label classification prediction (with 'top'), but "
                "'predictions' is not a list of per-class entries."
            ),
            context="workflow_execution | runtime_input_validation",
        )
    class_names_mapping: Dict[int, str] = {}
    class_id_to_confidence: Dict[int, float] = {}
    for entry in predictions:
        class_id = int(entry["class_id"])
        class_names_mapping[class_id] = str(entry[CLASS_NAME_KEY])
        class_id_to_confidence[class_id] = float(entry["confidence"])
    top_name = value.get("top")
    top_class_id = next(
        (
            class_id
            for class_id, name in class_names_mapping.items()
            if name == top_name
        ),
        None,
    )
    if top_class_id is None:
        top_class_id = (
            max(class_id_to_confidence, key=class_id_to_confidence.get)
            if class_id_to_confidence
            else 0
        )
    confidence_vector = _dense_confidence_vector(class_id_to_confidence)
    image_metadata = _build_classification_image_metadata(
        value=value, class_names_mapping=class_names_mapping
    )
    return ClassificationPrediction(
        class_id=torch.tensor(
            [top_class_id], dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.tensor(
            [confidence_vector], dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        images_metadata=[image_metadata],
    )


def _build_multi_label_prediction(value: dict) -> MultiLabelClassificationPrediction:
    predictions = value["predictions"]
    if not isinstance(predictions, dict):
        raise RuntimeInputError(
            public_message=(
                "Detected a multi-label classification prediction (with "
                "'predicted_classes'), but 'predictions' is not a name->entry map."
            ),
            context="workflow_execution | runtime_input_validation",
        )
    class_names_mapping: Dict[int, str] = {}
    class_id_to_confidence: Dict[int, float] = {}
    name_to_class_id: Dict[str, int] = {}
    for name, entry in predictions.items():
        class_id = int(entry["class_id"])
        class_names_mapping[class_id] = str(name)
        class_id_to_confidence[class_id] = float(entry["confidence"])
        name_to_class_id[str(name)] = class_id
    predicted_class_ids = [
        name_to_class_id[str(name)]
        for name in value.get("predicted_classes", [])
        if str(name) in name_to_class_id
    ]
    confidence_vector = _dense_confidence_vector(class_id_to_confidence)
    image_metadata = _build_classification_image_metadata(
        value=value, class_names_mapping=class_names_mapping
    )
    return MultiLabelClassificationPrediction(
        class_ids=torch.tensor(
            predicted_class_ids, dtype=torch.long, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        confidence=torch.tensor(
            confidence_vector, dtype=torch.float32, device=WORKFLOWS_IMAGE_TENSOR_DEVICE
        ),
        image_metadata=image_metadata,
    )


__all__ = [
    "deserialize_image_kind",
    "deserialize_detections_kind",
    "deserialize_rle_detections_kind",
    "deserialize_native_classification_prediction_kind",
    "deserialize_native_embedding_kind",
    "deserialize_native_tensor_kind",
]
