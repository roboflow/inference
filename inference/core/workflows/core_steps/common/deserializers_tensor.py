"""
Tensor-native sibling of `common/deserializers.py`. Same public function
names; loader swaps the import based on `ENABLE_TENSOR_DATA_REPRESENTATION`.

Per the plan's locked decision [ITERATE 4.A], the numpy file is left
untouched. Functions here add tensor-aware code paths and delegate to
the numpy implementations for everything else.
"""

from typing import Any, List, Optional, Tuple

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
    PARENT_ID_KEY,
    PARENT_ORIGIN_KEY,
    RLE_MASK_KEY_IN_INFERENCE_RESPONSE,
    ROOT_PARENT_ID_KEY,
    ROOT_PARENT_ORIGIN_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

DEFAULT_OBJECT_DETECTION_PREDICTION_TYPE = "object-detection"
DEFAULT_INSTANCE_SEGMENTATION_PREDICTION_TYPE = "instance-segmentation"


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
            tensor_image=image,
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
            tensor_image=value,
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


__all__ = [
    "deserialize_image_kind",
    "deserialize_detections_kind",
    "deserialize_rle_detections_kind",
]
