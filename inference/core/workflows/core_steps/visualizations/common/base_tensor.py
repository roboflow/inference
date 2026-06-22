from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field

from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    TensorNativePrediction,
    split_key_point_prediction,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    TRACKER_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.keypoints_detection import KeyPoints
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_numpy_mask

OUTPUT_IMAGE_KEY: str = "image"

#: ``sv.Detections.data`` column carrying the resolved class-name string. This is
#: the supervision-native key the annotators (e.g. ``sv.LabelAnnotator``) and the
#: numpy visualisers read via ``predictions["class_name"]``.
CLASS_NAME_DATA_FIELD: str = "class_name"


def to_supervision_for_annotation(
    prediction: Union[TensorNativePrediction, TensorNativeDetections],
    materialise_masks: bool = True,
) -> sv.Detections:
    """Materialise a tensor-native prediction into an ``sv.Detections`` carrying
    everything the supervision annotators read.

    ``materialise_masks=False`` skips the dense-mask materialisation entirely
    (``mask`` stays ``None``) for annotators that never read ``.mask`` — e.g. the
    label annotator — avoiding the device->host mask transfer/decode for them.

    This is the single, sanctioned native -> sv conversion used by the
    visualisation block siblings: the annotators (``sv.BoxAnnotator`` /
    ``sv.LabelAnnotator`` / ``sv.MaskAnnotator`` / ...) require an
    ``sv.Detections``, and the visualiser output is an annotated image (never a
    native detection), so there is no round-trip back to a native object.

    The reconstructed ``sv.Detections`` carries:

    * ``xyxy`` / ``class_id`` / ``confidence`` (and a dense boolean ``mask`` for
      instance segmentation, materialised in a single bulk transfer/decode for
      the whole stack),
    * ``tracker_id`` (from ``bboxes_metadata[i]["tracker_id"]`` when present),
    * ``data["class_name"]`` resolved from ``image_metadata["class_names"]``
      (``{int class_id: str name}``), falling back to ``f"class_{id}"``,
    * ``data[DETECTION_ID_KEY]`` (from ``bboxes_metadata``),
    * ``data[IMAGE_DIMENSIONS_KEY]`` (broadcast from ``image_metadata``), and
    * any extra per-box ``bboxes_metadata`` keys (``time_in_zone``,
      ``area``-derived keys, etc.) that specific annotators consume.

    For the keypoint-detection tuple input, the bounding-box component is used
    (the keypoint annotators take an ``sv.KeyPoints`` from the native
    ``KeyPoints`` component via its own ``.to_supervision()``, separately).
    """
    if isinstance(prediction, tuple):
        _, detections = split_key_point_prediction(prediction)
    elif isinstance(prediction, KeyPoints):
        raise ValueError(
            "A bare `KeyPoints` prediction (without its bounding-box component) "
            "cannot be visualised by this block: the supervision annotators "
            "require the bounding-box `Detections`. Provide the keypoint-detection "
            "tuple `(KeyPoints, Detections)` instead."
        )
    else:
        detections = prediction
    image_metadata = detections.image_metadata or {}
    detections_number = int(detections.xyxy.shape[0])
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is None:
        bboxes_metadata = [{} for _ in range(detections_number)]
    class_names_mapping = image_metadata.get(CLASS_NAMES_KEY) or {}
    xyxy = detections.xyxy.detach().cpu().numpy().astype(np.float32)
    class_id = detections.class_id.detach().cpu().numpy().astype(int)
    confidence = detections.confidence.detach().cpu().numpy().astype(np.float32)
    mask = (
        _materialise_mask(detections, detections_number)
        if materialise_masks
        else None
    )
    tracker_id = _materialise_tracker_id(bboxes_metadata)
    data = _materialise_data(
        bboxes_metadata=bboxes_metadata,
        class_id=class_id,
        class_names_mapping=class_names_mapping,
        image_metadata=image_metadata,
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


def _materialise_mask(
    detections: TensorNativeDetections,
    detections_number: int,
) -> Optional[np.ndarray]:
    if not isinstance(detections, InstanceDetections):
        return None
    if detections_number == 0:
        return None
    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        # RLE: decode every instance in one call (never one-at-a-time) -> (N, H, W) bool.
        return coco_rle_masks_to_numpy_mask(mask)
    # Dense torch.Tensor (N, H, W): a single bulk device->host transfer instead of
    # N per-instance `.detach().to("cpu").numpy()` round-trips (each a blocking CUDA
    # sync). Mirrors `InstanceDetections.to_supervision`'s dense branch.
    return mask.detach().cpu().numpy().astype(bool)


def _materialise_tracker_id(
    bboxes_metadata: List[dict],
) -> Optional[np.ndarray]:
    tracker_ids = [data.get(TRACKER_ID_KEY) for data in bboxes_metadata]
    if any(tracker_id is None for tracker_id in tracker_ids):
        return None
    return np.asarray([int(tracker_id) for tracker_id in tracker_ids])


def _materialise_data(
    bboxes_metadata: List[dict],
    class_id: np.ndarray,
    class_names_mapping: dict,
    image_metadata: dict,
    detections_number: int,
) -> dict:
    class_names = [
        _resolve_class_name(int(value), class_names_mapping) for value in class_id
    ]
    data: dict = {CLASS_NAME_DATA_FIELD: np.asarray(class_names, dtype=object)}
    detection_ids = [
        str(per_box.get(DETECTION_ID_KEY, "")) for per_box in bboxes_metadata
    ]
    data[DETECTION_ID_KEY] = np.asarray(detection_ids, dtype=object)
    image_dimensions = image_metadata.get(IMAGE_DIMENSIONS_KEY)
    if image_dimensions is not None:
        data[IMAGE_DIMENSIONS_KEY] = np.asarray(
            [list(image_dimensions) for _ in range(detections_number)]
        )
    extra_keys = set()
    for per_box in bboxes_metadata:
        extra_keys.update(per_box.keys())
    extra_keys.discard(DETECTION_ID_KEY)
    extra_keys.discard(TRACKER_ID_KEY)
    for key in extra_keys:
        data[key] = np.asarray(
            [per_box.get(key) for per_box in bboxes_metadata], dtype=object
        )
    return data


def _resolve_class_name(class_id: int, class_names_mapping: dict) -> str:
    class_name = class_names_mapping.get(class_id)
    if class_name is None:
        return f"class_{class_id}"
    return str(class_name)


class VisualizationManifest(WorkflowBlockManifest, ABC):
    model_config = ConfigDict(
        json_schema_extra={
            "license": "Apache-2.0",
            "block_type": "visualization",
        }
    )
    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="The image to visualize on.",
        examples=["$inputs.image", "$steps.cropping.crops"],
        validation_alias=AliasChoices("image", "images"),
    )
    copy_image: Union[bool, Selector(kind=[BOOLEAN_KIND])] = Field(  # type: ignore
        description="Enable this option to create a copy of the input image for visualization, preserving the original. Use this when stacking multiple visualizations.",
        default=True,
        examples=[True, False],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name=OUTPUT_IMAGE_KEY,
                kind=[
                    IMAGE_KIND,
                ],
            ),
        ]


class VisualizationBlock(WorkflowBlock, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[VisualizationManifest]:
        pass

    @abstractmethod
    def getAnnotator(self, *args, **kwargs) -> sv.annotators.base.BaseAnnotator:
        pass

    @abstractmethod
    def run(
        self, image: WorkflowImageData, copy_image: bool, *args, **kwargs
    ) -> BlockResult:
        pass


class PredictionsVisualizationManifest(VisualizationManifest, ABC):
    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Model predictions to visualize.",
        examples=["$steps.object_detection_model.predictions"],
    )


class PredictionsVisualizationBlock(VisualizationBlock, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    @abstractmethod
    def get_manifest(cls) -> Type[VisualizationManifest]:
        pass

    @abstractmethod
    def getAnnotator(self, *args, **kwargs) -> sv.annotators.base.BaseAnnotator:
        pass

    @abstractmethod
    def run(
        self,
        image: WorkflowImageData,
        predictions: Union[TensorNativePrediction, TensorNativeDetections],
        copy_image: bool,
        *args,
        **kwargs,
    ) -> BlockResult:
        pass
