from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import numpy as np
import supervision as sv
import torch
from pydantic import AliasChoices, ConfigDict, Field
from supervision.detection.compact_mask import CompactMask

from inference.core.env import WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS
from inference.core.workflows.core_steps.common.rle_compact import (
    instances_rle_to_compact_mask,
)
from inference.core.workflows.core_steps.common.tensor_native import (
    HOST_MIRROR_KEYS,
    TensorNativeDetections,
    TensorNativePrediction,
    read_host_mirror,
    split_key_point_prediction,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
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

OUTPUT_IMAGE_KEY: str = "image"


def predictions_are_empty(detections) -> bool:
    """True when there is provably nothing to draw (no boxes at all)."""
    if detections is None:
        return True
    xyxy = getattr(detections, "xyxy", None)
    if xyxy is None:
        return True
    try:
        return int(xyxy.shape[0]) == 0
    except (AttributeError, TypeError):
        try:
            return len(xyxy) == 0
        except TypeError:
            return False


def empty_predictions_passthrough(
    image: WorkflowImageData, detections, copy_image: bool
) -> Optional[dict]:
    """``{OUTPUT_IMAGE_KEY: ...}`` when there is nothing to draw, else ``None``.

    With no detections every annotator is a no-op, but the sv fallback still
    pays ``image.numpy_image`` first - on the tensor pipeline that is a
    full-resolution device->host materialisation (~30 ms per 2K frame under
    load on Orin NX, measured on the ~60% of live camera frames with no
    detections - the dominant cost of visualization blocks in that regime).
    Passing the input representation through keeps the exact output
    semantics of an empty annotate (an independent copy when
    ``copy_image=True``, shared backing otherwise) without ever leaving the
    device.
    """
    if not predictions_are_empty(detections):
        return None
    if image.is_tensor_materialised():
        tensor = image.tensor_image
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                tensor_image=tensor.clone() if copy_image else tensor,
            )
        }
    numpy_image = image.numpy_image
    return {
        OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
            origin_image_data=image,
            numpy_image=numpy_image.copy() if copy_image else numpy_image,
        )
    }


def resolve_overlap_winners(
    flat: torch.Tensor,
    priority: torch.Tensor,
    num_cells: int,
    num_candidates: int,
) -> torch.Tensor:
    """Later-wins ownership resolution for one flat indexed store.

    ``flat`` (int64, values in ``[0, num_cells)``) holds the destination cell
    of every painted pixel; ``priority`` (int32, values in
    ``[0, num_candidates)``) is the paint order. Returns, per painted pixel,
    the winning priority of its cell — the amax over every pixel targeting
    that cell — as an int64 tensor provably within ``[0, num_candidates)``,
    ready to index a per-candidate color table.

    The owner buffer starts as a full ``-1`` sentinel and the scatter uses
    ``include_self=True``, so the reduction at each cell is the documented
    ``amax({-1} ∪ {priorities targeting the cell})`` — well defined for any
    duplication pattern. Gathered cells are exactly the scattered cells, so
    every gathered value is a real priority, never the sentinel.

    The previous formulation — ``torch.empty`` + ``include_self=False`` —
    leaned on two undocumented implementation details: PyTorch's internal
    pre-fill of the dtype minimum at scattered cells, and that pre-fill
    staying ordered before the reduce kernel. If either breaks (any leak of
    an uninitialized or ``INT32_MIN`` value into the gather), the winner
    becomes an out-of-range color index and the paint dies in an
    asynchronous CUDA device assert — SIGABRT with no Python traceback.

    Defence in depth on top of the provable formulation: the env-gated
    strict mode (``WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS``) syncs
    and raises a detailed error on any out-of-range scatter index or owner,
    and the final clamp keeps the color gather in-bounds even under a
    kernel malfunction (one mispainted pixel instead of a dead process).
    """
    if WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS:
        _validate_scatter_indices(flat, num_cells)
    owner = torch.full((num_cells,), -1, dtype=torch.int32, device=flat.device)
    owner.scatter_reduce_(0, flat, priority, reduce="amax", include_self=True)
    winners = owner[flat].long()
    if WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS:
        _validate_winners(winners, num_candidates)
    return winners.clamp_(0, num_candidates - 1)


def _validate_scatter_indices(flat: torch.Tensor, num_cells: int) -> None:
    if int(flat.numel()) == 0:
        return
    lo, hi = int(flat.min().item()), int(flat.max().item())
    if lo < 0 or hi >= num_cells:
        raise RuntimeError(
            f"overlap resolver received out-of-range pixel indices: "
            f"min={lo}, max={hi}, valid range [0, {num_cells})"
        )


def _validate_winners(winners: torch.Tensor, num_candidates: int) -> None:
    bad = (winners < 0) | (winners >= num_candidates)
    if bool(bad.any().item()):
        bad_values = winners[bad]
        raise RuntimeError(
            f"overlap resolver produced {int(bad.sum().item())} out-of-range "
            f"owners (valid range [0, {num_candidates})): first values "
            f"{bad_values[:8].tolist()} — scatter_reduce amax returned a "
            f"value never scattered, which would have been an asynchronous "
            f"CUDA device assert in the color gather"
        )


#: ``sv.Detections.data`` key the supervision annotators read class names from.
CLASS_NAME_DATA_FIELD: str = "class_name"


def to_supervision_for_annotation(
    prediction: Union[TensorNativePrediction, TensorNativeDetections],
    materialise_masks: bool = True,
) -> sv.Detections:
    """Materialise a tensor-native prediction into an ``sv.Detections`` carrying
    everything the supervision annotators read.

    ``materialise_masks=False`` leaves ``mask`` as ``None``, skipping the
    device->host mask transfer/decode for annotators that never read it.

    The reconstructed ``sv.Detections`` carries:

    * ``xyxy`` / ``class_id`` / ``confidence`` (plus ``mask`` for instance
      segmentation),
    * ``tracker_id`` (from ``bboxes_metadata[i]["tracker_id"]`` when present),
    * ``data["class_name"]`` resolved from ``image_metadata["class_names"]``
      (``{int class_id: str name}``), falling back to ``f"class_{id}"``,
    * ``data[DETECTION_ID_KEY]`` (from ``bboxes_metadata``),
    * ``data[IMAGE_DIMENSIONS_KEY]`` (broadcast from ``image_metadata``), and
    * any extra per-box ``bboxes_metadata`` keys (``time_in_zone``,
      ``area``-derived keys, etc.) that specific annotators consume.

    For the keypoint-detection tuple input, only the bounding-box component is
    converted.
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
    # Prefer the per-box host mirror written by
    # ``attach_native_detection_metadata``: when EVERY box carries it, the sv
    # view's xyxy/class_id/confidence are assembled on the host with ZERO device
    # reads — a ``.cpu()`` here queues behind unrelated kernels on the default
    # CUDA stream (up to ~65 ms per batch measured on Jetson under load).
    # Mirror-less predictions (remote/deserialized/transformed) keep the
    # tensor-read path unchanged; both paths produce bit-identical arrays.
    host_mirror = read_host_mirror(bboxes_metadata, detections_number)
    if host_mirror is not None:
        xyxy, class_id, confidence = host_mirror
    else:
        xyxy = detections.xyxy.detach().cpu().numpy().astype(np.float32)
        class_id = detections.class_id.detach().cpu().numpy().astype(int)
        confidence = detections.confidence.detach().cpu().numpy().astype(np.float32)
    mask = (
        _materialise_mask(detections, detections_number, xyxy)
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
    xyxy: np.ndarray,
) -> Optional[Union[np.ndarray, CompactMask]]:
    if not isinstance(detections, InstanceDetections):
        return None
    if detections_number == 0:
        return None
    mask = detections.mask
    if isinstance(mask, InstancesRLEMasks):
        # RLE is transcoded without decoding the full-frame (N, H, W) boolean
        # stack; the boxes provide the per-crop bounds.
        return instances_rle_to_compact_mask(mask, xyxy)
    # Dense (N, H, W) masks: one bulk device->host transfer instead of N
    # per-instance round-trips (each a blocking CUDA sync).
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
    # Per-image lineage that flag-off carries per-box in ``sv.Detections.data``
    # (the single per-image value broadcast to every row). The tensor-native
    # path stores these once in ``image_metadata``; re-broadcast them here so a
    # custom-text Label lookup (e.g. ``predictions["parent_id"]``) resolves
    # identically to flag-off instead of raising ``KeyError``. Only keys present
    # on this image are emitted, matching flag-off (which omits ``inference_id``
    # when the model did not supply one). Fixed-anchor annotators never read
    # these, so adding them leaves their rendered output unchanged.
    for lineage_key in (
        PARENT_ID_KEY,
        ROOT_PARENT_ID_KEY,
        PREDICTION_TYPE_KEY,
        INFERENCE_ID_KEY,
    ):
        if lineage_key in image_metadata:
            data[lineage_key] = np.asarray(
                [image_metadata[lineage_key]] * detections_number, dtype=object
            )
    extra_keys = set()
    for per_box in bboxes_metadata:
        extra_keys.update(per_box.keys())
    extra_keys.discard(DETECTION_ID_KEY)
    extra_keys.discard(TRACKER_ID_KEY)
    # The private host mirror of the box tensors is an internal transport
    # channel, not per-box user data — it must never surface in ``.data``.
    extra_keys.difference_update(HOST_MIRROR_KEYS)
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
