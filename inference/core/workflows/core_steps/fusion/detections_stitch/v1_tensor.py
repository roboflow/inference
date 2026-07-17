from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
import torch
import torchvision
from pydantic import ConfigDict, Field
from supervision import OverlapFilter
from supervision.config import ORIENTED_BOX_COORDINATES

from inference.core import logger
from inference.core.workflows.core_steps.common.tensor_native import (
    embed_rle_masks_in_larger_canvas,
    take_prediction_by_indices,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

TensorNativeDetections = Union[Detections, InstanceDetections]

LONG_DESCRIPTION = """
Merge detections from multiple image slices or crops back into a single unified detection result by converting coordinates from slice/crop space to original image coordinates, combining all detections, and optionally filtering overlapping detections to enable SAHI workflows, multi-stage detection pipelines, and coordinate-space merging workflows where detections from sub-images need to be reconstructed as if they were detected on the original image.

## How This Block Works

This block merges detections that were made on multiple sub-parts (slices or crops) of the same input image, reconstructing them as a single detection result in the original image coordinate space. The block:

1. Receives reference image and slice/crop predictions:
   - Takes the original reference image that was sliced or cropped
   - Receives predictions from detection models that processed each slice/crop
   - Predictions must contain parent coordinate metadata indicating slice/crop position
2. Retrieves crop offsets for each detection:
   - Extracts parent coordinates from each detection's metadata
   - Gets the offset (x, y position) indicating where each slice/crop was located in the original image
   - Uses this offset to transform coordinates from slice space to original image space
3. Manages crop metadata:
   - Updates image dimensions in detection metadata to match reference image dimensions
   - Validates that detections were not scaled (scaled detections are not supported)
   - Attaches parent coordinate information to detections for proper coordinate transformation
4. Transforms coordinates to original image space:
   - Moves bounding box coordinates (xyxy) from slice/crop coordinates to original image coordinates
   - Transforms segmentation masks from slice/crop space to original image space (if present)
   - Applies offset to align detections with their position in the original image
5. Merges all transformed detections:
   - Combines all re-aligned detections from all slices/crops into a single detection result
   - Creates unified detection output containing all detections from all sub-images
6. Applies overlap filtering (optional):
   - **None strategy**: Returns all merged detections without filtering (may contain duplicates from overlapping slices)
   - **NMS (Non-Maximum Suppression)**: Removes lower-confidence detections when IoU exceeds threshold, keeping only the highest confidence detection for each overlapping region
   - **NMM (Non-Maximum Merge)**: Combines overlapping detections instead of discarding them, merging detections that exceed IoU threshold
7. Returns merged detections:
   - Outputs unified detection result in original image coordinate space
   - Reduces dimensionality by 1 (multiple slice detections → single image detections)
   - All detections are now referenced to the original image dimensions and coordinates

This block is essential for SAHI (Slicing Adaptive Inference) workflows where an image is sliced, each slice is processed separately, and results need to be merged back. Overlapping slices can produce duplicate detections for the same object, so overlap filtering (NMS/NMM) helps clean up these duplicates. The coordinate transformation ensures that detection coordinates are correctly positioned relative to the original image, not the slices.

## Common Use Cases

- **SAHI Workflows**: Complete SAHI technique by merging detections from image slices back to original image coordinates (e.g., merge slice detections from SAHI processing, reconstruct full-image detections from slices, combine small object detection results), enabling SAHI detection workflows
- **Multi-Stage Detection**: Merge detections from secondary high-resolution models applied to dynamically cropped regions (e.g., coarse detection → crop → precise detection → merge, two-stage detection pipelines, hierarchical detection workflows), enabling multi-stage detection workflows
- **Small Object Detection**: Combine detection results from sliced images processed separately for small object detection (e.g., merge detections from aerial image slices, combine slice detection results, reconstruct detections from tiled images), enabling small object detection workflows
- **High-Resolution Processing**: Merge detections from high-resolution images processed in smaller chunks (e.g., merge detections from satellite image tiles, combine results from medical image regions, reconstruct detections from large image segments), enabling high-resolution detection workflows
- **Coordinate Space Unification**: Convert detections from multiple coordinate spaces (slice/crop space) to a single unified coordinate space (original image space) for consistent processing (e.g., unify detection coordinates, merge coordinate spaces, standardize detection positions), enabling coordinate unification workflows
- **Overlapping Region Handling**: Handle duplicate detections from overlapping slices or crops by applying overlap filtering (e.g., remove duplicate detections from overlapping slices, merge overlapping detections, clean up overlapping results), enabling overlap resolution workflows

## Connecting to Other Blocks

This block receives slice/crop predictions and reference images, and produces merged detections:

- **After detection models in SAHI workflows** following Image Slicer → Detection Model → Detections Stitch pattern to merge slice detections (e.g., merge SAHI slice detections, reconstruct full-image detections, combine slice results), enabling SAHI completion workflows
- **After secondary detection models** in multi-stage pipelines following Dynamic Crop → Detection Model → Detections Stitch pattern to merge cropped detections (e.g., merge cropped region detections, combine two-stage detection results, unify multi-stage outputs), enabling multi-stage detection workflows
- **Before visualization blocks** to visualize merged detection results on the original image (e.g., visualize merged detections, display stitched results, show unified detection output), enabling visualization workflows
- **Before filtering or analytics blocks** to process merged detection results (e.g., filter merged detections, analyze stitched results, process unified outputs), enabling analysis workflows
- **Before sink or storage blocks** to store or export merged detection results (e.g., save merged detections, export stitched results, store unified outputs), enabling storage workflows
- **In workflow outputs** to provide merged detections as final workflow output (e.g., return merged detections, output stitched results, provide unified detection output), enabling output workflows

## Requirements

This block requires a reference image (the original image that was sliced/cropped) and predictions from detection models that processed slices/crops. The predictions must contain parent coordinate metadata (PARENT_COORDINATES_KEY) indicating the position of each slice/crop in the original image. The block does not support scaled detections (detections that were resized relative to the parent image). Predictions should be from object detection or instance segmentation models. The block supports three overlap filtering strategies: "none" (no filtering, may include duplicates), "nms" (Non-Maximum Suppression, removes lower-confidence overlapping detections, default), and "nmm" (Non-Maximum Merge, combines overlapping detections). The IoU threshold (default 0.3) determines when detections are considered overlapping for filtering purposes. For more information on SAHI technique, see: https://ieeexplore.ieee.org/document/9897990.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Stitch",
            "version": "v1",
            "short_description": "Merges detections made against multiple pieces of input image into single detection.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "fusion",
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-reel",
                "blockPriority": 10,
                "supervision": True,
            },
        }
    )
    type: Literal["roboflow_core/detections_stitch@v1"]
    reference_image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Original reference image that was sliced or cropped to produce the input predictions. This image is used to determine the target coordinate space and image dimensions for the merged detections. All detection coordinates will be transformed to match this reference image's coordinate system. The same image that was provided to Image Slicer or Dynamic Crop blocks should be used here to ensure proper coordinate alignment.",
        examples=["$inputs.image", "$steps.input_image.output"],
    )
    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Model predictions (object detection or instance segmentation) from detection models that processed image slices or crops. These predictions must contain parent coordinate metadata indicating the position of each slice/crop in the original image. Predictions are collected from multiple slices/crops and merged into a single unified detection result. The block converts coordinates from slice/crop space to original image space and combines all detections.",
        examples=[
            "$steps.object_detection.predictions",
            "$steps.instance_segmentation.predictions",
            "$steps.slice_model.predictions",
        ],
    )
    overlap_filtering_strategy: Union[
        Literal["none", "nms", "nmm"],
        Selector(kind=[STRING_KIND]),
    ] = Field(
        default="nms",
        description="Strategy for handling overlapping detections when merging results from overlapping slices/crops. 'none': No filtering applied, all detections are kept (may include duplicates from overlapping regions). 'nms' (Non-Maximum Suppression, default): Removes lower-confidence detections when IoU exceeds threshold, keeping only the highest confidence detection for each overlapping region. 'nmm' (Non-Maximum Merge): Combines overlapping detections instead of discarding them, merging detections that exceed IoU threshold. Use 'none' when you want to preserve all detections, 'nms' to remove duplicates (recommended for most cases), or 'nmm' to combine overlapping detections.",
        examples=["none", "nms", "nmm", "$inputs.filtering_strategy"],
    )
    iou_threshold: Union[
        FloatZeroToOne,
        Selector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Intersection over Union (IoU) threshold for overlap filtering. Range: 0.0 to 1.0. When overlap filtering strategy is 'nms' or 'nmm', detections with IoU above this threshold are considered overlapping. For NMS: overlapping detections with IoU above threshold result in lower-confidence detection being removed. For NMM: overlapping detections with IoU above threshold are merged. Lower values (e.g., 0.2-0.3) are more aggressive, removing/merging more detections. Higher values (e.g., 0.5-0.7) are more permissive, only handling highly overlapping detections. Default 0.3 works well for most use cases with overlapping slices.",
        examples=[0.2, 0.3, 0.4, 0.5, "$inputs.iou_threshold"],
    )

    @classmethod
    def get_dimensionality_reference_property(cls) -> Optional[str]:
        return "reference_image"

    @classmethod
    def get_input_dimensionality_offsets(cls) -> Dict[str, int]:
        return {"predictions": 1}

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsStitchBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        reference_image: WorkflowImageData,
        predictions: Batch[TensorNativeDetections],
        overlap_filtering_strategy: Optional[Literal["none", "nms", "nmm"]],
        iou_threshold: Optional[float],
    ) -> BlockResult:
        # Use reference image to ensure all masks have the same dimensions
        reference_height, reference_width = reference_image.numpy_image.shape[:2]
        resolution_wh = (reference_width, reference_height)

        re_aligned_predictions = []
        for detections in predictions:
            detections_copy = deepcopy(detections)
            offset = retrieve_crop_offset(detections=detections_copy)
            detections_copy = manage_crops_metadata(
                detections=detections_copy, image=reference_image
            )
            re_aligned_detections = move_detections(
                detections=detections_copy,
                offset=offset,
                resolution_wh=resolution_wh,
            )
            re_aligned_predictions.append(re_aligned_detections)
        overlap_filter = choose_overlap_filter_strategy(
            overlap_filtering_strategy=overlap_filtering_strategy,
        )
        merged = merge_detections(detections_list=re_aligned_predictions)
        if overlap_filter is OverlapFilter.NONE:
            return {"predictions": merged}
        if overlap_filter is OverlapFilter.NON_MAX_SUPPRESSION:
            return {"predictions": with_nms(detections=merged, threshold=iou_threshold)}
        return {"predictions": with_nmm(detections=merged, threshold=iou_threshold)}


def retrieve_crop_offset(detections: TensorNativeDetections) -> Optional[np.ndarray]:
    if len(detections) == 0:
        return None
    image_metadata = detections.image_metadata or {}
    if PARENT_COORDINATES_KEY not in image_metadata:
        raise RuntimeError(
            f"Offset for crops is expected to be saved in image_metadata key {PARENT_COORDINATES_KEY} "
            f"of the tensor-native detections, but could not be found. Probably block producing the "
            f"detections lack this part of implementation or has a bug."
        )
    return np.asarray(image_metadata[PARENT_COORDINATES_KEY][:2]).copy()


def manage_crops_metadata(
    detections: TensorNativeDetections,
    image: WorkflowImageData,
) -> TensorNativeDetections:
    if len(detections) == 0:
        return detections

    image_metadata = detections.image_metadata or {}
    if SCALING_RELATIVE_TO_PARENT_KEY in image_metadata:
        scale = image_metadata[SCALING_RELATIVE_TO_PARENT_KEY]
        if abs(scale - 1.0) > 1e-4:
            raise ValueError(
                f"Scaled bounding boxes were passed to Detections Stitch block "
                f"which is not supported. Block is supposed to merge predictions "
                f"from multiple crops of the same image into single prediction, but "
                f"scaling cannot be used in the meantime. This error probably indicate "
                f"wrong step output plugged as input of this step."
            )

    height, width = image.numpy_image.shape[:2]
    image_metadata = dict(image_metadata)
    image_metadata[IMAGE_DIMENSIONS_KEY] = [height, width]
    image_metadata = attach_parents_coordinates_to_image_metadata(
        image_metadata=image_metadata,
        image=image,
    )
    detections.image_metadata = image_metadata

    return detections


def attach_parents_coordinates_to_image_metadata(
    image_metadata: dict,
    image: WorkflowImageData,
) -> dict:
    root = image.workflow_root_ancestor_metadata
    parent = image.parent_metadata
    root_coordinates = root.origin_coordinates
    parent_coordinates = parent.origin_coordinates
    image_metadata[ROOT_PARENT_ID_KEY] = root.parent_id
    image_metadata[ROOT_PARENT_COORDINATES_KEY] = [
        root_coordinates.left_top_x,
        root_coordinates.left_top_y,
    ]
    image_metadata[ROOT_PARENT_DIMENSIONS_KEY] = [
        root_coordinates.origin_height,
        root_coordinates.origin_width,
    ]
    image_metadata[PARENT_ID_KEY] = parent.parent_id
    image_metadata[PARENT_COORDINATES_KEY] = [
        parent_coordinates.left_top_x,
        parent_coordinates.left_top_y,
    ]
    image_metadata[PARENT_DIMENSIONS_KEY] = [
        parent_coordinates.origin_height,
        parent_coordinates.origin_width,
    ]
    return image_metadata


def move_detections(
    detections: TensorNativeDetections,
    offset: Optional[np.ndarray],
    resolution_wh: Optional[Tuple[int, int]],
) -> TensorNativeDetections:
    """
    Shift detections by ``offset``, keeping every geometry field consistent:
    axis-aligned boxes, segmentation masks, and oriented-box corners.

    Mirrors ``supervision.detection.tools.inference_slicer.move_detections``;
    kept local since that helper is not part of supervision's public API.
    """
    if len(detections) == 0:
        return detections
    if offset is None:
        raise ValueError("To move non-empty detections offset is needed, but not given")
    # Translate xyxy on-device: add [dx, dy, dx, dy] broadcast over the box rows,
    # staying on detections.xyxy.device/dtype (no D2H->H2D round-trip per slice).
    dx, dy = float(offset[0]), float(offset[1])
    xyxy_shift = torch.as_tensor(
        [dx, dy, dx, dy],
        dtype=detections.xyxy.dtype,
        device=detections.xyxy.device,
    )
    detections.xyxy = detections.xyxy + xyxy_shift
    bboxes_metadata = detections.bboxes_metadata
    if bboxes_metadata is not None and any(
        ORIENTED_BOX_COORDINATES in (box_metadata or {})
        for box_metadata in bboxes_metadata
    ):
        # OBB corners live per-detection in `bboxes_metadata[i]["xyxyxyxy"]` with
        # shape (4, 2); broadcast `offset` (shape (2,)) over the trailing axis to
        # translate each (x, y). Without this, downstream OBB-aware NMS/NMM compares
        # corners in tile-local coords against `xyxy` already moved to image coords.
        for box_metadata in bboxes_metadata:
            if box_metadata is not None and ORIENTED_BOX_COORDINATES in box_metadata:
                box_metadata[ORIENTED_BOX_COORDINATES] = (
                    np.asarray(box_metadata[ORIENTED_BOX_COORDINATES]) + offset
                )
    if isinstance(detections, InstanceDetections) and detections.mask is not None:
        if resolution_wh is None:
            raise ValueError(
                "To move non-empty detections with segmentation mask, resolution_wh is needed, but not given."
            )
        detections.mask = _move_native_masks(
            mask=detections.mask, offset=offset, resolution_wh=resolution_wh
        )
    return detections


def _move_native_masks(
    mask: Union[torch.Tensor, InstancesRLEMasks],
    offset: np.ndarray,
    resolution_wh: Tuple[int, int],
) -> InstancesRLEMasks:
    # Move masks in RLE form via embed_rle_masks_in_larger_canvas: the slice-resolution
    # masks are placed onto the (H, W) reference canvas at the slice's top-left offset,
    # entirely in RLE (the big canvas is never densified, no full-frame D2H). The output
    # is always RLE — dense input is first converted slice-by-slice to RLE.
    target_width, target_height = resolution_wh
    target_size_hw = (target_height, target_width)
    x0, y0 = int(offset[0]), int(offset[1])
    if isinstance(mask, InstancesRLEMasks):
        rle_masks = mask
    else:
        # DENSE input: convert each slice mask to RLE, wrap as InstancesRLEMasks.
        dense_masks = mask.detach().to(dtype=torch.bool)
        if dense_masks.shape[0] == 0:
            slice_height, slice_width = int(dense_masks.shape[1]), int(
                dense_masks.shape[2]
            )
            rle_masks = InstancesRLEMasks(
                image_size=(slice_height, slice_width), masks=[]
            )
        else:
            slice_height, slice_width = int(dense_masks.shape[1]), int(
                dense_masks.shape[2]
            )
            counts = [
                torch_mask_to_coco_rle(single_mask)["counts"]
                for single_mask in dense_masks
            ]
            rle_masks = InstancesRLEMasks(
                image_size=(slice_height, slice_width), masks=counts
            )
    return embed_rle_masks_in_larger_canvas(
        masks=rle_masks,
        offset_xy=(x0, y0),
        target_size_hw=target_size_hw,
    )


def merge_detections(
    detections_list: List[TensorNativeDetections],
) -> TensorNativeDetections:
    """Concatenate native detections from every crop into a single prediction.

    Mirrors ``sv.Detections.merge`` but operates directly on the
    ``inference_models`` dataclasses: ``xyxy`` / ``class_id`` / ``confidence``
    tensors are concatenated, ``bboxes_metadata`` lists are joined, masks (dense
    or RLE) are stacked, and per-image ``image_metadata`` (parent coordinates,
    dimensions, class-name maps) is taken from the re-aligned crops with their
    ``class_names`` maps unioned. NOTE (shared-helper candidate): a native
    merge/concat of detections is needed across several blocks and should be
    consolidated by the maintainer.
    """
    non_empty = [detections for detections in detections_list if len(detections) > 0]
    is_instance_segmentation = any(
        isinstance(detections, InstanceDetections) for detections in detections_list
    )
    image_metadata = _merge_image_metadata(detections_list)
    if len(non_empty) == 0:
        if is_instance_segmentation:
            return InstanceDetections(
                xyxy=torch.zeros((0, 4), dtype=torch.float32),
                class_id=torch.zeros((0,), dtype=torch.long),
                confidence=torch.zeros((0,), dtype=torch.float32),
                mask=torch.zeros((0, 0, 0), dtype=torch.bool),
                image_metadata=image_metadata,
                bboxes_metadata=None,
            )
        return Detections(
            xyxy=torch.zeros((0, 4), dtype=torch.float32),
            class_id=torch.zeros((0,), dtype=torch.long),
            confidence=torch.zeros((0,), dtype=torch.float32),
            image_metadata=image_metadata,
            bboxes_metadata=None,
        )
    xyxy = torch.cat([detections.xyxy for detections in non_empty], dim=0)
    class_id = torch.cat([detections.class_id for detections in non_empty], dim=0)
    confidence = torch.cat([detections.confidence for detections in non_empty], dim=0)
    bboxes_metadata: List[dict] = []
    for detections in non_empty:
        per_detection = detections.bboxes_metadata
        if per_detection is None:
            per_detection = [{} for _ in range(len(detections))]
        bboxes_metadata.extend(per_detection)
    if is_instance_segmentation:
        mask = _merge_masks(non_empty)
        return InstanceDetections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            mask=mask,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _merge_image_metadata(
    detections_list: List[TensorNativeDetections],
) -> Optional[dict]:
    merged: Optional[dict] = None
    class_names: Dict[int, str] = {}
    for detections in detections_list:
        image_metadata = detections.image_metadata
        if not image_metadata:
            continue
        if merged is None:
            merged = dict(image_metadata)
        per_image_class_names = image_metadata.get(CLASS_NAMES_KEY)
        if per_image_class_names:
            class_names.update(per_image_class_names)
    if merged is None:
        return None
    if class_names:
        merged[CLASS_NAMES_KEY] = class_names
    return merged


def _ensure_class_names_map(image_metadata: dict, class_id: torch.Tensor) -> dict:
    """Guarantee image_metadata carries a class_names map covering every output
    class_id so the serializer's per-row class_id lookup never fails when the
    merged metadata was None/map-less. Existing names are preserved; any
    surviving class_id without an entry falls back to f"class_{id}"."""
    image_metadata = dict(image_metadata)
    class_names = dict(image_metadata.get(CLASS_NAMES_KEY) or {})
    class_names = {int(key): value for key, value in class_names.items()}
    for class_id_value in class_id.detach().to("cpu").tolist():
        class_id_int = int(class_id_value)
        if class_id_int not in class_names:
            class_names[class_id_int] = f"class_{class_id_int}"
    image_metadata[CLASS_NAMES_KEY] = class_names
    return image_metadata


def _merge_masks(
    detections_list: List[InstanceDetections],
) -> Union[torch.Tensor, InstancesRLEMasks]:
    any_rle = any(
        isinstance(detections.mask, InstancesRLEMasks) for detections in detections_list
    )
    if any_rle:
        image_size = None
        rle_masks: List[bytes] = []
        for detections in detections_list:
            mask = detections.mask
            if isinstance(mask, InstancesRLEMasks):
                image_size = mask.image_size
                rle_masks.extend(mask.masks)
            else:
                numpy_masks = mask.detach().to("cpu").numpy().astype(bool)
                for single_mask in numpy_masks:
                    rle = torch_mask_to_coco_rle(
                        torch.as_tensor(single_mask, dtype=torch.bool)
                    )
                    image_size = tuple(rle["size"])
                    rle_masks.append(rle["counts"])
        return InstancesRLEMasks(image_size=image_size, masks=rle_masks)
    return torch.cat([detections.mask for detections in detections_list], dim=0)


def with_nms(
    detections: TensorNativeDetections,
    threshold: float,
) -> TensorNativeDetections:
    if len(detections) == 0:
        return detections
    # NMS runs on-device via torchvision.ops.batched_nms over the merged native
    # xyxy/confidence/class_id tensors (box IoU, class-aware — matching the prior
    # sv box NMS, which this block fed without masks). The kept indices index the
    # native detections directly; no sv.Detections is materialised. batched_nms
    # returns kept indices sorted by descending score; sort ascending so the
    # surviving rows keep their original relative order.
    boxes = detections.xyxy.to(dtype=torch.float32)
    scores = detections.confidence.to(dtype=torch.float32)
    # torchvision's NMS kernel requires float boxes/scores and an int64 idxs
    # (the kernel is not implemented for int32 class ids).
    class_ids = detections.class_id.to(dtype=torch.long)
    keep = torchvision.ops.batched_nms(
        boxes=boxes,
        scores=scores,
        idxs=class_ids,
        iou_threshold=threshold,
    )
    surviving_indices = torch.sort(keep).values.detach().to("cpu").tolist()
    survivors = take_prediction_by_indices(
        prediction=detections, indices=surviving_indices
    )
    # Guarantee a class_names map covering the surviving class_ids so a non-empty
    # NMS result never trips the serializer when the merged metadata is map-less.
    survivors.image_metadata = _ensure_class_names_map(
        survivors.image_metadata or {}, survivors.class_id
    )
    return survivors


def with_nmm(
    detections: TensorNativeDetections,
    threshold: float,
) -> TensorNativeDetections:
    if len(detections) == 0:
        return detections
    if isinstance(detections, InstanceDetections) and isinstance(
        detections.mask, torch.Tensor
    ):
        # Dense device masks: run the torch NMM port, which keeps the (N, H, W)
        # masks on device end-to-end (only the tiny xyxy/confidence/class_id and
        # the (N, N) resized-mask intersection matrix cross the PCIe bus).
        # Decision- and value-identical to the sv path below; the sv path stays
        # as insurance and for RLE / mask-less inputs.
        try:
            return _with_nmm_dense_masks_torch(
                detections=detections, threshold=threshold
            )
        except Exception as error:
            logger.warning(
                f"Torch NMM path of detections_stitch failed ({error}); "
                f"falling back to supervision-based NMM."
            )
    return _with_nmm_sv(detections=detections, threshold=threshold)


def _with_nmm_sv(
    detections: TensorNativeDetections,
    threshold: float,
) -> TensorNativeDetections:
    # sv.Detections is used here purely as the NMM algorithm (mirroring the numpy
    # block's `merged.with_nmm`). NMM rewrites box geometry (and union-merges
    # masks), so the merged sv output cannot be mapped 1:1 back to native rows;
    # the native result is built fresh from the sv output's xyxy/class_id/
    # confidence/mask, with class names preserved from the input image_metadata
    # and fresh per-detection ids. No sv.Detections is returned.
    image_metadata = detections.image_metadata or {}
    is_instance_segmentation = isinstance(detections, InstanceDetections)
    masks = None
    if is_instance_segmentation and detections.mask is not None:
        if isinstance(detections.mask, InstancesRLEMasks):
            masks = coco_rle_masks_to_numpy_mask(detections.mask)
        else:
            masks = detections.mask.detach().to("cpu").numpy().astype(bool)
    nmm_output = sv.Detections(
        xyxy=detections.xyxy.detach().to("cpu").numpy().astype(float),
        confidence=detections.confidence.detach().to("cpu").numpy().astype(float),
        class_id=detections.class_id.detach().to("cpu").numpy().astype(int),
        mask=masks,
    ).with_nmm(threshold=threshold)
    number_of_detections = len(nmm_output)
    bboxes_metadata = [
        {DETECTION_ID_KEY: str(uuid4())} for _ in range(number_of_detections)
    ]
    device = detections.xyxy.device
    xyxy = torch.as_tensor(
        np.asarray(nmm_output.xyxy), dtype=torch.float32, device=device
    ).reshape(-1, 4)
    class_id = torch.as_tensor(
        np.asarray(nmm_output.class_id), dtype=torch.long, device=device
    )
    confidence = torch.as_tensor(
        np.asarray(nmm_output.confidence), dtype=torch.float32, device=device
    )
    # Guarantee a class_names map covering the output class_ids so a non-empty
    # NMM result never trips the serializer when the input metadata is None/map-less.
    image_metadata = _ensure_class_names_map(image_metadata, class_id)
    if is_instance_segmentation:
        if nmm_output.mask is not None:
            mask = torch.as_tensor(
                np.asarray(nmm_output.mask), dtype=torch.bool, device=device
            )
        else:
            mask = torch.zeros((number_of_detections, 0, 0), dtype=torch.bool)
        return InstanceDetections(
            xyxy=xyxy,
            class_id=class_id,
            confidence=confidence,
            mask=mask,
            image_metadata=image_metadata,
            bboxes_metadata=bboxes_metadata,
        )
    return Detections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


# Constants of the torch NMM port. _NMM_MASK_DIMENSION mirrors the default
# `mask_dimension` of `sv...mask_non_max_merge` (grouping decisions are taken on
# masks resized to this max dimension). The float budget caps the transient
# float32 copy of the flattened resized masks used by the pairwise-intersection
# matmul; above it the matmul is tiled (results are identical either way — the
# counts are exact integers, see `_pairwise_mask_intersection_on_device`).
_NMM_MASK_DIMENSION = 640
_NMM_PAIRWISE_FLOAT_BUDGET_BYTES = 128 * 1024 * 1024


def _with_nmm_dense_masks_torch(
    detections: InstanceDetections,
    threshold: float,
) -> InstanceDetections:
    """Mask-based NMM equivalent to `sv.Detections.with_nmm` (class-aware, IoU
    metric) that never ships the dense (N, H, W) masks through host memory.

    supervision semantics replicated exactly (from supervision 0.29.1 sources):

    * grouping decisions run on masks resized to max-dim 640 with the
      nearest-index grid of `sv...resize_masks` (`np.linspace(0, dim - 1,
      new_dim).astype(int)` sampling — upsamples when the mask is smaller);
    * per class id (ascending `np.unique` order), detections are seeded by
      descending confidence (`scores.argsort()` pop-from-the-back) and the
      merge candidate is the *union* of already-absorbed resized masks;
      absorption is retried against the growing union until a fixed point
      (`sv..._group_overlapping_masks`);
    * IoU arithmetic mirrors `sv..._mask_iou_batch_split`: float32
      integer-exact intersection counts, `union = area_a + area_b - inter` in
      float32, division into a float64 zeros buffer where union != 0;
    * each group merges into one output detection (`sv._merge_detection_group`):
      union box via float32 min/max, confidence = box-area-weighted mean of
      member confidences (`np.dot(f32 areas, f64 confs) / total_area`, cast to
      float32; the winner's confidence when total area <= 0), the winning
      (highest-confidence) member's class id, mask = logical OR of the ORIGINAL
      full-resolution masks; singleton groups pass through unchanged.

    Device traffic: one D2H of xyxy/confidence/class_id, one D2H of the (N, N)
    intersection matrix + (N,) areas of the resized masks, one tiny D2H per
    union-growth round (rare: only groups that absorb members and re-test), and
    one small H2D of the merged xyxy/confidence/class_id. Full-resolution masks
    never leave the device.
    """
    device = detections.xyxy.device
    masks = detections.mask.detach()
    if masks.dtype is not torch.bool:
        masks = masks.to(dtype=torch.bool)
    number_of_input_detections = int(masks.shape[0])
    # Host copies of the small per-detection fields with the same dtypes the
    # sv-based path feeds into sv.Detections (float64 boxes / confidences,
    # int class ids) so every downstream decision is bit-identical.
    xyxy_host = detections.xyxy.detach().to("cpu").numpy().astype(float)
    confidence_host = detections.confidence.detach().to("cpu").numpy().astype(float)
    class_id_host = detections.class_id.detach().to("cpu").numpy().astype(int)
    resized_masks = _resize_masks_like_supervision(
        masks=masks, max_dimension=_NMM_MASK_DIMENSION
    )
    flat_masks = resized_masks.reshape(number_of_input_detections, -1)
    intersection, areas = _pairwise_mask_intersection_on_device(flat_masks=flat_masks)
    intersection_host = intersection.to("cpu").numpy()
    areas_host = areas.to("cpu").numpy()
    merge_groups = _mask_non_max_merge_groups(
        confidence=confidence_host,
        class_id=class_id_host,
        intersection=intersection_host,
        areas=areas_host,
        flat_masks=flat_masks,
        iou_threshold=threshold,
    )
    xyxy_out, confidence_out, class_id_out = _merge_detection_groups_bookkeeping(
        merge_groups=merge_groups,
        xyxy=xyxy_host,
        confidence=confidence_host,
        class_id=class_id_host,
    )
    mask = _union_masks_on_device(masks=masks, merge_groups=merge_groups)
    number_of_detections = len(merge_groups)
    bboxes_metadata = [
        {DETECTION_ID_KEY: str(uuid4())} for _ in range(number_of_detections)
    ]
    xyxy = torch.as_tensor(xyxy_out, dtype=torch.float32, device=device).reshape(-1, 4)
    class_id = torch.as_tensor(class_id_out, dtype=torch.long, device=device)
    confidence = torch.as_tensor(confidence_out, dtype=torch.float32, device=device)
    image_metadata = _ensure_class_names_map(detections.image_metadata or {}, class_id)
    return InstanceDetections(
        xyxy=xyxy,
        class_id=class_id,
        confidence=confidence,
        mask=mask.to(device),
        image_metadata=image_metadata,
        bboxes_metadata=bboxes_metadata,
    )


def _resize_masks_like_supervision(
    masks: torch.Tensor, max_dimension: int
) -> torch.Tensor:
    """Torch replica of `sv...resize_masks`: nearest-index resampling so the
    largest mask dimension becomes `max_dimension` (aspect ratio kept). The
    integer sampling grids are computed on host with the exact numpy expression
    supervision uses, so the resampled masks match sv's pixel-for-pixel."""
    height, width = int(masks.shape[1]), int(masks.shape[2])
    scale = min(max_dimension / height, max_dimension / width)
    new_height = int(scale * height)
    new_width = int(scale * width)
    if new_height == height and new_width == width:
        # np.linspace(0, dim - 1, dim).astype(int) is the identity grid.
        return masks
    y_indices = torch.as_tensor(
        np.linspace(0, height - 1, new_height).astype(int),
        dtype=torch.long,
        device=masks.device,
    )
    x_indices = torch.as_tensor(
        np.linspace(0, width - 1, new_width).astype(int),
        dtype=torch.long,
        device=masks.device,
    )
    return masks.index_select(1, y_indices).index_select(2, x_indices)


def _pairwise_mask_intersection_on_device(
    flat_masks: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(N, N) float32 intersection pixel counts and (N,) float32 areas of the
    flattened bool masks, computed on the masks' device.

    Counts are exact integers: the matmul operands are 0/1 (exact in float32
    and in TF32's 10-bit mantissa) and every partial sum stays below
    2**24 (P <= 640 * 640 after resize), so any accumulation order — cuBLAS,
    TF32-with-fp32-accumulate, CPU BLAS — yields the exact count, matching
    supervision's float32 numpy matmul bit-for-bit. Memory: the transient
    float32 copy of the flattened masks is tiled once it would exceed
    _NMM_PAIRWISE_FLOAT_BUDGET_BYTES (the bool masks themselves stay resident).
    """
    number_of_masks, pixels = int(flat_masks.shape[0]), int(flat_masks.shape[1])
    if number_of_masks * pixels * 4 <= _NMM_PAIRWISE_FLOAT_BUDGET_BYTES:
        flat_f32 = flat_masks.to(dtype=torch.float32)
        return flat_f32 @ flat_f32.T, flat_f32.sum(dim=1)
    chunk = max(1, _NMM_PAIRWISE_FLOAT_BUDGET_BYTES // (2 * max(pixels, 1) * 4))
    intersection = torch.empty(
        (number_of_masks, number_of_masks),
        dtype=torch.float32,
        device=flat_masks.device,
    )
    areas = torch.empty(
        (number_of_masks,), dtype=torch.float32, device=flat_masks.device
    )
    for row_start in range(0, number_of_masks, chunk):
        row_end = min(row_start + chunk, number_of_masks)
        rows_f32 = flat_masks[row_start:row_end].to(dtype=torch.float32)
        areas[row_start:row_end] = rows_f32.sum(dim=1)
        for col_start in range(row_start, number_of_masks, chunk):
            col_end = min(col_start + chunk, number_of_masks)
            if col_start == row_start:
                cols_f32 = rows_f32
            else:
                cols_f32 = flat_masks[col_start:col_end].to(dtype=torch.float32)
            block = rows_f32 @ cols_f32.T
            intersection[row_start:row_end, col_start:col_end] = block
            if col_start != row_start:
                intersection[col_start:col_end, row_start:row_end] = block.T
    return intersection, areas


def _mask_non_max_merge_groups(
    confidence: np.ndarray,
    class_id: np.ndarray,
    intersection: np.ndarray,
    areas: np.ndarray,
    flat_masks: torch.Tensor,
    iou_threshold: float,
) -> List[List[int]]:
    """Port of `sv...mask_non_max_merge` (class-aware, IoU metric): per class id
    in ascending order, run the greedy growing-union grouping and translate the
    class-local groups back to global row indices."""
    merge_groups: List[List[int]] = []
    for category_id in np.unique(class_id):
        current_indices = np.where(class_id == category_id)[0]
        local_groups = _group_overlapping_masks_greedy(
            scores=confidence[current_indices],
            global_indices=current_indices,
            intersection=intersection,
            areas=areas,
            flat_masks=flat_masks,
            iou_threshold=iou_threshold,
        )
        for local_group in local_groups:
            merge_groups.append(current_indices[local_group].tolist())
    return merge_groups


def _group_overlapping_masks_greedy(
    scores: np.ndarray,
    global_indices: np.ndarray,
    intersection: np.ndarray,
    areas: np.ndarray,
    flat_masks: torch.Tensor,
    iou_threshold: float,
) -> List[List[int]]:
    """Port of `sv..._group_overlapping_masks` returning groups of positions
    into `global_indices` (sv's class-local indices). The first absorption round
    of every group is answered from the precomputed pairwise intersection
    matrix (the candidate is still a single mask); only rounds against a grown
    union candidate query the device — one small D2H each."""
    merge_groups: List[List[int]] = []
    order = scores.argsort()
    while len(order) > 0:
        idx = int(order[-1])
        order = order[:-1]
        if len(order) == 0:
            merge_groups.append([idx])
            break
        candidate_group = [idx]
        first_round = True
        while len(order) > 0:
            remaining_global = global_indices[order]
            if first_round:
                seed_global = int(global_indices[idx])
                intersection_vector = intersection[remaining_global, seed_global]
                candidate_area = areas[seed_global]
            else:
                intersection_vector, candidate_area = _union_candidate_iou_inputs(
                    flat_masks=flat_masks,
                    member_global_ids=global_indices[candidate_group],
                    remaining_global_ids=remaining_global,
                )
            # Same arithmetic as sv._mask_iou_batch_split: float32 union,
            # division into a float64 zeros buffer where union != 0.
            union_area = areas[remaining_global] + candidate_area - intersection_vector
            ious = np.divide(
                intersection_vector,
                union_area,
                out=np.zeros_like(intersection_vector, dtype=float),
                where=union_area != 0,
            )
            ious = np.nan_to_num(ious)
            above_threshold = ious >= iou_threshold
            if not above_threshold.any():
                break
            above_idx = order[above_threshold]
            candidate_group.extend(np.flip(above_idx).tolist())
            order = order[~above_threshold]
            first_round = False
        merge_groups.append(candidate_group)
    return merge_groups


def _union_candidate_iou_inputs(
    flat_masks: torch.Tensor,
    member_global_ids: np.ndarray,
    remaining_global_ids: np.ndarray,
) -> Tuple[np.ndarray, np.float32]:
    """Intersection counts of the remaining resized masks against the union of
    the current group members, plus the union's area — computed on device and
    shipped as one packed vector (single small D2H sync). Counts are exact
    integers, matching sv's float32 numpy arithmetic bit-for-bit."""
    device = flat_masks.device
    members = torch.as_tensor(member_global_ids, dtype=torch.long, device=device)
    remaining = torch.as_tensor(remaining_global_ids, dtype=torch.long, device=device)
    candidate_f32 = (
        flat_masks.index_select(0, members).any(dim=0).to(dtype=torch.float32)
    )
    intersection_vector = (
        flat_masks.index_select(0, remaining).to(dtype=torch.float32) @ candidate_f32
    )
    packed = torch.cat([intersection_vector, candidate_f32.sum().reshape(1)])
    packed_host = packed.to("cpu").numpy()
    return packed_host[:-1], np.float32(packed_host[-1])


def _merge_detection_groups_bookkeeping(
    merge_groups: List[List[int]],
    xyxy: np.ndarray,
    confidence: np.ndarray,
    class_id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of the AABB branch of `sv._merge_detection_group` (+ the trivial
    concatenation of `sv.Detections.merge`), on host numpy with the exact
    dtypes sv uses so merged boxes and confidences are bit-identical."""
    number_of_groups = len(merge_groups)
    xyxy_out = np.empty((number_of_groups, 4), dtype=np.float64)
    confidence_out = np.empty((number_of_groups,), dtype=np.float64)
    class_id_out = np.empty((number_of_groups,), dtype=np.int64)
    for position, group in enumerate(merge_groups):
        if len(group) == 1:
            index = group[0]
            xyxy_out[position] = xyxy[index]
            confidence_out[position] = confidence[index]
            class_id_out[position] = class_id[index]
            continue
        group_confidences = confidence[group]
        winner_index = int(np.argmax(group_confidences))
        all_xyxy = xyxy[group].astype(np.float32)
        box_areas = (all_xyxy[:, 2] - all_xyxy[:, 0]) * (
            all_xyxy[:, 3] - all_xyxy[:, 1]
        )
        total_area = float(box_areas.sum())
        if total_area > 0:
            merged_confidence = np.float32(
                float(np.dot(box_areas, group_confidences) / total_area)
            )
        else:
            merged_confidence = group_confidences[winner_index]
        xyxy_out[position] = [
            all_xyxy[:, 0].min(),
            all_xyxy[:, 1].min(),
            all_xyxy[:, 2].max(),
            all_xyxy[:, 3].max(),
        ]
        confidence_out[position] = merged_confidence
        class_id_out[position] = class_id[group[winner_index]]
    return xyxy_out, confidence_out, class_id_out


def _union_masks_on_device(
    masks: torch.Tensor, merge_groups: List[List[int]]
) -> torch.Tensor:
    """Merged output masks built from the ORIGINAL full-resolution device masks:
    singleton groups are gathered with one index_select, multi-member groups are
    unioned with one batched index_add_ (member count per pixel > 0 == logical
    OR — exact bool result, matching sv's np.logical_or.reduce). Fixed kernel
    count regardless of the number of groups."""
    device = masks.device
    number_of_groups = len(merge_groups)
    height, width = int(masks.shape[1]), int(masks.shape[2])
    mask_out = torch.empty(
        (number_of_groups, height, width), dtype=torch.bool, device=device
    )
    single_positions = [
        position for position, group in enumerate(merge_groups) if len(group) == 1
    ]
    multi_positions = [
        position for position, group in enumerate(merge_groups) if len(group) > 1
    ]
    if single_positions:
        single_ids = torch.as_tensor(
            [merge_groups[position][0] for position in single_positions],
            dtype=torch.long,
            device=device,
        )
        positions = torch.as_tensor(single_positions, dtype=torch.long, device=device)
        mask_out[positions] = masks.index_select(0, single_ids)
    if multi_positions:
        member_ids: List[int] = []
        member_slots: List[int] = []
        for slot, position in enumerate(multi_positions):
            for member in merge_groups[position]:
                member_ids.append(member)
                member_slots.append(slot)
        largest_group = max(len(merge_groups[position]) for position in multi_positions)
        # uint8 accumulation wraps at 256 members per group; escape to int32
        # for (degenerate) larger groups.
        accumulator_dtype = torch.uint8 if largest_group < 256 else torch.int32
        accumulator = torch.zeros(
            (len(multi_positions), height, width),
            dtype=accumulator_dtype,
            device=device,
        )
        member_ids_t = torch.as_tensor(member_ids, dtype=torch.long, device=device)
        member_slots_t = torch.as_tensor(member_slots, dtype=torch.long, device=device)
        accumulator.index_add_(
            0, member_slots_t, masks.index_select(0, member_ids_t).to(accumulator_dtype)
        )
        positions = torch.as_tensor(multi_positions, dtype=torch.long, device=device)
        mask_out[positions] = accumulator != 0
    return mask_out


def choose_overlap_filter_strategy(
    overlap_filtering_strategy: Literal["none", "nms", "nmm"],
) -> sv.OverlapFilter:
    if overlap_filtering_strategy == "none":
        return sv.OverlapFilter.NONE
    if overlap_filtering_strategy == "nms":
        return sv.OverlapFilter.NON_MAX_SUPPRESSION
    elif overlap_filtering_strategy == "nmm":
        return sv.OverlapFilter.NON_MAX_MERGE
    raise ValueError(
        f"Invalid overlap filtering strategy: {overlap_filtering_strategy}"
    )
