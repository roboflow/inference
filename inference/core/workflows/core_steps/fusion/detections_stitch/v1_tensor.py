from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field
from supervision import OverlapFilter, move_boxes, move_masks
from supervision.config import ORIENTED_BOX_COORDINATES

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import (
    coco_rle_masks_to_numpy_mask,
    torch_mask_to_coco_rle,
)

from inference.core.workflows.core_steps.common.tensor_native import (
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
            f"Offset for crops is expected to be saved in data key {PARENT_COORDINATES_KEY} "
            f"of sv.Detections, but could not be found. Probably block producing sv.Detections "
            f"lack this part of implementation or has a bug."
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
    # sv.move_boxes is a pure numpy translation; read xyxy out as numpy, shift,
    # write the result back as a tensor (no sv.Detections is ever materialised).
    moved_xyxy = move_boxes(
        xyxy=detections.xyxy.detach().to("cpu").numpy(), offset=offset
    )
    detections.xyxy = torch.as_tensor(
        moved_xyxy, dtype=detections.xyxy.dtype, device=detections.xyxy.device
    )
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
) -> Union[torch.Tensor, InstancesRLEMasks]:
    # sv.move_masks is a pure numpy operation (pad/shift to the target resolution);
    # decode the native masks to a numpy stack, move them, then re-encode in the
    # original representation (dense torch or RLE). No sv.Detections is built.
    is_rle = isinstance(mask, InstancesRLEMasks)
    if is_rle:
        numpy_masks = coco_rle_masks_to_numpy_mask(mask)
    else:
        numpy_masks = mask.detach().to("cpu").numpy().astype(bool)
    moved_masks = move_masks(
        masks=numpy_masks, offset=offset, resolution_wh=resolution_wh
    )
    if is_rle:
        target_height, target_width = resolution_wh[1], resolution_wh[0]
        rle_masks = [
            torch_mask_to_coco_rle(
                torch.as_tensor(single_mask, dtype=torch.bool)
            )["counts"]
            for single_mask in moved_masks
        ]
        return InstancesRLEMasks(
            image_size=(target_height, target_width), masks=rle_masks
        )
    return torch.as_tensor(moved_masks, dtype=torch.bool, device=mask.device)


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
    confidence = torch.cat(
        [detections.confidence for detections in non_empty], dim=0
    )
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


def _merge_masks(
    detections_list: List[InstanceDetections],
) -> Union[torch.Tensor, InstancesRLEMasks]:
    any_rle = any(
        isinstance(detections.mask, InstancesRLEMasks)
        for detections in detections_list
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
    # sv.Detections is used here purely as the NMS algorithm (mirroring the numpy
    # block's `merged.with_nms`). A synthetic row index travels through `.data`
    # so the surviving original rows can be recovered and sliced natively; no
    # sv.Detections is returned.
    number_of_detections = len(detections)
    nms_input = sv.Detections(
        xyxy=detections.xyxy.detach().to("cpu").numpy().astype(float),
        confidence=detections.confidence.detach().to("cpu").numpy().astype(float),
        class_id=detections.class_id.detach().to("cpu").numpy().astype(int),
        data={"__row_index__": np.arange(number_of_detections)},
    ).with_nms(threshold=threshold)
    surviving_indices = nms_input.data["__row_index__"].astype(int).tolist()
    return take_prediction_by_indices(
        prediction=detections, indices=surviving_indices
    )


def with_nmm(
    detections: TensorNativeDetections,
    threshold: float,
) -> TensorNativeDetections:
    if len(detections) == 0:
        return detections
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
