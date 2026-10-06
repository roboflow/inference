from copy import copy, deepcopy
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.utils import (
    attach_parents_coordinates_to_sv_detections,
)
from roboflow_workflows.execution_engine.constants import (
    IMAGE_DIMENSIONS_KEY,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    IMAGE_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    STRING_KIND,
    FloatZeroToOne,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from supervision import OverlapFilter, move_boxes, move_masks
from supervision.config import ORIENTED_BOX_COORDINATES
from supervision.detection.compact_mask import CompactMask

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
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
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
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        return [WorkOperation.DETECTION_PROCESSING]

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )

    def discover_dependent_resources(self) -> List[DependentResource]:
        return []


class DetectionsStitchBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        reference_image: WorkflowImageData,
        predictions: Batch[sv.Detections],
        overlap_filtering_strategy: Optional[Literal["none", "nms", "nmm"]],
        iou_threshold: Optional[float],
    ) -> BlockResult:
        # Use reference image to ensure all masks have the same dimensions
        reference_height, reference_width = reference_image.numpy_image.shape[:2]
        resolution_wh = (reference_width, reference_height)

        # Masks travel separately from the rest of the detections and are
        # only materialised at reference resolution once the survivors are
        # known. Moving every crop mask to a full-size dense array first costs
        # N x H x W bytes before any filtering: a 1080p frame sliced 12 ways
        # with ~25 masks per slice took ~3 GiB inside this block and
        # OOM-killed an 8 GiB video worker (2026-09-28).
        #
        # With overlap filtering every crop's masks become crop-scoped
        # CompactMask (RLE of each mask's bounding box); merge / NMS / NMM run
        # on the compact form, so peak memory is bounded by the survivors.
        # Without filtering every mask survives and the output is dense, so
        # the RLE round trip is pure overhead (7x slower on fragmented
        # masks); the crop masks are copied straight into the output stack.
        overlap_filter = choose_overlap_filter_strategy(
            overlap_filtering_strategy=overlap_filtering_strategy,
        )
        re_aligned_predictions = []
        crop_masks: List[Tuple[np.ndarray, np.ndarray]] = []
        masks_seen = False
        masks_missing = False
        for detections in predictions:
            mask = detections.mask
            detections_copy = _copy_without_mask(detections=detections)
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
            if len(detections_copy) == 0:
                continue
            if mask is None:
                masks_missing = True
                continue
            masks_seen = True
            crop_masks.append((np.asarray(mask), offset))
        if masks_seen and masks_missing:
            raise ValueError(
                "Detections Stitch block received a mix of predictions with and "
                "without segmentation masks; every non-empty crop must carry "
                "masks, or none may."
            )

        merged = sv.Detections.merge(detections_list=re_aligned_predictions)
        if overlap_filter is OverlapFilter.NONE:
            if crop_masks:
                merged.mask = stitch_masks_dense(
                    crop_masks=crop_masks, resolution_wh=resolution_wh
                )
            return {"predictions": merged}

        if crop_masks:
            merged.mask = CompactMask.merge(
                [
                    compact_mask_for_crop(
                        masks, offset=offset, resolution_wh=resolution_wh
                    )
                    for masks, offset in crop_masks
                ]
            )
        if overlap_filter is OverlapFilter.NON_MAX_SUPPRESSION:
            filtered = merged.with_nms(threshold=iou_threshold)
        else:
            filtered = merged.with_nmm(threshold=iou_threshold)
        result = _with_dense_masks(detections=filtered)
        return {"predictions": result}


def _copy_without_mask(detections: sv.Detections) -> sv.Detections:
    # Deep copy sharing nothing with the input, minus the mask: crop masks are
    # carried separately as CompactMask, so the dense crop stack is not copied.
    shallow = copy(detections)
    shallow.mask = None
    detections_copy = deepcopy(shallow)
    return detections_copy


def visible_crop_region(
    crop_shape: Tuple[int, int],
    *,
    offset: np.ndarray,
    resolution_wh: Tuple[int, int],
) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
    """Find the part of a crop that lands inside the reference frame.

    Mirrors the clipping of ``move_masks``: a crop may start before the
    reference origin (negative offset) or extend past its far edge.

    Args:
        crop_shape: ``(crop_h, crop_w)`` of the crop masks.
        offset: ``(x, y)`` position of the crop in the reference image.
        resolution_wh: ``(width, height)`` of the reference image.

    Returns:
        ``((source_y1, source_y2, source_x1, source_x2), (target_y1, target_y2,
        target_x1, target_x2))`` - the crop slice and the reference slice it
        maps onto - or ``None`` if the crop is entirely outside the frame.
    """
    reference_width, reference_height = resolution_wh
    offset_x, offset_y = int(offset[0]), int(offset[1])
    crop_height, crop_width = crop_shape
    source_x1, source_y1 = max(0, -offset_x), max(0, -offset_y)
    source_x2 = min(crop_width, reference_width - offset_x)
    source_y2 = min(crop_height, reference_height - offset_y)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return None
    target_x1, target_y1 = offset_x + source_x1, offset_y + source_y1
    target_x2 = target_x1 + (source_x2 - source_x1)
    target_y2 = target_y1 + (source_y2 - source_y1)
    return (
        (source_y1, source_y2, source_x1, source_x2),
        (target_y1, target_y2, target_x1, target_x2),
    )


def stitch_masks_dense(
    crop_masks: List[Tuple[np.ndarray, np.ndarray]],
    *,
    resolution_wh: Tuple[int, int],
) -> np.ndarray:
    """Place every crop's masks in one dense reference-size stack.

    Equivalent to ``move_masks`` per crop followed by ``np.concatenate``, but
    the output ``(N, reference_h, reference_w)`` array is the only
    reference-size allocation: no per-crop full-frame stacks, concatenation
    copy or compact intermediates. Only each mask's tight bounding box is
    written, so the rest of the (calloc-zeroed) output is never touched;
    writing the whole crop window per mask was 2x slower and left ~50% more
    resident memory on 300 masks from 640x640 crops.

    Args:
        crop_masks: ``(masks, offset)`` per crop, in output order; ``masks``
            is dense ``(n_i, crop_h, crop_w)`` and ``offset`` is the crop's
            ``(x, y)`` position in the reference image.
        resolution_wh: ``(width, height)`` of the reference image.

    Returns:
        Boolean ``(sum(n_i), height, width)`` array.
    """
    reference_width, reference_height = resolution_wh
    total = sum(masks.shape[0] for masks, _ in crop_masks)
    stitched = np.zeros((total, reference_height, reference_width), dtype=bool)
    row = 0
    for masks, offset in crop_masks:
        count = masks.shape[0]
        region = visible_crop_region(
            masks.shape[1:], offset=offset, resolution_wh=resolution_wh
        )
        if region is not None:
            (sy1, sy2, sx1, sx2), (ty1, _, tx1, _) = region
            visible = masks[:, sy1:sy2, sx1:sx2]
            # half-open boxes; an empty mask comes back as all zeros and is
            # skipped, which leaves its output rows False
            tight = sv.mask_to_xyxy(masks=visible, coordinate_convention="exclusive")
            for index, (x1, y1, x2, y2) in enumerate(tight.tolist()):
                if x2 <= x1 or y2 <= y1:
                    continue
                stitched[row + index, ty1 + y1 : ty1 + y2, tx1 + x1 : tx1 + x2] = (
                    visible[index, y1:y2, x1:x2]
                )
        row += count
    return stitched


def compact_mask_for_crop(
    masks: np.ndarray,
    *,
    offset: Optional[np.ndarray],
    resolution_wh: Tuple[int, int],
) -> CompactMask:
    """Position crop masks in the reference image as compact masks.

    Decodes to the same dense array as ``move_masks(masks, offset,
    resolution_wh)``, including the clipping of a crop that extends past the
    reference frame, without allocating anything of reference size.

    Args:
        masks: Dense ``(N, crop_h, crop_w)`` boolean masks in crop coordinates.
        offset: ``(x, y)`` position of the crop in the reference image.
        resolution_wh: ``(width, height)`` of the reference image.

    Returns:
        ``CompactMask`` of ``N`` masks whose image shape is the reference image.

    Raises:
        ValueError: If ``offset`` is missing for a non-empty crop.
    """
    if offset is None:
        raise ValueError("To move non-empty detections offset is needed, but not given")

    reference_width, reference_height = resolution_wh
    region = visible_crop_region(
        masks.shape[1:], offset=offset, resolution_wh=resolution_wh
    )
    if region is None:
        visible = np.zeros((masks.shape[0], 1, 1), dtype=bool)
        target_x1, target_y1 = 0, 0
    else:
        (sy1, sy2, sx1, sx2), (target_y1, _, target_x1, _) = region
        visible = np.ascontiguousarray(masks[:, sy1:sy2, sx1:sx2])

    visible_shape = (visible.shape[1], visible.shape[2])
    tight_boxes = sv.mask_to_xyxy(masks=visible)
    compact = CompactMask.from_dense(visible, tight_boxes, visible_shape)
    positioned = compact.with_offset(
        target_x1,
        target_y1,
        (reference_height, reference_width),
    )
    return positioned


def _with_dense_masks(detections: sv.Detections) -> sv.Detections:
    # supervision <0.30 does not carry CompactMask through Detections.merge or
    # the annotators, so the block's output stays dense. Only the survivors of
    # overlap filtering are decoded, which is what bounds this block's memory.
    if isinstance(detections.mask, CompactMask):
        detections.mask = detections.mask.to_dense()
    return detections


def retrieve_crop_offset(detections: sv.Detections) -> Optional[np.ndarray]:
    if len(detections) == 0:
        return None
    if PARENT_COORDINATES_KEY not in detections.data:
        raise RuntimeError(
            f"Offset for crops is expected to be saved in data key {PARENT_COORDINATES_KEY} "
            f"of sv.Detections, but could not be found. Probably block producing sv.Detections "
            f"lack this part of implementation or has a bug."
        )
    return detections.data[PARENT_COORDINATES_KEY][0][:2].copy()


def manage_crops_metadata(
    detections: sv.Detections,
    image: WorkflowImageData,
) -> sv.Detections:
    if len(detections) == 0:
        return detections

    if SCALING_RELATIVE_TO_PARENT_KEY in detections.data:
        scale = detections[SCALING_RELATIVE_TO_PARENT_KEY][0]
        if abs(scale - 1.0) > 1e-4:
            raise ValueError(
                f"Scaled bounding boxes were passed to Detections Stitch block "
                f"which is not supported. Block is supposed to merge predictions "
                f"from multiple crops of the same image into single prediction, but "
                f"scaling cannot be used in the meantime. This error probably indicate "
                f"wrong step output plugged as input of this step."
            )

    height, width = image.numpy_image.shape[:2]
    detections[IMAGE_DIMENSIONS_KEY] = np.array([[height, width]] * len(detections))

    return attach_parents_coordinates_to_sv_detections(
        detections=detections,
        image=image,
    )


def move_detections(
    detections: sv.Detections,
    offset: Optional[np.ndarray],
    resolution_wh: Optional[Tuple[int, int]],
) -> sv.Detections:
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
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
    if ORIENTED_BOX_COORDINATES in detections.data:
        # OBB corners live in `data["xyxyxyxy"]` with shape (N, 4, 2); broadcast
        # `offset` (shape (2,)) over the trailing axis to translate each (x, y).
        # Without this, downstream OBB-aware NMS/NMM compares corners in
        # tile-local coords against `xyxy` already moved to image coords.
        detections.data[ORIENTED_BOX_COORDINATES] = (
            detections.data[ORIENTED_BOX_COORDINATES] + offset
        )
    if detections.mask is not None:
        if resolution_wh is None:
            raise ValueError(
                "To move non-empty detections with segmentation mask, resolution_wh is needed, but not given."
            )
        detections.mask = move_masks(
            masks=detections.mask, offset=offset, resolution_wh=resolution_wh
        )
    return detections


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
