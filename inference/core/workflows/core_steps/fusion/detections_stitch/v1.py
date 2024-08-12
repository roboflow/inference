from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision import OverlapFilter, move_boxes, move_masks

from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    FLOAT_ZERO_TO_ONE_KIND,
    STRING_KIND,
    FloatZeroToOne,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
This block merges detections that were inferred for multiple sub-parts of the same input image
into single detection. 

Block may be helpful in the following scenarios:
* to apply [Slicing Adaptive Inference (SAHI)](https://ieeexplore.ieee.org/document/9897990) technique, 
as a final step of procedure, which involves Image Slicer block and model block at previous stages.
* to merge together detections performed by precise, high-resolution model applied as secondary
model after coarse detection is performed in the first stage and Dynamic Crop is applied later. 
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
        }
    )
    type: Literal["roboflow_core/detections_stitch@v1"]
    crops: Union[WorkflowImageSelector, StepOutputImageSelector] = Field(
        description="Crops used to generate predictions to be merged.",
        examples=["$steps.cropping.crops"],
    )
    predictions: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="The output of a detection model describing the bounding boxes to be merged.",
        examples=["$steps.my_object_detection_model.predictions"],
    )
    overlap_filtering_strategy: Union[
        Literal["none", "nms", "nmm"],
        WorkflowParameterSelector(kind=[STRING_KIND]),
    ] = Field(
        default="nms",
        description="Which strategy to employ when filtering overlapping boxes. "
        "None does nothing, NMS discards surplus detections, NMM merges them.",
        examples=["nms", "$inputs.overlap_filtering_strategy"],
    )
    iou_threshold: Union[
        FloatZeroToOne,
        WorkflowParameterSelector(kind=[FLOAT_ZERO_TO_ONE_KIND]),
    ] = Field(
        default=0.3,
        description="Parameter of overlap filtering strategy. If box intersection over union is above this "
        " ratio, discard or merge the lower confidence box.",
        examples=[0.4, "$inputs.iou_threshold"],
    )

    @classmethod
    def get_output_dimensionality_offset(cls) -> int:
        return -1

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
                    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class DetectionsStitchBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        crops: Batch[WorkflowImageData],
        predictions: Batch[sv.Detections],
        overlap_filtering_strategy: Optional[Literal["none", "nms", "nmm"]],
        iou_threshold: Optional[float],
    ) -> BlockResult:
        re_aligned_predictions = []
        for crop, detections in zip(crops, predictions):
            origin_coordinates = crops[0].parent_metadata.origin_coordinates
            offset = np.array(
                [origin_coordinates.left_top_x, origin_coordinates.left_top_y]
            )
            resolution_wh = (crop.numpy_image.shape[1], crop.numpy_image.shape[0])
            re_aligned_detections = move_detections(
                detections=detections.copy(), offset=offset, resolution_wh=resolution_wh
            )
            re_aligned_predictions.append(re_aligned_detections)

        overlap_filter = choose_overlap_filter_strategy(
            overlap_filtering_strategy=overlap_filtering_strategy,
        )
        merged = sv.Detections.merge(detections_list=re_aligned_predictions)
        if overlap_filter is OverlapFilter.NONE:
            return {"predictions": merged}
        if overlap_filter is OverlapFilter.NON_MAX_SUPPRESSION:
            return {"predictions": merged.with_nms(threshold=iou_threshold)}
        return {"predictions": merged.with_nmm(threshold=iou_threshold)}


def move_detections(
    detections: sv.Detections,
    offset: np.ndarray,
    resolution_wh: Tuple[int, int],
) -> sv.Detections:
    """
    Copied from: https://github.com/roboflow/supervision/blob/5123085037ec594524fc8f9d9b71b1cd9f487e8d/supervision/detection/tools/inference_slicer.py#L17-L16
    to avoid fragile contract with supervision, as this function is not element of public
    API.
    Args:
        detections (sv.Detections): Detections object to be moved.
        offset (np.ndarray): An array of shape `(2,)` containing offset values in format
            is `[dx, dy]`.
        resolution_wh (Tuple[int, int]): The width and height of the desired mask
            resolution. Required for segmentation detections.

    Returns:
        (sv.Detections) repositioned Detections object.
    """
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
    if detections.mask is not None:
        if resolution_wh is None:
            raise ValueError(
                "Resolution width and height are required for moving segmentation "
                "detections. This should be the same as (width, height) of image shape."
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
