import logging
import warnings
from copy import deepcopy
from typing import List, Literal, Optional, Tuple, Type, Union

import numpy as np
import supervision as sv
from pydantic import ConfigDict, Field
from supervision import OverlapFilter, move_boxes, move_masks

from inference.core.workflows.core_steps.common.utils import scale_sv_detections
from inference.core.workflows.core_steps.warnings import (
    RoboflowCoreBlocksIncompatibilityWarning,
)
from inference.core.workflows.execution_engine.constants import (
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    ROOT_PARENT_COORDINATES_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
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
            detections_copy = deepcopy(detections)
            resolution_wh = retrieve_crop_wh(detections=detections_copy)
            offset = retrieve_crop_offset(detections=detections_copy)
            detections_copy = manage_crops_metadata(
                detections=detections_copy,
                offset=offset,
                image_lineage=crop.lineage,
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
        merged = sv.Detections.merge(detections_list=re_aligned_predictions)
        if overlap_filter is OverlapFilter.NONE:
            return {"predictions": merged}
        if overlap_filter is OverlapFilter.NON_MAX_SUPPRESSION:
            return {"predictions": merged.with_nms(threshold=iou_threshold)}
        return {"predictions": merged.with_nmm(threshold=iou_threshold)}


def retrieve_crop_wh(detections: sv.Detections) -> Optional[Tuple[int, int]]:
    if len(detections) == 0:
        return None
    if PARENT_DIMENSIONS_KEY not in detections.data:
        raise RuntimeError(
            f"Dimensions for crops is expected to be saved in data key {PARENT_DIMENSIONS_KEY} "
            f"of sv.Detections, but could not be found. Probably block producing sv.Detections "
            f"lack this part of implementation or has a bug."
        )
    return (
        detections.data[PARENT_COORDINATES_KEY][0][1].item(),
        detections.data[PARENT_COORDINATES_KEY][0][0].item(),
    )


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
    offset: Optional[np.ndarray],
    image_lineage: List[ImageParentMetadata],
) -> sv.Detections:
    if len(detections) == 0:
        return detections
    if offset is None:
        raise ValueError(
            "To process non-empty detections offset is needed, but not given"
        )
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
    if PARENT_COORDINATES_KEY in detections.data:
        detections.data[PARENT_COORDINATES_KEY] -= offset
    if ROOT_PARENT_COORDINATES_KEY in detections.data:
        detections.data[ROOT_PARENT_COORDINATES_KEY] -= offset
    warnings.warn(
        f"Workflow block roboflow_core/detections_stitch@v1 could not correctly "
        f"re-assign predictions parent ID unwrapping previous crop operation symmetrically. "
        f"That may cause errors at latter stages of execution and is caused by change in "
        f"`WorkflowImageData` - namely adding metadata lineage. If one of previous block "
        f"executing custom crop operation do not maintain lineage, at this phase it is impossible "
        f"to recover. This will produce error only if next blocks rely on lineage property.",
        category=RoboflowCoreBlocksIncompatibilityWarning,
    )
    if len(image_lineage) < 2:
        warnings.warn(
            f"Workflow block roboflow_core/detections_stitch@v1 could not correctly "
            f"re-assign predictions parent ID unwrapping previous crop operation symmetrically. "
            f"That may cause errors at latter stages of execution and is caused by change in "
            f"`WorkflowImageData` - namely adding metadata lineage. If one of previous block "
            f"executing custom crop operation do not maintain lineage, at this phase it is impossible "
            f"to recover. This will produce error only if next blocks rely on lineage property.",
            category=RoboflowCoreBlocksIncompatibilityWarning,
        )
        return detections
    parent_id = image_lineage[-2].parent_id
    detections.data[PARENT_ID_KEY] = np.array([parent_id] * len(detections))
    return detections


def move_detections(
    detections: sv.Detections,
    offset: Optional[np.ndarray],
    resolution_wh: Optional[Tuple[int, int]],
) -> sv.Detections:
    """
    Copied from: https://github.com/roboflow/supervision/blob/5123085037ec594524fc8f9d9b71b1cd9f487e8d/supervision/detection/tools/inference_slicer.py#L17-L16
    to avoid fragile contract with supervision, as this function is not element of public
    API.
    """
    if len(detections) == 0:
        return detections
    if offset is None:
        raise ValueError("To move non-empty detections offset is needed, but not given")
    detections.xyxy = move_boxes(xyxy=detections.xyxy, offset=offset)
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
