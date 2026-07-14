from typing import List, Literal, Optional, Type, Union
from uuid import uuid4

import torch
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    TensorNativePrediction,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    IMAGE_DIMENSIONS_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections

LONG_DESCRIPTION = """
Expand or contract detection bounding boxes by applying fixed offsets to their width and height, adding padding around detections to include more context, adjust bounding box sizes for downstream processing, or compensate for tight detections, supporting both pixel-based and percentage-based offset units for flexible bounding box adjustment.

## How This Block Works

This block adjusts the size of detection bounding boxes by adding offsets to their dimensions, effectively expanding or contracting the boxes to include more or less context around detected objects. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) containing bounding boxes
2. Processes each detection's bounding box coordinates independently
3. Calculates offsets based on the selected unit type:
   - **Pixel-based offsets**: Adds/subtracts a fixed number of pixels on each side (offset_width//2 pixels on left/right, offset_height//2 pixels on top/bottom)
   - **Percentage-based offsets**: Calculates offsets as a percentage of the bounding box's width and height (offset_width% of box width, offset_height% of box height)
4. Applies the offsets to expand the bounding boxes:
   - Subtracts half the width offset from x_min and adds half to x_max (expands horizontally)
   - Subtracts half the height offset from y_min and adds half to y_max (expands vertically)
5. Clips the adjusted bounding boxes to image boundaries (ensures coordinates stay within image dimensions using min/max constraints)
6. Updates detection metadata:
   - Sets parent_id_key to reference the original detection IDs (preserves traceability)
   - Generates new detection IDs for the offset detections (tracks that these are modified versions)
7. Preserves all other detection properties (masks, keypoints, polygons, class labels, confidence scores) unchanged
8. Returns the modified detections with expanded or contracted bounding boxes

The block applies offsets symmetrically around the center of each bounding box, expanding the box equally on all sides based on the width and height offsets. Positive offsets expand boxes (add padding), while the implementation always expands boxes outward. The pixel-based mode applies fixed pixel offsets regardless of box size, useful for consistent padding. The percentage-based mode applies offsets proportional to box size, useful when padding should scale with the detected object size. Boxes are automatically clipped to image boundaries to prevent invalid coordinates.

## Common Use Cases

- **Context Padding for Analysis**: Expand tight bounding boxes to include more surrounding context (e.g., add padding around detected objects for better classification, expand boxes to include object context for feature extraction, add margin around text detections for OCR), enabling improved analysis with additional context
- **Detection Size Adjustment**: Adjust bounding box sizes to match downstream processing requirements (e.g., expand boxes for models that need larger input regions, adjust box sizes to accommodate specific analysis needs, modify detections for compatibility with other blocks), enabling size customization for workflow compatibility
- **Tight Detection Compensation**: Expand overly tight bounding boxes that cut off parts of objects (e.g., add padding to tight object detections, expand boxes that miss object edges, compensate for models that produce undersized boxes), enabling better object coverage
- **Multi-Stage Workflow Preparation**: Prepare detections with adjusted sizes for secondary processing (e.g., expand initial detections before running secondary models, adjust box sizes for specialized analysis blocks, prepare detections with context for detailed processing), enabling optimized multi-stage workflows
- **Crop Region Optimization**: Adjust bounding boxes before cropping to include desired context (e.g., add padding before dynamic cropping to include surrounding context, expand boxes to capture more area for analysis, adjust crop regions for better feature extraction), enabling optimized region extraction
- **Visualization and Display**: Adjust bounding box sizes for better visualization or display purposes (e.g., expand boxes for clearer annotations, adjust box sizes for presentation, modify detections for visualization consistency), enabling improved visual outputs

## Connecting to Other Blocks

This block receives detection predictions and produces adjusted detections with modified bounding boxes:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to expand or adjust bounding box sizes before further processing, enabling size-optimized detections for downstream analysis
- **Before dynamic crop blocks** to adjust bounding box sizes before cropping, enabling optimized crop regions with desired context or padding
- **Before classification or analysis blocks** that benefit from additional context around detections (e.g., classification with context, feature extraction from expanded regions, detailed analysis with padding), enabling improved analysis with context
- **In multi-stage detection workflows** where initial detections need size adjustments before secondary processing (e.g., expand initial detections before running specialized models, adjust box sizes for compatibility, prepare detections for optimized processing), enabling flexible multi-stage workflows
- **Before visualization blocks** to adjust bounding box sizes for display purposes (e.g., expand boxes for clearer annotations, adjust sizes for presentation, modify detections for visualization consistency), enabling optimized visual outputs
- **Before blocks that process detection regions** where bounding box size matters (e.g., OCR on text regions with padding, feature extraction from expanded regions, specialized models requiring specific box sizes), enabling size-optimized region processing
"""

SHORT_DESCRIPTION = "Apply a padding around the width and height of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detection Offset",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-distribute-spacing-horizontal",
                "blockPriority": 3,
            },
        }
    )
    type: Literal["roboflow_core/detection_offset@v1", "DetectionOffset"]
    predictions: Selector(
        kind=[
            TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions containing bounding boxes to adjust. Supports object detection, instance segmentation, or keypoint detection predictions. The bounding boxes in these predictions will be expanded or contracted based on the offset_width and offset_height values. All detection properties (masks, keypoints, polygons, classes, confidence) are preserved unchanged - only bounding box coordinates are modified.",
        examples=["$steps.object_detection_model.predictions"],
    )
    offset_width: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Offset value to apply to bounding box width. Must be a positive integer. If units is 'Pixels', this is the number of pixels added to the box width (divided equally between left and right sides - offset_width//2 pixels on each side). If units is 'Percent (%)', this is the percentage of the bounding box width to add (calculated as percentage of the box's width, then divided between left and right). Positive values expand boxes horizontally. Boxes are clipped to image boundaries automatically.",
        examples=[10, "$inputs.offset_x"],
        validation_alias=AliasChoices("offset_width", "offset_x"),
    )
    offset_height: Union[PositiveInt, Selector(kind=[INTEGER_KIND])] = Field(
        description="Offset value to apply to bounding box height. Must be a positive integer. If units is 'Pixels', this is the number of pixels added to the box height (divided equally between top and bottom sides - offset_height//2 pixels on each side). If units is 'Percent (%)', this is the percentage of the bounding box height to add (calculated as percentage of the box's height, then divided between top and bottom). Positive values expand boxes vertically. Boxes are clipped to image boundaries automatically.",
        examples=[10, "$inputs.offset_y"],
        validation_alias=AliasChoices("offset_height", "offset_y"),
    )
    units: Literal["Percent (%)", "Pixels"] = Field(
        default="Pixels",
        description="Unit type for offset values: 'Pixels' for fixed pixel offsets (same number of pixels for all boxes regardless of size) or 'Percent (%)' for percentage-based offsets (proportional to each bounding box's dimensions). Pixel offsets provide consistent padding in absolute terms. Percentage offsets scale with box size, providing proportional padding. Use pixels when you need consistent absolute padding. Use percentage when padding should scale with detected object size.",
        examples=["Pixels", "Percent (%)"],
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(
                name="predictions",
                kind=[
                    TENSOR_NATIVE_OBJECT_DETECTION_PREDICTION_KIND,
                    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    TENSOR_NATIVE_KEYPOINT_DETECTION_PREDICTION_KIND,
                ],
            ),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionOffsetBlockV1(WorkflowBlock):
    # TODO: This block breaks parent coordinates.
    # Issue report: https://github.com/roboflow/inference/issues/380

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[TensorNativePrediction],
        offset_width: int,
        offset_height: int,
        units: str = "Pixels",
    ) -> BlockResult:
        use_percentage = units == "Percent (%)"
        return [
            {
                "predictions": offset_detections(
                    prediction=prediction,
                    offset_width=offset_width,
                    offset_height=offset_height,
                    use_percentage=use_percentage,
                )
            }
            for prediction in predictions
        ]


def _read_image_dimensions(detections: TensorNativeDetections) -> tuple:
    """Return ``(height, width)`` from the prediction's ``image_metadata`` (one
    ``[height, width]`` pair shared by every box of the prediction). Raises when
    ``IMAGE_DIMENSIONS_KEY`` is absent rather than degrading to an un-clamped
    offset."""
    image_metadata = detections.image_metadata or {}
    image_dimensions = image_metadata.get(IMAGE_DIMENSIONS_KEY)
    if image_dimensions is None:
        raise ValueError(
            "Detection Offset block requires image dimensions to clamp the "
            f"offset boxes, but `{IMAGE_DIMENSIONS_KEY}` is missing from the "
            "prediction's image_metadata."
        )
    return int(image_dimensions[0]), int(image_dimensions[1])


def _offset_xyxy(
    xyxy: torch.Tensor,
    offset_width: int,
    offset_height: int,
    image_height: Optional[int],
    image_width: Optional[int],
    use_percentage: bool,
) -> torch.Tensor:
    """Expand each box outward, clamped to image bounds. Percentage mode pads by
    a fraction of each box's own width/height; pixel mode pads by a fixed
    (offset // 2) on each side."""
    x1 = xyxy[:, 0]
    y1 = xyxy[:, 1]
    x2 = xyxy[:, 2]
    y2 = xyxy[:, 3]
    if use_percentage:
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = (box_width * offset_width / 200).to(torch.int64).to(xyxy.dtype)
        pad_y = (box_height * offset_height / 200).to(torch.int64).to(xyxy.dtype)
    else:
        pad_x = torch.full_like(x1, float(offset_width // 2))
        pad_y = torch.full_like(y1, float(offset_height // 2))
    new_x1 = torch.clamp(x1 - pad_x, min=0)
    new_y1 = torch.clamp(y1 - pad_y, min=0)
    new_x2 = x2 + pad_x
    new_y2 = y2 + pad_y
    if image_width is not None:
        new_x2 = torch.clamp(new_x2, max=image_width)
    if image_height is not None:
        new_y2 = torch.clamp(new_y2, max=image_height)
    return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=1)


def _rebuild_detections(
    detections: TensorNativeDetections,
    new_xyxy: torch.Tensor,
    new_bboxes_metadata: Optional[List[dict]],
) -> TensorNativeDetections:
    """Rebuild a native ``Detections`` / ``InstanceDetections`` with offset boxes;
    class_id / confidence / mask / image_metadata are carried unchanged."""
    if isinstance(detections, InstanceDetections):
        return InstanceDetections(
            xyxy=new_xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=detections.mask,
            image_metadata=detections.image_metadata,
            bboxes_metadata=new_bboxes_metadata,
        )
    return Detections(
        xyxy=new_xyxy,
        class_id=detections.class_id,
        confidence=detections.confidence,
        image_metadata=detections.image_metadata,
        bboxes_metadata=new_bboxes_metadata,
    )


def _offset_bboxes_metadata(
    detections: TensorNativeDetections,
    number_of_detections: int,
    parent_id_key: str,
    detection_id_key: str,
) -> List[dict]:
    """Set each box's parent_id to its prior detection_id and mint a fresh
    detection_id."""
    existing = detections.bboxes_metadata or [{} for _ in range(number_of_detections)]
    new_metadata = []
    for index in range(number_of_detections):
        entry = dict(existing[index] or {}) if index < len(existing) else {}
        entry[parent_id_key] = entry.get(detection_id_key)
        entry[detection_id_key] = str(uuid4())
        new_metadata.append(entry)
    return new_metadata


def offset_detections(
    prediction: TensorNativePrediction,
    offset_width: int,
    offset_height: int,
    parent_id_key: str = PARENT_ID_KEY,
    detection_id_key: str = DETECTION_ID_KEY,
    use_percentage: bool = False,
) -> TensorNativePrediction:
    if prediction is None:
        return prediction
    # Only boxes are offset — keypoints are left untouched and re-wrapped onto
    # the output. A prediction with a missing/empty bbox component is returned
    # unchanged.
    if isinstance(prediction, tuple):
        key_points, detections = prediction
    else:
        key_points, detections = None, prediction
    if (
        detections is None
        or detections.xyxy is None
        or int(detections.xyxy.shape[0]) == 0
    ):
        return prediction
    number_of_detections = int(detections.xyxy.shape[0])
    image_height, image_width = _read_image_dimensions(detections)
    new_xyxy = _offset_xyxy(
        xyxy=detections.xyxy,
        offset_width=offset_width,
        offset_height=offset_height,
        image_height=image_height,
        image_width=image_width,
        use_percentage=use_percentage,
    )
    new_bboxes_metadata = _offset_bboxes_metadata(
        detections=detections,
        number_of_detections=number_of_detections,
        parent_id_key=parent_id_key,
        detection_id_key=detection_id_key,
    )
    new_detections = _rebuild_detections(
        detections=detections,
        new_xyxy=new_xyxy,
        new_bboxes_metadata=new_bboxes_metadata,
    )
    if key_points is not None:
        return key_points, new_detections
    return new_detections
