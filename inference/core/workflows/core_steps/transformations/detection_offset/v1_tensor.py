import uuid
from copy import deepcopy
from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
from pydantic import AliasChoices, ConfigDict, Field, PositiveInt

from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    INTEGER_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

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
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
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
                    OBJECT_DETECTION_PREDICTION_KIND,
                    INSTANCE_SEGMENTATION_PREDICTION_KIND,
                    KEYPOINT_DETECTION_PREDICTION_KIND,
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
        predictions: Batch[sv.Detections],
        offset_width: int,
        offset_height: int,
        units: str = "Pixels",
    ) -> BlockResult:
        use_percentage = units == "Percent (%) - of bounding box width / height"
        return [
            {
                "predictions": offset_detections(
                    detections=detections,
                    offset_width=offset_width,
                    offset_height=offset_height,
                    use_percentage=use_percentage,
                )
            }
            for detections in predictions
        ]


def offset_detections(
    detections: sv.Detections,
    offset_width: int,
    offset_height: int,
    parent_id_key: str = PARENT_ID_KEY,
    detection_id_key: str = DETECTION_ID_KEY,
    use_percentage: bool = False,
) -> sv.Detections:
    if len(detections) == 0:
        return detections
    _detections = deepcopy(detections)
    image_dimensions = detections.data["image_dimensions"]
    if use_percentage:
        _detections.xyxy = np.array(
            [
                (
                    max(0, x1 - int(box_width * offset_width / 200)),
                    max(0, y1 - int(box_height * offset_height / 200)),
                    min(
                        image_dimensions[i][1],
                        x2 + int(box_width * offset_width / 200),
                    ),
                    min(
                        image_dimensions[i][0],
                        y2 + int(box_height * offset_height / 200),
                    ),
                )
                for i, (x1, y1, x2, y2) in enumerate(_detections.xyxy)
                for box_width, box_height in [(x2 - x1, y2 - y1)]
            ]
        )
    else:
        _detections.xyxy = np.array(
            [
                (
                    max(0, x1 - offset_width // 2),
                    max(0, y1 - offset_height // 2),
                    min(image_dimensions[i][1], x2 + offset_width // 2),
                    min(image_dimensions[i][0], y2 + offset_height // 2),
                )
                for i, (x1, y1, x2, y2) in enumerate(_detections.xyxy)
            ]
        )
    _detections[parent_id_key] = detections[detection_id_key].copy()
    _detections[detection_id_key] = [str(uuid.uuid4()) for _ in detections]
    return _detections
