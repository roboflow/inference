from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
    OperationDefinition,
)
from inference.core.workflows.core_steps.transformations.detections_transformation.v1 import (
    execute_transformation,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
)
from inference.core.workflows.execution_engine.entities.types import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Conditionally filter out model predictions."

LONG_DESCRIPTION = """
Filter detection predictions based on customizable conditions, selectively removing detections that don't meet specified criteria (e.g., class names, confidence scores, bounding box properties) while preserving only the detections that match your filtering logic.

## How This Block Works

This block applies conditional filtering to detection predictions using a flexible query language system. The block:

1. Takes detection predictions (object detection, instance segmentation, or keypoint detection) and filtering operation definitions as input
2. Evaluates each detection against the filtering conditions specified in the `operations` parameter
3. Extracts detection properties (e.g., class_name, confidence, bounding box coordinates) using property extraction operations
4. Compares extracted properties against criteria using binary statements (e.g., class_name in list, confidence > threshold)
5. Filters out detections that don't match the conditions, keeping only detections that satisfy the filter criteria
6. Returns filtered predictions containing only the detections that passed the filter conditions

The block uses a query language system that supports extracting various detection properties (class names, confidence scores, bounding box coordinates, etc.) and applying conditional logic to filter detections. Filtering operations can check if properties are in lists, compare numeric values, check string equality, or use other comparators. The `operations_parameters` dictionary provides runtime values (like class name lists or thresholds) that are referenced in the filtering operations, allowing dynamic filtering criteria that can change based on workflow inputs or computed values. Multiple filtering operations can be chained together to create complex filtering logic.

## Common Use Cases

- **Class-Based Filtering**: Filter detections to keep only specific object classes (e.g., keep only "person" and "car" detections, remove all others), enabling focused processing on relevant object types while excluding unwanted detections
- **Confidence Threshold Filtering**: Remove low-confidence detections to improve detection quality (e.g., keep detections with confidence > 0.7, filter out uncertain predictions), ensuring downstream processing works with reliable detections
- **Multi-Criteria Filtering**: Apply multiple filtering conditions simultaneously (e.g., keep detections where class_name is in allowed list AND confidence > threshold), combining class and confidence filtering for precise control
- **Dynamic Filtering Based on Workflow State**: Use workflow inputs or computed values to determine filtering criteria (e.g., filter classes based on user input, adjust confidence threshold based on lighting conditions), enabling adaptive filtering that responds to changing conditions
- **Pre-Processing for Downstream Blocks**: Filter detections before passing to visualization, counting, or storage blocks (e.g., remove false positives before counting, filter out background classes before visualization), reducing noise and improving accuracy of subsequent operations
- **Selective Processing Workflows**: Route different filtered subsets to different downstream blocks (e.g., filter high-confidence detections to one path, low-confidence to another), enabling conditional processing based on detection quality or type

## Connecting to Other Blocks

The filtered predictions from this block can be connected to:

- **Detection model blocks** (e.g., Object Detection Model, Instance Segmentation Model, Keypoint Detection Model) to receive predictions that are filtered based on class, confidence, or other properties
- **Visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Label Visualization) to display only the filtered detections, reducing visual clutter and focusing on relevant objects
- **Counting and analytics blocks** (e.g., Line Counter, Time in Zone, Velocity) to count or analyze only specific filtered classes or confidence levels, ensuring accurate metrics for the objects of interest
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload, Webhook Sink) to save or transmit only filtered detection results, reducing storage and bandwidth usage by excluding irrelevant detections
- **Other transformation blocks** (e.g., Detections Merge, Detections Transform, Detection Offset) to apply additional transformations to the filtered subset, enabling complex processing pipelines on filtered detections
- **Flow control blocks** (e.g., Continue If, Rate Limiter) to conditionally trigger downstream processing based on whether filtered detections meet certain criteria, enabling conditional workflows based on filtered results
"""

OPERATIONS_EXAMPLE = [
    {
        "type": "DetectionsFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractDetectionProperty",
                                "property_name": "class_name",
                            }
                        ],
                    },
                    "comparator": {"type": "in (Sequence)"},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "classes",
                    },
                },
            ],
        },
    }
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Filter",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "flow_control",
                "icon": "far fa-filter",
                "blockPriority": 1,
            },
        }
    )
    type: Literal["roboflow_core/detections_filter@v1", "DetectionsFilter"]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions to filter (object detection, instance segmentation, or keypoint detection). Each detection is evaluated against the filtering conditions specified in the operations parameter. Only detections that match the filter criteria are included in the output. Supports batch processing, allowing filtering of multiple detection sets simultaneously.",
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[AllOperationsType] = Field(
        description="Definition of filtering logic using the query language system. Specifies one or more filtering operations (e.g., DetectionsFilter) that use StatementGroup syntax to define conditional logic. Each operation can extract detection properties (class_name, confidence, coordinates, etc.) and compare them using binary statements (e.g., class_name in list, confidence > threshold). Multiple operations can be chained to create complex filtering logic. The operations reference parameter names from operations_parameters to access runtime values.",
        examples=[OPERATIONS_EXAMPLE]
    )
    operations_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Dictionary mapping parameter names (referenced in operations) to actual values from the workflow. These parameters provide runtime values used in filtering operations (e.g., class name lists, confidence thresholds). Keys match parameter names used in the operations definition, and values are selectors referencing workflow inputs, step outputs, or computed values. Example: {'classes': '$inputs.allowed_classes', 'threshold': 0.7} where 'classes' and 'threshold' are referenced in the operations.",
        examples=[
            {
                "classes": "$inputs.classes",
            }
        ],
        default_factory=lambda: {},
    )

    @classmethod
    def get_parameters_accepting_batches(cls) -> List[str]:
        return ["predictions"]

    @classmethod
    def get_parameters_accepting_batches_and_scalars(cls) -> List[str]:
        return ["operations_parameters"]

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
            )
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsFilterBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[sv.Detections],
        operations: List[OperationDefinition],
        operations_parameters: Dict[str, Any],
    ) -> BlockResult:
        return execute_transformation(
            predictions=predictions,
            operations=operations,
            operations_parameters=operations_parameters,
        )
