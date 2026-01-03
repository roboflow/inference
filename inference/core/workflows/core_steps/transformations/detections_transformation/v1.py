from copy import copy
from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    DEFAULT_OPERAND_NAME,
    AllOperationsType,
    OperationDefinition,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.core_steps.common.utils import (
    grab_batch_parameters,
    grab_non_batch_parameters,
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

LONG_DESCRIPTION = """
Apply customizable transformations to detection predictions using UQL (Query Language) operation chains, enabling flexible modification of bounding boxes, filtering detections, extracting properties, resizing boxes, and other detection manipulations through configurable operation sequences for advanced detection processing workflows.

## How This Block Works

This block transforms detection predictions by applying a chain of UQL operations that can modify, filter, extract, or manipulate detection data. The block:

1. Receives detection predictions (object detection, instance segmentation, or keypoint detection) and a list of UQL operations to apply
2. Validates that operations_parameters doesn't contain reserved parameter names
3. Builds an operations chain from the provided UQL operation definitions, creating a sequence of transformations to apply in order
4. Separates operations_parameters into batch parameters (aligned with predictions) and non-batch parameters (applied to all predictions)
5. Processes each prediction batch by applying the operations chain:
   - Zips predictions with batch parameters to align data per batch item
   - Combines batch and non-batch parameters into evaluation parameters for each prediction
   - Applies the operations chain to the detections with the combined parameters
   - Validates that the output is still sv.Detections (operations must preserve detection type)
6. Returns the transformed detections for each input batch

The block supports a wide variety of UQL operations including filtering (DetectionsFilter), property extraction (ExtractDetectionProperty), bounding box transformations (resizing, scaling), and other detection manipulations. Operations are applied sequentially, allowing complex transformations through operation chaining. The block validates that transformations preserve the detection type, ensuring outputs remain compatible with other detection-processing blocks. Batch and non-batch parameters enable flexible operation parameterization, supporting both per-detection and global parameter values.

## Common Use Cases

- **Advanced Detection Filtering**: Apply complex filtering logic to detection predictions (e.g., filter detections by class names using conditional statements, filter by confidence thresholds with multiple conditions, apply custom filtering criteria based on detection properties), enabling sophisticated detection selection workflows
- **Bounding Box Transformations**: Modify bounding box sizes, positions, or properties (e.g., resize bounding boxes proportionally, scale boxes by percentage, adjust box coordinates, transform box dimensions), enabling flexible bounding box manipulation
- **Property Extraction and Filtering**: Extract detection properties and filter based on extracted values (e.g., extract class names and filter by class lists, extract confidence scores and filter by thresholds, extract properties for conditional processing), enabling property-based detection processing
- **Multi-Conditional Processing**: Apply complex conditional transformations based on multiple detection criteria (e.g., transform detections based on class and confidence combinations, apply different operations for different detection types, conditionally modify detections based on multiple properties), enabling sophisticated conditional detection processing
- **Detection Data Enrichment**: Extract and add properties to detections for downstream processing (e.g., extract class names for filtering, compute detection properties, add metadata to detections), enabling enriched detection data for complex workflows
- **Custom Detection Manipulation**: Apply custom transformations not available in dedicated blocks (e.g., complex multi-step detection modifications, custom filtering and transformation combinations, specialized detection processing workflows), enabling flexible custom detection processing

## Connecting to Other Blocks

This block receives detection predictions and produces transformed detections:

- **After detection blocks** (e.g., Object Detection, Instance Segmentation, Keypoint Detection) to apply custom transformations, filtering, or modifications to detection predictions, enabling flexible detection processing workflows
- **Before dynamic crop blocks** to filter or modify detections before cropping (e.g., filter detections by class before cropping, transform box sizes before cropping, extract specific detections for cropping), enabling optimized region extraction workflows
- **Before classification or analysis blocks** to prepare detections with custom filtering or transformations (e.g., filter detections for specific analysis, transform boxes for compatibility, prepare detections with custom criteria), enabling customized detection preparation
- **In multi-stage detection workflows** where detections need custom transformations between stages (e.g., filter and transform initial detections before secondary processing, apply custom modifications between detection stages, conditionally process detections based on criteria), enabling sophisticated multi-stage workflows
- **Before visualization blocks** to filter or transform detections for display (e.g., filter detections for visualization, transform boxes for presentation, customize detections for display purposes), enabling optimized visual outputs
- **After detection blocks and before other transformation blocks** to apply custom logic between transformations (e.g., filter after detection and before cropping, transform between detection stages, apply conditional modifications), enabling complex transformation pipelines with custom logic
"""

SHORT_DESCRIPTION = "Apply transformations on detected bounding boxes."

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
            "name": "Detections Transformation",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "transformation",
            "ui_manifest": {
                "section": "transformation",
                "icon": "fal fa-expand",
                "blockPriority": 4,
            },
        }
    )
    type: Literal[
        "roboflow_core/detections_transformation@v1", "DetectionsTransformation"
    ]
    predictions: Selector(
        kind=[
            OBJECT_DETECTION_PREDICTION_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            KEYPOINT_DETECTION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Detection predictions to transform using UQL operations. Supports object detection, instance segmentation, or keypoint detection predictions. The detections will be transformed by the operations chain defined in the operations field. All transformations must preserve the detection type (output must remain sv.Detections). The block processes batch inputs and applies transformations per batch item.",
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[AllOperationsType] = Field(
        description="List of UQL (Query Language) operations to apply sequentially to the detections. Operations are executed in order, with each operation receiving the output of the previous operation. Supported operations include DetectionsFilter (filtering detections by conditions), ExtractDetectionProperty (extracting properties from detections), bounding box transformations (resizing, scaling), and other UQL operations that accept and return sv.Detections. Operations can be parameterized using operations_parameters. The operations chain must transform sv.Detections to sv.Detections (type must be preserved).",
        examples=[OPERATIONS_EXAMPLE],
    )
    operations_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="Dictionary mapping parameter names (used in operations) to workflow data sources or values. Parameters are referenced in operations (e.g., in conditional statements, filter operations) and provided at runtime. Supports both batch parameters (aligned with predictions, one value per batch item) and non-batch parameters (same value for all batch items). Parameters are automatically separated into batch and non-batch based on their data structure. Cannot use reserved parameter names. Use this to parameterize operations dynamically (e.g., provide class lists for filtering, provide thresholds for conditions, supply values for operations that need runtime parameters).",
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


class DetectionsTransformationBlockV1(WorkflowBlock):

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


def execute_transformation(
    predictions: Batch[sv.Detections],
    operations: List[OperationDefinition],
    operations_parameters: Dict[str, Any],
) -> BlockResult:
    if DEFAULT_OPERAND_NAME in operations_parameters:
        raise ValueError(
            f"Detected reserved parameter name: {DEFAULT_OPERAND_NAME} declared in `operations_parameters` "
            f"of `DetectionsTransformation` block."
        )
    operations_chain = build_operations_chain(operations=operations)
    batch_parameters = grab_batch_parameters(
        operations_parameters=operations_parameters,
        main_batch_size=len(predictions),
    )
    non_batch_parameters = grab_non_batch_parameters(
        operations_parameters=operations_parameters,
    )
    batch_parameters_keys = list(batch_parameters.keys())
    batches_to_align = [predictions] + [
        batch_parameters[k] for k in batch_parameters_keys
    ]
    results = []
    for payload in zip(*batches_to_align):
        detections = payload[0]
        single_evaluation_parameters = copy(non_batch_parameters)
        for key, value in zip(batch_parameters_keys, payload[1:]):
            single_evaluation_parameters[key] = value
        transformed_detections = operations_chain(
            detections,
            global_parameters=single_evaluation_parameters,
        )
        if not isinstance(transformed_detections, sv.Detections):
            raise ValueError(
                "Definition of operation chain provided to `DetectionsTransformation` block "
                f"transforms sv.Detections into different type: {type(transformed_detections)} "
                "which is not allowed."
            )
        results.append({"predictions": transformed_detections})
    return results
