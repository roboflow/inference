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
Block changes detected Bounding Boxes in a way specified in configuration.

It supports such operations as changing the size of Bounding Boxes. 
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
        description="Model predictions to transform.",
        examples=["$steps.object_detection_model.predictions"],
    )
    operations: List[AllOperationsType] = Field(
        description="Transformations to be applied on the predictions.",
        examples=[OPERATIONS_EXAMPLE],
    )
    operations_parameters: Dict[
        str,
        Selector(),
    ] = Field(
        description="References to additional parameters that may be provided in runtime to parameterize operations",
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
