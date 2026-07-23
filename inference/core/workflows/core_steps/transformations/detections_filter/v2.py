from copy import copy
from typing import Any, Dict, List, Literal, Optional, Type, Union

import supervision as sv
from pydantic import ConfigDict, Field, model_validator

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
    CLASSIFICATION_PREDICTION_KIND,
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
Filter detection or classification predictions using customizable UQL conditions while preserving the prediction container expected by downstream workflow blocks.

## How This Block Works

For object detection, instance segmentation, and keypoint detection inputs, this block behaves like Detections Filter v1: it evaluates a `DetectionsFilter` predicate for each spatial detection and returns an indexed `sv.Detections` subset.

For classification inputs, it evaluates a `ClassificationFilter` predicate for each selected class prediction. Single-label results retain matching entries in `predictions` and recompute `top` and `confidence`. Multi-label results retain matching entries in `predicted_classes` while preserving the complete class score map in `predictions`.

Use `ExtractDetectionProperty` inside `DetectionsFilter` conditions and `ExtractClassificationPredictionProperty` inside `ClassificationFilter` conditions. Both support filtering by class name, class ID, or confidence where applicable.

When every candidate is removed, detection inputs return an empty `sv.Detections`; single-label classification returns `predictions=[]`, `top=""`, and `confidence=0.0`; multi-label classification returns `predicted_classes=[]`.

## Version Differences

Version 2 adds classification prediction inputs and classification-specific filtering. Existing v1 detection configurations remain valid after changing only the block type to `roboflow_core/detections_filter@v2`.
"""

CLASSIFICATION_OPERATIONS_EXAMPLE = [
    {
        "type": "ClassificationFilter",
        "filter_operation": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {
                        "type": "DynamicOperand",
                        "operations": [
                            {
                                "type": "ExtractClassificationPredictionProperty",
                                "property_name": "confidence",
                            }
                        ],
                    },
                    "comparator": {"type": "(Number) >="},
                    "right_operand": {
                        "type": "DynamicOperand",
                        "operand_name": "threshold",
                    },
                }
            ],
        },
    }
]

DETECTION_KINDS = [
    OBJECT_DETECTION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Detections Filter",
            "version": "v2",
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
    type: Literal["roboflow_core/detections_filter@v2"]
    predictions: Selector(kind=DETECTION_KINDS + [CLASSIFICATION_PREDICTION_KIND]) = (
        Field(
            description="Detection or classification predictions to filter. Use a "
            "`DetectionsFilter` operation for detection inputs and a "
            "`ClassificationFilter` operation for classification inputs.",
            examples=[
                "$steps.object_detection_model.predictions",
                "$steps.classification_model.predictions",
            ],
        )
    )
    operations: List[AllOperationsType] = Field(
        description="UQL operation chain defining the filter. Existing detection "
        "operation chains from v1 remain supported. Classification chains use "
        "`ClassificationFilter` with per-class property extraction.",
        examples=[CLASSIFICATION_OPERATIONS_EXAMPLE],
    )
    operations_parameters: Dict[str, Selector()] = Field(
        description="Runtime values referenced by the operation chain. Values may be "
        "scalars or batch-aligned selectors.",
        examples=[{"threshold": "$inputs.threshold"}],
        default_factory=lambda: {},
    )

    @model_validator(mode="after")
    def validate_filter_family(self) -> "BlockManifest":
        operation_types = {operation.type for operation in self.operations}
        if {"DetectionsFilter", "ClassificationFilter"}.issubset(operation_types):
            raise ValueError(
                "Detections Filter v2 cannot mix `DetectionsFilter` and "
                "`ClassificationFilter` operations in one chain."
            )
        return self

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
                kind=DETECTION_KINDS + [CLASSIFICATION_PREDICTION_KIND],
            )
        ]

    def get_actual_outputs(self) -> List[OutputDefinition]:
        if any(
            operation.type == "ClassificationFilter" for operation in self.operations
        ):
            return [
                OutputDefinition(
                    name="predictions",
                    kind=[CLASSIFICATION_PREDICTION_KIND],
                )
            ]
        return [OutputDefinition(name="predictions", kind=DETECTION_KINDS)]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class DetectionsFilterBlockV2(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        predictions: Batch[Union[sv.Detections, dict]],
        operations: List[OperationDefinition],
        operations_parameters: Dict[str, Any],
    ) -> BlockResult:
        return execute_transformation(
            predictions=predictions,
            operations=operations,
            operations_parameters=operations_parameters,
        )


def execute_transformation(
    predictions: Batch[Union[sv.Detections, dict]],
    operations: List[OperationDefinition],
    operations_parameters: Dict[str, Any],
) -> BlockResult:
    if DEFAULT_OPERAND_NAME in operations_parameters:
        raise ValueError(
            f"Detected reserved parameter name: {DEFAULT_OPERAND_NAME} declared in "
            "`operations_parameters` of Detections Filter v2."
        )
    operation_types = {operation.type for operation in operations}
    classification_mode = "ClassificationFilter" in operation_types
    if classification_mode and "DetectionsFilter" in operation_types:
        raise ValueError(
            "Detections Filter v2 cannot mix `DetectionsFilter` and "
            "`ClassificationFilter` operations in one chain."
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
        batch_parameters[key] for key in batch_parameters_keys
    ]
    results = []
    for payload in zip(*batches_to_align):
        prediction = payload[0]
        _validate_prediction_matches_mode(
            prediction=prediction,
            classification_mode=classification_mode,
        )
        single_evaluation_parameters = copy(non_batch_parameters)
        for key, value in zip(batch_parameters_keys, payload[1:]):
            single_evaluation_parameters[key] = value
        transformed_prediction = operations_chain(
            prediction,
            global_parameters=single_evaluation_parameters,
        )
        _validate_prediction_matches_mode(
            prediction=transformed_prediction,
            classification_mode=classification_mode,
        )
        results.append({"predictions": transformed_prediction})
    return results


def _validate_prediction_matches_mode(
    prediction: Any,
    classification_mode: bool,
) -> None:
    if classification_mode and not isinstance(prediction, dict):
        raise ValueError(
            "A `ClassificationFilter` operation requires classification predictions."
        )
    if classification_mode and (
        "image" not in prediction
        or "predictions" not in prediction
        or (
            "predicted_classes" not in prediction
            and ("top" not in prediction or "confidence" not in prediction)
        )
    ):
        raise ValueError(
            "A `ClassificationFilter` operation must preserve the classification "
            "prediction response structure."
        )
    if not classification_mode and not isinstance(prediction, sv.Detections):
        raise ValueError(
            "Classification predictions require a `ClassificationFilter` operation."
        )
