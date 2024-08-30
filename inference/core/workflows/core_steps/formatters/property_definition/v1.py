from typing import Any, List, Literal, Optional, Type

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
    BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
    BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
    StepOutputSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

LONG_DESCRIPTION = """
Define a field using properties from previous workflow steps.

Example use-cases:
* extraction of all class names for object-detection predictions
* extraction of class / confidence from classification result
* extraction ocr text from OCR result
"""

SHORT_DESCRIPTION = "Define a variable from model predictions, such as the class names, confidences, or number of detections."


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Property Definition",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "formatter",
            "search_keywords": [
                "property",
                "field",
                "number",
                "count",
                "classes",
                "confidences",
                "labels",
                "coordinates",
            ],
        }
    )
    type: Literal[
        "roboflow_core/property_definition@v1",
        "PropertyDefinition",
        "PropertyExtraction",
    ]
    data: StepOutputSelector(
        kind=[
            BATCH_OF_OBJECT_DETECTION_PREDICTION_KIND,
            BATCH_OF_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            BATCH_OF_KEYPOINT_DETECTION_PREDICTION_KIND,
            BATCH_OF_CLASSIFICATION_PREDICTION_KIND,
        ]
    ) = Field(
        description="Reference data to extract property from",
        examples=["$steps.my_step.predictions"],
    )
    operations: List[AllOperationsType]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class PropertyDefinitionBlockV1(WorkflowBlock):

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        data: Any,
        operations: List[AllOperationsType],
    ) -> BlockResult:
        operations_chain = build_operations_chain(operations=operations)
        return {"output": operations_chain(data, global_parameters={})}
