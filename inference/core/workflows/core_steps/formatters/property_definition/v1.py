from typing import Any, List, Literal, Optional, Type, Union

from pydantic import ConfigDict, Field

from inference.core.workflows.core_steps.common.query_language.entities.operations import (
    AllOperationsType,
)
from inference.core.workflows.core_steps.common.query_language.operations.core import (
    build_operations_chain,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import Selector
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
            "ui_manifest": {
                "section": "advanced",
                "icon": "fal fa-gear-code",
                "blockPriority": 0,
                "popular": True,
            },
        }
    )
    type: Literal[
        "roboflow_core/property_definition@v1",
        "PropertyDefinition",
        "PropertyExtraction",
    ]
    data: Selector() = Field(
        description="Data to extract property from.",
        examples=["$steps.my_step.predictions"],
    )
    operations: List[AllOperationsType] = Field(
        description="List of operations to perform on the data.",
        examples=[
            [{"type": "DetectionsPropertyExtract", "property_name": "class_name"}]
        ],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="output")]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


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
