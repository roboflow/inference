from typing import List, Optional, Type, Union

import pydantic
from packaging.version import Version
from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated

from inference.core.entities.responses.workflows import WorkflowsBlocksSchemaDescription
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)


def parse_workflow_definition(
    raw_workflow_definition: dict,
    dynamic_blocks: List[BlockSpecification],
    execution_engine_version: Optional[Version] = None,
) -> ParsedWorkflowDefinition:
    workflow_definition_class = build_workflow_definition_entity(
        dynamic_blocks=dynamic_blocks,
        execution_engine_version=execution_engine_version,
    )
    try:
        workflow_definition = workflow_definition_class.model_validate(
            raw_workflow_definition
        )
        return ParsedWorkflowDefinition(
            version=workflow_definition.version,
            inputs=workflow_definition.inputs,
            steps=workflow_definition.steps,
            outputs=workflow_definition.outputs,
        )
    except pydantic.ValidationError as e:
        raise WorkflowSyntaxError(
            public_message="Could not parse workflow definition. Details provided in inner error.",
            context="workflow_compilation | workflow_definition_parsing",
            inner_error=e,
        ) from e


def build_workflow_definition_entity(
    dynamic_blocks: List[BlockSpecification],
    execution_engine_version: Optional[Version] = None,
) -> Type[BaseModel]:
    blocks = (
        load_workflow_blocks(execution_engine_version=execution_engine_version)
        + dynamic_blocks
    )
    steps_manifests = tuple(block.manifest_class for block in blocks)
    block_manifest_types_union = Union[steps_manifests]
    block_type = Annotated[block_manifest_types_union, Field(discriminator="type")]
    return create_model(
        "WorkflowSpecificationV1",
        version=(str, ...),
        inputs=(List[InputType], ...),
        steps=(List[block_type], ...),
        outputs=(List[JsonField], ...),
    )


def get_workflow_schema_description() -> WorkflowsBlocksSchemaDescription:
    workflow_definition_class = build_workflow_definition_entity([])
    schema = workflow_definition_class.model_json_schema()
    return WorkflowsBlocksSchemaDescription(schema=schema)
