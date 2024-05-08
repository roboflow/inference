from typing import List, Literal, Type, Union

import pydantic
from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated

from inference.core.workflows.entities.base import InputType, JsonField
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.compiler.entities import (
    ParsedWorkflowDefinition,
)
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)


def parse_workflow_definition(
    raw_workflow_definition: dict,
) -> ParsedWorkflowDefinition:
    workflow_definition_class = build_workflow_definition_entity()
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


def build_workflow_definition_entity() -> Type[BaseModel]:
    blocks = load_workflow_blocks()
    steps_manifests = tuple(block.manifest_class for block in blocks)
    block_manifest_types_union = Union[steps_manifests]
    block_type = Annotated[block_manifest_types_union, Field(discriminator="type")]
    return create_model(
        "WorkflowSpecificationV1",
        version=(Literal["1.0"], ...),
        inputs=(List[InputType], ...),
        steps=(List[block_type], ...),
        outputs=(List[JsonField], ...),
    )
