from typing import Annotated, List, Literal, Type, Union

from pydantic import BaseModel, Field, create_model

from inference.enterprise.workflows.entities.outputs import JsonField
from inference.enterprise.workflows.entities.workflows_specification import InputType
from inference.enterprise.workflows.execution_engine.compiler.blocks_loader import (
    load_workflow_blocks,
)
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    ParsedWorkflowDefinition,
)


def parse_workflow_definition(
    raw_workflow_definition: dict,
) -> ParsedWorkflowDefinition:
    workflow_definition_class = build_workflow_definition_entity()
    workflow_definition = workflow_definition_class.model_validate(
        raw_workflow_definition
    )
    return ParsedWorkflowDefinition(
        version=workflow_definition.version,
        inputs=workflow_definition.inputs,
        steps=workflow_definition.steps,
        outputs=workflow_definition.outputs,
    )


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
