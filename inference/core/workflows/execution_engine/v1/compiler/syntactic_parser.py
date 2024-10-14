from typing import Dict, List, Optional, Type, Union

import pydantic
from pydantic import BaseModel, Field, create_model
from typing_extensions import Annotated

from inference.core.entities.responses.workflows import WorkflowsBlocksSchemaDescription
from inference.core.workflows.errors import WorkflowSyntaxError
from inference.core.workflows.execution_engine.entities.base import InputType, JsonField
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.cache import (
    BasicWorkflowsCache,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)

WORKFLOW_DEFINITION_ENTITIES_CACHE = BasicWorkflowsCache[Type[BaseModel]](
    cache_size=64,
    hash_functions=[
        (
            "available_blocks",
            lambda blocks: "<|>".join(
                block.block_source + block.identifier for block in blocks
            ),
        )
    ],
)


@execution_phase(
    name="workflow_definition_parsing",
    categories=["execution_engine_operation"],
)
def parse_workflow_definition(
    raw_workflow_definition: dict,
    available_blocks: List[BlockSpecification],
    profiler: Optional[WorkflowsProfiler] = None,
) -> ParsedWorkflowDefinition:
    workflow_definition_class = build_workflow_definition_entity(
        available_blocks=available_blocks,
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
    available_blocks: List[BlockSpecification],
) -> Type[BaseModel]:
    cache_key = WORKFLOW_DEFINITION_ENTITIES_CACHE.get_hash_key(
        available_blocks=available_blocks
    )
    cached_value = WORKFLOW_DEFINITION_ENTITIES_CACHE.get(key=cache_key)
    if cached_value is not None:
        return cached_value
    steps_manifests = tuple(block.manifest_class for block in available_blocks)
    block_manifest_types_union = Union[steps_manifests]
    block_type = Annotated[block_manifest_types_union, Field(discriminator="type")]
    entity = create_model(
        "WorkflowSpecificationV1",
        version=(str, ...),
        inputs=(List[InputType], ...),
        steps=(List[block_type], ...),
        outputs=(List[JsonField], ...),
    )
    WORKFLOW_DEFINITION_ENTITIES_CACHE.cache(
        key=cache_key,
        value=entity,
    )
    return entity


def get_workflow_schema_description() -> WorkflowsBlocksSchemaDescription:
    available_blocks = load_workflow_blocks()
    workflow_definition_class = build_workflow_definition_entity(
        available_blocks=available_blocks
    )
    schema = workflow_definition_class.model_json_schema()
    return WorkflowsBlocksSchemaDescription(schema=schema)
