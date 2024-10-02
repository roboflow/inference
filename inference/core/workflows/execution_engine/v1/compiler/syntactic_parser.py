from typing import Dict, List, Optional, Type, Union

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
from inference.core.workflows.execution_engine.profiling.core import (
    WorkflowsProfiler,
    execution_phase,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)


class WorkflowDefinitionEntitiesCache:

    def __init__(self, cache_size: int):
        self._cache_size = max(1, cache_size)
        self._entries: Dict[str, Type[BaseModel]] = {}

    def add_entry(
        self, blocks: List[BlockSpecification], entry: Type[BaseModel]
    ) -> None:
        hash_value = self._hash_blocks(blocks=blocks)
        if hash_value in self._entries:
            self._entries[hash_value] = entry
            return None
        if len(self._entries) == self._cache_size:
            key_to_drop = next(self._entries.__iter__())
            self._entries.pop(key_to_drop)
        self._entries[hash_value] = entry

    def cache_hit(self, blocks: List[BlockSpecification]) -> bool:
        hash_value = self._hash_blocks(blocks=blocks)
        return hash_value in self._entries

    def get(self, blocks: List[BlockSpecification]) -> Type[BaseModel]:
        hash_value = self._hash_blocks(blocks=blocks)
        return self._entries[hash_value]

    def _hash_blocks(self, blocks: List[BlockSpecification]) -> str:
        return "<|>".join(block.block_source + block.identifier for block in blocks)


WORKFLOW_DEFINITION_ENTITIES_CACHE = WorkflowDefinitionEntitiesCache(cache_size=64)


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
    if WORKFLOW_DEFINITION_ENTITIES_CACHE.cache_hit(blocks=available_blocks):
        return WORKFLOW_DEFINITION_ENTITIES_CACHE.get(blocks=available_blocks)
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
    WORKFLOW_DEFINITION_ENTITIES_CACHE.add_entry(
        blocks=available_blocks,
        entry=entity,
    )
    return entity


def get_workflow_schema_description() -> WorkflowsBlocksSchemaDescription:
    available_blocks = load_workflow_blocks()
    workflow_definition_class = build_workflow_definition_entity(
        available_blocks=available_blocks
    )
    schema = workflow_definition_class.model_json_schema()
    return WorkflowsBlocksSchemaDescription(schema=schema)
