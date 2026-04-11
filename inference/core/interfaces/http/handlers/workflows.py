# TODO - for everyone: start migrating other handlers to bring relief to http_api.py
import copy
from typing import Any, Dict, List, Optional, Set, Union

from packaging.specifiers import SpecifierSet

from inference.core.cache.air_gapped import has_cached_model_variant
from inference.core.entities.responses.workflows import (
    DescribeInterfaceResponse,
    ExternalBlockPropertyPrimitiveDefinition,
    ExternalWorkflowsBlockSelectorDefinition,
    UniversalQueryLanguageDescription,
    WorkflowsBlocksDescription,
)
from inference.core.env import ENABLE_BUILDER
from inference.core.workflows.core_steps.common.query_language.introspection.core import (
    prepare_operations_descriptions,
    prepare_operators_descriptions,
)
from inference.core.workflows.errors import WorkflowExecutionEngineVersionError
from inference.core.workflows.execution_engine.core import (
    retrieve_requested_execution_engine_version,
)
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)
from inference.core.workflows.execution_engine.introspection.connections_discovery import (
    discover_blocks_connections,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    compile_dynamic_blocks,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicBlockDefinition,
)
from inference.core.workflows.execution_engine.v1.introspection.inputs_discovery import (
    describe_workflow_inputs,
)
from inference.core.workflows.execution_engine.v1.introspection.outputs_discovery import (
    describe_workflow_outputs,
)
from inference.core.workflows.execution_engine.v1.introspection.types_discovery import (
    discover_kinds_schemas,
    discover_kinds_typing_hints,
)
from inference.core.workflows.prototypes.block import BlockAirGappedInfo


def handle_describe_workflows_blocks_request(
    dynamic_blocks_definitions: Optional[List[DynamicBlockDefinition]] = None,
    requested_execution_engine_version: Optional[str] = None,
    api_key: Optional[str] = None,
    air_gapped: bool = False,
) -> WorkflowsBlocksDescription:
    if dynamic_blocks_definitions is None:
        dynamic_blocks_definitions = []
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=dynamic_blocks_definitions,
        api_key=api_key,
    )
    blocks_description = describe_available_blocks(
        dynamic_blocks=dynamic_blocks,
        execution_engine_version=requested_execution_engine_version,
    )
    blocks_connections = discover_blocks_connections(
        blocks_description=blocks_description,
    )
    kinds_connections = {
        kind_name: [
            ExternalWorkflowsBlockSelectorDefinition(
                manifest_type_identifier=c.manifest_type_identifier,
                property_name=c.property_name,
                property_description=c.property_description,
                compatible_element=c.compatible_element,
                is_list_element=c.is_list_element,
                is_dict_element=c.is_dict_element,
            )
            for c in connections
        ]
        for kind_name, connections in blocks_connections.kinds_connections.items()
    }
    primitives_connections = [
        ExternalBlockPropertyPrimitiveDefinition(
            manifest_type_identifier=primitives_connection.manifest_type_identifier,
            property_name=primitives_connection.property_name,
            property_description=primitives_connection.property_description,
            type_annotation=primitives_connection.type_annotation,
        )
        for primitives_connection in blocks_connections.primitives_connections
    ]
    uql_operations_descriptions = prepare_operations_descriptions()
    uql_operators_descriptions = prepare_operators_descriptions()
    universal_query_language_description = (
        UniversalQueryLanguageDescription.from_internal_entities(
            operations_descriptions=uql_operations_descriptions,
            operators_descriptions=uql_operators_descriptions,
        )
    )
    result = WorkflowsBlocksDescription(
        blocks=blocks_description.blocks,
        declared_kinds=blocks_description.declared_kinds,
        kinds_connections=kinds_connections,
        primitives_connections=primitives_connections,
        universal_query_language_description=universal_query_language_description,
        dynamic_block_definition_schema=DynamicBlockDefinition.schema(),
    )
    if air_gapped and ENABLE_BUILDER:
        result = enrich_with_air_gapped_info(result)
    return result


def enrich_with_air_gapped_info(
    result: WorkflowsBlocksDescription,
) -> WorkflowsBlocksDescription:
    """Post-process block descriptions to include air-gapped availability info.

    Deep-copies block schemas before mutating so the LRU-cached objects are
    not modified.
    """
    enriched_blocks = []
    for block in result.blocks:
        manifest_cls = block.manifest_class
        air_gapped_info = _get_air_gapped_info_for_block(manifest_cls)
        enriched_schema = copy.deepcopy(block.block_schema)
        if "json_schema_extra" not in enriched_schema:
            enriched_schema["json_schema_extra"] = {}
        enriched_schema["json_schema_extra"][
            "air_gapped_info"
        ] = air_gapped_info.to_dict()
        enriched_blocks.append(
            block.model_copy(update={"block_schema": enriched_schema})
        )
    return result.model_copy(update={"blocks": enriched_blocks})


def _get_air_gapped_info_for_block(
    manifest_cls: Any,
) -> BlockAirGappedInfo:
    """Determine air-gapped availability for a single block manifest class.

    Resolution order:

    1. **Cloud-only blocks** — ``get_air_gapped_availability()`` returns
       ``available=False`` (e.g. blocks wrapping OpenAI / Anthropic APIs).
    2. **Foundation-model blocks** — ``get_supported_model_variants()``
       returns a non-None list; availability depends on whether any
       variant has cached weights.
    3. **Default** — the block is available (pure logic, local-network, etc.).

    Compatible task types from ``get_compatible_task_types()`` are always
    attached when present.
    """
    task_types = manifest_cls.get_compatible_task_types() if hasattr(manifest_cls, "get_compatible_task_types") else []

    # 1. Explicit cloud/internet declaration
    if not hasattr(manifest_cls, "get_air_gapped_availability"):
        return BlockAirGappedInfo(available=True, compatible_task_types=task_types)
    availability = manifest_cls.get_air_gapped_availability()
    if not availability.available:
        return BlockAirGappedInfo(
            available=False,
            reason=availability.reason,
            compatible_task_types=task_types,
        )

    # 2. Foundation models with locally-cacheable weights
    model_variants = manifest_cls.get_supported_model_variants() if hasattr(manifest_cls, "get_supported_model_variants") else None
    if model_variants is not None:
        cached = has_cached_model_variant(model_variants)
        # Use the first variant as a representative identifier for the UI.
        # The list is ordered by convention (default/primary variant first),
        # and individual variant selection happens at workflow-execution time.
        representative_id = model_variants[0] if model_variants else None
        return BlockAirGappedInfo(
            available=cached,
            reason=None if cached else "missing_cache_artifacts",
            model_id=representative_id,
            compatible_task_types=task_types,
        )

    # 3. Default: block is available (pure logic blocks, local-network blocks, etc.)
    return BlockAirGappedInfo(
        available=True,
        compatible_task_types=task_types,
    )


def handle_describe_workflows_interface(
    definition: dict,
) -> DescribeInterfaceResponse:
    requested_execution_engine_version = retrieve_requested_execution_engine_version(
        workflow_definition=definition
    )
    if not SpecifierSet(f">=1.0.0,<2.0.0").contains(requested_execution_engine_version):
        raise WorkflowExecutionEngineVersionError(
            public_message="Describing workflow outputs is only supported for Execution Engine v1.",
            context="describing_workflow_outputs",
        )
    inputs = describe_workflow_inputs(definition=definition)
    outputs = describe_workflow_outputs(definition=definition)
    unique_kinds = get_unique_kinds(inputs=inputs, outputs=outputs)
    typing_hints = discover_kinds_typing_hints(kinds_names=unique_kinds)
    kinds_schemas = discover_kinds_schemas(kinds_names=unique_kinds)
    return DescribeInterfaceResponse(
        inputs=inputs,
        outputs=outputs,
        typing_hints=typing_hints,
        kinds_schemas=kinds_schemas,
    )


def get_unique_kinds(
    inputs: Dict[str, List[str]],
    outputs: Dict[str, Union[List[str], Dict[str, List[str]]]],
) -> Set[str]:
    all_kinds = set()
    for input_element_kinds in inputs.values():
        all_kinds.update(input_element_kinds)
    for output_definition in outputs.values():
        if isinstance(output_definition, list):
            all_kinds.update(output_definition)
        if isinstance(output_definition, dict):
            for output_field_kinds in output_definition.values():
                all_kinds.update(output_field_kinds)
    return all_kinds


def filter_out_unwanted_workflow_outputs(
    workflow_results: List[dict],
    excluded_fields: Optional[List[str]],
) -> List[dict]:
    if not excluded_fields:
        return workflow_results
    excluded_fields = set(excluded_fields)
    filtered_results = []
    for result_element in workflow_results:
        filtered_result = {}
        for key, value in result_element.items():
            if key in excluded_fields:
                continue
            filtered_result[key] = value
        filtered_results.append(filtered_result)
    return filtered_results
