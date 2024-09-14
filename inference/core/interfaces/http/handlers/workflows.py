# TODO - for everyone: start migrating other handlers to bring relief to http_api.py
from typing import List, Optional

from packaging.version import Version

from inference.core.entities.responses.workflows import (
    ExternalBlockPropertyPrimitiveDefinition,
    ExternalWorkflowsBlockSelectorDefinition,
    UniversalQueryLanguageDescription,
    WorkflowsBlocksDescription,
    DescribeOutputResponse,
)
from inference.core.workflows.core_steps.common.query_language.introspection.core import (
    prepare_operations_descriptions,
    prepare_operators_descriptions,
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
from inference.core.entities.requests.workflows import DescribeOutputRequest


def handle_describe_workflows_blocks_request(
    dynamic_blocks_definitions: Optional[List[DynamicBlockDefinition]] = None,
    requested_execution_engine_version: Optional[str] = None,
) -> WorkflowsBlocksDescription:
    if dynamic_blocks_definitions is None:
        dynamic_blocks_definitions = []
    dynamic_blocks = compile_dynamic_blocks(
        dynamic_blocks_definitions=dynamic_blocks_definitions,
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
    return WorkflowsBlocksDescription(
        blocks=blocks_description.blocks,
        declared_kinds=blocks_description.declared_kinds,
        kinds_connections=kinds_connections,
        primitives_connections=primitives_connections,
        universal_query_language_description=universal_query_language_description,
        dynamic_block_definition_schema=DynamicBlockDefinition.schema(),
    )


def handle_describe_workflows_output(
    workflow_request: DescribeOutputRequest,
    workflow_specification: dict,
) -> DescribeOutputResponse:
    # Map each block to it's output properties
    block_output_map = {}
    blocks_description = describe_available_blocks(
        dynamic_blocks=workflow_request.dynamic_blocks_definitions,
        execution_engine_version=workflow_request.execution_engine_version,
    )
    for block in blocks_description.blocks:
        key = block.manifest_type_identifier
        output_property_kinds = get_output_property_kinds(block.outputs_manifest)
        block_output_map[key] = output_property_kinds
        if block.manifest_type_identifier_aliases:
            for alias in block.manifest_type_identifier_aliases:
                block_output_map[alias] = output_property_kinds

    workflow_steps = workflow_specification["steps"]
    workflow_outputs = workflow_specification["outputs"]

    # For each workflow output, return it's output properties
    workflow_response_definition = {}
    for output in workflow_outputs:
        selector = output["selector"]
        step_name = extract_step_name(selector)
        step = next((s for s in workflow_steps if s.get("name") == step_name), None)
        if step is None:
            continue
        output_properties = block_output_map[step["type"]]
        workflow_response_definition[selector] = output_properties

    return workflow_response_definition


def extract_step_name(selector: str) -> str:
    parts = selector.split(".")
    if len(parts) >= 2 and parts[0] == "$steps":
        return parts[1]
    return ""


def get_output_property_kinds(outputs_manifest):
    ret = {}
    if outputs_manifest:
        for output in outputs_manifest:
            ret[output.name] = [kind.name for kind in output.kind]
    return ret
