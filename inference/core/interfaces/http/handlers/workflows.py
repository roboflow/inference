# TODO - for everyone: start migrating other handlers to bring relief to http_api.py
from typing import List, Optional

from packaging.specifiers import SpecifierSet

from inference.core.entities.responses.workflows import (
    DescribeOutputResponse,
    ExternalBlockPropertyPrimitiveDefinition,
    ExternalWorkflowsBlockSelectorDefinition,
    UniversalQueryLanguageDescription,
    WorkflowsBlocksDescription,
)
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
from inference.core.workflows.execution_engine.v1.introspection.outputs_discovery import (
    describe_workflows_output,
)


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
    specification: dict,
) -> DescribeOutputResponse:
    requested_execution_engine_version = retrieve_requested_execution_engine_version(
        workflow_definition=specification
    )
    if not SpecifierSet(f">=1.0.0,<2.0.0").contains(requested_execution_engine_version):
        raise WorkflowExecutionEngineVersionError(
            public_message="Describing workflow outputs is only supported for Execution Engine v1.",
            context="describing_workflow_outputs",
        )
    outputs = describe_workflows_output(definition=specification)
    return DescribeOutputResponse(outputs=outputs)
