from collections import defaultdict
from typing import Dict, Generator, List, Set, Tuple, Type

from inference.core.workflows.execution_engine.entities.types import (
    ANY_DATA_AS_SELECTED_ELEMENT,
    BATCH_AS_SELECTED_ELEMENT,
    STEP_AS_SELECTED_ELEMENT,
    STEP_OUTPUT_AS_SELECTED_ELEMENT,
    WILDCARD_KIND,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockDescription,
    BlockManifestMetadata,
    BlockPropertyPrimitiveDefinition,
    BlockPropertySelectorDefinition,
    BlocksConnections,
    BlocksDescription,
    DiscoveredConnections,
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.core.workflows.execution_engine.introspection.schema_parser import (
    parse_block_manifest,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


def discover_blocks_connections(
    blocks_description: BlocksDescription,
) -> DiscoveredConnections:
    block_type2manifest_type = {
        block.block_class: block.manifest_class for block in blocks_description.blocks
    }
    all_schemas = parse_all_schemas(blocks_description=blocks_description)
    output_kind2schemas = get_all_outputs_kind_major(
        blocks_description=blocks_description
    )
    detailed_input_kind2schemas = get_all_inputs_kind_major(
        blocks_description=blocks_description,
        all_schemas=all_schemas,
    )
    compatible_elements = {
        STEP_OUTPUT_AS_SELECTED_ELEMENT,
        BATCH_AS_SELECTED_ELEMENT,
        ANY_DATA_AS_SELECTED_ELEMENT,
    }
    coarse_input_kind2schemas = convert_kinds_mapping_to_block_wise_format(
        detailed_input_kind2schemas=detailed_input_kind2schemas,
        compatible_elements=compatible_elements,
    )
    input_property_wise_connections = {}
    output_property_wise_connections = {}
    for block_type in all_schemas.keys():
        input_property_wise_connections[block_type] = discover_block_input_connections(
            starting_block=block_type,
            all_schemas=all_schemas,
            output_kind2schemas=output_kind2schemas,
            compatible_elements=compatible_elements,
        )
        manifest_type = block_type2manifest_type[block_type]
        output_property_wise_connections[block_type] = (
            discover_block_output_connections(
                manifest_type=manifest_type,
                input_kind2schemas=coarse_input_kind2schemas,
            )
        )
    input_block_wise_connections = (
        convert_property_connections_to_block_wise_connections(
            property_wise_connections=input_property_wise_connections,
        )
    )
    output_block_wise_connections = (
        convert_property_connections_to_block_wise_connections(
            property_wise_connections=output_property_wise_connections,
        )
    )
    input_connections = BlocksConnections(
        property_wise=input_property_wise_connections,
        block_wise=input_block_wise_connections,
    )
    output_connections = BlocksConnections(
        property_wise=output_property_wise_connections,
        block_wise=output_block_wise_connections,
    )
    primitives_connections = build_primitives_connections(
        all_schemas=all_schemas,
        blocks_description=blocks_description,
    )
    return DiscoveredConnections(
        input_connections=input_connections,
        output_connections=output_connections,
        kinds_connections=detailed_input_kind2schemas,
        primitives_connections=primitives_connections,
    )


def parse_all_schemas(
    blocks_description: BlocksDescription,
) -> Dict[Type[WorkflowBlock], BlockManifestMetadata]:
    return {
        block.block_class: parse_block_manifest(manifest_type=block.manifest_class)
        for block in blocks_description.blocks
    }


def get_all_outputs_kind_major(
    blocks_description: BlocksDescription,
) -> Dict[str, Set[Type[WorkflowBlock]]]:
    kind_major_step_outputs = defaultdict(set)
    for block in blocks_description.blocks:
        kind_major_step_outputs[WILDCARD_KIND.name].add(block.block_class)
        for output in block.outputs_manifest:
            for kind in output.kind:
                kind_major_step_outputs[kind.name].add(block.block_class)
    return kind_major_step_outputs


def get_all_inputs_kind_major(
    blocks_description: BlocksDescription,
    all_schemas: Dict[Type[WorkflowBlock], BlockManifestMetadata],
) -> Dict[str, Set[BlockPropertySelectorDefinition]]:
    kind_major_step_inputs = defaultdict(set)
    for (
        block_description,
        selector,
        allowed_reference,
    ) in enlist_blocks_selectors_and_references(
        blocks_description=blocks_description,
        all_schemas=all_schemas,
    ):
        if allowed_reference.selected_element == STEP_AS_SELECTED_ELEMENT:
            continue
        for single_kind in allowed_reference.kind:
            kind_major_step_inputs[single_kind.name].add(
                BlockPropertySelectorDefinition(
                    block_type=block_description.block_class,
                    manifest_type_identifier=block_description.manifest_type_identifier,
                    property_name=selector.property_name,
                    property_description=selector.property_description,
                    compatible_element=allowed_reference.selected_element,
                    is_list_element=selector.is_list_element,
                    is_dict_element=selector.is_dict_element,
                )
            )
        kind_major_step_inputs[WILDCARD_KIND.name].add(
            BlockPropertySelectorDefinition(
                block_type=block_description.block_class,
                manifest_type_identifier=block_description.manifest_type_identifier,
                property_name=selector.property_name,
                property_description=selector.property_description,
                compatible_element=allowed_reference.selected_element,
                is_list_element=selector.is_list_element,
                is_dict_element=selector.is_dict_element,
            )
        )
    return kind_major_step_inputs


def enlist_blocks_selectors_and_references(
    blocks_description: BlocksDescription,
    all_schemas: Dict[Type[WorkflowBlock], BlockManifestMetadata],
) -> Generator[
    Tuple[BlockDescription, SelectorDefinition, ReferenceDefinition], None, None
]:
    for block_description in blocks_description.blocks:
        for selector in all_schemas[block_description.block_class].selectors.values():
            for allowed_reference in selector.allowed_references:
                yield block_description, selector, allowed_reference


def discover_block_input_connections(
    starting_block: Type[WorkflowBlock],
    all_schemas: Dict[Type[WorkflowBlock], BlockManifestMetadata],
    output_kind2schemas: Dict[str, Set[Type[WorkflowBlock]]],
    compatible_elements: Set[str],
) -> Dict[str, Set[Type[WorkflowBlock]]]:
    result = {}
    for selector in all_schemas[starting_block].selectors.values():
        blocks_matching_property = set()
        for allowed_reference in selector.allowed_references:
            if allowed_reference.selected_element not in compatible_elements:
                continue
            for single_kind in allowed_reference.kind:
                blocks_matching_property.update(
                    output_kind2schemas.get(single_kind.name, set())
                )
        result[selector.property_name] = blocks_matching_property
    return result


def discover_block_output_connections(
    manifest_type: Type[WorkflowBlockManifest],
    input_kind2schemas: Dict[str, Set[Type[WorkflowBlock]]],
) -> Dict[str, Set[Type[WorkflowBlock]]]:
    result = {}
    for output in manifest_type.describe_outputs():
        compatible_blocks = set()
        for single_kind in output.kind:
            compatible_blocks.update(input_kind2schemas[single_kind.name])
        result[output.name] = compatible_blocks
    return result


def convert_property_connections_to_block_wise_connections(
    property_wise_connections: Dict[
        Type[WorkflowBlock], Dict[str, Set[Type[WorkflowBlock]]]
    ],
) -> Dict[Type[WorkflowBlock], Set[Type[WorkflowBlock]]]:
    result = {}
    for block_type, properties_connections in property_wise_connections.items():
        block_connections = set()
        for property_connections in properties_connections.values():
            block_connections.update(property_connections)
        result[block_type] = block_connections
    return result


def convert_kinds_mapping_to_block_wise_format(
    detailed_input_kind2schemas: Dict[str, Set[BlockPropertySelectorDefinition]],
    compatible_elements: Set[str],
) -> Dict[str, Set[Type[WorkflowBlock]]]:
    result = defaultdict(set)
    for kind_name, block_properties_definitions in detailed_input_kind2schemas.items():
        for definition in block_properties_definitions:
            if definition.compatible_element not in compatible_elements:
                continue
            result[kind_name].add(definition.block_type)
    return result


def build_primitives_connections(
    all_schemas: Dict[Type[WorkflowBlock], BlockManifestMetadata],
    blocks_description: BlocksDescription,
) -> List[BlockPropertyPrimitiveDefinition]:
    result = []
    for block in blocks_description.blocks:
        for property_definition in all_schemas[
            block.block_class
        ].primitive_types.values():
            definition = BlockPropertyPrimitiveDefinition(
                block_type=block.block_class,
                manifest_type_identifier=block.manifest_type_identifier,
                property_name=property_definition.property_name,
                property_description=property_definition.property_description,
                type_annotation=property_definition.type_annotation,
            )
            result.append(definition)
    return result
