import itertools
from collections import OrderedDict, defaultdict
from dataclasses import replace
from typing import Dict, Optional, Set, Type

from inference.core.workflows.execution_engine.entities.types import (
    KIND_KEY,
    REFERENCE_KEY,
    SELECTED_ELEMENT_KEY,
    SELECTOR_POINTS_TO_BATCH_KEY,
    Kind,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    BlockManifestMetadata,
    PrimitiveTypeDefinition,
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest

EXCLUDED_PROPERTIES = {"type"}

TYPE_MAPPING = {
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "string": "str",
    "null": "None",
}

SET_TYPE_NAME = "Set"
LIST_TYPE_NAME = "List"
UNION_TYPE_NAME = "Union"
NONE_TYPE_NAME = "None"
OPTIONAL_TYPE_NAME = "Optional"
ANY_TYPE_NAME = "Any"
ANY_DICT_TYPE_NAME = "Dict[Any, Any]"
DICT_TYPE_NAME_PREFIX = "Dict"
STR_TYPE_NAME = "str"

ITEMS_KEY = "items"
UNIQUE_ITEMS_KEY = "uniqueItems"
TUPLE_ITEMS_KEY = "prefixItems"
TYPE_KEY = "type"
REF_KEY = "$ref"
ADDITIONAL_PROPERTIES_KEY = "additionalProperties"
PROPERTIES_KEY = "properties"
DESCRIPTION_KEY = "description"
ALL_OF_KEY = "allOf"
ANY_OF_KEY = "anyOf"
ONE_OF_KEY = "oneOf"
OBJECT_TYPE = "object"


def parse_block_manifest(
    manifest_type: Type[WorkflowBlockManifest],
) -> BlockManifestMetadata:
    schema = manifest_type.model_json_schema()
    inputs_dimensionality_offsets = manifest_type.get_input_dimensionality_offsets()
    dimensionality_reference_property = (
        manifest_type.get_dimensionality_reference_property()
    )
    inputs_accepting_batches = set(manifest_type.get_parameters_accepting_batches())
    inputs_accepting_batches_and_scalars = set(
        manifest_type.get_parameters_accepting_batches_and_scalars()
    )
    return parse_block_manifest_schema(
        schema=schema,
        inputs_dimensionality_offsets=inputs_dimensionality_offsets,
        dimensionality_reference_property=dimensionality_reference_property,
        inputs_accepting_batches=inputs_accepting_batches,
        inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
    )


def parse_block_manifest_schema(
    schema: dict,
    inputs_dimensionality_offsets: Dict[str, int],
    dimensionality_reference_property: Optional[str],
    inputs_accepting_batches: Set[str],
    inputs_accepting_batches_and_scalars: Set[str],
) -> BlockManifestMetadata:
    primitive_types = retrieve_primitives_from_schema(
        schema=schema,
    )
    selectors = retrieve_selectors_from_schema(
        schema=schema,
        inputs_dimensionality_offsets=inputs_dimensionality_offsets,
        dimensionality_reference_property=dimensionality_reference_property,
        inputs_accepting_batches=inputs_accepting_batches,
        inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
    )
    return BlockManifestMetadata(
        primitive_types=primitive_types,
        selectors=selectors,
    )


def retrieve_primitives_from_schema(schema: dict) -> Dict[str, PrimitiveTypeDefinition]:
    result = []
    for property_name, property_definition in schema[PROPERTIES_KEY].items():
        if property_name in EXCLUDED_PROPERTIES:
            continue
        property_description = property_definition.get(DESCRIPTION_KEY, "not available")
        primitive_metadata = retrieve_primitive_type_from_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=property_definition,
        )
        if primitive_metadata is not None:
            result.append(primitive_metadata)
    return OrderedDict((r.property_name, r) for r in result)


def retrieve_primitive_type_from_property(
    property_name: str,
    property_description: str,
    property_definition: dict,
) -> Optional[PrimitiveTypeDefinition]:
    if REFERENCE_KEY in property_definition:
        return None
    if ITEMS_KEY in property_definition:
        result = retrieve_primitive_type_from_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=property_definition[ITEMS_KEY],
        )
        if result is None:
            return None
        high_level_type = (
            SET_TYPE_NAME
            if property_definition.get(UNIQUE_ITEMS_KEY, False) is True
            else LIST_TYPE_NAME
        )
        return replace(
            result, type_annotation=f"{high_level_type}[{result.type_annotation}]"
        )
    if TUPLE_ITEMS_KEY in property_definition:
        nested_annotations = [
            retrieve_primitive_type_from_property(
                property_name=property_name,
                property_description=property_description,
                property_definition=nested_definition,
            )
            for nested_definition in property_definition[TUPLE_ITEMS_KEY]
        ]
        inner_types = ", ".join(a.type_annotation for a in nested_annotations)
        return PrimitiveTypeDefinition(
            property_name=property_name,
            property_description=property_description,
            type_annotation=f"Tuple[{inner_types}]",
        )
    if property_definition.get(TYPE_KEY) in TYPE_MAPPING:
        type_name = TYPE_MAPPING[property_definition[TYPE_KEY]]
        return PrimitiveTypeDefinition(
            property_name=property_name,
            property_description=property_description,
            type_annotation=type_name,
        )
    if REF_KEY in property_definition:
        return PrimitiveTypeDefinition(
            property_name=property_name,
            property_description=property_description,
            type_annotation=property_definition[REF_KEY].split("/")[-1],
        )
    if property_defines_union(property_definition=property_definition):
        return retrieve_primitive_type_from_union_property(
            property_name=property_name,
            property_description=property_description,
            union_definition=property_definition,
        )
    if property_definition.get(TYPE_KEY) == "object":
        return retrieve_primitive_type_from_dict_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=property_definition,
        )
    return PrimitiveTypeDefinition(
        property_name=property_name,
        property_description=property_description,
        type_annotation=ANY_TYPE_NAME,
    )


def retrieve_primitive_type_from_union_property(
    property_name: str,
    property_description: str,
    union_definition: dict,
) -> Optional[PrimitiveTypeDefinition]:
    union_types = (
        union_definition.get(ANY_OF_KEY, [])
        + union_definition.get(ONE_OF_KEY, [])
        + union_definition.get(ALL_OF_KEY, [])
    )
    primitive_union_types = [e for e in union_types if REFERENCE_KEY not in e]
    union_types_metadata = []
    for union_type in primitive_union_types:
        union_type_metadata = retrieve_primitive_type_from_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=union_type,
        )
        if union_type_metadata is None:
            continue
        union_types_metadata.append(union_type_metadata)
    if not union_types_metadata:
        return None
    type_names = {m.type_annotation for m in union_types_metadata}
    if NONE_TYPE_NAME in type_names:
        high_level_type = OPTIONAL_TYPE_NAME
        type_names.remove(NONE_TYPE_NAME)
    else:
        high_level_type = UNION_TYPE_NAME
    final_type_name = ", ".join(list(sorted(type_names)))
    if len(type_names) == 0:
        return None
    if len(type_names) > 1:
        final_type_name = f"{high_level_type}[{final_type_name}]"
    return PrimitiveTypeDefinition(
        property_name=property_name,
        property_description=property_description,
        type_annotation=final_type_name,
    )


def retrieve_primitive_type_from_dict_property(
    property_name: str,
    property_description: str,
    property_definition: dict,
) -> Optional[PrimitiveTypeDefinition]:
    if ADDITIONAL_PROPERTIES_KEY in property_definition:
        dict_value_type = retrieve_primitive_type_from_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=property_definition[ADDITIONAL_PROPERTIES_KEY],
        )
        if dict_value_type is None:
            return None
        return PrimitiveTypeDefinition(
            property_name=property_name,
            property_description=property_description,
            type_annotation=f"{DICT_TYPE_NAME_PREFIX}[{STR_TYPE_NAME}, {dict_value_type.type_annotation}]",
        )
    return PrimitiveTypeDefinition(
        property_name=property_name,
        property_description=property_description,
        type_annotation=ANY_DICT_TYPE_NAME,
    )


def retrieve_selectors_from_schema(
    schema: dict,
    inputs_dimensionality_offsets: Dict[str, int],
    dimensionality_reference_property: Optional[str],
    inputs_accepting_batches: Set[str],
    inputs_accepting_batches_and_scalars: Set[str],
) -> Dict[str, SelectorDefinition]:
    result = []
    for property_name, property_definition in schema[PROPERTIES_KEY].items():
        if property_name in EXCLUDED_PROPERTIES:
            continue
        property_dimensionality_offset = inputs_dimensionality_offsets.get(
            property_name, 0
        )
        is_dimensionality_reference_property = (
            property_name == dimensionality_reference_property
        )
        property_description = property_definition.get(DESCRIPTION_KEY, "not available")
        if ITEMS_KEY in property_definition:
            selector = retrieve_selectors_from_simple_property(
                property_name=property_name,
                property_description=property_description,
                property_definition=property_definition[ITEMS_KEY],
                property_dimensionality_offset=property_dimensionality_offset,
                is_dimensionality_reference_property=is_dimensionality_reference_property,
                is_list_element=True,
                inputs_accepting_batches=inputs_accepting_batches,
                inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
            )
        elif (
            property_definition.get(TYPE_KEY) == OBJECT_TYPE
            and ADDITIONAL_PROPERTIES_KEY in property_definition
        ):
            selector = retrieve_selectors_from_simple_property(
                property_name=property_name,
                property_description=property_description,
                property_definition=property_definition[ADDITIONAL_PROPERTIES_KEY],
                property_dimensionality_offset=property_dimensionality_offset,
                is_dimensionality_reference_property=is_dimensionality_reference_property,
                is_dict_element=True,
                inputs_accepting_batches=inputs_accepting_batches,
                inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
            )
        else:
            selector = retrieve_selectors_from_simple_property(
                property_name=property_name,
                property_description=property_description,
                property_definition=property_definition,
                property_dimensionality_offset=property_dimensionality_offset,
                is_dimensionality_reference_property=is_dimensionality_reference_property,
                inputs_accepting_batches=inputs_accepting_batches,
                inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
            )
        if selector is not None:
            result.append(selector)
    return OrderedDict((r.property_name, r) for r in result)


def retrieve_selectors_from_simple_property(
    property_name: str,
    property_description: str,
    property_definition: dict,
    property_dimensionality_offset: int,
    is_dimensionality_reference_property: bool,
    inputs_accepting_batches: Set[str],
    inputs_accepting_batches_and_scalars: Set[str],
    is_list_element: bool = False,
    is_dict_element: bool = False,
) -> Optional[SelectorDefinition]:
    if REFERENCE_KEY in property_definition:
        declared_points_to_batch = property_definition.get(
            SELECTOR_POINTS_TO_BATCH_KEY, False
        )
        if declared_points_to_batch == "dynamic":
            if property_name in inputs_accepting_batches_and_scalars:
                points_to_batch = {True, False}
            else:
                points_to_batch = {property_name in inputs_accepting_batches}
        else:
            points_to_batch = {declared_points_to_batch}
        allowed_references = [
            ReferenceDefinition(
                selected_element=property_definition[SELECTED_ELEMENT_KEY],
                kind=[
                    Kind.model_validate(k)
                    for k in property_definition.get(KIND_KEY, [])
                ],
                points_to_batch=points_to_batch,
            )
        ]
        return SelectorDefinition(
            property_name=property_name,
            property_description=property_description,
            allowed_references=allowed_references,
            is_list_element=is_list_element,
            is_dict_element=is_dict_element,
            dimensionality_offset=property_dimensionality_offset,
            is_dimensionality_reference_property=is_dimensionality_reference_property,
        )
    if ITEMS_KEY in property_definition:
        if is_list_element or is_dict_element:
            # ignoring nested references above first level of depth
            return None
        return retrieve_selectors_from_simple_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=property_definition[ITEMS_KEY],
            property_dimensionality_offset=property_dimensionality_offset,
            is_dimensionality_reference_property=is_dimensionality_reference_property,
            inputs_accepting_batches=inputs_accepting_batches,
            inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
            is_list_element=True,
        )
    if property_defines_union(property_definition=property_definition):
        return retrieve_selectors_from_union_definition(
            property_name=property_name,
            property_description=property_description,
            union_definition=property_definition,
            is_list_element=is_list_element,
            is_dict_element=is_dict_element,
            property_dimensionality_offset=property_dimensionality_offset,
            is_dimensionality_reference_property=is_dimensionality_reference_property,
            inputs_accepting_batches=inputs_accepting_batches,
            inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
        )
    return None


def property_defines_union(property_definition: dict) -> bool:
    return (
        ANY_OF_KEY in property_definition
        or ONE_OF_KEY in property_definition
        or ALL_OF_KEY in property_definition
    )


def retrieve_selectors_from_union_definition(
    property_name: str,
    property_description: str,
    union_definition: dict,
    is_list_element: bool,
    is_dict_element: bool,
    property_dimensionality_offset: int,
    is_dimensionality_reference_property: bool,
    inputs_accepting_batches: Set[str],
    inputs_accepting_batches_and_scalars: Set[str],
) -> Optional[SelectorDefinition]:
    union_types = (
        union_definition.get(ANY_OF_KEY, [])
        + union_definition.get(ONE_OF_KEY, [])
        + union_definition.get(ALL_OF_KEY, [])
    )
    results = []
    for type_definition in union_types:
        result = retrieve_selectors_from_simple_property(
            property_name=property_name,
            property_description=property_description,
            property_definition=type_definition,
            property_dimensionality_offset=property_dimensionality_offset,
            is_dimensionality_reference_property=is_dimensionality_reference_property,
            inputs_accepting_batches=inputs_accepting_batches,
            inputs_accepting_batches_and_scalars=inputs_accepting_batches_and_scalars,
            is_list_element=is_list_element,
        )
        if result is None:
            continue
        results.append(result)
    results_references = list(
        itertools.chain.from_iterable(r.allowed_references for r in results)
    )
    results_references_kind_by_selected_element = defaultdict(set)
    results_references_batch_pointing_by_selected_element = defaultdict(set)
    for reference in results_references:
        results_references_kind_by_selected_element[reference.selected_element].update(
            reference.kind
        )
        results_references_batch_pointing_by_selected_element[
            reference.selected_element
        ].update(reference.points_to_batch)
    merged_references = []
    for (
        reference_selected_element,
        kind,
    ) in results_references_kind_by_selected_element.items():
        merged_references.append(
            ReferenceDefinition(
                selected_element=reference_selected_element,
                kind=list(kind),
                points_to_batch=results_references_batch_pointing_by_selected_element[
                    reference_selected_element
                ],
            )
        )
    if not merged_references:
        return None
    return SelectorDefinition(
        property_name=property_name,
        property_description=property_description,
        allowed_references=merged_references,
        is_list_element=is_list_element,
        is_dict_element=is_dict_element,
        dimensionality_offset=property_dimensionality_offset,
        is_dimensionality_reference_property=is_dimensionality_reference_property,
    )
