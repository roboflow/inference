from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, create_model

from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    WILDCARD_KIND,
    Kind,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.dynamic_blocs.block_scaffolding import (
    assembly_custom_python_block,
)
from inference.core.workflows.execution_engine.dynamic_blocs.entities import (
    BLOCK_SOURCE,
    DynamicBlockDefinition,
    DynamicInputDefinition,
    DynamicOutputDefinition,
    ManifestDescription,
    SelectorType,
    ValueType,
)
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_all_defined_kinds,
)
from inference.core.workflows.execution_engine.introspection.utils import (
    get_full_type_name,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def compile_dynamic_blocks(
    dynamic_blocks_definitions: List[dict],
) -> List[BlockSpecification]:
    all_defined_kinds = load_all_defined_kinds()
    kinds_lookup = {kind.name: kind for kind in all_defined_kinds}
    dynamic_blocks = [
        DynamicBlockDefinition.model_validate(dynamic_block)
        for dynamic_block in dynamic_blocks_definitions
    ]
    compiled_blocks = []
    for dynamic_block in dynamic_blocks:
        block_specification = create_dynamic_block_specification(
            dynamic_block_definition=dynamic_block,
            kinds_lookup=kinds_lookup,
        )
        compiled_blocks.append(block_specification)
    return compiled_blocks


def create_dynamic_block_specification(
    dynamic_block_definition: DynamicBlockDefinition,
    kinds_lookup: Dict[str, Kind],
) -> BlockSpecification:
    unique_identifier = str(uuid4())
    block_manifest = assembly_dynamic_block_manifest(
        unique_identifier=unique_identifier,
        manifest_description=dynamic_block_definition.manifest,
        kinds_lookup=kinds_lookup,
    )
    block_class = assembly_custom_python_block(
        unique_identifier=unique_identifier,
        manifest=block_manifest,
        python_code=dynamic_block_definition.code,
    )
    return BlockSpecification(
        block_source=BLOCK_SOURCE,
        identifier=get_full_type_name(selected_type=block_class),
        block_class=block_class,
        manifest_class=block_manifest,
    )


def assembly_dynamic_block_manifest(
    unique_identifier: str,
    manifest_description: ManifestDescription,
    kinds_lookup: Dict[str, Kind],
) -> Type[WorkflowBlockManifest]:
    inputs_definitions = build_inputs(
        block_type=manifest_description.block_type,
        inputs=manifest_description.inputs,
        kinds_lookup=kinds_lookup,
    )
    manifest_class = create_model(
        f"DynamicBlockManifest[{unique_identifier}]",
        __config__=ConfigDict(extra="allow"),
        name=(str, ...),
        type=(Literal[manifest_description.block_type], ...),
        **inputs_definitions,
    )
    outputs_definitions = build_outputs_definitions(
        outputs=manifest_description.outputs,
        kinds_lookup=kinds_lookup,
    )
    return assembly_manifest_class_methods(
        manifest_class=manifest_class,
        outputs_definitions=outputs_definitions,
        manifest_description=manifest_description,
    )


PYTHON_TYPES_MAPPING = {
    ValueType.ANY: Any,
    ValueType.INTEGER: int,
    ValueType.FLOAT: float,
    ValueType.BOOLEAN: bool,
    ValueType.DICT: dict,
    ValueType.LIST: list,
    ValueType.STRING: str,
}


def build_inputs(
    block_type: str,
    inputs: Dict[str, DynamicInputDefinition],
    kinds_lookup: Dict[str, Kind],
) -> Dict[str, Tuple[type, Field]]:
    result = {}
    for input_name, input_definition in inputs.items():
        result[input_name] = build_input(
            block_type=block_type,
            input_name=input_name,
            input_definition=input_definition,
            kinds_lookup=kinds_lookup,
        )
    return result


def build_input(
    block_type: str,
    input_name: str,
    input_definition: DynamicInputDefinition,
    kinds_lookup: Dict[str, Kind],
) -> Tuple[type, Field]:
    input_type = build_input_field_type(
        block_type=block_type,
        input_name=input_name,
        input_definition=input_definition,
        kinds_lookup=kinds_lookup,
    )
    field_metadata = build_input_field_metadata(input_definition=input_definition)
    return input_type, field_metadata


def build_input_field_type(
    block_type: str,
    input_name: str,
    input_definition: DynamicInputDefinition,
    kinds_lookup: Dict[str, Kind],
) -> type:
    input_type_union_elements = collect_python_types_for_selectors(
        input_definition=input_definition,
        kinds_lookup=kinds_lookup,
    )
    input_type_union_elements += collect_python_types_for_values(
        input_definition=input_definition
    )
    if not input_type_union_elements:
        raise ValueError(
            f"There is no definition of input type found for property: {input_name} of "
            f"dynamic block {block_type}."
        )
    if len(input_type_union_elements) > 1:
        input_type = Union[tuple(input_type_union_elements)]
    else:
        input_type = input_type_union_elements[0]
    if input_definition.is_optional:
        input_type = Optional[input_type]
    return input_type


def collect_python_types_for_selectors(
    input_definition: DynamicInputDefinition,
    kinds_lookup: Dict[str, Kind],
) -> List[type]:
    result = []
    for selector_type in input_definition.selector_types:
        selector_kind_names = input_definition.selector_data_kind.get(
            selector_type, ["*"]
        )
        selector_kind = []
        for kind_name in selector_kind_names:
            if kind_name not in kinds_lookup:
                raise ValueError(
                    f"Could not find kind with name {kind_name} within kinds "
                    f"recognised by Execution Engine: {list(kinds_lookup.keys())}."
                )
            selector_kind.append(kinds_lookup[kind_name])
        if selector_type is SelectorType.INPUT_IMAGE:
            result.append(WorkflowImageSelector)
        elif selector_type is SelectorType.INPUT_PARAMETER:
            result.append(WorkflowParameterSelector(kind=selector_kind))
        else:
            result.append(StepOutputSelector(kind=selector_kind))
    return result


def collect_python_types_for_values(
    input_definition: DynamicInputDefinition,
) -> List[type]:
    result = []
    for value_type_name in input_definition.value_types:
        value_type = PYTHON_TYPES_MAPPING[value_type_name]
        result.append(value_type)
    return result


def build_input_field_metadata(input_definition: DynamicInputDefinition) -> Field:
    if not input_definition.has_default_value:
        return Field()
    default_value = input_definition.default_value
    field_metadata_params = {}
    if default_holds_compound_object(default_value=default_value):
        field_metadata_params["default_factory"] = lambda: default_value
    else:
        field_metadata_params["default"] = default_value
    field_metadata = Field(**field_metadata_params)
    return field_metadata


def default_holds_compound_object(default_value: Any) -> bool:
    return (
        isinstance(default_value, list)
        or isinstance(default_value, dict)
        or isinstance(default_value, set)
    )


def build_outputs_definitions(
    outputs: Dict[str, DynamicOutputDefinition],
    kinds_lookup: Dict[str, Kind],
) -> List[OutputDefinition]:
    result = []
    for name, definition in outputs.items():
        if not definition.kind:
            result.append(OutputDefinition(name=name, kind=[WILDCARD_KIND]))
        else:
            actual_kinds = [
                kinds_lookup.get(kind_name, Kind(name=kind_name))
                for kind_name in definition.kind
            ]
            result.append(OutputDefinition(name=name, kind=actual_kinds))
    return result


def collect_input_dimensionality_offsets(
    inputs: Dict[str, DynamicInputDefinition],
) -> Dict[str, int]:
    result = {}
    for name, definition in inputs.items():
        if definition.dimensionality_offset != 0:
            result[name] = definition.dimensionality_offset
    return result


def assembly_manifest_class_methods(
    manifest_class: Type[BaseModel],
    outputs_definitions: List[OutputDefinition],
    manifest_description: ManifestDescription,
) -> Type[WorkflowBlockManifest]:
    describe_outputs = lambda cls: outputs_definitions
    setattr(manifest_class, "describe_outputs", classmethod(describe_outputs))
    setattr(manifest_class, "get_actual_outputs", describe_outputs)
    accepts_batch_input = lambda cls: manifest_description.accepts_batch_input
    setattr(manifest_class, "accepts_batch_input", classmethod(accepts_batch_input))
    input_dimensionality_offsets = collect_input_dimensionality_offsets(
        inputs=manifest_description.inputs
    )
    get_input_dimensionality_offsets = lambda cls: input_dimensionality_offsets
    setattr(
        manifest_class,
        "get_input_dimensionality_offsets",
        classmethod(get_input_dimensionality_offsets),
    )
    dimensionality_reference = pick_dimensionality_reference_property(
        inputs=manifest_description.inputs
    )
    get_dimensionality_reference_property = lambda cls: dimensionality_reference
    setattr(
        manifest_class,
        "get_dimensionality_reference_property",
        classmethod(get_dimensionality_reference_property),
    )
    get_output_dimensionality_offset = (
        lambda cls: manifest_description.output_dimensionality_offset
    )
    setattr(
        manifest_class,
        "get_output_dimensionality_offset",
        classmethod(get_output_dimensionality_offset),
    )
    accepts_empty_values = lambda cls: manifest_description.accepts_empty_values
    setattr(manifest_class, "accepts_empty_values", classmethod(accepts_empty_values))
    return manifest_class


def pick_dimensionality_reference_property(
    inputs: Dict[str, DynamicInputDefinition]
) -> Optional[str]:
    references = []
    for name, definition in inputs.items():
        if definition.is_dimensionality_reference:
            references.append(name)
    if not references:
        return None
    if len(references) == 1:
        return references[0]
    raise ValueError("Not expected to have multiple dimensionality references")
