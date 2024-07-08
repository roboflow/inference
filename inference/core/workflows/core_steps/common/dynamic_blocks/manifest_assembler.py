from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from inference.core.workflows.core_steps.common.dynamic_blocks.entities import (
    DynamicInputDefinition,
    DynamicOutputDefinition,
    ManifestDescription,
    SelectorType,
    ValueType,
)
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import (
    WILDCARD_KIND,
    Kind,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)


def assembly_dynamic_block_manifest(
    block_name: str,
    block_type: str,
    manifest_description: ManifestDescription,
    kinds_lookup: Optional[Dict[str, Kind]] = None,
) -> Type[BaseModel]:
    if not kinds_lookup:
        kinds_lookup = {}
    model_name = create_block_type_name(block_name=block_name)
    inputs_definitions = build_inputs(
        inputs=manifest_description.inputs, kinds_lookup=kinds_lookup
    )
    manifest_class = create_model(
        model_name,
        __config__=ConfigDict(extra="allow"),
        name=(str, ...),
        type=(Literal[block_type], ...),
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


def create_block_type_name(block_name: str) -> str:
    block_title = (
        block_name.strip().replace("-", " ").replace("_", " ").title().replace(" ", "")
    )
    return f"DynamicBlock{block_title}"


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
    inputs: Dict[str, DynamicInputDefinition],
    kinds_lookup: Dict[str, Kind],
) -> Dict[str, Tuple[type, Field]]:
    result = {}
    for input_name, input_definition in inputs.items():
        result[input_name] = build_input(
            input_definition=input_definition, kinds_lookup=kinds_lookup
        )
    return result


def build_input(
    input_definition: DynamicInputDefinition,
    kinds_lookup: Dict[str, Kind],
) -> Tuple[type, Field]:
    input_type = build_input_field_type(
        input_definition=input_definition, kinds_lookup=kinds_lookup
    )
    field_metadata = build_input_field_metadata(input_definition=input_definition)
    return input_type, field_metadata


def build_input_field_type(
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
        input_type_union_elements.append(Any)
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
            selector_kind.append(kinds_lookup.get(kind_name, Kind(name=kind_name)))
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
) -> Type[BaseModel]:
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
