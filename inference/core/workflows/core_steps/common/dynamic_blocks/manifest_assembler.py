from typing import Literal, Dict, Tuple, Any, Union, Optional, List

from pydantic import create_model, ConfigDict, Field

from inference.core.workflows.core_steps.common.dynamic_blocks.entities import DynamicBlockManifest, \
    DynamicInputDefinition, \
    SelectorType, ValueType, DynamicOutputDefinition
from inference.core.workflows.entities.base import OutputDefinition
from inference.core.workflows.entities.types import Kind, WorkflowImageSelector, WorkflowParameterSelector, \
    StepOutputSelector, WILDCARD_KIND
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def assembly_dynamic_block_manifest(
    block_name: str,
    block_type: str,
    dynamic_manifest: DynamicBlockManifest,
    kinds_lookup: Dict[str, Kind]
) -> WorkflowBlockManifest:
    model_name = f"DynamicBlock{block_name}Type{block_type}"
    inputs_definitions = build_inputs(inputs=dynamic_manifest.inputs, kinds_lookup=kinds_lookup)
    model = create_model(
        model_name,
        __config__=ConfigDict(extra="allow"),
        name=(str, ...),
        type=(Literal[block_type], ...),
        **inputs_definitions,
    )
    outputs_definitions = build_outputs_definitions(
        outputs=dynamic_manifest.outputs,
        kinds_lookup=kinds_lookup,
    )
    describe_outputs = lambda cls: outputs_definitions
    setattr(model, "describe_outputs", classmethod(describe_outputs))
    setattr(model, "get_actual_outputs", describe_outputs)
    accepts_batch_input = lambda cls: dynamic_manifest.accepts_batch_input
    setattr(model, "accepts_batch_input", classmethod(accepts_batch_input))
    input_dimensionality_offsets = collect_input_dimensionality_offsets(inputs=dynamic_manifest.inputs)
    get_input_dimensionality_offsets = lambda cls: input_dimensionality_offsets
    setattr(model, "get_input_dimensionality_offsets", classmethod(get_input_dimensionality_offsets))
    dimensionality_reference = pick_dimensionality_referencE_property(inputs=dynamic_manifest.inputs)
    get_dimensionality_reference_property = lambda cls: dimensionality_reference
    setattr(model, "get_dimensionality_reference_property", classmethod(get_dimensionality_reference_property))
    get_output_dimensionality_offset = lambda cls: dynamic_manifest.output_dimensionality_offset
    setattr(model, "get_output_dimensionality_offset", classmethod(get_output_dimensionality_offset))
    accepts_batch_input = lambda cls: dynamic_manifest.accepts_batch_input
    setattr(model, "accepts_batch_input", classmethod(accepts_batch_input))
    accepts_empty_values = lambda cls: dynamic_manifest.accepts_empty_values
    setattr(model, "accepts_empty_values", classmethod(accepts_empty_values))
    return model


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
        input_type_union_elements = []
        for selector_type in input_definition.selector_types:
            selector_kind_names = input_definition.selector_data_kind.get(selector_type, ["*"])
            selector_kind = []
            for kind_name in selector_kind_names:
                selector_kind.append(kinds_lookup[kind_name])
            if selector_type is SelectorType.INPUT_IMAGE:
                input_type_union_elements.append(WorkflowImageSelector)
            elif selector_type is SelectorType.INPUT_PARAMETER:
                input_type_union_elements.append(WorkflowParameterSelector(kind=selector_kind))
            else:
                input_type_union_elements.append(StepOutputSelector(kind=selector_kind))
        for value_type_name in input_definition.value_types:
            value_type = PYTHON_TYPES_MAPPING[value_type_name]
            input_type_union_elements.append(value_type)
        if not input_type_union_elements:
            input_type_union_elements.append(Any)
        if len(input_type_union_elements) > 1:
            input_type = Union[tuple(input_type_union_elements)]
        else:
            input_type = input_type_union_elements[0]
        if input_definition.is_optional:
            input_type = Optional[input_type]
        field_metadata = Field()
        if input_definition.has_default_value:
            default_value = input_definition.default_value
            field_metadata_params = {}
            if isinstance(default_value, list) or isinstance(default_value, dict) or isinstance(default_value, set):
                field_metadata_params["default_factory"] = lambda: default_value
            else:
                field_metadata_params["default"] = default_value
            field_metadata = Field(**field_metadata_params)
        result[input_name] = input_type, field_metadata
    return result


def build_outputs_definitions(
    outputs: Dict[str, DynamicOutputDefinition],
    kinds_lookup: Dict[str, Kind],
) -> List[OutputDefinition]:
    result = []
    for name, definition in outputs.items():
        if not definition.kind:
            result.append(OutputDefinition(name=name, kind=[WILDCARD_KIND]))
        else:
            actual_kinds = [kinds_lookup[kind_name] for kind_name in definition.kind]
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


def pick_dimensionality_referencE_property(inputs: Dict[str, DynamicInputDefinition]) -> Optional[str]:
    references = []
    for name, definition in inputs.items():
        if definition.is_dimensionality_reference:
            references.append(name)
    if not references:
        return None
    if len(references) == 1:
        return references[0]
    raise ValueError("Not expected to have multiple dimensionality references")


if __name__ == '__main__':
    lookup = {"image": Kind(name="image"), "predictions": Kind(name="predictions")}
    dynamic_manifest = DynamicBlockManifest(
        inputs={
            "images": DynamicInputDefinition(
                is_dimensionality_reference=True,
                selector_types=[SelectorType.INPUT_IMAGE, SelectorType.STEP_OUTPUT],
                selector_data_kind={
                    SelectorType.INPUT_IMAGE: ["image"],
                    SelectorType.STEP_OUTPUT: ["image"],
                },
            ),
            "predictions": DynamicInputDefinition(
                selector_types=[SelectorType.STEP_OUTPUT],
                selector_data_kind={
                    SelectorType.STEP_OUTPUT: ["predictions"],
                },
                dimensionality_offset=1,
            ),
            "param": DynamicInputDefinition(
                is_optional=True,
                has_default_value=True,
                value_types=[ValueType.STRING, ValueType.FLOAT],
                default_value=None,
            )
        },
        outputs={"result": DynamicOutputDefinition()},
    )

    result = assembly_dynamic_block_manifest(
        block_name="my_block",
        block_type="custom_block",
        dynamic_manifest=dynamic_manifest,
        kinds_lookup=lookup,
    )

    result_instance = result(
        name="a",
        type="custom_block",
        images="$inputs.image",
        predictions="$steps.step.predictions",
    )
    print(result_instance)
    print(result_instance.describe_outputs())
    print(result_instance.get_actual_outputs())
    print(result_instance.get_input_dimensionality_offsets())
    print(result_instance.get_dimensionality_reference_property())
    print(result_instance.get_output_dimensionality_offset())
    print(result_instance.accepts_batch_input())
    print(result_instance.accepts_empty_values())


