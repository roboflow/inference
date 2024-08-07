from typing import Type, Union
from unittest import mock

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticUndefinedType

from inference.core.workflows.errors import DynamicBlockError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    WILDCARD_KIND,
    Kind,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks import block_assembler
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_assembler import (
    build_input_field_metadata,
    build_outputs_definitions,
    collect_input_dimensionality_offsets,
    collect_python_types_for_selectors,
    collect_python_types_for_values,
    create_dynamic_block_specification,
    pick_dimensionality_reference_property,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    DynamicBlockDefinition,
    DynamicInputDefinition,
    DynamicOutputDefinition,
    ManifestDescription,
    PythonCode,
    SelectorType,
    ValueType,
)


def test_pick_dimensionality_reference_property_when_there_is_no_reference_property() -> (
    None
):
    # given
    inputs = {
        "a": DynamicInputDefinition(
            type="DynamicInputDefinition", selector_types=[SelectorType.INPUT_PARAMETER]
        ),
        "b": DynamicInputDefinition(
            type="DynamicInputDefinition", value_types=[ValueType.INTEGER]
        ),
    }

    # when
    result = pick_dimensionality_reference_property(
        block_type="some",
        inputs=inputs,
    )

    # then
    assert result is None


def test_pick_dimensionality_reference_property_when_there_is_single_reference_property() -> (
    None
):
    # given
    inputs = {
        "a": DynamicInputDefinition(
            type="DynamicInputDefinition",
            selector_types=[SelectorType.INPUT_PARAMETER],
            is_dimensionality_reference=True,
        ),
        "b": DynamicInputDefinition(
            type="DynamicInputDefinition", value_types=[ValueType.INTEGER]
        ),
    }

    # when
    result = pick_dimensionality_reference_property(
        block_type="some",
        inputs=inputs,
    )

    # then
    assert result == "a", "Expected `a` to be picked as dimensionality reference"


def test_pick_dimensionality_reference_property_when_there_are_multiple_reference_properties() -> (
    None
):
    # given
    inputs = {
        "a": DynamicInputDefinition(
            type="DynamicInputDefinition",
            selector_types=[SelectorType.INPUT_PARAMETER],
            is_dimensionality_reference=True,
        ),
        "b": DynamicInputDefinition(
            type="DynamicInputDefinition",
            value_types=[ValueType.INTEGER],
            is_dimensionality_reference=True,
        ),
    }

    # when
    with pytest.raises(DynamicBlockError):
        _ = pick_dimensionality_reference_property(
            block_type="some",
            inputs=inputs,
        )


def test_build_outputs_definitions_when_build_should_succeed() -> None:
    # given
    outputs = {
        "a": DynamicOutputDefinition(type="DynamicOutputDefinition"),
        "b": DynamicOutputDefinition(
            type="DynamicOutputDefinition", kind=["string", "integer"]
        ),
    }
    kinds_lookup = {
        "*": WILDCARD_KIND,
        "string": Kind(name="string"),
        "integer": Kind(name="integer"),
    }

    # when
    result = build_outputs_definitions(
        block_type="some",
        outputs=outputs,
        kinds_lookup=kinds_lookup,
    )

    # then
    assert result == [
        OutputDefinition(name="a", kind=[WILDCARD_KIND]),
        OutputDefinition(
            name="b", kind=[kinds_lookup["string"], kinds_lookup["integer"]]
        ),
    ], "Expected outputs to be built such that `a` has * kind and `b` has exactly the kinds that were defined"


def test_build_outputs_definitions_when_build_should_fail_on_not_recognised_kind() -> (
    None
):
    # given
    outputs = {
        "a": DynamicOutputDefinition(type="DynamicOutputDefinition"),
        "b": DynamicOutputDefinition(
            type="DynamicOutputDefinition", kind=["string", "integer"]
        ),
    }
    kinds_lookup = {
        "*": WILDCARD_KIND,
        "string": Kind(name="string"),
    }

    # when
    with pytest.raises(DynamicBlockError):
        _ = build_outputs_definitions(
            block_type="some",
            outputs=outputs,
            kinds_lookup=kinds_lookup,
        )


def test_collect_input_dimensionality_offsets() -> None:
    # given
    inputs = {
        "a": DynamicInputDefinition(
            type="DynamicInputDefinition",
            selector_types=[SelectorType.INPUT_PARAMETER],
            dimensionality_offset=1,
        ),
        "b": DynamicInputDefinition(
            type="DynamicInputDefinition",
            selector_types=[SelectorType.INPUT_PARAMETER],
        ),
        "c": DynamicInputDefinition(
            type="DynamicInputDefinition",
            selector_types=[SelectorType.INPUT_PARAMETER],
            dimensionality_offset=-1,
        ),
    }

    # when
    result = collect_input_dimensionality_offsets(inputs=inputs)

    # then
    assert result == {
        "a": 1,
        "c": -1,
    }, "Expected only entries with non-default value be given in results"


def test_build_input_field_metadata_for_field_without_default_value() -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        selector_types=[SelectorType.INPUT_PARAMETER],
        dimensionality_offset=1,
    )

    # when
    result = build_input_field_metadata(input_definition=input_definition)

    # then
    assert isinstance(result.default, PydanticUndefinedType)


def test_build_input_field_metadata_for_field_without_default_being_none() -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        value_types=[ValueType.INTEGER],
        is_optional=True,
        has_default_value=True,
    )

    # when
    result = build_input_field_metadata(input_definition=input_definition)

    # then
    assert result.default is None


def test_build_input_field_metadata_for_field_without_default_being_primitive() -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        value_types=[ValueType.INTEGER],
        is_optional=True,
        has_default_value=True,
        default_value=3.0,
    )

    # when
    result = build_input_field_metadata(input_definition=input_definition)

    # then
    assert result.default == 3


@pytest.mark.parametrize("default_type", [list, set, dict])
def test_build_input_field_metadata_for_field_without_default_being_compound(
    default_type: Union[Type[list], Type[set], Type[dict]],
) -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        value_types=[ValueType.LIST],
        has_default_value=True,
        default_value=default_type(),
    )

    # when
    result = build_input_field_metadata(input_definition=input_definition)

    # then
    assert (
        result.default_factory() == default_type()
    ), "Expected default_factory used creates new instance of compound element"


@pytest.mark.parametrize(
    "default_value", [[2, 3, 4], {"a", "b", "c"}, {"a": 1, "b": 2}]
)
def test_build_input_field_metadata_for_field_without_default_being_non_empty_compound(
    default_value: Union[set, list, dict],
) -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        value_types=[ValueType.LIST],
        has_default_value=True,
        default_value=default_value,
    )

    # when
    result = build_input_field_metadata(input_definition=input_definition)

    # then
    assert (
        result.default_factory() == default_value
    ), "Expected default_factory to create identical instance of compound data"
    assert id(result.default_factory()) != id(
        default_value
    ), "Expected default_factory to create new instance of compound data"


def test_collect_python_types_for_values_when_types_can_be_resolved() -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        value_types=[ValueType.LIST, ValueType.INTEGER],
    )

    # when
    result = collect_python_types_for_values(
        block_type="some",
        input_name="a",
        input_definition=input_definition,
    )

    # then
    assert result == [list, int], "Expected python types to be resolved properly"


@mock.patch.object(block_assembler, "PYTHON_TYPES_MAPPING", {})
def test_collect_python_types_for_values_when_type_cannot_be_resolved() -> None:
    # given
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        value_types=[ValueType.LIST, ValueType.INTEGER],
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = collect_python_types_for_values(
            block_type="some",
            input_name="a",
            input_definition=input_definition,
        )


def test_collect_python_types_for_selectors_when_collection_should_succeed() -> None:
    # given
    kinds_lookup = {
        "*": WILDCARD_KIND,
        "string": Kind(name="string"),
        "integer": Kind(name="integer"),
    }
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        selector_types=[
            SelectorType.INPUT_PARAMETER,
            SelectorType.INPUT_IMAGE,
            SelectorType.STEP_OUTPUT_IMAGE,
            SelectorType.STEP_OUTPUT,
        ],
        selector_data_kind={SelectorType.STEP_OUTPUT: ["string", "integer"]},
    )

    # when
    result = collect_python_types_for_selectors(
        block_type="some",
        input_name="a",
        input_definition=input_definition,
        kinds_lookup=kinds_lookup,
    )

    # then

    assert len(result) == 4, "Expected union of 4 types"
    assert repr(result[0]) == repr(
        WorkflowParameterSelector(kind=[WILDCARD_KIND])
    ), "First element of union is to be input param of kind *"
    assert repr(result[1]) == repr(
        WorkflowImageSelector
    ), "Second element of union is to be input image selector"
    assert repr(result[2]) == repr(
        StepOutputImageSelector
    ), "Third element of union is to be step output image selector"
    assert repr(result[3]) == repr(
        StepOutputSelector(kind=[kinds_lookup["string"], kinds_lookup["integer"]])
    ), "Last element of union is to be step output selector of kinds string integer"


def test_collect_python_types_for_selectors_when_collection_should_fail_on_unknown_kind() -> (
    None
):
    # given
    kinds_lookup = {
        "*": WILDCARD_KIND,
        "string": Kind(name="string"),
    }
    input_definition = DynamicInputDefinition(
        type="DynamicInputDefinition",
        selector_types=[
            SelectorType.INPUT_PARAMETER,
            SelectorType.INPUT_IMAGE,
            SelectorType.STEP_OUTPUT_IMAGE,
            SelectorType.STEP_OUTPUT,
        ],
        selector_data_kind={SelectorType.STEP_OUTPUT: ["string", "integer"]},
    )

    # when
    with pytest.raises(DynamicBlockError):
        _ = collect_python_types_for_selectors(
            block_type="some",
            input_name="a",
            input_definition=input_definition,
            kinds_lookup=kinds_lookup,
        )


PYTHON_CODE = """
def run(self, a, b):
    return {"output": b[::-1]}
"""


def test_create_dynamic_block_specification() -> None:
    # given
    kinds_lookup = {
        "*": WILDCARD_KIND,
        "string": Kind(name="string"),
        "integer": Kind(name="integer"),
    }
    dynamic_block_definition = DynamicBlockDefinition(
        type="DynamicBlockDefinition",
        manifest=ManifestDescription(
            type="ManifestDescription",
            block_type="MyBlock",
            inputs={
                "a": DynamicInputDefinition(
                    type="DynamicInputDefinition",
                    selector_types=[
                        SelectorType.INPUT_PARAMETER,
                        SelectorType.STEP_OUTPUT,
                    ],
                    selector_data_kind={
                        SelectorType.STEP_OUTPUT: ["string", "integer"]
                    },
                ),
                "b": DynamicInputDefinition(
                    type="DynamicInputDefinition",
                    value_types=[ValueType.LIST],
                    has_default_value=True,
                    default_value=[1, 2, 3],
                ),
            },
            outputs={
                "a": DynamicOutputDefinition(type="DynamicOutputDefinition"),
                "b": DynamicOutputDefinition(
                    type="DynamicOutputDefinition", kind=["string", "integer"]
                ),
            },
            output_dimensionality_offset=1,
            accepts_batch_input=True,
        ),
        code=PythonCode(
            type="PythonCode",
            run_function_code=PYTHON_CODE,
        ),
    )

    # when
    result = create_dynamic_block_specification(
        dynamic_block_definition=dynamic_block_definition,
        kinds_lookup=kinds_lookup,
    )

    # then
    assert result.block_source == "dynamic_workflows_blocks"
    assert result.manifest_class.describe_outputs() == [
        OutputDefinition(name="a", kind=[WILDCARD_KIND]),
        OutputDefinition(
            name="b", kind=[kinds_lookup["string"], kinds_lookup["integer"]]
        ),
    ], "Expected outputs to be built such that `a` has * kind and `b` has exactly the kinds that were defined"
    assert (
        result.manifest_class.accepts_batch_input() is True
    ), "Manifest defined to accept batch input"
    assert (
        result.manifest_class.accepts_empty_values() is False
    ), "Manifest defined not to accept empty input"
    assert (
        result.manifest_class.get_input_dimensionality_offsets() == {}
    ), "No explicit offsets defined"
    assert (
        result.manifest_class.get_dimensionality_reference_property() is None
    ), "No dimensionality reference property expected"
    assert (
        result.manifest_class.get_output_dimensionality_offset() == 1
    ), "Expected output dimensionality offset announced"

    block_instance = result.block_class()
    code_run_result = block_instance.run(a="some", b=[1, 2, 3])
    assert code_run_result == {
        "output": [3, 2, 1]
    }, "Expected code to work properly and revert second param"

    _ = result.manifest_class.model_validate(
        {"name": "some", "type": "MyBlock", "a": "$steps.some.a", "b": [1, 2, 3, 4, 5]}
    )  # no error expected

    _ = result.manifest_class.model_validate(
        {
            "name": "some",
            "type": "MyBlock",
            "a": "$steps.some.a",
        }
    )  # no error expected, default value for "b" defined

    with pytest.raises(ValidationError):
        _ = result.manifest_class.model_validate(
            {"name": "some", "type": "MyBlock", "a": "some", "b": [1, 2, 3, 4, 5]}
        )  # error expected - value "a" without selector

    with pytest.raises(ValidationError):
        _ = result.manifest_class.model_validate(
            {"name": "some", "type": "MyBlock", "a": "$steps.some.a", "b": 1}
        )  # error expected - value "b" not a list
