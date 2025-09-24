import pytest

from inference.core.workflows.errors import (
    BlockInitParameterNotProvidedError,
    BlockInterfaceError,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
    call_if_callable,
    initialise_step,
    retrieve_init_parameter_values,
    retrieve_init_parameters_values,
)
from tests.workflows.unit_tests.execution_engine.compiler.plugin_with_test_blocks.blocks import (
    ExampleBlockWithFaultyInit,
    ExampleBlockWithFaultyInitManifest,
    ExampleBlockWithInit,
    ExampleBlockWithInitManifest,
)


def test_call_if_callable_when_callable_given() -> None:
    # given
    calls = []

    def my_callable() -> int:
        calls.append(1)
        return 1

    # when
    result = call_if_callable(value=my_callable)

    # then
    assert result == 1
    assert calls == [1], "Expected to call function passed as argument"


def test_call_if_callable_when_simple_value_given() -> None:
    # when
    result = call_if_callable(value=37)

    # then
    assert result == 37, "Expected not to do anything apart from returning value"


def test_retrieve_init_parameter_values_when_explicit_parameters_define_value() -> None:
    # given
    explicit_init_parameters = {
        "some.param": 39,
    }

    # when
    result = retrieve_init_parameter_values(
        block_name="block",
        block_init_parameter="param",
        block_source="some",
        explicit_init_parameters=explicit_init_parameters,
        initializers={},
    )

    # then
    assert result == 39


def test_retrieve_init_parameter_values_when_initializers_define_value() -> None:
    # given
    initializers = {
        "some.param": 42,
    }

    # when
    result = retrieve_init_parameter_values(
        block_name="block",
        block_init_parameter="param",
        block_source="some",
        explicit_init_parameters={},
        initializers=initializers,
    )

    # then
    assert result == 42


def test_retrieve_init_parameter_values_when_explicit_parameters_define_generic_value_for_specific_param() -> (
    None
):
    # given
    explicit_init_parameters = {
        "param": 39,
    }

    # when
    result = retrieve_init_parameter_values(
        block_name="block",
        block_init_parameter="param",
        block_source="some",
        explicit_init_parameters=explicit_init_parameters,
        initializers={},
    )

    # then
    assert result == 39


def test_retrieve_init_parameter_values_when_initializers_define_generic_value_for_specific_param() -> (
    None
):
    initializers = {
        "param": lambda: 42,
    }

    # when
    result = retrieve_init_parameter_values(
        block_name="block",
        block_init_parameter="param",
        block_source="some",
        explicit_init_parameters={},
        initializers=initializers,
    )

    # then
    assert result == 42


def test_retrieve_init_parameter_values_when_parameter_cannot_be_resolved() -> None:
    # when
    with pytest.raises(BlockInitParameterNotProvidedError):
        _ = retrieve_init_parameter_values(
            block_name="block",
            block_init_parameter="param",
            block_source="some",
            explicit_init_parameters={},
            initializers={},
        )


def test_retrieve_init_parameters_values() -> None:
    # given
    explicit_init_parameters = {
        "some.param_1": 39,
        "param_2": 42,
    }
    initializers = {
        "some.param_3": lambda: 37,
        "param_4": 99,
    }

    # when
    result = retrieve_init_parameters_values(
        block_name="block",
        block_init_parameters=["param_1", "param_2", "param_3", "param_4"],
        block_source="some",
        explicit_init_parameters=explicit_init_parameters,
        initializers=initializers,
    )

    # then
    assert result == {
        "param_1": 39,
        "param_2": 42,
        "param_3": 37,
        "param_4": 99,
    }


def test_initialise_step_when_initialisation_is_successful() -> None:
    # given
    block_specification = BlockSpecification(
        block_source="test_plugin",
        identifier="test_plugin.ExampleBlockWithInit",
        block_class=ExampleBlockWithInit,
        manifest_class=ExampleBlockWithInit.get_manifest(),
    )
    manifest = ExampleBlockWithInitManifest(
        type="ExampleBlockWithInit",
        name="some",
        predictions=["$steps.a.predictions"],
    )

    # when
    result = initialise_step(
        step_manifest=manifest,
        block_specification=block_specification,
        explicit_init_parameters={
            "a": 9,
            "b": 30,
        },
        initializers={},
    )

    # then
    assert result.step.a == 9, "Expected a parameter to be set into 9"
    assert result.step.b == 30, "Expected b parameter to be set into 30"


def test_initialise_step_when_initialisation_failed() -> None:
    # given
    block_specification = BlockSpecification(
        block_source="test_plugin",
        identifier="test_plugin.ExampleBlockWithFaultyInit",
        block_class=ExampleBlockWithFaultyInit,
        manifest_class=ExampleBlockWithFaultyInit.get_manifest(),
    )
    manifest = ExampleBlockWithFaultyInitManifest(
        type="ExampleBlockWithFaultyInit",
        name="some",
        predictions=["$steps.a.predictions"],
    )

    # when
    with pytest.raises(BlockInterfaceError):
        _ = initialise_step(
            step_manifest=manifest,
            block_specification=block_specification,
            explicit_init_parameters={
                "a": 9,
                "b": lambda: 30,
            },
            initializers={},
        )
