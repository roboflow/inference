import pytest

from inference.enterprise.workflows.entities.base import OutputDefinition
from inference.enterprise.workflows.errors import (
    ExecutionEngineRuntimeError,
    InvalidBlockBehaviourError,
)
from inference.enterprise.workflows.execution_engine.executor.execution_cache import (
    ExecutionCache,
)


def test_execution_cache_when_attempted_to_register_outputs_of_unregistered_step() -> (
    None
):
    # given
    cache = ExecutionCache.init()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.register_step_outputs(
            step_name="non_existing", outputs=[{"some": "data"}]
        )


def test_execution_cache_when_attempted_to_register_malformed_outputs() -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    with pytest.raises(InvalidBlockBehaviourError):
        cache.register_step_outputs(step_name="some", outputs=[39])


def test_execution_cache_getting_output_when_step_is_not_registered() -> None:
    # given
    cache = ExecutionCache.init()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.get_output(selector="$steps.some.value")


def test_execution_cache_getting_output_when_step_output_is_not_registered() -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.get_output(selector="$steps.some.value")


@pytest.mark.parametrize("selector", ["$steps.some", "$inputs.param"])
def test_execution_cache_getting_output_when_selector_does_not_point_to_step_output(
    selector: str,
) -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.get_output(selector=selector)


def test_execution_cache_getting_output_when_step_output_is_registered_but_empty() -> (
    None
):
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    result = cache.get_output(selector="$steps.some.my_output")

    # then
    assert result == []


def test_execution_cache_getting_output_when_step_output_is_registered_and_not_empty() -> (
    None
):
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )
    cache.register_step_outputs(
        step_name="some", outputs=[{"my_output": "one"}, {"my_output": "two"}]
    )

    # when
    result = cache.get_output(selector="$steps.some.my_output")

    # then
    assert result == ["one", "two"]


def test_execution_cache_getting_output_when_step_not_declared_output_is_registered_and_not_empty() -> (
    None
):
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )
    cache.register_step_outputs(
        step_name="some",
        outputs=[
            {"my_output": "one", "not_declared": 1},
            {"my_output": "two", "not_declared": 2},
        ],
    )

    # when
    result = cache.get_output(selector="$steps.some.not_declared")

    # then
    assert result == [1, 2]


def test_output_represent_batch_when_selector_points_to_not_declared_step() -> None:
    # given
    cache = ExecutionCache.init()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.output_represent_batch(
            selector="$steps.non_existing.predictions",
        )


def test_output_represent_batch_when_selector_points_to_not_declared_property() -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.output_represent_batch(
            selector="$steps.some.not_declared",
        )


@pytest.mark.parametrize("selector", ["$steps.some", "$inputs.param"])
def test_output_represent_batch_when_selector_does_not_point_to_step_output(
    selector: str,
) -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        cache.output_represent_batch(selector=selector)


def test_output_represent_batch_when_selector_points_to_registered_element_supporting_batches() -> (
    None
):
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    result = cache.output_represent_batch(selector="$steps.some.my_output")

    # then
    assert result is True


def test_output_represent_batch_when_selector_points_to_registered_element_not_supporting_batches() -> (
    None
):
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=False,
    )

    # when
    result = cache.output_represent_batch(selector="$steps.some.my_output")

    # then
    assert result is False


def test_contains_step_when_step_is_registered() -> None:
    # given
    cache = ExecutionCache.init()

    # when
    result = cache.contains_step(step_name="not_declared")

    # then
    assert result is False


def test_contains_step_when_step_is_not_registered() -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    result = cache.contains_step(step_name="some")

    # then
    assert result is True


def test_is_value_registered_when_step_is_not_registered() -> None:
    # given
    cache = ExecutionCache.init()

    # when
    result = cache.is_value_registered(selector="$steps.not_declared.some")

    # then
    assert result is False


def test_is_value_registered_when_step_output_is_not_registered() -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    result = cache.is_value_registered(selector="$steps.some.not_registered")

    # then
    assert result is False


@pytest.mark.parametrize("selector", ["$steps.some", "$inputs.param"])
def test_is_value_registered_when_selector_does_not_point_to_step_output(
    selector: str,
) -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    result = cache.is_value_registered(selector=selector)

    # then
    assert result is False


def test_is_value_registered_when_selector_points_to_registered_value() -> None:
    # given
    cache = ExecutionCache.init()
    cache.register_step(
        step_name="some",
        output_definitions=[OutputDefinition(name="my_output")],
        compatible_with_batches=True,
    )

    # when
    result = cache.is_value_registered(selector="$steps.some.my_output")

    # then
    assert result is True
