import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.dynamic_batches_manager import (
    DynamicBatchesManager,
    assembly_root_batch_indices,
)
from tests.workflows.unit_tests.execution_engine.executor.execution_data_manager.common import (
    prepare_execution_graph_for_tests,
)


def test_assembly_root_batch_indices() -> None:
    # given
    execution_graph = prepare_execution_graph_for_tests(
        steps_names=["some"],
        are_batch_oriented=[True],
        steps_outputs=[[OutputDefinition(name="a")]],
    )
    runtime_parameters = {"image": ["dummy-image-1", "dummy-image-2"]}

    # when
    result = assembly_root_batch_indices(
        execution_graph=execution_graph,
        runtime_parameters=runtime_parameters,
    )

    # then
    assert len(result) == 1, "Expected one batch input to be found"
    value = next(iter(result.values()))
    assert value == [(0,), (1,)], "Expected 2 input indices starting from (0, )"


def test_registration_of_lineage_indices() -> None:
    # given
    manager = DynamicBatchesManager(lineage2indices={})

    # when
    manager.register_element_indices_for_lineage(
        lineage=["root", "my-step-increasing-lineage"],
        indices=[(0, 0), (0, 1), (1, 0)],
    )

    # then
    assert manager.is_lineage_registered(
        lineage=["root", "my-step-increasing-lineage"]
    ), "Expected lineage to be registered"
    assert manager.get_indices_for_data_lineage(
        lineage=["root", "my-step-increasing-lineage"]
    ) == [(0, 0), (0, 1), (1, 0)], "Expected indices to be possible to be recovered"


def test_getting_indices_of_not_registered_lineage() -> None:
    # given
    manager = DynamicBatchesManager(lineage2indices={})

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        _ = manager.get_indices_for_data_lineage(
            lineage=["root", "my-step-increasing-lineage"]
        )


def test_is_lineage_registered_for_non_registered_lineage() -> None:
    manager = DynamicBatchesManager(lineage2indices={})

    # when
    result = manager.is_lineage_registered(
        lineage=["root", "my-step-increasing-lineage"]
    )

    # then
    assert result is False
