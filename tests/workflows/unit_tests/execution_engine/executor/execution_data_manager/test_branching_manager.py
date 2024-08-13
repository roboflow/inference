import pytest

from inference.core.workflows.errors import ExecutionEngineRuntimeError
from inference.core.workflows.execution_engine.v1.executor.execution_data_manager.branching_manager import (
    BranchingManager,
)


def test_register_batch_oriented_mask_for_the_first_time() -> None:
    # given
    manager = BranchingManager.init()

    # when
    manager.register_batch_oriented_mask(
        execution_branch="my-branch", mask={(0,), (1,), (2,), (3,)}
    )

    # then
    assert (
        manager.is_execution_branch_registered(execution_branch="my-branch") is True
    ), "Expected mask to be registered"
    assert (
        manager.is_execution_branch_batch_oriented(execution_branch="my-branch") is True
    ), "Expected mask to be registered as batch oriented"
    assert manager.get_mask(execution_branch="my-branch") == {
        (0,),
        (1,),
        (2,),
        (3,),
    }, "Expected mask to be possible to retrieved"


def test_register_batch_oriented_mask_for_the_second_time() -> None:
    # given
    manager = BranchingManager.init()

    # when
    manager.register_batch_oriented_mask(
        execution_branch="my-branch", mask={(0,), (1,), (2,), (3,)}
    )
    with pytest.raises(ExecutionEngineRuntimeError):
        manager.register_batch_oriented_mask(
            execution_branch="my-branch", mask={(0,), (1,)}
        )

    # then
    assert (
        manager.is_execution_branch_registered(execution_branch="my-branch") is True
    ), "Expected mask to be registered"
    assert (
        manager.is_execution_branch_batch_oriented(execution_branch="my-branch") is True
    ), "Expected mask to be registered as batch oriented"
    assert manager.get_mask(execution_branch="my-branch") == {
        (0,),
        (1,),
        (2,),
        (3,),
    }, "Expected mask to be possible to retrieved"


def test_register_non_batch_oriented_mask_for_the_first_time() -> None:
    # given
    manager = BranchingManager.init()

    # when
    manager.register_non_batch_mask(
        execution_branch="my-branch",
        mask=True,
    )

    # then
    assert (
        manager.is_execution_branch_registered(execution_branch="my-branch") is True
    ), "Expected mask to be registered"
    assert (
        manager.is_execution_branch_batch_oriented(execution_branch="my-branch")
        is False
    ), "Expected mask to be registered as non-batch oriented"
    assert (
        manager.get_mask(execution_branch="my-branch") is True
    ), "Expected mask to be possible to retrieved"


def test_register_non_batch_oriented_mask_for_the_second_time() -> None:
    # given
    manager = BranchingManager.init()

    # when
    manager.register_non_batch_mask(
        execution_branch="my-branch",
        mask=True,
    )
    with pytest.raises(ExecutionEngineRuntimeError):
        manager.register_non_batch_mask(
            execution_branch="my-branch",
            mask=False,
        )

    # then
    assert (
        manager.is_execution_branch_registered(execution_branch="my-branch") is True
    ), "Expected mask to be registered"
    assert (
        manager.is_execution_branch_batch_oriented(execution_branch="my-branch")
        is False
    ), "Expected mask to be registered as non-batch oriented"
    assert (
        manager.get_mask(execution_branch="my-branch") is True
    ), "Expected mask to be possible to retrieved"


def test_getting_mask_for_not_registered_branch() -> None:
    # given
    manager = BranchingManager.init()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        manager.get_mask(execution_branch="my-branch")


def test_getting_info_about_not_registered_branch() -> None:
    # given
    manager = BranchingManager.init()

    # when
    with pytest.raises(ExecutionEngineRuntimeError):
        manager.is_execution_branch_batch_oriented(execution_branch="my-branch")


def test_is_execution_branch_registered_for_non_registered_mask() -> None:
    # given
    manager = BranchingManager.init()

    # when
    result = manager.is_execution_branch_registered(execution_branch="my-branch")

    # then
    assert result is False
