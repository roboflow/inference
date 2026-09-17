"""The workflows-local default observer: runs the work, records nothing.

`NullExecutionObserver` is what the module gets when no host binds one, so
these tests pin the two things a standalone workflows run depends on: the
result of the wrapped call reaches the caller unchanged (including the
exception path), and no context is established that a later run could inherit.
"""

import pytest

from inference.core.workflows.prototypes.observer import (
    NULL_EXECUTION_OBSERVER,
    ExecutionObserver,
    NullExecutionObserver,
)


def test_null_observer_satisfies_the_protocol() -> None:
    assert isinstance(NullExecutionObserver(), ExecutionObserver)
    assert isinstance(NULL_EXECUTION_OBSERVER, NullExecutionObserver)


def test_observe_workflow_run_returns_the_wrapped_result() -> None:
    # given
    observer = NullExecutionObserver()

    # when
    result = observer.observe_workflow_run(
        workflow=object(),
        runtime_parameters={"image": []},
        workflow_id="wf-1",
        fps=30.0,
        is_preview=True,
        run=lambda: [{"a": 1}],
    )

    # then
    assert result == [{"a": 1}]


def test_observe_workflow_run_propagates_the_error() -> None:
    observer = NullExecutionObserver()

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        observer.observe_workflow_run(
            workflow=object(),
            runtime_parameters={},
            workflow_id=None,
            fps=0,
            is_preview=False,
            run=boom,
        )


def test_capture_step_context_establishes_nothing() -> None:
    assert NullExecutionObserver().capture_step_context() is None


def test_step_scope_is_a_transparent_context_manager() -> None:
    # given
    observer = NullExecutionObserver()
    entered = []

    # when
    with observer.step_scope(context=None, step_name="a_step"):
        entered.append(True)

    # then
    assert entered == [True]


def test_step_scope_does_not_swallow_errors() -> None:
    observer = NullExecutionObserver()
    with pytest.raises(ValueError):
        with observer.step_scope(context=None, step_name="a_step"):
            raise ValueError("from the step")


def test_observe_block_run_returns_the_wrapped_result() -> None:
    observer = NullExecutionObserver()
    result = observer.observe_block_run(
        block=object(),
        block_args=(),
        block_kwargs={"a": 1},
        run=lambda: {"result": 2},
    )
    assert result == {"result": 2}


def test_observe_model_run_returns_the_wrapped_result() -> None:
    observer = NullExecutionObserver()
    result = observer.observe_model_run(
        block=object(),
        model_id="sam2video",
        images=[],
        run=lambda: [{"masks": None}],
    )
    assert result == [{"masks": None}]
