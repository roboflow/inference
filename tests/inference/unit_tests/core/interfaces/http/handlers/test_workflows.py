from inference.core.interfaces.http.handlers.workflows import (
    filter_out_unwanted_workflow_outputs,
)


def test_filter_out_unwanted_workflow_outputs_when_nothing_to_filter() -> None:
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=None,
    )

    # then
    assert result == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_filter_out_unwanted_workflow_outputs_when_empty_filter() -> None:
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=[],
    )

    # then
    assert result == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_filter_out_unwanted_workflow_outputs_when_fields_to_be_filtered() -> None:
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=["a"],
    )

    # then
    assert result == [
        {"b": 2},
        {"b": 4},
    ]


def test_filter_out_unwanted_workflow_outputs_when_filter_defines_non_existing_fields() -> (
    None
):
    # given
    workflow_results = [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]

    # when
    result = filter_out_unwanted_workflow_outputs(
        workflow_results=workflow_results,
        excluded_fields=["non-existing"],
    )

    # then
    assert result == [
        {"a": 1, "b": 2},
        {"a": 3, "b": 4},
    ]


def test_schedule_execution_engine_close_runs_after_the_tasks_queued_by_the_run() -> (
    None
):
    # given: a sink queued fire-and-forget work during the run
    from unittest.mock import MagicMock

    from fastapi import BackgroundTasks

    from inference.core.interfaces.http.handlers.workflows import (
        schedule_execution_engine_close,
    )

    order = []
    background_tasks = BackgroundTasks()
    background_tasks.add_task(lambda: order.append("sink task"))
    execution_engine = MagicMock()
    execution_engine.close.side_effect = lambda: order.append("close")

    # when
    schedule_execution_engine_close(
        execution_engine=execution_engine, background_tasks=background_tasks
    )

    # then: nothing closes during the request, and the close comes last
    execution_engine.close.assert_not_called()
    for task in background_tasks.tasks:
        task.func(*task.args, **task.kwargs)
    assert order == ["sink task", "close"]


def test_schedule_execution_engine_close_without_background_tasks_leaves_it_to_gc() -> (
    None
):
    from unittest.mock import MagicMock

    from inference.core.interfaces.http.handlers.workflows import (
        schedule_execution_engine_close,
    )

    execution_engine = MagicMock()

    schedule_execution_engine_close(
        execution_engine=execution_engine, background_tasks=None
    )

    execution_engine.close.assert_not_called()


def test_schedule_execution_engine_close_accepts_engines_without_close() -> None:
    from fastapi import BackgroundTasks

    from inference.core.interfaces.http.handlers.workflows import (
        schedule_execution_engine_close,
    )

    background_tasks = BackgroundTasks()

    schedule_execution_engine_close(
        execution_engine=object(), background_tasks=background_tasks
    )

    assert background_tasks.tasks == []
