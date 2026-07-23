from unittest.mock import MagicMock

from inference.core.workflows.execution_engine.v1.executor.core import (
    close_stream_pipelines,
    flush_stream_pipeline_outputs,
    run_step,
)


def test_disabled_step_is_not_executed() -> None:
    workflow = MagicMock(disabled_steps=frozenset({"sink"}))
    execution_data_manager = MagicMock()

    run_step(
        step_selector="$steps.sink",
        workflow=workflow,
        execution_data_manager=execution_data_manager,
        profiler=MagicMock(),
    )

    execution_data_manager.is_step_simd.assert_not_called()


def test_disabled_step_stream_pipeline_is_not_flushed() -> None:
    sink = MagicMock()
    workflow = MagicMock(
        steps={"sink": sink},
        disabled_steps=frozenset({"sink"}),
    )

    result = flush_stream_pipeline_outputs(
        workflow=workflow,
        execution_data_manager=MagicMock(),
    )

    assert result == []
    sink.step.flush_stream_pipeline_outputs.assert_not_called()


def test_disabled_step_stream_pipeline_is_not_closed() -> None:
    sink = MagicMock()
    workflow = MagicMock(
        steps={"sink": sink},
        disabled_steps=frozenset({"sink"}),
    )

    close_stream_pipelines(workflow=workflow)

    sink.step.close_stream_pipeline.assert_not_called()
