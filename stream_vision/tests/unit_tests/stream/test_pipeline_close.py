"""How a workflow pipeline's end reaches the Execution Engine's two close phases."""

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from queue import Queue
from typing import Callable, Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest
from streamvision.stream import utils
from streamvision.stream.pipeline import InferencePipeline
from streamvision.stream.utils import on_pipeline_end


def _make_pipeline(on_end: Optional[Callable[[], None]]) -> InferencePipeline:
    return InferencePipeline(
        on_video_frame=MagicMock(),
        video_sources=[],
        predictions_queue=Queue(),
        watchdog=MagicMock(),
        status_update_handlers=[],
        on_pipeline_end=on_end,
    )


def _workflow_pipeline_end(
    execution_engine, thread_pool_executor: ThreadPoolExecutor
) -> partial:
    # The closure `init_with_workflow` builds.
    return partial(
        on_pipeline_end,
        thread_pool_executor=thread_pool_executor,
        cancel_thread_pool_tasks_on_exit=True,
        profiler=MagicMock(),
        profiling_directory="./unused",
        execution_engine_thread_pool_executor=None,
        execution_engine=execution_engine,
        block_close_deadline_seconds=None,
    )


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", False)
def test_join_passes_the_deadline_set_on_the_pipeline_to_prepare_close() -> None:
    # given
    execution_engine = MagicMock()
    pipeline = _make_pipeline(
        _workflow_pipeline_end(execution_engine, ThreadPoolExecutor(max_workers=1))
    )

    # when
    pipeline.set_block_close_deadline(4.0)
    pipeline.join()

    # then
    execution_engine.prepare_close.assert_called_once_with(4.0)
    execution_engine.close.assert_called_once_with()


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", False)
@mock.patch.object(utils, "WORKFLOWS_BLOCK_CLOSE_DEADLINE_SECONDS", 30.0)
def test_join_without_a_set_deadline_uses_the_configured_default() -> None:
    # given
    execution_engine = MagicMock()
    pipeline = _make_pipeline(
        _workflow_pipeline_end(execution_engine, ThreadPoolExecutor(max_workers=1))
    )

    # when
    pipeline.join()

    # then
    execution_engine.prepare_close.assert_called_once_with(30.0)


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", False)
def test_setting_the_deadline_keeps_the_other_pipeline_end_arguments() -> None:
    # given
    execution_engine = MagicMock()
    thread_pool_executor = ThreadPoolExecutor(max_workers=1)
    pipeline = _make_pipeline(
        _workflow_pipeline_end(execution_engine, thread_pool_executor)
    )

    # when
    pipeline.set_block_close_deadline(2.0)
    pipeline.set_block_close_deadline(3.0)

    # then
    on_end = pipeline._on_pipeline_end
    assert on_end.func is on_pipeline_end
    assert on_end.keywords["block_close_deadline_seconds"] == 3.0
    assert on_end.keywords["thread_pool_executor"] is thread_pool_executor
    assert on_end.keywords["execution_engine"] is execution_engine
    thread_pool_executor.shutdown()


@pytest.mark.parametrize("on_end", [None, MagicMock(), partial(print, "end")])
def test_setting_the_deadline_leaves_custom_logic_pipeline_ends_alone(on_end) -> None:
    # given a pipeline built with custom logic
    pipeline = _make_pipeline(on_end)

    # when
    pipeline.set_block_close_deadline(2.0)

    # then
    assert pipeline._on_pipeline_end is on_end


@mock.patch.object(utils, "ENABLE_WORKFLOWS_PROFILING", False)
def test_two_concurrent_joins_close_the_workflow_blocks_once() -> None:
    # given a real engine whose block close is slow; two threads join the pipeline
    pytest.importorskip("roboflow_workflows")
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    execution_engine = ExecutionEngine.init(
        workflow_definition={
            "version": "1.0",
            "inputs": [{"type": "WorkflowImage", "name": "image"}],
            "steps": [
                {
                    "type": "roboflow_core/cache_get@v1",
                    "name": "cached",
                    "image": "$inputs.image",
                    "key": "detections",
                }
            ],
            "outputs": [
                {"type": "JsonField", "name": "out", "selector": "$steps.cached.*"}
            ],
        },
        init_parameters={},
    )
    block = execution_engine._engine._compiled_workflow.steps["cached"].step
    calls = []
    calls_lock = threading.Lock()
    original_close = block.close

    def record(name: str) -> None:
        with calls_lock:
            calls.append(name)

    def slow_close() -> None:
        record("close")
        threading.Event().wait(0.1)
        original_close()

    block.prepare_close = lambda deadline_seconds: record(
        f"prepare_close:{deadline_seconds}"
    )
    block.close = slow_close
    pipeline = _make_pipeline(
        _workflow_pipeline_end(execution_engine, ThreadPoolExecutor(max_workers=1))
    )
    pipeline.set_block_close_deadline(5.0)
    start = threading.Barrier(2)

    def join() -> None:
        start.wait(timeout=5)
        pipeline.join()
        record("joined")

    threads = [threading.Thread(target=join) for _ in range(2)]

    # when
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # then: each phase ran once and neither join returned before the close finished
    assert calls == ["prepare_close:5.0", "close", "joined", "joined"]
