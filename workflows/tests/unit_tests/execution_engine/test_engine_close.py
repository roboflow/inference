import logging
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.engine import BaseExecutionEngine
from roboflow_workflows.execution_engine.v1.executor.core import (
    close_workflow_blocks,
    prepare_close_workflow_blocks,
)

CACHE_GET_DEFINITION = {
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
    "outputs": [{"type": "JsonField", "name": "out", "selector": "$steps.cached.*"}],
}


class ClosableBlock:
    def __init__(self, fail: bool = False):
        self.close_calls = 0
        self.fail = fail

    def close(self) -> None:
        self.close_calls += 1
        if self.fail:
            raise RuntimeError("cannot close")


class PreparableBlock:
    def __init__(self, fail: bool = False):
        self.deadlines = []
        self.fail = fail

    def prepare_close(self, deadline_seconds) -> None:
        self.deadlines.append(deadline_seconds)
        if self.fail:
            raise RuntimeError("cannot prepare")


def compiled_workflow(**steps) -> SimpleNamespace:
    return SimpleNamespace(
        steps={name: SimpleNamespace(step=step) for name, step in steps.items()}
    )


def test_close_workflow_blocks_closes_every_closable_block() -> None:
    first, second = ClosableBlock(), ClosableBlock()
    workflow = compiled_workflow(first=first, plain=object(), second=second)

    close_workflow_blocks(workflow=workflow)

    assert first.close_calls == 1
    assert second.close_calls == 1


def test_close_workflow_blocks_isolates_a_failing_block(caplog) -> None:
    failing, healthy = ClosableBlock(fail=True), ClosableBlock()
    workflow = compiled_workflow(failing=failing, healthy=healthy)

    with caplog.at_level(logging.WARNING):
        close_workflow_blocks(workflow=workflow)

    assert healthy.close_calls == 1
    assert "Could not close workflow step `failing`: cannot close" in caplog.text


def test_close_workflow_blocks_ignores_non_callable_close_attributes() -> None:
    block = SimpleNamespace(close="not callable")

    close_workflow_blocks(workflow=compiled_workflow(block=block))


def test_execution_engine_close_delegates_to_the_versioned_engine() -> None:
    engine = MagicMock(spec=BaseExecutionEngine)

    ExecutionEngine(engine=engine).close()

    engine.close.assert_called_once_with()


def test_base_engine_close_defaults_to_a_no_op() -> None:
    class MinimalEngine(BaseExecutionEngine):
        @classmethod
        def init(cls, *args, **kwargs):
            return cls()

        def run(self, *args, **kwargs):
            return []

    assert MinimalEngine().close() is None


def test_compiled_engine_closes_its_blocks_once_and_tolerates_repeated_close() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=CACHE_GET_DEFINITION, init_parameters={}
    )
    block = engine._engine._compiled_workflow.steps["cached"].step
    calls = []
    original_close = block.close
    block.close = lambda: calls.append("close") or original_close()

    engine.close()
    engine.close()

    assert calls == ["close"]


def test_prepare_close_workflow_blocks_passes_the_deadline_to_every_preparable_block() -> (
    None
):
    first, second = PreparableBlock(), PreparableBlock()
    closable = ClosableBlock()
    workflow = compiled_workflow(
        first=first, plain=object(), closable=closable, second=second
    )

    prepare_close_workflow_blocks(workflow, deadline_seconds=4.0)

    assert first.deadlines == [4.0]
    assert second.deadlines == [4.0]
    assert closable.close_calls == 0


def test_prepare_close_workflow_blocks_defaults_to_no_deadline() -> None:
    block = PreparableBlock()

    prepare_close_workflow_blocks(compiled_workflow(block=block))

    assert block.deadlines == [None]


def test_prepare_close_workflow_blocks_isolates_a_failing_block(caplog) -> None:
    failing, healthy = PreparableBlock(fail=True), PreparableBlock()
    workflow = compiled_workflow(failing=failing, healthy=healthy)

    with caplog.at_level(logging.WARNING):
        prepare_close_workflow_blocks(workflow, deadline_seconds=1.0)

    assert healthy.deadlines == [1.0]
    assert (
        "Could not prepare workflow step `failing` for close: cannot prepare"
        in caplog.text
    )


def test_prepare_close_workflow_blocks_ignores_non_callable_attributes() -> None:
    block = SimpleNamespace(prepare_close="not callable")

    prepare_close_workflow_blocks(compiled_workflow(block=block), deadline_seconds=1.0)


def test_execution_engine_prepare_close_delegates_to_the_versioned_engine() -> None:
    engine = MagicMock(spec=BaseExecutionEngine)

    ExecutionEngine(engine=engine).prepare_close(deadline_seconds=2.5)

    engine.prepare_close.assert_called_once_with(2.5)


def test_execution_engine_prepare_close_tolerates_an_engine_without_it() -> None:
    engine = SimpleNamespace(close=MagicMock())
    wrapper = ExecutionEngine(engine=engine)

    wrapper.prepare_close(deadline_seconds=2.5)
    wrapper.close()

    engine.close.assert_called_once_with()


def test_base_engine_prepare_close_defaults_to_a_no_op() -> None:
    class MinimalEngine(BaseExecutionEngine):
        @classmethod
        def init(cls, *args, **kwargs):
            return cls()

        def run(self, *args, **kwargs):
            return []

    assert MinimalEngine().prepare_close(deadline_seconds=1.0) is None


def test_compiled_engine_prepares_its_blocks_once_with_the_deadline() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=CACHE_GET_DEFINITION, init_parameters={}
    )
    block = engine._engine._compiled_workflow.steps["cached"].step
    deadlines = []
    block.prepare_close = deadlines.append

    engine.prepare_close(deadline_seconds=3.0)
    engine.prepare_close(deadline_seconds=9.0)

    assert deadlines == [3.0]


def test_compiled_engine_does_not_prepare_blocks_after_close() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=CACHE_GET_DEFINITION, init_parameters={}
    )
    block = engine._engine._compiled_workflow.steps["cached"].step
    calls = []
    block.prepare_close = lambda deadline: calls.append(("prepare", deadline))
    original_close = block.close
    block.close = lambda: calls.append(("close",)) or original_close()

    engine.close()
    engine.prepare_close(deadline_seconds=3.0)

    assert calls == [("close",)]


def test_concurrent_close_closes_blocks_once_and_waits_for_the_first_close() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=CACHE_GET_DEFINITION, init_parameters={}
    )
    block = engine._engine._compiled_workflow.steps["cached"].step
    close_entered = threading.Event()
    release_close = threading.Event()
    calls = []
    original_close = block.close

    def slow_close() -> None:
        calls.append("close")
        close_entered.set()
        assert release_close.wait(timeout=5)
        original_close()
        calls.append("closed")

    block.close = slow_close
    first = threading.Thread(target=engine.close)
    second = threading.Thread(target=lambda: engine.close() or calls.append("second"))

    first.start()
    assert close_entered.wait(timeout=5)
    second.start()
    second.join(timeout=0.2)
    second_blocked_while_closing = second.is_alive()
    release_close.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert second_blocked_while_closing
    assert calls == ["close", "closed", "second"]


def test_concurrent_prepare_close_and_close_run_each_phase_once() -> None:
    engine = ExecutionEngine.init(
        workflow_definition=CACHE_GET_DEFINITION, init_parameters={}
    )
    block = engine._engine._compiled_workflow.steps["cached"].step
    calls = []
    lock = threading.Lock()
    original_close = block.close

    def record(name: str) -> None:
        with lock:
            calls.append(name)

    block.prepare_close = lambda deadline: record("prepare")
    block.close = lambda: record("close") or original_close()
    start = threading.Barrier(4)

    def end_pipeline() -> None:
        start.wait(timeout=5)
        engine.prepare_close(deadline_seconds=1.0)
        engine.close()

    threads = [threading.Thread(target=end_pipeline) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    # Every thread prepares before closing, so the first phase always runs first.
    assert calls == ["prepare", "close"]
