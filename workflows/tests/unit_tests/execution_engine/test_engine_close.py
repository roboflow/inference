import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.engine import BaseExecutionEngine
from roboflow_workflows.execution_engine.v1.executor.core import close_workflow_blocks


class ClosableBlock:
    def __init__(self, fail: bool = False):
        self.close_calls = 0
        self.fail = fail

    def close(self) -> None:
        self.close_calls += 1
        if self.fail:
            raise RuntimeError("cannot close")


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


def test_compiled_engine_closes_its_blocks_and_tolerates_repeated_close() -> None:
    definition = {
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
    }
    engine = ExecutionEngine.init(workflow_definition=definition, init_parameters={})
    block = engine._engine._compiled_workflow.steps["cached"].step
    calls = []
    original_close = block.close
    block.close = lambda: calls.append("close") or original_close()

    engine.close()
    engine.close()

    assert calls == ["close", "close"]
