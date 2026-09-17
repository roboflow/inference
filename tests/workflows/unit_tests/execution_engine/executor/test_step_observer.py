"""The engine's half of the capture/attach split, with no host bound.

The engine already re-enters a snapshot of the submitting thread's whole
`contextvars` context per task (`executor/utils.py`), so a host usually needs
nothing extra. A host that must *attach and detach* something - OpenTelemetry -
does, and the engine owns *when* that happens. These tests pin the when, with a
fake observer; nothing here imports the server.
"""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import get_ident
from unittest import mock

from inference.core.workflows.execution_engine.v1.executor import core


class RecordingObserver:
    def __init__(self):
        self.captured_in = None
        self.scopes = []
        self.capture_count = 0

    def observe_workflow_run(
        self, *, workflow, runtime_parameters, workflow_id, fps, is_preview, run
    ):
        return run()

    def capture_step_context(self):
        self.capture_count += 1
        self.captured_in = get_ident()
        return {"token": self.capture_count}

    @contextmanager
    def step_scope(self, *, context, step_name):
        self.scopes.append((get_ident(), context, step_name))
        yield

    def observe_block_run(self, *, block, block_args, block_kwargs, run):
        return run()

    def observe_model_run(self, *, block, model_id, images, run):
        return run()


@mock.patch.object(core, "run_step")
def test_step_context_is_captured_once_in_the_caller_and_entered_per_step(
    run_step_mock,
) -> None:
    # given
    observer = RecordingObserver()

    # when
    with ThreadPoolExecutor(max_workers=2) as executor:
        core.execute_steps(
            next_steps=["$steps.first", "$steps.second"],
            workflow=mock.MagicMock(),
            execution_data_manager=mock.MagicMock(),
            max_concurrent_steps=2,
            workflow_execution_id="exec-1",
            executor=executor,
            observer=observer,
        )

    # then - one snapshot, taken here; one scope per step, entered elsewhere
    assert observer.capture_count == 1
    assert observer.captured_in == get_ident()
    assert sorted(step_name for _, _, step_name in observer.scopes) == [
        "first",
        "second",
    ]
    assert all(context == {"token": 1} for _, context, _ in observer.scopes)
    assert all(thread_id != get_ident() for thread_id, _, _ in observer.scopes)


@mock.patch.object(core, "run_step")
def test_step_scope_wraps_the_step_and_sees_its_failure(run_step_mock) -> None:
    # given
    observer = RecordingObserver()
    exits = []

    @contextmanager
    def recording_scope(*, context, step_name):
        try:
            yield
        finally:
            exits.append(step_name)

    observer.step_scope = recording_scope
    run_step_mock.side_effect = RuntimeError("block failed")
    workflow = mock.MagicMock()
    workflow.steps = {"only": mock.MagicMock()}

    # when
    try:
        core.safe_execute_step(
            step_selector="$steps.only",
            workflow=workflow,
            execution_data_manager=mock.MagicMock(),
            observer=observer,
            step_context={"token": 1},
        )
    except Exception:
        pass

    # then
    assert exits == ["only"]


@mock.patch.object(core, "run_step")
def test_the_default_observer_needs_no_binding(run_step_mock) -> None:
    # given - a direct engine caller with no host: must not raise
    # when
    core.safe_execute_step(
        step_selector="$steps.only",
        workflow=mock.MagicMock(),
        execution_data_manager=mock.MagicMock(),
    )

    # then
    run_step_mock.assert_called_once()
