"""The server's binding of the workflows `ExecutionObserver`.

Workflows declares what it is doing; this module is where "what it is doing"
becomes a usage row and an OpenTelemetry span. Nothing here re-implements
either: the three billed hooks call `usage_collector` - the real decorator,
with the real extraction - and the tracing hooks call the real telemetry
helpers. `test_workflows_observer_row_parity.py` compares the rows these
produce against the rows the decorated entry points produced before the port,
under a frozen clock.

The three decorated functions below exist for one reason: the collector reads
a row's fields off the *decorated function's own parameter names*
(`collect_func_params`). So each shim declares exactly the names its category's
extractor looks for, and calls the engine's continuation in its body. That is
also why they are module-level: the collector memoizes signatures keyed by
function object, and a per-call closure would pin one entry per workflow run.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, TypeVar

from inference.core.telemetry import (
    attach_context,
    capture_context,
    detach_context,
    start_span,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    consume_block_duration,
)
from inference.usage_tracking.block_execution import (
    clear_measured_block_execution,
    record_measured_block_execution,
)
from inference.usage_tracking.collector import usage_collector
from inference.usage_tracking.stream_session import stream_session_id

T = TypeVar("T")


@usage_collector("workflows")
def _billed_workflow_run(workflow, runtime_parameters, run):
    """One whole workflow run, billed and traced.

    `workflow` and `runtime_parameters` are declared so the collector can read
    the workflow's api key, its step list and the run's image source off them -
    exactly the names the engine's `run_workflow` used to expose.
    """
    with start_span("workflow.run"):
        return run()


@usage_collector("workflow_block")
def _billed_block_run(self, block_args, block_kwargs, run):
    """One custom-Python block invocation, billed.

    `self` is the block; the collector reads its usage resource id, api key and
    step metadata. `block_kwargs` stays nested so a workflow-authored input
    cannot bind to one of the decorator's own keyword arguments, and so batch
    inputs stay countable.

    The duration the engine measured is relayed in a `finally`, before the
    decorator extracts the row, so a block that raised is still billed for the
    time it actually ran.
    """
    try:
        return run()
    finally:
        _relay_measured_block_duration()


@usage_collector("model")
def _billed_model_run(self, model_id, images, run):
    """A block's own model call, billed as a model row.

    `model_id` and `images` are the two names the model extractor reads; the
    api key comes off the block.
    """
    return run()


def _relay_measured_block_duration() -> None:
    """Hand the engine's measurement to the collector's channel.

    The host channel is cleared first, unconditionally: a measurement left
    behind by an invocation whose usage recording failed must not be billed to
    this one.
    """
    measured = consume_block_duration()
    clear_measured_block_execution()
    if measured is None:
        return
    record_measured_block_execution(duration=measured.duration, source=measured.source)


@dataclass(frozen=True)
class ServerStepContext:
    """What a step worker thread needs from the thread that submitted it.

    The engine already re-enters a snapshot of the submitting thread's whole
    `contextvars` context per task, so both fields normally arrive anyway; this
    is the explicit second layer, and OpenTelemetry needs it because its
    context must be attached and detached, not merely present. Both fields are
    re-bound in the worker including when they are None: pool threads are
    reused across requests.
    """

    otel_context: Any
    stream_session_id: Optional[str]


class UsageTrackingExecutionObserver:
    """Binds workflow execution to usage tracking and OpenTelemetry."""

    def observe_workflow_run(
        self,
        *,
        workflow: Any,
        runtime_parameters: Dict[str, Any],
        workflow_id: Optional[str],
        fps: float,
        is_preview: bool,
        run: Callable[[], T],
    ) -> T:
        return _billed_workflow_run(
            workflow=workflow,
            runtime_parameters=runtime_parameters,
            run=run,
            usage_fps=fps,
            usage_workflow_id=workflow_id or "",
            usage_workflow_preview=is_preview,
        )

    def capture_step_context(self) -> ServerStepContext:
        return ServerStepContext(
            otel_context=capture_context(),
            stream_session_id=stream_session_id.get(),
        )

    @contextmanager
    def step_scope(self, *, context: Any, step_name: str) -> Iterator[None]:
        stream_session_id.set(
            context.stream_session_id if context is not None else None
        )
        otel_token = attach_context(
            context.otel_context if context is not None else None
        )
        try:
            with start_span("workflow.step", {"workflow.step": step_name}):
                yield
        finally:
            detach_context(otel_token)

    def observe_block_run(
        self,
        *,
        block: Any,
        block_args: Tuple[Any, ...],
        block_kwargs: Dict[str, Any],
        run: Callable[[], T],
    ) -> T:
        return _billed_block_run(
            self=block, block_args=block_args, block_kwargs=block_kwargs, run=run
        )

    def observe_model_run(
        self,
        *,
        block: Any,
        model_id: Optional[str],
        images: Any,
        run: Callable[[], T],
    ) -> T:
        return _billed_model_run(self=block, model_id=model_id, images=images, run=run)
