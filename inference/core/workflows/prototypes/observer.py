"""Observation hooks the host may bind to watch workflow execution.

Billing and tracing are host concerns: an inference server bills workflow runs
and parents its spans, a standalone workflows process does neither. The engine
therefore reports what it is doing through this port and never knows which of
those two it is talking to.

Three hooks *wrap* the work rather than bracketing it. That is deliberate: a
host's usage accounting is typically a decorator that must call the function
itself - it times the call, records on both the success and the exception
path, and reads what it records off the call's own arguments. A context
manager cannot delegate the call, so bracketing hooks would force the host to
restructure its accounting rather than reuse it.

Step context is split in two because the engine runs steps on a
``ThreadPoolExecutor``: ``capture_step_context()`` runs in the thread that
submits the work, ``step_scope()`` in the worker that executes it. The opaque
token in between is whatever the host needs to carry across that boundary.
Note this is the host's *second* layer: the engine already re-enters a
snapshot of the submitting thread's whole ``contextvars`` context per task
(``execution_engine/v1/executor/utils.py``). Hosts that need an explicit
attach/detach - OpenTelemetry does - use these hooks; hosts that do not can
return ``None`` and yield.
"""

from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from inference.core.workflows.execution_engine.v1.compiler.entities import (
        CompiledWorkflow,
    )

T = TypeVar("T")


@runtime_checkable
class ExecutionObserver(Protocol):
    """What the execution engine reports, and the host may act on."""

    def observe_workflow_run(
        self,
        *,
        workflow: "CompiledWorkflow",
        runtime_parameters: Dict[str, Any],
        workflow_id: Optional[str],
        fps: float,
        is_preview: bool,
        run: Callable[[], T],
    ) -> T:
        """Wrap one whole workflow run.

        The compiled workflow is passed whole rather than as extracted fields:
        a host that bills runs identifies the workflow from its definition and
        its init parameters, and re-deriving that here would move the host's
        rules into the engine.
        """
        ...

    def capture_step_context(self) -> Any:
        """Snapshot, in the submitting thread, whatever the workers need."""
        ...

    def step_scope(self, *, context: Any, step_name: str) -> ContextManager[None]:
        """Establish that snapshot inside a worker thread, for one step."""
        ...

    def observe_block_run(
        self,
        *,
        block: Any,
        block_args: Tuple[Any, ...],
        block_kwargs: Dict[str, Any],
        run: Callable[[], T],
    ) -> T:
        """Wrap one invocation of an assembled custom-Python block.

        ``block_kwargs`` stays a nested mapping: the block's parameter names
        come from the workflow definition unvalidated, so spreading them would
        let an input collide with a host-reserved argument name.
        """
        ...

    def observe_model_run(
        self,
        *,
        block: Any,
        model_id: Optional[str],
        images: Any,
        run: Callable[[], T],
    ) -> T:
        """Wrap a block's own model call - blocks that load a model directly
        rather than through the models provider."""
        ...


class NullExecutionObserver:
    """Observes nothing. The default, so workflows runs with no host bound."""

    def observe_workflow_run(
        self,
        *,
        workflow: "CompiledWorkflow",
        runtime_parameters: Dict[str, Any],
        workflow_id: Optional[str],
        fps: float,
        is_preview: bool,
        run: Callable[[], T],
    ) -> T:
        return run()

    def capture_step_context(self) -> Any:
        return None

    @contextmanager
    def step_scope(self, *, context: Any, step_name: str) -> Iterator[None]:
        yield

    def observe_block_run(
        self,
        *,
        block: Any,
        block_args: Tuple[Any, ...],
        block_kwargs: Dict[str, Any],
        run: Callable[[], T],
    ) -> T:
        return run()

    def observe_model_run(
        self,
        *,
        block: Any,
        model_id: Optional[str],
        images: Any,
        run: Callable[[], T],
    ) -> T:
        return run()


# A shared instance rather than a class: the steps initialiser calls anything
# callable it finds in the initializer registry (`call_if_callable`), and an
# instance is not callable, so it is handed to blocks as-is.
NULL_EXECUTION_OBSERVER = NullExecutionObserver()
