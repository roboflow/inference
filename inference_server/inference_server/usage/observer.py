"""Request-scoped observer of a workflow run feeding the request's usage row."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

from roboflow_workflows.execution_engine.entities.base import Batch
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    consume_block_duration,
)
from roboflow_workflows.prototypes.observer import NULL_EXECUTION_OBSERVER

from inference_server.usage.request_hook import (
    CUSTOM_PYTHON_RUNS,
    MODEL_INVOCATIONS,
    UNKNOWN_RESOURCE_ID,
    _execution_duration,
    add_custom_python_run,
    add_model_invocation,
)

logger = logging.getLogger(__name__)

CUSTOM_PYTHON_BLOCK_KIND = "custom_python"

T = TypeVar("T")


def request_observer() -> Any:
    """Observer bound to the usage row of the request being served.

    Returns:
        A ``UsageExecutionObserver`` over the ``models`` and ``custom_python``
        lists of the request, or the null observer when no usage row is being
        recorded for the request.
    """
    models = MODEL_INVOCATIONS.get()
    custom_python = CUSTOM_PYTHON_RUNS.get()
    if models is None or custom_python is None:
        return NULL_EXECUTION_OBSERVER

    observer = UsageExecutionObserver(models=models, custom_python=custom_python)

    return observer


class UsageExecutionObserver:
    """Feed the usage row of a request with what its workflow run executes.

    Both lists are the objects captured when the request started, so every
    thread the engine runs steps on appends to the same row; model runs of
    providers reach the ``models`` list through the request's bridge, which
    carries it. Nothing here records a row of its own, and a bookkeeping
    failure never reaches the run.
    """

    def __init__(
        self,
        *,
        models: List[Dict[str, Any]],
        custom_python: List[Dict[str, Any]],
    ) -> None:
        """Bind the observer to the lists of one request.

        Args:
            models: ``models`` list of the request's row.
            custom_python: ``custom_python`` list of the request's row.
        """
        self._models = models
        self._custom_python = custom_python

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
        """Run the workflow unchanged.

        Args:
            workflow: Compiled workflow, unused.
            runtime_parameters: Inputs of the run, unused.
            workflow_id: Identifier of the workflow, unused.
            fps: Frames per second of the source, unused.
            is_preview: Whether the run is a preview, unused.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        return run()

    def capture_step_context(self) -> None:
        """Hand nothing to the step worker threads."""
        return None

    @contextmanager
    def step_scope(self, *, context: Any, step_name: str) -> Iterator[None]:
        """Enter a step's worker thread without binding anything.

        Args:
            context: Value ``capture_step_context`` returned, unused.
            step_name: Name of the step, unused.
        """
        yield

    def observe_block_run(
        self,
        *,
        block: Any,
        block_args: Tuple[Any, ...],
        block_kwargs: Dict[str, Any],
        run: Callable[[], T],
    ) -> T:
        """Run a block and, for a custom Python block, record its duration.

        The duration is the one the engine measured for this invocation, else
        the wall time of ``run``. A block that raises is still recorded.

        Args:
            block: The block instance.
            block_args: Positional inputs of the block, unused.
            block_kwargs: Keyword inputs of the block, unused.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        if not _is_custom_python_block(block):
            return run()
        started = time.perf_counter()
        try:
            return run()
        finally:
            self._record_block_run(block, time.perf_counter() - started)

    def observe_model_run(
        self,
        *,
        block: Any,
        model_id: Optional[str],
        images: Any,
        run: Callable[[], T],
    ) -> T:
        """Run a block's own model call and record it as a model invocation.

        Args:
            block: The block instance, unused.
            model_id: Identifier of the model the block runs.
            images: Images handed to the model; one entry per element.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        started = time.perf_counter()
        try:
            return run()
        finally:
            self._record_model_run(model_id, images, time.perf_counter() - started)

    def _record_block_run(self, block: Any, wall_duration: float) -> None:
        try:
            measured = consume_block_duration()
            duration = measured.duration if measured is not None else wall_duration
            entry: Dict[str, Any] = {"block_type": _block_type(block)}
            step_name = getattr(block, "_workflow_step_name", None)
            if step_name:
                entry["step_name"] = str(step_name)
            entry["execution_duration"] = _execution_duration(duration)
            add_custom_python_run(self._custom_python, entry)
        except Exception as failure:
            logger.debug(
                "Custom Python run was not recorded: %s", type(failure).__name__
            )

    def _record_model_run(
        self, model_id: Optional[str], images: Any, duration: float
    ) -> None:
        try:
            entry = {
                "model_id": _model_id(model_id),
                "frames": _frames(images),
                "execution_duration": duration,
            }
            add_model_invocation(self._models, entry)
        except Exception as failure:
            logger.debug("Model run was not recorded: %s", type(failure).__name__)


def _is_custom_python_block(block: Any) -> bool:
    try:
        kind = getattr(block, "_usage_block_kind", None)
    except Exception:
        return False

    return kind == CUSTOM_PYTHON_BLOCK_KIND


def _block_type(block: Any) -> str:
    block_type = getattr(block, "_workflow_step_type", None) or getattr(
        block, "_usage_block_type", None
    )

    return str(block_type)


def _model_id(model_id: Optional[str]) -> str:
    if model_id is None or not str(model_id).strip():
        return UNKNOWN_RESOURCE_ID

    return str(model_id).strip()


def _frames(images: Any) -> int:
    if isinstance(images, (list, tuple, Batch)):
        return max(1, len(images))

    return 1
