"""Observers of a workflow run recording its usage rows.

``UsageExecutionObserver`` records the ``workflows``, ``workflow_block`` and
``model`` rows of a run on the scope bound by the request being served;
``StreamUsageExecutionObserver`` owns the scope of a stream pipeline and binds
it around every run.
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, TypeVar

from roboflow_workflows.execution_engine.entities.base import Batch
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    consume_block_duration,
)
from roboflow_workflows.prototypes.observer import NULL_EXECUTION_OBSERVER

from inference_server.usage.rows import (
    USAGE_SCOPE,
    UsageScope,
    bound_scope,
    bound_stream_session,
    bound_workflow_preview,
    current_stream_session_id,
    record_block_usage,
    record_model_usage,
    record_workflow_usage,
)

logger = logging.getLogger(__name__)

CUSTOM_PYTHON_BLOCK_KIND = "custom_python"

T = TypeVar("T")


def request_observer() -> Any:
    """Observer of the workflow runs of the request being served.

    Returns:
        A ``UsageExecutionObserver`` when the request binds a usage scope, or
        the null observer when no usage row is being recorded for it.
    """
    if USAGE_SCOPE.get() is None:
        return NULL_EXECUTION_OBSERVER

    observer = UsageExecutionObserver()

    return observer


class UsageExecutionObserver:
    """Record the rows of a workflow run on the scope of the current context.

    Every hook reads the ``UsageScope`` bound where it runs: the request's
    handler binds it, and the engine re-enters the submitting thread's
    context in every step worker. Model runs of providers reach the scope
    through the request's bridge, which carries it. A bookkeeping failure
    never reaches the run.
    """

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
        """Run the workflow and record its ``workflows`` row.

        The preview flag is published to the custom Python block rows of the
        run. A run that raises is recorded with its error and re-raised.

        Args:
            workflow: Compiled workflow; its definition and API key attribute
                the row.
            runtime_parameters: Inputs of the run, unused.
            workflow_id: Identifier the engine derived; the hash of the step
                list attributes the row when None.
            fps: Frames per second of the source the run processes.
            is_preview: Whether the run is a preview.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        scope = USAGE_SCOPE.get()
        if scope is None:
            return run()
        started = time.perf_counter()
        with bound_workflow_preview(is_preview):
            try:
                result = run()
            except Exception as error:
                self._record_workflow_run(
                    scope,
                    workflow,
                    workflow_id=workflow_id,
                    fps=fps,
                    is_preview=is_preview,
                    duration=time.perf_counter() - started,
                    error=error,
                )
                raise
        self._record_workflow_run(
            scope,
            workflow,
            workflow_id=workflow_id,
            fps=fps,
            is_preview=is_preview,
            duration=time.perf_counter() - started,
            error=None,
        )

        return result

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
        """Run a block and, for a custom Python block, record its row.

        The duration is the one the engine measured for this invocation, else
        the wall time of ``run``. A block that raises is still recorded.

        Args:
            block: The block instance.
            block_args: Positional inputs of the block, unused.
            block_kwargs: Keyword inputs of the block; the largest batch among
                them is the frame count of the row.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        if not _is_custom_python_block(block):
            return run()
        started = time.perf_counter()
        try:
            result = run()
        except Exception as error:
            self._record_block_run(
                block, block_kwargs, time.perf_counter() - started, error
            )
            raise
        self._record_block_run(block, block_kwargs, time.perf_counter() - started, None)

        return result

    def observe_model_run(
        self,
        *,
        block: Any,
        model_id: Optional[str],
        images: Any,
        run: Callable[[], T],
    ) -> T:
        """Run a block's own model call and record its ``model`` row.

        Args:
            block: The block instance; its API key attributes the row.
            model_id: Identifier of the model the block runs.
            images: Images handed to the model; one frame per element.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        started = time.perf_counter()
        try:
            result = run()
        except Exception as error:
            self._record_model_run(
                block, model_id, images, time.perf_counter() - started, error
            )
            raise
        self._record_model_run(
            block, model_id, images, time.perf_counter() - started, None
        )

        return result

    def _record_workflow_run(
        self,
        scope: UsageScope,
        workflow: Any,
        *,
        workflow_id: Optional[str],
        fps: float,
        is_preview: bool,
        duration: float,
        error: Optional[Exception],
    ) -> None:
        try:
            record_workflow_usage(
                scope,
                workflow=workflow,
                workflow_id=workflow_id,
                fps=fps,
                is_preview=is_preview,
                duration=duration,
                error=error,
            )
        except Exception as failure:
            logger.debug(
                "Usage of the workflow run was not recorded: %s",
                type(failure).__name__,
            )

    def _record_block_run(
        self,
        block: Any,
        block_kwargs: Dict[str, Any],
        wall_duration: float,
        error: Optional[Exception],
    ) -> None:
        try:
            measured = consume_block_duration()
            scope = USAGE_SCOPE.get()
            if scope is None:
                return
            record_block_usage(
                scope,
                block=block,
                frames=_block_frames(block_kwargs),
                duration=wall_duration,
                measured=measured,
                error=error,
            )
        except Exception as failure:
            logger.debug(
                "Custom Python run was not recorded: %s", type(failure).__name__
            )

    def _record_model_run(
        self,
        block: Any,
        model_id: Optional[str],
        images: Any,
        duration: float,
        error: Optional[Exception],
    ) -> None:
        try:
            scope = USAGE_SCOPE.get()
            if scope is None:
                return
            record_model_usage(
                scope,
                model_id=model_id,
                api_key=getattr(block, "_api_key", None),
                frames=_frames(images),
                duration=duration,
                details={},
                megapixel_buckets=None,
                error=error,
            )
        except Exception as failure:
            logger.debug("Model run was not recorded: %s", type(failure).__name__)


@dataclass(frozen=True)
class StreamStepContext:
    """What a step worker thread of a pipeline run needs from the run's thread.

    Args:
        scope: Usage scope of the pipeline.
        stream_session_id: Session of the pipeline the run belongs to.
    """

    scope: UsageScope
    stream_session_id: Optional[str]


class StreamUsageExecutionObserver(UsageExecutionObserver):
    """Record the rows of every workflow run of a stream pipeline.

    The pipeline process serves no request, so the observer owns the usage
    scope: it binds it around every run and inside the step worker threads,
    and the pipeline host binds it under ``scope_binding`` so the bridge built
    for the pipeline captures it. The stream session of the run is kept on
    the scope so rows recorded from the bridge's loop thread carry it too.
    """

    def __init__(self, collector: Any, *, api_key: Optional[str] = None) -> None:
        """Bind the observer to the collector of the pipeline process.

        Args:
            collector: ``UsageCollector`` the rows are recorded on.
            api_key: Key of the pipeline; a row without a key of its own is
                attributed to it.
        """
        self._scope = UsageScope(collector=collector, api_key=api_key)

    @property
    def scope(self) -> UsageScope:
        """Usage scope of the pipeline."""
        return self._scope

    @contextmanager
    def scope_binding(self) -> Iterator[None]:
        """Bind the pipeline's scope in the current context.

        Yields:
            Nothing; the scope is unbound again on exit.
        """
        with bound_scope(self._scope):
            yield

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
        """Run the workflow under the pipeline's scope and record its row.

        Args:
            workflow: Compiled workflow; its definition and API key attribute
                the row.
            runtime_parameters: Inputs of the run, unused.
            workflow_id: Identifier the engine derived.
            fps: Frames per second of the source the run processes.
            is_preview: Whether the run is a preview.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        self._scope.stream_session_id = current_stream_session_id()
        with bound_scope(self._scope):
            result = super().observe_workflow_run(
                workflow=workflow,
                runtime_parameters=runtime_parameters,
                workflow_id=workflow_id,
                fps=fps,
                is_preview=is_preview,
                run=run,
            )

        return result

    def capture_step_context(self) -> StreamStepContext:
        """Snapshot the scope and stream session for a worker thread.

        Returns:
            The context ``step_scope`` re-binds in the worker.
        """
        context = StreamStepContext(
            scope=self._scope,
            stream_session_id=current_stream_session_id(),
        )

        return context

    @contextmanager
    def step_scope(self, *, context: Any, step_name: str) -> Iterator[None]:
        """Bind the scope and stream session inside a worker thread.

        Args:
            context: Value ``capture_step_context`` returned; nothing is bound
                when ``None``.
            step_name: Name of the step, unused.
        """
        if context is None:
            yield
            return

        with bound_scope(context.scope):
            with bound_stream_session(context.stream_session_id):
                yield


def _is_custom_python_block(block: Any) -> bool:
    try:
        kind = getattr(block, "_usage_block_kind", None)
    except Exception:
        return False

    return kind == CUSTOM_PYTHON_BLOCK_KIND


def _frames(images: Any) -> int:
    if isinstance(images, (list, tuple, Batch)):
        return max(1, len(images))

    return 1


def _batch_elements(value: Any) -> int:
    if not isinstance(value, Batch):
        return 1

    elements = sum(_batch_elements(element) for element in value)

    return elements


def _block_frames(block_kwargs: Any) -> int:
    if not isinstance(block_kwargs, dict):
        return 1
    batch_sizes = [
        _batch_elements(value)
        for value in block_kwargs.values()
        if isinstance(value, Batch)
    ]
    if not batch_sizes:
        return 1

    frames = max(max(batch_sizes), 1)

    return frames
