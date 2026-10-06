"""Observers of a workflow run feeding a usage row.

``UsageExecutionObserver`` feeds the row of the HTTP request being served;
``StreamUsageExecutionObserver`` records a row of its own for every run of a
stream pipeline.
"""

import logging
import math
import numbers
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar

from roboflow_workflows.execution_engine.entities.base import Batch
from roboflow_workflows.execution_engine.v1.dynamic_blocks.block_duration import (
    consume_block_duration,
)
from roboflow_workflows.prototypes.observer import NULL_EXECUTION_OBSERVER

from inference_server import configuration
from inference_server.usage.request_hook import (
    CUSTOM_PYTHON_RUNS,
    MODEL_INVOCATIONS,
    REQUEST_CATEGORY,
    UNKNOWN_RESOURCE_ID,
    _exception_error_details,
    _execution_duration,
    _specification_resource_id,
    _workflow_steps,
    add_custom_python_run,
    add_model_invocation,
)

try:
    from streamvision.stream.session import stream_session_id
except ImportError:
    stream_session_id = None

logger = logging.getLogger(__name__)

CUSTOM_PYTHON_BLOCK_KIND = "custom_python"
WORKFLOW_API_KEY_PARAMETER = "workflows_core.api_key"

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


@dataclass(frozen=True)
class StreamStepContext:
    """What a step worker thread of a pipeline run needs from the run's thread.

    Args:
        models: ``models`` list of the run.
        custom_python: ``custom_python`` list of the run.
        stream_session_id: Session of the pipeline the run belongs to.
    """

    models: List[Dict[str, Any]]
    custom_python: List[Dict[str, Any]]
    stream_session_id: Optional[str]


class StreamUsageExecutionObserver(UsageExecutionObserver):
    """Record one ``request`` row per workflow run of a stream pipeline.

    The pipeline process serves no request, so the observer owns the row: it
    times every run, collects the model invocations and custom Python runs the
    run makes into lists it holds for the pipeline's lifetime, and records the
    row on the pipeline process's collector when the run returns or raises.
    The lists are bound to the holders the HTTP row uses for the duration of
    every run and under ``holders_scope`` so the bridge built for the
    pipeline captures them; the bridge, the step worker threads and this
    observer all append to the same lists. A bookkeeping failure never reaches
    the run.
    """

    def __init__(
        self,
        collector: Any,
        workflow_id: Optional[str] = None,
        specification: Optional[dict] = None,
    ) -> None:
        """Bind the observer to the collector of the pipeline process.

        Args:
            collector: ``UsageCollector`` the rows are recorded on.
            workflow_id: Identifier the client requested for a named workflow,
                ``None`` for an inline specification; every row is attributed
                to it, the way the HTTP rule does.
            specification: Definition hashed into the attribution when no
                ``workflow_id`` is given; the compiled workflow's definition
                applies when ``None``.
        """
        super().__init__(models=[], custom_python=[])
        self._collector = collector
        self._workflow_id = workflow_id
        self._specification = specification

    @contextmanager
    def holders_scope(self) -> Iterator[None]:
        """Bind the observer's lists to the holders of the current thread.

        Yields:
            Nothing; the holders are unbound again on exit.
        """
        with _bound_holders(self._models, self._custom_python):
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
        """Run the workflow and record its ``request`` row.

        Args:
            workflow: Compiled workflow; its definition and API key attribute
                the row.
            runtime_parameters: Inputs of the run, unused.
            workflow_id: Identifier the engine derived; ignored, the row is
                attributed by the inputs the observer was built with.
            fps: Frames per second of the source the run processes.
            is_preview: Whether the run is a preview.
            run: The engine's continuation.

        Returns:
            Whatever ``run`` returns.
        """
        self._models.clear()
        self._custom_python.clear()
        started = time.perf_counter()
        with _bound_holders(self._models, self._custom_python):
            try:
                result = run()
            except Exception as error:
                self._record_run(
                    workflow,
                    workflow_id=workflow_id,
                    fps=fps,
                    is_preview=is_preview,
                    duration=time.perf_counter() - started,
                    error=error,
                )
                raise
        self._record_run(
            workflow,
            workflow_id=workflow_id,
            fps=fps,
            is_preview=is_preview,
            duration=time.perf_counter() - started,
            error=None,
        )

        return result

    def capture_step_context(self) -> StreamStepContext:
        """Snapshot the run's lists and stream session for a worker thread.

        Returns:
            The context ``step_scope`` re-binds in the worker.
        """
        context = StreamStepContext(
            models=self._models,
            custom_python=self._custom_python,
            stream_session_id=_current_stream_session_id(),
        )

        return context

    @contextmanager
    def step_scope(self, *, context: Any, step_name: str) -> Iterator[None]:
        """Bind the run's lists and stream session inside a worker thread.

        Args:
            context: Value ``capture_step_context`` returned; nothing is bound
                when ``None``.
            step_name: Name of the step, unused.
        """
        if context is None:
            yield
            return

        with _bound_holders(context.models, context.custom_python):
            with _bound_stream_session(context.stream_session_id):
                yield

    def _record_run(
        self,
        workflow: Any,
        *,
        workflow_id: Optional[str],
        fps: float,
        is_preview: bool,
        duration: float,
        error: Optional[Exception],
    ) -> None:
        try:
            row = self._row(
                workflow,
                workflow_id=workflow_id,
                fps=fps,
                is_preview=is_preview,
                duration=duration,
                error=error,
            )
            self._collector.record_usage(**row)
        except Exception as failure:
            logger.debug(
                "Usage of the workflow run was not recorded: %s",
                type(failure).__name__,
            )

    def _row(
        self,
        workflow: Any,
        *,
        workflow_id: Optional[str],
        fps: float,
        is_preview: bool,
        duration: float,
        error: Optional[Exception],
    ) -> Dict[str, Any]:
        api_key = _workflow_api_key(workflow)
        specification = _workflow_json(workflow)
        details: Dict[str, Any] = {}
        if configuration.DEDICATED_DEPLOYMENT_ID:
            details["dedicated_deployment_id"] = configuration.DEDICATED_DEPLOYMENT_ID
        if configuration.DEVICE_ID:
            details["device_id"] = configuration.DEVICE_ID
        if specification is not None:
            details["steps"] = _workflow_steps(specification)
        details["is_preview"] = is_preview
        details["models"] = list(self._models)
        details["custom_python"] = list(self._custom_python)
        error_details: Dict[str, Any] = {}
        if error is not None:
            error_details = _exception_error_details(error, (api_key,))
        details.update(error_details)
        frames = 1
        if not _is_positive_fps(fps):
            fps = 0.0

        row = {
            "api_key": api_key or "",
            "category": REQUEST_CATEGORY,
            "resource_id": _run_resource_id(
                self._workflow_id,
                (
                    self._specification
                    if self._specification is not None
                    else specification
                ),
            ),
            "resource_details": details,
            "frames": frames,
            "execution_duration": _execution_duration(duration),
            "fps": fps,
            "source_duration": _source_duration(frames, fps),
            "billable": True,
            "is_preview": is_preview,
            "error_type": error_details.get("error_type"),
            "error_status_code": error_details.get("error_status_code"),
            "exec_session_id": _current_stream_session_id(),
        }

        return row


@contextmanager
def _bound_holders(
    models: List[Dict[str, Any]], custom_python: List[Dict[str, Any]]
) -> Iterator[None]:
    models_token = MODEL_INVOCATIONS.set(models)
    custom_python_token = CUSTOM_PYTHON_RUNS.set(custom_python)
    try:
        yield
    finally:
        CUSTOM_PYTHON_RUNS.reset(custom_python_token)
        MODEL_INVOCATIONS.reset(models_token)


@contextmanager
def _bound_stream_session(session_id: Optional[str]) -> Iterator[None]:
    if stream_session_id is None:
        yield
        return

    token = stream_session_id.set(session_id)
    try:
        yield
    finally:
        stream_session_id.reset(token)


def _current_stream_session_id() -> Optional[str]:
    if stream_session_id is None:
        return None

    session_id = stream_session_id.get()

    return session_id


def _workflow_api_key(workflow: Any) -> Optional[str]:
    init_parameters = getattr(workflow, "init_parameters", None)
    if not isinstance(init_parameters, dict):
        return None

    api_key = init_parameters.get(WORKFLOW_API_KEY_PARAMETER)

    return api_key


def _workflow_json(workflow: Any) -> Optional[dict]:
    workflow_json = getattr(workflow, "workflow_json", None)
    if not isinstance(workflow_json, dict):
        return None

    return workflow_json


def _run_resource_id(workflow_id: Optional[str], specification: Optional[dict]) -> str:
    if workflow_id:
        return str(workflow_id)
    if specification is not None:
        return _specification_resource_id(specification)

    return UNKNOWN_RESOURCE_ID


def _is_positive_fps(fps: Any) -> bool:
    return isinstance(fps, numbers.Real) and math.isfinite(fps) and fps > 0


def _source_duration(frames: int, fps: Any) -> float:
    if not _is_positive_fps(fps):
        return 0.0

    source_duration = frames / fps

    return source_duration


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
