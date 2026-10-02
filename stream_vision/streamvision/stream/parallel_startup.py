"""Owned, bounded startup transaction for the opt-in workflow stream path."""

import contextvars
import logging
import sys
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from threading import Event, Lock
from time import perf_counter
from typing import Any, Callable, Dict, Optional, Tuple

from roboflow_workflows.execution_engine.entities.base import WorkflowParameter
from roboflow_workflows.execution_engine.v1.compiler.core import (
    collect_input_substitutions,
)
from roboflow_workflows.execution_engine.v1.compiler.entities import (
    ParsedWorkflowDefinition,
)
from roboflow_workflows.execution_engine.v1.core import (
    _is_eligible_for_generic_preloading,
    _pre_load_roboflow_platform_models,
    _resolve_runtime_dependency_model_id,
    _retrieve_step_execution_mode,
    _verify_pre_loaded_models_presence,
)
from roboflow_workflows.execution_engine.v1.executor.runtime_input_assembler import (
    assemble_inference_parameter,
)
from roboflow_workflows.execution_engine.v1.executor.runtime_input_validator import (
    validate_runtime_input,
)
from streamvision.stream.session import stream_session_id

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _StartupResources:
    """Host-owned provider, optional capacity budget and teardown callback.

    Standalone callers supply their provider through workflow init parameters;
    only a host that owns that provider should supply a cleanup callback.
    """

    provider: Any
    available_capacity: Optional[int] = None
    cleanup: Optional[Callable[[], None]] = None


class _ReadyModelManager:
    """Retain one manager; block compiler/custom-block use during preparation."""

    def __init__(self, startup):
        self._startup = startup

    def __getattr__(self, name):
        self._startup._wait_models()
        return getattr(self._startup.manager, name)

    def __bool__(self):
        self._startup._wait_models()
        return bool(self._startup.manager)

    def __contains__(self, key):
        self._startup._wait_models()
        return key in self._startup.manager

    def __getitem__(self, key):
        self._startup._wait_models()
        return self._startup.manager[key]

    def __len__(self):
        self._startup._wait_models()
        return len(self._startup.manager)

    def __iter__(self):
        self._startup._wait_models()
        return iter(self._startup.manager)


class _ParallelWorkflowStartup:
    def __init__(
        self,
        *,
        pipeline: Any,
        resources: _StartupResources,
        cancelled: Optional[Event],
        model_limit: int,
        report: Callable[[str, dict], None],
    ):
        self.pipeline = pipeline
        self.manager = resources.provider
        self._resources = resources
        self.cancelled = cancelled if cancelled is not None else Event()
        self._failed = Event()
        self._failure = None
        self._failure_lock = Lock()
        self._model_limit = model_limit
        self._report = report
        self._pool = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="workflow-startup"
        )
        self._models: Optional[Future] = None
        self._sources: Optional[Future] = None
        self._attempted_sources = []
        self._closed = False
        self._transferred = False
        self._close_lock = Lock()
        self.timings = {}
        self._timings_lock = Lock()
        self._origin = perf_counter()
        # Copy the caller's selected CUDA device, without importing/initializing
        # CUDA merely to start a CPU pipeline. Native backends retain their usual
        # primary context. The preparation stream is synchronized before transfer.
        self._torch = sys.modules.get("torch")
        self._cuda_device = None
        if self._torch is not None and self._torch.cuda.is_initialized():
            self._cuda_device = self._torch.cuda.current_device()
        self.gated_manager = _ReadyModelManager(self)

    def _check_cancelled(self):
        if self.cancelled.is_set() or self._failed.is_set():
            raise CancelledError("Parallel workflow startup cancelled")

    def _phase(self, name: str, operation: Callable[[], Any]) -> Any:
        session = getattr(self.pipeline, "_stream_session_id", None)
        token = stream_session_id.set(session) if session is not None else None
        started = perf_counter() - self._origin
        outcome = "failed"
        try:
            self._check_cancelled()
            result = operation()
            self._check_cancelled()
            outcome = "ready"
            return result
        except BaseException as error:
            with self._failure_lock:
                if self._failure is None:
                    self._failure = error
            self._failed.set()
            raise
        finally:
            if token is not None:
                stream_session_id.reset(token)
            finished = perf_counter() - self._origin
            timing = {
                "start_seconds": started,
                "end_seconds": finished,
                "duration_seconds": finished - started,
                "outcome": outcome,
            }
            with self._timings_lock:
                self.timings[name] = timing
            try:
                self._report(name, timing)
            except Exception:
                logger.warning("Could not report workflow startup phase")

    def _submit(self, name: str, operation: Callable[[], Any]) -> Future:
        context = contextvars.copy_context()
        return self._pool.submit(context.run, self._phase, name, operation)

    def _start_sources(self):
        def start():
            for source in self.pipeline._video_sources:
                self._check_cancelled()
                self._attempted_sources.append(source)
                device = (
                    self._torch.cuda.device(self._cuda_device)
                    if self._cuda_device is not None
                    else nullcontext()
                )
                with device:
                    source.start()
                self.pipeline._started_sources.append(source)

        self._sources = self._submit("sources", start)

    def _on_workflow_parsed(
        self,
        definition: ParsedWorkflowDefinition,
        *,
        init_parameters: Dict[str, Any],
        api_key: Optional[str],
        workflows_parameters: Optional[Dict[str, Any]] = None,
        frame_input_names: Tuple[str, ...] = (),
    ) -> None:
        self._check_cancelled()
        if self._models is not None:
            raise RuntimeError("Dependencies may only be prepared once")

        # Only ordinary fixed parameters can be known without a frame. Use the
        # same defaults, manifest substitution validation and ID resolvers as run.
        parameters = {
            item.name: assemble_inference_parameter(
                parameter=item.name,
                runtime_parameters=workflows_parameters or {},
                default_value=item.default_value,
            )
            for item in definition.inputs
            if isinstance(item, WorkflowParameter)
            and item.name not in frame_input_names
        }
        substitutions = [
            item
            for item in collect_input_substitutions(definition)
            if item.input_parameter_name in parameters
            and parameters[item.input_parameter_name] is not None
        ]
        validate_runtime_input(
            runtime_parameters=parameters, input_substitutions=substitutions
        )
        mode = _retrieve_step_execution_mode(init_parameters=init_parameters)
        dependencies, seen = [], set()
        capacity = self._resources.available_capacity
        limit = self._model_limit
        if isinstance(capacity, int):
            limit = min(limit, max(0, capacity))
        for manifest in definition.steps:
            for dependency in manifest.discover_dependent_resources() or []:
                if not _is_eligible_for_generic_preloading(dependency, mode):
                    continue
                metadata = dependency.metadata
                if metadata.requires_runtime_resolution():
                    resolved = _resolve_runtime_dependency_model_id(
                        dependency, parameters
                    )
                    if resolved is None:
                        continue
                    metadata = metadata.model_copy(update={"model_id": resolved})
                    dependency = dependency.model_copy(update={"metadata": metadata})
                model_id = metadata.model_id
                if len(dependencies) >= limit:
                    continue
                if model_id in seen or model_id in self.manager:
                    continue
                seen.add(model_id)
                dependencies.append(dependency)

        def prepare():
            device = (
                self._torch.cuda.device(self._cuda_device)
                if self._cuda_device is not None
                else nullcontext()
            )
            with device:
                try:
                    for dependency in dependencies:
                        self._check_cancelled()
                        _pre_load_roboflow_platform_models(
                            dependencies=[dependency],
                            model_manager=self.manager,
                            api_key=api_key,
                            step_execution_mode=mode,
                        )
                    _verify_pre_loaded_models_presence(
                        model_manager=self.manager, expected_model_ids=seen
                    )
                finally:
                    torch = sys.modules.get("torch")
                    if (
                        dependencies
                        and torch is not None
                        and torch.cuda.is_initialized()
                    ):
                        torch.cuda.synchronize()

        self._models = self._submit("models", prepare)

    def _wait_models(self):
        if self._models is not None:
            self._models.result()
        if not self._transferred:
            self._check_cancelled()

    def _finish(self):
        # Do not use Future.cancel as an ownership barrier: running native work
        # cannot be interrupted. Always drain both workers before transfer/cleanup.
        self._pool.shutdown(wait=True)
        if self._failure is not None:
            raise self._failure
        for future in (self._sources, self._models):
            if future is not None:
                future.result()
        self._check_cancelled()
        # After transfer, the pipeline checks cancellation at start/frame boundaries.
        # Keep provider access available to flush/shutdown handlers during teardown.
        self._transferred = True
        self.pipeline._parallel_startup = self
        self.pipeline._sources_started_during_init = True
        self.pipeline.startup_phase_timings = dict(self.timings)
        self.pipeline._sources_startup_finished.set()

    def _stop_sources_locked(self):
        errors, remaining_sources = [], []
        for source in self._attempted_sources:
            try:
                source.terminate(
                    wait_on_frames_consumption=False, purge_frames_buffer=True
                )
                if source in self.pipeline._started_sources:
                    self.pipeline._started_sources.remove(source)
            except Exception as error:
                remaining_sources.append(source)
                errors.append(error)
        self._attempted_sources = remaining_sources
        return errors

    def _stop_sources(self):
        # terminate() and join() may be called by different lifecycle threads.
        with self._close_lock:
            errors = self._stop_sources_locked()
            if errors:
                raise RuntimeError(
                    "Parallel startup source cleanup failed"
                ) from errors[0]

    def _close(self):
        # Hold the lock through draining: concurrent callers must not report
        # cleanup complete while another caller still owns native startup work.
        with self._close_lock:
            if self._closed:
                return
            self._failed.set()
            self._pool.shutdown(wait=True)
            errors = self._stop_sources_locked()
            if self._resources.cleanup is not None:
                try:
                    self._resources.cleanup()
                except Exception as error:
                    errors.append(error)
            if errors:
                # Keep the owner and failed resources available for a retry.
                raise RuntimeError(
                    "Parallel startup resource cleanup failed"
                ) from errors[0]

            self.pipeline._started_sources.clear()
            self._closed = True
