from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Type, Union

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from roboflow_workflows.errors import (
    NotSupportedExecutionEngineError,
    WorkflowDefinitionError,
    WorkflowEnvironmentConfigurationError,
)
from roboflow_workflows.execution_engine.entities.engine import BaseExecutionEngine
from roboflow_workflows.execution_engine.profiling.core import WorkflowsProfiler
from roboflow_workflows.execution_engine.v1.core import (
    EXECUTION_ENGINE_V1_VERSION,
    OMITTED_STEP_ERROR_HANDLER,
    ExecutionEngineV1,
    _OmittedStepErrorHandler,
)

REGISTERED_ENGINES = {
    EXECUTION_ENGINE_V1_VERSION: ExecutionEngineV1,
}


def get_available_versions() -> List[str]:
    return [str(v) for v in sorted(REGISTERED_ENGINES.keys())]


class ExecutionEngine(BaseExecutionEngine):

    @classmethod
    def init(
        cls,
        workflow_definition: dict,
        init_parameters: Optional[Dict[str, Any]] = None,
        max_concurrent_steps: int = 1,
        prevent_local_images_loading: bool = False,
        workflow_id: Optional[str] = None,
        profiler: Optional[WorkflowsProfiler] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        step_error_handler: Optional[
            Union[str, Callable[[str, Exception], None], _OmittedStepErrorHandler]
        ] = OMITTED_STEP_ERROR_HANDLER,
        dependencies_pre_init: Optional[List[str]] = None,
    ) -> "ExecutionEngine":
        requested_engine_version = retrieve_requested_execution_engine_version(
            workflow_definition=workflow_definition,
        )
        engine_type = _select_execution_engine(
            requested_engine_version=requested_engine_version
        )
        engine = engine_type.init(
            workflow_definition=workflow_definition,
            init_parameters=init_parameters,
            max_concurrent_steps=max_concurrent_steps,
            prevent_local_images_loading=prevent_local_images_loading,
            workflow_id=workflow_id,
            profiler=profiler,
            executor=executor,
            step_error_handler=step_error_handler,
            dependencies_pre_init=dependencies_pre_init,
        )
        return cls(engine=engine)

    def __init__(
        self,
        engine: BaseExecutionEngine,
    ):
        self._engine = engine

    def run(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
        _is_preview: bool = False,
        serialize_results: bool = False,
        defer_stream_pipeline_flush: bool = False,
        resolve_output_futures: bool = True,
    ) -> List[Dict[str, Any]]:
        return self._engine.run(
            runtime_parameters=runtime_parameters,
            fps=fps,
            _is_preview=_is_preview,
            serialize_results=serialize_results,
            defer_stream_pipeline_flush=defer_stream_pipeline_flush,
            resolve_output_futures=resolve_output_futures,
        )

    def flush_stream_pipeline(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
        _is_preview: bool = False,
        serialize_results: bool = False,
    ) -> List[Dict[str, Any]]:
        return self._engine.flush_stream_pipeline(
            runtime_parameters=runtime_parameters,
            fps=fps,
            _is_preview=_is_preview,
            serialize_results=serialize_results,
        )

    def prepare_close(self, deadline_seconds: Optional[float] = None) -> None:
        """Start the first phase of closing: let blocks begin finishing their work.

        Calls ``prepare_close(deadline_seconds)`` on every block that defines one,
        for example so a video recorder stops ingesting frames and starts muxing
        pending clips. Hosts call it before shutting down their thread pools, then
        call ``close()`` after them. A failing block is logged and skipped. Safe to
        call more than once and from several threads; only the first call acts.

        Args:
            deadline_seconds: Seconds the host allows, from now, for the blocks to
                finish before ``close()``; ``None`` when the host sets no limit.
        """
        prepare_close = getattr(self._engine, "prepare_close", None)
        if callable(prepare_close):
            prepare_close(deadline_seconds)

    def close(self) -> None:
        """Release resources the workflow's blocks hold between runs.

        Calls ``close()`` on every block that defines one, such as the background
        connections of reader blocks. Safe to call more than once and from several
        threads: blocks are closed once, and a concurrent call returns after that
        close has finished. The engine must not be run afterwards.
        """
        self._engine.close()


def retrieve_requested_execution_engine_version(workflow_definition: dict) -> Version:
    raw_version = workflow_definition.get("version")
    if raw_version:
        try:
            return Version(raw_version)
        except (TypeError, ValueError) as e:
            raise WorkflowDefinitionError(
                public_message=f"Workflow definition contains `version` defined as `{raw_version}` which cannot be "
                f"parsed as valid version definition. Error details: {e}",
                inner_error=e,
                context="workflow_compilation | engine_initialisation",
            )
    if not REGISTERED_ENGINES:
        raise WorkflowEnvironmentConfigurationError(
            public_message="No Execution Engine versions registered to be used.",
            context="workflow_compilation | engine_initialisation",
        )
    return max(REGISTERED_ENGINES.keys())


def _select_execution_engine(
    requested_engine_version: Version,
) -> Type[BaseExecutionEngine]:
    requested_engine_version_specifier_set = _prepare_requested_version_specifier_set(
        requested_engine_version=requested_engine_version,
    )
    matching_versions = []
    for version in REGISTERED_ENGINES:
        if requested_engine_version_specifier_set.contains(version):
            matching_versions.append(version)
    if not matching_versions:
        raise NotSupportedExecutionEngineError(
            public_message=f"Workflow definition requested Execution Engine in version: "
            f"`{requested_engine_version_specifier_set}` which cannot be found in existing setup. "
            f"Available Execution Engines versions: `{list(REGISTERED_ENGINES.keys())}`.",
            context="workflow_compilation | engine_initialisation",
        )
    if len(matching_versions) > 1:
        raise WorkflowEnvironmentConfigurationError(
            public_message="Found multiple Execution Engines versions matching workflow definition "
            f"`{matching_versions}`. This indicates misconfiguration. "
            f"Please raise issue at https://github.com/roboflow/inference/issues",
            context="workflow_compilation | engine_initialisation",
        )
    return REGISTERED_ENGINES[matching_versions[0]]


def _prepare_requested_version_specifier_set(
    requested_engine_version: Version,
) -> SpecifierSet:
    next_major_version = requested_engine_version.major + 1
    return SpecifierSet(f">={requested_engine_version},<{next_major_version}.0.0")
