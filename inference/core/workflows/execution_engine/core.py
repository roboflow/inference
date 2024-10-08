from typing import Any, Dict, List, Optional, Type

from packaging.specifiers import SpecifierSet
from packaging.version import Version

from inference.core.workflows.errors import (
    NotSupportedExecutionEngineError,
    WorkflowDefinitionError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.entities.engine import (
    BaseExecutionEngine,
)
from inference.core.workflows.execution_engine.profiling.core import WorkflowsProfiler
from inference.core.workflows.execution_engine.v1.core import (
    EXECUTION_ENGINE_V1_VERSION,
    ExecutionEngineV1,
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
    ) -> List[Dict[str, Any]]:
        return self._engine.run(
            runtime_parameters=runtime_parameters,
            fps=fps,
            _is_preview=_is_preview,
        )


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
