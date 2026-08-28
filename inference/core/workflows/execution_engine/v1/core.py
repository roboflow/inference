import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Set, Union

from packaging.version import Version

from inference.core.env import WORKFLOWS_STEP_EXECUTION_MODE
from inference.core.logger import logger
from inference.core.workflows.errors import (
    RuntimeInputError,
    WorkflowEnvironmentConfigurationError,
)
from inference.core.workflows.execution_engine.entities.engine import (
    BaseExecutionEngine,
)
from inference.core.workflows.execution_engine.profiling.core import (
    NullWorkflowsProfiler,
    WorkflowsProfiler,
)
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)
from inference.core.workflows.execution_engine.v1.compiler.utils import (
    deduce_blocks_dependencies,
    get_last_chunk_of_selector,
    is_input_selector,
)
from inference.core.workflows.execution_engine.v1.executor.core import (
    flush_stream_pipeline_workflow,
    run_workflow,
)
from inference.core.workflows.execution_engine.v1.executor.runtime_input_assembler import (
    assemble_runtime_parameters,
)
from inference.core.workflows.execution_engine.v1.executor.runtime_input_validator import (
    validate_runtime_input,
)
from inference.core.workflows.execution_engine.v1.step_error_handlers import (
    extended_roboflow_errors_handler,
    legacy_step_error_handler,
)
from inference.core.workflows.prototypes.block import (
    DependentResource,
    DependentResourceType,
    ModelExecutionLocation,
    ModelRequiredAction,
    StepExecutionMode,
    is_workflow_selector,
)

EXECUTION_ENGINE_V1_VERSION = Version("1.15.1")

DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER = os.getenv(
    "DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER", "extended_roboflow_errors"
)

REGISTERED_STEP_ERROR_HANDLERS = {
    "legacy": legacy_step_error_handler,
    "extended_roboflow_errors": extended_roboflow_errors_handler,
}

PRE_INIT_SUPPORTED_DEPENDENCIES = {DependentResourceType.ROBOFLOW_PLATFORM_MODEL}


def _parse_dependencies_pre_init(
    dependencies_pre_init: List[str],
) -> Set[DependentResourceType]:
    supported_values = [t.value for t in PRE_INIT_SUPPORTED_DEPENDENCIES]
    parsed = set()
    for raw_value in dependencies_pre_init:
        try:
            resource_type = DependentResourceType(str(raw_value).replace("-", "_"))
        except ValueError:
            raise WorkflowEnvironmentConfigurationError(
                public_message=f"`dependencies_pre_init` contains unrecognised value "
                f"'{raw_value}'. Supported values: {supported_values}.",
                context="workflow_compilation | engine_initialisation",
            )
        if resource_type not in PRE_INIT_SUPPORTED_DEPENDENCIES:
            raise WorkflowEnvironmentConfigurationError(
                public_message=f"`dependencies_pre_init` value '{raw_value}' is not "
                f"supported for pre-loading. Supported values: {supported_values}.",
                context="workflow_compilation | engine_initialisation",
            )
        parsed.add(resource_type)
    return parsed


def _retrieve_init_parameter(
    init_parameters: Dict[str, Any], parameter_name: str
) -> Optional[Any]:
    for key in (f"workflows_core.{parameter_name}", parameter_name):
        if key in init_parameters:
            value = init_parameters[key]
            return value() if callable(value) else value
    return None


def _retrieve_step_execution_mode(
    init_parameters: Dict[str, Any],
) -> StepExecutionMode:
    # Mirrors what blocks receive from the steps initialiser: an explicit
    # init parameter wins, otherwise the environment default applies.
    value = _retrieve_init_parameter(
        init_parameters=init_parameters, parameter_name="step_execution_mode"
    )
    if value is None:
        value = WORKFLOWS_STEP_EXECUTION_MODE
    if isinstance(value, StepExecutionMode):
        return value
    return StepExecutionMode(value)


def _is_locally_executed_platform_model(
    dependency: DependentResource,
    step_execution_mode: StepExecutionMode,
) -> bool:
    """True for declarations that will pull model weights into this process.

    Excludes non-model resources, ACCESS-only usage, remote-only execution,
    and `ENVIRONMENT_DEFINED` execution when the effective step execution
    mode is not LOCAL.
    """
    if dependency.resource_type is not DependentResourceType.ROBOFLOW_PLATFORM_MODEL:
        return False
    metadata = dependency.metadata
    if metadata.required_action is not ModelRequiredAction.EXECUTION:
        return False
    if metadata.execution_location is ModelExecutionLocation.REMOTE:
        return False
    if (
        metadata.execution_location is ModelExecutionLocation.ENVIRONMENT_DEFINED
        and step_execution_mode is not StepExecutionMode.LOCAL
    ):
        return False
    return True


def _verify_pre_loaded_models_presence(
    model_manager: Any, expected_model_ids: Set[str]
) -> None:
    # Registration happens sequentially without capacity reservation — a
    # size/memory-bounded model manager may evict earlier entries while
    # loading later ones. Eviction is not an error (models lazily re-load at
    # execution time), but it silently defeats the purpose of pre-loading, so
    # it deserves a warning.
    missing_model_ids = [
        model_id
        for model_id in sorted(expected_model_ids)
        if model_id not in model_manager
    ]
    if missing_model_ids:
        logger.warning(
            "Pre-loading of workflow dependencies registered models %s, but they "
            "are no longer present in the model manager — most likely evicted by "
            "the manager's size or memory limits while subsequent models were "
            "loading. The workflow will lazily re-load them at execution time.",
            missing_model_ids,
        )


def _pre_load_roboflow_platform_models(
    dependencies: List[DependentResource],
    model_manager: Any,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> List[DependentResource]:
    """Registers concrete Roboflow platform models in the model manager.

    Returns dependencies whose model id is an `$inputs.<name>` selector — they
    can only be resolved on the first run, once runtime parameters are known.
    `$steps.<name>.<property>` references are dropped (never statically
    resolvable) and so are declarations that will not pull weights locally
    (see `_is_locally_executed_platform_model`).
    """
    pending, loaded_model_ids = [], set()
    for dependency in dependencies:
        if not _is_locally_executed_platform_model(
            dependency=dependency, step_execution_mode=step_execution_mode
        ):
            continue
        metadata = dependency.metadata
        if metadata.requires_runtime_resolution():
            if is_input_selector(selector_or_value=metadata.model_id):
                pending.append(dependency)
            continue
        if metadata.model_id in loaded_model_ids:
            continue
        loaded_model_ids.add(metadata.model_id)
        model_manager.add_model(
            model_id=metadata.model_id,
            api_key=api_key,
            **(metadata.model_registration_kwargs or {}),
        )
    if loaded_model_ids:
        _verify_pre_loaded_models_presence(
            model_manager=model_manager, expected_model_ids=loaded_model_ids
        )
    return pending


def _resolve_and_pre_load_runtime_dependencies(
    pending_dependencies: List[DependentResource],
    runtime_parameters: Dict[str, Any],
    model_manager: Any,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> None:
    loaded_model_ids = set()
    for dependency in pending_dependencies:
        if not _is_locally_executed_platform_model(
            dependency=dependency, step_execution_mode=step_execution_mode
        ):
            continue
        if not is_input_selector(selector_or_value=dependency.metadata.model_id):
            # Safeguard: only `$inputs.<name>` references are resolvable here.
            # Without it, a `$steps.<name>.<property>` entry would take its
            # last chunk and could accidentally match an unrelated input.
            continue
        input_name = get_last_chunk_of_selector(selector=dependency.metadata.model_id)
        resolved_value = runtime_parameters.get(input_name)
        if not isinstance(resolved_value, str) or not resolved_value:
            continue
        if is_workflow_selector(resolved_value):
            continue
        model_id_resolver = dependency.metadata.model_id_resolver
        if model_id_resolver is not None:
            # Declarations of synthesized ids (e.g. `clip/<version>`) attach a
            # resolver turning the substituted input value into the final id.
            try:
                resolved_value = model_id_resolver(resolved_value)
            except Exception as error:
                raise RuntimeInputError(
                    public_message=f"Could not resolve model id of dependent resource "
                    f"declared as `{dependency.metadata.model_id}` while pre-loading "
                    f"workflow dependencies - value `{resolved_value}` submitted for "
                    f"input `{input_name}` is invalid. Details: {error}",
                    context="workflow_execution | runtime_input_validation",
                    inner_error=error,
                ) from error
            if resolved_value is None:
                # Resolver declared the value statically unresolvable (the
                # final id depends on more than this one input) — skip
                # pre-loading and let execution resolve it.
                continue
        if resolved_value in loaded_model_ids:
            continue
        loaded_model_ids.add(resolved_value)
        model_manager.add_model(
            model_id=resolved_value,
            api_key=api_key,
            **(dependency.metadata.model_registration_kwargs or {}),
        )
    if loaded_model_ids:
        _verify_pre_loaded_models_presence(
            model_manager=model_manager, expected_model_ids=loaded_model_ids
        )


class ExecutionEngineV1(BaseExecutionEngine):

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
            Union[str, Callable[[str, Exception], None]]
        ] = DEFAULT_WORKFLOWS_STEP_ERROR_HANDLER,
        dependencies_pre_init: Optional[List[str]] = None,
    ) -> "ExecutionEngineV1":
        if init_parameters is None:
            init_parameters = {}
        if isinstance(step_error_handler, str):
            if step_error_handler not in REGISTERED_STEP_ERROR_HANDLERS:
                raise WorkflowEnvironmentConfigurationError(
                    public_message=f"Execution engine was initialised with step_error_handler='{step_error_handler}' "
                    f"which is not registered. Supported values: "
                    f"{list(REGISTERED_STEP_ERROR_HANDLERS.keys())}",
                    context="workflow_compilation | engine_initialisation",
                )
            step_error_handler = REGISTERED_STEP_ERROR_HANDLERS[step_error_handler]
        init_parameters["dynamic_workflows_blocks.api_key"] = init_parameters.get(
            "dynamic_workflows_blocks.api_key",
            init_parameters.get("workflows_core.api_key"),
        )

        if profiler is None:
            profiler = NullWorkflowsProfiler.init()
        compiled_workflow = compile_workflow(
            workflow_definition=workflow_definition,
            init_parameters=init_parameters,
            execution_engine_version=EXECUTION_ENGINE_V1_VERSION,
            profiler=profiler,
        )
        pre_init_dependencies_types = (
            _parse_dependencies_pre_init(dependencies_pre_init=dependencies_pre_init)
            if dependencies_pre_init
            else set()
        )
        pending_runtime_dependencies: List[DependentResource] = []
        pre_init_model_manager, pre_init_api_key = None, None
        pre_init_step_execution_mode = None
        if DependentResourceType.ROBOFLOW_PLATFORM_MODEL in pre_init_dependencies_types:
            pre_init_model_manager = _retrieve_init_parameter(
                init_parameters=init_parameters, parameter_name="model_manager"
            )
            if pre_init_model_manager is None:
                raise WorkflowEnvironmentConfigurationError(
                    public_message="`dependencies_pre_init` requested pre-loading of "
                    "Roboflow platform models, but `model_manager` cannot be found "
                    "in `init_parameters`.",
                    context="workflow_compilation | engine_initialisation",
                )
            pre_init_api_key = _retrieve_init_parameter(
                init_parameters=init_parameters, parameter_name="api_key"
            )
            pre_init_step_execution_mode = _retrieve_step_execution_mode(
                init_parameters=init_parameters
            )
            dependencies = deduce_blocks_dependencies(
                compiled_workflow=compiled_workflow
            )
            pending_runtime_dependencies = _pre_load_roboflow_platform_models(
                dependencies=dependencies,
                model_manager=pre_init_model_manager,
                api_key=pre_init_api_key,
                step_execution_mode=pre_init_step_execution_mode,
            )
        return cls(
            compiled_workflow=compiled_workflow,
            max_concurrent_steps=max_concurrent_steps,
            prevent_local_images_loading=prevent_local_images_loading,
            profiler=profiler,
            workflow_id=workflow_id,
            internal_id=workflow_definition.get("id"),
            executor=executor,
            step_error_handler=step_error_handler,
            pending_runtime_dependencies=pending_runtime_dependencies,
            pre_init_model_manager=pre_init_model_manager,
            pre_init_api_key=pre_init_api_key,
            pre_init_step_execution_mode=pre_init_step_execution_mode,
        )

    def __init__(
        self,
        compiled_workflow: CompiledWorkflow,
        max_concurrent_steps: int,
        prevent_local_images_loading: bool,
        profiler: WorkflowsProfiler,
        workflow_id: Optional[str] = None,
        internal_id: Optional[str] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        step_error_handler: Optional[Callable[[str, Exception], None]] = None,
        pending_runtime_dependencies: Optional[List[DependentResource]] = None,
        pre_init_model_manager: Optional[Any] = None,
        pre_init_api_key: Optional[str] = None,
        pre_init_step_execution_mode: Optional[StepExecutionMode] = None,
    ):
        self._compiled_workflow = compiled_workflow
        self._max_concurrent_steps = max_concurrent_steps
        self._prevent_local_images_loading = prevent_local_images_loading
        self._workflow_id = workflow_id
        self._profiler = profiler
        self._internal_id = internal_id
        self._executor = executor
        self._step_error_handler = step_error_handler
        self._pending_runtime_dependencies = pending_runtime_dependencies or []
        self._pre_init_model_manager = pre_init_model_manager
        self._pre_init_api_key = pre_init_api_key
        self._pre_init_step_execution_mode = pre_init_step_execution_mode
        self._pending_dependencies_resolution_attempted = False

    def run(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
        _is_preview: bool = False,
        serialize_results: bool = False,
        defer_stream_pipeline_flush: bool = False,
        resolve_output_futures: bool = True,
    ) -> List[Dict[str, Any]]:
        self._profiler.start_workflow_run()
        runtime_parameters = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=self._compiled_workflow.workflow_definition.inputs,
            kinds_deserializers=self._compiled_workflow.kinds_deserializers,
            prevent_local_images_loading=self._prevent_local_images_loading,
            profiler=self._profiler,
        )
        validate_runtime_input(
            runtime_parameters=runtime_parameters,
            input_substitutions=self._compiled_workflow.input_substitutions,
            profiler=self._profiler,
        )
        if (
            self._pending_runtime_dependencies
            and not self._pending_dependencies_resolution_attempted
        ):
            self._pending_dependencies_resolution_attempted = True
            _resolve_and_pre_load_runtime_dependencies(
                pending_dependencies=self._pending_runtime_dependencies,
                runtime_parameters=runtime_parameters,
                model_manager=self._pre_init_model_manager,
                api_key=self._pre_init_api_key,
                step_execution_mode=self._pre_init_step_execution_mode,
            )
        usage_workflow_id = self._internal_id
        if self._workflow_id and not usage_workflow_id:
            logger.debug(
                "Workflow ID is set to '%s' however internal Workflow ID is missing",
                self._workflow_id,
            )
            usage_workflow_id = self._workflow_id
        result = run_workflow(
            workflow=self._compiled_workflow,
            runtime_parameters=runtime_parameters,
            max_concurrent_steps=self._max_concurrent_steps,
            usage_fps=fps,
            usage_workflow_id=usage_workflow_id,
            usage_workflow_preview=_is_preview,
            kinds_serializers=self._compiled_workflow.kinds_serializers,
            serialize_results=serialize_results,
            profiler=self._profiler,
            executor=self._executor,
            step_error_handler=self._step_error_handler,
            defer_stream_pipeline_flush=defer_stream_pipeline_flush,
            resolve_output_futures=resolve_output_futures,
        )
        self._profiler.end_workflow_run()
        return result

    def flush_stream_pipeline(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
        _is_preview: bool = False,
        serialize_results: bool = False,
    ) -> List[Dict[str, Any]]:
        self._profiler.start_workflow_run()
        runtime_parameters = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=self._compiled_workflow.workflow_definition.inputs,
            kinds_deserializers=self._compiled_workflow.kinds_deserializers,
            prevent_local_images_loading=self._prevent_local_images_loading,
            profiler=self._profiler,
        )
        validate_runtime_input(
            runtime_parameters=runtime_parameters,
            input_substitutions=self._compiled_workflow.input_substitutions,
            profiler=self._profiler,
        )
        result = flush_stream_pipeline_workflow(
            workflow=self._compiled_workflow,
            runtime_parameters=runtime_parameters,
            max_concurrent_steps=self._max_concurrent_steps,
            kinds_serializers=self._compiled_workflow.kinds_serializers,
            serialize_results=serialize_results,
            profiler=self._profiler,
            executor=self._executor,
            step_error_handler=self._step_error_handler,
        )
        self._profiler.end_workflow_run()
        return result
