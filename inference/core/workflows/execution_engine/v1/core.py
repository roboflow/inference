from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union

from packaging.version import Version

from inference.core.logger import logger
from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
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
from inference.core.workflows.execution_engine.v1.executor.core import run_workflow
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

EXECUTION_ENGINE_V1_VERSION = Version("1.7.0")

REGISTERED_STEP_ERROR_HANDLERS = {
    "legacy": legacy_step_error_handler,
    "extended_roboflow_errors": extended_roboflow_errors_handler,
}


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
        ] = "extended_roboflow_errors",
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
        return cls(
            compiled_workflow=compiled_workflow,
            max_concurrent_steps=max_concurrent_steps,
            prevent_local_images_loading=prevent_local_images_loading,
            profiler=profiler,
            workflow_id=workflow_id,
            internal_id=workflow_definition.get("id"),
            executor=executor,
            step_error_handler=step_error_handler,
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
    ):
        self._compiled_workflow = compiled_workflow
        self._max_concurrent_steps = max_concurrent_steps
        self._prevent_local_images_loading = prevent_local_images_loading
        self._workflow_id = workflow_id
        self._profiler = profiler
        self._internal_id = internal_id
        self._executor = executor
        self._step_error_handler = step_error_handler

    def run(
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
        )
        self._profiler.end_workflow_run()
        return result
