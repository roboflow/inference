from typing import Any, Dict, List, Optional

from packaging.version import Version

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

EXECUTION_ENGINE_V1_VERSION = Version("1.1.2")


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
    ) -> "ExecutionEngineV1":
        if init_parameters is None:
            init_parameters = {}
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
        )

    def __init__(
        self,
        compiled_workflow: CompiledWorkflow,
        max_concurrent_steps: int,
        prevent_local_images_loading: bool,
        profiler: WorkflowsProfiler,
        workflow_id: Optional[str] = None,
    ):
        self._compiled_workflow = compiled_workflow
        self._max_concurrent_steps = max_concurrent_steps
        self._prevent_local_images_loading = prevent_local_images_loading
        self._workflow_id = workflow_id
        self._profiler = profiler

    def run(
        self,
        runtime_parameters: Dict[str, Any],
        fps: float = 0,
        _is_preview: bool = False,
    ) -> List[Dict[str, Any]]:
        self._profiler.start_workflow_run()
        runtime_parameters = assemble_runtime_parameters(
            runtime_parameters=runtime_parameters,
            defined_inputs=self._compiled_workflow.workflow_definition.inputs,
            prevent_local_images_loading=self._prevent_local_images_loading,
            profiler=self._profiler,
        )
        validate_runtime_input(
            runtime_parameters=runtime_parameters,
            input_substitutions=self._compiled_workflow.input_substitutions,
            profiler=self._profiler,
        )
        result = run_workflow(
            workflow=self._compiled_workflow,
            runtime_parameters=runtime_parameters,
            max_concurrent_steps=self._max_concurrent_steps,
            usage_fps=fps,
            usage_workflow_id=self._workflow_id,
            usage_workflow_preview=_is_preview,
            profiler=self._profiler,
        )
        self._profiler.end_workflow_run()
        return result
