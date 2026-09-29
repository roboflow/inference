from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, Optional

from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
    WorkflowsProfiler,
)
from roboflow_workflows.execution_engine.v1.dynamic_blocks.debug_logs import (
    register_debug_session,
)
from roboflow_workflows.http_contract.describe import (
    filter_out_unwanted_workflow_outputs,
)
from roboflow_workflows.http_contract.entities import (
    WorkflowInferenceRequest,
    WorkflowInferenceResponse,
)
from roboflow_workflows.prototypes.observer import NULL_EXECUTION_OBSERVER

from inference_server import configuration
from inference_server.workflows import host


def make_profiler(enable_profiling: bool) -> WorkflowsProfiler:
    if configuration.ENABLE_WORKFLOWS_PROFILING and enable_profiling:
        return BaseWorkflowsProfiler.init(
            max_runs_in_buffer=configuration.WORKFLOWS_PROFILER_BUFFER_SIZE,
        )
    return NullWorkflowsProfiler.init()


def build_init_parameters(
    *,
    provider: Any,
    api_key: Optional[str],
    background_tasks: Any,
    disable_sinks: bool,
    inner_workflow_dispatch_depth: int,
    step_execution_mode: Optional[str] = None,
) -> Dict[str, Any]:
    init_parameters: Dict[str, Any] = {
        "workflows_core.model_manager": provider,
        "workflows_core.api_key": api_key,
        "workflows_core.background_tasks": background_tasks,
        "workflows_core.disable_sinks": disable_sinks,
        "workflows_core.inner_workflow_dispatch_depth": inner_workflow_dispatch_depth,
        "workflows_core.execution_observer": NULL_EXECUTION_OBSERVER,
        "workflows_core.configuration": host.SERVER_WORKFLOWS_CONFIGURATION,
        **host.workflows_platform_bindings(),
    }
    if step_execution_mode is not None:
        init_parameters["workflows_core.step_execution_mode"] = step_execution_mode
    host.bind_image_codec(init_parameters)
    return init_parameters


def run_workflow_sync(
    *,
    specification: dict,
    workflow_request: WorkflowInferenceRequest,
    init_parameters: Dict[str, Any],
    profiler: WorkflowsProfiler,
    executor: Optional[ThreadPoolExecutor],
    workflow_id: Optional[str],
    is_preview: bool,
    debug: bool,
) -> WorkflowInferenceResponse:
    execution_engine = ExecutionEngine.init(
        workflow_definition=specification,
        init_parameters=init_parameters,
        max_concurrent_steps=configuration.WORKFLOWS_MAX_CONCURRENT_STEPS,
        prevent_local_images_loading=True,
        profiler=profiler,
        executor=executor,
        workflow_id=workflow_id,
        step_error_handler=host.step_error_handler,
    )
    debug_context = register_debug_session() if debug else nullcontext()
    with debug_context as debug_session:
        try:
            workflow_results = execution_engine.run(
                runtime_parameters=workflow_request.inputs,
                serialize_results=True,
                _is_preview=is_preview,
            )
        except Exception as error:
            if debug_session is not None:
                logs_snapshot = debug_session.output_streams.snapshot()
                if logs_snapshot:
                    error.python_blocks_output_streams = logs_snapshot
                trace_entries = debug_session.debug_traces.snapshot()
                if trace_entries:
                    error.python_blocks_debug_traces = trace_entries
            raise
        python_blocks_output_streams = (
            debug_session.output_streams.snapshot()
            if debug and debug_session is not None
            else None
        ) or None
        python_blocks_debug_traces = (
            debug_session.debug_traces.snapshot()
            if debug and debug_session is not None
            else None
        ) or None
    with profiler.profile_execution_phase(
        name="workflow_results_filtering",
        categories=["inference_package_operation"],
    ):
        outputs = filter_out_unwanted_workflow_outputs(
            workflow_results=workflow_results,
            excluded_fields=workflow_request.excluded_fields,
        )
    return WorkflowInferenceResponse(
        outputs=outputs,
        profiler_trace=profiler.export_trace(),
        python_blocks_output_streams=python_blocks_output_streams,
        python_blocks_debug_traces=python_blocks_debug_traces,
    )
