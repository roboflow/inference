"""Workflow-related HTTP routes (describe, run, blocks, validate)."""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Query, Request
from fastapi.responses import Response
from concurrent.futures import ThreadPoolExecutor

from inference.core import logger
from inference.core.entities.requests.workflows import (
    DescribeBlocksRequest,
    PredefinedWorkflowDescribeInterfaceRequest,
    PredefinedWorkflowInferenceRequest,
    WorkflowSpecificationDescribeInterfaceRequest,
    WorkflowSpecificationInferenceRequest,
)
from inference.core.entities.responses.workflows import (
    DescribeInterfaceResponse,
    ExecutionEngineVersions,
    WorkflowInferenceResponse,
    WorkflowsBlocksDescription,
    WorkflowsBlocksSchemaDescription,
    WorkflowValidationStatus,
)
from inference.core.env import (
    ENABLE_WORKFLOWS_PROFILING,
    GCP_SERVERLESS,
    LAMBDA,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
    WORKFLOWS_STEP_EXECUTION_MODE,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.handlers.workflows import (
    filter_out_unwanted_workflow_outputs,
    handle_describe_workflows_blocks_request,
    handle_describe_workflows_interface,
)
from inference.core.interfaces.http.middlewares.gzip import gzip_response_if_requested
from inference.core.interfaces.http.orjson_utils import orjson_response
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import (
    ExecutionEngine,
    get_available_versions,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
    WorkflowsProfiler,
)
from inference.core.workflows.execution_engine.v1.compiler.syntactic_parser import (
    get_workflow_schema_description,
    parse_workflow_definition,
)
from inference.usage_tracking.collector import usage_collector


def create_workflows_router(
    model_manager: ModelManager,
    shared_thread_pool_executor: Optional[ThreadPoolExecutor],
) -> APIRouter:
    router = APIRouter()

    def process_workflow_inference_request(
        workflow_request,
        workflow_specification: dict,
        background_tasks: Optional[BackgroundTasks],
        profiler: WorkflowsProfiler,
    ) -> WorkflowInferenceResponse:
        workflow_init_parameters = {
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": workflow_request.api_key,
            "workflows_core.background_tasks": background_tasks,
        }
        execution_engine = ExecutionEngine.init(
            workflow_definition=workflow_specification,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            prevent_local_images_loading=True,
            profiler=profiler,
            executor=shared_thread_pool_executor,
            workflow_id=workflow_request.workflow_id,
        )
        is_preview = False
        if hasattr(workflow_request, "is_preview"):
            is_preview = workflow_request.is_preview
        workflow_results = execution_engine.run(
            runtime_parameters=workflow_request.inputs,
            serialize_results=True,
            _is_preview=is_preview,
        )
        with profiler.profile_execution_phase(
            name="workflow_results_filtering",
            categories=["inference_package_operation"],
        ):
            outputs = filter_out_unwanted_workflow_outputs(
                workflow_results=workflow_results,
                excluded_fields=workflow_request.excluded_fields,
            )
        profiler_trace = profiler.export_trace()
        response = WorkflowInferenceResponse(
            outputs=outputs,
            profiler_trace=profiler_trace,
        )
        return orjson_response(response=response)

    @router.post(
        "/{workspace_name}/workflows/{workflow_id}/describe_interface",
        response_model=DescribeInterfaceResponse,
        summary="Endpoint to describe interface of predefined workflow",
        description="Checks Roboflow API for workflow definition, once acquired - describes workflow inputs and outputs",
    )
    @with_route_exceptions
    def describe_predefined_workflow_interface(
        workspace_name: str,
        workflow_id: str,
        workflow_request: PredefinedWorkflowDescribeInterfaceRequest,
    ) -> DescribeInterfaceResponse:
        workflow_specification = get_workflow_specification(
            api_key=workflow_request.api_key,
            workspace_id=workspace_name,
            workflow_id=workflow_id,
            use_cache=workflow_request.use_cache,
            workflow_version_id=workflow_request.workflow_version_id,
        )
        return handle_describe_workflows_interface(
            definition=workflow_specification,
        )

    @router.post(
        "/workflows/describe_interface",
        response_model=DescribeInterfaceResponse,
        summary="Endpoint to describe interface of workflow given in request",
        description="Parses workflow definition and retrieves describes inputs and outputs",
    )
    @with_route_exceptions
    def describe_workflow_interface(
        workflow_request: WorkflowSpecificationDescribeInterfaceRequest,
    ) -> DescribeInterfaceResponse:
        return handle_describe_workflows_interface(
            definition=workflow_request.specification,
        )

    @router.post(
        "/{workspace_name}/workflows/{workflow_id}",
        response_model=WorkflowInferenceResponse,
        summary="Endpoint to run predefined workflow",
        description="Checks Roboflow API for workflow definition, once acquired - parses and executes injecting runtime parameters from request body",
    )
    @router.post(
        "/infer/workflows/{workspace_name}/{workflow_id}",
        response_model=WorkflowInferenceResponse,
        summary="[LEGACY] Endpoint to run predefined workflow",
        description="Checks Roboflow API for workflow definition, once acquired - parses and executes injecting runtime parameters from request body. This endpoint is deprecated and will be removed end of Q2 2024",
        deprecated=True,
    )
    @with_route_exceptions
    @usage_collector("request")
    def infer_from_predefined_workflow(
        workspace_name: str,
        workflow_id: str,
        workflow_request: PredefinedWorkflowInferenceRequest,
        background_tasks: BackgroundTasks,
    ) -> WorkflowInferenceResponse:
        if ENABLE_WORKFLOWS_PROFILING and workflow_request.enable_profiling:
            profiler = BaseWorkflowsProfiler.init(
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
            )
        else:
            profiler = NullWorkflowsProfiler.init()
        with profiler.profile_execution_phase(
            name="workflow_definition_fetching",
            categories=["inference_package_operation"],
        ):
            workflow_specification = get_workflow_specification(
                api_key=workflow_request.api_key,
                workspace_id=workspace_name,
                workflow_id=workflow_id,
                use_cache=workflow_request.use_cache,
                workflow_version_id=workflow_request.workflow_version_id,
            )
        if not workflow_request.workflow_id:
            workflow_request.workflow_id = workflow_id
        if not workflow_specification.get("id"):
            logger.warning(
                "Internal workflow ID missing in specification for '%s'",
                workflow_id,
            )
        return process_workflow_inference_request(
            workflow_request=workflow_request,
            workflow_specification=workflow_specification,
            background_tasks=(
                background_tasks if not (LAMBDA or GCP_SERVERLESS) else None
            ),
            profiler=profiler,
        )

    @router.post(
        "/workflows/run",
        response_model=WorkflowInferenceResponse,
        summary="Endpoint to run workflow specification provided in payload",
        description="Parses and executes workflow specification, injecting runtime parameters from request body.",
    )
    @router.post(
        "/infer/workflows",
        response_model=WorkflowInferenceResponse,
        summary="[LEGACY] Endpoint to run workflow specification provided in payload",
        description="Parses and executes workflow specification, injecting runtime parameters from request body. This endpoint is deprecated and will be removed end of Q2 2024.",
        deprecated=True,
    )
    @with_route_exceptions
    @usage_collector("request")
    def infer_from_workflow(
        workflow_request: WorkflowSpecificationInferenceRequest,
        background_tasks: BackgroundTasks,
    ) -> WorkflowInferenceResponse:
        if ENABLE_WORKFLOWS_PROFILING and workflow_request.enable_profiling:
            profiler = BaseWorkflowsProfiler.init(
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
            )
        else:
            profiler = NullWorkflowsProfiler.init()
        return process_workflow_inference_request(
            workflow_request=workflow_request,
            workflow_specification=workflow_request.specification,
            background_tasks=(
                background_tasks if not (LAMBDA or GCP_SERVERLESS) else None
            ),
            profiler=profiler,
        )

    @router.get(
        "/workflows/execution_engine/versions",
        response_model=ExecutionEngineVersions,
        summary="Returns available Execution Engine versions sorted from oldest to newest",
        description="Returns available Execution Engine versions sorted from oldest to newest",
    )
    @with_route_exceptions
    def get_execution_engine_versions() -> ExecutionEngineVersions:
        versions = get_available_versions()
        return ExecutionEngineVersions(versions=versions)

    @router.get(
        "/workflows/blocks/describe",
        response_model=WorkflowsBlocksDescription,
        summary="[LEGACY] Endpoint to get definition of workflows blocks that are accessible",
        description="Endpoint provides detailed information about workflows building blocks that are "
        "accessible in the inference server. This information could be used to programmatically "
        "build / display workflows.",
        deprecated=True,
    )
    @with_route_exceptions
    def describe_workflows_blocks_get(
        request: Request,
    ) -> Union[WorkflowsBlocksDescription, Response]:
        result = handle_describe_workflows_blocks_request()
        return gzip_response_if_requested(request=request, response=result)

    @router.post(
        "/workflows/blocks/describe",
        response_model=WorkflowsBlocksDescription,
        summary="[EXPERIMENTAL] Endpoint to get definition of workflows blocks that are accessible",
        description="Endpoint provides detailed information about workflows building blocks that are "
        "accessible in the inference server. This information could be used to programmatically "
        "build / display workflows. Additionally - in request body one can specify list of "
        "dynamic blocks definitions which will be transformed into blocks and used to generate "
        "schemas and definitions of connections",
    )
    @with_route_exceptions
    def describe_workflows_blocks_post(
        request: Request,
        request_payload: Optional[DescribeBlocksRequest] = None,
    ) -> Union[WorkflowsBlocksDescription, Response]:
        dynamic_blocks_definitions = None
        requested_execution_engine_version = None
        api_key = None
        if request_payload is not None:
            dynamic_blocks_definitions = (
                request_payload.dynamic_blocks_definitions
            )
            requested_execution_engine_version = (
                request_payload.execution_engine_version
            )
            api_key = request_payload.api_key or request.query_params.get(
                "api_key", None
            )
        result = handle_describe_workflows_blocks_request(
            dynamic_blocks_definitions=dynamic_blocks_definitions,
            requested_execution_engine_version=requested_execution_engine_version,
            api_key=api_key,
        )
        return gzip_response_if_requested(request=request, response=result)

    @router.get(
        "/workflows/definition/schema",
        response_model=WorkflowsBlocksSchemaDescription,
        summary="Endpoint to fetch the workflows block schema",
        description="Endpoint to fetch the schema of all available blocks. This information can be "
        "used to validate workflow definitions and suggest syntax in the JSON editor.",
    )
    @with_route_exceptions
    def get_workflow_schema(
        request: Request,
    ) -> WorkflowsBlocksSchemaDescription:
        result = get_workflow_schema_description()
        return gzip_response_if_requested(request, response=result)

    @router.post(
        "/workflows/blocks/dynamic_outputs",
        response_model=List[OutputDefinition],
        summary="[EXPERIMENTAL] Endpoint to get definition of dynamic output for workflow step",
        description="Endpoint to be used when step outputs can be discovered only after "
        "filling manifest with data.",
    )
    @with_route_exceptions
    def get_dynamic_block_outputs(
        step_manifest: Dict[str, Any],
    ) -> List[OutputDefinition]:
        dummy_workflow_definition = {
            "version": "1.0",
            "inputs": [],
            "steps": [step_manifest],
            "outputs": [],
        }
        available_blocks = load_workflow_blocks()
        parsed_definition = parse_workflow_definition(
            raw_workflow_definition=dummy_workflow_definition,
            available_blocks=available_blocks,
        )
        parsed_manifest = parsed_definition.steps[0]
        return parsed_manifest.get_actual_outputs()

    @router.post(
        "/workflows/validate",
        response_model=WorkflowValidationStatus,
        summary="[EXPERIMENTAL] Endpoint to validate",
        description="Endpoint provides a way to check validity of JSON workflow definition.",
    )
    @with_route_exceptions
    def validate_workflow(
        specification: dict,
        api_key: Optional[str] = Query(
            None,
            description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
        ),
    ) -> WorkflowValidationStatus:
        step_execution_mode = StepExecutionMode(WORKFLOWS_STEP_EXECUTION_MODE)
        workflow_init_parameters = {
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": api_key,
            "workflows_core.background_tasks": None,
            "workflows_core.step_execution_mode": step_execution_mode,
        }
        _ = ExecutionEngine.init(
            workflow_definition=specification,
            init_parameters=workflow_init_parameters,
            max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
            prevent_local_images_loading=True,
        )
        return WorkflowValidationStatus(status="ok")

    return router
