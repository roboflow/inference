import gzip
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, Query, Request, Response
from pydantic import BaseModel
from roboflow_workflows.execution_engine.core import (
    ExecutionEngine,
    get_available_versions,
)
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)
from roboflow_workflows.execution_engine.v1.compiler.syntactic_parser import (
    get_workflow_schema,
    parse_workflow_definition,
)
from roboflow_workflows.http_contract.describe import (
    describe_workflow_interface,
    describe_workflows_blocks,
)
from roboflow_workflows.http_contract.entities import (
    DescribeBlocksRequest,
    DescribeInterfaceResponse,
    ExecutionEngineVersions,
    PredefinedWorkflowDescribeInterfaceRequest,
    PredefinedWorkflowInferenceRequest,
    WorkflowInferenceResponse,
    WorkflowsBlocksDescription,
    WorkflowsBlocksSchemaDescription,
    WorkflowSpecificationDescribeInterfaceRequest,
    WorkflowSpecificationInferenceRequest,
    WorkflowValidationStatus,
)
from starlette.concurrency import run_in_threadpool

from inference_server import configuration
from inference_server.legacy.bridge import SyncLegacyBridge
from inference_server.legacy.common import orjson_response, resolve_api_key
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.workflows import execution, host
from inference_server.workflows.errors import with_workflow_errors
from inference_server.workflows.models_provider import GatewayModelsProvider

router = APIRouter(tags=["workflows"])

MISSING_API_KEY_MESSAGE = (
    "Required Roboflow API key is missing. Pass it as the `api_key` field of the "
    "request payload or as the `Authorization: Bearer <api_key>` header."
)


def _models_provider(request: Request, api_key: Optional[str]) -> GatewayModelsProvider:
    bridge = SyncLegacyBridge(
        request.app.state.legacy_bridge, request.app.state.loop_bridge
    )
    return GatewayModelsProvider(bridge, api_key)


def _gzip_if_requested(
    request: Request, result: BaseModel
) -> Union[BaseModel, Response]:
    if "gzip" not in request.headers.get("Accept-Encoding", ""):
        return result
    response = Response(content=result.model_dump_json())
    response.body = gzip.compress(response.body)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Content-Length"] = str(len(response.body))
    return response


async def _run_workflow(
    request: Request,
    workflow_request,
    specification: dict,
    background_tasks: BackgroundTasks,
    api_key: Optional[str],
    profiler,
) -> Response:
    init_parameters = execution.build_init_parameters(
        provider=_models_provider(request, api_key),
        api_key=api_key,
        background_tasks=background_tasks,
        disable_sinks=workflow_request.disable_sinks,
        inner_workflow_dispatch_depth=workflow_request.inner_workflow_dispatch_depth,
    )
    result = await run_in_threadpool(
        execution.run_workflow_sync,
        specification=specification,
        workflow_request=workflow_request,
        init_parameters=init_parameters,
        profiler=profiler,
        executor=request.app.state.workflows_executor,
        workflow_id=workflow_request.workflow_id,
        is_preview=getattr(workflow_request, "is_preview", False),
        debug=workflow_request.debug,
    )
    return orjson_response(result)


@router.post(
    "/{workspace_name}/workflows/{workflow_id}/describe_interface",
    response_model=DescribeInterfaceResponse,
    summary="Endpoint to describe interface of predefined workflow",
    description="Checks Roboflow API for workflow definition, once acquired - describes workflow inputs and outputs",
)
@with_workflow_errors
async def describe_predefined_workflow_interface(
    request: Request,
    workspace_name: str,
    workflow_id: str,
    workflow_request: PredefinedWorkflowDescribeInterfaceRequest,
) -> DescribeInterfaceResponse:
    api_key = resolve_api_key(request, None, workflow_request.api_key)
    if api_key is None:
        raise LegacyHTTPError(400, MISSING_API_KEY_MESSAGE)
    specification = await run_in_threadpool(
        host.get_workflow_specification,
        api_key=api_key,
        workspace_id=workspace_name,
        workflow_id=workflow_id,
        use_cache=workflow_request.use_cache,
        workflow_version_id=workflow_request.workflow_version_id,
    )
    return await run_in_threadpool(
        describe_workflow_interface, definition=specification
    )


@router.post(
    "/workflows/describe_interface",
    response_model=DescribeInterfaceResponse,
    summary="Endpoint to describe interface of workflow given in request",
    description="Parses workflow definition and retrieves describes inputs and outputs",
)
@with_workflow_errors
async def describe_workflow_specification_interface(
    request: Request,
    workflow_request: WorkflowSpecificationDescribeInterfaceRequest,
) -> DescribeInterfaceResponse:
    api_key = resolve_api_key(request, None, workflow_request.api_key)
    if api_key is None:
        raise LegacyHTTPError(400, MISSING_API_KEY_MESSAGE)
    return await run_in_threadpool(
        describe_workflow_interface, definition=workflow_request.specification
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
@with_workflow_errors
async def infer_from_predefined_workflow(
    request: Request,
    workspace_name: str,
    workflow_id: str,
    workflow_request: PredefinedWorkflowInferenceRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    api_key = resolve_api_key(request, None, workflow_request.api_key)
    profiler = execution.make_profiler(workflow_request.enable_profiling)
    with profiler.profile_execution_phase(
        name="workflow_definition_fetching",
        categories=["inference_package_operation"],
    ):
        specification = await run_in_threadpool(
            host.get_workflow_specification,
            api_key=api_key,
            workspace_id=workspace_name,
            workflow_id=workflow_id,
            use_cache=workflow_request.use_cache,
            workflow_version_id=workflow_request.workflow_version_id,
        )
    if not workflow_request.workflow_id:
        workflow_request.workflow_id = workflow_id
    return await _run_workflow(
        request=request,
        workflow_request=workflow_request,
        specification=specification,
        background_tasks=background_tasks,
        api_key=api_key,
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
@with_workflow_errors
async def infer_from_workflow(
    request: Request,
    workflow_request: WorkflowSpecificationInferenceRequest,
    background_tasks: BackgroundTasks,
) -> Response:
    api_key = resolve_api_key(request, None, workflow_request.api_key)
    return await _run_workflow(
        request=request,
        workflow_request=workflow_request,
        specification=workflow_request.specification,
        background_tasks=background_tasks,
        api_key=api_key,
        profiler=execution.make_profiler(workflow_request.enable_profiling),
    )


@router.get(
    "/workflows/execution_engine/versions",
    response_model=ExecutionEngineVersions,
    summary="Returns available Execution Engine versions sorted from oldest to newest",
    description="Returns available Execution Engine versions sorted from oldest to newest",
)
@with_workflow_errors
async def get_execution_engine_versions() -> ExecutionEngineVersions:
    return ExecutionEngineVersions(versions=get_available_versions())


@router.get(
    "/workflows/blocks/describe",
    response_model=WorkflowsBlocksDescription,
    summary="[LEGACY] Endpoint to get definition of workflows blocks that are accessible",
    description="Endpoint provides detailed information about workflows building blocks that are "
    "accessible in the inference server. This information could be used to programmatically "
    "build / display workflows.",
    deprecated=True,
)
@with_workflow_errors
async def describe_blocks(
    request: Request,
    # NOTE: accepted for wire compatibility and ignored - the air-gapped builder is not ported.
    air_gapped: bool = Query(False),
) -> Union[WorkflowsBlocksDescription, Response]:
    result = await run_in_threadpool(
        describe_workflows_blocks,
        workspace_resolver=host.WORKSPACE_RESOLVER,
    )
    return _gzip_if_requested(request, result)


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
@with_workflow_errors
async def describe_blocks_with_dynamic_definitions(
    request: Request,
    request_payload: Optional[DescribeBlocksRequest] = None,
    # NOTE: accepted for wire compatibility and ignored - the air-gapped builder is not ported.
    air_gapped: bool = Query(False),
) -> Union[WorkflowsBlocksDescription, Response]:
    dynamic_blocks_definitions = None
    requested_execution_engine_version = None
    body_api_key = None
    if request_payload is not None:
        dynamic_blocks_definitions = request_payload.dynamic_blocks_definitions
        requested_execution_engine_version = request_payload.execution_engine_version
        body_api_key = request_payload.api_key
    api_key = resolve_api_key(
        request, request.query_params.get("api_key"), body_api_key
    )
    result = await run_in_threadpool(
        describe_workflows_blocks,
        dynamic_blocks_definitions=dynamic_blocks_definitions,
        requested_execution_engine_version=requested_execution_engine_version,
        api_key=api_key,
        workspace_resolver=host.WORKSPACE_RESOLVER,
    )
    return _gzip_if_requested(request, result)


@router.get(
    "/workflows/definition/schema",
    response_model=WorkflowsBlocksSchemaDescription,
    summary="Endpoint to fetch the workflows block schema",
    description="Endpoint to fetch the schema of all available blocks. This information can be "
    "used to validate workflow definitions and suggest syntax in the JSON editor.",
)
@with_workflow_errors
async def get_workflow_definition_schema(
    request: Request,
) -> Union[WorkflowsBlocksSchemaDescription, Response]:
    schema = await run_in_threadpool(get_workflow_schema)
    return _gzip_if_requested(request, WorkflowsBlocksSchemaDescription(schema=schema))


@router.post(
    "/workflows/blocks/dynamic_outputs",
    response_model=List[OutputDefinition],
    summary="[EXPERIMENTAL] Endpoint to get definition of dynamic output for workflow step",
    description="Endpoint to be used when step outputs can be discovered only after "
    "filling manifest with data.",
)
@with_workflow_errors
async def get_dynamic_block_outputs(
    step_manifest: Dict[str, Any],
) -> List[OutputDefinition]:
    return await run_in_threadpool(_dynamic_block_outputs, step_manifest)


def _dynamic_block_outputs(step_manifest: Dict[str, Any]) -> List[OutputDefinition]:
    dummy_workflow_definition = {
        "version": "1.0",
        "inputs": [],
        "steps": [step_manifest],
        "outputs": [],
    }
    parsed_definition = parse_workflow_definition(
        raw_workflow_definition=dummy_workflow_definition,
        available_blocks=load_workflow_blocks(),
    )
    return parsed_definition.steps[0].get_actual_outputs()


@router.post(
    "/workflows/validate",
    response_model=WorkflowValidationStatus,
    summary="[EXPERIMENTAL] Endpoint to validate",
    description="Endpoint provides a way to check validity of JSON workflow definition.",
)
@with_workflow_errors
async def validate_workflow(
    request: Request,
    specification: dict,
    api_key: Optional[str] = Query(
        None,
        description="Roboflow API Key that will be passed to the model during initialization for artifact retrieval",
    ),
) -> WorkflowValidationStatus:
    resolved_key = resolve_api_key(request, api_key, None)
    init_parameters = execution.build_init_parameters(
        provider=_models_provider(request, resolved_key),
        api_key=resolved_key,
        background_tasks=None,
        disable_sinks=False,
        inner_workflow_dispatch_depth=0,
        step_execution_mode=host.SERVER_WORKFLOWS_CONFIGURATION.engine.step_execution_mode,
    )
    await run_in_threadpool(
        ExecutionEngine.init,
        workflow_definition=specification,
        init_parameters=init_parameters,
        max_concurrent_steps=configuration.WORKFLOWS_MAX_CONCURRENT_STEPS,
        prevent_local_images_loading=True,
        step_error_handler=host.step_error_handler,
    )
    return WorkflowValidationStatus(status="ok")
