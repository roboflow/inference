"""
V1 API endpoints for workflow execution.

This module provides workflow execution endpoints using multipart form data
for efficient image handling and header-based authentication.
"""

from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    Path,
    Query,
    Request,
)

from inference.core import logger
from inference.core.entities.requests.workflows import (
    PredefinedWorkflowInferenceRequest,
    WorkflowInferenceRequest,
    WorkflowSpecificationInferenceRequest,
)
from inference.core.entities.responses.workflows import WorkflowInferenceResponse
from inference.core.env import (
    ENABLE_WORKFLOWS_PROFILING,
    GCP_SERVERLESS,
    LAMBDA,
    WORKFLOWS_PROFILER_BUFFER_SIZE,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.exceptions import InputImageLoadError
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.interfaces.http.handlers.workflows import (
    filter_out_unwanted_workflow_outputs,
)
from inference.core.interfaces.http.orjson_utils import (
    orjson_response_keeping_parent_id,
)
from inference.core.interfaces.http.v1.auth import get_validated_api_key
from inference.core.interfaces.http.v1.multipart import (
    parse_workflow_multipart,
    merge_images_into_inputs,
)
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import get_workflow_specification
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.profiling.core import (
    BaseWorkflowsProfiler,
    NullWorkflowsProfiler,
)
from inference.usage_tracking.collector import usage_collector

router = APIRouter()


def create_workflow_endpoints(model_manager: ModelManager) -> APIRouter:
    """
    Create and configure v1 workflow endpoints.

    Args:
        model_manager: The model manager instance

    Returns:
        Configured APIRouter with all workflow endpoints
    """

    def process_workflow_inference_request_v1(
        workflow_request: WorkflowInferenceRequest,
        workflow_specification: dict,
        background_tasks: Optional[BackgroundTasks],
        profiler,
    ) -> WorkflowInferenceResponse:
        """
        Process a workflow inference request (v1 internal helper).

        Args:
            workflow_request: The workflow request
            workflow_specification: The workflow definition
            background_tasks: FastAPI background tasks
            profiler: Workflow profiler instance

        Returns:
            Workflow inference response
        """
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
            workflow_id=workflow_request.workflow_id,
        )

        is_preview = False
        workflow_results = execution_engine.run(
            runtime_parameters=workflow_request.inputs,
        )

        outputs = filter_out_unwanted_workflow_outputs(
            execution_graph=execution_engine.execution_graph,
            workflow_results=workflow_results,
            excluded_fields=workflow_request.excluded_fields,
        )

        response = WorkflowInferenceResponse(
            outputs=outputs,
            is_preview=is_preview,
        )

        if workflow_request.enable_profiling:
            response.profiling_data = profiler.export_data()

        return response

    @router.post(
        "/workflows/{workspace_id}/{workflow_id}",
        response_model=WorkflowInferenceResponse,
        summary="Run Predefined Workflow (v1)",
        description="Execute a predefined workflow using multipart form data for images",
        tags=["v1", "Workflows"],
    )
    @with_route_exceptions
    @usage_collector("request")
    async def run_predefined_workflow_v1(
        workspace_id: str = Path(..., description="Workspace ID"),
        workflow_id: str = Path(..., description="Workflow ID"),
        request: Request = None,
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_validated_api_key),
        use_cache: bool = Query(True, description="Use cached workflow definition"),
        enable_profiling: bool = Query(False, description="Enable workflow profiling"),
    ):
        """
        Execute a predefined workflow (v1).

        Expected multipart/form-data format:
        - inputs: JSON string with non-image workflow inputs (optional)
        - <input_name>: Binary image file (one per image input)

        Example:
        - inputs: '{"confidence": 0.5, "prompt": "detect people"}'
        - image: <binary image data>
        - mask: <binary mask data>

        The image field names should match the workflow input names.
        If a workflow input is provided in the inputs JSON as an image dict
        (with type/value), it will be used instead of the multipart field.

        Authentication:
        - Header: Authorization: Bearer <api_key>
        - Header: X-Roboflow-Api-Key: <api_key>
        - Query param: ?api_key=<api_key>

        Returns:
            Workflow execution results
        """
        logger.debug(f"[V1] Predefined workflow request: {workspace_id}/{workflow_id}")

        # Parse multipart form data
        form_data = await request.form()
        images, inputs, _ = await parse_workflow_multipart(form_data)

        # Merge images into inputs using convention-based matching
        merged_inputs = await merge_images_into_inputs(images, inputs)

        logger.debug(
            f"[V1] Workflow inputs: {len(merged_inputs)} total "
            f"({len(images)} from multipart images)"
        )

        # Setup profiler
        if ENABLE_WORKFLOWS_PROFILING and enable_profiling:
            profiler = BaseWorkflowsProfiler.init(
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
            )
        else:
            profiler = NullWorkflowsProfiler.init()

        # Fetch workflow specification
        with profiler.profile_execution_phase(
            name="workflow_definition_fetching",
            categories=["inference_package_operation"],
        ):
            workflow_specification = get_workflow_specification(
                api_key=api_key,
                workspace_id=workspace_id,
                workflow_id=workflow_id,
                use_cache=use_cache,
            )

        # Create workflow request
        workflow_request = PredefinedWorkflowInferenceRequest(
            api_key=api_key,
            inputs=merged_inputs,
            excluded_fields=None,
            enable_profiling=enable_profiling,
            workflow_id=workflow_id,
            use_cache=use_cache,
        )

        # Process workflow
        response = process_workflow_inference_request_v1(
            workflow_request=workflow_request,
            workflow_specification=workflow_specification,
            background_tasks=(
                background_tasks if not (LAMBDA or GCP_SERVERLESS) else None
            ),
            profiler=profiler,
        )

        logger.debug(f"[V1] Workflow completed: {workspace_id}/{workflow_id}")
        return orjson_response_keeping_parent_id(response)

    @router.post(
        "/workflows/run",
        response_model=WorkflowInferenceResponse,
        summary="Run Workflow Specification (v1)",
        description="Execute a workflow specification provided in the request using multipart form data",
        tags=["v1", "Workflows"],
    )
    @with_route_exceptions
    @usage_collector("request")
    async def run_workflow_specification_v1(
        request: Request = None,
        background_tasks: BackgroundTasks = None,
        api_key: str = Depends(get_validated_api_key),
        enable_profiling: bool = Query(False, description="Enable workflow profiling"),
    ):
        """
        Execute a workflow specification (v1).

        Expected multipart/form-data format:
        - specification: JSON string with workflow definition (required)
        - inputs: JSON string with non-image workflow inputs (optional)
        - <input_name>: Binary image file (one per image input)

        Example:
        - specification: '{"version": "1.0", "steps": [...]}'
        - inputs: '{"confidence": 0.5}'
        - image: <binary image data>

        Authentication:
        - Header: Authorization: Bearer <api_key>
        - Header: X-Roboflow-Api-Key: <api_key>
        - Query param: ?api_key=<api_key>

        Returns:
            Workflow execution results
        """
        logger.debug("[V1] Workflow specification request")

        # Parse multipart form data
        form_data = await request.form()
        images, inputs, specification = await parse_workflow_multipart(form_data)

        if not specification:
            raise InputImageLoadError(
                message="Workflow specification is required in 'specification' field",
                public_message="Workflow specification is required",
            )

        # Merge images into inputs
        merged_inputs = await merge_images_into_inputs(images, inputs)

        logger.debug(
            f"[V1] Workflow inputs: {len(merged_inputs)} total "
            f"({len(images)} from multipart images)"
        )

        # Setup profiler
        if ENABLE_WORKFLOWS_PROFILING and enable_profiling:
            profiler = BaseWorkflowsProfiler.init(
                max_runs_in_buffer=WORKFLOWS_PROFILER_BUFFER_SIZE,
            )
        else:
            profiler = NullWorkflowsProfiler.init()

        # Create workflow request
        workflow_request = WorkflowSpecificationInferenceRequest(
            api_key=api_key,
            inputs=merged_inputs,
            excluded_fields=None,
            enable_profiling=enable_profiling,
            specification=specification,
        )

        # Process workflow
        response = process_workflow_inference_request_v1(
            workflow_request=workflow_request,
            workflow_specification=specification,
            background_tasks=(
                background_tasks if not (LAMBDA or GCP_SERVERLESS) else None
            ),
            profiler=profiler,
        )

        logger.debug("[V1] Workflow specification completed")
        return orjson_response_keeping_parent_id(response)

    return router
