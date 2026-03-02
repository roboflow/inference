"""Stream / inference pipelines HTTP routes."""

from typing import Optional

from fastapi import APIRouter, Request

from inference.core import logger
from inference.core.interfaces.stream_manager.api.entities import (
    CommandResponse,
    ConsumePipelineResponse,
    InferencePipelineStatusResponse,
    InitializeWebRTCPipelineResponse,
    ListPipelinesResponse,
)
from inference.core.interfaces.stream_manager.api.stream_manager_client import (
    StreamManagerClient,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    ConsumeResultsPayload,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions_async


def create_stream_router(stream_manager_client: StreamManagerClient) -> APIRouter:
    router = APIRouter()

    @router.get(
        "/inference_pipelines/list",
        response_model=ListPipelinesResponse,
        summary="[EXPERIMENTAL] List active InferencePipelines",
        description="[EXPERIMENTAL] Listing all active InferencePipelines processing videos",
    )
    @with_route_exceptions_async
    async def list_pipelines(_: Request) -> ListPipelinesResponse:
        return await stream_manager_client.list_pipelines()

    @router.get(
        "/inference_pipelines/{pipeline_id}/status",
        response_model=InferencePipelineStatusResponse,
        summary="[EXPERIMENTAL] Get status of InferencePipeline",
        description="[EXPERIMENTAL] Get status of InferencePipeline",
    )
    @with_route_exceptions_async
    async def get_status(pipeline_id: str) -> InferencePipelineStatusResponse:
        return await stream_manager_client.get_status(
            pipeline_id=pipeline_id
        )

    @router.post(
        "/inference_pipelines/initialise",
        response_model=CommandResponse,
        summary="[EXPERIMENTAL] Starts new InferencePipeline",
        description="[EXPERIMENTAL] Starts new InferencePipeline",
    )
    @with_route_exceptions_async
    async def initialise(request: InitialisePipelinePayload) -> CommandResponse:
        return await stream_manager_client.initialise_pipeline(
            initialisation_request=request
        )

    @router.post(
        "/inference_pipelines/initialise_webrtc",
        response_model=InitializeWebRTCPipelineResponse,
        summary="[EXPERIMENTAL] Establishes WebRTC peer connection and starts new InferencePipeline consuming video track",
        description="[EXPERIMENTAL] Establishes WebRTC peer connection and starts new InferencePipeline consuming video track",
    )
    @with_route_exceptions_async
    async def initialise_webrtc_inference_pipeline(
        request: InitialiseWebRTCPipelinePayload,
    ) -> InitializeWebRTCPipelineResponse:
        logger.debug("Received initialise webrtc inference pipeline request")
        resp = await stream_manager_client.initialise_webrtc_pipeline(
            initialisation_request=request
        )
        logger.debug("Returning initialise webrtc inference pipeline response")
        return resp

    @router.post(
        "/inference_pipelines/{pipeline_id}/pause",
        response_model=CommandResponse,
        summary="[EXPERIMENTAL] Pauses the InferencePipeline",
        description="[EXPERIMENTAL] Pauses the InferencePipeline",
    )
    @with_route_exceptions_async
    async def pause(pipeline_id: str) -> CommandResponse:
        return await stream_manager_client.pause_pipeline(
            pipeline_id=pipeline_id
        )

    @router.post(
        "/inference_pipelines/{pipeline_id}/resume",
        response_model=CommandResponse,
        summary="[EXPERIMENTAL] Resumes the InferencePipeline",
        description="[EXPERIMENTAL] Resumes the InferencePipeline",
    )
    @with_route_exceptions_async
    async def resume(pipeline_id: str) -> CommandResponse:
        return await stream_manager_client.resume_pipeline(
            pipeline_id=pipeline_id
        )

    @router.post(
        "/inference_pipelines/{pipeline_id}/terminate",
        response_model=CommandResponse,
        summary="[EXPERIMENTAL] Terminates the InferencePipeline",
        description="[EXPERIMENTAL] Terminates the InferencePipeline",
    )
    @with_route_exceptions_async
    async def terminate(pipeline_id: str) -> CommandResponse:
        return await stream_manager_client.terminate_pipeline(
            pipeline_id=pipeline_id
        )

    @router.get(
        "/inference_pipelines/{pipeline_id}/consume",
        response_model=ConsumePipelineResponse,
        summary="[EXPERIMENTAL] Consumes InferencePipeline result",
        description="[EXPERIMENTAL] Consumes InferencePipeline result",
    )
    @with_route_exceptions_async
    async def consume(
        pipeline_id: str,
        request: Optional[ConsumeResultsPayload] = None,
    ) -> ConsumePipelineResponse:
        if request is None:
            request = ConsumeResultsPayload()
        return await stream_manager_client.consume_pipeline_result(
            pipeline_id=pipeline_id,
            excluded_fields=request.excluded_fields,
        )

    return router
