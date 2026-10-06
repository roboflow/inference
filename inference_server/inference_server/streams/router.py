"""The legacy `/inference_pipelines/*` routes, forwarded to the stream manager.

Every route calls the `StreamManagerClient` the lifespan places on
`app.state.stream_manager_client`; the manager's error family is answered by
`with_legacy_errors`. Importing this module imports the stream manager
entities, which freeze the process stream configuration: install it first.
"""

import logging
from typing import Optional

from fastapi import APIRouter, FastAPI, Request
from streamvision.stream_manager.api.entities import (
    CommandResponse,
    ConsumePipelineResponse,
    InferencePipelineStatusResponse,
    InitializeWebRTCPipelineResponse,
    ListPipelinesResponse,
)
from streamvision.stream_manager.api.stream_manager_client import StreamManagerClient
from streamvision.stream_manager.manager_app.entities import (
    ConsumeResultsPayload,
    InitialisePipelinePayload,
    InitialiseWebRTCPipelinePayload,
)

from inference_server.legacy.common import resolve_api_key
from inference_server.legacy.errors import with_legacy_errors

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streams"])


def include_streams_router(app: FastAPI) -> None:
    """Mount the `/inference_pipelines/*` routes on the application.

    Args:
        app: Application serving the routes; it must carry the stream manager
            client on `app.state.stream_manager_client` while serving them.
    """
    app.include_router(router)


def _stream_manager_client(request: Request) -> StreamManagerClient:
    client = getattr(request.app.state, "stream_manager_client", None)
    assert client is not None, "The stream manager client is not installed."

    return client


@router.get(
    "/inference_pipelines/list",
    response_model=ListPipelinesResponse,
    summary="[EXPERIMENTAL] List active InferencePipelines",
    description="[EXPERIMENTAL] Listing all active InferencePipelines processing videos",
)
@with_legacy_errors
async def list_pipelines(request: Request) -> ListPipelinesResponse:
    response = await _stream_manager_client(request).list_pipelines()

    return response


@router.get(
    "/inference_pipelines/{pipeline_id}/status",
    response_model=InferencePipelineStatusResponse,
    summary="[EXPERIMENTAL] Get status of InferencePipeline",
    description="[EXPERIMENTAL] Get status of InferencePipeline",
)
@with_legacy_errors
async def get_status(
    request: Request, pipeline_id: str
) -> InferencePipelineStatusResponse:
    response = await _stream_manager_client(request).get_status(pipeline_id=pipeline_id)

    return response


@router.post(
    "/inference_pipelines/initialise",
    response_model=CommandResponse,
    summary="[EXPERIMENTAL] Starts new InferencePipeline",
    description="[EXPERIMENTAL] Starts new InferencePipeline",
)
@with_legacy_errors
async def initialise(
    request: Request, payload: InitialisePipelinePayload
) -> CommandResponse:
    payload.api_key = resolve_api_key(request, None, payload.api_key)
    response = await _stream_manager_client(request).initialise_pipeline(
        initialisation_request=payload
    )

    return response


@router.post(
    "/inference_pipelines/initialise_webrtc",
    response_model=InitializeWebRTCPipelineResponse,
    summary="[EXPERIMENTAL] Establishes WebRTC peer connection and starts new InferencePipeline consuming video track",
    description="[EXPERIMENTAL] Establishes WebRTC peer connection and starts new InferencePipeline consuming video track",
)
@with_legacy_errors
async def initialise_webrtc_inference_pipeline(
    request: Request, payload: InitialiseWebRTCPipelinePayload
) -> InitializeWebRTCPipelineResponse:
    payload.api_key = resolve_api_key(request, None, payload.api_key)
    logger.debug("Received initialise webrtc inference pipeline request")
    response = await _stream_manager_client(request).initialise_webrtc_pipeline(
        initialisation_request=payload
    )
    logger.debug("Returning initialise webrtc inference pipeline response")

    return response


@router.post(
    "/inference_pipelines/{pipeline_id}/pause",
    response_model=CommandResponse,
    summary="[EXPERIMENTAL] Pauses the InferencePipeline",
    description="[EXPERIMENTAL] Pauses the InferencePipeline",
)
@with_legacy_errors
async def pause(request: Request, pipeline_id: str) -> CommandResponse:
    response = await _stream_manager_client(request).pause_pipeline(
        pipeline_id=pipeline_id
    )

    return response


@router.post(
    "/inference_pipelines/{pipeline_id}/resume",
    response_model=CommandResponse,
    summary="[EXPERIMENTAL] Resumes the InferencePipeline",
    description="[EXPERIMENTAL] Resumes the InferencePipeline",
)
@with_legacy_errors
async def resume(request: Request, pipeline_id: str) -> CommandResponse:
    response = await _stream_manager_client(request).resume_pipeline(
        pipeline_id=pipeline_id
    )

    return response


@router.post(
    "/inference_pipelines/{pipeline_id}/terminate",
    response_model=CommandResponse,
    summary="[EXPERIMENTAL] Terminates the InferencePipeline",
    description="[EXPERIMENTAL] Terminates the InferencePipeline",
)
@with_legacy_errors
async def terminate(request: Request, pipeline_id: str) -> CommandResponse:
    response = await _stream_manager_client(request).terminate_pipeline(
        pipeline_id=pipeline_id
    )

    return response


@router.get(
    "/inference_pipelines/{pipeline_id}/consume",
    response_model=ConsumePipelineResponse,
    summary="[EXPERIMENTAL] Consumes InferencePipeline result",
    description="[EXPERIMENTAL] Consumes InferencePipeline result",
)
@with_legacy_errors
async def consume(
    request: Request,
    pipeline_id: str,
    payload: Optional[ConsumeResultsPayload] = None,
) -> ConsumePipelineResponse:
    if payload is None:
        payload = ConsumeResultsPayload()
    response = await _stream_manager_client(request).consume_pipeline_result(
        pipeline_id=pipeline_id,
        excluded_fields=payload.excluded_fields,
    )

    return response
