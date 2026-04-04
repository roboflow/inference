"""WebRTC worker initialization HTTP routes."""

import re
from fastapi import APIRouter, Request
from pydantic import ValidationError
from inference.core import logger
from inference.core.env import BUILDER_ORIGIN
from inference.core.exceptions import (
    MissingApiKeyError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    WebRTCConfigurationError,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions_async
from inference.core.interfaces.stream_manager.api.entities import (
    CommandContext,
    InitializeWebRTCResponse,
)
from inference.core.interfaces.stream_manager.manager_app.entities import OperationStatus
from inference.core.interfaces.webrtc_worker import start_worker
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.workflows.errors import WorkflowError, WorkflowSyntaxError


def create_webrtc_worker_router() -> APIRouter:
    router = APIRouter()
    @router.post(
        "/initialise_webrtc_worker",
        response_model=InitializeWebRTCResponse,
        summary="[EXPERIMENTAL] Establishes WebRTC peer connection and processes video stream in spawned process or modal function",
        description="[EXPERIMENTAL] Establishes WebRTC peer connection and processes video stream in spawned process or modal function",
    )
    @with_route_exceptions_async
    async def initialise_webrtc_worker(
        request: WebRTCWorkerRequest,
        r: Request,
    ) -> InitializeWebRTCResponse:
        if str(r.headers.get("origin")).lower() == BUILDER_ORIGIN.lower():
            if re.search(
                r"^https://[^.]+\.roboflow\.[^./]+/", str(r.url).lower()
            ):
                request.is_preview = True

        logger.debug("Received initialise_webrtc_worker request")
        worker_result: WebRTCWorkerResult = await start_worker(
            webrtc_request=request,
        )
        if worker_result.exception_type is not None:
            if worker_result.exception_type == "WorkflowSyntaxError":
                raise WorkflowSyntaxError(
                    public_message=worker_result.error_message,
                    context=worker_result.error_context,
                    inner_error=worker_result.inner_error,
                )
            if worker_result.exception_type == "WorkflowError":
                raise WorkflowError(
                    public_message=worker_result.error_message,
                    context=worker_result.error_context,
                )
            expected_exceptions = {
                "Exception": Exception,
                "KeyError": KeyError,
                "MissingApiKeyError": MissingApiKeyError,
                "NotImplementedError": NotImplementedError,
                "RoboflowAPINotAuthorizedError": RoboflowAPINotAuthorizedError,
                "RoboflowAPINotNotFoundError": RoboflowAPINotNotFoundError,
                "ValidationError": ValidationError,
                "WebRTCConfigurationError": WebRTCConfigurationError,
            }
            exc = expected_exceptions.get(
                worker_result.exception_type, Exception
            )(worker_result.error_message)
            logger.error(
                f"Initialise webrtc worker failed with %s: %s",
                worker_result.exception_type,
                worker_result.error_message,
            )
            raise exc
        logger.debug("Returning initialise_webrtc_worker response")
        return InitializeWebRTCResponse(
            context=CommandContext(),
            status=OperationStatus.SUCCESS,
            sdp=worker_result.answer.sdp,
            type=worker_result.answer.type,
        )
    return router
