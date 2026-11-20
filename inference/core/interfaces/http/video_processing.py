"""Connect synchronous action-recognition work to the HTTP request lifetime."""

from typing import TYPE_CHECKING

from anyio import from_thread
from fastapi import HTTPException, Request

from inference.core.entities.requests.action_recognition import (
    ActionRecognitionInferenceRequest,
)
from inference.core.entities.responses.action_recognition import (
    ActionRecognitionInferenceResponse,
)
from inference.core.env import ACTION_RECOGNITION_PROCESSING_TIMEOUT_SECONDS
from inference.core.interfaces.http.middlewares.disconnect import (
    REQUEST_DISCONNECT_STATE_KEY,
)
from inference.core.utils.video_processing import (
    VideoProcessingCancelledError,
    VideoProcessingControl,
    VideoProcessingTimeoutError,
)

if TYPE_CHECKING:
    from inference.core.managers.base import ModelManager


def infer_action_recognition_request(
    model_manager: "ModelManager",
    *,
    model_id: str,
    inference_request: ActionRecognitionInferenceRequest,
    request: Request
) -> ActionRecognitionInferenceResponse:
    """Run a clip from a synchronous HTTP route with cooperative cancellation.

    Args:
        model_manager (ModelManager): Manager that owns the loaded model.
        model_id (str): Identifier registered with the manager.
        inference_request (ActionRecognitionInferenceRequest): Parsed clip request.
        request (Request): HTTP request whose connection the server observes.

    Returns:
        ActionRecognitionInferenceResponse: Complete clip results.

    Raises:
        HTTPException: Processing exceeds its deadline or the caller disconnects.
    """
    disconnect_state = request.scope.get("state", {}).get(REQUEST_DISCONNECT_STATE_KEY)
    if disconnect_state is not None:
        # The sync route runs in an AnyIO worker; activate the watcher on its
        # event loop, then poll only the thread-safe flag between model calls.
        from_thread.run_sync(disconnect_state.start_monitoring)
        is_disconnected = disconnect_state.disconnected.is_set
    else:
        # Preserve use of this helper in bare ASGI apps without our middleware.
        def is_disconnected():
            disconnected = from_thread.run(request.is_disconnected)
            return disconnected

    control = VideoProcessingControl(
        timeout_seconds=ACTION_RECOGNITION_PROCESSING_TIMEOUT_SECONDS,
        is_disconnected=is_disconnected,
    )
    try:
        control.check()
        response = model_manager.infer_from_request_sync(
            model_id, inference_request, processing_control=control
        )
        control.check()
        return response
    except VideoProcessingCancelledError as error:
        raise HTTPException(status_code=499, detail=str(error)) from error
    except VideoProcessingTimeoutError as error:
        raise HTTPException(status_code=504, detail=str(error)) from error
