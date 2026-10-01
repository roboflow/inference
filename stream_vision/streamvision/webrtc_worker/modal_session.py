"""Container-side run of one Modal WebRTC session: watchdog, peer connection, usage."""

import asyncio
import datetime
import logging
import os
from typing import Any, Awaitable, Callable, Optional

from streamvision.camera.source_reference_sanitizer import sanitize_source_reference
from streamvision.stream.environment import (
    WEBRTC_MODAL_MIN_CPU_CORES,
    WEBRTC_MODAL_MIN_RAM_MB,
    WEBRTC_MODAL_WATCHDOG_TIMEMOUT,
    WEBRTC_SESSION_HEARTBEAT_URL,
)
from streamvision.webrtc_worker.entities import WebRTCWorkerRequest, WebRTCWorkerResult
from streamvision.webrtc_worker.host import get_webrtc_worker_host
from streamvision.webrtc_worker.watchdog import Watchdog

try:
    import modal
except ImportError:
    modal = None

logger = logging.getLogger(__name__)

# https://modal.com/docs/guide/environment_variables#environment-variables
MODAL_CLOUD_PROVIDER = os.getenv("MODAL_CLOUD_PROVIDER")
MODAL_IMAGE_ID = os.getenv("MODAL_IMAGE_ID")
MODAL_REGION = os.getenv("MODAL_REGION")
MODAL_TASK_ID = os.getenv("MODAL_TASK_ID")
MODAL_ENVIRONMENT = os.getenv("MODAL_ENVIRONMENT")
MODAL_IDENTITY_TOKEN = os.getenv("MODAL_IDENTITY_TOKEN")


async def run_rtc_peer_connection_with_watchdog(
    webrtc_request: WebRTCWorkerRequest,
    send_answer: Callable[[WebRTCWorkerResult], Awaitable[None]],
    model_manager: Any,
    watchdog: Watchdog,
):
    from streamvision.webrtc_worker.webrtc import init_rtc_peer_connection_with_loop

    rtc_peer_connection_task = asyncio.create_task(
        init_rtc_peer_connection_with_loop(
            webrtc_request=webrtc_request,
            send_answer=send_answer,
            model_manager=model_manager,
            heartbeat_callback=watchdog.heartbeat,
            connection_established_callback=watchdog.mark_connection_established,
        )
    )

    loop = asyncio.get_running_loop()

    def on_timeout(message: Optional[str] = ""):
        msg = "Cancelled by watchdog"
        if message:
            msg += f": {message}"
        # Use call_soon_threadsafe since this callback is invoked from the watchdog thread
        loop.call_soon_threadsafe(rtc_peer_connection_task.cancel, msg)

    watchdog.on_timeout = on_timeout
    watchdog.start()

    try:
        await rtc_peer_connection_task
        logger.info("Task completed uninterrupted")
    except modal.exception.InputCancellation:
        logger.warning("Modal function was cancelled")
    except asyncio.CancelledError as exc:
        logger.warning("WebRTC connection task was cancelled (%s)", exc)


def run_modal_session(
    webrtc_request: WebRTCWorkerRequest,
    q: Any,
    *,
    workflow_id: str,
    model_manager: Optional[Any],
    cold_start: bool,
    function_call_number_on_container: int,
    container_startup_time_seconds: float,
) -> None:
    """Run one WebRTC session inside a Modal container and report its usage.

    Args:
        webrtc_request: Request of the session.
        q: Modal queue the answer is delivered on.
        workflow_id: Resource identifier the host resolved for the session.
        model_manager: Host-owned models object, passed through untouched.
        cold_start: Whether the container started cold.
        function_call_number_on_container: Ordinal of this call on the container.
        container_startup_time_seconds: Time the container took to start.

    Raises:
        Exception: No frame was processed during the session.
    """
    _exec_session_started = datetime.datetime.now()
    webrtc_request.processing_session_started = _exec_session_started
    # Modal cancels based on time taken during entry hook
    if function_call_number_on_container == 1 and cold_start:
        logger.info(
            "Subtracting container startup time (%s) from processing session started (%s)",
            container_startup_time_seconds,
            webrtc_request.processing_session_started,
        )
        webrtc_request.processing_session_started -= datetime.timedelta(
            seconds=container_startup_time_seconds
        )
    logger.info("WebRTC session started at %s", _exec_session_started.isoformat())
    logger.info(
        "webrtc_realtime_processing: %s",
        webrtc_request.webrtc_realtime_processing,
    )
    logger.info("stream_output: %s", webrtc_request.stream_output)
    logger.info("data_output: %s", webrtc_request.data_output)
    logger.info("declared_fps: %s", webrtc_request.declared_fps)
    logger.info(
        "rtsp_url: %s",
        sanitize_source_reference(webrtc_request.rtsp_url or ""),
    )
    logger.info("processing_timeout: %s", webrtc_request.processing_timeout)
    logger.info("requested_region: %s", webrtc_request.requested_region)
    logger.info("watchdog_timeout: %s", WEBRTC_MODAL_WATCHDOG_TIMEMOUT)
    logger.info("requested_plan: %s", webrtc_request.requested_plan)
    logger.info(
        "ICE servers: %s",
        len(
            webrtc_request.webrtc_config.iceServers
            if webrtc_request.webrtc_config
            else []
        ),
    )
    logger.info(
        "WEBRTC_MODAL_MIN_CPU_CORES: %s",
        WEBRTC_MODAL_MIN_CPU_CORES or "not set",
    )
    logger.info("WEBRTC_MODAL_MIN_RAM_MB: %s", WEBRTC_MODAL_MIN_RAM_MB or "not set")
    logger.info("MODAL_CLOUD_PROVIDER: %s", MODAL_CLOUD_PROVIDER)
    logger.info("MODAL_IMAGE_ID: %s", MODAL_IMAGE_ID)
    logger.info("MODAL_REGION: %s", MODAL_REGION)
    logger.info("MODAL_TASK_ID: %s", MODAL_TASK_ID)
    logger.info("MODAL_ENVIRONMENT: %s", MODAL_ENVIRONMENT)
    logger.info("MODAL_IDENTITY_TOKEN set: %s", bool(MODAL_IDENTITY_TOKEN))

    async def send_answer(obj: WebRTCWorkerResult):
        logger.info("Sending webrtc answer")
        if obj.error_message:
            logger.error("Error: %s (%s)", obj.error_message, obj.exception_type)
        await q.put.aio(obj)

    if webrtc_request.processing_timeout == 0:
        error_msg = "Processing timeout is 0, skipping processing"
        logger.info(error_msg)
        asyncio.run(send_answer(WebRTCWorkerResult(error_message=error_msg)))
        return
    if (
        not webrtc_request.webrtc_offer
        or not webrtc_request.webrtc_offer.sdp
        or not webrtc_request.webrtc_offer.type
    ):
        error_msg = "Webrtc offer is missing, skipping processing"
        logger.info(error_msg)
        asyncio.run(send_answer(WebRTCWorkerResult(error_message=error_msg)))
        return

    watchdog = Watchdog(
        api_key=webrtc_request.api_key,
        timeout_seconds=WEBRTC_MODAL_WATCHDOG_TIMEMOUT,
        workspace_id=getattr(webrtc_request, "workspace_id", None),
        session_id=getattr(webrtc_request, "session_id", None),
        heartbeat_url=WEBRTC_SESSION_HEARTBEAT_URL,
    )

    try:
        asyncio.run(
            run_rtc_peer_connection_with_watchdog(
                webrtc_request=webrtc_request,
                send_answer=send_answer,
                model_manager=model_manager,
                watchdog=watchdog,
            )
        )
    except modal.exception.InputCancellation:
        logger.warning("Modal function was cancelled")
    except asyncio.CancelledError as exc:
        logger.warning("WebRTC connection task was cancelled (%s)", exc)
    except Exception:
        logger.exception("WebRTC session failed")
    finally:
        # This synchronous owner runs after asyncio.run has closed its loop.
        watchdog.stop()

    _exec_session_stopped = datetime.datetime.now()
    logger.info(
        "WebRTC session stopped at %s",
        _exec_session_stopped.isoformat(),
    )

    no_frames_processed = watchdog.total_heartbeats == 0

    video_source = "realtime browser stream"
    if webrtc_request.rtsp_url:
        video_source = "rtsp"
    elif not webrtc_request.webrtc_realtime_processing:
        video_source = "buffered browser stream"
    else:
        video_source = "realtime browser stream"

    host = get_webrtc_worker_host()
    host.record_session_usage(
        webrtc_request=webrtc_request,
        workflow_id=workflow_id,
        video_source=video_source,
        session_started=_exec_session_started,
        session_stopped=_exec_session_stopped,
        connection_established=watchdog.connection_established,
    )

    logger.info("Function completed")

    if no_frames_processed:
        if watchdog.connection_established:
            host.push_usage_payloads()
            raise Exception(
                "WebRTC connection was established but no frames were processed. "
                "This typically indicates an invalid RTSP stream URL or corrupted video file."
            )
        else:
            raise Exception(
                "WebRTC connection could not be established. "
                "No frames were processed."
            )
    host.push_usage_payloads()
