import asyncio
import multiprocessing

from inference.core.env import (
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WEBRTC_MODAL_USAGE_QUOTA_ENABLED,
)
from inference.core.exceptions import CreditsExceededError
from inference.core.interfaces.webrtc_worker.cpu import rtc_peer_connection_process
from inference.core.interfaces.webrtc_worker.entities import (
    RTCIceServer,
    WebRTCConfig,
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.logger import logger


async def start_worker(
    webrtc_request: WebRTCWorkerRequest,
) -> WebRTCWorkerResult:
    if webrtc_request.webrtc_turn_config:
        webrtc_request.webrtc_config = WebRTCConfig(
            iceServers=[
                RTCIceServer(
                    urls=[webrtc_request.webrtc_turn_config.urls],
                    username=webrtc_request.webrtc_turn_config.username,
                    credential=webrtc_request.webrtc_turn_config.credential,
                )
            ]
        )

    if WEBRTC_MODAL_TOKEN_ID and WEBRTC_MODAL_TOKEN_SECRET:
        try:
            from inference.core.interfaces.webrtc_worker.modal import (
                spawn_rtc_peer_connection_modal,
            )
            from inference.core.interfaces.webrtc_worker.utils import is_over_quota
        except ImportError:
            raise ImportError(
                "Modal not installed, please install it using 'pip install modal'"
            )
        if WEBRTC_MODAL_USAGE_QUOTA_ENABLED:
            if is_over_quota(webrtc_request.api_key):
                logger.error("API key over quota")
                raise CreditsExceededError("API key over quota")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            spawn_rtc_peer_connection_modal,
            webrtc_request,
        )
        return result
    else:
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=rtc_peer_connection_process,
            kwargs={
                "webrtc_request": webrtc_request,
                "answer_conn": child_conn,
            },
            daemon=False,
        )
        p.start()
        child_conn.close()

        loop = asyncio.get_running_loop()
        answer = WebRTCWorkerResult.model_validate(
            await loop.run_in_executor(None, parent_conn.recv)
        )
        parent_conn.close()

        return answer
