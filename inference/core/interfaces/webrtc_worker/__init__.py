import asyncio
import multiprocessing

from inference.core.env import WEBRTC_MODAL_TOKEN_ID, WEBRTC_MODAL_TOKEN_SECRET
from inference.core.interfaces.stream_manager.manager_app.entities import (
    InitialiseWebRTCPipelinePayload,
)

from .cpu import rtc_peer_connection_process
from .modal import spawn_rtc_peer_connection_modal


async def start_worker(
    webrtc_request: InitialiseWebRTCPipelinePayload,
):
    if modal is not None and WEBRTC_MODAL_TOKEN_ID and WEBRTC_MODAL_TOKEN_SECRET:
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
        answer = await loop.run_in_executor(None, parent_conn.recv)
        parent_conn.close()

        return p.pid, p, answer
