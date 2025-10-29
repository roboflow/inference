import asyncio
from multiprocessing.connection import Connection

from inference.core.interfaces.stream_manager.manager_app.entities import (
    InitialiseWebRTCPipelinePayload,
)

from .webrtc import init_rtc_peer_connection_with_loop


def rtc_peer_connection_process(
    webrtc_request: InitialiseWebRTCPipelinePayload,
    answer_conn: Connection,
):
    def send_answer(obj):
        answer_conn.send(obj)
        answer_conn.close()

    asyncio.run(
        init_rtc_peer_connection_with_loop(
            webrtc_request=webrtc_request,
            send_answer=send_answer,
        )
    )
