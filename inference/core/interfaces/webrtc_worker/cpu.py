import asyncio
from multiprocessing.connection import Connection

from .entities import WebRTCWorkerRequest
from .webrtc import init_rtc_peer_connection_with_loop


def rtc_peer_connection_process(
    webrtc_request: WebRTCWorkerRequest,
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
