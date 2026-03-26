import asyncio
import os
from multiprocessing.connection import Connection

from inference.core import logger
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.webrtc import (
    init_rtc_peer_connection_with_loop,
)


def rtc_peer_connection_process(
    webrtc_request: WebRTCWorkerRequest,
    answer_conn: Connection,
):
    def send_answer(obj: WebRTCWorkerResult):
        answer_conn.send(obj)
        answer_conn.close()

    try:
        asyncio.run(
            init_rtc_peer_connection_with_loop(
                webrtc_request=webrtc_request,
                send_answer=send_answer,
            )
        )
        logger.info("WebRTC process terminated")
    except Exception:
        logger.exception("WebRTC worker process failed")
        answer_conn.close()
    finally:
        # Skip Python interpreter cleanup to avoid onnxruntime crash
        # on ARM64/Jetson during garbage collection at exit
        os._exit(0)
