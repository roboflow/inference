import argparse
import asyncio
import json
import logging
import sys
import urllib.parse
from enum import Enum
from pathlib import Path
from threading import Event, Thread
from typing import Optional

import cv2 as cv
import numpy as np
import requests
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av.logging import ERROR, set_libav_level

from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCData,
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)
from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.roboflow_api import get_workflow_specification
from inference.core.utils.async_utils import Queue

logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG,
    format="%(asctime)s - [%(levelname)s] [%(funcName)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)


class WebcamFrameGrabberState(Enum):
    STOPPED = "STOPPED"
    STOPPING = "STOPPING"
    STARTING = "STARTING"
    STARTED = "STARTED"


class FramesGrabber:
    def __init__(
        self,
        source_path: Optional[str] = None,
    ):
        if source_path:
            self._cap = cv.VideoCapture(source_path)
        else:
            self._cap = cv.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam")
        self._fps = self._cap.get(cv.CAP_PROP_FPS)

    def get_frame(self) -> Optional[np.ndarray]:
        ret, np_frame = self._cap.read()
        if not ret:
            return None
        return np_frame

    def get_fps(self) -> Optional[float]:
        return self._fps


class StreamTrack(VideoStreamTrack):
    def __init__(
        self,
        asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
        source_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop
        if asyncio_loop is None:
            self._loop = asyncio.get_event_loop()

        self._camera = FramesGrabber(source_path=source_path)

        self.track: Optional[RemoteStreamTrack] = None
        self._recv_task: Optional[asyncio.Task] = None
        self.recv_queue: "Queue[Optional[VideoFrame]]" = Queue(loop=self._loop)

        self._av_logging_set: bool = False

    def set_track(self, track: RemoteStreamTrack):
        self.track = track
        self._recv_task = self._loop.create_task(self._recv_loop(), name="recv_loop")

    async def stop_recv_loop(self):
        if self._recv_task:
            self._recv_task.cancel()
        await self.recv_queue.async_put(None)

    async def _recv_loop(self):
        try:
            while self.track.readyState != "ended":
                frame: VideoFrame = await self.track.recv()
                await self.recv_queue.async_put(frame)

        except asyncio.CancelledError:
            logger.info("WebRTC recv loop cancelled")
        except MediaStreamError:
            if not self.complete:
                logger.error("WebRTC recv loop finished due to MediaStreamError")
        except Exception as exc:
            logger.error("Error in WebRTC recv loop: %s", exc)

        await self.recv_queue.async_put(None)

    async def recv(self):
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            set_libav_level(ERROR)
            self._av_logging_set = True

        np_frame = await self._loop.run_in_executor(
            None,
            self._camera.get_frame,
        )
        if np_frame is None:
            logger.info("%s: No more frames", self.__class__.__name__)
            await self.stop_recv_loop()
            raise MediaStreamError("No more frames")

        new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
        new_frame.pts, new_frame.time_base = await self.next_timestamp()

        return new_frame


class RTCPeerConnectionWithDataChannel(RTCPeerConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_channel: Optional[RTCDataChannel] = None
        self.webcam_stream_track: Optional[StreamTrack] = None
        self.closed_event: Event = Event()


async def init_rtc_peer_connection_with_local_description(
    asyncio_loop: asyncio.AbstractEventLoop,
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
    source_path: Optional[str] = None,
) -> RTCPeerConnectionWithDataChannel:
    webcam_stream_track = StreamTrack(
        asyncio_loop=asyncio_loop,
        source_path=source_path,
    )

    if webrtc_turn_config:
        turn_server = RTCIceServer(
            urls=[webrtc_turn_config.urls],
            username=webrtc_turn_config.username,
            credential=webrtc_turn_config.credential,
        )
        peer_connection = RTCPeerConnectionWithDataChannel(
            configuration=RTCConfiguration(iceServers=[turn_server]),
        )
    else:
        peer_connection = RTCPeerConnectionWithDataChannel()
    relay = MediaRelay()

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("track received")
        webcam_stream_track.set_track(track=relay.subscribe(track))
        peer_connection.webcam_stream_track = webcam_stream_track

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("connection state: %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            await webcam_stream_track.stop_recv_loop()
            peer_connection.closed_event.set()
            await peer_connection.close()

    data_channel = peer_connection.createDataChannel("inference")

    @data_channel.on("message")
    def on_message(message):
        print(message)

    peer_connection.data_channel = data_channel
    peer_connection.addTrack(webcam_stream_track)

    offer: RTCSessionDescription = await peer_connection.createOffer()
    await peer_connection.setLocalDescription(offer)
    while peer_connection.iceGatheringState != "complete":
        logger.debug("Waiting for ice gathering to complete")
        await asyncio.sleep(0.1)

    return peer_connection


class FileMustExist(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not values.strip() or not Path(values.strip()).exists():
            raise argparse.ArgumentError(argument=self, message="Incorrect path")
        setattr(namespace, self.dest, values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebRTC webcam stream")
    parser.add_argument(
        "--source-path",
        required=False,
        type=str,
        default=None,
        action=FileMustExist,
        help="Path to video file, if not provided webcam will be used",
    )
    parser.add_argument("--workflow-id", required=True, type=str)
    parser.add_argument("--workspace-id", required=True, type=str)
    parser.add_argument("--inference-server-url", required=True, type=str)
    parser.add_argument("--api-key", required=True, type=str)
    parser.add_argument("--realtime", required=False, action="store_true")
    parser.add_argument("--turn-url", required=False, type=str)
    parser.add_argument("--turn-username", required=False, type=str)
    parser.add_argument("--turn-credential", required=False, type=str)
    return parser.parse_args()


def main():
    args = parse_args()

    workflow_specification = get_workflow_specification(
        api_key=args.api_key,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
    )

    webrtc_turn_config = None
    if args.turn_url:
        webrtc_turn_config = WebRTCTURNConfig(
            urls=args.turn_url,
            username=args.turn_username,
            credential=args.turn_credential,
        )

    asyncio_loop = asyncio.new_event_loop()
    loop_thread = Thread(target=asyncio_loop.run_forever, daemon=True)
    loop_thread.start()
    asyncio.set_event_loop(asyncio_loop)
    future = asyncio.run_coroutine_threadsafe(
        init_rtc_peer_connection_with_local_description(
            asyncio_loop=asyncio_loop,
            webrtc_turn_config=webrtc_turn_config,
            source_path=args.source_path,
        ),
        asyncio_loop,
    )
    peer_connection = future.result()

    request = WebRTCWorkerRequest(
        api_key=args.api_key,
        workflow_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workflow_id=args.workflow_id,
            workspace_name=args.workspace_id,
            image_input_name="image",
            workflows_parameters={},
            workflows_thread_pool_workers=4,
            cancel_thread_pool_tasks_on_exit=True,
            video_metadata_input_name="video_metadata",
        ),
        webrtc_offer=WebRTCOffer(
            type=peer_connection.localDescription.type,
            sdp=peer_connection.localDescription.sdp,
        ),
        webrtc_turn_config=webrtc_turn_config,
        stream_output=["video"],
        data_output=["preds"],
        webrtc_realtime_processing=False if args.source_path is None else True,
    )

    https_verify = True
    if args.inference_server_url.startswith("https://") and (
        "localhost" in args.inference_server_url
        or "127.0.0.1" in args.inference_server_url
    ):
        https_verify = False

    response = requests.post(
        urllib.parse.urljoin(args.inference_server_url, "initialise_webrtc_worker"),
        json=request.model_dump(),
        verify=https_verify,
    )
    webrtc_answer = response.json()
    if response.status_code != 200:
        raise Exception(f"Failed to initialise WebRTC pipeline: {response.text}")

    future = asyncio.run_coroutine_threadsafe(
        peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=webrtc_answer["sdp"], type=webrtc_answer["type"])
        ),
        asyncio_loop,
    )
    future.result()

    while not peer_connection.closed_event.is_set():
        frame: Optional[VideoFrame] = (
            peer_connection.webcam_stream_track.recv_queue.sync_get()
        )
        if frame is None:
            logger.info("No more frames")
            break
        cv.imshow("Processed frame", frame.to_ndarray(format="bgr24"))
        key = cv.waitKey(1)

        if key == -1:
            continue

        if key == ord("q"):
            break

        if chr(key) in "1234567890abcdefghijkz" and (
            not peer_connection.data_channel
            or peer_connection.data_channel.readyState != "open"
        ):
            logger.error("Data channel not open")
            continue

        if chr(key) in "1234567890":
            if chr(key) == "0":
                message = json.dumps(
                    WebRTCData(
                        stream_output="",
                        data_output=None,
                    ).model_dump()
                )
                logger.info("Turning off stream output via data channel")
            else:
                max_ind = max(0, len(workflow_specification.get("outputs", [])) - 1)
                output_ind = min(key - ord("1"), max_ind)
                output_name = workflow_specification.get("outputs")[output_ind].get(
                    "name", ""
                )
                message = json.dumps(
                    WebRTCData(
                        stream_output=output_name,
                        data_output=None,
                    ).model_dump()
                )
                logger.info("Setting stream output via data channel: %s", output_name)
            peer_connection.data_channel.send(message)

        if chr(key) in "abcdefghijkz":
            if chr(key) == "z":
                message = json.dumps(
                    WebRTCData(
                        stream_output=None,
                        data_output="",
                    ).model_dump()
                )
                logger.info("Turning off data output via data channel")
            else:
                max_ind = max(0, len(workflow_specification.get("outputs", [])) - 1)
                output_ind = min(key - ord("a"), max_ind)
                output_name = workflow_specification.get("outputs")[output_ind].get(
                    "name", ""
                )
                message = json.dumps(
                    WebRTCData(
                        stream_output=None,
                        data_output=output_name,
                    ).model_dump()
                )
                logger.info("Setting data output via data channel: %s", output_name)
            peer_connection.data_channel.send(message)

    cv.destroyAllWindows()
    asyncio.run_coroutine_threadsafe(
        peer_connection.webcam_stream_track.stop_recv_loop(),
        asyncio_loop,
    ).result()
    asyncio.run_coroutine_threadsafe(
        peer_connection.close(),
        asyncio_loop,
    ).result()
    asyncio_loop.stop()
    loop_thread.join()


if __name__ == "__main__":
    main()
