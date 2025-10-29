import argparse
import asyncio
import json
import logging
import sys
import time
import urllib.parse
from asyncio.exceptions import TimeoutError
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

from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)
from inference.core.interfaces.stream_manager.manager_app.entities import (
    InitialiseWebRTCPipelinePayload,
    MemorySinkConfiguration,
    VideoConfiguration,
    WebRTCData,
    WebRTCOffer,
    WebRTCTURNConfig,
    WorkflowConfiguration,
)
from inference.core.roboflow_api import get_workflow_specification
from inference.core.utils.async_utils import Queue as SyncAsyncQueue
from inference.core.workflows.execution_engine.v1.compiler.core import compile_workflow
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)

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


class WebcamFramesGrabber:
    def __init__(
        self,
        async_loop: Optional[asyncio.AbstractEventLoop] = None,
        maxsize: int = 10,
    ):
        self._frames_sink_queue: "SyncAsyncQueue[np.ndarray]" = SyncAsyncQueue(
            loop=async_loop, maxsize=maxsize
        )
        self._stop_event: Event = Event()
        self._frames_reader_thread: Optional[Thread] = None
        self._fps: Optional[float] = None
        self._state: WebcamFrameGrabberState = WebcamFrameGrabberState.STOPPED

    def frames_reader(self):
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam")
        self._fps = cap.get(cv.CAP_PROP_FPS)
        self._state = WebcamFrameGrabberState.STARTING
        logger.info("%s: %s", self.__class__.__name__, self._state.name)
        while True:
            if self._stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                break
            if self._frames_sink_queue.sync_full():
                logger.debug("%s: frames queue full", self.__class__.__name__)
                self._frames_sink_queue.sync_get_nowait()
            self._frames_sink_queue.sync_put(frame)
            if self._state == WebcamFrameGrabberState.STARTING:
                self._state = WebcamFrameGrabberState.STARTED
                logger.info("%s: %s", self.__class__.__name__, self._state.name)
        self._state = WebcamFrameGrabberState.STOPPING
        logger.info("%s: %s", self.__class__.__name__, self._state.name)
        cap.release()

    def start(self):
        if self._frames_reader_thread:
            return
        while not self._frames_sink_queue.sync_empty():
            self._frames_sink_queue.sync_get_nowait()
        self._frames_reader_thread = Thread(target=self.frames_reader, daemon=True)
        self._frames_reader_thread.start()

    def stop(self):
        if not self._frames_reader_thread:
            return
        self._stop_event.set()
        self._frames_reader_thread.join()
        self._frames_reader_thread = None

        if self._frames_sink_queue.sync_full():
            logger.debug("%s: frames queue full", self.__class__.__name__)
            self._frames_sink_queue.sync_get_nowait()
        logger.debug("%s: poison pill", self.__class__.__name__)
        self._frames_sink_queue.sync_put(None)

        self._stop_event.clear()
        self._state = WebcamFrameGrabberState.STOPPED
        self._fps = None
        logger.info("%s: %s", self.__class__.__name__, self._state.name)

    def get_fps(self) -> Optional[float]:
        if (
            self._state
            not in {WebcamFrameGrabberState.STARTING, WebcamFrameGrabberState.STARTED}
            or self._fps is None
        ):
            raise RuntimeError("FPS not initialized")
        return self._fps


class WebcamStreamTrack(VideoStreamTrack):
    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        webcam_frames_grabber_queue: "SyncAsyncQueue[np.ndarray]",
        peer_frames_queue: "SyncAsyncQueue[VideoFrame]",
        webcam_fps: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.track: Optional[RemoteStreamTrack] = None
        self._id = time.time_ns()
        self._processed = 0
        self._asyncio_loop = asyncio_loop
        self._track_active: bool = True
        self._webcam_frames_grabber_queue: "SyncAsyncQueue[np.ndarray]" = (
            webcam_frames_grabber_queue
        )
        self._webcam_fps = webcam_fps
        self._peer_frames_queue: "SyncAsyncQueue[VideoFrame]" = peer_frames_queue
        self._av_logging_set: bool = False

    def set_track(self, track: RemoteStreamTrack):
        self.track = track

    def close(self):
        self._track_active = False

    async def recv(self):
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            set_libav_level(ERROR)
            self._av_logging_set = True

        if not self._track_active:
            if await self._peer_frames_queue.async_full():
                logger.debug("%s: results queue full", self.__class__.__name__)
                await self._peer_frames_queue.async_get_nowait()
            logger.debug("%s: poison pill", self.__class__.__name__)
            await self._peer_frames_queue.async_put(None)
            raise MediaStreamError("Track not active")

        try:
            frame: VideoFrame = await asyncio.wait_for(self.track.recv(), timeout=0.005)
            if await self._peer_frames_queue.async_full():
                logger.debug("%s: results queue full", self.__class__.__name__)
                await self._peer_frames_queue.async_get_nowait()
            await self._peer_frames_queue.async_put(frame)
        except (asyncio.TimeoutError, MediaStreamError):
            logger.debug("%s: timeout - peer not ready", self.__class__.__name__)
            pass

        frame = await self._webcam_frames_grabber_queue.async_get()
        if frame is None:
            logger.debug("%s: poison pill received", self.__class__.__name__)
            raise MediaStreamError("No more frames")
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts, new_frame.time_base = await self.next_timestamp()
        self._processed += 1

        return new_frame


class RTCPeerConnectionWithTrack(RTCPeerConnection):
    def __init__(self, webcam_stream_track: WebcamStreamTrack, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.webcam_stream_track: WebcamStreamTrack = webcam_stream_track
        self.data_channel: Optional[RTCDataChannel] = None


async def init_rtc_peer_connection_with_local_description(
    webcam_frames_grabber_queue: "SyncAsyncQueue[np.ndarray]",
    peer_frames_queue: "SyncAsyncQueue[VideoFrame]",
    asyncio_loop: asyncio.AbstractEventLoop,
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
    webcam_fps: Optional[float] = None,
) -> RTCPeerConnectionWithTrack:
    webcam_stream_track = WebcamStreamTrack(
        asyncio_loop=asyncio_loop,
        webcam_frames_grabber_queue=webcam_frames_grabber_queue,
        peer_frames_queue=peer_frames_queue,
        webcam_fps=webcam_fps,
    )

    if webrtc_turn_config:
        turn_server = RTCIceServer(
            urls=[webrtc_turn_config.urls],
            username=webrtc_turn_config.username,
            credential=webrtc_turn_config.credential,
        )
        peer_connection = RTCPeerConnectionWithTrack(
            webcam_stream_track=webcam_stream_track,
            configuration=RTCConfiguration(iceServers=[turn_server]),
        )
    else:
        peer_connection = RTCPeerConnectionWithTrack(
            webcam_stream_track=webcam_stream_track,
        )
    relay = MediaRelay()

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("track received")
        webcam_stream_track.set_track(track=relay.subscribe(track))

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Conn state: %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            webcam_stream_track.close()
            await webcam_frames_grabber_queue.async_put(None)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WebRTC webcam stream")
    parser.add_argument("--workflow-id", required=True, type=str)
    parser.add_argument("--workspace-id", required=True, type=str)
    parser.add_argument("--inference-server-url", required=True, type=str)
    parser.add_argument("--webrtc-endpoint", required=True, type=str, default="initialise_webrtc")
    parser.add_argument("--api-key", required=True, type=str)
    parser.add_argument("--show-camera-preview", required=False, action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    workflow_specification = get_workflow_specification(
        api_key=args.api_key,
        workspace_id=args.workspace_id,
        workflow_id=args.workflow_id,
    )

    def start_loop(loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.run_forever()

    loop = asyncio.new_event_loop()
    t = Thread(target=start_loop, args=(loop,), daemon=True)
    t.start()
    asyncio.set_event_loop(loop)

    webcam_frames_grabber = WebcamFramesGrabber(async_loop=loop, maxsize=10)
    webcam_frames_grabber.start()
    while webcam_frames_grabber._state != WebcamFrameGrabberState.STARTED:
        time.sleep(0.1)
    fps = webcam_frames_grabber.get_fps()

    peer_frames_queue = SyncAsyncQueue(loop=loop, maxsize=10)

    future = asyncio.run_coroutine_threadsafe(
        init_rtc_peer_connection_with_local_description(
            webcam_frames_grabber_queue=webcam_frames_grabber._frames_sink_queue,
            peer_frames_queue=peer_frames_queue,
            asyncio_loop=loop,
            webcam_fps=fps,
        ),
        loop,
    )
    peer_connection: RTCPeerConnectionWithTrack = future.result()

    output_name = workflow_specification.get("outputs")[0].get("name", "")
    request = InitialiseWebRTCPipelinePayload(
        video_configuration=VideoConfiguration(
            type="VideoConfiguration",
            video_reference="",
            max_fps=None,
            source_buffer_filling_strategy=BufferFillingStrategy.DROP_OLDEST,
            source_buffer_consumption_strategy=BufferConsumptionStrategy.EAGER,
            video_source_properties=None,
            batch_collection_timeout=None,
        ),
        processing_configuration=WorkflowConfiguration(
            type="WorkflowConfiguration",
            workflow_id=args.workflow_id,
            workspace_name=args.workspace_id,
            image_input_name="image",
            workflows_parameters={},
            workflows_thread_pool_workers=4,
            cancel_thread_pool_tasks_on_exit=True,
            video_metadata_input_name="video_metadata",
        ),
        sink_configuration=MemorySinkConfiguration(
            type="MemorySinkConfiguration",
            results_buffer_size=64,
        ),
        api_key=args.api_key,
        decoding_buffer_size=5,
        webrtc_peer_timeout=10,
        webcam_fps=fps,
        webrtc_offer=WebRTCOffer(
            type=peer_connection.localDescription.type,
            sdp=peer_connection.localDescription.sdp,
        ),
        stream_output=["video"],
        data_output=["preds"],
    )

    response = requests.post(
        urllib.parse.urljoin(
            args.inference_server_url, args.webrtc_endpoint
        ),
        json=request.model_dump(),
        verify=False,
    )
    webrtc_answer = response.json()
    if response.status_code != 200:
        raise Exception(f"Failed to initialise WebRTC pipeline: {response.text}")

    future = asyncio.run_coroutine_threadsafe(
        peer_connection.setRemoteDescription(
            RTCSessionDescription(sdp=webrtc_answer["sdp"], type=webrtc_answer["type"])
        ),
        loop,
    )
    future.result()

    while webcam_frames_grabber._state == WebcamFrameGrabberState.STARTED:
        try:
            frame: VideoFrame = peer_frames_queue.sync_get(timeout=0.005)
        except TimeoutError:
            logger.debug("Waiting for frame from peer")
            continue
        if not frame:
            break
        img = frame.to_ndarray(format="bgr24")
        cv.imshow("Peer", img)
        key = cv.waitKey(1)

        if key == ord("q"):
            break

        if key in (ord(n) for n in "1234567890abcdefghijkz") and (
            not peer_connection.data_channel
            or peer_connection.data_channel.readyState != "open"
        ):
            logger.error("Data channel not open")
            continue

        if key in (ord(n) for n in "1234567890"):
            if key == ord("0"):
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

        if key in (ord(n) for n in "abcdefghijkz"):
            if key == ord("z"):
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
    webcam_frames_grabber.stop()
    future = asyncio.run_coroutine_threadsafe(peer_connection.close(), loop)
    future.result()
    loop.stop()
    t.join()


main()
