import asyncio
import datetime
import json
import logging
from multiprocessing import Process, Pipe, get_start_method, set_start_method
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCDataChannel,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay, MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack, RTCRtpReceiver
from av import VideoFrame
from av import logging as av_logging

from inference.core import logger
from inference.core.env import DEBUG_AIORTC_QUEUES, DEBUG_WEBRTC_PROCESSING_LATENCY
from inference.core.interfaces.camera.entities import (
    SourceProperties,
    StatusUpdate,
    UpdateSeverity,
    VideoFrameProducer,
)
from inference.core.interfaces.stream.inference_pipeline import (
    INFERENCE_THREAD_FINISHED_EVENT,
)
from inference.core.interfaces.stream.watchdog import BasePipelineWatchDog
from inference.core.interfaces.stream_manager.manager_app.entities import (
    WebRTCData,
    WebRTCOffer,
    WebRTCTURNConfig,
)
from inference.core.utils.async_utils import Queue as SyncAsyncQueue
from inference.core.utils.function import experimental
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData

logging.getLogger("aiortc").setLevel(logging.WARNING)


def overlay_text_on_np_frame(frame: np.ndarray, text: List[str]):
    for i, l in enumerate(text):
        frame = cv.putText(
            frame,
            l,
            (10, 20 + 30 * i),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    return frame


def get_frame_from_workflow_output(
    workflow_output: Dict[str, Union[WorkflowImageData, Any]], frame_output_key: str
) -> Optional[np.ndarray]:
    latency: Optional[datetime.timedelta] = None
    np_image: Optional[np.ndarray] = None

    step_output = workflow_output.get(frame_output_key)
    if isinstance(step_output, WorkflowImageData):
        if (
            DEBUG_WEBRTC_PROCESSING_LATENCY
            and step_output.video_metadata
            and step_output.video_metadata.frame_timestamp is not None
        ):
            latency = (
                datetime.datetime.now() - step_output.video_metadata.frame_timestamp
            )
        np_image = step_output.numpy_image
    elif isinstance(step_output, dict):
        for frame_output in step_output.values():
            if isinstance(frame_output, WorkflowImageData):
                if (
                    DEBUG_WEBRTC_PROCESSING_LATENCY
                    and frame_output.video_metadata
                    and frame_output.video_metadata.frame_timestamp is not None
                ):
                    latency = (
                        datetime.datetime.now()
                        - frame_output.video_metadata.frame_timestamp
                    )
                np_image = frame_output.numpy_image

    # logger.warning since inference pipeline is noisy on INFO level
    if DEBUG_WEBRTC_PROCESSING_LATENCY and latency is not None:
        logger.warning("Processing latency: %ss", latency.total_seconds())

    return np_image


class VideoTransformTrack(VideoStreamTrack):
    def __init__(
        self,
        to_inference_queue: "SyncAsyncQueue[VideoFrame]",
        from_inference_queue: "SyncAsyncQueue[np.ndarray]",
        asyncio_loop: asyncio.AbstractEventLoop,
        fps_probe_frames: int,
        webcam_fps: Optional[float] = None,
        media_relay: Optional[MediaRelay] = None,
        drain_remote_stream_track: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop

        self._reader_task: Optional[asyncio.Task] = None

        self.track: Optional[RemoteStreamTrack] = None
        self._track_active: bool = False

        self._id = time.time_ns()
        self._processed = 0

        self.to_inference_queue: "SyncAsyncQueue[VideoFrame]" = to_inference_queue
        self.from_inference_queue: "SyncAsyncQueue[np.ndarray]" = from_inference_queue

        self._fps_probe_frames = fps_probe_frames
        self._probe_count: int = 0
        self._fps_probe_t1: Optional[float] = None
        self._fps_probe_t2: Optional[float] = None
        self.incoming_stream_fps: Optional[float] = webcam_fps

        self._last_processed_frame: Optional[VideoFrame] = None

        self._av_logging_set: bool = False

        self._last_queue_log_time: float = 0.0

        self._drain_remote_stream_track = drain_remote_stream_track

    def set_track(
        self,
        track: RemoteStreamTrack,
        relay_track: Optional[RemoteStreamTrack] = None,
        rtp_receiver: Optional[RTCRtpReceiver] = None,
    ):
        if not self.track:
            self.track = track
        self._relay_track = relay_track
        self._rtp_receiver = rtp_receiver

    def start(self):
        self._track_active = True
        self._reader_task = self._loop.create_task(
            self._raw_frames_reader_loop(), name="raw_frames_reader"
        )

    def stop(self):
        self._track_active = False
        self._reader_task.cancel()

    def close(self):
        self.stop()
        self._track_active = False

    async def _raw_frames_reader_loop(self):
        try:
            while self._track_active:
                current_time = time.time()
                # logger.warning since inference pipeline is noisy on INFO level
                if (
                    DEBUG_AIORTC_QUEUES
                    and current_time - self._last_queue_log_time >= 5.0
                ):
                    logger.warning("=== AIORTC QUEUE SIZES ===")

                    logger.warning(
                        "from_inference_queue: %s",
                        self.from_inference_queue._queue.qsize(),
                    )
                    logger.warning(
                        "to_inference_queue: %s", self.to_inference_queue._queue.qsize()
                    )

                    if self.track and hasattr(self.track, "_queue"):
                        logger.warning(
                            f"RemoteStreamTrack._queue: {self.track._queue.qsize()} (UNBOUNDED!)"
                        )

                    if (
                        self._relay_track
                        and hasattr(self._relay_track, "_queue")
                        and self._relay_track._queue
                    ):
                        logger.warning(
                            f"RelayStreamTrack._queue: {self._relay_track._queue.qsize()} (UNBOUNDED!)"
                        )

                    if self._rtp_receiver:
                        if hasattr(
                            self._rtp_receiver, "_RTCRtpReceiver__decoder_queue"
                        ):
                            decoder_queue_size = (
                                self._rtp_receiver._RTCRtpReceiver__decoder_queue.qsize()
                            )
                            logger.warning(
                                f"RTCRtpReceiver.__decoder_queue: {decoder_queue_size} (UNBOUNDED!)"
                            )

                        if hasattr(
                            self._rtp_receiver, "_RTCRtpReceiver__jitter_buffer"
                        ):
                            jb = self._rtp_receiver._RTCRtpReceiver__jitter_buffer
                            if hasattr(jb, "_packets"):
                                filled = sum(1 for p in jb._packets if p is not None)
                                logger.warning(
                                    f"RTCRtpReceiver.JitterBuffer: {filled}/{jb._capacity} (bounded)"
                                )

                    logger.warning("========================")
                    self._last_queue_log_time = current_time

                frame: VideoFrame = await self.track.recv()
                if self._drain_remote_stream_track:
                    # Prevent spam in log, only inform about major drains
                    if self.track._queue.qsize() > 10:
                        logger.warning(
                            "Draining RemoteStreamTrack._queue: %s (UNBOUNDED)",
                            self.track._queue.qsize(),
                        )
                        while self.track._queue.qsize() > 0:
                            frame: VideoFrame = await self.track.recv()

                if self.incoming_stream_fps is None:
                    self._probe_count += 1
                    now = time.time()
                    if self._probe_count == 1:
                        self._fps_probe_t1 = now
                    elif self._probe_count == self._fps_probe_frames:
                        self._fps_probe_t2 = now
                        dt = max(1e-6, (self._fps_probe_t2 - self._fps_probe_t1))
                        self.incoming_stream_fps = (self._fps_probe_frames - 1) / dt
                        logger.info("Incoming stream fps: %s", self.incoming_stream_fps)

                await self.to_inference_queue.async_put(frame)
            logger.info("WebRTC reader loop finished")

        except asyncio.CancelledError:
            logger.info("WebRTC reader loop cancelled")
        except MediaStreamError:
            if not self.complete:
                logger.error("WebRTC reader loop finished due to MediaStreamError")
        except Exception as exc:
            logger.error("Error in WebRTC reader loop: %s", exc)

    async def recv(self):
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True
        self._processed += 1

        np_frame: Optional[np.ndarray] = None
        np_frame = await self.from_inference_queue.async_get()

        if np_frame is None:
            if self._last_processed_frame:
                new_frame = self._last_processed_frame
            else:
                np_frame = overlay_text_on_np_frame(
                    np.zeros((720, 1280, 3), dtype=np.uint8),
                    ["Inference pipeline is starting..."],
                )
                new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
        else:
            new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
            self._last_processed_frame = new_frame

        # below method call may sleep
        pts, time_base = await self.next_timestamp()
        new_frame.pts = pts
        new_frame.time_base = time_base

        return new_frame


class WebRTCVideoFrameProducer(VideoFrameProducer):
    @experimental(
        reason="Usage of WebRTCVideoFrameProducer with `InferencePipeline` is an experimental feature."
        "Please report any issues here: https://github.com/roboflow/inference/issues"
    )
    def __init__(
        self,
        to_inference_queue: "SyncAsyncQueue[VideoFrame]",
        webrtc_video_transform_track: VideoTransformTrack,
    ):
        self.to_inference_queue: "SyncAsyncQueue[VideoFrame]" = to_inference_queue
        self._w: Optional[int] = None
        self._h: Optional[int] = None
        self._video_transform_track = webrtc_video_transform_track

    def grab(self) -> bool:
        return self._video_transform_track._track_active

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame: Optional[VideoFrame] = self.to_inference_queue.sync_get()

        if frame is None:
            return False, None

        img = frame.to_ndarray(format="bgr24")
        return True, img

    def release(self):
        self._video_transform_track.close()

    def isOpened(self) -> bool:
        return self._video_transform_track._track_active

    def discover_source_properties(self) -> SourceProperties:
        while not self._video_transform_track.incoming_stream_fps:
            time.sleep(0.1)
        return SourceProperties(
            width=self._w,
            height=self._h,
            total_frames=-1,
            is_file=False,
            fps=self._video_transform_track.incoming_stream_fps,
            is_reconnectable=False,
        )

    def initialize_source_properties(self, properties: Dict[str, float]):
        pass


class RTCPeerConnectionWithFPS(RTCPeerConnection):
    def __init__(
        self,
        video_transform_track: VideoTransformTrack,
        asyncio_loop: asyncio.AbstractEventLoop,
        stream_output: Optional[str] = None,
        data_output: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop
        self.video_transform_track: VideoTransformTrack = video_transform_track
        self._consumers_signalled: bool = False
        self.stream_output: Optional[str] = stream_output
        self.data_output: Optional[str] = data_output
        self.data_channel: Optional[RTCDataChannel] = None


class WebRTCPipelineWatchDog(BasePipelineWatchDog):
    def __init__(
        self,
        webrtc_peer_connection: RTCPeerConnectionWithFPS,
        asyncio_loop: asyncio.AbstractEventLoop,
    ):
        super().__init__()
        self._webrtc_peer_connection = webrtc_peer_connection
        self._asyncio_loop = asyncio_loop

    def on_status_update(self, status_update: StatusUpdate) -> None:
        if status_update.event_type == INFERENCE_THREAD_FINISHED_EVENT:
            logger.debug(
                "InferencePipeline thread finished, closing WebRTC peer connection"
            )
            asyncio.run_coroutine_threadsafe(
                self._webrtc_peer_connection.close(), self._asyncio_loop
            )
        if status_update.severity.value <= UpdateSeverity.DEBUG.value:
            return None
        self._stream_updates.append(status_update)


async def init_rtc_peer_connection(
    webrtc_offer: WebRTCOffer,
    to_inference_queue: "SyncAsyncQueue[VideoFrame]",
    from_inference_queue: "SyncAsyncQueue[np.ndarray]",
    asyncio_loop: asyncio.AbstractEventLoop,
    fps_probe_frames: int,
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
    webrtc_realtime_processing: bool = True,
    webcam_fps: Optional[float] = None,
    stream_output: Optional[str] = None,
    data_output: Optional[str] = None,
) -> RTCPeerConnectionWithFPS:
    relay = MediaRelay()
    video_transform_track = VideoTransformTrack(
        to_inference_queue=to_inference_queue,
        from_inference_queue=from_inference_queue,
        asyncio_loop=asyncio_loop,
        webcam_fps=webcam_fps,
        fps_probe_frames=fps_probe_frames,
        media_relay=relay,
        drain_remote_stream_track=webrtc_realtime_processing,
    )

    if webrtc_turn_config:
        turn_server = RTCIceServer(
            urls=[webrtc_turn_config.urls],
            username=webrtc_turn_config.username,
            credential=webrtc_turn_config.credential,
        )
        peer_connection = RTCPeerConnectionWithFPS(
            video_transform_track=video_transform_track,
            configuration=RTCConfiguration(iceServers=[turn_server]),
            asyncio_loop=asyncio_loop,
            stream_output=stream_output,
            data_output=data_output,
        )
    else:
        peer_connection = RTCPeerConnectionWithFPS(
            video_transform_track=video_transform_track,
            asyncio_loop=asyncio_loop,
            stream_output=stream_output,
            data_output=data_output,
        )

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("track received")
        # can be called with buffered=False
        relay_track = relay.subscribe(track)

        # Find the RTCRtpReceiver for this track
        rtp_receiver = None
        for transceiver in peer_connection.getTransceivers():
            if transceiver.receiver.track == track:
                rtp_receiver = transceiver.receiver
                break

        video_transform_track.set_track(
            track=track, relay_track=relay_track, rtp_receiver=rtp_receiver
        )
        video_transform_track.start()
        peer_connection.addTrack(video_transform_track)

    @peer_connection.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        logger.info("Data channel %s received", channel.label)

        @channel.on("message")
        def on_message(message):
            logger.info("Data channel message received: %s", message)
            try:
                message = WebRTCData(**json.loads(message))
            except json.JSONDecodeError:
                logger.error("Failed to decode webrtc data payload: %s", message)
                return
            if message.stream_output is not None:
                peer_connection.stream_output = message.stream_output or None
            if message.data_output is not None:
                peer_connection.data_output = message.data_output or None

        peer_connection.data_channel = channel

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
            logger.debug("Signalling WebRTC termination to frames consumer")
            try:
                await to_inference_queue.async_put_nowait(None)
            except asyncio.QueueFull:
                await to_inference_queue.async_get_nowait()
                await to_inference_queue.async_put_nowait(None)
            peer_connection._consumers_signalled = True
            logger.info("'connectionstatechange' event handler finished")

    await peer_connection.setRemoteDescription(
        RTCSessionDescription(sdp=webrtc_offer.sdp, type=webrtc_offer.type)
    )
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    logger.debug(f"WebRTC connection status: {peer_connection.connectionState}")

    return peer_connection


class RTCPeerConnectionWithLoop(RTCPeerConnection):
    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop


class VideoTransformTrackWithLoop(VideoStreamTrack):
    def __init__(
        self,
        asyncio_loop: asyncio.AbstractEventLoop,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._loop = asyncio_loop

        self.track: Optional[RemoteStreamTrack] = None
        self._track_active: bool = False

        self._av_logging_set: bool = False

    def set_track(
        self,
        track: RemoteStreamTrack,
    ):
        if not self.track:
            self.track = track

    def close(self):
        self._track_active = False

    async def recv(self):
        # Silencing swscaler warnings in multi-threading environment
        if not self._av_logging_set:
            av_logging.set_libav_level(av_logging.ERROR)
            self._av_logging_set = True

        frame: VideoFrame = await self.track.recv()
        return frame


async def _wait_ice_complete(peer_connection: RTCPeerConnectionWithLoop, timeout=2.0):
    if peer_connection.iceGatheringState == "complete":
        return
    fut = asyncio.get_running_loop().create_future()

    @peer_connection.on("icegatheringstatechange")
    def _():
        if not fut.done() and peer_connection.iceGatheringState == "complete":
            fut.set_result(True)

    try:
        await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError:
        pass


async def init_rtc_peer_connection_with_loop(
    webrtc_offer: WebRTCOffer,
    send_answer: Callable[[Any], None],
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
    asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> RTCPeerConnectionWithLoop:
    relay = MediaRelay()
    video_transform_track = VideoTransformTrackWithLoop(
        asyncio_loop=asyncio_loop,
    )

    if webrtc_turn_config:
        turn_server = RTCIceServer(
            urls=[webrtc_turn_config.urls],
            username=webrtc_turn_config.username,
            credential=webrtc_turn_config.credential,
        )
        peer_connection = RTCPeerConnectionWithLoop(
            configuration=RTCConfiguration(iceServers=[turn_server]),
            asyncio_loop=asyncio_loop,
        )
    else:
        peer_connection = RTCPeerConnectionWithLoop(
            asyncio_loop=asyncio_loop,
        )

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.info("track received")
        # can be called with buffered=False
        relay.subscribe(track)
        video_transform_track.set_track(track=track)
        peer_connection.addTrack(video_transform_track)

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.info("Connection state is %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
        logger.info("'connectionstatechange' event handler finished")

    await peer_connection.setRemoteDescription(
        RTCSessionDescription(sdp=webrtc_offer.sdp, type=webrtc_offer.type)
    )
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)

    logger.debug(f"WebRTC connection status: {peer_connection.connectionState}")

    # 3) (Optional) wait for ICE gathering to finish so SDP has candidates
    await _wait_ice_complete(peer_connection, timeout=2.0)

    # Send the final answer back to parent (one-shot), then keep running
    send_answer(
        {
            "sdp": peer_connection.localDescription.sdp,
            "type": peer_connection.localDescription.type,
        }
    )

    # Stay alive until the peer disconnects
    while peer_connection.connectionState not in {"failed", "closed"}:
        await asyncio.sleep(0.2)


def rtc_peer_connection_process(
    offer_sdp: str,
    offer_type: str,
    turn_urls: str,
    turn_username: str,
    turn_credential: str,
    answer_conn=None,
):
    def send_answer(obj):
        answer_conn.send(obj)
        answer_conn.close()

    offer = WebRTCOffer(type=offer_type, sdp=offer_sdp)
    turn_config = WebRTCTURNConfig(
        urls=turn_urls, username=turn_username, credential=turn_credential
    )
    asyncio.run(
        init_rtc_peer_connection_with_loop(
            webrtc_offer=offer,
            webrtc_turn_config=turn_config,
            send_answer=send_answer,
        )
    )


# Prefer 'spawn' so we don't fork an active event loop (portable & safe)
try:
    if get_start_method(allow_none=True) != "spawn":
        set_start_method("spawn", force=True)
except RuntimeError:
    pass


async def start_worker(
    webrtc_offer: WebRTCOffer,
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
):
    parent_conn, child_conn = Pipe(duplex=False)
    p = Process(
        target=rtc_peer_connection_process,
        kwargs={
            "offer_sdp": webrtc_offer.sdp,
            "offer_type": webrtc_offer.type,
            "turn_urls": webrtc_turn_config.urls,
            "turn_username": webrtc_turn_config.username,
            "turn_credential": webrtc_turn_config.credential,
            "answer_conn": child_conn,
        },
        daemon=False,
    )
    p.start()
    child_conn.close()

    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(None, parent_conn.recv)
    parent_conn.close()

    # answer = json.loads(answer_line.decode("utf-8").strip())
    return p.pid, p, answer
