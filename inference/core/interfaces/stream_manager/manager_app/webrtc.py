import asyncio
import concurrent.futures
import datetime
import time
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
    VideoStreamTrack,
)
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame
from av import logging as av_logging

from inference.core import logger
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
    WebRTCOffer,
    WebRTCTURNConfig,
)
from inference.core.utils.async_utils import Queue as SyncAsyncQueue
from inference.core.utils.function import experimental

FALLBACK_FPS: float = 10


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


class VideoTransformTrack(VideoStreamTrack):
    def __init__(
        self,
        to_inference_queue: "SyncAsyncQueue[VideoFrame]",
        from_inference_queue: "SyncAsyncQueue[np.ndarray]",
        asyncio_loop: asyncio.AbstractEventLoop,
        processing_timeout: float,
        fps_probe_frames: int,
        min_consecutive_on_time: int,
        max_consecutive_timeouts: Optional[int] = None,
        webcam_fps: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.processing_timeout: float = processing_timeout

        self.track: Optional[RemoteStreamTrack] = None
        self._track_active: bool = True

        self._id = time.time_ns()
        self._processed = 0

        self.to_inference_queue: "SyncAsyncQueue[VideoFrame]" = to_inference_queue
        self.from_inference_queue: "SyncAsyncQueue[np.ndarray]" = from_inference_queue

        self._asyncio_loop = asyncio_loop
        self._pool = concurrent.futures.ThreadPoolExecutor()

        self._fps_probe_frames = fps_probe_frames
        self._fps_probe_t1: Optional[float] = None
        self._fps_probe_t2: Optional[float] = None
        self.incoming_stream_fps: Optional[float] = webcam_fps

        self._last_frame: Optional[VideoFrame] = None
        self._consecutive_timeouts: int = 0
        self._consecutive_on_time: int = 0
        self._max_consecutive_timeouts: Optional[int] = max_consecutive_timeouts
        self._min_consecutive_on_time: int = min_consecutive_on_time

        self._av_logging_set: bool = False

    def set_track(self, track: RemoteStreamTrack):
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
        self._processed += 1
        if not self.incoming_stream_fps:
            if not self._fps_probe_t1:
                logger.debug("Probing incoming stream FPS")
            if self._processed == 1:
                self._fps_probe_t1 = time.time()
            if self._processed == self._fps_probe_frames:
                self._fps_probe_t2 = time.time()
            if self._fps_probe_t1 == self._fps_probe_t2:
                logger.warning(
                    "All frames probed in the same time - could not calculate fps, assuming fallback %s FPS.",
                    FALLBACK_FPS,
                )
                self.incoming_stream_fps = FALLBACK_FPS
            elif self._fps_probe_t1 is not None and self._fps_probe_t2 is not None:
                self.incoming_stream_fps = (self._fps_probe_frames - 1) / (
                    self._fps_probe_t2 - self._fps_probe_t1
                )
                logger.info("Incoming stream fps: %s", self.incoming_stream_fps)

        if not await self.to_inference_queue.async_full():
            await self.to_inference_queue.async_put(frame)
        elif not self._last_frame:
            await self.to_inference_queue.async_get_nowait()
            await self.to_inference_queue.async_put_nowait(frame)

        np_frame: Optional[np.ndarray] = None
        try:
            np_frame = await self.from_inference_queue.async_get(
                timeout=self.processing_timeout
            )
            new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
            self._last_frame = new_frame

            if self._max_consecutive_timeouts:
                self._consecutive_on_time += 1
                if self._consecutive_on_time >= self._min_consecutive_on_time:
                    self._consecutive_timeouts = 0
        except asyncio.TimeoutError:
            if self._last_frame:
                if self._max_consecutive_timeouts:
                    self._consecutive_timeouts += 1
                    if self._consecutive_timeouts >= self._max_consecutive_timeouts:
                        self._consecutive_on_time = 0

        workflow_too_slow_message = [
            "Workflow is too heavy to process all frames on time..."
        ]
        if np_frame is None:
            if not self._last_frame:
                np_frame = overlay_text_on_np_frame(
                    frame.to_ndarray(format="bgr24"),
                    ["Inference pipeline is starting..."],
                )
                new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
            elif (
                self._max_consecutive_timeouts
                and self._consecutive_timeouts >= self._max_consecutive_timeouts
            ):
                np_frame = overlay_text_on_np_frame(
                    self._last_frame.to_ndarray(format="bgr24"),
                    workflow_too_slow_message,
                )
                new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
            else:
                new_frame = self._last_frame
        else:
            if (
                self._max_consecutive_timeouts
                and self._consecutive_timeouts >= self._max_consecutive_timeouts
            ):
                np_frame = overlay_text_on_np_frame(
                    self._last_frame.to_ndarray(format="bgr24"),
                    workflow_too_slow_message,
                )
                new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")
            else:
                new_frame = VideoFrame.from_ndarray(np_frame, format="bgr24")

        new_frame.pts = self._processed
        new_frame.time_base = frame.time_base

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
        self._is_opened = True

    def grab(self) -> bool:
        res = self.to_inference_queue.sync_get()
        if res is None:
            logger.debug("Received termination signal")
            return False
        return True

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        frame: VideoFrame = self.to_inference_queue.sync_get()
        if frame is None:
            logger.debug("Received termination signal")
            return False, None
        img = frame.to_ndarray(format="bgr24")

        return True, img

    def release(self):
        self._is_opened = False

    def isOpened(self) -> bool:
        return self._is_opened

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
    def __init__(self, video_transform_track: VideoTransformTrack, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_transform_track: VideoTransformTrack = video_transform_track
        self._consumers_signalled: bool = False


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
    processing_timeout: float,
    fps_probe_frames: int,
    max_consecutive_timeouts: int,
    min_consecutive_on_time: int,
    webrtc_turn_config: Optional[WebRTCTURNConfig] = None,
    webcam_fps: Optional[float] = None,
) -> RTCPeerConnectionWithFPS:
    video_transform_track = VideoTransformTrack(
        to_inference_queue=to_inference_queue,
        from_inference_queue=from_inference_queue,
        asyncio_loop=asyncio_loop,
        webcam_fps=webcam_fps,
        processing_timeout=processing_timeout,
        fps_probe_frames=fps_probe_frames,
        max_consecutive_timeouts=max_consecutive_timeouts,
        min_consecutive_on_time=min_consecutive_on_time,
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
        )
    else:
        peer_connection = RTCPeerConnectionWithFPS(
            video_transform_track=video_transform_track,
        )
    relay = MediaRelay()

    @peer_connection.on("track")
    def on_track(track: RemoteStreamTrack):
        logger.debug("Track %s received", track.kind)
        video_transform_track.set_track(track=relay.subscribe(track))
        peer_connection.addTrack(video_transform_track)

    @peer_connection.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.debug("Connection state is %s", peer_connection.connectionState)
        if peer_connection.connectionState in {"failed", "closed"}:
            logger.info("Stopping WebRTC peer")
            await peer_connection.close()
            logger.debug("Signalling WebRTC termination to frames consumer")
            if not await to_inference_queue.async_full():
                await to_inference_queue.async_put(None)
            else:
                await to_inference_queue.async_get_nowait()
                await to_inference_queue.async_put_nowait(None)
            peer_connection._consumers_signalled = True
            logger.debug("'connectionstatechange' event handler finished")

    await peer_connection.setRemoteDescription(
        RTCSessionDescription(sdp=webrtc_offer.sdp, type=webrtc_offer.type)
    )
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    logger.debug(f"WebRTC connection status: {peer_connection.connectionState}")

    return peer_connection
