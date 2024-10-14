import asyncio
import concurrent.futures
import time
from collections import deque
from threading import Event, Lock
from typing import Deque, Dict, Optional, Tuple

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc.contrib.media import MediaRelay
from aiortc.mediastreams import MediaStreamError
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame

from inference.core import logger
from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)
from inference.core.interfaces.stream_manager.manager_app.entities import WebRTCOffer
from inference.core.utils.async_utils import async_lock
from inference.core.utils.function import experimental


class VideoTransformTrack(VideoStreamTrack):
    def __init__(
        self,
        to_inference_queue: Deque,
        to_inference_lock: Lock,
        from_inference_queue: Deque,
        from_inference_lock: Lock,
        webrtc_peer_timeout: float = 1,
        fps_probe_frames: int = 10,
        webcam_fps: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if not webrtc_peer_timeout:
            webrtc_peer_timeout = 1
        self.webrtc_peer_timeout: float = webrtc_peer_timeout

        self.track: Optional[RemoteStreamTrack] = None
        self._id = time.time_ns()
        self._processed = 0
        self.to_inference_queue: Deque = to_inference_queue
        self.from_inference_queue: Deque = from_inference_queue
        self.to_inference_lock: Lock = to_inference_lock
        self.from_inference_lock: Lock = from_inference_lock
        self._pool = concurrent.futures.ThreadPoolExecutor()
        self._track_active: bool = True
        self._fps_probe_frames = fps_probe_frames
        self.incoming_stream_fps: Optional[float] = webcam_fps

    def set_track(self, track: RemoteStreamTrack):
        if not self.track:
            self.track = track

    def close(self):
        self._track_active = False

    async def recv(self):
        if not self.incoming_stream_fps:
            logger.debug("Probing incoming stream FPS")
            t1 = 0
            t2 = 0
            for i in range(self._fps_probe_frames):
                try:
                    frame: VideoFrame = await asyncio.wait_for(
                        self.track.recv(), self.webrtc_peer_timeout
                    )
                except (asyncio.TimeoutError, MediaStreamError):
                    logger.info(
                        "Timeout while waiting to receive frames sent through webrtc peer connection; assuming peer disconnected."
                    )
                    self.close()
                    raise MediaStreamError
                # drop first frame
                if i == 1:
                    t1 = time.time()
            t2 = time.time()
            if t1 == t2:
                logger.info(
                    "All frames probed in the same time - could not calculate fps."
                )
                raise MediaStreamError
            self.incoming_stream_fps = 9 / (t2 - t1)
            logger.debug("Incoming stream fps: %s", self.incoming_stream_fps)

        try:
            frame: VideoFrame = await asyncio.wait_for(
                self.track.recv(), self.webrtc_peer_timeout
            )
        except (asyncio.TimeoutError, MediaStreamError):
            logger.info(
                "Timeout while waiting to receive frames sent through webrtc peer connection; assuming peer disconnected."
            )
            self.close()
            raise MediaStreamError
        img = frame.to_ndarray(format="bgr24")

        dropped = 0
        async with async_lock(lock=self.to_inference_lock, pool=self._pool):
            self.to_inference_queue.appendleft(img)
        while self._track_active and not self.from_inference_queue:
            try:
                frame: VideoFrame = await asyncio.wait_for(
                    self.track.recv(), self.webrtc_peer_timeout
                )
            except (asyncio.TimeoutError, MediaStreamError):
                self.close()
                logger.info(
                    "Timeout while waiting to receive frames sent through webrtc peer connection; assuming peer disconnected."
                )
                raise MediaStreamError
            dropped += 1
        async with async_lock(lock=self.from_inference_lock, pool=self._pool):
            res = self.from_inference_queue.pop()

        logger.debug("Dropping %s every inference", dropped)
        new_frame = VideoFrame.from_ndarray(res, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        self._processed += 1
        return new_frame


class WebRTCVideoFrameProducer(VideoFrameProducer):
    @experimental(
        reason="Usage of WebRTCVideoFrameProducer with `InferencePipeline` is an experimental feature."
        "Please report any issues here: https://github.com/roboflow/inference/issues"
    )
    def __init__(
        self,
        to_inference_queue: deque,
        to_inference_lock: Lock,
        stop_event: Event,
        webrtc_video_transform_track: VideoTransformTrack,
    ):
        self.to_inference_queue: deque = to_inference_queue
        self.to_inference_lock: Lock = to_inference_lock
        self._stop_event = stop_event
        self._w: Optional[int] = None
        self._h: Optional[int] = None
        self._video_transform_track = webrtc_video_transform_track
        self._is_opened = True

    def grab(self) -> bool:
        return self._is_opened

    def retrieve(self) -> Tuple[bool, np.ndarray]:
        while not self._stop_event.is_set() and not self.to_inference_queue:
            time.sleep(0.1)
        if self._stop_event.is_set():
            logger.info("Received termination signal, closing.")
            self._is_opened = False
            return False, None
        with self.to_inference_lock:
            img = self.to_inference_queue.pop()
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


async def init_rtc_peer_connection(
    webrtc_offer: WebRTCOffer,
    to_inference_queue: Deque,
    to_inference_lock: Lock,
    from_inference_queue: Deque,
    from_inference_lock: Lock,
    webrtc_peer_timeout: float,
    feedback_stop_event: Event,
    webcam_fps: Optional[float] = None,
) -> RTCPeerConnectionWithFPS:
    video_transform_track = VideoTransformTrack(
        to_inference_lock=to_inference_lock,
        to_inference_queue=to_inference_queue,
        from_inference_lock=from_inference_lock,
        from_inference_queue=from_inference_queue,
        webrtc_peer_timeout=webrtc_peer_timeout,
        webcam_fps=webcam_fps,
    )

    peer_connection = RTCPeerConnectionWithFPS(
        video_transform_track=video_transform_track
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
            video_transform_track.close()
            logger.info("Signalling WebRTC termination to the caller")
            feedback_stop_event.set()
            await peer_connection.close()

    await peer_connection.setRemoteDescription(
        RTCSessionDescription(sdp=webrtc_offer.sdp, type=webrtc_offer.type)
    )
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    logger.debug(f"WebRTC connection status: {peer_connection.connectionState}")

    return peer_connection
