import asyncio
import concurrent.futures
import time
from threading import Event, Lock
from typing import Deque, Optional

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame

from inference.core import logger
from inference.core.interfaces.stream_manager.manager_app.entities import WebRTCOffer
from inference.core.utils.async_utils import async_lock


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        to_inference_queue: Deque,
        to_inference_lock: Lock,
        from_inference_queue: Deque,
        from_inference_lock: Lock,
        webrtc_peer_timeout: float = 1,
    ):
        if not webrtc_peer_timeout:
            webrtc_peer_timeout = 1
        self.webrtc_peer_timeout: float = webrtc_peer_timeout

        self.track: Optional[RemoteStreamTrack] = None
        self._processed = 0
        self._id = time.time_ns()
        self.to_inference_queue: Deque = to_inference_queue
        self.from_inference_queue: Deque = from_inference_queue
        self.to_inference_lock: Lock = to_inference_lock
        self.from_inference_lock: Lock = from_inference_lock
        self._loop = asyncio.get_event_loop()
        self._pool = concurrent.futures.ThreadPoolExecutor()
        self._img = None
        self._track_active: bool = True
        self.dummy_frame: Optional[VideoFrame] = None
        self.last_pts = 0
        self.last_time_base = 0

    def set_track(self, track: RemoteStreamTrack):
        if not self.track:
            self.track = track

    def close(self):
        self._track_active = False

    async def recv(self):
        try:
            frame: VideoFrame = await asyncio.wait_for(
                self.track.recv(), self.webrtc_peer_timeout
            )
            self.last_pts = frame.pts
            self.last_time_base = frame.time_base
            if not self.dummy_frame:
                self.dummy_frame = VideoFrame.from_ndarray(
                    np.zeros_like(frame.to_ndarray(format="bgr24")), format="bgr24"
                )
        except asyncio.TimeoutError:
            logger.info(
                "Timeout while waiting to receive frames sent through webrtc peer connection; assuming peer disconnected."
            )
            self._track_active = False
            if self.dummy_frame:
                self.dummy_frame.pts = self.last_pts
                self.dummy_frame.time_base = self.last_time_base
                return self.dummy_frame
            return VideoFrame.from_ndarray(
                np.zeros(shape=(640, 480, 3), dtype=np.uint8), format="bgr24"
            )
        img = frame.to_ndarray(format="bgr24")

        dropped = 0
        async with async_lock(lock=self.to_inference_lock, pool=self._pool):
            self.to_inference_queue.appendleft(img)
        while self._track_active and not self.from_inference_queue:
            try:
                frame: VideoFrame = await asyncio.wait_for(
                    self.track.recv(), self.webrtc_peer_timeout
                )
                self.last_pts = frame.pts
                self.last_time_base = frame.time_base
            except asyncio.TimeoutError:
                logger.info(
                    "Timeout while waiting to receive frames sent through webrtc peer connection; assuming peer disconnected."
                )
                self._track_active = False
                if self.dummy_frame:
                    self.dummy_frame.pts = self.last_pts
                    self.dummy_frame.time_base = self.last_time_base
                    return self.dummy_frame
                return VideoFrame.from_ndarray(
                    np.zeros(shape=(640, 480, 3), dtype=np.uint8), format="bgr24"
                )
            dropped += 1
        async with async_lock(lock=self.from_inference_lock, pool=self._pool):
            res = self.from_inference_queue.pop()

        logger.debug("Dropping %s every inference", dropped)
        self._processed += 1
        new_frame = VideoFrame.from_ndarray(res, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def init_rtc_peer_connection(
    webrtc_offer: WebRTCOffer,
    to_inference_queue: Deque,
    to_inference_lock: Lock,
    from_inference_queue: Deque,
    from_inference_lock: Lock,
    webrtc_peer_timeout: float,
    feedback_stop_event: Event,
) -> RTCPeerConnection:
    peer_connection = RTCPeerConnection()
    relay = MediaRelay()

    video_transform_track = VideoTransformTrack(
        to_inference_lock=to_inference_lock,
        to_inference_queue=to_inference_queue,
        from_inference_lock=from_inference_lock,
        from_inference_queue=from_inference_queue,
        webrtc_peer_timeout=webrtc_peer_timeout,
    )

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
