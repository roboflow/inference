import asyncio
import concurrent.futures
import time
from threading import Lock
from typing import Deque, Optional

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc.rtcrtpreceiver import RemoteStreamTrack
from av import VideoFrame

from inference.core.interfaces.stream_manager.manager_app.entities import WebRTCOffer
from inference.core.logger import logger
from inference.core.utils.async_utils import async_lock


class VideoTransformTrack(MediaStreamTrack):
    kind = "video"

    def __init__(
        self,
        to_inference_queue: Deque,
        to_inference_lock: Lock,
        from_inference_queue: Deque,
        from_inference_lock: Lock,
    ):
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
        self.peer_timed_out: bool = False
        self.dummy_frame: Optional[VideoFrame] = None
        self.last_pts = 0
        self.last_time_base = 0

    def set_track(self, track: RemoteStreamTrack):
        if not self.track:
            self.track = track

    async def recv(self):
        try:
            frame: VideoFrame = await asyncio.wait_for(self.track.recv(), 10)
            self.last_pts = frame.pts
            self.last_time_base = frame.time_base
            if not self.dummy_frame:
                self.dummy_frame = VideoFrame.from_ndarray(
                    np.zeros_like(frame.to_ndarray(format="bgr24")), format="bgr24"
                )
        except asyncio.TimeoutError:
            logger.info("No more frames sent through webrtc peer connection")
            self.peer_timed_out = True
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
        while not self.from_inference_queue:
            try:
                frame: VideoFrame = await asyncio.wait_for(self.track.recv(), 10)
                self.last_pts = frame.pts
                self.last_time_base = frame.time_base
            except asyncio.TimeoutError:
                logger.info("No more frames sent through webrtc peer connection")
                self.peer_timed_out = True
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
    peer_connection: RTCPeerConnection,
    peer_connection_ready_event: asyncio.Event,
    webrtc_offer: WebRTCOffer,
    to_inference_queue: Deque,
    to_inference_lock: Lock,
    from_inference_queue: Deque,
    from_inference_lock: Lock,
) -> RTCPeerConnection:
    relay = MediaRelay()

    video_transform_track = VideoTransformTrack(
        to_inference_lock=to_inference_lock,
        to_inference_queue=to_inference_queue,
        from_inference_lock=from_inference_lock,
        from_inference_queue=from_inference_queue,
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
            await peer_connection.close()

    await peer_connection.setRemoteDescription(
        RTCSessionDescription(sdp=webrtc_offer.sdp, type=webrtc_offer.type)
    )
    answer = await peer_connection.createAnswer()
    await peer_connection.setLocalDescription(answer)
    logger.debug(f"WebRTC connection status: {peer_connection.connectionState}")

    peer_connection_ready_event.set()

    while not video_transform_track.peer_timed_out:
        await asyncio.sleep(1)
    logger.info("We are done with this WebRTC peer")
