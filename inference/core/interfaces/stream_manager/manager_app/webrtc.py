import asyncio
import concurrent.futures
import time
from multiprocessing.synchronize import Lock as LockType
from typing import Deque, Optional

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
        to_inference_lock: LockType,
        from_inference_queue: Deque,
        from_inference_lock: LockType,
    ):
        self.track: Optional[RemoteStreamTrack] = None
        self._processed = 0
        self._id = time.time_ns()
        self.to_inference_queue: Deque = to_inference_queue
        self.from_inference_queue: Deque = from_inference_queue
        self.to_inference_lock: LockType = to_inference_lock
        self.from_inference_lock: LockType = from_inference_lock
        self._loop = asyncio.get_event_loop()
        self._pool = concurrent.futures.ThreadPoolExecutor()
        self._img = None
        self.no_more_frames: bool = False

    async def set_track(self, track: RemoteStreamTrack):
        if not self.track:
            self.track = track

    async def recv(self):
        try:
            frame: VideoFrame = await asyncio.wait_for(self.track.recv(), 10)
        except Exception as exc:
            logger.info("No more frames sent through webrtc peer connection")
            self.no_more_frames = True
            return
        img = frame.to_ndarray(format="bgr24")

        dropped = 0
        async with async_lock(lock=self.to_inference_lock, pool=self._pool):
            self.to_inference_queue.appendleft(img)
        while not self.from_inference_queue:
            frame: VideoFrame = await asyncio.wait_for(self.track.recv(), 10)
            dropped += 1
        async with async_lock(lock=self.from_inference_lock, pool=self._pool):
            res = self.from_inference_queue.pop()

        logger.debug("Dropping %s every inference", dropped)
        self._processed += 1
        new_frame = VideoFrame.from_ndarray(res, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame


async def create_rtc_peer_connection(
    webrtc_offer: WebRTCOffer,
    to_inference_queue: Deque,
    to_inference_lock: LockType,
    from_inference_queue: Deque,
    from_inference_lock: LockType,
) -> RTCPeerConnection:
    pc = RTCPeerConnection()
    relay = MediaRelay()

    video_transform_track = VideoTransformTrack(
        to_inference_lock=to_inference_lock,
        to_inference_queue=to_inference_queue,
        from_inference_lock=from_inference_lock,
        from_inference_queue=from_inference_queue,
    )

    @pc.on("track")
    def on_track(track):
        logger.debug("Track %s received", track.kind)
        video_transform_track.set_track(track=relay.subscribe(track))
        pc.addTrack(video_transform_track)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        logger.debug("Connection state is %s", pc.connectionState)
        if pc.connectionState in {"failed", "closed"}:
            logger.info("Stopping WebRTC server")
            await pc.close()
            global global_pc
            global_pc = None

    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=webrtc_offer.sdp, type=webrtc_offer.type)
    )
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.debug(f"WebRTC connection status: {pc.connectionState}")

    return pc
