import asyncio
import threading
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional

import av
import cv2
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
from av import VideoFrame
from queue import Queue

# Suppress FFmpeg colorspace conversion warnings
av.logging.set_level(av.logging.ERROR)


class _WebcamVideoTrack(VideoStreamTrack):
    """A VideoStreamTrack that pulls frames from OpenCV in the event loop."""

    def __init__(self, camera_index: int = 0, resolution: Optional[tuple[int, int]] = None):
        super().__init__()
        self._cap = cv2.VideoCapture(camera_index)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam")
        if resolution:
            w, h = resolution
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    def get_declared_fps(self) -> Optional[float]:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else None

    async def recv(self) -> VideoFrame:  # type: ignore[override]
        # Blocking read from webcam
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Failed to read from webcam")
        vf = VideoFrame.from_ndarray(frame, format="bgr24")
        vf.pts, vf.time_base = await self.next_timestamp()
        return vf

    def release(self) -> None:
        try:
            self._cap.release()
        except Exception:
            pass


@dataclass
class _VideoStream:
    _frames: "Queue[Optional[np.ndarray]]"

    def stream(self) -> Iterator[np.ndarray]:
        while True:
            frame = self._frames.get()
            if frame is None:
                break
            yield frame


class _DataChannel:
    def __init__(self) -> None:
        self._handlers: dict[str, List[Callable[[Any], None]]] = {"message": []}

    def bind(self, channel: RTCDataChannel) -> None:
        @channel.on("message")
        def _on_message(message: Any) -> None:  # noqa: ANN401
            for cb in list(self._handlers.get("message", [])):
                try:
                    cb(message if isinstance(message, dict) else message)
                except Exception:
                    # best-effort; do not crash
                    pass

    def on(self, event: str) -> Callable[[Callable[[Any], None]], Callable[[Any], None]]:
        def decorator(fn: Callable[[Any], None]) -> Callable[[Any], None]:
            self._handlers.setdefault(event, []).append(fn)
            return fn

        return decorator


class WebRTCSession(AbstractContextManager["WebRTCSession"]):
    """Minimal WebRTC session supporting webcam streaming and results display."""

    def __init__(
        self,
        api_url: str,
        api_key: Optional[str],
        *,
        workspace_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        workflow_specification: Optional[dict] = None,
        image_input_name: str = "image",
        resolution: Optional[tuple[int, int]] = None,
        webrtc_realtime_processing: bool = True,
        webrtc_turn_config: Optional[dict] = None,
        stream_output: Optional[List[Optional[str]]] = None,
        data_output: Optional[List[Optional[str]]] = None,
        declared_fps: Optional[float] = None,
        workflows_parameters: Optional[dict] = None,
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._workspace_name = workspace_name
        self._workflow_id = workflow_id
        self._workflow_specification = workflow_specification
        self._image_input_name = image_input_name
        self._resolution = resolution
        self._webrtc_realtime_processing = webrtc_realtime_processing
        self._webrtc_turn_config = webrtc_turn_config
        self._stream_output = stream_output or []
        self._data_output = data_output or []
        self._declared_fps = declared_fps
        self._workflows_parameters = workflows_parameters or {}

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._pc: Optional[RTCPeerConnection] = None
        self._video_queue_sync: "Queue[Optional[np.ndarray]]" = Queue(maxsize=8)
        self.video = _VideoStream(self._video_queue_sync)
        self.data = _DataChannel()
        self._track: Optional[_WebcamVideoTrack] = None

    def __enter__(self) -> "WebRTCSession":
        # Start event loop in background thread
        self._loop = asyncio.new_event_loop()

        def _run(loop: asyncio.AbstractEventLoop) -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop_thread = threading.Thread(target=_run, args=(self._loop,), daemon=True)
        self._loop_thread.start()
        # Build peer connection and initialize
        fut = asyncio.run_coroutine_threadsafe(self._init(), self._loop)
        fut.result()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        try:
            if self._loop and self._pc:
                asyncio.run_coroutine_threadsafe(self._pc.close(), self._loop).result()
        finally:
            try:
                if self._track:
                    self._track.release()
            finally:
                if self._loop:
                    self._loop.call_soon_threadsafe(self._loop.stop)
                if self._loop_thread:
                    self._loop_thread.join(timeout=2)

    def wait_for_disconnect(self, timeout: Optional[float] = None) -> None:
        # Simple wait by draining the loop until the queue receives None (not used here)
        try:
            while True:
                frame = self.video._frames.get(timeout=timeout)
                if frame is None:
                    break
        except Exception:
            pass

    async def _init(self) -> None:
        # Create local track
        self._track = _WebcamVideoTrack(resolution=self._resolution)
        if self._declared_fps is None:
            self._declared_fps = self._track.get_declared_fps()

        # Peer connection
        configuration = None
        if self._webrtc_turn_config:
            ice = RTCIceServer(
                urls=[self._webrtc_turn_config.get("urls")],
                username=self._webrtc_turn_config.get("username"),
                credential=self._webrtc_turn_config.get("credential"),
            )
            configuration = RTCConfiguration(iceServers=[ice])

        pc = RTCPeerConnection(configuration=configuration)
        relay = MediaRelay()

        @pc.on("track")
        def _on_track(track):  # noqa: ANN001
            subscribed = relay.subscribe(track)

            async def _reader():
                while True:
                    try:
                        f: VideoFrame = await subscribed.recv()
                    except Exception:
                        # connection closed or track ended
                        try:
                            self._video_queue_sync.put_nowait(None)
                        except Exception:
                            pass
                        break
                    img = f.to_ndarray(format="bgr24")
                    # backpressure: drop oldest
                    if self._video_queue_sync.full():
                        try:
                            _ = self._video_queue_sync.get_nowait()
                        except Exception:
                            pass
                    try:
                        self._video_queue_sync.put_nowait(img)
                    except Exception:
                        pass

            asyncio.ensure_future(_reader())

        ch = pc.createDataChannel("inference")
        self.data.bind(ch)

        pc.addTrack(self._track)
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        # Wait for ICE gathering to complete
        while pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        # Call server to initialize worker
        wf_conf: dict[str, Any] = {
            "type": "WorkflowConfiguration",
            "image_input_name": self._image_input_name,
            "workflows_parameters": self._workflows_parameters,
        }
        if self._workflow_specification is not None:
            wf_conf["workflow_specification"] = self._workflow_specification
        else:
            # workspace_name + workflow_id path
            wf_conf["workflow_id"] = self._workflow_id
            wf_conf["workspace_name"] = self._workspace_name

        payload = {
            "api_key": self._api_key,
            "workflow_configuration": wf_conf,
            "webrtc_offer": {
                "type": pc.localDescription.type,
                "sdp": pc.localDescription.sdp,
            },
            "webrtc_turn_config": self._webrtc_turn_config,
            "webrtc_realtime_processing": self._webrtc_realtime_processing,
            "stream_output": self._stream_output,
            "data_output": self._data_output,
            "declared_fps": self._declared_fps,
            "rtsp_url": None,
        }

        url = f"{self._api_url}/initialise_webrtc_worker"
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        ans = resp.json()
        answer = RTCSessionDescription(sdp=ans["sdp"], type=ans["type"])  # type: ignore[index]
        await pc.setRemoteDescription(answer)

        self._pc = pc
