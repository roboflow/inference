"""Video this process captures and publishes into a Cloud Streaming temporary source."""

import queue
import threading
import time
from abc import ABC, abstractmethod
from fractions import Fraction
from typing import Iterator, Optional

import cv2
import numpy as np

from inference_sdk.cloud_streaming.api import CloudStreamingError
from inference_sdk.cloud_streaming.tunnel import verified_rtsps

# The processor starts decoding at a keyframe, so keep them about a second apart.
KEYFRAME_INTERVAL = 30
MILLISECONDS = Fraction(1, 1000)
# A live input that stops delivering for this long is treated as gone.
INPUT_TIMEOUT_US = 10_000_000


def require_av():
    try:
        import av
    except ImportError as error:
        raise CloudStreamingError(
            "Publishing or decoding video needs PyAV: pip install 'inference-sdk[streaming]'"
        ) from error
    return av


class LocalSource(ABC):
    """Video this process captures; the stream runs on a temporary source that it publishes to."""

    name = "Python SDK"

    @abstractmethod
    def frames(self, stopped: threading.Event) -> Iterator[np.ndarray]:
        """BGR frames, in real time, until the video ends or `stopped` is set."""


class WebcamSource(LocalSource):
    """A local camera, opened with OpenCV."""

    def __init__(
        self, device: int = 0, width: Optional[int] = None, height: Optional[int] = None
    ):
        self.device = device
        self.width = width
        self.height = height
        self.name = f"Webcam {device}"

    def frames(self, stopped: threading.Event) -> Iterator[np.ndarray]:
        capture = cv2.VideoCapture(self.device)
        try:
            if self.width:
                capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height:
                capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not capture.isOpened():
                raise CloudStreamingError(f"cannot open camera {self.device}")
            while not stopped.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise CloudStreamingError(
                        f"camera {self.device} stopped delivering frames"
                    )
                yield frame
        finally:
            capture.release()


class VideoFileSource(LocalSource):
    """A video file, played at its own frame rate as a live source."""

    def __init__(self, path: str, loop: bool = False):
        self.path = path
        self.loop = loop
        self.name = "Video file"

    def frames(self, stopped: threading.Event) -> Iterator[np.ndarray]:
        av = require_av()
        while not stopped.is_set():
            with av.open(self.path) as container:
                started = time.monotonic()
                first = None
                for frame in container.decode(video=0):
                    if frame.time is not None:
                        first = frame.time if first is None else first
                        stopped.wait(
                            max(0.0, frame.time - first - (time.monotonic() - started))
                        )
                    if stopped.is_set():
                        return
                    yield frame.to_ndarray(format="bgr24")
            if not self.loop:
                return


class RTSPSource(LocalSource):
    """A live stream this machine can reach (RTSP, RTMP, HTTP MJPEG), such as a camera on the LAN."""

    def __init__(self, url: str):
        self.url = url
        self.name = "RTSP camera"

    def frames(self, stopped: threading.Event) -> Iterator[np.ndarray]:
        av = require_av()
        options = {"timeout": str(INPUT_TIMEOUT_US)}
        if self.url.startswith(("rtsp://", "rtsps://")):
            options["rtsp_transport"] = "tcp"
        with av.open(self.url, options=options) as container:
            for frame in container.decode(video=0):
                if stopped.is_set():
                    return
                yield frame.to_ndarray(format="bgr24")


class ManualSource(LocalSource):
    """Frames you push with `send()`, for example after your own preprocessing."""

    def __init__(self, max_queued: int = 10):
        self._frames: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queued)
        self.name = "Python SDK frames"

    def send(self, frame: np.ndarray) -> None:
        """Queue a BGR frame; the oldest queued frame is dropped when the queue is full."""
        while True:
            try:
                self._frames.put_nowait(frame)
                return
            except queue.Full:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    pass

    def frames(self, stopped: threading.Event) -> Iterator[np.ndarray]:
        while not stopped.is_set():
            try:
                yield self._frames.get(timeout=0.5)
            except queue.Empty:
                continue


class Publisher:
    """Encodes a local source as H.264 and publishes it to a temporary source over RTSPS."""

    def __init__(self, source: LocalSource, rtsps_url: str):
        self._source = source
        self._url = rtsps_url
        self._stopped = threading.Event()
        self._publishing = False
        self._settled = threading.Event()
        self.done = threading.Event()
        self.error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="cloud-streaming-publisher", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def wait_until_publishing(self, timeout: float) -> None:
        """Return once the relay accepted the video; raise when publishing failed or timed out."""
        self._settled.wait(timeout)
        if not self._publishing:
            raise CloudStreamingError(
                f"could not publish the video: {self.error or 'timed out'}"
            )

    def stop(self) -> None:
        self._stopped.set()
        self._thread.join()

    def _run(self) -> None:
        tunnel = None
        try:
            with verified_rtsps(self._url) as tunnel:
                self._publish(tunnel.url)
        except Exception as error:
            # A refused certificate surfaces from FFmpeg as a plain I/O error; report its cause.
            # FFmpeg's own errors name the url, credentials included, so keep only the reason.
            self.error = CloudStreamingError(
                (tunnel and tunnel.error)
                or getattr(error, "strerror", None)
                or str(error)
            )
        finally:
            self._settled.set()
            self.done.set()

    def _publish(self, url: str) -> None:
        av = require_av()
        container = av.open(
            url, mode="w", format="rtsp", options={"rtsp_transport": "tcp"}
        )
        try:
            stream = None
            started = 0.0
            last_pts = -1
            for image in self._source.frames(self._stopped):
                if stream is None:
                    stream = container.add_stream("libx264", rate=KEYFRAME_INTERVAL)
                    # 4:2:0 chroma needs even dimensions.
                    stream.width = image.shape[1] & ~1
                    stream.height = image.shape[0] & ~1
                    stream.pix_fmt = "yuv420p"
                    # Frames are stamped with wall-clock milliseconds, so the encoder counts in
                    # them too; at its 1/rate default, nearby frames would share a timestamp.
                    stream.codec_context.time_base = MILLISECONDS
                    stream.options = {
                        "preset": "veryfast",
                        "tune": "zerolatency",
                        "g": str(KEYFRAME_INTERVAL),
                    }
                    started = time.monotonic()
                frame = av.VideoFrame.from_ndarray(image, format="bgr24").reformat(
                    width=stream.width, height=stream.height, format="yuv420p"
                )
                last_pts = max(last_pts + 1, int((time.monotonic() - started) * 1000))
                frame.pts = last_pts
                frame.time_base = MILLISECONDS
                for packet in stream.encode(frame):
                    container.mux(packet)
                self._publishing = True
                self._settled.set()
            if stream is not None:
                for packet in stream.encode():
                    container.mux(packet)
        finally:
            container.close()
