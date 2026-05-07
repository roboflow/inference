"""VideoFrameProducer that subscribes to a ROS image topic via rosbridge.

Lets ``InferencePipeline`` consume a ``sensor_msgs/Image`` or
``sensor_msgs/CompressedImage`` topic by passing either a callable factory or
a ``rosbridge://...`` URL as the ``video_reference``.

``roslibpy`` is imported lazily so the module is safe to import without the
``ros`` extra installed.
"""

from __future__ import annotations

import threading
from queue import Empty, Queue
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from inference.core import logger
from inference.core.interfaces.camera.entities import (
    SourceProperties,
    VideoFrameProducer,
)
from inference.core.workflows.core_steps.common.rosbridge.connection import (
    RosHandle,
    get_registry,
)
from inference.core.workflows.core_steps.common.rosbridge.encoding import (
    decode_image_message,
)

ROSBRIDGE_URL_SCHEME = "rosbridge"
DEFAULT_ROSBRIDGE_PORT = 9090
DEFAULT_MESSAGE_TYPE = "sensor_msgs/CompressedImage"
_FIRST_FRAME_TIMEOUT_S = 30.0


def is_rosbridge_reference(reference: Any) -> bool:
    if not isinstance(reference, str):
        return False
    return reference.lower().startswith(f"{ROSBRIDGE_URL_SCHEME}://")


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


class RosbridgeImageFrameProducer(VideoFrameProducer):
    """``cv2.VideoCapture``-shaped wrapper around a rosbridge image topic."""

    def __init__(
        self,
        host: str,
        topic: str,
        port: int = DEFAULT_ROSBRIDGE_PORT,
        ssl: bool = False,
        message_type: str = DEFAULT_MESSAGE_TYPE,
        compression: str = "none",
        queue_size: int = 1,
        throttle_rate_ms: int = 0,
        first_frame_timeout: float = _FIRST_FRAME_TIMEOUT_S,
    ) -> None:
        self._host = host
        self._port = port
        self._ssl = ssl
        self._topic = topic if topic.startswith("/") else f"/{topic}"
        self._message_type = message_type
        self._compression = compression
        self._queue_size = max(1, int(queue_size))
        self._throttle_rate_ms = max(0, int(throttle_rate_ms))
        self._first_frame_timeout = first_frame_timeout

        self._frames: "Queue[Dict[str, Any]]" = Queue(maxsize=self._queue_size)
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_msg_meta: Optional[Tuple[int, int]] = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._opened = False

        self._handle: Optional[RosHandle] = None
        self._topic_handle: Optional[Any] = None

        self._open()

    @classmethod
    def from_url(cls, url: str) -> "RosbridgeImageFrameProducer":
        parsed = urlparse(url)
        if parsed.scheme.lower() != ROSBRIDGE_URL_SCHEME:
            raise ValueError(f"not a rosbridge URL: {url!r}")
        host = parsed.hostname
        if not host:
            raise ValueError(f"rosbridge URL missing host: {url!r}")
        port = parsed.port or DEFAULT_ROSBRIDGE_PORT
        topic = unquote(parsed.path or "")
        if not topic or topic == "/":
            raise ValueError(f"rosbridge URL missing topic path: {url!r}")
        params = {k: v[-1] for k, v in parse_qs(parsed.query).items()}
        return cls(
            host=host,
            port=port,
            ssl=_parse_bool(params.get("ssl", "false")),
            topic=topic,
            message_type=params.get("type", DEFAULT_MESSAGE_TYPE),
            compression=params.get("compression", "none"),
            queue_size=int(params.get("queue_size", "1")),
            throttle_rate_ms=int(params.get("throttle_rate_ms", "0")),
        )

    def _open(self) -> None:
        try:
            import roslibpy  # type: ignore
        except ImportError as e:
            raise ImportError(
                "roslibpy is required for rosbridge:// sources. "
                "Install with: pip install 'inference[ros]'"
            ) from e

        self._handle = get_registry().acquire(self._host, self._port, self._ssl)
        ros = self._handle.ros
        kwargs = {
            "name": self._topic,
            "message_type": self._message_type,
            "queue_size": self._queue_size,
            "throttle_rate": self._throttle_rate_ms,
        }
        if self._compression and self._compression.lower() != "none":
            kwargs["compression"] = self._compression
        self._topic_handle = roslibpy.Topic(ros, **kwargs)
        self._topic_handle.subscribe(self._on_message)
        self._opened = True
        logger.info(
            "rosbridge image source subscribed: %s @ %s:%s (type=%s, compression=%s)",
            self._topic,
            self._host,
            self._port,
            self._message_type,
            self._compression,
        )

    def _on_message(self, msg: Dict[str, Any]) -> None:
        if self._closed.is_set():
            return
        try:
            frame = decode_image_message(msg, self._message_type)
        except Exception:
            logger.exception("failed to decode rosbridge image message")
            return
        with self._lock:
            if self._frames.full():
                try:
                    self._frames.get_nowait()
                except Empty:
                    pass
            self._frames.put_nowait(frame)

    def grab(self) -> bool:
        if self._closed.is_set():
            return False
        timeout = self._first_frame_timeout if self._latest_frame is None else None
        try:
            frame = self._frames.get(timeout=timeout)
        except Empty:
            return False
        with self._lock:
            self._latest_frame = frame
            self._latest_msg_meta = (frame.shape[1], frame.shape[0])
        return True

    def retrieve(self) -> Tuple[bool, np.ndarray]:
        with self._lock:
            frame = self._latest_frame
        if frame is None:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        return True, frame

    def release(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            if self._topic_handle is not None:
                self._topic_handle.unsubscribe()
        except Exception:
            logger.exception("error unsubscribing rosbridge topic")
        finally:
            self._topic_handle = None
        if self._handle is not None:
            self._handle.release()
            self._handle = None
        self._opened = False

    def isOpened(self) -> bool:
        return self._opened and not self._closed.is_set()

    def discover_source_properties(self) -> SourceProperties:
        if self._latest_msg_meta is None:
            if not self.grab():
                raise RuntimeError(
                    "no frames received from rosbridge topic before timeout"
                )
        width, height = self._latest_msg_meta or (0, 0)
        return SourceProperties(
            width=int(width),
            height=int(height),
            total_frames=-1,
            is_file=False,
            fps=0.0,
            is_reconnectable=True,
        )

    def initialize_source_properties(self, properties: Dict[str, float]) -> None:
        # No-op: rosbridge image topics don't expose cv2.CAP_PROP_* knobs.
        return None
