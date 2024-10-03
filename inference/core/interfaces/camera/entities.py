import logging
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from multiprocessing.synchronize import Lock as LockType
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np

FrameTimestamp = datetime
FrameID = int


class UpdateSeverity(Enum):
    """Enumeration for defining different levels of update severity.

    Attributes:
        DEBUG (int): A debugging severity level.
        INFO (int): An informational severity level.
        WARNING (int): A warning severity level.
        ERROR (int): An error severity level.
    """

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR


@dataclass(frozen=True)
class StatusUpdate:
    """Represents a status update event in the system.

    Attributes:
        timestamp (datetime): The timestamp when the status update was created.
        severity (UpdateSeverity): The severity level of the update.
        event_type (str): A string representing the type of the event.
        payload (dict): A dictionary containing data relevant to the update.
        context (str): A string providing additional context about the update.
    """

    timestamp: datetime
    severity: UpdateSeverity
    event_type: str
    payload: dict
    context: str


@dataclass(frozen=True)
class VideoFrame:
    """Represents a single frame of video data.

    Attributes:
        image (np.ndarray): The image data of the frame as a NumPy array.
        frame_id (FrameID): A unique identifier for the frame.
        frame_timestamp (FrameTimestamp): The timestamp when the frame was captured.
        source_id (int): The index of the video_reference element which was passed to InferencePipeline for this frame
            (useful when multiple streams are passed to InferencePipeline).
        fps (Optional[float]): FPS of source (if possible to be acquired)
        comes_from_video_file (Optional[bool]): flag to determine if frame comes from video file
    """

    image: np.ndarray
    frame_id: FrameID
    frame_timestamp: FrameTimestamp
    fps: Optional[float] = None
    source_id: Optional[int] = None
    comes_from_video_file: Optional[bool] = None


@dataclass(frozen=True)
class SourceProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float


class VideoFrameProducer:
    def grab(self) -> bool:
        raise NotImplementedError

    def retrieve(self) -> Tuple[bool, np.ndarray]:
        raise NotImplementedError

    def release(self):
        raise NotImplementedError

    def isOpened(self) -> bool:
        raise NotImplementedError

    def discover_source_properties(self) -> SourceProperties:
        raise NotImplementedError

    def initialize_source_properties(self, properties: Dict[str, float]):
        pass


class WebRTCVideoFrameProducer(VideoFrameProducer):
    def __init__(self, to_inference_queue: deque, to_inference_lock: LockType):
        self.to_inference_queue: deque = to_inference_queue
        self.to_inference_lock: LockType = to_inference_lock
        self._w: Optional[int] = None
        self._h: Optional[int] = None
        self._fps_buff = []
        self._is_opened = True

    def grab(self) -> bool:
        return True

    def retrieve(self) -> Tuple[bool, np.ndarray]:
        while not self.to_inference_queue:
            time.sleep(0.1)
        self._is_opened = True
        with self.to_inference_lock:
            img = self.to_inference_queue.pop()
        return True, img

    def release(self):
        pass

    def isOpened(self) -> bool:
        return self._is_opened

    def discover_source_properties(self) -> SourceProperties:
        max_ts = max(self._fps_buff, key=lambda x: x["ts"]) if self._fps_buff else 0
        min_ts = min(self._fps_buff, key=lambda x: x["ts"]) if self._fps_buff else 0
        if max_ts == min_ts:
            max_ts += 0.1
        fps = len(self._fps_buff) / (max_ts - min_ts)
        return SourceProperties(
            width=self._w,
            height=self._h,
            total_frames=-1,
            is_file=False,
            fps=fps,
        )

    def initialize_source_properties(self, properties: Dict[str, float]):
        pass


VideoSourceIdentifier = Union[str, int, Callable[[], VideoFrameProducer]]
