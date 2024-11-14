import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
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
        fps (Optional[float]): declared FPS of source (if possible to be acquired)
        measured_fps (Optional[float]): measured FPS of live stream
        comes_from_video_file (Optional[bool]): flag to determine if frame comes from video file
    """

    image: np.ndarray
    frame_id: FrameID
    frame_timestamp: FrameTimestamp
    # TODO: in next major version of inference replace `fps` with `declared_fps`
    fps: Optional[float] = None
    measured_fps: Optional[float] = None
    source_id: Optional[int] = None
    comes_from_video_file: Optional[bool] = None


@dataclass(frozen=True)
class SourceProperties:
    width: int
    height: int
    total_frames: int
    is_file: bool
    fps: float
    is_reconnectable: Optional[bool] = None


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


VideoSourceIdentifier = Union[str, int, Callable[[], VideoFrameProducer]]
