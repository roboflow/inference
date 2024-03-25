import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

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
    """

    image: np.ndarray
    frame_id: FrameID
    frame_timestamp: FrameTimestamp
    source_id: Optional[int] = None
