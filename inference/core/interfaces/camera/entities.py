from dataclasses import dataclass
from datetime import datetime
from enum import Enum

FrameTimestamp = datetime
FrameID = int


class UpdateSeverity(Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class StatusUpdate:
    timestamp: datetime
    severity: UpdateSeverity
    event_type: str
    payload: dict
