"""Buffer strategies of a video source, importable without the decoder.

Request entities validate these values without needing cv2 or the video
source itself. `video_source` re-exports both enums, and each class keeps the
`__module__` it was historically defined in, so signatures, annotations and
pickles still name `video_source` exactly as before the extraction.
"""

from enum import Enum

_HISTORICAL_MODULE = f"{__name__.rsplit('.', 1)[0]}.video_source"


class BufferFillingStrategy(str, Enum):
    WAIT = "WAIT"
    DROP_OLDEST = "DROP_OLDEST"
    ADAPTIVE_DROP_OLDEST = "ADAPTIVE_DROP_OLDEST"
    DROP_LATEST = "DROP_LATEST"
    ADAPTIVE_DROP_LATEST = "ADAPTIVE_DROP_LATEST"


class BufferConsumptionStrategy(str, Enum):
    LAZY = "LAZY"
    EAGER = "EAGER"


BufferFillingStrategy.__module__ = _HISTORICAL_MODULE
BufferConsumptionStrategy.__module__ = _HISTORICAL_MODULE
