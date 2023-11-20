from typing import List, Optional, Tuple

import numpy as np

from inference.core.interfaces.camera.entities import FrameTimestamp, FrameID
from inference.core.interfaces.camera.video_source import VideoSource


class StreamMultiplexer:

    def __init__(self, sources: List[VideoSource]):
        self._sources = sources

    def get_frames(self) -> List[Optional[Tuple[FrameTimestamp, FrameID, np.ndarray]]]:
        return [
            source.read_frame() if source.frame_ready() else None
            for source in self._sources
        ]
