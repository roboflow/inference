from typing import List, Optional, Tuple

import numpy as np

from inference.core.interfaces.camera.entities import FrameTimestamp, FrameID, VideoFrame
from inference.core.interfaces.camera.exceptions import EndOfStreamError
from inference.core.interfaces.camera.video_source import VideoSource


class StreamMultiplexer:
    def __init__(self, sources: List[VideoSource]):
        self._sources = sources

    def get_frames(self) -> List[Optional[VideoFrame]]:
        frames = []
        for source in self._sources:
            try:
                if source.frame_ready():
                    frames.append(source.read_frame())
                else:
                    frames.append(None)
            except EndOfStreamError:
                frames.append(None)
        return frames
