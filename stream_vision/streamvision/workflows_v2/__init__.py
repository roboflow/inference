"""Workflows 2.0 video sources (optional; needs ``streamvision[workflows]``).

Usage::

    from streamvision.workflows_v2 import VideoStatistics, WORKFLOWS_V2_SOURCES

    catalogue = Catalogue(blocks, sources=[*WORKFLOWS_V2_SOURCES, ...])
    definition["sources"] = [
        {"type": "video/stream_set@v1", "name": "cams",
         "references": ["rtsp://cam-0/stream", "rtsp://cam-1/stream"],
         "batch_size": 1},
    ]
    statistics = VideoStatistics()
    session = compile_workflow(definition, catalogue=catalogue).create_session(
        {"video_statistics": statistics}
    )
    run = session.start(handlers=handlers)
    run.wait()
    statistics.members()["cams/1"].end_reason

``streamvision`` never imports this package itself.
"""

from streamvision.workflows_v2.reader import VideoSourceFailure, VideoStopError
from streamvision.workflows_v2.sources import (
    VideoStream,
    VideoStreamSet,
    frame_to_image,
)
from streamvision.workflows_v2.statistics import MemberStatistics, VideoStatistics

WORKFLOWS_V2_SOURCES = (VideoStream, VideoStreamSet)

__all__ = [
    "MemberStatistics",
    "VideoSourceFailure",
    "VideoStatistics",
    "VideoStopError",
    "VideoStream",
    "VideoStreamSet",
    "WORKFLOWS_V2_SOURCES",
    "frame_to_image",
]
