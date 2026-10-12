"""Workflows 2.0 video sources (optional; needs ``streamvision[workflows]``).

Usage::

    from roboflow_workflows.execution_engine.v2 import Catalogue, compile_workflow
    from streamvision.workflows_v2 import VideoStatistics

    catalogue = Catalogue.from_modules([
        "roboflow_workflows.execution_engine.v2.blocks",
        "streamvision.workflows_v2",
    ])
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

from roboflow_workflows.execution_engine.v2.catalogue import Catalogue

from streamvision.workflows_v2.reader import VideoSourceFailure, VideoStopError
from streamvision.workflows_v2.sources import (
    VideoStream,
    VideoStreamSet,
    frame_to_image,
)
from streamvision.workflows_v2.statistics import MemberStatistics, VideoStatistics

WORKFLOWS_V2_SOURCES = (VideoStream, VideoStreamSet)


def create_catalogue() -> Catalogue:
    """Expose the opt-in video sources through V2 module discovery.

    Returns:
        A catalogue containing ``video/stream`` and ``video/stream_set``.
    """
    catalogue = Catalogue(sources=WORKFLOWS_V2_SOURCES)

    return catalogue


WORKFLOWS_V2_CATALOGUE = create_catalogue

__all__ = [
    "MemberStatistics",
    "VideoSourceFailure",
    "VideoStatistics",
    "VideoStopError",
    "VideoStream",
    "VideoStreamSet",
    "WORKFLOWS_V2_CATALOGUE",
    "WORKFLOWS_V2_SOURCES",
    "create_catalogue",
    "frame_to_image",
]
