import math

from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.execution_engine.entities.base import VideoMetadata

logger = get_logger(__name__)

# Matches the trackers in ``core_steps/trackers`` and Inference's VideoSource.
DEFAULT_BYTE_TRACK_FPS = 30


def resolve_byte_track_fps(metadata: VideoMetadata, block_name: str) -> float:
    """Return the frame rate to initialize a video's ByteTrack with.

    supervision removes a lost track after
    ``int(frame_rate / 30 * lost_track_buffer)`` frames. A rate of 0 makes that
    0, so an object missed for three frames returns with a new tracker ID.
    Call this only when creating a video's tracker, so the warning appears
    once per ``video_identifier``.
    """
    fps = metadata.fps
    if (
        isinstance(fps, (int, float))
        and not isinstance(fps, bool)
        and math.isfinite(fps)
        and fps > 0
    ):
        return fps
    logger.warning(
        f"fps not available in VideoMetadata for {block_name} "
        f"(video_identifier={metadata.video_identifier!r}, fps={fps!r}), "
        f"defaulting to {DEFAULT_BYTE_TRACK_FPS} fps for ByteTrack initialisation"
    )
    return DEFAULT_BYTE_TRACK_FPS
