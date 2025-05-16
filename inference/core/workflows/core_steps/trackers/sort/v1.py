from typing import Literal, Union

from inference.core.workflows.execution_engine.entities.base import Selector
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
)
from .trackers.base import BaseTrackerBlock, BaseTrackerBlockManifest

__all__ = ["SortTrackerBlockV1"]

class SortTrackerBlockManifest(BaseTrackerBlockManifest):
    type: Literal["roboflow_core/sort_tracker@v1"]

    # SORT defaults
    minimum_consecutive_frames: Union[int, Selector(kind=[INTEGER_KIND])] = 3
    minimum_iou_threshold: Union[float, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = 0.3


class SortTrackerBlockV1(BaseTrackerBlock):
    """SORT tracker workflow block."""

    @classmethod
    def get_manifest(cls):  # noqa: D401
        return SortTrackerBlockManifest

    # ------------------------------------------------------------------

    def _instantiate_tracker(self, *, video_id: str, frame_rate: float, track_activation_threshold: float,
                             lost_track_buffer: int, minimum_consecutive_frames: int, minimum_iou_threshold: float,
                             **kwargs):
        from trackers.core.sort.tracker import SORTTracker

        return SORTTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate or 30.0,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold=minimum_iou_threshold,
        )