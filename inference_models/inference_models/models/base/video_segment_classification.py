from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import torch


@dataclass(frozen=True)
class VideoSegmentClassificationPrediction:
    """One classified frame segment; ranges may overlap."""

    start_frame_idx: int
    end_frame_idx: int
    class_name: str


WHOLE_VIDEO_SAMPLING_MODE = "whole_video"


@dataclass(frozen=True)
class VideoSampling:
    """The temporal contract a model is trained (or validated) for.

    Consumers window and sample video to match: window length in seconds,
    frames sampled per second, and the fewest frames worth classifying.

    ``whole_video`` mode models read a clip as one unit, with the frame
    budget spread over its full length, so consumers classify once at the
    end of the stream instead of on a window schedule.
    """

    window_seconds: float = 16.0
    sample_fps: float = 4.0
    min_frames: int = 4
    mode: str = "sliding_window"
    frame_budget: Optional[int] = None

    @property
    def classifies_whole_video(self) -> bool:
        return self.mode == WHOLE_VIDEO_SAMPLING_MODE

    @property
    def window_frames(self) -> int:
        """Frames per classification.

        Whole-video models give no meaning to the window length, so the
        budget the model declares wins over the length-derived count.
        """
        if self.frame_budget is not None and self.frame_budget > 0:
            return self.frame_budget
        return max(1, round(self.window_seconds * self.sample_fps))


class VideoSegmentClassificationModel(ABC):

    @property
    def video_sampling(self) -> VideoSampling:
        return VideoSampling()

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "VideoSegmentClassificationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> Optional[List[str]]:
        """Fixed-vocabulary models return their class list.

        Open-vocabulary models return None and take ``class_names`` per call.
        """
        pass

    @abstractmethod
    def infer(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[VideoSegmentClassificationPrediction]:
        """Classify RGB frames and return segments in their index space.

        Frames are numpy HWC arrays or torch CHW tensors. ``class_names``
        overrides a model-provided vocabulary and constrains open-vocabulary
        models.
        """
        pass

    def __call__(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[VideoSegmentClassificationPrediction]:
        return self.infer(
            frames=frames,
            class_names=class_names,
            fps=fps,
            **kwargs,
        )
