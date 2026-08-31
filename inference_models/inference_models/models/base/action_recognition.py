from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class ActionRecognitionPrediction:
    """One classified frame segment; ranges may overlap."""

    start_frame_idx: int
    end_frame_idx: int
    class_name: str


@dataclass(frozen=True)
class VideoSampling:
    """The temporal contract a model is trained (or validated) for.

    Consumers window and sample video to match: window length in seconds,
    frames sampled per second, the fewest frames worth classifying, and
    the longest frame side the model was trained on.

    """

    window_seconds: float = 16.0
    sample_fps: float = 4.0
    min_frames: int = 4
    max_frame_side: int = 360


class ActionRecognitionModel(ABC):

    @property
    def video_sampling(self) -> VideoSampling:
        return VideoSampling()

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "ActionRecognitionModel":
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
    ) -> List[ActionRecognitionPrediction]:
        """Classify RGB frames and return segments in their index space.

        Frames are numpy HWC arrays or torch CHW tensors. ``class_names``
        restricts a model's own vocabulary to a subset, and supplies the
        vocabulary for an open-vocabulary model.
        """
        pass

    def __call__(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[ActionRecognitionPrediction]:
        return self.infer(
            frames=frames,
            class_names=class_names,
            fps=fps,
            **kwargs,
        )


@dataclass(frozen=True)
class WindowSpec:
    """The source frames one classification reads, in order."""

    frame_indices: Tuple[int, ...]


def plan_windows(
    frame_count: int,
    source_fps: float,
    sampling: VideoSampling,
) -> List[WindowSpec]:
    """Cut a clip into the windows a model trained for ``sampling`` expects.

    Windows tile from the start of the clip and the trailing remainder is
    dropped, which is how training validates. A clip shorter than one window
    stays whole, at a reduced frame count, and a clip too short to reach
    ``min_frames`` yields nothing.
    """
    if frame_count <= 0 or source_fps <= 0:
        return []
    effective_fps = min(sampling.sample_fps, source_fps)
    if effective_fps <= 0:
        return []
    source_frames_per_sample = source_fps / effective_fps
    window_span = max(1, round(sampling.window_seconds * source_fps))
    samples_per_window = max(1, round(sampling.window_seconds * effective_fps))

    def _sample(first: int, count: int) -> Tuple[int, ...]:
        return tuple(
            min(frame_count - 1, first + round(index * source_frames_per_sample))
            for index in range(count)
        )

    windows = []
    window_start = 0
    while window_start + window_span <= frame_count:
        windows.append(
            WindowSpec(frame_indices=_sample(window_start, samples_per_window))
        )
        window_start += window_span
    if windows:
        return windows
    whole_clip_samples = max(1, int(frame_count / source_frames_per_sample))
    if whole_clip_samples < max(1, sampling.min_frames):
        return []
    return [WindowSpec(frame_indices=_sample(0, whole_clip_samples))]


def merge_segment(timeline: list, segment, stride: float) -> None:
    """Union ``segment`` into ``timeline`` in place, keeping ranges monotone.

    Two ranges of one class merge when no sampled frame lies in the gap
    between them, so ``stride`` is the sampling stride rounded up. Ranges only
    ever grow; nothing here extends or retracts a range on its own. Entries
    need ``start_frame_idx``, ``end_frame_idx`` and ``class_name``.
    """
    matching = [
        existing
        for existing in timeline
        if existing.class_name == segment.class_name
        and existing.start_frame_idx <= segment.end_frame_idx + stride
        and segment.start_frame_idx <= existing.end_frame_idx + stride
    ]
    if not matching:
        timeline.append(segment)
        return
    segment.start_frame_idx = min(
        segment.start_frame_idx, *(entry.start_frame_idx for entry in matching)
    )
    segment.end_frame_idx = max(
        segment.end_frame_idx, *(entry.end_frame_idx for entry in matching)
    )
    timeline[:] = [entry for entry in timeline if entry not in matching]
    timeline.append(segment)
