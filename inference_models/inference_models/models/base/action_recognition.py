import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


@dataclass(frozen=True)
class ActionRecognitionPrediction:
    """One classified frame segment; ranges may overlap."""

    start_frame_idx: int
    end_frame_idx: int
    class_name: str


SLIDING_WINDOW_MODE = "sliding_window"
WHOLE_VIDEO_MODE = "whole_video"
_MICROSECONDS = 1_000_000


@dataclass(frozen=True)
class VideoSampling:
    """The temporal contract a model is trained (or validated) for.

    Consumers window and sample video to match: window length in seconds,
    frames sampled per second, the fewest frames worth classifying, and
    the longest frame side the model was trained on.

    ``mode`` says how training cut a video. Under ``sliding_window`` a video
    becomes tiled windows of ``window_seconds``. Under ``whole_video`` it
    becomes one sample.

    ``max_frame_side`` and ``max_frames`` are trained values, so both stay
    ``None`` for a model that never trained on them, and the frames then reach
    the model at their own size and rate. Reading a model below what it
    expects costs the detail the answer is made of.

    ``max_frames`` is the budget one sample holds, which only a trained model
    has. A fine-tune records it, and a clip longer than the budget is sampled
    below ``sample_fps`` so the frames still span it. It stays ``None`` for a
    model that never trained on a budget, and then the clip is sampled at
    ``sample_fps`` however long it runs. The processor pools frames to its own
    token budget, so an unbounded count costs decode time and host memory
    rather than accelerator memory.
    """

    window_seconds: float = 16.0
    sample_fps: float = 4.0
    min_frames: int = 4
    max_frame_side: Optional[int] = None
    mode: str = SLIDING_WINDOW_MODE
    max_frames: Optional[int] = None


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
    """The source frames one classification reads, in order.

    ``sample_fps`` is the rate those frames stand for, which is what turns a
    frame index back into a timestamp. Under ``whole_video`` the step
    stretches, so this drops below the model's nominal rate.
    """

    frame_indices: Tuple[int, ...]
    sample_fps: float


def plan_windows(
    frame_count: int,
    source_fps: float,
    sampling: VideoSampling,
) -> List[WindowSpec]:
    """Cut a clip into the windows a model trained for ``sampling`` expects.

    Windows are measured in whole microseconds, not in frames, because that is
    what training does. Under ``sliding_window`` the clip tiles from the start
    and the trailing remainder is dropped, which is how training validates. A
    clip shorter than one window becomes a single sample spanning it, which is
    also what training does. Under ``whole_video`` the clip is one sample.
    """
    if frame_count <= 0 or source_fps <= 0 or sampling.sample_fps <= 0:
        return []
    duration_us = int(round(frame_count / source_fps * _MICROSECONDS))
    window_us = int(round(sampling.window_seconds * _MICROSECONDS))
    if sampling.mode == WHOLE_VIDEO_MODE or window_us <= 0 or duration_us <= window_us:
        return [_plan_interval(0, duration_us, frame_count, source_fps, sampling)]
    return [
        _plan_interval(
            index * window_us,
            (index + 1) * window_us,
            frame_count,
            source_fps,
            sampling,
        )
        for index in range(duration_us // window_us)
    ]


def _plan_interval(
    start_us: int,
    end_us: int,
    frame_count: int,
    source_fps: float,
    sampling: VideoSampling,
) -> WindowSpec:
    """The frames one sample reads, and the rate they stand for.

    Two grids exist, and a model reads the one it was trained on.

    A model that recorded ``max_frames`` was trained by the platform. That
    picks the count from the recorded rate alone, divides the interval into
    whole-microsecond timestamps, and takes the first frame at or after each
    one. A source too slow to supply that many distinct frames yields repeats,
    which is what training fed. The rate is ``count / duration``, which is
    what the trainer stamped the frames with.

    A model that recorded nothing has only its pretraining to match, and that
    used a plain rate. Its frames sit ``1 / sample_fps`` apart, capped at what
    the source can actually supply, and the rate is nominal. Reporting
    anything else moves every frame timestamp off the values the model saw,
    and the answer degrades to a summary.
    """
    span_us = end_us - start_us
    duration_seconds = span_us / _MICROSECONDS
    last_frame = frame_count - 1
    floor = max(1, sampling.min_frames)
    if sampling.max_frames is not None:
        count = max(
            floor,
            min(
                sampling.max_frames, int(round(duration_seconds * sampling.sample_fps))
            ),
        )
        step_us = span_us / count
        indices = tuple(
            min(last_frame, _frame_at_or_after(start_us + index * step_us, source_fps))
            for index in range(count)
        )
        return WindowSpec(frame_indices=indices, sample_fps=count / duration_seconds)
    effective_fps = min(sampling.sample_fps, source_fps)
    count = max(floor, int(round(duration_seconds * effective_fps)))
    step_us = _MICROSECONDS / effective_fps
    indices = tuple(
        min(
            last_frame,
            int(int(start_us + index * step_us) * source_fps / _MICROSECONDS),
        )
        for index in range(count)
    )
    return WindowSpec(frame_indices=indices, sample_fps=effective_fps)


def _frame_at_or_after(timestamp_us: float, source_fps: float) -> int:
    """The first frame no earlier than a timestamp, which is how training reads.

    The timestamp truncates to whole microseconds first, and the frame comes
    from one multiplication rather than a chain. Both keep binary floating
    point from drifting a frame off the trainer's grid.
    """
    return math.ceil(int(timestamp_us) * source_fps / _MICROSECONDS)


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
    start_frame_idx = min(
        segment.start_frame_idx, *(entry.start_frame_idx for entry in matching)
    )
    end_frame_idx = max(
        segment.end_frame_idx, *(entry.end_frame_idx for entry in matching)
    )
    timeline[:] = [entry for entry in timeline if entry not in matching]
    # Replaced rather than mutated: the result type this module declares is
    # frozen, so widening a range in place raises on its own predictions.
    timeline.append(
        _widened(
            segment=segment,
            start_frame_idx=start_frame_idx,
            end_frame_idx=end_frame_idx,
        )
    )


def _widened(segment, start_frame_idx: int, end_frame_idx: int):
    """A copy of ``segment`` covering the wider range.

    Timeline entries are frozen dataclasses on the model side and pydantic
    models on the response side, so the copy uses whichever protocol the entry
    offers rather than assuming one.
    """
    model_copy = getattr(segment, "model_copy", None)
    if callable(model_copy):
        return model_copy(
            update={
                "start_frame_idx": start_frame_idx,
                "end_frame_idx": end_frame_idx,
            }
        )
    return replace(
        segment, start_frame_idx=start_frame_idx, end_frame_idx=end_frame_idx
    )
