"""Draw an Action Recognition timeline on video, live or synchronized."""

import math
import zlib
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Deque, Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict, Field
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS,
)
from roboflow_workflows.core_steps.visualizations.common.fonts import resolve_font_path
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    ActionRecognitionResult,
    ImageParentMetadata,
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_RESULT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
    STILL_IMAGE_INPUT_SOFT_RESTRICTION,
    BlockResult,
    DependentResource,
    WorkflowBlock,
    WorkflowBlockManifest,
    actual_restrictions_of,
)
from roboflow_workflows.utils.image_encoding import (
    decode_encoded_image_bytes,
    encode_image_to_jpeg_bytes,
)

logger = get_logger(__name__)

TYPE = "roboflow_core/action_recognition_visualization@v1"
MAX_STREAMS = 16
DEFAULT_SOURCE_FPS = 30.0
MAX_LANES = 5
JPEG_QUALITY = 85
# How many past model calls set the playback delay. One slow call then stops
# holding the delay up after a few normal ones.
LAG_PEAKS_KEPT = 4
MAX_CACHED_TEXTS = 512
LABEL_FONT_FAMILY = "geist"

STATUS_ANALYZING = "analyzing"
STATUS_LIVE = "live"
STATUS_SYNCHRONIZED = "synchronized"
STATUS_BEHIND = "behind"
STATUS_NOT_ANALYZED = "not_analyzed"

LONG_DESCRIPTION = """
Draw the results of an Action Recognition Model on its video. Connect the
original image and the model's `predictions` output.

An action recognition result always describes frames the stream has already
passed. The block offers two honest ways to show that.

`live` (default) shows the current frame with no delay. A strip at the bottom
covers the last `history_seconds`: one lane per class, a bar for each action
range, and a hatched zone for frames the model has not analyzed yet. A badge
names the most recent action on the strip and how long ago it ended, so a
quiet latest window does not hide it. With no action on the strip, the badge
describes the last analyzed frame. This mode keeps no frames and adds no
latency.

`synchronized` plays the video late, so each label sits on the frame it
describes. The block picks the delay: it follows how far the model's results
trail the stream, which is about one stride plus one model call. A badge shows
the current delay. Until the first result arrives, the first frame is held
with an "Analyzing" badge, then playback starts from that frame. `max_delay_seconds` caps the delay. When the
model falls further behind than the cap, or the frame budget is too small,
playback continues with the status `behind` and frames may carry no label.
This mode suits live streams. A video file ends before its delayed tail is
shown; read the model's `timeline` on the last frame instead.

Frames for `synchronized` are downscaled to `max_frame_side`, compressed and
kept in host memory, within `max_buffer_mb` across all streams of the step.
When the budget is too small the oldest frames are dropped and the delay
shrinks; the block does not raise. At most 16 streams are tracked; the least
recently used one is dropped. A frame-number restart or a settings change
resets a stream. A stream without a declared FPS is treated as 30 FPS, like
the model.

`status` is `analyzing` before the first result, then `live`, `synchronized`,
`not_analyzed` (for a frame no successful window covered) or `behind`. `actions` lists the classes at
`actions_frame_number`: the displayed frame in `synchronized`; in `live`, the
last frame of the most recent action on the strip, or the last analyzed frame
when the strip has none. `displayed_frame_number` identifies the frame in the
image. Configure the video output to this block's `image`.
"""


class BlockManifest(WorkflowBlockManifest):
    """Configure how Action Recognition results appear on video."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition Visualization",
            "version": "v1",
            "short_description": "Draw an action timeline on live or synchronized video.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "ui_manifest": {"section": "video", "icon": "far fa-timeline"},
        }
    )
    type: Literal["roboflow_core/action_recognition_visualization@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Original video frame of the stream the model analyzes.",
        examples=["$inputs.image"],
    )
    predictions: Selector(kind=[ACTION_RECOGNITION_RESULT_KIND]) = Field(
        description=(
            "Predictions output of the same stream's Action Recognition Model, "
            "carrying the timeline and the analyzed frame bounds of each "
            "successful window."
        ),
        examples=["$steps.actions.predictions"],
    )
    display_mode: Union[
        Literal["live", "synchronized"], Selector(kind=[STRING_KIND])
    ] = Field(
        default="live",
        description=(
            "live shows the current frame and a timeline of the recent past. "
            "synchronized plays the video late so labels sit on their frames."
        ),
        examples=["live", "synchronized", "$inputs.display_mode"],
    )
    history_seconds: float = Field(
        default=30.0,
        ge=1,
        le=600,
        allow_inf_nan=False,
        description="Video time the timeline strip covers, ending at the displayed frame.",
        examples=[30.0],
    )
    max_delay_seconds: float = Field(
        default=60.0,
        ge=0,
        le=300,
        allow_inf_nan=False,
        description=(
            "Upper limit of the synchronized playback delay. Beyond it the "
            "video continues without waiting for results."
        ),
        examples=[60.0],
    )
    max_frame_side: int = Field(
        default=1280,
        ge=32,
        le=4096,
        description=(
            "Longest image side kept for synchronized playback. Larger frames "
            "are downscaled; smaller ones are not upscaled."
        ),
        examples=[1280, 640],
    )
    max_buffer_mb: int = Field(
        default=256,
        ge=1,
        le=16384,
        description=(
            "Memory for compressed synchronized frames across all streams of "
            "this step. A smaller budget shortens the delay instead of failing."
        ),
        examples=[256, 512],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        """Describe the drawn image and what it shows.

        Returns:
            Image, displayed frame number, actions, their frame number and status.
        """
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
            OutputDefinition(name="displayed_frame_number", kind=[INTEGER_KIND]),
            OutputDefinition(name="actions", kind=[LIST_OF_VALUES_KIND]),
            OutputDefinition(name="actions_frame_number", kind=[INTEGER_KIND]),
            OutputDefinition(name="status", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        """Return the compatible engine range.

        Returns:
            Supported execution engine versions.
        """
        return ">=1.4.0,<2.0.0"

    @classmethod
    def get_restrictions(cls) -> List[RuntimeRestriction]:
        """Declare persistent video-state requirements.

        Returns:
            Restrictions for stateless HTTP and still-image inputs.
        """
        return [
            STATEFUL_VIDEO_HTTP_SOFT_RESTRICTION,
            STILL_IMAGE_INPUT_SOFT_RESTRICTION,
        ]

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare rendering and buffering work.

        Returns:
            Operations performed on each frame.
        """
        return [WorkOperation.VISUALIZATION, WorkOperation.TEMPORAL_BUFFERING]

    def discover_dependent_resources(self) -> List[DependentResource]:
        """Declare external resources.

        Returns:
            An empty list; this block does not run a model.
        """
        return []

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Describe stateful video restrictions.

        Args:
            ignore_environment_restrictions: Whether to ignore host conditions.

        Returns:
            Discovered temporal restrictions.
        """
        result = actual_restrictions_of(
            declared=list(STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS),
            node_id=f"$steps.{self.name}",
            ignore_environment_restrictions=ignore_environment_restrictions,
        )
        return result


@dataclass
class _BufferedFrame:
    metadata: VideoMetadata
    parent_metadata: ImageParentMetadata
    root_metadata: ImageParentMetadata
    jpeg: bytes


@dataclass
class _Stream:
    fps: float
    signature: tuple
    first_frame: int
    last_frame: int
    frames: Deque[_BufferedFrame] = field(default_factory=deque)
    buffered_bytes: int = 0
    last_mark: Optional[int] = None
    last_displayed_frame: Optional[int] = None
    covered_ranges: List[Tuple[int, int]] = field(default_factory=list)
    lag_peaks: Deque[int] = field(default_factory=lambda: deque(maxlen=LAG_PEAKS_KEPT))
    current_peak: int = 0
    peak_is_estimate: bool = False
    budget_trimmed: bool = False

    def drop_oldest(self) -> int:
        """Drop the oldest stored frame and return its size in bytes."""
        dropped = len(self.frames.popleft().jpeg)
        self.buffered_bytes -= dropped

        return dropped


class ActionRecognitionVisualizationBlockV1(WorkflowBlock):
    """Draw action timelines, keeping delayed frames per video when needed."""

    def __init__(self):
        self._streams: "OrderedDict[str, _Stream]" = OrderedDict()
        try:
            self._font_path: Optional[str] = str(resolve_font_path(LABEL_FONT_FAMILY))
        except Exception as error:
            # Labels still draw, in OpenCV's ASCII-only font.
            logger.warning(
                "Action Recognition Visualization could not load its label "
                "font and uses a basic one: %s",
                error,
            )
            self._font_path = None

    def close(self) -> None:
        """Release all buffered synchronized frames for tracked streams."""
        self._streams.clear()

    @classmethod
    def get_manifest(cls) -> Type[BlockManifest]:
        """Return the block's manifest.

        Returns:
            The visualization contract.
        """
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        predictions: ActionRecognitionResult,
        display_mode: str = "live",
        history_seconds: float = 30.0,
        max_delay_seconds: float = 60.0,
        max_frame_side: int = 1280,
        max_buffer_mb: int = 256,
    ) -> BlockResult:
        """Draw the timeline on the current frame or on a delayed one.

        Args:
            image: Original frame carrying source video metadata.
            predictions: Current result bundle from the same video's Action
                Recognition Model, carrying timeline and analyzed frame bounds.
            display_mode: "live" or "synchronized".
            history_seconds: Video time the strip covers.
            max_delay_seconds: Upper limit of the synchronized delay.
            max_frame_side: Longest stored side for synchronized frames.
            max_buffer_mb: Memory for stored frames across streams.

        Returns:
            Image, displayed_frame_number, actions, actions_frame_number
            and status.

        Raises:
            ValueError: If the image is not a uint8 BGR frame, or the
                display mode is unknown.
        """
        if display_mode not in ("live", "synchronized"):
            raise ValueError('display_mode must be "live" or "synchronized".')

        pixels = image.numpy_image
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR video image.")

        timeline = predictions.timeline
        analyzed_from_frame = predictions.analyzed_from_frame
        analyzed_until_frame = predictions.analyzed_until_frame
        number = image.video_metadata.frame_number
        synchronized = display_mode == "synchronized"
        state = self._stream_state(
            image.video_metadata,
            signature=(display_mode, max_delay_seconds, max_frame_side, *pixels.shape),
            analyzed_until_frame=analyzed_until_frame,
        )
        _record_coverage(
            state,
            start_frame=analyzed_from_frame,
            end_frame=analyzed_until_frame,
        )
        history_frames = max(1, round(history_seconds * state.fps))
        if synchronized:
            pixels, metadata, parent_metadata, root_metadata, status = (
                self._synchronized_frame(
                    state,
                    image=image,
                    timeline=timeline,
                    analyzed_until_frame=analyzed_until_frame,
                    max_delay_seconds=max_delay_seconds,
                    max_frame_side=max_frame_side,
                    max_buffer_mb=max_buffer_mb,
                )
            )
            displayed = metadata.frame_number
            actions_frame = displayed if status == STATUS_SYNCHRONIZED else None
        else:
            pixels = pixels.copy()
            metadata = image.video_metadata
            parent_metadata = image.parent_metadata
            root_metadata = image.workflow_root_ancestor_metadata
            displayed = number
            if analyzed_until_frame is None:
                status, actions_frame = STATUS_ANALYZING, None
            else:
                candidate = min(number, analyzed_until_frame)
                covered = _is_frame_covered(
                    state,
                    frame_number=candidate,
                    analyzed_until_frame=analyzed_until_frame,
                    timeline=timeline,
                )
                status = STATUS_LIVE if covered else STATUS_NOT_ANALYZED
                # A quiet latest window must not hide an action the strip still
                # shows; the badge names it with its age instead.
                actions_frame = _last_action_frame(
                    timeline,
                    until_frame=candidate,
                    since_frame=number - history_frames,
                )
                if actions_frame is None and covered:
                    actions_frame = candidate

        state.last_mark = analyzed_until_frame

        oldest_relevant = displayed - history_frames
        if state.frames:
            oldest_relevant = min(
                oldest_relevant, state.frames[0].metadata.frame_number
            )
        _prune_coverage(state, min_frame=oldest_relevant)

        held = synchronized and status == STATUS_ANALYZING
        if held:
            # The held first frame is not the stream's present; dim it to say so.
            _darken(pixels, keep=0.45)
        actions = _actions_at(timeline, frame_number=actions_frame)
        _draw_badges(
            pixels,
            font_path=self._font_path,
            actions=actions,
            actions_frame=actions_frame,
            displayed=displayed,
            fps=state.fps,
            status=status,
            waited_frames=number - state.first_frame if held else None,
            delay_frames=number - displayed if synchronized and not held else None,
        )
        _draw_strip(
            pixels,
            font_path=self._font_path,
            timeline=timeline,
            displayed=displayed,
            covered_ranges=state.covered_ranges,
            first_frame=state.first_frame,
            fps=state.fps,
            history_seconds=history_seconds,
        )

        output_image = WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=root_metadata,
            numpy_image=pixels,
            video_metadata=metadata,
        )
        result = {
            "image": output_image,
            "displayed_frame_number": displayed,
            "actions": [entry.class_name for entry in actions],
            "actions_frame_number": actions_frame,
            "status": status,
        }

        return result

    def _stream_state(
        self,
        metadata: VideoMetadata,
        *,
        signature: tuple,
        analyzed_until_frame: Optional[int],
    ) -> _Stream:
        video_id = metadata.video_identifier
        number = metadata.frame_number
        state = self._streams.get(video_id)
        model_reset = (
            state is not None
            and state.last_mark is not None
            and (analyzed_until_frame is None or analyzed_until_frame < state.last_mark)
        )
        if state is not None and (
            number < state.last_frame or signature != state.signature or model_reset
        ):
            del self._streams[video_id]
            state = None

        if state is None:
            fps = metadata.fps
            if fps is None or not math.isfinite(fps) or fps <= 0:
                # Same fallback as the model, so both count frames alike.
                fps = DEFAULT_SOURCE_FPS
            state = _Stream(
                fps=float(fps),
                signature=signature,
                first_frame=number,
                last_frame=number,
            )
            self._streams[video_id] = state
            while len(self._streams) > MAX_STREAMS:
                self._streams.popitem(last=False)

        self._streams.move_to_end(video_id)
        state.last_frame = number

        return state

    def _synchronized_frame(
        self,
        state: _Stream,
        *,
        image: WorkflowImageData,
        timeline: List[ActionRecognitionPrediction],
        analyzed_until_frame: Optional[int],
        max_delay_seconds: float,
        max_frame_side: int,
        max_buffer_mb: int,
    ) -> Tuple[
        np.ndarray,
        VideoMetadata,
        ImageParentMetadata,
        ImageParentMetadata,
        str,
    ]:
        """Store the frame; return the delayed frame, its metadata and status."""
        number = image.video_metadata.frame_number
        if not state.frames or state.frames[-1].metadata.frame_number != number:
            pixels = image.numpy_image
            height, width = pixels.shape[:2]
            scale = max_frame_side / max(height, width)
            if scale < 1:
                size = (max(1, round(width * scale)), max(1, round(height * scale)))
                pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
            jpeg = encode_image_to_jpeg_bytes(pixels, jpeg_quality=JPEG_QUALITY)
            state.frames.append(
                _BufferedFrame(
                    metadata=image.video_metadata,
                    parent_metadata=image.parent_metadata,
                    root_metadata=image.workflow_root_ancestor_metadata,
                    jpeg=jpeg,
                )
            )
            state.buffered_bytes += len(jpeg)

        delay = _playback_delay(
            state,
            number=number,
            analyzed_until_frame=analyzed_until_frame,
        )
        waiting_for_first_result = delay is None
        if waiting_for_first_result:
            # No result yet: hold the first frame until the limit, then play.
            delay = number - state.first_frame

        max_delay_frames = math.ceil(max_delay_seconds * state.fps)
        hard_limit = number - max_delay_frames
        target = number - min(delay, max_delay_frames)
        # Never keep a frame older than max_delay_seconds behind the input,
        # even when input frame numbers jump past the limit in one step.
        while (
            len(state.frames) > 1 and state.frames[0].metadata.frame_number < hard_limit
        ):
            state.drop_oldest()

        # Advance toward target, but pause on the first unshown frame of any
        # detected action so catch-up does not silently drop an entire event.
        while (
            len(state.frames) > 1
            and state.frames[1].metadata.frame_number <= target
            and not _preserves_unshown_action(
                state,
                frame_number=state.frames[0].metadata.frame_number,
                timeline=timeline,
                analyzed_until_frame=analyzed_until_frame,
            )
        ):
            state.drop_oldest()
        self._enforce_budget(state, max_buffer_mb=max_buffer_mb)

        shown = state.frames[0]
        shown_number = shown.metadata.frame_number
        pixels = decode_encoded_image_bytes(shown.jpeg)
        budget = max_buffer_mb * 1024**2
        if state.buffered_bytes > budget:
            # A single frame can exceed max_buffer_mb; return it now and release
            # its bytes so total retained memory stays within the budget.
            while state.frames:
                state.drop_oldest()
            self._warn_budget_trimmed(state, max_buffer_mb=max_buffer_mb)

        if waiting_for_first_result and shown_number == state.first_frame:
            status = STATUS_ANALYZING
        elif analyzed_until_frame is not None and shown_number <= analyzed_until_frame:
            if _is_frame_covered(
                state,
                frame_number=shown_number,
                analyzed_until_frame=analyzed_until_frame,
                timeline=timeline,
            ):
                status = STATUS_SYNCHRONIZED
            else:
                status = STATUS_NOT_ANALYZED
        else:
            status = STATUS_BEHIND

        if status != STATUS_ANALYZING:
            state.last_displayed_frame = shown_number

        return (
            pixels,
            shown.metadata,
            shown.parent_metadata,
            shown.root_metadata,
            status,
        )

    def _enforce_budget(self, state: _Stream, *, max_buffer_mb: int) -> None:
        budget = max_buffer_mb * 1024**2
        used = sum(stream.buffered_bytes for stream in self._streams.values())
        # Least recently used first, so a stream that stopped gives its frames
        # back before a running one loses delay. The current stream is last
        # and keeps the frame it shows.
        for stream in self._streams.values():
            kept = 1 if stream is state else 0
            while used > budget and len(stream.frames) > kept:
                used -= stream.drop_oldest()
                self._warn_budget_trimmed(stream, max_buffer_mb=max_buffer_mb)

    @staticmethod
    def _warn_budget_trimmed(stream: _Stream, *, max_buffer_mb: int) -> None:
        if stream.budget_trimmed:
            return

        stream.budget_trimmed = True
        logger.warning(
            "Action Recognition Visualization dropped delayed frames "
            "to stay within max_buffer_mb=%s. The playback delay is "
            "shorter than the model needs; lower max_frame_side or "
            "raise the budget.",
            max_buffer_mb,
        )


def _record_coverage(
    state: _Stream,
    *,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> None:
    if start_frame is None or end_frame is None or start_frame > end_frame:
        return

    ranges = sorted([*state.covered_ranges, (start_frame, end_frame)])
    merged: List[Tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    state.covered_ranges = merged


def _prune_coverage(state: _Stream, *, min_frame: int) -> None:
    if not state.covered_ranges:
        return

    state.covered_ranges = [
        (start, end) for start, end in state.covered_ranges if end >= min_frame
    ]


def _is_frame_covered(
    state: _Stream,
    *,
    frame_number: int,
    analyzed_until_frame: Optional[int],
    timeline: List[ActionRecognitionPrediction],
) -> bool:
    if analyzed_until_frame is None or frame_number > analyzed_until_frame:
        return False

    if _actions_at(timeline, frame_number=frame_number):
        return True

    covered = any(start <= frame_number <= end for start, end in state.covered_ranges)

    return covered


def _preserves_unshown_action(
    state: _Stream,
    *,
    frame_number: int,
    timeline: List[ActionRecognitionPrediction],
    analyzed_until_frame: Optional[int],
) -> bool:
    if analyzed_until_frame is None or frame_number > analyzed_until_frame:
        return False

    if (
        state.last_displayed_frame is not None
        and frame_number <= state.last_displayed_frame
    ):
        return False

    for entry in timeline:
        if not (entry.start_frame_idx <= frame_number <= entry.end_frame_idx):
            continue

        if (
            state.last_displayed_frame is None
            or state.last_displayed_frame < entry.start_frame_idx
        ):
            return True

    return False


def _playback_delay(
    state: _Stream,
    *,
    number: int,
    analyzed_until_frame: Optional[int],
) -> Optional[int]:
    """Frames to play behind the stream so shown frames have results.

    Results trail the stream in a sawtooth: the lag grows one frame per frame
    and drops when a model call lands. Playing behind by the peak of that
    sawtooth keeps every shown frame analyzed and the motion smooth. Returns
    None until the first result exists. A mark that returns to None resets the
    stream before this runs, so a None mark always means no result yet.
    """
    if analyzed_until_frame is None:
        return None

    lag = number - analyzed_until_frame
    if state.last_mark is None:
        # The first result: show the held first frame with its result once
        # before advancing to subsequent frames.
        state.lag_peaks.append(max(0, number - state.first_frame))
        state.peak_is_estimate = True
        state.current_peak = lag
    elif analyzed_until_frame != state.last_mark:
        if state.peak_is_estimate:
            # A model load or a cold server can hold the first result
            # back far longer than a stride, so a measured peak replaces
            # the first wait as soon as one exists.
            state.lag_peaks.clear()
            state.peak_is_estimate = False
        # current_peak is measured one frame before the result lands; adding
        # one matches the initial first-frame hold (stride + call) so steady
        # playback neither skips nor repeats a frame when the estimate clears.
        measured = state.current_peak + 1 if state.current_peak > 0 else 0
        state.lag_peaks.append(measured)
        state.current_peak = lag
    state.current_peak = max(state.current_peak, lag)
    delay = max(max(state.lag_peaks, default=0), state.current_peak)

    return delay


def _actions_at(
    timeline: List[ActionRecognitionPrediction],
    *,
    frame_number: Optional[int],
) -> List[ActionRecognitionPrediction]:
    if frame_number is None:
        return []

    by_class: Dict[str, ActionRecognitionPrediction] = {}
    for entry in timeline:
        if entry.start_frame_idx <= frame_number <= entry.end_frame_idx:
            by_class.setdefault(entry.class_name, entry)
    actions = [by_class[name] for name in sorted(by_class)]
    return actions


def _last_action_frame(
    timeline: List[ActionRecognitionPrediction],
    *,
    until_frame: int,
    since_frame: int,
) -> Optional[int]:
    """Last frame in [since_frame, until_frame] that any action covers."""
    ends = [
        min(entry.end_frame_idx, until_frame)
        for entry in timeline
        if entry.start_frame_idx <= until_frame
    ]
    recent_ends = [end for end in ends if end >= since_frame]
    if not recent_ends:
        return None

    last_frame = max(recent_ends)
    return last_frame


def _class_color(entry: ActionRecognitionPrediction) -> Tuple[int, int, int]:
    # An open-vocabulary class has no id, so its name picks a stable color.
    index = (
        entry.class_id
        if entry.class_id >= 0
        else zlib.crc32(entry.class_name.encode("utf-8"))
    )
    color = sv.ColorPalette.DEFAULT.by_idx(index).as_bgr()
    return color


def _sizes(image: np.ndarray) -> Tuple[int, int, int]:
    """Bar height, padding and font size, scaled to the frame."""
    height, width = image.shape[:2]
    unit = max(8, round(min(height, width * 9 / 16) * 0.045))
    sizes = (unit, max(2, unit // 4), max(8, round(unit * 0.7)))
    return sizes


def _draw_badges(
    image: np.ndarray,
    *,
    font_path: Optional[str],
    actions: List[ActionRecognitionPrediction],
    actions_frame: Optional[int],
    displayed: int,
    fps: float,
    status: str,
    waited_frames: Optional[int],
    delay_frames: Optional[int],
) -> None:
    unit, pad, font_px = _sizes(image)
    neutral = (70, 70, 70)
    if actions:
        badges = [(entry.class_name, _class_color(entry)) for entry in actions]
    elif status == STATUS_NOT_ANALYZED:
        badges = [("Not analyzed", neutral)]
    elif actions_frame is None:
        waited = "" if waited_frames is None else f" {int(waited_frames / fps)} s"
        badges = [(f"Analyzing...{waited}", neutral)]
    else:
        badges = [("No action", neutral)]

    lag_frames = 0 if actions_frame is None else max(0, displayed - actions_frame)
    if lag_frames > 0:
        age_seconds = lag_frames / fps
        if age_seconds >= 1.0:
            age_label = f"{int(age_seconds)} s ago"
        elif age_seconds >= 0.1:
            age_label = f"{age_seconds:.1f} s ago"
        else:
            unit_label = "frame" if lag_frames == 1 else "frames"
            age_label = f"{lag_frames} {unit_label} ago"
        # In live display the label describes the past; say how far back.
        badges.append((age_label, None))
    if delay_frames:
        badges.append((f"Delay {delay_frames / fps:.1f} s", None))

    x, y = pad * 2, pad * 2
    for label, color in badges:
        mask = _fitted_text(
            label,
            px=font_px,
            font_path=font_path,
            max_width=image.shape[1] - x - pad * 3,
        )
        if mask is None:
            break

        right = x + mask.shape[1] + pad * 2
        if color is None:
            _darken(image[y : y + unit, x:right], keep=0.4)
            ink = (235, 235, 235)
        else:
            cv2.rectangle(image, (x, y), (right, y + unit), color, -1)
            blue, green, red = color
            luminance = 0.114 * blue + 0.587 * green + 0.299 * red
            ink = (20, 20, 20) if luminance > 140 else (245, 245, 245)
        _blend_text(image, mask, x=x + pad, y=y + (unit - font_px) // 2, color=ink)
        x = right + pad


def _draw_strip(
    image: np.ndarray,
    *,
    font_path: Optional[str],
    timeline: List[ActionRecognitionPrediction],
    displayed: int,
    covered_ranges: List[Tuple[int, int]],
    first_frame: int,
    fps: float,
    history_seconds: float,
) -> None:
    unit, pad, font_px = _sizes(image)
    height, width = image.shape[:2]
    history_frames = max(1, round(history_seconds * fps))
    start = displayed - history_frames
    visible = [
        entry
        for entry in timeline
        if entry.end_frame_idx >= start and entry.start_frame_idx <= displayed
    ]
    lanes = _pick_lanes(visible)
    axis_px = max(8, round(font_px * 0.8))
    lane_count = max(1, len(lanes))
    band_height = lane_count * (unit + pad) + pad + axis_px + pad * 2
    if band_height > height * 0.6:
        return

    band_top = height - band_height
    label_width = min(
        round(width * 0.26),
        max(
            [
                _text_mask(lane.class_name, px=font_px, font_path=font_path).shape[1]
                for lane in lanes
            ],
            default=0,
        ),
    )
    x0 = pad * 2 + (label_width + pad * 2 if label_width else 0)
    x1 = width - pad * 2
    if x1 - x0 < 8:
        return

    def x_of(frame_number: float) -> int:
        position = x0 + (frame_number - start) / history_frames * (x1 - x0)
        clipped = int(round(min(max(position, x0), x1)))
        return clipped

    _darken(image[band_top:], keep=0.3)
    lanes_top = band_top + pad
    lanes_bottom = lanes_top + lane_count * (unit + pad) - pad
    for index in range(lane_count):
        top = lanes_top + index * (unit + pad)
        cv2.rectangle(image, (x0, top), (x1, top + unit), (55, 55, 55), -1)
        if index >= len(lanes):
            continue

        lane = lanes[index]
        mask = _fitted_text(
            lane.class_name, px=font_px, font_path=font_path, max_width=label_width
        )
        if mask is not None:
            _blend_text(
                image,
                mask,
                x=pad * 2,
                y=top + (unit - font_px) // 2,
                color=(235, 235, 235),
            )
        color = _class_color(lane)
        for entry in visible:
            if entry.class_name != lane.class_name:
                continue

            left = x_of(entry.start_frame_idx)
            right = max(left + 1, x_of(min(entry.end_frame_idx + 1, displayed)))
            cv2.rectangle(image, (left, top + 1), (right, top + unit - 1), color, -1)

    # Frames no successful window covered, including those after the mark.
    uncovered = _uncovered_spans(
        covered_ranges,
        visible=visible,
        start=start,
        first_frame=first_frame,
        displayed=displayed,
    )
    for gap_start, gap_end in uncovered:
        gap_x0 = x_of(gap_start)
        gap_x1 = x1 if gap_end >= displayed else x_of(gap_end + 1)
        _draw_pending(
            image,
            x0=gap_x0,
            y0=lanes_top,
            x1=gap_x1,
            y1=lanes_bottom,
            spacing=max(4, unit // 2),
        )
    cv2.line(image, (x1, lanes_top - 1), (x1, lanes_bottom + 1), (255, 255, 255), 1)

    axis_y = lanes_bottom + pad
    axis_color = (190, 190, 190)
    # The left edge is relative: early in a stream it lies before frame one.
    start_label = _text_mask(
        f"-{round(history_frames / fps)} s", px=axis_px, font_path=font_path
    )
    _blend_text(image, start_label, x=x0, y=axis_y, color=axis_color)
    seconds = max(0, int(displayed / fps))
    end_label = _text_mask(
        f"{seconds // 60}:{seconds % 60:02d}", px=axis_px, font_path=font_path
    )
    _blend_text(image, end_label, x=x1 - end_label.shape[1], y=axis_y, color=axis_color)


def _uncovered_spans(
    covered_ranges: List[Tuple[int, int]],
    *,
    visible: List[ActionRecognitionPrediction],
    start: int,
    first_frame: int,
    displayed: int,
) -> List[Tuple[int, int]]:
    """Return inclusive frame ranges within ``[start, displayed]`` lacking coverage.

    Frames before ``first_frame`` predate the stream, so no window could cover
    them; hatching them would look like a failed window.
    """
    combined = sorted(
        [
            *covered_ranges,
            *((entry.start_frame_idx, entry.end_frame_idx) for entry in visible),
        ]
    )
    uncovered: List[Tuple[int, int]] = []
    cursor = max(start, first_frame)
    for range_start, range_end in combined:
        if range_end < cursor:
            continue
        if range_start > displayed:
            break
        if range_start > cursor:
            uncovered.append((cursor, min(displayed, range_start - 1)))
        cursor = max(cursor, range_end + 1)
        if cursor > displayed:
            break

    if cursor <= displayed:
        uncovered.append((cursor, displayed))

    return uncovered


def _pick_lanes(
    visible: List[ActionRecognitionPrediction],
) -> List[ActionRecognitionPrediction]:
    """One entry per class in view, limited to the most recently active."""
    latest: Dict[str, ActionRecognitionPrediction] = {}
    for entry in visible:
        known = latest.get(entry.class_name)
        if known is None or entry.end_frame_idx > known.end_frame_idx:
            latest[entry.class_name] = entry
    recent = sorted(latest.values(), key=lambda entry: -entry.end_frame_idx)
    # A lane keeps its place while its class stays in view.
    lanes = sorted(
        recent[:MAX_LANES],
        key=lambda entry: (entry.class_id, entry.class_name),
    )
    return lanes


def _draw_pending(
    image: np.ndarray,
    *,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    spacing: int,
) -> None:
    """Hatch the part of the strip the model has not analyzed yet."""
    if x1 - x0 < 2:
        return

    # cv2 draws only on contiguous arrays, and a slice of the frame is not.
    zone = np.ascontiguousarray(image[y0:y1, x0:x1])
    zone[:] = (zone * 0.6 + 40).astype(np.uint8)
    zone_height, zone_width = zone.shape[:2]
    for offset in range(-zone_height, zone_width, spacing):
        cv2.line(
            zone,
            (offset, zone_height),
            (offset + zone_height, 0),
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    image[y0:y1, x0:x1] = zone


def _darken(zone: np.ndarray, *, keep: float) -> None:
    zone[:] = (zone * keep).astype(np.uint8)


def _fitted_text(
    text: str,
    *,
    px: int,
    font_path: Optional[str],
    max_width: int,
) -> Optional[np.ndarray]:
    """Mask of ``text`` cut from the end to fit ``max_width``, or None."""
    mask = _text_mask(text, px=px, font_path=font_path)
    while mask.shape[1] > max_width and len(text) > 1:
        text = text[:-1]
        mask = _text_mask(text, px=px, font_path=font_path)
    if mask.shape[1] > max_width:
        return None

    return mask


# The same class names and clock labels are drawn on every frame.
@lru_cache(maxsize=MAX_CACHED_TEXTS)
def _text_mask(text: str, *, px: int, font_path: Optional[str]) -> np.ndarray:
    """Render ``text`` as an alpha mask, in OpenCV's font without a font file."""
    if font_path is None:
        text = text.encode("ascii", errors="replace").decode("ascii")
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = cv2.getFontScaleFromHeight(font, max(4, round(px * 0.7)), 1)
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
        mask = np.zeros(
            (text_height + baseline + 2, max(1, text_width + 2)),
            dtype=np.uint8,
        )
        cv2.putText(mask, text, (1, text_height + 1), font, scale, 255, 1, cv2.LINE_AA)
        return mask

    font = ImageFont.truetype(font_path, px)
    canvas = Image.new(
        "L",
        (max(1, math.ceil(font.getlength(text))), max(1, round(px * 1.3))),
        0,
    )
    ImageDraw.Draw(canvas).text((0, 0), text, fill=255, font=font)
    mask = np.asarray(canvas)
    return mask


def _blend_text(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    """Blend a text mask into ``image`` with its top-left corner at (x, y)."""
    height, width = image.shape[:2]
    x, y = max(0, x), max(0, y)
    mask = mask[: max(0, height - y), : max(0, width - x)]
    zone = image[y : y + mask.shape[0], x : x + mask.shape[1]]
    alpha = mask[:, :, None].astype(np.float32) / 255.0
    zone[:] = (zone * (1.0 - alpha) + np.array(color) * alpha).astype(np.uint8)
