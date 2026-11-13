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
from pydantic import ConfigDict, Field, model_validator
from roboflow_workflows._compat_names import get_logger
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS,
)
from roboflow_workflows.core_steps.visualizations.common.fonts import resolve_font_path
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    OutputDefinition,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_PREDICTION_KIND,
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

LONG_DESCRIPTION = """
Draw the results of an Action Recognition Model on its video. Connect the
original image, the model's `timeline` and its `analyzed_until_frame`.

An action recognition result always describes frames the stream has already
passed. The block offers two honest ways to show that.

`live` (default) shows the current frame with no delay. A strip at the bottom
covers the last `history_seconds`: one lane per class, a bar for each action
range, and a hatched zone for frames the model has not analyzed yet. A badge
names the actions at the last analyzed frame and how long ago that was. This
mode keeps no frames and adds no latency.

`synchronized` plays the video late, so each label sits on the frame it
describes. The block picks the delay: it follows how far the model's results
trail the stream, which is about one stride plus one model call. Until the
first result arrives, the first frame is held with an "Analyzing" badge, then
playback starts from that frame. `max_delay_seconds` caps the delay. When the
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

`status` is `analyzing` before the first result, then `live`, `synchronized`
or `behind`. `actions` lists the classes at `actions_frame_number`: the
displayed frame in `synchronized`, the last analyzed frame in `live`.
`displayed_frame_number` identifies the frame in the image. Configure the
video output to this block's `image`.
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
    timeline: Selector(kind=[ACTION_RECOGNITION_PREDICTION_KIND]) = Field(
        description="Timeline output of the same stream's Action Recognition Model.",
        examples=["$steps.actions.timeline"],
    )
    analyzed_until_frame: Optional[Selector(kind=[INTEGER_KIND])] = Field(
        default=None,
        description=(
            "The model's analyzed_until_frame output. It marks which frames "
            "have a result. Required for synchronized display."
        ),
        examples=["$steps.actions.analyzed_until_frame"],
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

    @model_validator(mode="after")
    def validate_synchronized_inputs(self) -> "BlockManifest":
        """Require the analyzed-frame mark when playback must follow it.

        Returns:
            The validated manifest.

        Raises:
            ValueError: If synchronized display has no analyzed_until_frame.
        """
        if self.display_mode == "synchronized" and self.analyzed_until_frame is None:
            raise ValueError(
                "synchronized display needs analyzed_until_frame from the "
                "Action Recognition Model."
            )

        return self

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
    def accepts_empty_values(cls) -> bool:
        """Keep drawing while the model has no analyzed frame to report.

        Returns:
            True, so a null analyzed_until_frame does not skip the step.
        """
        return True

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
    lag_peaks: Deque[int] = field(default_factory=lambda: deque(maxlen=LAG_PEAKS_KEPT))
    current_peak: int = 0
    peak_is_estimate: bool = False
    budget_trimmed: bool = False


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
        timeline: Optional[List[ActionRecognitionPrediction]],
        analyzed_until_frame: Optional[int] = None,
        display_mode: str = "live",
        history_seconds: float = 30.0,
        max_delay_seconds: float = 60.0,
        max_frame_side: int = 1280,
        max_buffer_mb: int = 256,
    ) -> BlockResult:
        """Draw the timeline on the current frame or on a delayed one.

        Args:
            image: Original frame carrying source video metadata.
            timeline: Current ranges from the same video's model output.
            analyzed_until_frame: Last frame the model has a result for.
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

        timeline = timeline or []
        number = image.video_metadata.frame_number
        synchronized = display_mode == "synchronized"
        state = self._stream_state(
            image.video_metadata,
            signature=(display_mode, max_delay_seconds, max_frame_side, *pixels.shape),
        )
        if synchronized:
            pixels, metadata, status = self._synchronized_frame(
                state,
                image=image,
                analyzed_until_frame=analyzed_until_frame,
                max_delay_seconds=max_delay_seconds,
                max_frame_side=max_frame_side,
                max_buffer_mb=max_buffer_mb,
            )
            displayed = metadata.frame_number
            actions_frame = displayed if status == STATUS_SYNCHRONIZED else None
        else:
            pixels, metadata = pixels.copy(), image.video_metadata
            displayed = number
            if analyzed_until_frame is None:
                status, actions_frame = STATUS_ANALYZING, None
            else:
                status = STATUS_LIVE
                actions_frame = min(number, analyzed_until_frame)

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
            waited_frames=number - state.first_frame if held else None,
        )
        _draw_strip(
            pixels,
            font_path=self._font_path,
            timeline=timeline,
            displayed=displayed,
            analyzed_until_frame=analyzed_until_frame,
            fps=state.fps,
            history_seconds=history_seconds,
        )

        result = {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=pixels,
                video_metadata=metadata,
            ),
            "displayed_frame_number": displayed,
            "actions": [entry.class_name for entry in actions],
            "actions_frame_number": actions_frame,
            "status": status,
        }
        return result

    def _stream_state(self, metadata: VideoMetadata, *, signature: tuple) -> _Stream:
        video_id = metadata.video_identifier
        number = metadata.frame_number
        state = self._streams.get(video_id)
        if state is not None and (
            number < state.last_frame or signature != state.signature
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
        analyzed_until_frame: Optional[int],
        max_delay_seconds: float,
        max_frame_side: int,
        max_buffer_mb: int,
    ) -> Tuple[np.ndarray, VideoMetadata, str]:
        """Store the frame; return the delayed frame to show and its status."""
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
                _BufferedFrame(metadata=image.video_metadata, jpeg=jpeg)
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
        target = number - min(delay, math.ceil(max_delay_seconds * state.fps))
        while len(state.frames) > 1 and state.frames[1].metadata.frame_number <= target:
            state.buffered_bytes -= len(state.frames.popleft().jpeg)
        self._enforce_budget(state, max_buffer_mb=max_buffer_mb)

        shown = state.frames[0]
        shown_number = shown.metadata.frame_number
        if waiting_for_first_result and shown_number == state.first_frame:
            status = STATUS_ANALYZING
        elif analyzed_until_frame is not None and shown_number <= analyzed_until_frame:
            status = STATUS_SYNCHRONIZED
        else:
            status = STATUS_BEHIND
        pixels = decode_encoded_image_bytes(shown.jpeg)
        return pixels, shown.metadata, status

    def _enforce_budget(self, state: _Stream, *, max_buffer_mb: int) -> None:
        budget = max_buffer_mb * 1024**2
        used = sum(stream.buffered_bytes for stream in self._streams.values())
        while used > budget and len(state.frames) > 1:
            dropped = len(state.frames.popleft().jpeg)
            state.buffered_bytes -= dropped
            used -= dropped
            if not state.budget_trimmed:
                state.budget_trimmed = True
                logger.warning(
                    "Action Recognition Visualization dropped delayed frames to "
                    "stay within max_buffer_mb=%s. The playback delay is shorter "
                    "than the model needs; lower max_frame_side or raise the "
                    "budget.",
                    max_buffer_mb,
                )


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
    None until the first result exists.
    """
    if analyzed_until_frame is None and state.last_mark is None:
        return None

    if analyzed_until_frame is None:
        lag = number - state.last_mark
    else:
        lag = number - analyzed_until_frame
        if state.last_mark is None:
            # The first result: play on from the held first frame. The wait
            # so far is one stride plus one call, the peak to expect. A peak
            # falls on the frame before a result lands, hence the one less.
            state.lag_peaks.append(max(0, number - state.first_frame - 1))
            state.peak_is_estimate = True
            state.current_peak = lag
        elif analyzed_until_frame != state.last_mark:
            if state.peak_is_estimate:
                # A model load or a cold server can hold the first result
                # back far longer than a stride, so a measured peak replaces
                # the first wait as soon as one exists.
                state.lag_peaks.clear()
                state.peak_is_estimate = False
            state.lag_peaks.append(state.current_peak)
            state.current_peak = lag
        state.last_mark = analyzed_until_frame
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
    waited_frames: Optional[int],
) -> None:
    unit, pad, font_px = _sizes(image)
    neutral = (70, 70, 70)
    if actions_frame is None:
        waited = "" if waited_frames is None else f" {int(waited_frames / fps)} s"
        badges = [(f"Analyzing...{waited}", neutral)]
    elif actions:
        badges = [(entry.class_name, _class_color(entry)) for entry in actions]
    else:
        badges = [("No action", neutral)]
    age_seconds = 0 if actions_frame is None else int((displayed - actions_frame) / fps)
    if age_seconds > 0:
        # In live display the label describes the past; say how far back.
        badges.append((f"{age_seconds} s ago", None))

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
    analyzed_until_frame: Optional[int],
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

    pending_from = start if analyzed_until_frame is None else analyzed_until_frame + 1
    if pending_from <= displayed:
        _draw_pending(
            image,
            x0=x_of(pending_from),
            y0=lanes_top,
            x1=x1,
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
