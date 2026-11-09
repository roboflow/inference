"""Delayed presentation of action intervals using the latest known timeline."""

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Literal, Optional, Type

import cv2
import numpy as np
from pydantic import ConfigDict, Field
from roboflow_workflows.core_steps.common.workload_presets import (
    STATEFUL_VIDEO_TEMPORAL_RESTRICTIONS,
)
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    OutputDefinition,
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

TYPE = "roboflow_core/action_recognition_visualization@v1"
MAX_STREAMS = 16
LONG_DESCRIPTION = """
Display earlier video frames with actions from the latest Action Recognition
Model `timeline`. Connect the original image and the same stream's timeline;
do not delay the timeline. The block selects only intervals
whose inclusive source-frame bounds contain the displayed frame.

`delay_seconds` is converted to frames using declared source FPS, pinned per
stream like the model. It is approximate video time, not wall-clock latency or
source PTS; missing/invalid FPS is rejected. Dropped frames can increase the
actual delay: the newest stored frame at or before the target is displayed.
No sleep or extra inference is performed. Model latency still affects playback.

Only matching action labels are drawn. During warmup the image is blank and
status is `buffering`.
`ready` means an old image is available, NOT that analysis is complete. An empty
actions list means no matching action is currently known; missing/failed analyses
and later overlapping windows can leave or revise gaps. There is no coverage or
finality guarantee. Predictions from the latest timeline are used on every call.

Images are copied into host RAM and downscaled, preserving aspect ratio, to
`max_frame_side` (default 640). This is a presentation output; use the original
image for models and geometric processing. The budget covers retained image
pixels across all streams, not temporary rendering allocations or Python objects.
Capacity is reserved for a full delay at source FPS; exceeding `max_buffer_mb`
raises an error instead of silently shortening the delay. At most 16 streams are
retained; restart the workflow to release ended streams. A frame-number restart,
resolution change or delay/resolution configuration change resets that stream.

Use a persistent video workflow with sequential frames. Duplicate frame numbers
are idempotent for buffering. Output image metadata and `displayed_frame_number`
identify the old frame; transport IDs/PTS may still identify the current request.
Configure the video output to this block's `image`. There is no end-of-stream
flush: this version targets continuous streams; a finite video's buffered tail
is not emitted. The block does not change the model, engine or video transport.
"""


class BlockManifest(WorkflowBlockManifest):
    """Configure bounded, delayed action-recognition presentation."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition Visualization",
            "version": "v1",
            "short_description": "Delay video and draw actions for the displayed frame.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "ui_manifest": {"section": "video", "icon": "far fa-clock-rotate-left"},
        }
    )
    type: Literal["roboflow_core/action_recognition_visualization@v1"]
    image: Selector(kind=[IMAGE_KIND]) = Field(
        description="Original video frame with stable stream identity and source FPS.",
        examples=["$inputs.image"],
    )
    timeline: Selector(kind=[ACTION_RECOGNITION_PREDICTION_KIND]) = Field(
        description="Latest cumulative timeline from the same stream's Action Recognition Model.",
        examples=["$steps.actions.timeline"],
    )
    delay_seconds: float = Field(
        default=10.0,
        ge=0,
        le=300,
        allow_inf_nan=False,
        description="Approximate video-time delay, converted using source FPS. Does not guarantee completed analysis.",
        examples=[10.0],
    )
    max_frame_side: int = Field(
        default=640,
        ge=32,
        le=4096,
        description="Maximum presentation image side in pixels. Downscales before buffering; never upscales.",
        examples=[640, 1280],
    )
    max_buffer_mb: int = Field(
        default=256,
        ge=1,
        le=16384,
        description="Total retained pixel budget in MiB across this block's streams. Insufficient capacity raises an error.",
        examples=[256, 512],
    )

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        """Describe the delayed image and its matching action labels.

        Returns:
            Image, displayed frame number (null during warmup), actions and status.
        """
        return [
            OutputDefinition(name="image", kind=[IMAGE_KIND]),
            OutputDefinition(name="displayed_frame_number", kind=[INTEGER_KIND]),
            OutputDefinition(name="actions", kind=[LIST_OF_VALUES_KIND]),
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
class _Stream:
    fps: float
    signature: tuple
    reserved_bytes: int
    frames: Deque[WorkflowImageData] = field(default_factory=deque)
    last_frame: int = -1


class ActionRecognitionVisualizationBlockV1(WorkflowBlock):
    """Own presentation frames in host RAM, isolated by video identifier."""

    def __init__(self):
        self._streams: Dict[str, _Stream] = {}

    @classmethod
    def get_manifest(cls) -> Type[BlockManifest]:
        """Return the block's manifest.

        Returns:
            The delayed visualization contract.
        """
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        timeline: List[ActionRecognitionPrediction],
        delay_seconds: float = 10.0,
        max_frame_side: int = 640,
        max_buffer_mb: int = 256,
    ) -> BlockResult:
        """Buffer a frame and annotate an older image using current knowledge.

        Args:
            image: Original frame carrying source video metadata.
            timeline: Current intervals from the same video's model output.
            delay_seconds: Approximate delay in source-video seconds.
            max_frame_side: Maximum stored and displayed image side.
            max_buffer_mb: Total retained pixel budget in MiB.

        Returns:
            Image, displayed_frame_number, actions and buffering/ready status.

        Raises:
            ValueError: If source FPS or configuration is invalid.
            MemoryError: If the full delay cannot fit the memory or stream limit.
        """
        if not math.isfinite(delay_seconds) or not 0 <= delay_seconds <= 300:
            raise ValueError("delay_seconds must be finite and between 0 and 300.")
        if not 32 <= max_frame_side <= 4096 or not 1 <= max_buffer_mb <= 16384:
            raise ValueError("Invalid max_frame_side or max_buffer_mb.")

        metadata = image.video_metadata
        if not metadata.video_identifier:
            raise ValueError("A stable video_identifier is required.")
        if metadata.fps is None or not math.isfinite(metadata.fps) or metadata.fps <= 0:
            raise ValueError("A positive, finite source FPS is required.")

        pixels = image.numpy_image
        if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
            raise ValueError("Expected a uint8 BGR video image.")

        height, width = pixels.shape[:2]
        scale = min(1.0, max_frame_side / max(height, width))
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        signature = (delay_seconds, max_frame_side, height, width)
        video_id, number = metadata.video_identifier, metadata.frame_number
        state = self._streams.get(video_id)
        if state and (number < state.last_frame or signature != state.signature):
            del self._streams[video_id]
            state = None

        fps = state.fps if state else float(metadata.fps)
        depth = math.ceil(delay_seconds * fps)
        # One predecessor for gaps, plus the current input. Reserve at source FPS
        # so insufficient memory fails before a partial warmup is presented.
        required = (depth + 2) * size[0] * size[1] * 3
        others = sum(
            s.reserved_bytes for key, s in self._streams.items() if key != video_id
        )
        if others + required > max_buffer_mb * 1024**2:
            raise MemoryError(
                "Delay buffer exceeds max_buffer_mb; reduce delay/resolution or increase the budget."
            )
        if state is None:
            if len(self._streams) >= MAX_STREAMS:
                raise MemoryError(
                    "Too many video streams; restart the workflow to release ended streams."
                )
            state = _Stream(fps=fps, signature=signature, reserved_bytes=required)
            self._streams[video_id] = state

        if number != state.last_frame or not state.frames:
            owned = (
                cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
                if size != (width, height)
                else pixels.copy()
            )
            stored = WorkflowImageData.copy_and_replace(
                origin_image_data=image,
                numpy_image=owned,
                video_metadata=metadata.model_copy(deep=True),
            )
            state.frames.append(stored)
            state.last_frame = number

        target = number - depth
        while (
            len(state.frames) > 1
            and state.frames[1].video_metadata.frame_number <= target
        ):
            state.frames.popleft()

        oldest = state.frames[0]
        available = oldest.video_metadata.frame_number <= target
        displayed = oldest.video_metadata.frame_number if available else None
        actions = sorted(
            {
                entry.class_name
                for entry in timeline
                if available
                and entry.start_frame_idx <= displayed <= entry.end_frame_idx
            }
        )
        output = (
            oldest.numpy_image.copy()
            if available
            else np.zeros((size[1], size[0], 3), dtype=np.uint8)
        )
        _draw_labels(output, actions=actions)
        result = {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=oldest if available else image,
                numpy_image=output,
            ),
            "displayed_frame_number": displayed,
            "actions": actions,
            "status": "ready" if available else "buffering",
        }
        return result


def _draw_labels(image, *, actions):
    if not actions:
        return

    height, width = image.shape[:2]
    scale = max(0.25, min(width / 900, height / 500, 0.7))
    line_height = max(12, round(32 * scale))
    capacity = max(1, (height - 8) // line_height)
    labels = actions[:capacity]
    cv2.rectangle(
        image,
        (0, 0),
        (width - 1, min(height - 1, len(labels) * line_height + 8)),
        (25, 25, 25),
        -1,
    )
    for index, label in enumerate(labels):
        label = label.encode("ascii", errors="replace").decode("ascii")
        while (
            label
            and cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            > width - 12
        ):
            label = label[:-1]
        cv2.putText(
            image,
            label,
            (6, (index + 1) * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
