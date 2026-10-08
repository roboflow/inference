"""Stateless overlays for the last action-recognition window and its history."""

import math
from typing import List, Literal, Optional, Type
from zlib import crc32

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from roboflow_workflows.core_steps.visualizations.common.base import (
    VisualizationManifest,
)
from roboflow_workflows.core_steps.visualizations.text_display.utils import (
    draw_background,
    draw_rounded_rectangle,
)
from roboflow_workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_PREDICTION_KIND,
    DICTIONARY_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import (
    Discovery,
    RuntimeRestriction,
    WorkOperation,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    DependentResource,
    WorkflowBlock,
)

TYPE = "roboflow_core/action_recognition_visualization@v1"
# OpenCV BGR, shared by labels and timeline; class names keep their colors.
COLORS = [(193, 240, 126), (255, 201, 142), (167, 180, 255), (130, 218, 255)]
TEXT = (247, 243, 237)
MUTED = (204, 190, 177)
BACKGROUND = (35, 22, 12)
LONG_DESCRIPTION = """
Draw Action Recognition results into an image, without custom Python or a player
change. Connect the source image and the model's `window` output. Compact mode
(default) shows all classes that fit from the last analyzed window, its coverage,
and an explicit overflow count. These are window labels, not current-frame actions.
No artificial confidence scores are displayed.

Timeline mode also requires the same model's `timeline` output. It clips that
accumulated history to the latest analyzed window and marks frames after the
window as not yet analyzed. Intervals can overlap or merge; they are not counts
of events. At most `max_classes` rows are shown, with an overflow count.

Both modes distinguish collecting context, a successful window with no reported
action under the class filter, and a failed analysis. An error hides old labels
and timeline bars until the model returns a valid result. The model owns the
state; this block does not store frames or history and never runs inference.
Times are estimates from source frame numbers and declared FPS; without declared
FPS, frame indices are displayed. Holding labels does not make synchronous
inference asynchronous or guarantee live video throughput. Output dimensions
and image metadata are preserved. Tensor images use the existing NumPy drawing
path, like Text Display.
"""


class BlockManifest(VisualizationManifest):
    """Configure a compact or timeline overlay from one model's outputs."""

    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition Visualization",
            "version": "v1",
            "short_description": "Show action labels or a timeline for the last analyzed window.",
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["action recognition", "video", "timeline"],
            "ui_manifest": {"section": "visualization", "icon": "far fa-chart-bar"},
        }
    )
    type: Literal["roboflow_core/action_recognition_visualization@v1"]
    window: Selector(kind=[DICTIONARY_KIND]) = Field(
        description="The Action Recognition Model window output: state, classes and analyzed coverage.",
        examples=["$steps.actions.window"],
    )
    mode: Literal["compact", "timeline"] = Field(
        default="compact",
        description="Compact window labels for small previews, or accumulated timeline intervals clipped to the window.",
        examples=["compact", "timeline"],
        json_schema_extra={"always_visible": True},
    )
    timeline: Optional[Selector(kind=[ACTION_RECOGNITION_PREDICTION_KIND])] = Field(
        default=None,
        description="The same model's timeline output. Required in timeline mode; unused in compact mode.",
        examples=["$steps.actions.timeline"],
        json_schema_extra={"relevant_for": {"mode": {"values": ["timeline"]}}},
    )
    max_classes: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Maximum displayed classes; additional classes are summarized with a count. Compact mode also limits labels to available width.",
        examples=[4, 6],
    )

    @model_validator(mode="after")
    def validate_timeline(self) -> "BlockManifest":
        """Require history when the selected mode draws timeline intervals.

        Returns:
            The validated manifest.

        Raises:
            ValueError: If timeline mode has no timeline selector.
        """
        if self.mode == "timeline" and self.timeline is None:
            raise ValueError("Timeline mode requires the model's timeline output.")
        return self

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        """Return the supported execution-engine version range.

        Returns:
            The engine compatibility range.
        """
        return ">=1.4.0,<2.0.0"

    def discover_work_operations(self) -> List[WorkOperation]:
        """Declare image rendering work.

        Returns:
            The visualization operation.
        """
        return [WorkOperation.VISUALIZATION]

    def discover_dependent_resources(self) -> List[DependentResource]:
        """Declare external resources.

        Returns:
            An empty list; rendering has no external resources.
        """
        return []

    def get_actual_restrictions(
        self, *, ignore_environment_restrictions: bool = False
    ) -> Discovery[RuntimeRestriction]:
        """Describe runtime restrictions for image rendering.

        Args:
            ignore_environment_restrictions: Whether to ignore host conditions.

        Returns:
            A complete, empty restriction set.
        """
        return Discovery[RuntimeRestriction](
            items=[], complete=True, unknown_reasons=[]
        )


class _Window(BaseModel):
    status: Literal["collecting", "ready", "error"] = Field(
        description="Latest analysis state."
    )
    classes: List[str] = Field(
        description="All classes from the latest successful call."
    )
    start_frame: Optional[int] = Field(
        description="First sampled source frame, inclusive."
    )
    end_frame: Optional[int] = Field(
        description="Last sampled source frame, inclusive."
    )
    fps: Optional[float] = Field(description="Declared source FPS, or null if assumed.")
    video_identifier: str = Field(description="Source video identity.")

    @model_validator(mode="after")
    def _validate_coverage(self):
        if self.status == "ready" and (
            self.start_frame is None
            or self.end_frame is None
            or self.end_frame < self.start_frame
        ):
            raise ValueError("A ready window requires ordered source frame bounds.")
        if self.fps is not None and (not math.isfinite(self.fps) or self.fps <= 0):
            raise ValueError("Window FPS must be positive and finite, or null.")
        return self


class ActionRecognitionVisualizationBlockV1(WorkflowBlock):
    """Render both modes without retaining per-stream state."""

    @classmethod
    def get_manifest(cls) -> Type[BlockManifest]:
        """Return the block's input and output contract.

        Returns:
            The visualization manifest.
        """
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        window: dict,
        mode: str = "compact",
        timeline: Optional[List[ActionRecognitionPrediction]] = None,
        max_classes: int = 4,
        copy_image: bool = True,
    ) -> BlockResult:
        """Draw the model window on an image with its source metadata intact.

        Args:
            image: Current source image, NumPy or tensor backed.
            window: The same stream's Action Recognition Model window output.
            mode: Compact labels or timeline intervals.
            timeline: The model's accumulated history, required for timeline mode.
            max_classes: Maximum visible classes before an overflow indicator.
            copy_image: Whether to preserve the input pixels.

        Returns:
            The annotated image under the image output key.

        Raises:
            ValueError: If coverage is invalid, streams differ, or history is missing.
        """
        snapshot = _Window.model_validate(window)
        if snapshot.video_identifier != image.video_metadata.video_identifier:
            raise ValueError(
                "Image and action-recognition window must use the same video."
            )
        if (
            snapshot.status == "ready"
            and image.video_metadata.frame_number < snapshot.end_frame
        ):
            raise ValueError(
                "Image precedes the analyzed window; use the matching model output."
            )
        if mode == "timeline" and timeline is None:
            raise ValueError("Timeline mode requires the model's timeline output.")
        output = image.numpy_image
        if copy_image:
            output = output.copy()
        else:
            image.declare_numpy_image_mutated()
        _render(
            output,
            window=snapshot,
            current_frame=image.video_metadata.frame_number,
            mode=mode,
            timeline=timeline or [],
            max_classes=max_classes,
        )
        result = {
            "image": WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=output
            )
        }
        return result


def _color(name):
    color = COLORS[crc32(name.encode("utf-8")) % len(COLORS)]
    return color


def _time(frame, fps):
    if fps is None:
        return f"f{frame}"
    seconds = frame / fps
    minutes = int(seconds // 60)
    label = f"~{minutes:02d}:{seconds % 60:04.1f}"
    return label


def _fit(text, width, scale=0.7):
    # This matches the existing OpenCV text renderer's character repertoire.
    text = text.encode("ascii", errors="replace").decode("ascii")
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= width:
        return text
    left, right = 0, len(text)
    while left < right:
        mid = (left + right + 1) // 2
        if (
            cv2.getTextSize(text[:mid] + "...", cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][
                0
            ]
            <= width
        ):
            left = mid
        else:
            right = mid - 1
    shortened = text[:left] + "..."
    return shortened


def _clip_history(timeline, start, end):
    # Model intervals and sampled bounds are inclusive source-frame indices.
    # Use half-open bounds internally so even a single-frame action has width.
    clipped = []
    for entry in timeline:
        lo = max(start, entry.start_frame_idx)
        hi = min(end, entry.end_frame_idx)
        if lo <= hi:
            clipped.append((entry.class_name, lo, hi + 1))
    return clipped


def _render(img, *, window, current_frame, mode, timeline, max_classes):
    height, width = img.shape[:2]
    unit = min(max(min(width / 1280, height / 720), 0.45), width / 480, height / 320)
    # Coordinates below are local to a bottom-left panel in a 1280x720 design.
    ready = window.status == "ready"
    history = (
        _clip_history(timeline, window.start_frame, window.end_frame)
        if ready and window.classes and mode == "timeline"
        else []
    )
    classes = sorted(set(window.classes)) if ready else []
    if mode == "timeline" and ready and window.classes:
        classes = sorted(set(classes) | {name for name, _, _ in history})
    visible = classes[:max_classes]
    show_timeline = mode == "timeline" and ready and bool(classes)
    panel_w = min(1232 if show_timeline else 840, width / unit - 48)
    narrow = panel_w < 840
    panel_h = 166 if ready and classes else 142
    if show_timeline:
        visible = visible[: max(1, int((height / unit - 210) / 42))]
        panel_h = 151 + 42 * len(visible)
    x0, y0 = round(24 * unit), height - round((24 + panel_h) * unit)

    def text(label, x, y, *, size=0.7, color=TEXT):
        cv2.putText(
            img,
            label,
            (x0 + round(x * unit), y0 + round(y * unit)),
            cv2.FONT_HERSHEY_SIMPLEX,
            size * unit,
            color,
            max(1, round(unit)),
            cv2.LINE_AA,
        )

    def box(x, y, w, h, color, radius=5):
        draw_rounded_rectangle(
            img=img,
            pt1=(x0 + round(x * unit), y0 + round(y * unit)),
            pt2=(x0 + round((x + w) * unit), y0 + round((y + h) * unit)),
            color=color,
            radius=round(radius * unit),
        )

    draw_background(
        img=img,
        x1=x0,
        y1=y0,
        x2=x0 + round(panel_w * unit),
        y2=y0 + round(panel_h * unit),
        bg_color_bgr=BACKGROUND,
        background_opacity=0.94,
        border_radius=round(12 * unit),
    )
    if not ready:
        failed = window.status == "error"
        text("ACTION RECOGNITION", 22, 34, size=0.55, color=MUTED)
        text(
            "Last analysis failed" if failed else "Collecting context",
            22,
            77,
            size=0.9,
            color=(182, 173, 255) if failed else (138, 210, 255),
        )
        text(
            "Waiting for a valid analysis" if failed else "No analyzed window yet",
            22,
            116,
            color=MUTED,
        )
        return

    coverage = f"{_time(window.start_frame, window.fps)} - {_time(window.end_frame, window.fps)}"
    text(
        "LAST WINDOW" if narrow else "LAST ANALYZED WINDOW",
        22,
        34,
        size=0.65,
        color=COLORS[0],
    )
    coverage_width = cv2.getTextSize(coverage, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
    text(coverage, max(220, panel_w - 22 - coverage_width), 34, size=0.65)
    if not classes:
        text("No actions reported", 22, 78, size=0.9)
        text("Under the selected class filter", 22, 117, color=MUTED)
        return

    if not show_timeline:
        cursor, shown = 22, 0
        for name in visible:
            label = _fit(name, min(260, panel_w - 230))
            chip_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0] + 30
            reserve = 145 if shown < len(classes) - 1 else 0
            if cursor + chip_w + reserve > panel_w - 22:
                break
            box(cursor, 55, chip_w, 40, _color(name))
            text(label, cursor + 15, 83, color=BACKGROUND)
            cursor += chip_w + 12
            shown += 1
        if shown < len(classes):
            text(f"+{len(classes) - shown} more", cursor, 83, color=MUTED)
        text(
            f"Held until next analysis | Frame {_time(current_frame, window.fps)}",
            22,
            142,
            size=0.6,
            color=MUTED,
        )
        return

    text(
        f"History in window | Frame {_time(current_frame, window.fps)}",
        22,
        67,
        size=0.6,
        color=MUTED,
    )
    start, end = window.start_frame, window.end_frame + 1
    axis_end = max(end, current_frame + 1)
    label_width = min(170, panel_w * 0.2)
    plot_x = label_width + 38
    plot_w = panel_w - plot_x - 30

    def position(frame):
        return plot_x + (frame - start) / (axis_end - start) * plot_w

    analyzed_x = position(end)
    for index, name in enumerate(visible):
        y = 88 + index * 42
        text(_fit(name, label_width, 0.65), 22, y + 22, size=0.65, color=_color(name))
        box(plot_x, y, analyzed_x - plot_x, 27, (68, 52, 36))
        if axis_end > end:
            # A separate grey track is explicitly labelled, never extrapolated.
            box(analyzed_x, y, plot_x + plot_w - analyzed_x, 27, (47, 38, 31), radius=0)
        for label, lo, hi in history:
            if label == name:
                box(
                    position(lo),
                    y,
                    max(1, position(hi) - position(lo)),
                    27,
                    _color(name),
                )
    axis_y = 104 + len(visible) * 42
    for fraction in ((0, 1) if narrow else (0, 0.5, 1)):
        frame = round(start + fraction * (axis_end - 1 - start))
        label = _time(frame, window.fps)
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        text(
            label,
            plot_x + fraction * plot_w - fraction * label_w,
            axis_y,
            size=0.5,
            color=MUTED,
        )
    footer = "Accumulated intervals"
    if len(classes) > len(visible):
        footer += f" | +{len(classes) - len(visible)} more classes"
    if not narrow or axis_end == end:
        text(footer, 22, panel_h - 17, size=0.5, color=MUTED)
    if axis_end > end:
        label = f"Grey: no result after {_time(window.end_frame, window.fps)}"
        if narrow and len(classes) > len(visible):
            label += f" | +{len(classes) - len(visible)} classes"
        label_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        text(label, panel_w - 22 - label_w, panel_h - 17, size=0.5, color=MUTED)
