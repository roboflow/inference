from collections import OrderedDict
from typing import List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field
from supervision.annotators.utils import resolve_text_background_xyxy, wrap_text

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    TensorNativePrediction,
    split_key_point_prediction,
)
from inference.core.workflows.core_steps.visualizations.common.base_colorable_tensor import (
    ColorableVisualizationBlock,
    ColorableVisualizationManifest,
)
from inference.core.workflows.core_steps.visualizations.common.base_tensor import (
    OUTPUT_IMAGE_KEY,
    empty_predictions_passthrough,
    resolve_overlap_winners,
    to_supervision_for_annotation,
)
from inference.core.workflows.core_steps.visualizations.common.label_text import (
    build_detection_labels,
)
from inference.core.workflows.core_steps.visualizations.common.utils import str_to_color
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    INTEGER_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/label_visualization@v1"
SHORT_DESCRIPTION = (
    "Draw labels on an image at specific coordinates based on provided detections."
)
LONG_DESCRIPTION = """
Draw text labels on detected objects with customizable content, position, styling, and background colors to display information like class names, confidence scores, tracking IDs, or other detection metadata.

## How This Block Works

This block takes an image and detection predictions and draws text labels on each detected object. The block:

1. Takes an image and predictions as input
2. Extracts label text for each detection based on the selected text option (class name, confidence, tracker ID, dimensions, area, time in zone, or index)
3. Determines label position based on the selected anchor point (center, corners, edges, or center of mass)
4. Applies background color styling based on the selected color palette, with colors assigned by class, index, or track ID
5. Renders text labels with customizable text color, scale, thickness, padding, and border radius using Supervision's LabelAnnotator
6. Returns an annotated image with text labels overlaid on the original image

The block supports various text content options including class names, confidence scores, combination of class and confidence, tracker IDs (for tracked objects), time in zone (for zone analysis), object dimensions (center coordinates and width/height), area, or detection index. Labels are rendered with colored backgrounds that match the object's assigned color from the palette, and text styling (color, size, thickness) can be customized for optimal visibility. The labels can be positioned at any anchor point relative to each detection, allowing flexible placement for different visualization needs.

## Common Use Cases

- **Information Display on Detections**: Add informative text labels showing class names, confidence scores, or other metadata directly on detected objects for quick identification and validation
- **Model Performance Visualization**: Display confidence scores or class predictions on detected objects to visualize model certainty, identify low-confidence detections, and validate model performance
- **Object Tracking Visualization**: Show tracker IDs on tracked objects to visualize object tracking across frames, monitor persistent object identities, or debug tracking algorithms
- **Zone Analysis and Monitoring**: Display "Time In Zone" labels on objects to visualize how long objects have been in specific zones for occupancy monitoring, dwell time analysis, or compliance tracking
- **Spatial Information Display**: Show object dimensions (center coordinates, width, height) or area measurements directly on detections for spatial analysis, measurement workflows, or quality control
- **Professional Presentation and Reporting**: Create clean, informative visualizations with labeled detections for reports, dashboards, or presentations that combine visual results with textual information

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Bounding Box Visualization, Polygon Visualization, Dot Visualization) to combine text labels with geometric annotations for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images with labels for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with labels to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with labels as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with labels for live monitoring, tracking visualization, or post-processing analysis
"""

#: The font every supervision label draw call uses
#: (``supervision/annotators/core.py`` ``CV2_FONT``).
_CV2_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: ``supervision.annotators.utils.PENDING_TRACK_ID`` / ``PENDING_TRACK_COLOR``:
#: with ``ColorLookup.TRACK``, ``resolve_color`` (utils.py:139) returns the
#: pending gray for BOTH the background and the text color whenever the
#: resolved id is -1 — before it ever consults the palette or the configured
#: text color.
_PENDING_TRACK_ID = -1
_PENDING_TRACK_COLOR_BGR = (128, 128, 128)

#: LRU bound on cached label sprites. The steady-state label vocabulary is
#: tiny (class names, `class 0.87`-style strings), but `Dimensions`-like texts
#: change every frame — the bound turns a pathological stream into an LRU
#: churn (a cache miss costs one small CPU patch render + one small H2D)
#: instead of an unbounded device-memory leak.
_SPRITE_CACHE_MAX_ENTRIES = 512

#: Slots in the per-block pinned staging ring for the per-frame paste tables.
#: Must be >= 2x the pipeline batch size: on slot reuse the previous copy from
#: that slot was enqueued a whole batch (one model step) earlier, so its event
#: has fired long before the slot comes around again — the reuse-wait below is
#: a pathological-backpressure guard, never the steady state.
_TABLE_RING_DEPTH = 8

#: Initial per-slot slab capacity (int64 elements). Covers 32 labels per frame
#: (2 table entries per label); grown geometrically on capacity misses.
_TABLE_SLAB_MIN_CAPACITY = 64


def _staged_upload(
    host_values: np.ndarray,
    device: torch.device,
    pending: Optional[List[Tuple[torch.Tensor, "torch.cuda.Event"]]] = None,
) -> torch.Tensor:
    """One-shot host→device upload that never drains the device queue.

    A pageable ``torch.from_numpy(...).to(device)`` follows cudaMemcpy
    semantics: the driver first waits for every kernel already queued on the
    stream before the bytes move (measured 55-65 ms per first label upload of
    a batch on Jetson, queued behind the mask-composite kernels). Staging
    through pinned memory with ``copy_(non_blocking=True)`` enqueues the
    transfer asynchronously instead.

    On CUDA the pinned staging tensor plus a recorded event is appended to
    ``pending`` so the caller provably keeps the host memory intact until the
    copy has executed (entries may be dropped only once ``event.query()`` is
    True). Callers that pass no ``pending`` (the ring-less direct-call path of
    ``gpu_paste_label_sprites``) fall back on torch's caching host allocator,
    which event-guards a freed pinned block internally before reusing it —
    safe, just not explicit.

    On CPU devices there is nothing to pin and the copy is a plain memcpy,
    but the identical ``copy_(..., non_blocking=True)`` is still dispatched
    (a same-device ``.to`` would short-circuit without dispatching) so the op
    trace carries the flag on every platform — the viz-phase transfer audit
    asserts it on the CPU test host too.
    """
    source = torch.from_numpy(np.ascontiguousarray(host_values))
    if device.type == "cuda":
        source = source.pin_memory()
    staged = torch.empty(tuple(source.shape), dtype=source.dtype, device=device)
    staged.copy_(source, non_blocking=True)
    if device.type == "cuda" and pending is not None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        pending.append((source, event))
    return staged


class _TableRingSlot:
    __slots__ = ("host_slab", "host_view", "device_slab", "event", "capacity")

    def __init__(self):
        self.host_slab: Optional[torch.Tensor] = None
        self.host_view: Optional[np.ndarray] = None
        self.device_slab: Optional[torch.Tensor] = None
        self.event = None
        self.capacity = 0


class _PinnedSlabRing:
    """Reusable pinned staging ring for the per-frame packed paste tables.

    Every warm-path ``gpu_paste_label_sprites`` call uploads one small int64
    table (the per-label paste offsets + lengths). Uploading it as a pageable
    copy would synchronize the stream (see ``_staged_upload``); allocating and
    pinning a fresh buffer per frame would thrash the pinned allocator. This
    ring keeps ``depth`` pre-pinned host slabs, each paired with a same-size
    device slab and a reuse event. ``upload_int64``:

    1. takes the next slot round-robin;
    2. waits on the slot's recorded event ONLY when the copy issued on the
       slot's previous use has not executed yet (``event.query()`` False) —
       with ``depth`` >= 2x the pipeline batch size that copy ran during an
       earlier model step, so the wait is a pathological-backpressure guard,
       never the steady state;
    3. grows the slab geometrically on a capacity miss (rare realloc, then
       re-pin — safe because step 2 just proved no copy is in flight from the
       old slab);
    4. writes the table into the pinned host slab (a plain numpy write, no
       torch dispatch) and enqueues ONE full-slab ``copy_(non_blocking=True)``
       into the device slab, then records the slot's event behind it.

    The returned device slab is safe for consumers enqueued later on the same
    stream (stream order puts the copy before them); the recorded event only
    guards the HOST slab against being overwritten while its copy is pending.
    On CPU devices pinning and events are skipped entirely — plain tensors and
    the same single ``copy_`` (still flagged ``non_blocking=True``), so the
    code runs identically on the CPU test host.
    """

    __slots__ = ("_depth", "_min_capacity", "_slots", "_cursor", "_initialised")

    def __init__(
        self,
        depth: int = _TABLE_RING_DEPTH,
        min_capacity: int = _TABLE_SLAB_MIN_CAPACITY,
    ):
        self._depth = int(depth)
        self._min_capacity = int(min_capacity)
        self._slots = [_TableRingSlot() for _ in range(self._depth)]
        self._cursor = 0
        self._initialised = False

    def _ensure_slot(
        self, slot: _TableRingSlot, needed: int, device: torch.device
    ) -> None:
        if (
            slot.device_slab is not None
            and slot.capacity >= needed
            and slot.device_slab.device == device
        ):
            return
        capacity = max(self._min_capacity, slot.capacity)
        while capacity < needed:
            capacity *= 2
        pin = device.type == "cuda"
        slot.host_slab = torch.empty(capacity, dtype=torch.int64, pin_memory=pin)
        slot.host_view = slot.host_slab.numpy()
        slot.device_slab = torch.empty(capacity, dtype=torch.int64, device=device)
        slot.capacity = capacity
        slot.event = None

    def upload_int64(self, values: np.ndarray, device: torch.device) -> torch.Tensor:
        """Stage 1-D int64 ``values`` and return the slot's device slab (its
        first ``len(values)`` elements hold the table; callers slice it)."""
        slot = self._slots[self._cursor]
        self._cursor = (self._cursor + 1) % self._depth
        if slot.event is not None and not slot.event.query():
            # Pathological backpressure only (the device is more than a whole
            # ring — >= 2 batches — behind the host); steady state never waits.
            slot.event.synchronize()
        needed = int(values.shape[0])
        if not self._initialised:
            # First touch sizes EVERY slot so steady-state frames (any ring
            # phase) never dispatch an allocation.
            for other in self._slots:
                self._ensure_slot(other, needed, device)
            self._initialised = True
        else:
            self._ensure_slot(slot, needed, device)
        slot.host_view[:needed] = values
        slot.device_slab.copy_(slot.host_slab, non_blocking=True)
        if device.type == "cuda":
            if slot.event is None:
                slot.event = torch.cuda.Event()
            slot.event.record(torch.cuda.current_stream(device))
        return slot.device_slab


class _SceneDependentLabelError(ValueError):
    """Raised when a label patch cannot be pre-rendered scene-independently.

    ``sv.LabelAnnotator`` draws the anti-aliased text directly on the scene, so
    any text ink that escapes the opaque label background (descenders when
    ``text_padding`` is smaller than the font's baseline extent, glyphs leaking
    into a rounded-off corner when ``border_radius`` is large relative to the
    padding) is blended with the *scene* pixels underneath. A cached sprite
    cannot reproduce that blend, so such configurations are refused here and
    the block falls back to the bit-identical sv path instead of approximating.
    """


class _LabelMeasurement:
    """The size math of ``LabelAnnotator._get_label_properties`` (supervision
    0.29.0, ``annotators/core.py:1301``) for one label: sv's own ``wrap_text``
    (v1 exposes no ``max_line_length``, so ``None`` — newline splitting only),
    per-line ``cv2.getTextSize`` with ``CV2_FONT``, ``max_width`` /
    ``total_height`` aggregation and ``2 * text_padding`` padding on both
    axes. ``margin`` bounds how far any ink (descenders below the last
    baseline, stroke thickness, the 1-px AA fringe) can reach beyond the
    background box, so a canvas with a ``margin`` border provably captures
    every touched pixel."""

    __slots__ = ("lines", "width_padded", "height_padded", "margin")

    def __init__(self, lines, width_padded, height_padded, margin):
        self.lines = lines
        self.width_padded = width_padded
        self.height_padded = height_padded
        self.margin = margin


def _measure_label(
    label, text_scale: float, text_thickness: int, text_padding: int
) -> _LabelMeasurement:
    lines = wrap_text(label, None)
    line_heights: List[int] = []
    line_widths: List[int] = []
    baselines: List[int] = []
    for line in lines:
        (line_w, line_h), baseline = cv2.getTextSize(
            line, _CV2_FONT, text_scale, text_thickness
        )
        line_widths.append(line_w)
        line_heights.append(line_h)
        baselines.append(baseline)
    max_width = max(line_widths) if line_widths else 0
    total_height = sum(line_heights) + (len(line_heights) - 1) * text_padding
    width_padded = max_width + 2 * text_padding
    height_padded = total_height + 2 * text_padding
    tg_baseline = cv2.getTextSize("Tg", _CV2_FONT, text_scale, text_thickness)[1]
    margin = max(baselines + [tg_baseline]) + int(text_thickness) + 3
    return _LabelMeasurement(lines, width_padded, height_padded, margin)


class _LabelSprite:
    """A single label patch pre-rendered with sv's exact draw calls.

    The patch is rendered once on CPU (see ``_render_label_sprite``) and kept
    as a sparse pixel list: host-side ``(rows, cols)`` canvas coordinates of
    the opaque (scene-independent) pixels plus their RGB values as a device
    tensor. Per frame the only work is offsetting cached flat-index templates
    — no re-render, no per-frame H2D of pixel payloads.

    ``(offset_x, offset_y)`` is where the sv background-box origin sits inside
    the sprite canvas, so pasting the canvas origin at ``(box_x1 - offset_x,
    box_y1 - offset_y)`` reproduces sv's drawing at ``box_xyxy`` exactly. For
    an interior sprite the offsets both equal the measurement margin; for a
    frame-clipped variant the canvas edges coincide with the frame edges on
    the clipped sides (cv2's anti-aliased rasterisation treats strokes cut by
    the canvas border differently from uncut ones, so the border must sit
    exactly where sv's frame border sits).
    """

    __slots__ = (
        "offset_x",
        "offset_y",
        "rows",
        "cols",
        "row_min",
        "row_max",
        "col_min",
        "col_max",
        "colors_dev",
        "_flat_by_width",
        "_pending",
    )

    def __init__(
        self,
        offset_x: int,
        offset_y: int,
        rows: np.ndarray,
        cols: np.ndarray,
        colors_dev: torch.Tensor,
    ):
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.rows = rows  # (K,) int32, canvas-relative
        self.cols = cols  # (K,) int32, canvas-relative
        self.row_min = int(rows.min()) if rows.size else 0
        self.row_max = int(rows.max()) if rows.size else -1
        self.col_min = int(cols.min()) if cols.size else 0
        self.col_max = int(cols.max()) if cols.size else -1
        self.colors_dev = colors_dev  # (K, 3) uint8 RGB on the pipeline device
        self._flat_by_width: dict = {}
        # (pinned staging tensor, recorded event) pairs for this sprite's
        # in-flight non-blocking uploads: the pinned source provably outlives
        # the async copy — dropped only once the event has fired.
        self._pending: List = []

    @property
    def pixel_count(self) -> int:
        return int(self.rows.shape[0])

    def _release_completed_staging(self) -> None:
        """Drop pinned staging buffers whose uploads provably executed (their
        recorded events query True). Never blocks; events recorded on one
        stream fire in order, so popping from the front suffices."""
        while self._pending and self._pending[0][1].query():
            self._pending.pop(0)

    def flat_template(self, frame_width: int) -> torch.Tensor:
        """Device-resident ``rows * frame_width + cols`` template, cached per
        frame width (streams keep a constant width, so this is a one-time
        staged non-blocking upload — a pageable ``.to(device)`` here would
        drain the queued viz kernels). Callers must add the paste offset out
        of place — the cached tensor is shared across frames."""
        self._release_completed_staging()
        flat = self._flat_by_width.get(frame_width)
        if flat is None:
            flat_np = self.rows.astype(np.int64) * frame_width + self.cols.astype(
                np.int64
            )
            flat = _staged_upload(flat_np, self.colors_dev.device, self._pending)
            self._flat_by_width[frame_width] = flat
        return flat


def _draw_label_patch(
    canvas: np.ndarray,
    box_xyxy: Tuple[int, int, int, int],
    lines: List[str],
    text_color_bgr: Tuple[int, int, int],
    background_color_bgr: Tuple[int, int, int],
    text_scale: float,
    text_thickness: int,
    text_padding: int,
    border_radius: int,
) -> None:
    """The exact per-label draw sequence of ``sv.LabelAnnotator`` (supervision
    0.29.0, ``annotators/core.py``): ``draw_rounded_rectangle`` (line 1437 —
    two filled rectangles + four filled circles, radius clipped to
    ``min(width, height) // 2``) followed by the multiline ``cv2.putText``
    loop of ``_draw_labels`` (line 1355 — empty lines advance by the height of
    ``"Tg"`` without drawing). cv2 rasterisation is invariant under integer
    translation, so drawing at the canvas-local box and pasting equals drawing
    at the scene-global box."""
    x1, y1, x2, y2 = box_xyxy
    width = x2 - x1
    height = y2 - y1
    radius = min(border_radius, min(width, height) // 2)
    cv2.rectangle(
        canvas, (x1 + radius, y1), (x2 - radius, y2), background_color_bgr, -1
    )
    cv2.rectangle(
        canvas, (x1, y1 + radius), (x2, y2 - radius), background_color_bgr, -1
    )
    for center in (
        (x1 + radius, y1 + radius),
        (x2 - radius, y1 + radius),
        (x1 + radius, y2 - radius),
        (x2 - radius, y2 - radius),
    ):
        cv2.circle(canvas, center, radius, background_color_bgr, -1)
    current_y = y1 + text_padding
    for line in lines:
        # sv measures "Tg" (a string with ascender + descender) to advance
        # over empty lines, and the line itself otherwise.
        text_h = cv2.getTextSize(
            line if line else "Tg", _CV2_FONT, text_scale, text_thickness
        )[0][1]
        if line:
            cv2.putText(
                img=canvas,
                text=line,
                org=(x1 + text_padding, current_y + text_h),
                fontFace=_CV2_FONT,
                fontScale=text_scale,
                color=text_color_bgr,
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )
        current_y += text_h + text_padding


def _render_label_sprite(
    measurement: _LabelMeasurement,
    text_color_bgr: Tuple[int, int, int],
    background_color_bgr: Tuple[int, int, int],
    text_scale: float,
    text_thickness: int,
    text_padding: int,
    border_radius: int,
    device: torch.device,
    box_in_canvas: Tuple[int, int, int, int],
    canvas_hw: Tuple[int, int],
    frame_edge_sides: Tuple[bool, bool, bool, bool],
) -> _LabelSprite:
    """Render one label patch with sv's exact draw calls and extract its
    scene-independent pixels.

    The patch is drawn twice with identical calls — once over a black canvas,
    once over a white one. Pixels where the two renders agree are fully
    determined by the drawing (opaque); pixels where they differ depend on
    what is underneath: either never touched (transparent — sv leaves the
    scene) or touched by anti-aliased text ink *outside* the opaque background
    (sv blends that ink with the scene — not representable by a cached sprite,
    so ``_SceneDependentLabelError`` sends the block to the sv path).

    ``box_in_canvas`` places the sv background box inside the canvas. On sides
    marked in ``frame_edge_sides`` (left, top, right, bottom) the caller
    aligned the canvas edge with the *frame* edge, so cv2 clips exactly where
    sv would — cv2's AA rasterisation of a stroke cut by the canvas border
    differs from an uncut stroke, so an edge-crossing label must be rendered
    against the identical border. The remaining sides carry a ``margin``-wide
    border that provably contains all reachable ink; if ink still lands on
    such a border ring the sprite is refused rather than silently clipped
    where sv would have drawn.
    """
    canvas_h, canvas_w = canvas_hw
    on_black = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    on_white = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for canvas in (on_black, on_white):
        _draw_label_patch(
            canvas=canvas,
            box_xyxy=box_in_canvas,
            lines=measurement.lines,
            text_color_bgr=text_color_bgr,
            background_color_bgr=background_color_bgr,
            text_scale=text_scale,
            text_thickness=text_thickness,
            text_padding=text_padding,
            border_radius=border_radius,
        )
    opaque = (on_black == on_white).all(axis=2)
    touched = opaque | (on_black != 0).any(axis=2) | (on_white != 255).any(axis=2)
    left_edge, top_edge, right_edge, bottom_edge = frame_edge_sides
    if (
        (not top_edge and touched[0].any())
        or (not bottom_edge and touched[-1].any())
        or (not left_edge and touched[:, 0].any())
        or (not right_edge and touched[:, -1].any())
    ):
        raise _SceneDependentLabelError(
            "label ink reaches the sprite canvas border on a side that is not "
            "a frame edge"
        )
    if (touched & ~opaque).any():
        raise _SceneDependentLabelError(
            "anti-aliased label text escapes the opaque background patch "
            "(text_padding too small for the font extent, or border_radius "
            "cutting under the text); sv blends such ink with the scene, "
            "which a cached sprite cannot reproduce"
        )
    rows, cols = np.nonzero(opaque)
    colors_rgb = np.ascontiguousarray(on_black[rows, cols][:, ::-1])  # BGR -> RGB
    # Cache-miss pixel payload: staged through pinned memory + a non-blocking
    # copy (a pageable upload would synchronize the stream and drain queued
    # viz kernels — measured 55-62 ms per miss on Jetson under load); the
    # staging buffer is parked on the sprite until its event fires.
    pending: List = []
    colors_dev = _staged_upload(colors_rgb, device, pending)
    sprite = _LabelSprite(
        offset_x=int(box_in_canvas[0]),
        offset_y=int(box_in_canvas[1]),
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        colors_dev=colors_dev,
    )
    sprite._pending.extend(pending)
    return sprite


def gpu_paste_label_sprites(
    scene_chw: torch.Tensor,
    sprites: List[_LabelSprite],
    canvas_origins: List[Tuple[int, int]],
    table_ring: Optional[_PinnedSlabRing] = None,
) -> torch.Tensor:
    """Composite pre-rendered label sprites into a CHW RGB uint8 tensor, in
    place, with a device-op budget that is flat in the label count.

    A per-label paste loop is dispatch-bound on Jetson (the bbox painter in
    this package measured 43 ms @ 50 boxes for a per-box loop), so all labels
    land in ONE indexed store:

    1. Host: per label, the cached flat-index template (device-resident, see
       ``_LabelSprite.flat_template``) is selected and its scalar paste offset
       computed from the sprite's canvas origin — the cached color tensor is
       reused untouched, so a cache-hit frame does no per-pixel host work and
       no pixel-payload H2D at all.
    2. One packed upload of the per-label (offset, length) table, staged
       through ``table_ring`` (the block's pinned slab ring) as a single
       NON-BLOCKING copy — a pageable upload here follows cudaMemcpy
       semantics and drains every queued kernel first (measured 62-65 ms per
       first label of a batch on Jetson behind the mask compositor). Then one
       ``repeat_interleave`` expansion, one ``cat`` + offset add — flat pixel
       indices for every sprite pixel of every label. Ring-less direct calls
       stage through a one-shot pinned buffer instead
       (``_staged_upload``).
    3. sv draws labels sequentially, so a later label's patch overwrites an
       earlier one. When the pasted rectangles overlap, that order is
       reproduced deterministically with a ``scatter_reduce_(amax)`` of the
       global pixel position (ascending in label order) followed by a winner
       gather — the same later-wins resolution the bbox painter uses.
       Disjoint labels (the common case) skip it.
    4. One final indexed store. This is the ONLY write to ``scene_chw``: any
       failure before it leaves the scene untouched, so the block's sv
       fallback can never double-draw a partially annotated frame (and the
       paste is an opaque overwrite, hence idempotent, unlike a blend).

    No ``.contiguous()`` is taken: ``.view`` raises on a non-contiguous scene
    (routing to the sv fallback) rather than silently writing into a copy —
    ``copy_image=False`` must mutate the caller's storage.

    Every sprite pixel must land inside the frame at its origin (the caller
    picked an interior sprite whose pixel bounds fit, or a frame-clipped
    variant whose canvas lies inside the frame); a violation raises before
    anything is written.
    """
    device = scene_chw.device
    height, width = int(scene_chw.shape[1]), int(scene_chw.shape[2])
    flat_parts: List[torch.Tensor] = []
    color_parts: List[torch.Tensor] = []
    piece_offsets: List[int] = []
    piece_lengths: List[int] = []
    piece_rects: List[Tuple[int, int, int, int]] = []
    for sprite, (origin_x, origin_y) in zip(sprites, canvas_origins):
        if sprite.pixel_count == 0:
            continue
        if (
            origin_x + sprite.col_min < 0
            or origin_y + sprite.row_min < 0
            or origin_x + sprite.col_max >= width
            or origin_y + sprite.row_max >= height
        ):
            raise ValueError("label sprite pixels fall outside the frame")
        flat_parts.append(sprite.flat_template(width))
        color_parts.append(sprite.colors_dev)
        piece_offsets.append(origin_y * width + origin_x)
        piece_lengths.append(sprite.pixel_count)
        piece_rects.append(
            (
                origin_x + sprite.col_min,
                origin_y + sprite.row_min,
                origin_x + sprite.col_max + 1,
                origin_y + sprite.row_max + 1,
            )
        )
    if not flat_parts:
        return scene_chw
    pieces = len(flat_parts)
    total = int(sum(piece_lengths))
    packed_host = np.empty(2 * pieces, dtype=np.int64)
    packed_host[:pieces] = piece_offsets
    packed_host[pieces:] = piece_lengths
    if table_ring is not None:
        packed = table_ring.upload_int64(packed_host, device)
    else:
        packed = _staged_upload(packed_host, device)
    offsets_t, lengths_t = packed[:pieces], packed[pieces : 2 * pieces]
    piece_of_px = torch.repeat_interleave(
        torch.arange(pieces, device=device), lengths_t, output_size=total
    )
    flat = torch.cat(flat_parts) if pieces > 1 else flat_parts[0]
    flat = flat + offsets_t[piece_of_px]
    colors = torch.cat(color_parts) if pieces > 1 else color_parts[0]
    # Pairwise overlap test of the pasted (frame-clipped) canvas rectangles:
    # disjoint labels make every pixel its own winner, so owner resolution is
    # skipped. Conservative (canvas margins are mostly transparent), which
    # only costs the extra pass, never correctness.
    rect = np.asarray(piece_rects, dtype=np.int64)
    inter_x = np.maximum(rect[:, 0][:, None], rect[:, 0][None, :]) < np.minimum(
        rect[:, 2][:, None], rect[:, 2][None, :]
    )
    inter_y = np.maximum(rect[:, 1][:, None], rect[:, 1][None, :]) < np.minimum(
        rect[:, 3][:, None], rect[:, 3][None, :]
    )
    labels_overlap = int((inter_x & inter_y).sum()) > pieces  # diagonal always True
    if labels_overlap:
        # Later-label-wins ownership, provably in [0, total) for any
        # duplication pattern (see resolve_overlap_winners for why the
        # previous empty + include_self=False formulation was retired).
        order = torch.arange(total, device=device, dtype=torch.int32)
        winners = resolve_overlap_winners(
            flat, order, num_cells=height * width, num_candidates=total
        )
        colors = colors[winners]
    scene_chw.view(3, -1)[:, flat] = colors.t()
    return scene_chw


def _gpu_label_paste_eligible(
    detections, color_axis: str, image: WorkflowImageData
) -> bool:
    """True when the sprite compositor can replace the sv path."""
    if color_axis not in ("CLASS", "INDEX", "TRACK"):
        # Custom lookups keep the battle-tested sv path.
        return False
    if not image.is_tensor_materialised():
        # Numpy-sourced images must behave EXACTLY as before: forcing
        # tensor_image would be a costly host-side conversion and the sv path
        # is faster there.
        return False
    xyxy = getattr(detections, "xyxy", None)
    if not isinstance(xyxy, torch.Tensor) or int(xyxy.shape[0]) == 0:
        # Nothing to draw; the sv path is a trivial no-op.
        return False
    return True


def _resolve_color_ids_for_labels(
    detections: sv.Detections, color_axis: str
) -> np.ndarray:
    """The palette indices sv's ``resolve_color_idx`` (supervision 0.29.0,
    ``annotators/utils.py:40``) would use on this materialised view, raising
    its exact ``ValueError``s when ids are missing (the sv fallback would then
    raise the very same error from the annotator)."""
    n = len(detections)
    if color_axis == "INDEX":
        return np.arange(n)
    if color_axis == "CLASS":
        if detections.class_id is None:
            raise ValueError(
                "Could not resolve color by class because "
                "Detections do not have class_id. If using an annotator, "
                "try setting color_lookup to sv.ColorLookup.INDEX or "
                "sv.ColorLookup.TRACK."
            )
        return detections.class_id.astype(int)
    if detections.tracker_id is None:
        raise ValueError(
            "Could not resolve color by track because "
            "Detections do not have tracker_id. Did you call "
            "tracker.update_with_detections(...) before annotating?"
        )
    return detections.tracker_id.astype(int)


class LabelManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "LabelVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Label Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-tag",
                "blockPriority": 2,
                "popular": True,
                "supervision": True,
                "warnings": [
                    {
                        "property": "copy_image",
                        "value": False,
                        "message": "This setting will mutate its input image. If the input is used by other blocks, it may cause unexpected behavior.",
                    }
                ],
            },
        }
    )

    text: Union[
        Literal[
            "Class",
            "Confidence",
            "Class and Confidence",
            "Index",
            "Dimensions",
            "Area",
            "Area (mask)",
            "Area (converted)",
            "Tracker Id",
            "Time In Zone",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="Class",
        description="Content to display in text labels. Options: 'Class' (class name), 'Confidence' (confidence score), 'Class and Confidence' (both), 'Tracker Id' (tracking ID for tracked objects), 'Time In Zone' (time spent in zone), 'Dimensions' (center coordinates and width x height), 'Area' (bounding box area in pixels), 'Area (mask)' (mask area in pixels from Mask Area Measurement block), 'Area (converted)' (mask area in converted units from Mask Area Measurement block), or 'Index' (detection index).",
        examples=["LABEL", "$inputs.text"],
        json_schema_extra={
            "always_visible": True,
        },
    )

    text_position: Union[
        Literal[
            "CENTER",
            "CENTER_LEFT",
            "CENTER_RIGHT",
            "TOP_CENTER",
            "TOP_LEFT",
            "TOP_RIGHT",
            "BOTTOM_LEFT",
            "BOTTOM_CENTER",
            "BOTTOM_RIGHT",
            "CENTER_OF_MASS",
        ],
        Selector(kind=[STRING_KIND]),
    ] = Field(  # type: ignore
        default="TOP_LEFT",
        description="Anchor position for placing labels relative to each detection's bounding box. Options include: CENTER (center of box), corners (TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT), edge midpoints (TOP_CENTER, CENTER_LEFT, CENTER_RIGHT, BOTTOM_CENTER), or CENTER_OF_MASS (center of mass of the object).",
        examples=["CENTER", "$inputs.text_position"],
    )

    text_color: Union[str, Selector(kind=[STRING_KIND])] = Field(  # type: ignore
        description="Color of the label text. Can be a color name (e.g., 'WHITE', 'BLACK') or color code in HEX format (e.g., '#FFFFFF') or RGB format (e.g., 'rgb(255, 255, 255)').",
        default="WHITE",
        examples=["WHITE", "#FFFFFF", "rgb(255, 255, 255)" "$inputs.text_color"],
    )

    text_scale: Union[float, Selector(kind=[FLOAT_KIND])] = Field(  # type: ignore
        description="Scale factor for text size. Higher values create larger text. Default is 1.0.",
        default=1.0,
        examples=[1.0, "$inputs.text_scale"],
    )

    text_thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of text characters in pixels. Higher values create bolder, thicker text for better visibility.",
        default=1,
        examples=[1, "$inputs.text_thickness"],
    )

    text_padding: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Padding around the text in pixels. Controls the spacing between the text and the label background border.",
        default=10,
        examples=[10, "$inputs.text_padding"],
    )

    border_radius: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Border radius of the label background in pixels. Set to 0 for square corners. Higher values create more rounded corners for a softer appearance.",
        default=0,
        examples=[0, "$inputs.border_radius"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class LabelVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}
        self._sprite_cache: "OrderedDict" = OrderedDict()
        # Pinned staging ring for the per-frame packed paste tables — the
        # steady-state path allocates/pins nothing per frame and its single
        # upload never blocks the stream (see _PinnedSlabRing).
        self._table_ring = _PinnedSlabRing()
        # One-shot latch so a permanently broken GPU fast path is visible in
        # production logs (WARNING) without emitting one record per frame.
        # Deliberately unsynchronised: a benign race can only cost a duplicate
        # warning, which is cheaper than a lock on the annotate path.
        self._gpu_fallback_warned = False

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return LabelManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        text_position: str,
        text_color: str,
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        border_radius: int,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    text_position,
                    text_color,
                    text_scale,
                    text_thickness,
                    text_padding,
                    border_radius,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            text_color = str_to_color(text_color)

            self.annotatorCache[key] = sv.LabelAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                text_position=getattr(sv.Position, text_position),
                text_color=text_color,
                text_scale=text_scale,
                text_thickness=text_thickness,
                text_padding=text_padding,
                border_radius=border_radius,
            )

        return self.annotatorCache[key]

    def _get_sprite(
        self,
        label,
        measurement: _LabelMeasurement,
        text_color_bgr: Tuple[int, int, int],
        background_color_bgr: Tuple[int, int, int],
        text_scale: float,
        text_thickness: int,
        text_padding: int,
        border_radius: int,
        device: torch.device,
        box_in_canvas: Tuple[int, int, int, int],
        canvas_hw: Tuple[int, int],
        frame_edge_sides: Tuple[bool, bool, bool, bool],
    ) -> _LabelSprite:
        """LRU-cached sprite lookup. The key covers everything that affects
        the patch's pixels: the label + typography + colors, plus the canvas
        geometry (interior sprites share one canonical geometry; frame-clipped
        variants are keyed by their clip window, which changes only when the
        label's overhang does). The raw label object is used as the key
        component — ``wrap_text`` special-cases falsy labels before
        stringifying, so pre-stringifying here could change sv's rendering;
        str-like labels hash consistently across ``str``/``np.str_``. In
        practice the label vocabulary is tiny (class names x rounded
        confidences), so steady-state hit rate is ~100% and no H2D happens per
        frame."""
        key = (
            label,
            text_color_bgr,
            background_color_bgr,
            float(text_scale),
            int(text_thickness),
            int(text_padding),
            int(border_radius),
            str(device),
            box_in_canvas,
            canvas_hw,
            frame_edge_sides,
        )
        sprite = self._sprite_cache.get(key)
        if sprite is not None:
            self._sprite_cache.move_to_end(key)
            return sprite
        sprite = _render_label_sprite(
            measurement=measurement,
            text_color_bgr=text_color_bgr,
            background_color_bgr=background_color_bgr,
            text_scale=float(text_scale),
            text_thickness=int(text_thickness),
            text_padding=int(text_padding),
            border_radius=int(border_radius),
            device=device,
            box_in_canvas=box_in_canvas,
            canvas_hw=canvas_hw,
            frame_edge_sides=frame_edge_sides,
        )
        self._sprite_cache[key] = sprite
        if len(self._sprite_cache) > _SPRITE_CACHE_MAX_ENTRIES:
            self._sprite_cache.popitem(last=False)
        return sprite

    def run(
        self,
        image: WorkflowImageData,
        predictions: Union[TensorNativePrediction, TensorNativeDetections],
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        text: Optional[str],
        text_position: Optional[str],
        text_color: Optional[str],
        text_scale: Optional[float],
        text_thickness: Optional[int],
        text_padding: Optional[int],
        border_radius: Optional[int],
    ) -> BlockResult:
        detections = (
            split_key_point_prediction(predictions)[1]
            if isinstance(predictions, tuple)
            else predictions
        )
        passthrough = empty_predictions_passthrough(
            image=image, detections=detections, copy_image=copy_image
        )
        if passthrough is not None:
            return passthrough
        # The Label annotator reads `.mask` for exactly two configurations, and
        # only for instance-segmentation input (there is no mask to materialise
        # otherwise, so `materialise_masks=True` is a no-op for OD input):
        #   * text == "Area": `sv.Detections.area` returns MASK area when a mask
        #     is present and BOX area when it is None — flag-off shows mask area
        #     on IS input, so the mask must be materialised to match.
        #   * text_position == "CENTER_OF_MASS": `sv.LabelAnnotator` anchors on
        #     the mask centroid; `get_anchors_coordinates` RAISES without a mask.
        # Every other label reads xyxy / confidence / per-box metadata, so the
        # device->host dense-mask copy is skipped for them. Both configurations
        # also keep the sv path below (the sprite compositor never materialises
        # masks).
        needs_masks = text == "Area" or text_position == "CENTER_OF_MASS"
        if not needs_masks and _gpu_label_paste_eligible(detections, color_axis, image):
            # GPU sprite path: labels are rendered once on CPU with sv's exact
            # draw calls, cached as device sprites, and pasted with a fixed
            # number of torch ops — the full-frame device->host materialisation
            # the sv path pays (~30 ms/2K frame on Orin NX) never happens.
            try:
                palette = self.getPalette(color_palette, palette_size, custom_colors)
                if not isinstance(palette, sv.ColorPalette):
                    raise TypeError("expected sv.ColorPalette")
                scene_t = image.tensor_image
                if int(scene_t.shape[0]) != 3:
                    raise ValueError("GPU label compositor requires a 3-channel image")
                # Mask-free sv view: one tiny xyxy/class/confidence D2H plus
                # per-box metadata — labels never read masks on this path.
                sv_view = to_supervision_for_annotation(
                    predictions, materialise_masks=False
                )
                labels = build_detection_labels(sv_view, text)
                if len(labels) != len(sv_view):
                    # sv's _validate_labels contract.
                    raise ValueError(
                        f"The number of labels ({len(labels)}) does not match "
                        f"the number of detections ({len(sv_view)}). Each "
                        "detection should have exactly 1 label."
                    )
                text_color_bgr = str_to_color(text_color).as_bgr()
                ids = _resolve_color_ids_for_labels(sv_view, color_axis)
                # Same anchor math as LabelAnnotator._get_label_properties
                # (supervision 0.29.0, annotators/core.py:1301): sv's own
                # get_anchors_coordinates on the same materialised view,
                # truncated to int; the label box comes from sv's
                # resolve_text_background_xyxy with the float32 round-trip the
                # annotator applies to it.
                anchors = sv_view.get_anchors_coordinates(
                    anchor=getattr(sv.Position, text_position)
                ).astype(int)
                frame_h = int(scene_t.shape[1])
                frame_w = int(scene_t.shape[2])
                sprites: List[_LabelSprite] = []
                canvas_origins: List[Tuple[int, int]] = []
                for idx, label in enumerate(labels):
                    if color_axis == "TRACK" and int(ids[idx]) == _PENDING_TRACK_ID:
                        # sv resolve_color returns the pending-track gray for
                        # BOTH background and text color (utils.py:139).
                        background_bgr = _PENDING_TRACK_COLOR_BGR
                        line_color_bgr = _PENDING_TRACK_COLOR_BGR
                    else:
                        background_bgr = palette.by_idx(int(ids[idx])).as_bgr()
                        line_color_bgr = text_color_bgr
                    measurement = _measure_label(
                        label,
                        float(text_scale),
                        int(text_thickness),
                        int(text_padding),
                    )
                    background_xyxy = resolve_text_background_xyxy(
                        center_coordinates=(int(anchors[idx][0]), int(anchors[idx][1])),
                        text_wh=(
                            measurement.width_padded,
                            measurement.height_padded,
                        ),
                        position=getattr(sv.Position, text_position),
                    )
                    background_xyxy = np.asarray(
                        background_xyxy, dtype=np.float32
                    ).astype(int)
                    bx1, by1, bx2, by2 = (int(value) for value in background_xyxy)
                    margin = measurement.margin
                    sprite = self._get_sprite(
                        label=label,
                        measurement=measurement,
                        text_color_bgr=line_color_bgr,
                        background_color_bgr=background_bgr,
                        text_scale=text_scale,
                        text_thickness=text_thickness,
                        text_padding=text_padding,
                        border_radius=border_radius,
                        device=scene_t.device,
                        box_in_canvas=(
                            margin,
                            margin,
                            margin + measurement.width_padded,
                            margin + measurement.height_padded,
                        ),
                        canvas_hw=(
                            measurement.height_padded + 1 + 2 * margin,
                            measurement.width_padded + 1 + 2 * margin,
                        ),
                        frame_edge_sides=(False, False, False, False),
                    )
                    origin_x = bx1 - sprite.offset_x
                    origin_y = by1 - sprite.offset_y
                    if not (
                        origin_x + sprite.col_min >= 0
                        and origin_y + sprite.row_min >= 0
                        and origin_x + sprite.col_max < frame_w
                        and origin_y + sprite.row_max < frame_h
                    ):
                        # The label crosses the frame boundary: cv2 clips at
                        # the frame edge and its AA rasterisation of a clipped
                        # stroke differs from an unclipped one, so a variant
                        # is rendered in a canvas whose edges coincide with
                        # the frame edges on the crossing sides (cached per
                        # clip window).
                        window_x1 = max(bx1 - margin, 0)
                        window_y1 = max(by1 - margin, 0)
                        window_x2 = min(bx2 + 1 + margin, frame_w)
                        window_y2 = min(by2 + 1 + margin, frame_h)
                        if window_x2 <= window_x1 or window_y2 <= window_y1:
                            # Entirely outside the frame: sv draws nothing
                            # visible.
                            continue
                        sprite = self._get_sprite(
                            label=label,
                            measurement=measurement,
                            text_color_bgr=line_color_bgr,
                            background_color_bgr=background_bgr,
                            text_scale=text_scale,
                            text_thickness=text_thickness,
                            text_padding=text_padding,
                            border_radius=border_radius,
                            device=scene_t.device,
                            box_in_canvas=(
                                bx1 - window_x1,
                                by1 - window_y1,
                                bx2 - window_x1,
                                by2 - window_y1,
                            ),
                            canvas_hw=(
                                window_y2 - window_y1,
                                window_x2 - window_x1,
                            ),
                            frame_edge_sides=(
                                bx1 - margin < 0,
                                by1 - margin < 0,
                                bx2 + 1 + margin > frame_w,
                                by2 + 1 + margin > frame_h,
                            ),
                        )
                        origin_x = bx1 - sprite.offset_x
                        origin_y = by1 - sprite.offset_y
                    sprites.append(sprite)
                    canvas_origins.append((origin_x, origin_y))
                # All validation and sprite rendering succeeded — only now may
                # the scene be touched (clone, or the single in-place store
                # inside the paste).
                if copy_image:
                    scene_t = scene_t.clone()
                annotated_tensor = gpu_paste_label_sprites(
                    scene_t, sprites, canvas_origins, table_ring=self._table_ring
                )
                if not copy_image:
                    # The paste mutated `image.tensor_image` storage in place
                    # (the sv-path contract for copy_image=False); invalidate
                    # the derived numpy/base64 caches.
                    image.declare_tensor_image_mutated()
                return {
                    OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                        origin_image_data=image, tensor_image=annotated_tensor
                    )
                }
            except Exception as gpu_error:
                if not self._gpu_fallback_warned:
                    self._gpu_fallback_warned = True
                    logger.warning(
                        "Label Visualization: GPU label compositor failed "
                        "(%s); falling back to the slower sv.LabelAnnotator "
                        "path (this materialises the frame on the host, paying "
                        "a device-to-host transfer per frame). Only the first "
                        "occurrence is logged at warning level; subsequent "
                        "fallbacks are logged at debug level.",
                        gpu_error,
                    )
                else:
                    logger.debug(
                        "GPU label compositor failed (%s); falling back to "
                        "sv.LabelAnnotator path.",
                        gpu_error,
                    )
        predictions = to_supervision_for_annotation(
            predictions, materialise_masks=needs_masks
        )
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            text_position,
            text_color,
            text_scale,
            text_thickness,
            text_padding,
            border_radius,
        )
        labels = build_detection_labels(predictions, text)
        scene = image.numpy_image
        if copy_image:
            scene = scene.copy()
        else:
            image.declare_numpy_image_mutated()
        annotated_image = annotator.annotate(
            scene=scene,
            detections=predictions,
            labels=labels,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
