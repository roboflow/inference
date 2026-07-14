from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field

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
    to_supervision_for_annotation,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    INTEGER_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TYPE: str = "roboflow_core/bounding_box_visualization@v1"
SHORT_DESCRIPTION = "Draw a box around detected objects in an image."
LONG_DESCRIPTION = """
Draw bounding boxes around detected objects in an image, with customizable colors, thickness, and corner roundness.

## How This Block Works

This block takes an image and detection predictions (from object detection, instance segmentation, or keypoint detection models) and draws rectangular bounding boxes around each detected object. The block:

1. Takes an image and predictions as input
2. Applies color styling based on the selected color palette, with colors assigned by class, index, or track ID
3. Draws bounding boxes using Supervision's BoxAnnotator (for square corners) or RoundBoxAnnotator (for rounded corners) based on the roundness setting
4. Applies the specified box thickness to control the line width of the bounding boxes
5. Returns an annotated image with bounding boxes overlaid on the original image

The block supports various color palettes (default, Roboflow, Matplotlib palettes, or custom colors) and can color boxes based on detection class, index, or tracker ID. When roundness is set to 0, square corners are used; when roundness is greater than 0, rounded corners are applied for a softer visual appearance. You can choose whether to modify the original image or create a copy for visualization, which is useful when stacking multiple visualization blocks.

## Common Use Cases

- **Model Validation and Debugging**: Visualize detection results to verify model performance, check bounding box accuracy, identify false positives or false negatives, and debug model outputs
- **Results Presentation**: Create annotated images for reports, dashboards, or presentations showing what objects were detected in images or video frames
- **Quality Control**: Overlay bounding boxes on production line images to visualize detected defects, products, or components for quality assurance workflows
- **Monitoring and Alerting**: Generate visual outputs showing detected objects for security monitoring, surveillance systems, or compliance tracking with annotated evidence
- **Training Data Review**: Review and validate training datasets by visualizing annotations and bounding boxes to ensure labeling accuracy and consistency
- **Interactive Applications**: Create user interfaces that display real-time detection results with bounding boxes for object tracking, counting, or identification applications

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Polygon Visualization, Mask Visualization) to stack multiple annotations on the same image for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save annotated images for documentation, archiving, or training data preparation
- **Webhook blocks** to send visualized results to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with bounding boxes for live monitoring or post-processing analysis
"""


class _BorderGeometry:
    """cv2-derived rasterisation template for one (thickness, corner radius).

    ``radius is None`` reproduces ``sv.BoxAnnotator`` (one ``cv2.rectangle``
    per box); an integer radius reproduces ``sv.RoundBoxAnnotator`` (four
    quarter ``cv2.ellipse`` arcs + four ``cv2.line`` edges per box). Neither
    is a simple outer-minus-inner frame: thick rectangles get round joins,
    thick lines get round caps, and even thicknesses paint ``thickness + 1``
    wide bands. Instead of re-deriving OpenCV's thick-line rasteriser, one
    reference box is drawn with the exact same cv2 primitive sequence and
    decomposed into four uniform straight bands plus four corner patches.
    Reconstruction is asserted pixel-perfect against cv2 on two reference
    sizes, so the GPU painter is bit-exact by construction for every box
    large enough to use the template (smaller boxes take the per-box cv2
    raster path).
    """

    def __init__(self, thickness: int, radius: Optional[int] = None):
        self.thickness = thickness
        self.radius = radius
        margin = 4 * thickness + 8
        # Upper-bound the corner span before o/i are measured so the first
        # raster is already big enough to host two corner patches + a body.
        span_bound = (0 if radius is None else radius) + 2 * (2 * thickness + 4) + 2
        reference_side = max(8 * thickness + 32, 2 * span_bound + 16)
        second_side = reference_side + 9  # catches any size-dependence
        ref = self._rasterise(margin, reference_side)
        ys, _ = np.nonzero(ref)
        self.outer = margin - int(ys.min())
        mid_column = np.nonzero(ref[:, margin + reference_side // 2])[0]
        self.inner = (
            int(mid_column[mid_column < margin + reference_side // 2].max()) - margin
        )
        self.corner = (0 if radius is None else radius) + 2 * (
            self.outer + self.inner
        ) + 2
        # Smallest box side (x2 - x1 + 1) the band+corner decomposition tiles.
        self.min_side = 2 * self.corner - 2 * self.outer
        c, o = self.corner, self.outer
        far = margin + reference_side + o  # outermost painted row/col
        self.corner_masks = np.stack(
            [
                ref[margin - o : margin - o + c, margin - o : margin - o + c],
                ref[margin - o : margin - o + c, far - c + 1 : far + 1],
                ref[far - c + 1 : far + 1, margin - o : margin - o + c],
                ref[far - c + 1 : far + 1, far - c + 1 : far + 1],
            ]
        )  # (4, C, C) bool: TL, TR, BL, BR
        # Local True coords per corner patch, for host-side anchor+offset
        # expansion (numpy) straight into the packed upload.
        self.corner_offsets = [
            (rows.astype(np.int64), cols.astype(np.int64))
            for rows, cols in (np.nonzero(mask) for mask in self.corner_masks)
        ]
        for side in (reference_side, second_side):
            self._assert_reconstruction(margin, side)

    def _draw(self, canvas: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """The exact cv2 primitive sequence the sv annotator issues per box."""
        if self.radius is None:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), 255, self.thickness)
            return
        radius = self.radius
        circle_centers = [
            (x1 + radius, y1 + radius),
            (x2 - radius, y1 + radius),
            (x2 - radius, y2 - radius),
            (x1 + radius, y2 - radius),
        ]
        lines = [
            ((x1 + radius, y1), (x2 - radius, y1)),
            ((x2, y1 + radius), (x2, y2 - radius)),
            ((x1 + radius, y2), (x2 - radius, y2)),
            ((x1, y1 + radius), (x1, y2 - radius)),
        ]
        start_angles = (180, 270, 0, 90)
        end_angles = (270, 360, 90, 180)
        for center, line, start_angle, end_angle in zip(
            circle_centers, lines, start_angles, end_angles
        ):
            cv2.ellipse(
                img=canvas,
                center=center,
                axes=(radius, radius),
                angle=0,
                startAngle=start_angle,
                endAngle=end_angle,
                color=255,
                thickness=self.thickness,
            )
            cv2.line(
                img=canvas,
                pt1=line[0],
                pt2=line[1],
                color=255,
                thickness=self.thickness,
            )

    def _rasterise(self, margin: int, side: int) -> np.ndarray:
        size = side + 2 * margin + 1
        canvas = np.zeros((size, size), np.uint8)
        self._draw(canvas, margin, margin, margin + side, margin + side)
        return canvas > 0

    def rasterise_box_border(
        self, x1: int, y1: int, x2: int, y2: int, height: int, width: int
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        """Exact border mask for one box that skips the template path, as
        ``(mask, row0, col0)`` anchored in frame coords (``None`` when the box
        cannot touch the frame).

        The canvas is the frame-intersected border region, NOT a padded
        crop: cv2 rasterises CLIPPED thick primitives differently from a
        cropped unclipped draw (a round line cap relocated by ``clipLine``
        shifts boundary pixels), so the canvas edges must coincide with the
        frame edges on every side the box crosses. On non-crossing sides the
        canvas keeps ``outer`` slack, beyond any painted pixel."""
        o = self.outer
        row0, col0 = max(y1 - o, 0), max(x1 - o, 0)
        row1, col1 = min(y2 + o, height - 1), min(x2 + o, width - 1)
        if row1 < row0 or col1 < col0:
            return None
        canvas = np.zeros((row1 - row0 + 1, col1 - col0 + 1), np.uint8)
        self._draw(canvas, x1 - col0, y1 - row0, x2 - col0, y2 - row0)
        return canvas > 0, row0, col0

    def box_regions(
        self, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int]]]:
        """Band rects (inclusive r1, r2, c1, c2) + corner anchors (r, c) for a
        normalized box; only valid when both sides are >= ``min_side``."""
        o, i, c = self.outer, self.inner, self.corner
        ox1, oy1, ox2, oy2 = x1 - o, y1 - o, x2 + o, y2 + o
        bands = [
            (oy1, y1 + i, ox1 + c, ox2 - c),  # top
            (y2 - i, oy2, ox1 + c, ox2 - c),  # bottom
            (oy1 + c, oy2 - c, ox1, x1 + i),  # left
            (oy1 + c, oy2 - c, x2 - i, ox2),  # right
        ]
        corners = [
            (oy1, ox1),
            (oy1, ox2 - c + 1),
            (oy2 - c + 1, ox1),
            (oy2 - c + 1, ox2 - c + 1),
        ]
        return bands, corners

    def _assert_reconstruction(self, margin: int, side: int) -> None:
        ref = self._rasterise(margin, side)
        rebuilt = np.zeros_like(ref)
        bands, corners = self.box_regions(
            margin, margin, margin + side, margin + side
        )
        for r1, r2, c1, c2 in bands:
            rebuilt[r1 : r2 + 1, c1 : c2 + 1] = True
        for idx, (r, c) in enumerate(corners):
            rebuilt[r : r + self.corner, c : c + self.corner] |= self.corner_masks[idx]
        if not np.array_equal(rebuilt, ref):
            raise ValueError(
                f"cv2 band+corner decomposition failed for thickness="
                f"{self.thickness}, radius={self.radius} (unexpected OpenCV "
                f"rasteriser geometry)"
            )


_GEOMETRY_CACHE: "OrderedDict[Tuple[int, Optional[int]], _BorderGeometry]" = (
    OrderedDict()
)
_GEOMETRY_CACHE_LIMIT = 128  # rounded radii are data-dependent; bound the cache
_EMPTY_I64 = np.zeros(0, dtype=np.int64)


def _get_geometry(thickness: int, radius: Optional[int] = None) -> _BorderGeometry:
    key = (thickness, radius)
    geometry = _GEOMETRY_CACHE.get(key)
    if geometry is None:
        geometry = _BorderGeometry(thickness, radius)
        _GEOMETRY_CACHE[key] = geometry
        if len(_GEOMETRY_CACHE) > _GEOMETRY_CACHE_LIMIT:
            _GEOMETRY_CACHE.popitem(last=False)
    else:
        _GEOMETRY_CACHE.move_to_end(key)
    return geometry


def gpu_draw_boxes(
    scene_chw: torch.Tensor,
    xyxy: np.ndarray,
    colors_rgb: np.ndarray,
    thickness: int,
    roundness: float = 0.0,
) -> torch.Tensor:
    """GPU-native torch replacement for ``sv.BoxAnnotator.annotate``
    (``roundness == 0``) and ``sv.RoundBoxAnnotator.annotate``
    (``roundness > 0``) on a CHW RGB uint8 device tensor (the
    ``WorkflowImageData.tensor_image`` contract). The scene tensor is mutated
    in place.

    A per-box torch paint loop is hopeless on Jetson — python-op/kernel-launch
    overhead dominates the tiny border writes — so the whole frame is painted
    with a FIXED number of torch ops regardless of box count:

    1. Host (numpy, all boxes at once): straight-band rectangles + corner
       patch pixel coords from the cv2-derived geometry templates (see
       ``_BorderGeometry``; rounded boxes group by their sv corner radius
       ``int(min_side // 2 * roundness)``). Boxes too small for the template
       decomposition are rasterised exactly by the same cv2 primitives on
       tiny canvases and contribute their pixel coords directly.
    2. One packed H2D upload of every host-built array (per-transfer fixed
       cost is brutal on some Jetsons: ~0.36 ms on AGX Orin vs ~0.06 ms on
       Orin Nano), then device-side ragged expansion of band rects to flat
       pixel indices (``repeat_interleave`` with host-computed
       ``output_size`` — no device sync anywhere).
    3. Overlap resolution: sv paints boxes sequentially so the HIGHEST
       detection index wins every contested pixel — reproduced exactly by one
       deterministic ``scatter_reduce_(amax)`` of box indices over the frame,
       then a single indexed color write. Duplicate indices all write the
       resolved winner's color, so the write is deterministic too.
    """
    geometry_square = _get_geometry(thickness) if roundness == 0 else None
    device = scene_chw.device
    height, width = int(scene_chw.shape[1]), int(scene_chw.shape[2])
    if height * width >= 2**31:
        raise ValueError("frame too large for int32 pixel indexing")
    n = int(xyxy.shape[0])

    xy = np.asarray(xyxy, dtype=np.int64)
    # cv2.rectangle normalizes swapped corners; detections are normalized
    # upstream, so this is only defensive for the rounded path.
    x1 = np.minimum(xy[:, 0], xy[:, 2])
    x2 = np.maximum(xy[:, 0], xy[:, 2])
    y1 = np.minimum(xy[:, 1], xy[:, 3])
    y2 = np.maximum(xy[:, 1], xy[:, 3])

    if roundness == 0:
        box_geometry = np.zeros(n, dtype=np.int64)  # one group, square
        groups: List[Tuple[_BorderGeometry, np.ndarray]] = [
            (geometry_square, np.arange(n))
        ]
    else:
        # sv.RoundBoxAnnotator: radius = int(min_side // 2 * roundness), the
        # smaller side chosen by strict width < height comparison.
        box_w, box_h = x2 - x1, y2 - y1
        radii = np.trunc(
            np.where(box_w < box_h, box_w // 2, box_h // 2) * roundness
        ).astype(np.int64)
        groups = [
            (_get_geometry(thickness, int(radius)), np.nonzero(radii == radius)[0])
            for radius in np.unique(radii)
        ]

    # ---- host-side prep (numpy): bands + corner coords + small boxes -----
    band_r1: List[np.ndarray] = []
    band_r2: List[np.ndarray] = []
    band_c1: List[np.ndarray] = []
    band_c2: List[np.ndarray] = []
    band_box: List[np.ndarray] = []
    coord_flat: List[np.ndarray] = []  # pre-resolved pixel coords (corners, small)
    coord_box: List[np.ndarray] = []

    def _append_patch_coords(
        rows: np.ndarray, cols: np.ndarray, box_index: int
    ) -> None:
        keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        coord_flat.append(rows[keep] * width + cols[keep])
        coord_box.append(np.full(int(keep.sum()), box_index, dtype=np.int64))

    # Expanded per-box paint bounds, for the host-side overlap test below.
    exp_x1 = np.empty(n, dtype=np.int64)
    exp_x2 = np.empty(n, dtype=np.int64)
    exp_y1 = np.empty(n, dtype=np.int64)
    exp_y2 = np.empty(n, dtype=np.int64)

    for geometry, indices in groups:
        o, inner, c = geometry.outer, geometry.inner, geometry.corner
        gx1, gx2, gy1, gy2 = x1[indices], x2[indices], y1[indices], y2[indices]
        exp_x1[indices], exp_x2[indices] = gx1 - o, gx2 + o
        exp_y1[indices], exp_y2[indices] = gy1 - o, gy2 + o
        big = ((gx2 - gx1 + 1) >= geometry.min_side) & (
            (gy2 - gy1 + 1) >= geometry.min_side
        )
        if geometry.radius is not None:
            # cv2 rasterises CLIPPED thick lines/arcs differently from a
            # cropped unclipped draw (round caps relocate), so rounded boxes
            # whose border region crosses the frame edge take the exact
            # frame-clipped raster path. cv2.rectangle is crop/clip-identical
            # (fuzz-verified), so square boxes keep the template everywhere.
            big &= (
                (gx1 - o >= 0)
                & (gy1 - o >= 0)
                & (gx2 + o <= width - 1)
                & (gy2 + o <= height - 1)
            )
        big_pos = np.nonzero(big)[0]
        if len(big_pos):
            bx1, bx2 = gx1[big_pos], gx2[big_pos]
            by1, by2 = gy1[big_pos], gy2[big_pos]
            big_idx = indices[big_pos]
            ox1, oy1, ox2, oy2 = bx1 - o, by1 - o, bx2 + o, by2 + o
            # Band rects (inclusive r1, r2, c1, c2) + owning detection index.
            band_r1.append(np.concatenate([oy1, by2 - inner, oy1 + c, oy1 + c]))
            band_r2.append(np.concatenate([by1 + inner, oy2, oy2 - c, oy2 - c]))
            band_c1.append(np.concatenate([ox1 + c, ox1 + c, ox1, bx2 - inner]))
            band_c2.append(np.concatenate([ox2 - c, ox2 - c, bx1 + inner, ox2]))
            band_box.append(np.tile(big_idx, 4))
            # Corner patches: anchor + cached local offsets, vectorized over
            # the group's boxes (host-side; ships in the packed upload).
            anchors = (
                (oy1, ox1),
                (oy1, ox2 - c + 1),
                (oy2 - c + 1, ox1),
                (oy2 - c + 1, ox2 - c + 1),
            )
            for slot, (anchor_r, anchor_c) in enumerate(anchors):
                off_r, off_c = geometry.corner_offsets[slot]
                rows = (anchor_r[:, None] + off_r[None, :]).ravel()
                cols = (anchor_c[:, None] + off_c[None, :]).ravel()
                keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                coord_flat.append(rows[keep] * width + cols[keep])
                coord_box.append(np.repeat(big_idx, len(off_r))[keep])
        for pos in np.nonzero(~big)[0]:
            # Corner geometry overlaps below min_side (joins/caps/arcs
            # collide), or a rounded box crosses the frame edge: rasterise
            # the exact border with the same cv2 primitives on a small
            # frame-clipped host canvas.
            idx = int(indices[pos])
            rastered = geometry.rasterise_box_border(
                int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx]),
                height, width,
            )
            if rastered is None:
                continue
            border, row0, col0 = rastered
            rows, cols = np.nonzero(border)
            _append_patch_coords(rows + row0, cols + col0, idx)

    if band_r1:
        rect_r1 = np.concatenate(band_r1)
        rect_r2 = np.concatenate(band_r2)
        rect_c1 = np.concatenate(band_c1)
        rect_c2 = np.concatenate(band_c2)
        rect_box = np.concatenate(band_box)
        np.clip(rect_r1, 0, None, out=rect_r1)
        np.clip(rect_c1, 0, None, out=rect_c1)
        np.clip(rect_r2, None, height - 1, out=rect_r2)
        np.clip(rect_c2, None, width - 1, out=rect_c2)
        heights = np.maximum(rect_r2 - rect_r1 + 1, 0)
        widths = np.maximum(rect_c2 - rect_c1 + 1, 0)
    else:
        rect_r1 = rect_c1 = heights = widths = rect_box = _EMPTY_I64

    pre_flat = np.concatenate(coord_flat) if coord_flat else _EMPTY_I64
    pre_box = np.concatenate(coord_box) if coord_box else _EMPTY_I64
    # Exact expansion sizes, computed host-side so the device-side ragged
    # expansion never needs a GPU->CPU sync. Zero-area rects (fully clamped
    # away) simply repeat 0 times.
    total_rows = int(heights.sum())
    total_px = int((heights * widths).sum())
    if total_px == 0 and len(pre_flat) == 0:
        return scene_chw

    # Host-side pairwise overlap test on the expanded paint bounds (O(n^2) on
    # tiny arrays). Disjoint borders make every pixel's winner its own box, so
    # the owner-resolution kernels can be skipped entirely — the common case.
    inter = (
        (np.maximum(exp_x1[:, None], exp_x1[None, :])
         <= np.minimum(exp_x2[:, None], exp_x2[None, :]))
        & (np.maximum(exp_y1[:, None], exp_y1[None, :])
           <= np.minimum(exp_y2[:, None], exp_y2[None, :]))
    )
    boxes_overlap = int(inter.sum()) > n  # diagonal is always True

    # ---- single packed H2D transfer --------------------------------------
    # int32 throughout: pixel indices fit (frame size is guarded in run());
    # halving the element width halves the transfer and every device-side
    # memory pass. torch's index ops demand int64, so `flat` is cast once.
    segments = [
        rect_r1,
        heights,
        rect_c1,
        widths,
        rect_box,
        pre_flat,
        pre_box,
        colors_rgb.astype(np.int64).ravel(),
    ]
    packed = torch.from_numpy(np.concatenate(segments).astype(np.int32)).to(device)
    views = []
    offset = 0
    for segment in segments:
        views.append(packed[offset : offset + len(segment)])
        offset += len(segment)
    r1_t, heights_t, c1_t, widths_t, box_t, pre_flat_t, pre_box_t, colors_flat_t = views

    # ---- device-side two-level ragged expansion (rect -> row -> pixel) ----
    flat_parts: List[torch.Tensor] = []
    box_parts: List[torch.Tensor] = []
    if total_px:
        rect_count = int(heights.shape[0])
        rect_of_row = torch.repeat_interleave(
            torch.arange(rect_count, device=device),
            heights_t.long(),
            output_size=total_rows,
        )
        row_starts = heights_t.cumsum(0) - heights_t
        row_intra = (
            torch.arange(total_rows, device=device, dtype=torch.int32)
            - row_starts[rect_of_row]
        )
        row_base = (r1_t[rect_of_row] + row_intra) * width + c1_t[rect_of_row]
        row_width = widths_t[rect_of_row]
        row_box = box_t[rect_of_row]
        px_of_row = torch.repeat_interleave(
            torch.arange(total_rows, device=device),
            row_width.long(),
            output_size=total_px,
        )
        col_starts = row_width.cumsum(0) - row_width
        px_intra = (
            torch.arange(total_px, device=device, dtype=torch.int32)
            - col_starts[px_of_row]
        )
        flat_parts.append(row_base[px_of_row] + px_intra)
        box_parts.append(row_box[px_of_row])
    if len(pre_flat):
        flat_parts.append(pre_flat_t)
        box_parts.append(pre_box_t)

    flat = torch.cat(flat_parts) if len(flat_parts) > 1 else flat_parts[0]
    pixel_box = torch.cat(box_parts) if len(box_parts) > 1 else box_parts[0]
    flat_long = flat.long()  # index ops require int64
    colors_dev = colors_flat_t.view(n, 3).to(torch.uint8)
    if boxes_overlap:
        # sv paints boxes sequentially, so the HIGHEST detection index wins
        # every contested pixel: one deterministic amax scatter of box ids.
        # include_self=False: uninitialized cells never participate, and every
        # gathered position below was scattered to — no fill kernel needed.
        owner = torch.empty(height * width, dtype=torch.int32, device=device)
        owner.scatter_reduce_(
            0, flat_long, pixel_box, reduce="amax", include_self=False
        )
        winner_colors = colors_dev[owner[flat_long].long()]  # (P, 3) uint8
    else:
        winner_colors = colors_dev[pixel_box.long()]
    # .view (not .reshape): guarantees the write lands in the caller's storage
    # (raises on a non-contiguous scene -> caught by the block's sv fallback).
    scene_chw.view(3, -1)[:, flat_long] = winner_colors.t()
    return scene_chw

    # ---- single packed H2D transfer --------------------------------------
    segments = [
        rows,
        row_width,
        row_c1,
        row_box,
        pre_flat,
        pre_box,
        colors_rgb.astype(np.int64).ravel(),
    ]
    packed = torch.from_numpy(np.concatenate(segments)).to(device)
    views = []
    offset = 0
    for segment in segments:
        views.append(packed[offset : offset + len(segment)])
        offset += len(segment)
    rows_t, width_t, c1_t, box_t, pre_flat_t, pre_box_t, colors_flat_t = views

    # ---- device-side expansion --------------------------------------------
    flat_parts: List[torch.Tensor] = []
    box_parts: List[torch.Tensor] = []
    if total_px:
        row_of_px = torch.repeat_interleave(
            torch.arange(int(rows_t.shape[0]), device=device),
            width_t,
            output_size=total_px,
        )
        col_starts = width_t.cumsum(0) - width_t
        cols = c1_t[row_of_px] + (
            torch.arange(total_px, device=device) - col_starts[row_of_px]
        )
        flat_parts.append(rows_t[row_of_px] * width + cols)
        box_parts.append(box_t[row_of_px])
    if len(pre_flat):
        flat_parts.append(pre_flat_t)
        box_parts.append(pre_box_t)

    flat = torch.cat(flat_parts) if len(flat_parts) > 1 else flat_parts[0]
    pixel_box = torch.cat(box_parts) if len(box_parts) > 1 else box_parts[0]
    # include_self=False: uninitialized cells never participate, and every
    # gathered position below was scattered to — no fill kernel needed.
    owner = torch.empty(height * width, dtype=torch.int64, device=device)
    owner.scatter_reduce_(0, flat, pixel_box, reduce="amax", include_self=False)
    colors_dev = colors_flat_t.view(n, 3).to(torch.uint8)
    winner_colors = colors_dev[owner[flat]]  # (P, 3) uint8
    # .view (not .reshape): guarantees the write lands in the caller's storage
    # (raises on a non-contiguous scene -> caught by the block's sv fallback).
    scene_chw.view(3, -1)[:, flat] = winner_colors.t()
    return scene_chw


def _gpu_box_draw_eligible(
    detections, color_axis: str, thickness, image: WorkflowImageData
) -> bool:
    """True when the torch painter can replicate the sv path exactly."""
    if color_axis not in ("CLASS", "INDEX"):
        # TRACK / custom lookups keep the sv path.
        return False
    if not isinstance(thickness, int) or thickness < 1:
        return False
    if not image.is_tensor_materialised():
        # Forcing tensor_image on a numpy-sourced image is a costly host-side
        # conversion — the sv path is faster there.
        return False
    xyxy = getattr(detections, "xyxy", None)
    if not isinstance(xyxy, torch.Tensor) or int(xyxy.shape[0]) == 0:
        # Nothing to paint; the sv path is a trivial no-op.
        return False
    return True


class BoundingBoxManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "BoundingBoxVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Bounding Box Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-object-group",
                "blockPriority": 0,
                "supervision": True,
                "popular": True,
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

    thickness: Union[int, Selector(kind=[INTEGER_KIND])] = Field(  # type: ignore
        description="Thickness of the bounding box edges in pixels. Higher values create thicker, more visible box outlines.",
        default=2,
        examples=[2, "$inputs.thickness"],
    )

    roundness: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Roundness of the bounding box corners, ranging from 0.0 (square corners) to 1.0 (fully rounded corners). When set to 0.0, square-cornered boxes are used; higher values create progressively more rounded corners.",
        default=0.0,
        examples=[0.0, "$inputs.roundness"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class BoundingBoxVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BoundingBoxManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        thickness: int,
        roundness: float,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(str, [color_palette, palette_size, color_axis, thickness, roundness])
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            if roundness == 0:
                self.annotatorCache[key] = sv.BoxAnnotator(
                    color=palette,
                    color_lookup=getattr(sv.ColorLookup, color_axis),
                    thickness=thickness,
                )
            else:
                self.annotatorCache[key] = sv.RoundBoxAnnotator(
                    color=palette,
                    color_lookup=getattr(sv.ColorLookup, color_axis),
                    thickness=thickness,
                    roundness=roundness,
                )
        return self.annotatorCache[key]

    def run(
        self,
        image: WorkflowImageData,
        predictions: Union[TensorNativePrediction, TensorNativeDetections],
        copy_image: bool,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: Optional[str],
        thickness: Optional[int],
        roundness: Optional[float],
    ) -> BlockResult:
        detections = (
            split_key_point_prediction(predictions)[1]
            if isinstance(predictions, tuple)
            else predictions
        )
        if _gpu_box_draw_eligible(detections, color_axis, thickness, image):
            try:
                palette = self.getPalette(color_palette, palette_size, custom_colors)
                if not isinstance(palette, sv.ColorPalette):
                    raise TypeError("expected sv.ColorPalette")
                # Same colors sv's resolve_color would pick: CLASS ->
                # palette.by_idx(class_id), INDEX -> palette.by_idx(det index).
                if color_axis == "CLASS":
                    ids = detections.class_id.detach().cpu().numpy().astype(int)
                else:
                    ids = np.arange(int(detections.xyxy.shape[0]))
                colors_rgb = np.asarray(
                    [palette.by_idx(int(idx)).as_rgb() for idx in ids],
                    dtype=np.uint8,
                )
                # Same int conversion as the sv annotator loop (truncation
                # toward zero, incl. negative coords).
                xyxy = detections.xyxy.detach().cpu().numpy().astype(int)
                # Tensor pipeline contract: the image is a CHW RGB device
                # tensor — zero-copy in, tensor out (downstream materialises
                # numpy lazily only if something asks for it).
                scene_t = image.tensor_image
                if int(scene_t.shape[0]) != 3:
                    raise ValueError("GPU box painter requires a 3-channel image")
                if copy_image:
                    scene_t = scene_t.clone()
                annotated_tensor = gpu_draw_boxes(
                    scene_t, xyxy, colors_rgb, int(thickness), float(roundness)
                )
                if not copy_image:
                    # The painter mutated `image.tensor_image` storage in
                    # place (the sv-path contract for copy_image=False);
                    # invalidate the derived numpy/base64 caches.
                    image.declare_tensor_image_mutated()
                return {
                    OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                        origin_image_data=image, tensor_image=annotated_tensor
                    )
                }
            except Exception as gpu_error:
                logger.debug(
                    "GPU box painter failed (%s); falling back to "
                    "sv.BoxAnnotator path.",
                    gpu_error,
                )
        predictions = to_supervision_for_annotation(predictions)
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            thickness,
            roundness,
        )
        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
