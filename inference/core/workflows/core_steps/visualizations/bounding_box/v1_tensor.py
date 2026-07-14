from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

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

_EMPTY_I64 = np.zeros(0, dtype=np.int64)

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


@lru_cache(maxsize=256)
def _quarter_arc_offsets(radius: int, thickness: int) -> Tuple[np.ndarray, np.ndarray]:
    """(dy, dx) pixel offsets of a top-left quarter ring around an arc
    center — a `thickness`-wide annulus at `radius`. Mirrored by sign for
    the other three corners. Cached per (radius, thickness) across frames."""
    outer = thickness // 2
    span = radius + outer
    yy, xx = np.mgrid[-span:1, -span:1]
    dist = np.sqrt(yy**2 + xx**2)
    ring = (dist >= radius - (thickness - 1 - outer) - 0.5) & (
        dist <= radius + outer + 0.5
    )
    return yy[ring].astype(np.int64), xx[ring].astype(np.int64)


def gpu_draw_boxes(
    scene_chw: torch.Tensor,
    xyxy: np.ndarray,
    colors_rgb: np.ndarray,
    thickness: int,
    roundness: float = 0.0,
) -> torch.Tensor:
    """Draw box borders on a CHW RGB uint8 device tensor, in place.

    Approximate rendering: each border is a plain ``thickness``-wide band
    centered on the box edge. ``roundness > 0`` rounds the corners with
    analytic quarter-ring arcs at sv's corner radius
    (``int(min_side // 2 * roundness)``). cv2 (the sv path) rasterises
    joins/caps/arcs slightly differently, so output is visually equivalent,
    not bit-identical.

    Painted with a fixed number of torch ops regardless of box count — a
    per-box loop is dispatch-bound on Jetson (measured 43 ms @ 50 boxes):

    1. Host: 4 band rectangles per box, clamped to the frame.
    2. One packed int32 H2D upload (per-transfer fixed cost is ~0.36 ms on
       AGX Orin), then two-level ragged expansion to flat pixel indices via
       ``repeat_interleave`` with host-computed ``output_size`` — no
       GPU->CPU syncs.
    3. One indexed store. When boxes overlap, sv's later-box-wins paint
       order is reproduced with a deterministic ``scatter_reduce_(amax)``
       of box indices; disjoint boxes (the common case) skip it.
    """
    device = scene_chw.device
    height, width = int(scene_chw.shape[1]), int(scene_chw.shape[2])
    if height * width >= 2**31:
        raise ValueError("frame too large for int32 pixel indexing")
    n = int(xyxy.shape[0])
    outer = thickness // 2  # band extends `outer` outward, rest inward

    xy = np.asarray(xyxy, dtype=np.int64)
    bx1 = np.minimum(xy[:, 0], xy[:, 2])
    bx2 = np.maximum(xy[:, 0], xy[:, 2])
    by1 = np.minimum(xy[:, 1], xy[:, 3])
    by2 = np.maximum(xy[:, 1], xy[:, 3])
    if roundness > 0:
        # sv.RoundBoxAnnotator's corner radius, from the smaller box side.
        radii = (np.minimum(bx2 - bx1, by2 - by1) // 2 * roundness).astype(np.int64)
    else:
        radii = np.zeros(n, dtype=np.int64)
    x1, x2, y1, y2 = bx1 - outer, bx2 + outer, by1 - outer, by2 + outer

    # 4 bands per box (inclusive coords). Square corners: top/bottom span
    # the full width, left/right fill between them. Rounded: every band is
    # inset by the corner radius; quarter-ring arcs (below) fill the joins.
    # Degenerate boxes just overlap bands of the same color — harmless.
    t = thickness
    inset = radii - outer  # 0 boxes: -outer == square full-width bands
    rect_r1 = np.concatenate([y1, y2 - t + 1, by1 + radii, by1 + radii])
    rect_r2 = np.concatenate([y1 + t - 1, y2, by2 - radii, by2 - radii])
    rect_c1 = np.concatenate([x1 + inset + outer, x1 + inset + outer, x1, x2 - t + 1])
    rect_c2 = np.concatenate([x2 - inset - outer, x2 - inset - outer, x1 + t - 1, x2])
    rect_box = np.tile(np.arange(n), 4)
    np.clip(rect_r1, 0, None, out=rect_r1)
    np.clip(rect_c1, 0, None, out=rect_c1)
    np.clip(rect_r2, None, height - 1, out=rect_r2)
    np.clip(rect_c2, None, width - 1, out=rect_c2)
    heights = np.maximum(rect_r2 - rect_r1 + 1, 0)
    widths = np.maximum(rect_c2 - rect_c1 + 1, 0)
    total_rows = int(heights.sum())
    total_px = int((heights * widths).sum())
    if total_px == 0:
        return scene_chw

    # Pairwise overlap test on the expanded bounds: disjoint borders make
    # every pixel's winner its own box, so owner resolution can be skipped.
    inter_x = np.maximum(x1[:, None], x1[None, :]) <= np.minimum(
        x2[:, None], x2[None, :]
    )
    inter_y = np.maximum(y1[:, None], y1[None, :]) <= np.minimum(
        y2[:, None], y2[None, :]
    )
    boxes_overlap = int((inter_x & inter_y).sum()) > n  # diagonal always True

    # Rounded corners: quarter-ring arc pixels per box, grouped by radius so
    # the ring offsets are computed (and lru-cached) once per radius. Coords
    # are pre-resolved to flat indices on host and ride the packed upload.
    arc_flat = arc_box = _EMPTY_I64
    if roundness > 0:
        flat_parts, box_parts = [], []
        for radius in np.unique(radii):
            idx = np.nonzero(radii == radius)[0]
            dy, dx = _quarter_arc_offsets(int(radius), thickness)
            centers_y = (
                by1[idx] + radius,
                by1[idx] + radius,
                by2[idx] - radius,
                by2[idx] - radius,
            )
            centers_x = (
                bx1[idx] + radius,
                bx2[idx] - radius,
                bx1[idx] + radius,
                bx2[idx] - radius,
            )
            signs = ((1, 1), (1, -1), (-1, 1), (-1, -1))
            for cy, cx, (sy, sx) in zip(centers_y, centers_x, signs):
                rows = (cy[:, None] + sy * dy[None, :]).ravel()
                cols = (cx[:, None] + sx * dx[None, :]).ravel()
                keep = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
                flat_parts.append(rows[keep] * width + cols[keep])
                box_parts.append(np.repeat(idx, len(dy))[keep])
        arc_flat = np.concatenate(flat_parts)
        arc_box = np.concatenate(box_parts)

    # Single packed upload; int32 halves the transfer and every device-side
    # memory pass (torch index ops demand int64 — `flat` is cast once).
    segments = [
        rect_r1,
        heights,
        rect_c1,
        widths,
        rect_box,
        arc_flat,
        arc_box,
        colors_rgb.astype(np.int64).ravel(),
    ]
    packed = torch.from_numpy(np.concatenate(segments).astype(np.int32)).to(device)
    views = []
    offset = 0
    for segment in segments:
        views.append(packed[offset : offset + len(segment)])
        offset += len(segment)
    r1_t, heights_t, c1_t, widths_t, box_t, arc_flat_t, arc_box_t, colors_flat_t = views

    # Two-level ragged expansion: rect -> row -> pixel.
    rect_of_row = torch.repeat_interleave(
        torch.arange(4 * n, device=device), heights_t.long(), output_size=total_rows
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
        torch.arange(total_rows, device=device), row_width.long(), output_size=total_px
    )
    col_starts = row_width.cumsum(0) - row_width
    px_intra = (
        torch.arange(total_px, device=device, dtype=torch.int32) - col_starts[px_of_row]
    )
    flat = row_base[px_of_row] + px_intra
    pixel_box = row_box[px_of_row]
    if len(arc_flat):
        flat = torch.cat([flat, arc_flat_t])
        pixel_box = torch.cat([pixel_box, arc_box_t])
    flat = flat.long()

    colors_dev = colors_flat_t.view(n, 3).to(torch.uint8)
    if boxes_overlap:
        # include_self=False: uninitialized cells never participate, and
        # every gathered position below was scattered to.
        owner = torch.empty(height * width, dtype=torch.int32, device=device)
        owner.scatter_reduce_(0, flat, pixel_box, reduce="amax", include_self=False)
        winner_colors = colors_dev[owner[flat].long()]  # (P, 3) uint8
    else:
        winner_colors = colors_dev[pixel_box.long()]
    # .view (not .reshape): guarantees the write lands in the caller's storage
    # (raises on a non-contiguous scene -> caught by the block's sv fallback).
    scene_chw.view(3, -1)[:, flat] = winner_colors.t()
    return scene_chw


def _gpu_box_draw_eligible(
    detections, color_axis: str, thickness, image: WorkflowImageData
) -> bool:
    """True when the torch painter can replace the sv path."""
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
