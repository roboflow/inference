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
    """cv2-derived rasterisation template for one box thickness.

    ``cv2.rectangle`` with ``thickness >= 2`` is NOT an outer-rect-minus-
    inner-rect frame: the four edges are thick lines with round joins, so the
    outermost corner pixels stay unpainted (and even thicknesses paint
    ``thickness + 1`` wide bands). Instead of re-deriving OpenCV's thick-line
    rasteriser, one reference rectangle is drawn with cv2 itself and decomposed
    into four uniform straight bands plus four corner patches. Reconstruction
    is asserted pixel-perfect against cv2 on two reference sizes, so the GPU
    painter is bit-exact by construction for every box large enough to use the
    template (smaller boxes take the per-box cv2 raster path).
    """

    def __init__(self, thickness: int):
        self.thickness = thickness
        reference_side = 8 * thickness + 32
        second_side = reference_side + 9  # catches any size-dependence
        margin = 4 * thickness + 8
        ref = self._rasterise(margin, reference_side, thickness)
        ys, xs = np.nonzero(ref)
        self.outer = margin - int(ys.min())
        mid_column = np.nonzero(ref[:, margin + reference_side // 2])[0]
        self.inner = (
            int(mid_column[mid_column < margin + reference_side // 2].max()) - margin
        )
        self.corner = 2 * (self.outer + self.inner) + 2
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
        for side in (reference_side, second_side):
            self._assert_reconstruction(margin, side, thickness)

    @staticmethod
    def _rasterise(margin: int, side: int, thickness: int) -> np.ndarray:
        size = side + 2 * margin + 1
        canvas = np.zeros((size, size), np.uint8)
        cv2.rectangle(
            canvas, (margin, margin), (margin + side, margin + side), 255, thickness
        )
        return canvas > 0

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

    def _assert_reconstruction(self, margin: int, side: int, thickness: int) -> None:
        ref = self._rasterise(margin, side, thickness)
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
                f"cv2.rectangle band+corner decomposition failed for "
                f"thickness={thickness} (unexpected OpenCV rasteriser geometry)"
            )


_GEOMETRY_CACHE: Dict[int, _BorderGeometry] = {}
_EMPTY_I64 = np.zeros(0, dtype=np.int64)
_DEVICE_CORNER_CACHE: Dict[Tuple[int, str], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_geometry(thickness: int) -> _BorderGeometry:
    if thickness not in _GEOMETRY_CACHE:
        _GEOMETRY_CACHE[thickness] = _BorderGeometry(thickness)
    return _GEOMETRY_CACHE[thickness]


def _get_device_corner_offsets(
    geometry: _BorderGeometry, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-corner-slot local True offsets of the corner templates, padded to a
    common length: ``(off_r, off_c, valid)`` each ``(4, K_max)`` on ``device``.
    Uploaded once per (thickness, device) and reused for every frame."""
    key = (geometry.thickness, str(device))
    if key not in _DEVICE_CORNER_CACHE:
        per_slot = [np.nonzero(mask) for mask in geometry.corner_masks]
        k_max = max(len(rows) for rows, _ in per_slot)
        off_r = np.zeros((4, k_max), dtype=np.int64)
        off_c = np.zeros((4, k_max), dtype=np.int64)
        valid = np.zeros((4, k_max), dtype=bool)
        for slot, (rows, cols) in enumerate(per_slot):
            off_r[slot, : len(rows)] = rows
            off_c[slot, : len(cols)] = cols
            valid[slot, : len(rows)] = True
        _DEVICE_CORNER_CACHE[key] = (
            torch.from_numpy(off_r).to(device),
            torch.from_numpy(off_c).to(device),
            torch.from_numpy(valid).to(device),
        )
    return _DEVICE_CORNER_CACHE[key]


def gpu_draw_boxes(
    scene_chw: torch.Tensor,
    xyxy: np.ndarray,
    colors_rgb: np.ndarray,
    thickness: int,
) -> torch.Tensor:
    """GPU-native torch replacement for ``sv.BoxAnnotator.annotate`` on a CHW
    RGB uint8 device tensor (the ``WorkflowImageData.tensor_image`` contract).
    The scene tensor is mutated in place.

    A per-box paint loop is hopeless on Jetson — python-op/kernel-launch
    overhead (~12 ops per box) dominates the tiny border writes. Instead the
    whole frame is painted with a FIXED number of torch ops regardless of box
    count:

    1. Host (numpy, all boxes at once): band rectangles + corner-patch anchors
       from the cv2-derived geometry templates (see ``_BorderGeometry``),
       clamped to the frame. Boxes too small for the template decomposition
       are rasterised exactly by cv2.rectangle on tiny canvases and contribute
       their pixel coords directly.
    2. Device: ragged expansion of band rects to flat pixel indices
       (``repeat_interleave`` with host-computed ``output_size`` — no device
       sync), plus broadcast expansion of the padded corner offsets.
    3. Overlap resolution: sv paints boxes sequentially so the HIGHEST
       detection index wins every contested pixel — reproduced exactly by one
       deterministic ``scatter_reduce_(amax)`` of box indices over the
       frame, then a single indexed color write. Duplicate indices all write
       the resolved winner's color, so the write is deterministic too.
    """
    geometry = _get_geometry(thickness)
    device = scene_chw.device
    height, width = int(scene_chw.shape[1]), int(scene_chw.shape[2])
    o, c = geometry.outer, geometry.corner
    inner = geometry.inner
    n = int(xyxy.shape[0])

    xy = np.asarray(xyxy, dtype=np.int64)
    # cv2.rectangle normalizes swapped corners.
    x1 = np.minimum(xy[:, 0], xy[:, 2])
    x2 = np.maximum(xy[:, 0], xy[:, 2])
    y1 = np.minimum(xy[:, 1], xy[:, 3])
    y2 = np.maximum(xy[:, 1], xy[:, 3])
    big = ((x2 - x1 + 1) >= geometry.min_side) & ((y2 - y1 + 1) >= geometry.min_side)

    # ---- host-side prep (numpy) ----------------------------------------
    rows = row_width = row_c1 = row_box = _EMPTY_I64
    big_idx = np.nonzero(big)[0]
    if len(big_idx):
        bx1, bx2 = x1[big_idx], x2[big_idx]
        by1, by2 = y1[big_idx], y2[big_idx]
        ox1, oy1, ox2, oy2 = bx1 - o, by1 - o, bx2 + o, by2 + o
        # Band rects (inclusive r1, r2, c1, c2) + owning detection index.
        rect_r1 = np.concatenate([oy1, by2 - inner, oy1 + c, oy1 + c])
        rect_r2 = np.concatenate([by1 + inner, oy2, oy2 - c, oy2 - c])
        rect_c1 = np.concatenate([ox1 + c, ox1 + c, ox1, bx2 - inner])
        rect_c2 = np.concatenate([ox2 - c, ox2 - c, bx1 + inner, ox2])
        rect_box = np.tile(big_idx, 4)
        np.clip(rect_r1, 0, None, out=rect_r1)
        np.clip(rect_c1, 0, None, out=rect_c1)
        np.clip(rect_r2, None, height - 1, out=rect_r2)
        np.clip(rect_c2, None, width - 1, out=rect_c2)
        heights = np.maximum(rect_r2 - rect_r1 + 1, 0)
        widths = np.maximum(rect_c2 - rect_c1 + 1, 0)
        # Row-level expansion on host: total row count is small (~band height
        # x 4 x boxes), so numpy is cheap and gives exact output sizes for the
        # device-side column expansion — no GPU->CPU sync anywhere.
        total_rows = int(heights.sum())
        if total_rows:
            rect_of_row = np.repeat(np.arange(len(heights)), heights)
            row_starts = np.cumsum(heights) - heights
            rows = rect_r1[rect_of_row] + (
                np.arange(total_rows) - row_starts[rect_of_row]
            )
            row_width = widths[rect_of_row]
            keep = row_width > 0
            rows, row_width = rows[keep], row_width[keep]
            row_c1 = rect_c1[rect_of_row][keep]
            row_box = rect_box[rect_of_row][keep]
        # Corner-patch anchors (r, c) per corner slot (TL, TR, BL, BR).
        anchor_r = np.concatenate([oy1, oy1, oy2 - c + 1, oy2 - c + 1])
        anchor_c = np.concatenate([ox1, ox2 - c + 1, ox1, ox2 - c + 1])
        anchor_slot = np.repeat(np.arange(4), len(big_idx))
        anchor_box = np.tile(big_idx, 4)
    else:
        anchor_r = anchor_c = anchor_slot = anchor_box = _EMPTY_I64

    small_flat = small_box = _EMPTY_I64
    small_idx = np.nonzero(~big)[0]
    if len(small_idx):
        small_flat_parts: List[np.ndarray] = []
        small_box_parts: List[np.ndarray] = []
        for idx in small_idx:
            # Joins overlap below min_side: rasterise the exact border with
            # cv2 on a tiny host canvas (bounded by min_side + 2*outer).
            bw, bh = int(x2[idx] - x1[idx]), int(y2[idx] - y1[idx])
            canvas = np.zeros((bh + 1 + 2 * o, bw + 1 + 2 * o), np.uint8)
            cv2.rectangle(canvas, (o, o), (o + bw, o + bh), 255, thickness)
            rr, cc = np.nonzero(canvas)
            rr = rr + int(y1[idx]) - o
            cc = cc + int(x1[idx]) - o
            keep = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
            small_flat_parts.append(rr[keep] * width + cc[keep])
            small_box_parts.append(np.full(int(keep.sum()), idx, dtype=np.int64))
        small_flat = np.concatenate(small_flat_parts)
        small_box = np.concatenate(small_box_parts)

    total_px = int(row_width.sum())
    if total_px == 0 and len(anchor_r) == 0 and len(small_flat) == 0:
        return scene_chw

    # ---- single packed H2D transfer -------------------------------------
    # Per-transfer fixed cost is brutal on some Jetsons (~0.36 ms on AGX Orin
    # vs ~0.06 ms on Orin Nano), so every host-built array ships in ONE
    # upload and is sliced back into views on-device.
    segments = [
        rows,
        row_width,
        row_c1,
        row_box,
        anchor_r,
        anchor_c,
        anchor_slot,
        anchor_box,
        small_flat,
        small_box,
        colors_rgb.astype(np.int64).ravel(),
    ]
    packed = torch.from_numpy(np.concatenate(segments)).to(device)
    views = []
    offset = 0
    for segment in segments:
        views.append(packed[offset : offset + len(segment)])
        offset += len(segment)
    (
        rows_t,
        width_t,
        c1_t,
        box_t,
        a_r,
        a_c,
        slot_t,
        corner_box_t,
        small_flat_t,
        small_box_t,
        colors_flat_t,
    ) = views

    # ---- device-side expansion ------------------------------------------
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
    if len(anchor_r):
        off_r, off_c, off_valid = _get_device_corner_offsets(geometry, device)
        rr = a_r.unsqueeze(1) + off_r[slot_t]  # (A, K_max)
        cc = a_c.unsqueeze(1) + off_c[slot_t]
        keep = (
            off_valid[slot_t]
            & (rr >= 0)
            & (rr < height)
            & (cc >= 0)
            & (cc < width)
        )
        flat_parts.append((rr * width + cc)[keep])
        box_parts.append(
            corner_box_t.unsqueeze(1).expand(-1, int(off_r.shape[1]))[keep]
        )
    if len(small_flat):
        flat_parts.append(small_flat_t)
        box_parts.append(small_box_t)

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
    detections, color_axis: str, roundness: float, thickness
) -> bool:
    """True when the torch painter can replicate the sv path exactly."""
    if roundness != 0:
        # sv.RoundBoxAnnotator (elliptic arcs) keeps the battle-tested sv path.
        return False
    if color_axis not in ("CLASS", "INDEX"):
        # TRACK / custom lookups keep the sv path.
        return False
    if not isinstance(thickness, int) or thickness < 1:
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
        if _gpu_box_draw_eligible(detections, color_axis, roundness, thickness):
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
                    scene_t, xyxy, colors_rgb, int(thickness)
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
