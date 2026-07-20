from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.tensor_native import (
    TensorNativeDetections,
    TensorNativePrediction,
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
from inference.core.workflows.execution_engine.entities.tensor_native_types import (
    TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_ZERO_TO_ONE_KIND,
    SEMANTIC_SEGMENTATION_PREDICTION_KIND,
    FloatZeroToOne,
    Selector,
)
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

TYPE: str = "roboflow_core/mask_visualization@v1"
SHORT_DESCRIPTION = "Apply a mask over detected objects in an image."
LONG_DESCRIPTION = """
Fill segmentation masks with semi-transparent color overlays, creating solid color fills that precisely follow the shape of detected objects from instance segmentation predictions.

## How This Block Works

This block takes an image and instance segmentation predictions (with masks) and fills the mask regions with colored overlays. The block:

1. Takes an image and instance segmentation predictions (with masks) as input
2. Extracts segmentation masks for each detected object from the predictions
3. Applies color styling to each mask based on the selected color palette, with colors assigned by class, index, or track ID
4. Fills the mask regions with solid colors using Supervision's MaskAnnotator
5. Blends the colored mask overlays with the original image using the specified opacity level
6. Returns an annotated image where mask regions are filled with semi-transparent colors, while non-masked areas remain unchanged

The block fills the exact shape of each object's segmentation mask with colored overlays, creating solid color fills that precisely follow object boundaries. Unlike polygon visualization (which draws outlines) or bounding box visualizations (which use rectangular regions), mask visualization fills the entire mask area with color, providing clear visual indication of the segmented regions. The opacity parameter controls how transparent the mask overlay is, allowing you to see the original image details through the colored mask (lower opacity) or create more opaque fills (higher opacity) that better obscure background details. This block requires instance segmentation predictions with mask data, as it specifically works with segmentation masks to create precise, shape-following color fills.

## Common Use Cases

- **Instance Segmentation Visualization**: Visualize instance segmentation results by filling mask regions with colors to clearly show segmented objects, validate segmentation quality, or highlight detected regions in analysis workflows
- **Precise Shape-Following Overlays**: Fill objects with colors that exactly match their segmented shapes, useful for applications requiring accurate region visualization such as medical imaging, quality control, or precise object identification
- **Mask-Based Object Highlighting**: Highlight segmented objects with colored overlays that follow exact object boundaries, providing clear visual distinction between different objects or object classes
- **Segmentation Model Validation**: Visualize segmentation predictions with colored mask fills to verify model performance, identify segmentation errors, or validate mask accuracy in model development and debugging workflows
- **Medical and Scientific Imaging**: Display segmented regions in medical imaging, microscopy, or scientific analysis applications where colored mask overlays help visualize tissue boundaries, cell regions, or measured areas
- **Mask Quality Inspection**: Use colored mask fills to inspect segmentation quality, verify mask boundaries, or identify areas where segmentation may need improvement in training data or model outputs

## Connecting to Other Blocks

The annotated image from this block can be connected to:

- **Other visualization blocks** (e.g., Label Visualization, Polygon Visualization, Bounding Box Visualization) to combine mask fills with additional annotations (labels, outlines) for comprehensive visualization
- **Data storage blocks** (e.g., Local File Sink, CSV Formatter, Roboflow Dataset Upload) to save images with mask overlays for documentation, reporting, or analysis
- **Webhook blocks** to send visualized results with mask fills to external systems, APIs, or web applications for display in dashboards or monitoring tools
- **Notification blocks** (e.g., Email Notification, Slack Notification) to send annotated images with mask overlays as visual evidence in alerts or reports
- **Video output blocks** to create annotated video streams or recordings with mask fills for live monitoring, segmentation visualization, or post-processing analysis
"""


def _coco_rle_counts_to_runs(counts) -> np.ndarray:
    """Decode a COCO compressed-RLE ``counts`` payload into run lengths.

    Vectorised numpy port of pycocotools' ``rleFrString``: each value is a
    varint of base-48 chars carrying 5 data bits + a continuation bit (0x20),
    with sign bit 0x10 in the final char, and from the 4th value on each count
    stored as a delta against the value two back. Uncompressed payloads (a
    list of ints) pass through. Returns int64 run lengths alternating
    background/foreground over the column-major (Fortran) pixel order,
    background first.
    """
    if isinstance(counts, (list, tuple, np.ndarray)):
        return np.asarray(counts, dtype=np.int64)
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    chars = np.frombuffer(counts, dtype=np.uint8).astype(np.int64) - 48
    if chars.size == 0:
        return np.zeros(0, dtype=np.int64)
    ends = (chars & 0x20) == 0  # final char of each varint
    ends_idx = np.flatnonzero(ends)
    starts_idx = np.concatenate(([0], ends_idx[:-1] + 1))
    value_id = np.cumsum(np.concatenate(([False], ends[:-1])))
    bit_shift = 5 * (np.arange(chars.size) - starts_idx[value_id])
    values = np.zeros(ends_idx.size, dtype=np.int64)
    np.add.at(values, value_id, (chars & 0x1F) << bit_shift)
    negative = (chars[ends_idx] & 0x10) != 0
    values[negative] -= np.int64(1) << (5 * (ends_idx - starts_idx + 1)[negative])
    # Undo the delta coding: values 3+ each add the decoded value two back,
    # which is a running sum over the odd and even positions independently.
    if values.size > 3:
        values[3::2] = values[1] + np.cumsum(values[3::2])
    if values.size > 4:
        values[4::2] = values[2] + np.cumsum(values[4::2])
    return values


def _rle_foreground_pixels_in_roi(
    masks: "InstancesRLEMasks",
    roi: tuple,
    device: torch.device,
):
    """Turn COCO-RLE masks into ``(flat ROI pixel index, detection id)`` device
    tensors without materialising any dense mask.

    Host work is proportional to the encoded byte size (the varint decode is
    vectorised numpy); only the per-run tables cross the bus — one packed H2D
    upload — and the run→pixel expansion happens on the device with a
    host-known ``output_size`` (no device→host sync).
    """
    uy1, ux1, uy2, ux2 = roi
    height = int(masks.image_size[0])  # runs are column-major over (h, w)
    starts_l, lens_l, dets_l = [], [], []
    for det_idx, payload in enumerate(masks.masks):
        runs = _coco_rle_counts_to_runs(payload)
        bounds = np.concatenate(([0], np.cumsum(runs)))
        fg_starts, fg_lens = bounds[1::2], runs[1::2]
        keep = fg_lens > 0
        starts_l.append(fg_starts[: fg_lens.size][keep])
        lens_l.append(fg_lens[keep])
        dets_l.append(np.full(int(keep.sum()), det_idx, dtype=np.int64))
    starts = np.concatenate(starts_l) if starts_l else np.zeros(0, dtype=np.int64)
    lens = np.concatenate(lens_l) if lens_l else np.zeros(0, dtype=np.int64)
    total = int(lens.sum())
    empty = torch.zeros(0, dtype=torch.int64, device=device)
    if total == 0:
        return empty, empty
    offsets = np.concatenate(([0], np.cumsum(lens)[:-1]))
    packed = torch.from_numpy(
        np.stack([starts, lens, np.concatenate(dets_l), offsets])
    ).to(device)
    run_starts, run_lens, run_dets, run_offsets = packed
    run_ids = torch.repeat_interleave(
        torch.arange(run_lens.shape[0], device=device),
        run_lens,
        output_size=total,
    )
    pix_f = run_starts[run_ids] + (
        torch.arange(total, device=device, dtype=torch.int64) - run_offsets[run_ids]
    )
    rows, cols = pix_f % height, pix_f // height
    inside = (rows >= uy1) & (rows < uy2) & (cols >= ux1) & (cols < ux2)
    rows, cols = rows[inside], cols[inside]
    return (rows - uy1) * (ux2 - ux1) + (cols - ux1), run_dets[run_ids][inside]


def gpu_mask_composite(
    scene: torch.Tensor,
    predictions: "InstanceDetections",
    colors_rgb: "np.ndarray",
    opacity: float,
) -> torch.Tensor:
    """GPU-native, tensor-only replacement for ``sv.MaskAnnotator.annotate``.

    Tensor pipeline contract on both ends: ``scene`` is a CHW RGB uint8 torch
    tensor (``WorkflowImageData.tensor_image``), mutated IN PLACE and returned
    — no numpy, no layout conversion, no host round-trip. Callers that need
    the original must pass a clone.

    Accepts both tensor-pipeline mask carriers:

    * a dense bool ``torch.Tensor`` ``(N, H, W)`` on the pipeline device, and
    * ``InstancesRLEMasks`` (COCO compressed RLE, column-major) — decoded
      straight into the per-pixel accumulators on the device. The dense
      ``(N, H, W)`` stack is never materialised: the RLE path is
      ``O(total foreground pixels)`` in memory, not ``O(N·H·W)``.

    Overlap semantics — intentionally simpler than supervision's: every mask
    covering a pixel contributes equally, i.e. the overlay color is the MEAN
    of the covering masks' colors, alpha-composited once with the scene:

        count      = Σ_i mask_i                       # (h, w)
        color_sum  = Σ_i mask_i · color_i · opacity   # (h, w, 3)
        out        = color_sum / count + (1-opacity) · scene   where count > 0

    This is order-independent and diverges from supervision's
    smallest-area-owns-the-pixel painter's algorithm on OVERLAPPING pixels
    only; pixels covered by exactly one mask still match ``sv.MaskAnnotator``
    bit-for-bit (same premultiplied blend; ``torch.round_`` is
    round-half-to-even, matching ``cv2.addWeighted``'s ``cvRound``; a convex
    combination of uint8 values needs no clamp). Both carriers share the same
    accumulate-and-blend core (dense masks via one ``nonzero``, RLE via the
    run scatter), and fp32 accumulation keeps counts and color sums exact for
    any realistic N.

    Masks' True pixels are assumed to lie inside their detection boxes (the
    tensor pipeline decodes masks per-box), so all work is restricted to the
    union ROI of the boxes; one tiny ``xyxy`` D2H is the only mandatory host
    sync (the dense path's ``nonzero`` adds one more).

    Args:
        scene: CHW RGB uint8 torch tensor on the pipeline device. Mutated in
            place. (No ``.contiguous()`` is taken: it could silently copy and
            break the in-place contract — ``copy_image=False`` operates on the
            caller's storage.)
        predictions: ``InstanceDetections`` whose ``mask`` is a dense bool
            ``torch.Tensor`` ``(N, H, W)`` or an ``InstancesRLEMasks`` whose
            ``image_size`` matches the scene.
        colors_rgb: ``(N, 3)`` uint8 per-detection colors (RGB), resolved with
            the same palette logic the sv annotator would use.
        opacity: overlay opacity, matches ``sv.MaskAnnotator(opacity=...)``.

    Returns:
        ``scene`` (same tensor, annotated in place).
    """
    mask_carrier = predictions.mask
    is_rle = isinstance(mask_carrier, InstancesRLEMasks)
    device = scene.device
    # All painting/blending below works on an HWC view; the permute is a
    # zero-copy view, so in-place writes land in the CHW storage.
    scene_hwc = scene.permute(1, 2, 0)

    H, W = int(scene_hwc.shape[0]), int(scene_hwc.shape[1])
    mask_hw = (
        tuple(int(s) for s in mask_carrier.image_size)
        if is_rle
        else (int(mask_carrier.shape[1]), int(mask_carrier.shape[2]))
    )
    if mask_hw != (H, W):
        # Silent slicing on a mismatched canvas would paint misaligned masks;
        # raising sends the block to the sv fallback instead.
        raise ValueError(f"mask canvas {mask_hw} does not match scene {(H, W)}")
    # Union ROI of all boxes — one tiny D2H of xyxy (the only host sync).
    # supervision xyxy has inclusive max coords, hence the +1.
    xy = predictions.xyxy.detach().cpu().numpy()
    ux1 = max(0, int(np.floor(xy[:, 0].min())))
    uy1 = max(0, int(np.floor(xy[:, 1].min())))
    ux2 = min(W, int(np.floor(xy[:, 2].max())) + 1)
    uy2 = min(H, int(np.floor(xy[:, 3].max())) + 1)
    if ux2 > ux1 and uy2 > uy1:
        roi_h, roi_w = uy2 - uy1, ux2 - ux1
        # fp32 accumulation: counts and premultiplied color sums are integers
        # and halves well below 2^24, so sums stay exact regardless of the
        # (nondeterministic) atomic add order.
        acc_dtype = torch.float32
        lut_premul = (
            torch.from_numpy(np.ascontiguousarray(colors_rgb))
            .to(device=device, dtype=acc_dtype)
            .mul_(opacity)
        )  # (N, 3)
        if is_rle:
            flat_idx, det_ids = _rle_foreground_pixels_in_roi(
                mask_carrier, (uy1, ux1, uy2, ux2), device
            )
        else:
            det_ids, rows, cols = mask_carrier[:, uy1:uy2, ux1:ux2].nonzero(
                as_tuple=True
            )
            flat_idx = rows * roi_w + cols
        count = torch.zeros(roi_h * roi_w, dtype=acc_dtype, device=device)
        count.index_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=acc_dtype))
        color_sum = torch.zeros(roi_h * roi_w, 3, dtype=acc_dtype, device=device)
        color_sum.index_add_(0, flat_idx, lut_premul[det_ids])
        hit = (count > 0).view(roi_h, roi_w, 1)
        scene_roi = scene_hwc[uy1:uy2, ux1:ux2]
        # mean premultiplied mask color, then fused axpy with the scene
        blended = color_sum.div_(count.clamp_(min=1.0).unsqueeze(1)).view(
            roi_h, roi_w, 3
        )
        blended.add_(scene_roi.to(acc_dtype), alpha=1.0 - opacity)
        blended_u8 = blended.round_().to(torch.uint8)
        # where() = one full-ROI write; boolean indexing would need 2 nonzero
        # passes + gather/scatter.
        scene_roi.copy_(torch.where(hit, blended_u8, scene_roi))

    return scene


def _resolve_color_ids(
    predictions: "InstanceDetections", color_axis: str
) -> np.ndarray:
    """The palette indices sv's ``resolve_color_idx`` would use, raising its
    exact ``ValueError``s when the ids are missing — but BEFORE any mask work,
    so a doomed run doesn't densify RLE masks on the sv fallback path just to
    crash on the same check."""
    n = int(predictions.xyxy.shape[0])
    if color_axis == "INDEX":
        return np.arange(n)
    if color_axis == "CLASS":
        if predictions.class_id is None:
            raise ValueError(
                "Could not resolve color by class because "
                "Detections do not have class_id. If using an annotator, "
                "try setting color_lookup to sv.ColorLookup.INDEX or "
                "sv.ColorLookup.TRACK."
            )
        return predictions.class_id.detach().cpu().numpy().astype(int)
    # TRACK: the tensor pipeline carries tracker ids in per-box metadata (the
    # same place to_supervision_for_annotation reads them from).
    metadata = predictions.bboxes_metadata or []
    tracker_ids = [box.get("tracker_id") for box in metadata]
    if len(tracker_ids) != n or any(tid is None for tid in tracker_ids):
        raise ValueError(
            "Could not resolve color by track because "
            "Detections do not have tracker_id. Did you call "
            "tracker.update_with_detections(...) before annotating?"
        )
    return np.asarray([int(tid) for tid in tracker_ids])


def _gpu_composite_eligible(predictions, color_axis: str) -> bool:
    """True when the torch compositor supports the inputs."""
    if color_axis not in ("CLASS", "INDEX", "TRACK"):
        # Custom lookups keep the battle-tested sv path.
        return False
    if not isinstance(predictions, InstanceDetections):
        return False
    n = int(predictions.xyxy.shape[0])
    if n == 0:
        # Nothing to paint; the sv path is a trivial no-op and avoids an
        # empty-crop edge case in the compositor.
        return False
    # Both tensor-pipeline carriers are supported: a dense (N, H, W) bool
    # torch.Tensor and COCO-RLE (InstancesRLEMasks). Crop views are built from
    # xyxy on whatever device the mask lives on (the loader only registers
    # this block for the tensor pipeline, so device gating is not this
    # block's job).
    mask_carrier = getattr(predictions, "mask", None)
    if isinstance(mask_carrier, InstancesRLEMasks):
        return len(mask_carrier.masks) == n
    return (
        isinstance(mask_carrier, torch.Tensor)
        and mask_carrier.ndim == 3
        and mask_carrier.dtype == torch.bool
        and int(mask_carrier.shape[0]) == n
    )


class MaskManifest(ColorableVisualizationManifest):
    type: Literal[f"{TYPE}", "MaskVisualization"]
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Mask Visualization",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "visualization",
            "search_keywords": ["annotator"],
            "ui_manifest": {
                "section": "visualization",
                "icon": "far fa-mask",
                "blockPriority": 12,
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

    predictions: Selector(
        kind=[
            TENSOR_NATIVE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            TENSOR_NATIVE_RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
            SEMANTIC_SEGMENTATION_PREDICTION_KIND,
        ]
    ) = Field(  # type: ignore
        description="Segmentation predictions containing masks for detected objects. The block uses segmentation masks to create colored fills that precisely follow object or class boundaries. Requires segmentation model outputs with mask data, which may be RLE-encoded.",
        examples=["$steps.instance_segmentation_model.predictions"],
    )

    opacity: Union[FloatZeroToOne, Selector(kind=[FLOAT_ZERO_TO_ONE_KIND])] = Field(  # type: ignore
        description="Opacity of the mask overlay, ranging from 0.0 (fully transparent) to 1.0 (fully opaque). Controls the transparency of the colored mask fill. Lower values (e.g., 0.3-0.5) create semi-transparent overlays that allow original image details to show through, while higher values (e.g., 0.7-1.0) create more opaque fills that better obscure background details. Typical values range from 0.4 to 0.7 for balanced visualization where both the mask and underlying image are visible.",
        default=0.5,
        examples=[0.5, "$inputs.opacity"],
    )

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class MaskVisualizationBlockV1(ColorableVisualizationBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.annotatorCache = {}

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return MaskManifest

    def getAnnotator(
        self,
        color_palette: str,
        palette_size: int,
        custom_colors: List[str],
        color_axis: str,
        opacity: float,
    ) -> sv.annotators.base.BaseAnnotator:
        key = "_".join(
            map(
                str,
                [
                    color_palette,
                    palette_size,
                    color_axis,
                    opacity,
                ],
            )
        )

        if key not in self.annotatorCache:
            palette = self.getPalette(color_palette, palette_size, custom_colors)

            self.annotatorCache[key] = sv.MaskAnnotator(
                color=palette,
                color_lookup=getattr(sv.ColorLookup, color_axis),
                opacity=opacity,
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
        opacity: Optional[float],
    ) -> BlockResult:
        # is_tensor_materialised(): never force a CHW tensor out of a
        # numpy-sourced image — that conversion is pure CPU overhead and the
        # sv path is faster on such frames.
        if _gpu_composite_eligible(predictions, color_axis) and (
            image.is_tensor_materialised()
        ):
            # Raises sv's ValueError when class/tracker ids are missing —
            # deliberately outside the try: the sv fallback would fail the
            # same way, only after a pointless mask materialisation.
            ids = _resolve_color_ids(predictions, color_axis)
            try:
                palette = self.getPalette(color_palette, palette_size, custom_colors)
                if not isinstance(palette, sv.ColorPalette):
                    raise TypeError("expected sv.ColorPalette")
                # Same colors sv's resolve_color_idx would pick:
                # palette.by_idx(class_id | det index | tracker_id).
                colors_rgb = np.asarray(
                    [palette.by_idx(int(idx)).as_rgb() for idx in ids],
                    dtype=np.uint8,
                )
                # Tensor pipeline contract: the image is a CHW RGB device
                # tensor — zero-copy in, tensor out (downstream materialises
                # numpy lazily only if something asks for it).
                scene_t = image.tensor_image
                if int(scene_t.shape[0]) != 3:
                    raise ValueError("GPU mask compositor requires a 3-channel image")
                if copy_image:
                    scene_t = scene_t.clone()
                annotated_tensor = gpu_mask_composite(
                    scene_t,
                    predictions,
                    colors_rgb,
                    float(opacity),
                )
                if not copy_image:
                    # The compositor mutated `image.tensor_image` storage
                    # in place (the sv-path contract for copy_image=False);
                    # invalidate the derived numpy/base64 caches.
                    image.declare_tensor_image_mutated()
                return {
                    OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                        origin_image_data=image, tensor_image=annotated_tensor
                    )
                }
            except Exception as gpu_error:
                logger.debug(
                    "GPU mask compositor failed (%s); falling back to "
                    "sv.MaskAnnotator path.",
                    gpu_error,
                )
        predictions = to_supervision_for_annotation(predictions)
        annotator = self.getAnnotator(
            color_palette,
            palette_size,
            custom_colors,
            color_axis,
            opacity,
        )

        scene = image.numpy_image
        if copy_image:
            scene = scene.copy()
        else:
            image.declare_numpy_image_mutated()
        annotated_image = annotator.annotate(
            scene=scene,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
