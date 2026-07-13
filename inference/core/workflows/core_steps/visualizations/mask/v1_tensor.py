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


def _dense_crops_and_offsets(mask, xyxy, img_h: int, img_w: int):
    """Build per-instance crop views from a dense ``(N, H, W)`` bool mask.

    Mirrors ``CompactMask.from_dense`` bit-for-bit: each box is clipped in the
    supervision ``xyxy`` convention (inclusive max coords) with
    ``x1c = clip(int(x1), 0, W-1)`` etc., and the crop is the zero-copy device
    view ``mask[i, y1c:y2c+1, x1c:x2c+1]``. A degenerate box (``x2c < x1c`` or
    ``y2c < y1c``) yields a ``None`` crop (painted as nothing — exactly what
    ``from_dense``'s 1x1 all-False crop does).

    Returns ``(crops, offsets)``: ``crops`` a list of bool tensor views
    (``None`` for degenerate boxes) and ``offsets`` a list of ``(x1c, y1c)``.
    """
    n = int(xyxy.shape[0])
    xyxy_cpu = xyxy.detach().cpu().numpy()
    crops: list = []
    offsets: list = []
    for i in range(n):
        x1, y1, x2, y2 = xyxy_cpu[i][0], xyxy_cpu[i][1], xyxy_cpu[i][2], xyxy_cpu[i][3]
        x1c = int(max(0, min(int(x1), img_w - 1)))
        y1c = int(max(0, min(int(y1), img_h - 1)))
        x2c = int(max(0, min(int(x2), img_w - 1)))
        y2c = int(max(0, min(int(y2), img_h - 1)))
        offsets.append((x1c, y1c))
        if x2c < x1c or y2c < y1c:
            crops.append(None)  # degenerate -> skip paint (from_dense parity)
            continue
        crops.append(mask[i, y1c : y2c + 1, x1c : x2c + 1])
    return crops, offsets


def gpu_mask_composite(
    scene,
    predictions: "InstanceDetections",
    colors_bgr: "np.ndarray",
    opacity: float,
    return_tensor: bool = False,
    scene_layout: str = "hwc_bgr",
):
    """GPU-native torch replacement for ``sv.MaskAnnotator.annotate``.

    Paints per-instance mask crops into an ownership index map in EXACTLY
    supervision's paint order (``np.flip(np.argsort(area))`` — largest first, so
    the smallest overlapping mask wins), then alpha-blends the per-pixel winning
    color with the scene in a single vectorised pass over the union ROI of all
    crops. This is the painter's-algorithm equivalence: because later paints
    overwrite earlier ones in both implementations and the paint order is
    identical (including stable-sort tie handling — equal areas painted
    higher-index-first so the LOWER index wins), the per-pixel winning color is
    identical, and so is the blend.

    Blend rounding: ``cv2.addWeighted`` saturate-casts with ``cvRound``
    (round-half-to-even); ``torch.round_`` matches it bit-for-bit for the exact
    half values produced by uint8 blends — all intermediates (<= 255.5) are
    exactly representable in fp16. Round-half-away (``(x + 0.5).floor()``)
    would diverge by 1 on ~half the odd sums. No clamp is needed: a convex
    combination of uint8 values stays in [0, 255].

    Two ``predictions.mask`` carriers feed the same paint/blend core:

    * ``InstancesRLEMasks`` GPU-crop side-channel — ``crop_masks_gpu`` (device
      bool tensors) at their ``crop_offsets``, with areas read from
      ``crop_rles`` when present. These fields do not exist on today's
      ``InstancesRLEMasks`` and are read via ``getattr``; the branch stays
      inert until a model-side change populates them.
    * dense ``torch.Tensor`` ``(N, H, W)`` bool on the pipeline device — per-instance crop
      views are built on device from ``predictions.xyxy``
      (``CompactMask.from_dense`` inclusive-clip convention) and areas are
      counted over those crop views (``crop.sum()``, batched into one D2H).
      The dense mask's True pixels are contained in its box, so this equals
      the full-mask pixel count — exactly ``sv.Detections.area`` for a dense
      mask — without a full-stack reduce.

    Args:
        scene: uint8 frame. With ``scene_layout="hwc_bgr"`` (default): an HWC
            BGR numpy array (uploaded to the mask device once) or an HWC BGR torch tensor.
            With ``scene_layout="chw_rgb"``: a CHW RGB torch tensor — the
            ``WorkflowImageData.tensor_image`` contract — composited without
            any layout conversion or host round-trip. A numpy scene is never
            mutated; a TENSOR scene IS mutated in place (callers that need the
            original must pass a clone).
        predictions: ``InstanceDetections`` whose ``mask`` is either an
            ``InstancesRLEMasks`` (``crop_masks_gpu`` + ``crop_offsets``) or a
            dense bool ``torch.Tensor`` ``(N, H, W)``.
        colors_bgr: ``(N, 3)`` uint8 per-detection colors (BGR), resolved with
            the same palette logic the sv annotator would use.
        opacity: overlay opacity, matches ``sv.MaskAnnotator(opacity=...)``.
        return_tensor: when True, return the device uint8 tensor (no download).
        scene_layout: ``"hwc_bgr"`` (default) or ``"chw_rgb"``. ``colors_bgr``
            stays BGR either way; the CHW path flips the LUT internally.

    Returns:
        uint8 numpy array in the input layout (same contract as ``annotate``),
        or the device tensor when ``return_tensor=True``.
    """
    mask_carrier = predictions.mask
    dense = isinstance(mask_carrier, torch.Tensor)
    if dense:
        device = mask_carrier.device
    else:
        crop_masks_gpu = getattr(mask_carrier, "crop_masks_gpu", None)
        crop_offsets = getattr(mask_carrier, "crop_offsets", None)
        crop_rles = getattr(mask_carrier, "crop_rles", None)
        device = crop_masks_gpu[0].device

    chw = scene_layout == "chw_rgb"
    if isinstance(scene, torch.Tensor):
        # No .contiguous() here: it could silently copy and break the in-place
        # mutation contract (copy_image=False operates on the caller's storage).
        scene_t = scene if scene.device == device else scene.to(device)
    else:
        if chw:
            raise ValueError("scene_layout='chw_rgb' requires a torch tensor scene")
        scene_t = torch.from_numpy(np.ascontiguousarray(scene)).to(device)
    # All painting/blending below works on an HWC view; for a CHW scene the
    # permute is a zero-copy view, so in-place writes land in the CHW storage.
    scene_hwc = scene_t.permute(1, 2, 0) if chw else scene_t

    H, W = int(scene_hwc.shape[0]), int(scene_hwc.shape[1])
    # Areas == mask pixel counts, which is what `sv.Detections.area` returns
    # for masked detections (dense or compact).
    if dense:
        # Dense carrier: crop views from xyxy (from_dense convention). Areas are
        # counted over the CROP views only — the True pixels of a dense mask are
        # contained in its box, so `crop.sum() == mask.sum()` exactly (same
        # ordering as `sv.Detections.area`), while touching only the box region
        # instead of one full pass over the (N, H, W) stack. The tiny per-crop
        # sums are batched into ONE (N,) D2H sync; a None (degenerate) crop
        # contributes area 0.
        mh, mw = int(mask_carrier.shape[1]), int(mask_carrier.shape[2])
        crops, offsets = _dense_crops_and_offsets(
            mask_carrier, predictions.xyxy, mh, mw
        )
        zero = torch.zeros((), dtype=torch.int64, device=device)
        areas = (
            torch.stack(
                [c.sum(dtype=torch.int64) if c is not None else zero for c in crops]
            )
            .cpu()
            .numpy()
        )
    else:
        crops = crop_masks_gpu
        offsets = crop_offsets
        n = len(crops)
        # Prefer the CPU-side crop RLE runs (odd-index runs are the True runs —
        # supervision CompactMask convention) to avoid per-crop GPU sum kernels
        # + a device sync.
        if crop_rles is not None and len(crop_rles) == n:
            areas = np.asarray(
                [int(np.asarray(runs)[1::2].sum()) for runs in crop_rles],
                dtype=np.int64,
            )
        else:
            areas = torch.stack([c.sum() for c in crops]).to(torch.int64).cpu().numpy()
    # Replicate supervision's `_paint_masks_by_area` order EXACTLY:
    # np.argsort is stable ascending; np.flip reverses -> descending area,
    # ties painted higher-index-first so the LOWER index wins (painted last).
    order = np.flip(np.argsort(areas))
    idx_map = torch.full((H, W), -1, dtype=torch.int16, device=device)
    ux1, uy1, ux2, uy2 = W, H, 0, 0
    for raw_i in order:
        i = int(raw_i)
        crop = crops[i]
        if crop is None:  # degenerate dense box -> paints nothing
            continue
        ch, cw = int(crop.shape[0]), int(crop.shape[1])
        x1, y1 = int(offsets[i][0]), int(offsets[i][1])
        x2, y2 = min(x1 + cw, W), min(y1 + ch, H)
        if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
            continue
        crop_mask = crop[: y2 - y1, : x2 - x1]
        if dense:
            # Dense crops are strided views into the (N, H, W) stack; a small
            # contiguous copy makes the masked_fill_ writes coalesce (the RLE
            # crops are already contiguous, so leave that path untouched).
            crop_mask = crop_mask.contiguous()
        # masked_fill_ = single fused kernel (no nonzero pass like `roi[m]=i`)
        idx_map[y1:y2, x1:x2].masked_fill_(crop_mask, i)
        ux1, uy1 = min(ux1, x1), min(uy1, y1)
        ux2, uy2 = max(ux2, x2), max(uy2, y2)
    if ux2 > ux1 and uy2 > uy1:
        # Pre-multiplied LUT: gather directly yields the color contribution
        # `color * opacity`, saving two full-ROI passes. The LUT channel order
        # must match the scene layout. fp16 on CUDA (all blend intermediates
        # <= 255.5 are exactly representable); fp32 on CPU, where half ops are
        # slow — identical results either way.
        blend_dtype = torch.float16 if device.type == "cuda" else torch.float32
        lut_colors = colors_bgr[:, ::-1] if chw else colors_bgr
        lut_premul = (
            torch.from_numpy(np.ascontiguousarray(lut_colors))
            .to(device=device, dtype=blend_dtype)
            .mul_(opacity)
        )  # (N, 3)
        sub = idx_map[uy1:uy2, ux1:ux2]
        hit = (sub >= 0).unsqueeze(-1)
        scene_roi = scene_hwc[uy1:uy2, ux1:ux2]
        blended = lut_premul[sub.clamp(min=0).long()]
        # fused axpy: blended += (1 - opacity) * scene  (single kernel)
        blended.add_(scene_roi.to(blend_dtype), alpha=1.0 - opacity)
        # torch.round_ = round-half-to-even, matching cv2.addWeighted's
        # cvRound (see docstring).
        blended_u8 = blended.round_().to(torch.uint8)
        # where() = one full-ROI write; boolean indexing would need 2 nonzero
        # passes + gather/scatter.
        scene_roi.copy_(torch.where(hit, blended_u8, scene_roi))

    if return_tensor:
        return scene_t
    return scene_t.cpu().numpy()


def _gpu_composite_eligible(predictions, color_axis: str) -> bool:
    """True when the torch compositor can replicate the sv path exactly."""
    if color_axis not in ("CLASS", "INDEX"):
        # TRACK / custom lookups keep the battle-tested sv path.
        return False
    if not isinstance(predictions, InstanceDetections):
        return False
    mask_carrier = getattr(predictions, "mask", None)
    n = int(predictions.xyxy.shape[0])
    if n == 0:
        # Nothing to paint; the sv path is a trivial no-op and avoids an
        # empty-crop edge case in the compositor.
        return False
    # Dense carrier: torch.Tensor (N, H, W) bool — the tensor pipeline's
    # default. Crop views are built from xyxy on whatever device the mask
    # lives on (the loader only registers this block for the tensor pipeline,
    # so device gating is not this block's job).
    if isinstance(mask_carrier, torch.Tensor):
        return (
            mask_carrier.ndim == 3
            and mask_carrier.dtype == torch.bool
            and int(mask_carrier.shape[0]) == n
        )
    # RLE carrier: needs the device-crop side-channel. Those fields do not
    # exist on today's InstancesRLEMasks (read via getattr), so this branch
    # stays inert until a model-side change populates them.
    if not isinstance(mask_carrier, InstancesRLEMasks):
        return False
    crops = getattr(mask_carrier, "crop_masks_gpu", None)
    offsets = getattr(mask_carrier, "crop_offsets", None)
    if not crops or not offsets:
        return False
    if len(crops) != n or len(offsets) != n:
        return False
    return all(isinstance(c, torch.Tensor) and c.dtype == torch.bool for c in crops)


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
        if _gpu_composite_eligible(predictions, color_axis):
            try:
                palette = self.getPalette(color_palette, palette_size, custom_colors)
                if not isinstance(palette, sv.ColorPalette):
                    raise TypeError("expected sv.ColorPalette")
                # Same colors sv's resolve_color would pick: CLASS ->
                # palette.by_idx(class_id), INDEX -> palette.by_idx(det index).
                if color_axis == "CLASS":
                    ids = predictions.class_id.detach().cpu().numpy().astype(int)
                else:
                    ids = np.arange(int(predictions.xyxy.shape[0]))
                colors_bgr = np.asarray(
                    [palette.by_idx(int(idx)).as_bgr() for idx in ids],
                    dtype=np.uint8,
                )
                # Composite in whichever representation the image already
                # carries — never force a cross-layout conversion (numpy->CHW
                # runs strided copies on the CPU; tensor->numpy is a device
                # round-trip).
                if image.is_tensor_materialised():
                    # Video path: CHW RGB tensor already on device — zero-copy
                    # in, tensor out (downstream materialises numpy lazily
                    # only if something asks for it).
                    scene_t = image.tensor_image
                    if int(scene_t.shape[0]) != 3:
                        raise ValueError(
                            "GPU mask compositor requires a 3-channel image"
                        )
                    if copy_image:
                        scene_t = scene_t.clone()
                    annotated_tensor = gpu_mask_composite(
                        scene_t,
                        predictions,
                        colors_bgr,
                        float(opacity),
                        return_tensor=True,
                        scene_layout="chw_rgb",
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
                # Numpy-resident image (HTTP path): upload the HWC BGR buffer
                # as-is and return numpy.
                annotated_image = gpu_mask_composite(
                    image.numpy_image,
                    predictions,
                    colors_bgr,
                    float(opacity),
                )
                if not copy_image:
                    # sv path mutates the input scene in place when
                    # copy_image=False; mirror that contract.
                    np.copyto(image.numpy_image, annotated_image)
                    annotated_image = image.numpy_image
                    image.declare_numpy_image_mutated()
                return {
                    OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                        origin_image_data=image, numpy_image=annotated_image
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

        annotated_image = annotator.annotate(
            scene=image.numpy_image.copy() if copy_image else image.numpy_image,
            detections=predictions,
        )
        return {
            OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                origin_image_data=image, numpy_image=annotated_image
            )
        }
