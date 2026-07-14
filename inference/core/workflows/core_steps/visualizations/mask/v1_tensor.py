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


def gpu_mask_composite(
    scene,
    predictions: "InstanceDetections",
    colors_bgr: "np.ndarray",
    opacity: float,
    return_tensor: bool = False,
    scene_layout: str = "hwc_bgr",
):
    """GPU-native torch replacement for ``sv.MaskAnnotator.annotate``.

    Supervision's painter's algorithm (``_paint_masks_by_area``) paints masks
    in ``np.flip(np.argsort(area))`` order — largest first — so the SMALLEST
    overlapping mask owns each pixel. Rather than replaying that paint loop,
    ownership is resolved in one vectorised kernel pair over the union ROI of
    all boxes:

        cost = where(mask, area_of_mask, +inf)   # (N, h, w)
        owner = cost.argmin(dim=0)               # smallest area wins

    Tie handling matches too: for equal areas ``argmin`` returns the FIRST
    (lowest) index, and in supervision's stable flip-sort equal areas are
    painted higher-index-first so the lower index is painted last and wins.
    The winning color per pixel is therefore identical to the sv annotator's,
    and so is the blend. (A loop-free "blend-all" variant — averaging every
    overlapping mask's color instead of resolving an owner — measures ~1.5 ms
    faster on Jetson Orin Nano but changes overlap colors and breaks sv
    parity; rejected.)

    Blend rounding: ``cv2.addWeighted`` saturate-casts with ``cvRound``
    (round-half-to-even); ``torch.round_`` matches it bit-for-bit for the exact
    half values produced by uint8 blends — all intermediates (<= 255.5) are
    exactly representable in fp16. Round-half-away (``(x + 0.5).floor()``)
    would diverge by 1 on ~half the odd sums. No clamp is needed: a convex
    combination of uint8 values stays in [0, 255].

    ``predictions.mask`` must be a dense ``torch.Tensor`` ``(N, H, W)`` bool on
    the pipeline device. Its True pixels are assumed to lie inside the
    detection's box (the tensor pipeline decodes masks per-box), so areas
    summed over the union ROI equal the full-mask pixel counts — exactly
    ``sv.Detections.area``.

    Args:
        scene: uint8 frame. With ``scene_layout="hwc_bgr"`` (default): an HWC
            BGR numpy array (uploaded to the mask device once) or an HWC BGR torch tensor.
            With ``scene_layout="chw_rgb"``: a CHW RGB torch tensor — the
            ``WorkflowImageData.tensor_image`` contract — composited without
            any layout conversion or host round-trip. A numpy scene is never
            mutated; a TENSOR scene IS mutated in place (callers that need the
            original must pass a clone).
        predictions: ``InstanceDetections`` whose ``mask`` is a dense bool
            ``torch.Tensor`` ``(N, H, W)``.
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
    device = mask_carrier.device

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
    # Union ROI of all boxes — one tiny D2H of xyxy (the only host sync).
    # supervision xyxy has inclusive max coords, hence the +1.
    xy = predictions.xyxy.detach().cpu().numpy()
    ux1 = max(0, int(np.floor(xy[:, 0].min())))
    uy1 = max(0, int(np.floor(xy[:, 1].min())))
    ux2 = min(W, int(np.floor(xy[:, 2].max())) + 1)
    uy2 = min(H, int(np.floor(xy[:, 3].max())) + 1)
    if ux2 > ux1 and uy2 > uy1:
        m = mask_carrier[:, uy1:uy2, ux1:ux2]
        # Areas over the ROI == full-mask pixel counts (True pixels live inside
        # the boxes) == sv.Detections.area. fp32: counts can exceed fp16 max.
        areas = m.sum(dim=(1, 2)).to(torch.float32)
        # Ownership resolved in one kernel pair (see docstring): smallest-area
        # mask wins each pixel; argmin's first-index tie rule matches sv's
        # stable flip-sort ties.
        cost = torch.where(
            m, areas.view(-1, 1, 1), torch.tensor(float("inf"), device=device)
        )
        owner = cost.argmin(dim=0)  # (h, w)
        hit = m.any(dim=0).unsqueeze(-1)
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
        scene_roi = scene_hwc[uy1:uy2, ux1:ux2]
        blended = lut_premul[owner]
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
    # Dense carrier only: torch.Tensor (N, H, W) bool — the tensor pipeline's
    # default. RLE-carrier predictions take the sv fallback. Crop views are
    # built from xyxy on whatever device the mask lives on (the loader only
    # registers this block for the tensor pipeline, so device gating is not
    # this block's job).
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
