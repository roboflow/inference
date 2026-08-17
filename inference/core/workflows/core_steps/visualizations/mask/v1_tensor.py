from typing import List, Literal, Optional, Type, Union

import numpy as np
import supervision as sv
import torch
from pydantic import ConfigDict, Field

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
    empty_predictions_passthrough,
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

#: supervision's pending-track sentinel and its color (``PENDING_TRACK_ID`` /
#: ``PENDING_TRACK_COLOR = Color.GREY``, supervision 0.29.0
#: ``annotators/utils.py``). Values are copied so the runtime path never
#: touches supervision.
_PENDING_TRACK_ID = -1
_PENDING_TRACK_COLOR_RGB = (128, 128, 128)

_SUPPORTED_COLOR_AXES = ("CLASS", "INDEX", "TRACK")


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


def _rle_to_dense_masks(
    masks: "InstancesRLEMasks", device: torch.device
) -> torch.Tensor:
    """Decode COCO-RLE payloads into a dense bool ``(N, H, W)`` mask stack on
    ``device``, feeding the same fixed-shape composite the dense carrier uses.

    The RLE bytes live on the host, so the varint decode is inherently host
    work; the per-run tables cross the bus once (H2D upload — it does not
    drain the device queue) and the run→pixel expansion plus the scatter into
    the dense stack happen on the device with host-known sizes: no
    device→host sync anywhere.
    """
    height, width = (int(size) for size in masks.image_size)
    n = len(masks.masks)
    flat_idx, det_ids = _rle_foreground_pixels_in_roi(
        masks, (0, 0, height, width), device
    )
    dense = torch.zeros(n * height * width, dtype=torch.bool, device=device)
    dense[det_ids * (height * width) + flat_idx] = True
    return dense.view(n, height, width)


def gpu_mask_composite(
    scene: torch.Tensor,
    mask: Union[torch.Tensor, "InstancesRLEMasks", np.ndarray],
    colors_rgb: Union[torch.Tensor, np.ndarray],
    opacity: float,
) -> torch.Tensor:
    """Torch-native mask compositor replicating ``sv.MaskAnnotator.annotate``
    with a ZERO-SYNC dense hot path.

    ``scene`` is a CHW RGB uint8 torch tensor (``WorkflowImageData``'s
    ``tensor_image`` layout), mutated IN PLACE and returned. The function is
    device-agnostic: the same code runs on CUDA tensors (the tensor pipeline)
    and CPU tensors (the numpy-image path of the block).

    Sync-freedom contract (the whole point of this block): for a dense bool
    ``(N, H, W)`` mask on the scene's device, the composite enqueues a FIXED,
    N-independent number of device ops and never reads a device value back to
    the host — no ``.cpu()``, no ``.item()``, no ``nonzero``, no data-dependent
    shapes. A sync-free viz phase queues a whole batch back-to-back instead of
    serialising each image against the previous one's kernels (measured ~35 ms
    of pure queue-wait per batch on Jetson NX before this rewrite). There is
    deliberately NO ROI narrowing (the old union-of-boxes crop needed an
    ``xyxy`` device→host read) and NO large-N scatter branch: full-frame fp32
    bandwidth at realistic N (a few dozen instances) is far cheaper than one
    default-stream sync, and the GEMM formulation below keeps the traffic at
    one bool read of the mask stack plus a fixed number of full-frame fp32
    passes.

    Compositing math — identical to the previous revision, kept on purpose:

        count      = Σ_i mask_i                       # (H·W,)
        color_sum  = colorsᵀ·opacity @ masks          # (3, H·W) — one GEMM
        out        = round(color_sum / count + (1-opacity) · scene)  where count > 0

    Overlap semantics are intentionally simpler than supervision's: every mask
    covering a pixel contributes equally (MEAN of the covering colors,
    alpha-composited once), which is order-independent and diverges from
    supervision's smallest-area-on-top painter's algorithm on OVERLAPPING
    pixels only. Pixels covered by exactly one mask match ``sv.MaskAnnotator``
    bit-for-bit: same premultiplied blend, and ``torch.round_`` is
    round-half-to-even like ``cv2.addWeighted``'s ``cvRound``. A convex
    combination of uint8 values needs no clamp. fp32 accumulation is exact for
    any realistic N (counts and premultiplied sums stay far below 2^24) and
    the GEMM is deterministic — note it assumes the default
    ``torch.backends.cuda.matmul.allow_tf32 = False``; enabling TF32 globally
    may shift overlap pixels by ±1.

    Accepted mask carriers:

    * dense bool ``torch.Tensor`` ``(N, H, W)`` — the zero-sync hot path
      (non-bool dtypes are cast on device; a carrier on another device is
      moved to the scene's device, which is only ever paid on the host-bound
      numpy-image path);
    * ``np.ndarray`` ``(N, H, W)`` — uploaded once, then the dense path;
    * ``InstancesRLEMasks`` (COCO compressed RLE, column-major — the SAM3 /
      semantic-segmentation carrier) — decoded host-side (inherent: the bytes
      live on the host) and scattered into a dense stack on device, then the
      identical fixed-shape composite. No device→host sync either.

    All validation happens before any write; the single in-place write to
    ``scene`` is staged last, so a raising call never leaves a partially
    painted image.

    Args:
        scene: CHW RGB uint8 torch tensor, any device. Mutated in place —
            callers that need the original must pass a clone.
        mask: one of the carriers above; the mask canvas must match the scene.
        colors_rgb: ``(N, 3)`` uint8 per-detection RGB colors (device tensor
            preferred; numpy is uploaded).
        opacity: overlay opacity, matches ``sv.MaskAnnotator(opacity=...)``.

    Returns:
        ``scene`` (same tensor, annotated in place).
    """
    if scene.ndim != 3 or int(scene.shape[0]) != 3:
        raise ValueError(
            "mask visualization requires a 3-channel CHW RGB scene tensor, "
            f"got shape {tuple(scene.shape)}"
        )
    height, width = int(scene.shape[1]), int(scene.shape[2])
    if isinstance(mask, InstancesRLEMasks):
        canvas = tuple(int(size) for size in mask.image_size)
        if canvas != (height, width):
            raise ValueError(
                f"mask canvas {canvas} does not match scene {(height, width)}"
            )
        mask = _rle_to_dense_masks(mask, scene.device)
    else:
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(np.ascontiguousarray(mask))
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
            raise ValueError(
                "mask visualization requires a dense (N, H, W) mask tensor, "
                "a (N, H, W) numpy array or InstancesRLEMasks, got "
                f"{type(mask).__name__}"
            )
        canvas = (int(mask.shape[1]), int(mask.shape[2]))
        if canvas != (height, width):
            raise ValueError(
                f"mask canvas {canvas} does not match scene {(height, width)}"
            )
        if mask.device != scene.device:
            # Only reachable on the host-bound numpy-image path (device masks
            # with a CPU scene); the tensor pipeline keeps everything on one
            # device and never dispatches a device change here.
            mask = mask.to(scene.device)
        if mask.dtype != torch.bool:
            mask = mask != 0
    if isinstance(colors_rgb, np.ndarray):
        colors_rgb = torch.from_numpy(np.ascontiguousarray(colors_rgb)).to(scene.device)
    n = int(mask.shape[0])
    if int(colors_rgb.shape[0]) != n:
        raise ValueError(
            f"got {int(colors_rgb.shape[0])} colors for {n} masks — one RGB "
            "color per detection is required"
        )
    pixels = height * width
    masks_flat = mask.reshape(n, pixels).to(torch.float32)  # (N, P)
    count = masks_flat.sum(dim=0)  # (P,)
    hit = (count > 0).unsqueeze(0)  # (1, P)
    colors_premul = colors_rgb.to(dtype=torch.float32).mul_(float(opacity))  # (N, 3)
    # One deterministic GEMM instead of an (N, 3, H, W) broadcast or an
    # index_add scatter: (3, N) @ (N, P) → the premultiplied color sum per
    # pixel, with no giant intermediate and no data-dependent indexing.
    color_sum = colors_premul.t() @ masks_flat  # (3, P)
    scene_flat = scene.reshape(3, pixels)
    blended = color_sum.div_(count.clamp_(min=1.0))
    blended = blended.add_(scene_flat.to(torch.float32), alpha=1.0 - float(opacity))
    blended_u8 = blended.round_().to(torch.uint8)
    # Single staged write: `where` materialises the full output first, then
    # one in-place copy lands it in the caller's storage.
    scene.copy_(torch.where(hit, blended_u8, scene_flat).view(3, height, width))
    return scene


def _resolve_color_ids(
    predictions: "InstanceDetections",
    color_axis: str,
    device: torch.device,
) -> torch.Tensor:
    """Palette ids as an ``(N,)`` int64 tensor on ``device`` — the same ids
    supervision's ``resolve_color_idx`` would produce, without any
    device→host sync:

    * ``INDEX`` → ``arange(n)`` built on device (``n`` comes from
      ``xyxy.shape[0]`` — shape metadata, no sync);
    * ``CLASS`` → ``class_id`` stays a device tensor (dtype cast only, never a
      ``.cpu()``);
    * ``TRACK`` → tracker ids read from the host-side ``bboxes_metadata``
      dicts and uploaded H2D (uploads do not drain the device queue). The
      pending-track sentinel ``-1`` is passed through for the caller to map to
      supervision's gray.

    Missing ids raise ``ValueError`` BEFORE any mask work, so a doomed run
    never pays a mask decode first.
    """
    n = int(predictions.xyxy.shape[0])
    if color_axis == "INDEX":
        return torch.arange(n, dtype=torch.int64, device=device)
    if color_axis == "CLASS":
        class_id = predictions.class_id
        if class_id is None:
            raise ValueError(
                "Could not resolve color by class because "
                "Detections do not have class_id. If using an annotator, "
                "try setting color_lookup to sv.ColorLookup.INDEX or "
                "sv.ColorLookup.TRACK."
            )
        return class_id.detach().to(device=device, dtype=torch.int64)
    if color_axis == "TRACK":
        metadata = predictions.bboxes_metadata or []
        tracker_ids = [box.get("tracker_id") for box in metadata]
        if len(tracker_ids) != n or any(
            tracker_id is None for tracker_id in tracker_ids
        ):
            raise ValueError(
                "Could not resolve color by track because "
                "Detections do not have tracker_id. Did you call "
                "tracker.update_with_detections(...) before annotating?"
            )
        return torch.tensor(
            [int(tracker_id) for tracker_id in tracker_ids],
            dtype=torch.int64,
            device=device,
        )
    raise ValueError(
        f"mask visualization supports color_axis in {_SUPPORTED_COLOR_AXES}, "
        f"got {color_axis!r}"
    )


def _validate_inputs(predictions, color_axis: str) -> None:
    """Raise a clear ``ValueError`` for inputs the torch compositor cannot
    paint — the previous silent supervision fallback is gone, so invalid
    inputs now fail loudly instead of quietly taking a different code path."""
    if color_axis not in _SUPPORTED_COLOR_AXES:
        raise ValueError(
            f"mask visualization supports color_axis in {_SUPPORTED_COLOR_AXES}, "
            f"got {color_axis!r}"
        )
    if not isinstance(predictions, InstanceDetections):
        raise ValueError(
            "mask visualization requires instance segmentation predictions "
            f"(InstanceDetections with masks), got {type(predictions).__name__}"
        )
    n = int(predictions.xyxy.shape[0])
    mask = predictions.mask
    if isinstance(mask, InstancesRLEMasks):
        if len(mask.masks) != n:
            raise ValueError(
                f"predictions carry {len(mask.masks)} RLE masks for {n} boxes — "
                "one mask per detection is required"
            )
    elif isinstance(mask, (torch.Tensor, np.ndarray)):
        if mask.ndim != 3 or int(mask.shape[0]) != n:
            raise ValueError(
                "predictions must carry a dense (N, H, W) mask stack with one "
                f"mask per detection; got shape {tuple(mask.shape)} for {n} boxes"
            )
    else:
        raise ValueError(
            "predictions carry no usable mask (expected a dense (N, H, W) "
            "tensor/array or InstancesRLEMasks, got "
            f"{type(mask).__name__}) — mask visualization requires "
            "segmentation masks"
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
        # (color_palette, palette_size, custom_colors, device) → (P, 3) uint8
        # RGB LUT. supervision's ColorPalette is consulted ONCE here, at
        # cache-build time (pure configuration); the per-frame tensor path is
        # supervision-free.
        self._palette_lut_cache = {}
        # sv.MaskAnnotator cache for the numpy-sourced-image path.
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

    def _get_palette_lut(
        self,
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            color_palette,
            int(palette_size) if palette_size is not None else None,
            tuple(custom_colors or ()),
            str(device),
        )
        lut = self._palette_lut_cache.get(key)
        if lut is None:
            palette = self.getPalette(color_palette, palette_size, custom_colors)
            palette_colors = getattr(palette, "colors", None)
            if not palette_colors:
                raise ValueError(
                    f"color palette {color_palette!r} did not resolve to a "
                    "color palette with at least one color"
                )
            lut = torch.tensor(
                [color.as_rgb() for color in palette_colors],
                dtype=torch.uint8,
                device=device,
            )
            self._palette_lut_cache[key] = lut
        return lut

    def _annotate_scene(
        self,
        scene: torch.Tensor,
        predictions: "InstanceDetections",
        color_palette: Optional[str],
        palette_size: Optional[int],
        custom_colors: Optional[List[str]],
        color_axis: str,
        opacity: float,
    ) -> torch.Tensor:
        device = scene.device
        ids = _resolve_color_ids(predictions, color_axis, device)
        lut = self._get_palette_lut(color_palette, palette_size, custom_colors, device)
        # Same color sv's `by_idx` picks: idx % palette size, on device. (For
        # negative ids torch's remainder wraps instead of raising like sv's
        # by_idx — checking would require a device→host read.)
        colors_rgb = lut[ids.remainder(int(lut.shape[0]))]
        if color_axis == "TRACK":
            # sv resolve_color: the pending-track id (-1) gets Color.GREY.
            pending = torch.tensor(
                _PENDING_TRACK_COLOR_RGB, dtype=torch.uint8, device=device
            )
            colors_rgb = torch.where(
                (ids == _PENDING_TRACK_ID).unsqueeze(1), pending, colors_rgb
            )
        return gpu_mask_composite(scene, predictions.mask, colors_rgb, float(opacity))

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
        passthrough = empty_predictions_passthrough(
            image=image, detections=predictions, copy_image=copy_image
        )
        if passthrough is not None:
            return passthrough
        if image.is_tensor_materialised():
            _validate_inputs(predictions, color_axis)
            # Tensor pipeline contract: CHW RGB device tensor in, tensor out —
            # zero device→host syncs on the dense path (downstream materialises
            # numpy lazily only if something asks for it).
            scene = image.tensor_image
            if copy_image:
                scene = scene.clone()
            annotated = self._annotate_scene(
                scene,
                predictions,
                color_palette,
                palette_size,
                custom_colors,
                color_axis,
                opacity,
            )
            if not copy_image:
                # The compositor mutated `image.tensor_image` storage in
                # place; invalidate the derived numpy/base64 caches.
                image.declare_tensor_image_mutated()
            return {
                OUTPUT_IMAGE_KEY: WorkflowImageData.copy_and_replace(
                    origin_image_data=image, tensor_image=annotated
                )
            }
        # Numpy-sourced image (flag-on cv2-fallback frames): behave EXACTLY as
        # before — the battle-tested sv.MaskAnnotator path, byte-identical to
        # the numpy v1 block. Forcing a CHW tensor out of such a frame would be
        # pure host-side conversion overhead.
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
