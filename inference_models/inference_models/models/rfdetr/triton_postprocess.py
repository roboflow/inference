"""Fused RF-DETR post-processing kernels (Triton + torch.compile fallback).

Fuses sigmoid + max-reduce + cxcywh→xyxy + denormalize + rescale into a
single GPU kernel launch, eliminating the ~25us Python→CUDA dispatch gaps
that dominate when these operations are expressed as separate PyTorch calls.

Each Triton program instance processes one of the 300 object queries:
  - Loads 91 logit values, applies sigmoid, finds max confidence + class ID
  - Loads 4 box coordinates (cx, cy, w, h in [0,1])
  - Converts to xyxy pixel coordinates with denormalization and rescale
  - Writes 6 floats: [confidence, class_id, x1, y1, x2, y2]

The caller does a single bulk D2H copy of the 300×6 output, then applies
threshold + sort + class remapping on CPU (trivial on ≤300 elements).
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def fused_postprocess_kernel(
    logits_ptr,        # [num_queries, num_classes] float32
    bboxes_ptr,        # [num_queries, 4] float32 (cx, cy, w, h normalised)
    output_ptr,        # [num_queries, 6] float32 (conf, cls, x1, y1, x2, y2)
    num_classes: tl.constexpr,  # 91
    # Rescale parameters (passed as scalars to avoid any extra memory traffic)
    dw,                # denorm width (float)
    dh,                # denorm height (float)
    inv_sw,            # 1.0 / scale_width (float)
    inv_sh,            # 1.0 / scale_height (float)
    pad_l,             # pad_left (float)
    pad_t,             # pad_top (float)
    crop_x,            # static_crop_offset.offset_x (float)
    crop_y,            # static_crop_offset.offset_y (float)
    BLOCK_C: tl.constexpr,  # next-power-of-2 >= num_classes (128 for 91)
):
    query_id = tl.program_id(0)

    # --- sigmoid + max reduction over classes ---
    cls_offsets = tl.arange(0, BLOCK_C)
    mask = cls_offsets < num_classes
    logit_ptrs = logits_ptr + query_id * num_classes + cls_offsets
    logits = tl.load(logit_ptrs, mask=mask, other=-float('inf'))

    # sigmoid: 1 / (1 + exp(-x))
    probs = 1.0 / (1.0 + tl.exp(-logits))

    # max + argmax
    confidence = tl.max(probs, axis=0)
    class_id = tl.argmax(probs, axis=0)

    # --- box transform: cxcywh → xyxy + denorm + rescale ---
    cx = tl.load(bboxes_ptr + query_id * 4 + 0)
    cy = tl.load(bboxes_ptr + query_id * 4 + 1)
    w  = tl.load(bboxes_ptr + query_id * 4 + 2)
    h  = tl.load(bboxes_ptr + query_id * 4 + 3)

    half_w = w * 0.5
    half_h = h * 0.5

    # cxcywh → xyxy (normalised) → denormalise → remove padding → undo scale → add crop
    x1 = ((cx - half_w) * dw - pad_l) * inv_sw + crop_x
    y1 = ((cy - half_h) * dh - pad_t) * inv_sh + crop_y
    x2 = ((cx + half_w) * dw - pad_l) * inv_sw + crop_x
    y2 = ((cy + half_h) * dh - pad_t) * inv_sh + crop_y

    # --- store output: [confidence, class_id, x1, y1, x2, y2] ---
    out_base = output_ptr + query_id * 6
    tl.store(out_base + 0, confidence)
    tl.store(out_base + 1, class_id.to(tl.float32))
    tl.store(out_base + 2, x1)
    tl.store(out_base + 3, y1)
    tl.store(out_base + 4, x2)
    tl.store(out_base + 5, y2)


def launch_fused_postprocess(
    logits: torch.Tensor,
    bboxes: torch.Tensor,
    output: torch.Tensor,
    num_classes: int,
    dw: float,
    dh: float,
    inv_sw: float,
    inv_sh: float,
    pad_l: float,
    pad_t: float,
    crop_x: float,
    crop_y: float,
) -> None:
    """Launch the Triton kernel for one image."""
    num_queries = logits.shape[0]
    BLOCK_C = triton.next_power_of_2(num_classes)
    fused_postprocess_kernel[(num_queries,)](
        logits, bboxes, output,
        num_classes=num_classes,
        dw=dw, dh=dh, inv_sw=inv_sw, inv_sh=inv_sh,
        pad_l=pad_l, pad_t=pad_t, crop_x=crop_x, crop_y=crop_y,
        BLOCK_C=BLOCK_C,
    )


@torch.compile(fullgraph=True)
def compiled_fused_postprocess(
    logits: torch.Tensor,
    bboxes: torch.Tensor,
    dw: float,
    dh: float,
    inv_sw: float,
    inv_sh: float,
    pad_l: float,
    pad_t: float,
    crop_x: float,
    crop_y: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """torch.compile fallback: fused sigmoid+max+box transform.

    Lets Inductor fuse the elementwise + reduction ops into fewer kernels
    than eager PyTorch would produce. Returns (confidence, class_ids, xyxy)
    for all queries — caller filters on CPU.
    """
    probs = logits.sigmoid()
    confidence, class_ids = probs.max(dim=1)
    cx = bboxes[:, 0]
    cy = bboxes[:, 1]
    half_w = bboxes[:, 2] * 0.5
    half_h = bboxes[:, 3] * 0.5
    x1 = ((cx - half_w) * dw - pad_l) * inv_sw + crop_x
    y1 = ((cy - half_h) * dh - pad_t) * inv_sh + crop_y
    x2 = ((cx + half_w) * dw - pad_l) * inv_sw + crop_x
    y2 = ((cy + half_h) * dh - pad_t) * inv_sh + crop_y
    xyxy = torch.stack([x1, y1, x2, y2], dim=1)
    return confidence, class_ids, xyxy
