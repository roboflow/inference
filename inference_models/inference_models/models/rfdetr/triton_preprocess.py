"""Fused Triton preprocessing kernel for RF-DETR.

Byte-exact port of PIL's separable bilinear-antialias resize (the algorithm
torchvision's `TF.resize(pil, ..., antialias=True)` uses on PIL inputs), with
the subsequent `/255` + ImageNet normalize fused into the same pass.

PIL's scheme (src/libImaging/Resample.c):

    PRECISION_BITS = 22
    scale       = in_size / out_size
    filterscale = max(1.0, scale)
    support     = 1.0 * filterscale          # triangle radius = 1
    ksize       = ceil(support) * 2 + 1
    center(o)   = (o + 0.5) * scale
    xmin(o)     = int(center - support + 0.5)                  clipped to [0, in]
    xmax(o)     = int(center + support + 0.5)                  clipped to [0, in]
    w_f(o, k)   = triangle((k + xmin - center + 0.5) / filterscale)
    w_f normalised to sum to 1 per output pixel
    w_i(o, k)   = round(w_f(o, k) * (1 << PRECISION_BITS))      int32
    out(o)      = clamp((Σ w_i(o, k) * src_u8) + (1 << (PRECISION_BITS-1)) >> PRECISION_BITS, 0, 255)

Single fused kernel: the horizontal uint8 intermediate lives in registers
rather than a DRAM scratch buffer. For each output tile we loop over
KSIZE_Y source rows; for each contributing source row we recompute the
horizontal convolution (int32 fixed-point, uint8 quantize) on the fly,
multiply by the vertical weight, and accumulate. Final: uint8 quantize,
BGR↔RGB swap, /255, ImageNet normalize, fp32 CHW store.

A separable two-pass variant (horizontal then vertical, via a DRAM uint8
intermediate) is ~0.4 fps faster end-to-end on the 312² RF-DETR workload
because it avoids redoing KSIZE_X MACs per output row. We picked the fused
version for simplicity (no intermediate buffer, one launch, one piece of
math).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


PRECISION_BITS = 22


def _bilinear_antialias_weights_1d_int(
    in_size: int, out_size: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """PIL's precompute_coeffs, int32 fixed-point form."""
    scale = in_size / out_size
    filterscale = max(1.0, scale)
    support = filterscale
    ksize = int(math.ceil(support)) * 2 + 1

    starts = np.zeros(out_size, dtype=np.int32)
    weights_fp = np.zeros((out_size, ksize), dtype=np.float64)
    inv_fs = 1.0 / filterscale

    for o in range(out_size):
        center = (o + 0.5) * scale
        xmin = int(center - support + 0.5)
        if xmin < 0:
            xmin = 0
        xmax = int(center + support + 0.5)
        if xmax > in_size:
            xmax = in_size
        actual = xmax - xmin
        starts[o] = xmin
        total = 0.0
        for k in range(actual):
            t = (k + xmin - center + 0.5) * inv_fs
            t_abs = -t if t < 0.0 else t
            w = 1.0 - t_abs if t_abs < 1.0 else 0.0
            weights_fp[o, k] = w
            total += w
        if total != 0.0:
            weights_fp[o, :actual] /= total

    weights_int = np.rint(weights_fp * (1 << PRECISION_BITS)).astype(np.int32)
    return starts, weights_int, ksize


if TRITON_AVAILABLE:

    _HALF = 1 << (PRECISION_BITS - 1)

    @triton.jit
    def _fused_resize_normalize_kernel(
        src_ptr,
        dst_ptr,
        ymin_ptr,
        xmin_ptr,
        wy_ptr,
        wx_ptr,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        dst_stride_c,
        dst_stride_h,
        target_h,
        target_w,
        inv_std_255_r,
        inv_std_255_g,
        inv_std_255_b,
        offset_r,
        offset_g,
        offset_b,
        CH_R: tl.constexpr,
        CH_G: tl.constexpr,
        CH_B: tl.constexpr,
        KSIZE_Y: tl.constexpr,
        KSIZE_X: tl.constexpr,
        PRECISION_BITS_C: tl.constexpr,
        HALF_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """One kernel per (tile_y, tile_x) over target image.

        In : src uint8 HWC (src_h, src_w, 3), source color order.
        Out: dst fp32 CHW (1, 3, target_h, target_w), network color order,
             (pixel/255 - mean)/std.
        """
        pid_y = tl.program_id(0)
        pid_x = tl.program_id(1)

        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_y = offs_y < target_h
        mask_x = offs_x < target_w
        mask_out = mask_y[:, None] & mask_x[None, :]

        ymin = tl.load(ymin_ptr + offs_y, mask=mask_y, other=0)
        xmin = tl.load(xmin_ptr + offs_x, mask=mask_x, other=0)

        # Vertical pass accumulators (int32 fixed-point) for 3 channels.
        vacc_0 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
        vacc_1 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
        vacc_2 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

        for ky in tl.static_range(KSIZE_Y):
            # Source row contributing to each output row in this tile.
            sy = ymin + ky
            sy_c = tl.maximum(tl.minimum(sy, src_h - 1), 0)
            wy = tl.load(wy_ptr + offs_y * KSIZE_Y + ky, mask=mask_y, other=0)

            # Horizontal pass for (output_rows_in_tile, output_cols_in_tile):
            # for each source column in the kernel, gather src[sy_c, sx_c, :]
            # and accumulate with wx[output_col, kx].
            hacc_0 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
            hacc_1 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
            hacc_2 = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)

            for kx in tl.static_range(KSIZE_X):
                sx = xmin + kx
                sx_c = tl.maximum(tl.minimum(sx, src_w - 1), 0)
                wx = tl.load(wx_ptr + offs_x * KSIZE_X + kx, mask=mask_x, other=0)
                base = sy_c[:, None] * src_stride_h + sx_c[None, :] * src_stride_w
                p0 = tl.load(src_ptr + base + 0, mask=mask_out, other=0).to(tl.int32)
                p1 = tl.load(src_ptr + base + 1, mask=mask_out, other=0).to(tl.int32)
                p2 = tl.load(src_ptr + base + 2, mask=mask_out, other=0).to(tl.int32)
                wx_2d = wx[None, :]
                hacc_0 += p0 * wx_2d
                hacc_1 += p1 * wx_2d
                hacc_2 += p2 * wx_2d

            # Horizontal uint8 quantization (byte-exact to PIL's intermediate).
            hacc_0 = (hacc_0 + HALF_C) >> PRECISION_BITS_C
            hacc_1 = (hacc_1 + HALF_C) >> PRECISION_BITS_C
            hacc_2 = (hacc_2 + HALF_C) >> PRECISION_BITS_C
            hacc_0 = tl.minimum(tl.maximum(hacc_0, 0), 255)
            hacc_1 = tl.minimum(tl.maximum(hacc_1, 0), 255)
            hacc_2 = tl.minimum(tl.maximum(hacc_2, 0), 255)

            wy_2d = wy[:, None]
            vacc_0 += hacc_0 * wy_2d
            vacc_1 += hacc_1 * wy_2d
            vacc_2 += hacc_2 * wy_2d

        # Vertical uint8 quantization.
        q_0 = (vacc_0 + HALF_C) >> PRECISION_BITS_C
        q_1 = (vacc_1 + HALF_C) >> PRECISION_BITS_C
        q_2 = (vacc_2 + HALF_C) >> PRECISION_BITS_C
        q_0 = tl.minimum(tl.maximum(q_0, 0), 255)
        q_1 = tl.minimum(tl.maximum(q_1, 0), 255)
        q_2 = tl.minimum(tl.maximum(q_2, 0), 255)

        # Source-to-output channel remap (triton requires constexpr branches).
        if CH_R == 0:
            q_r = q_0
        elif CH_R == 1:
            q_r = q_1
        else:
            q_r = q_2
        if CH_G == 0:
            q_g = q_0
        elif CH_G == 1:
            q_g = q_1
        else:
            q_g = q_2
        if CH_B == 0:
            q_b = q_0
        elif CH_B == 1:
            q_b = q_1
        else:
            q_b = q_2

        # (pixel/255 - mean)/std  ==  pixel * (1/(255*std)) + (-mean/std)
        out_r = q_r.to(tl.float32) * inv_std_255_r + offset_r
        out_g = q_g.to(tl.float32) * inv_std_255_g + offset_g
        out_b = q_b.to(tl.float32) * inv_std_255_b + offset_b

        out_row = offs_y[:, None] * dst_stride_h + offs_x[None, :]
        tl.store(dst_ptr + 0 * dst_stride_c + out_row, out_r, mask=mask_out)
        tl.store(dst_ptr + 1 * dst_stride_c + out_row, out_g, mask=mask_out)
        tl.store(dst_ptr + 2 * dst_stride_c + out_row, out_b, mask=mask_out)


class _ResampleTables:
    """Cache of per-axis PIL-int32 weight tables for one (src, dst) pair."""

    __slots__ = (
        "ymin_gpu",
        "xmin_gpu",
        "wy_gpu",
        "wx_gpu",
        "ksize_y",
        "ksize_x",
    )

    def __init__(
        self,
        ymin_gpu: torch.Tensor,
        xmin_gpu: torch.Tensor,
        wy_gpu: torch.Tensor,
        wx_gpu: torch.Tensor,
        ksize_y: int,
        ksize_x: int,
    ) -> None:
        self.ymin_gpu = ymin_gpu
        self.xmin_gpu = xmin_gpu
        self.wy_gpu = wy_gpu
        self.wx_gpu = wx_gpu
        self.ksize_y = ksize_y
        self.ksize_x = ksize_x


def build_resample_tables(
    src_h: int,
    src_w: int,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> _ResampleTables:
    ymin, wy, ksize_y = _bilinear_antialias_weights_1d_int(src_h, target_h)
    xmin, wx, ksize_x = _bilinear_antialias_weights_1d_int(src_w, target_w)
    return _ResampleTables(
        ymin_gpu=torch.from_numpy(ymin).to(device=device, non_blocking=True),
        xmin_gpu=torch.from_numpy(xmin).to(device=device, non_blocking=True),
        wy_gpu=torch.from_numpy(wy.ravel()).to(device=device, non_blocking=True),
        wx_gpu=torch.from_numpy(wx.ravel()).to(device=device, non_blocking=True),
        ksize_y=ksize_y,
        ksize_x=ksize_x,
    )


def triton_preprocess_rfdetr_stretch(
    src: torch.Tensor,
    tables: _ResampleTables,
    target_h: int,
    target_w: int,
    means: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    stds: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    swap_rb: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused PIL-exact resize + color swap + normalize.

    Args:
        src: uint8 CUDA tensor, shape (H, W, 3), HWC layout.
        tables: precomputed int32 resample tables from `build_resample_tables`.
        target_h, target_w: output spatial dims.
        means, stds: normalization in output channel order (R, G, B for
            network_input.color_mode == 'rgb').
        swap_rb: if True, source channel 0 → output B (BGR input, RGB network).
        out: optional preallocated fp32 (1, 3, H, W) CUDA tensor.

    Returns:
        fp32 (1, 3, target_h, target_w) on the same device as `src`.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("triton is not installed")
    if not src.is_cuda:
        raise ValueError(f"expected CUDA src tensor, got device={src.device}")
    if src.dtype != torch.uint8:
        raise ValueError(f"expected uint8 src, got {src.dtype}")
    if src.ndim != 3 or src.shape[2] != 3:
        raise ValueError(f"expected HWC 3-channel, got shape={tuple(src.shape)}")

    src = src.contiguous()
    src_h, src_w = int(src.shape[0]), int(src.shape[1])
    src_stride_h = int(src.stride(0))
    src_stride_w = int(src.stride(1))

    if out is None:
        out = torch.empty(
            (1, 3, target_h, target_w), dtype=torch.float32, device=src.device
        )
    else:
        if tuple(out.shape) != (1, 3, target_h, target_w):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected "
                f"(1, 3, {target_h}, {target_w})"
            )
        if out.dtype != torch.float32 or not out.is_cuda:
            raise ValueError("out must be fp32 CUDA tensor")

    dst_stride_c = target_h * target_w
    dst_stride_h = target_w

    if swap_rb:
        ch_r, ch_g, ch_b = 2, 1, 0
    else:
        ch_r, ch_g, ch_b = 0, 1, 2

    inv_std_255_r = 1.0 / (255.0 * stds[0])
    inv_std_255_g = 1.0 / (255.0 * stds[1])
    inv_std_255_b = 1.0 / (255.0 * stds[2])
    offset_r = -means[0] / stds[0]
    offset_g = -means[1] / stds[1]
    offset_b = -means[2] / stds[2]

    BLOCK_H = 16
    BLOCK_W = 16
    grid = (
        (target_h + BLOCK_H - 1) // BLOCK_H,
        (target_w + BLOCK_W - 1) // BLOCK_W,
    )
    _fused_resize_normalize_kernel[grid](
        src,
        out,
        tables.ymin_gpu,
        tables.xmin_gpu,
        tables.wy_gpu,
        tables.wx_gpu,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        dst_stride_c,
        dst_stride_h,
        target_h,
        target_w,
        float(inv_std_255_r),
        float(inv_std_255_g),
        float(inv_std_255_b),
        float(offset_r),
        float(offset_g),
        float(offset_b),
        CH_R=ch_r,
        CH_G=ch_g,
        CH_B=ch_b,
        KSIZE_Y=tables.ksize_y,
        KSIZE_X=tables.ksize_x,
        PRECISION_BITS_C=PRECISION_BITS,
        HALF_C=_HALF,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
    )
    return out
