"""Triton preprocessing kernels for RF-DETR.

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

The runtime implementation is the consolidated two-pass path:
horizontal PIL-antialias resize into a uint8 CHW scratch buffer, followed by
the vertical pass plus `/255` + ImageNet normalization into fp32 CHW output.

Tensor contracts:

* ``src`` is a CUDA uint8 HWC image with shape ``(raw_h, raw_w, 3)``. The
  hot TRT path currently passes a full frame with no static crop, but the
  kernels also accept crop offsets and logical crop dimensions.
* ``tmp`` is a CUDA uint8 CHW scratch tensor with shape
  ``(3, src_h, target_w)``. It stores the horizontally resized image after
  the same fixed-point rounding PIL applies between its separable passes.
* ``out`` is a CUDA fp32 NCHW tensor with shape ``(1, 3, target_h, target_w)``
  in network channel order. Each element is ``(uint8 / 255 - mean) / std``.
* ``ResampleTables`` owns the per-axis int32 fixed-point start/weight tables
  that PIL would precompute for this source/target shape pair.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch

from inference_models.errors import (
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


PRECISION_BITS = 22
_PREPROC_BLOCK_H_ENV = "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_BLOCK_H"
_PREPROC_BLOCK_W_ENV = "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_BLOCK_W"
_PREPROC_HORIZONTAL_BLOCK_H_ENV = (
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_HORIZONTAL_BLOCK_H"
)
_PREPROC_HORIZONTAL_BLOCK_W_ENV = (
    "INFERENCE_MODELS_RFDETR_TRITON_PREPROC_HORIZONTAL_BLOCK_W"
)


def _read_power_of_two_env(name: str, default: int) -> int:
    """Read an optional Triton block-size override from the environment.

    The preprocess kernels use power-of-two block sizes so Triton can form
    static tensor shapes for vectorized loads/stores. This helper keeps those
    launch-shape constraints local to the kernel wrapper and raises a
    user-facing ``ModelRuntimeError`` for invalid values.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ModelRuntimeError(
            message=f"{name} must be an integer, got {raw!r}.",
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        ) from error
    if value <= 0 or value & (value - 1) != 0:
        raise ModelRuntimeError(
            message=f"{name} must be a positive power of two, got {value}.",
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )
    if value > 512:
        raise ModelRuntimeError(
            message=f"{name} must be <= 512, got {value}.",
            help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
        )
    return value


def _bilinear_antialias_weights_1d_int(
    in_size: int, out_size: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build one axis of PIL-compatible bilinear-antialias tables.

    Args:
        in_size: Number of pixels along the source axis after static cropping.
        out_size: Number of pixels along the resized output axis.

    Returns:
        ``(starts, weights_int, ksize)`` where ``starts`` has shape
        ``(out_size,)`` and gives the first source sample for each output
        coordinate, ``weights_int`` has shape ``(out_size, ksize)`` and stores
        PIL's normalized triangle weights in ``PRECISION_BITS`` fixed-point
        format, and ``ksize`` is the compile-time convolution width used by the
        Triton loop for that axis.
    """
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
    def horizontal_resize_uint8_all_channels_kernel(
        src_ptr,
        tmp_ptr,
        xmin_ptr,
        wx_ptr,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        src_stride_c,
        crop_offset_y,
        crop_offset_x,
        target_w,
        CH_R: tl.constexpr,
        CH_G: tl.constexpr,
        CH_B: tl.constexpr,
        KSIZE_X: tl.constexpr,
        PRECISION_BITS_C: tl.constexpr,
        HALF_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Compute PIL's horizontal resize pass for one tile.

        Args:
            src_ptr: CUDA uint8 HWC source image, shape ``(raw_h, raw_w, 3)``.
            tmp_ptr: CUDA uint8 CHW scratch output, shape
                ``(3, src_h, target_w)``.
            xmin_ptr: CUDA int32 starts table, shape ``(target_w,)``.
            wx_ptr: CUDA int32 flattened weights table, shape
                ``(target_w * KSIZE_X,)``.
            src_h/src_w: Logical source height/width after crop. These drive
                bounds checks for the resized region.
            src_stride_h/src_stride_w/src_stride_c: Source strides in elements,
                used so the kernel accepts both contiguous HWC tensors and HWC
                views over CUDA CHW tensors without a layout conversion.
            crop_offset_y/crop_offset_x: Offset into ``src_ptr`` for static
                crop support. The TRT fast path passes zero.
            target_w: Width of the resized network input.
            CH_R/CH_G/CH_B: Source channel indices to emit into network
                channel order. For BGR input feeding an RGB model this is
                ``2, 1, 0``.

        Output:
            Writes the horizontally resized and PIL-rounded uint8 values into
            ``tmp_ptr`` in CHW order. The vertical kernel consumes this scratch
            buffer as its input image.
        """
        # Program ids tile over logical source rows and target columns. The
        # y-axis is still source height because this pass only resizes width.
        pid_y = tl.program_id(0)
        pid_x = tl.program_id(1)

        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_y = offs_y < src_h
        mask_x = offs_x < target_w
        mask_out = mask_y[:, None] & mask_x[None, :]

        # For each output x, PIL precomputes the first contributing source x
        # and a fixed-width row of int32 fixed-point triangle weights.
        xmin = tl.load(xmin_ptr + offs_x, mask=mask_x, other=0)
        sy = offs_y + crop_offset_y

        hacc_r = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
        hacc_g = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
        hacc_b = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
        for kx in tl.static_range(KSIZE_X):
            sx = xmin + kx
            sx_c = tl.maximum(tl.minimum(sx, src_w - 1), 0) + crop_offset_x
            wx = tl.load(wx_ptr + offs_x * KSIZE_X + kx, mask=mask_x, other=0)
            base = sy[:, None] * src_stride_h + sx_c[None, :] * src_stride_w
            # Load source pixels in the network's channel order so the channel
            # swap replaces the original PIL image conversion step.
            p_r = tl.load(
                src_ptr + base + CH_R * src_stride_c, mask=mask_out, other=0
            ).to(tl.int32)
            p_g = tl.load(
                src_ptr + base + CH_G * src_stride_c, mask=mask_out, other=0
            ).to(tl.int32)
            p_b = tl.load(
                src_ptr + base + CH_B * src_stride_c, mask=mask_out, other=0
            ).to(tl.int32)
            wx_2d = wx[None, :]
            # Fixed-point horizontal convolution: sum(src * PIL_weight_int).
            hacc_r += p_r * wx_2d
            hacc_g += p_g * wx_2d
            hacc_b += p_b * wx_2d

        # Match PIL's intermediate uint8 rounding before the vertical pass.
        q_r = (hacc_r + HALF_C) >> PRECISION_BITS_C
        q_g = (hacc_g + HALF_C) >> PRECISION_BITS_C
        q_b = (hacc_b + HALF_C) >> PRECISION_BITS_C
        q_r = tl.minimum(tl.maximum(q_r, 0), 255)
        q_g = tl.minimum(tl.maximum(q_g, 0), 255)
        q_b = tl.minimum(tl.maximum(q_b, 0), 255)

        out_row = offs_y[:, None] * target_w + offs_x[None, :]
        channel_stride = src_h * target_w
        # CHW scratch keeps the following vertical pass contiguous along x for
        # one output channel at a time.
        tl.store(tmp_ptr + 0 * channel_stride + out_row, q_r, mask=mask_out)
        tl.store(tmp_ptr + 1 * channel_stride + out_row, q_g, mask=mask_out)
        tl.store(tmp_ptr + 2 * channel_stride + out_row, q_b, mask=mask_out)

    @triton.jit
    def vertical_normalize_from_horizontal_kernel(
        tmp_ptr,
        dst_ptr,
        ymin_ptr,
        wy_ptr,
        src_h,
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
        KSIZE_Y: tl.constexpr,
        PRECISION_BITS_C: tl.constexpr,
        HALF_C: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_W: tl.constexpr,
    ):
        """Compute PIL's vertical resize pass and torchvision normalization.

        Args:
            tmp_ptr: CUDA uint8 CHW horizontal scratch, shape
                ``(3, src_h, target_w)``.
            dst_ptr: CUDA fp32 NCHW output, shape
                ``(1, 3, target_h, target_w)``.
            ymin_ptr: CUDA int32 starts table, shape ``(target_h,)``.
            wy_ptr: CUDA int32 flattened weights table, shape
                ``(target_h * KSIZE_Y,)``.
            src_h: Logical source height after crop and after the horizontal
                pass. This is the height of ``tmp_ptr``.
            dst_stride_c/dst_stride_h: Output strides in elements.
            target_h/target_w: Resized network input shape.
            inv_std_255_*: Precomputed ``1 / (255 * std[channel])`` values.
            offset_*: Precomputed ``-mean[channel] / std[channel]`` values.

        Output:
            Writes normalized fp32 NCHW data into ``dst_ptr``. This is the
            tensor consumed directly by TensorRT.
        """
        # Program ids tile over output rows, output columns, and the three
        # output channels. Channel-specific normalization is selected by pid_c.
        pid_y = tl.program_id(0)
        pid_x = tl.program_id(1)
        pid_c = tl.program_id(2)

        offs_y = pid_y * BLOCK_H + tl.arange(0, BLOCK_H)
        offs_x = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        mask_y = offs_y < target_h
        mask_x = offs_x < target_w
        mask_out = mask_y[:, None] & mask_x[None, :]

        # For each output y, load the first source row and PIL fixed-point
        # weights for the vertical half of the separable resize.
        ymin = tl.load(ymin_ptr + offs_y, mask=mask_y, other=0)

        vacc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.int32)
        for ky in tl.static_range(KSIZE_Y):
            sy = ymin + ky
            sy_c = tl.maximum(tl.minimum(sy, src_h - 1), 0)
            wy = tl.load(wy_ptr + offs_y * KSIZE_Y + ky, mask=mask_y, other=0)
            base = sy_c[:, None] * target_w + offs_x[None, :]
            p = tl.load(
                tmp_ptr + pid_c * src_h * target_w + base, mask=mask_out, other=0
            ).to(tl.int32)
            # Fixed-point vertical convolution over the horizontally rounded
            # scratch buffer, matching PIL's second resample pass.
            vacc += p * wy[:, None]

        # Final PIL uint8 rounding/clamping before torchvision's to_tensor.
        q = (vacc + HALF_C) >> PRECISION_BITS_C
        q = tl.minimum(tl.maximum(q, 0), 255)

        # Fuse TF.to_tensor() (`q / 255`) and TF.normalize().
        inv_std_255 = tl.where(
            pid_c == 0,
            inv_std_255_r,
            tl.where(pid_c == 1, inv_std_255_g, inv_std_255_b),
        )
        offset = tl.where(
            pid_c == 0,
            offset_r,
            tl.where(pid_c == 1, offset_g, offset_b),
        )
        out = q.to(tl.float32) * inv_std_255 + offset

        out_row = offs_y[:, None] * dst_stride_h + offs_x[None, :]
        tl.store(dst_ptr + pid_c * dst_stride_c + out_row, out, mask=mask_out)


class ResampleTables:
    """CUDA cache of PIL fixed-point resize tables for one shape pair.

    Attributes:
        ymin_gpu: int32 tensor with shape ``(target_h,)``. ``ymin_gpu[y]`` is
            the first source row contributing to output row ``y``.
        xmin_gpu: int32 tensor with shape ``(target_w,)``. ``xmin_gpu[x]`` is
            the first source column contributing to output column ``x``.
        wy_gpu: int32 flattened tensor with shape ``(target_h * ksize_y,)``.
            Row ``y`` contains the fixed-point vertical weights for output row
            ``y``.
        wx_gpu: int32 flattened tensor with shape ``(target_w * ksize_x,)``.
            Row ``x`` contains the fixed-point horizontal weights for output
            column ``x``.
        ksize_y/ksize_x: Static loop bounds for the vertical and horizontal
            Triton kernels. They are determined by PIL's antialias support
            radius for the current source/target scale.
    """

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


def resolve_two_pass_launch_config() -> Tuple[int, int, int, int]:
    """Resolve block sizes for the two-pass Triton implementation.

    Returns:
        ``(vertical_block_h, vertical_block_w, horizontal_block_h,
        horizontal_block_w)``. Defaults are tuned for the RF-DETR TRT workload,
        while environment variables allow microbenchmark sweeps without code
        changes.
    """
    return (
        _read_power_of_two_env(_PREPROC_BLOCK_H_ENV, 1),
        _read_power_of_two_env(_PREPROC_BLOCK_W_ENV, 128),
        _read_power_of_two_env(_PREPROC_HORIZONTAL_BLOCK_H_ENV, 1),
        _read_power_of_two_env(_PREPROC_HORIZONTAL_BLOCK_W_ENV, 128),
    )


def build_resample_tables(
    src_h: int,
    src_w: int,
    target_h: int,
    target_w: int,
    device: torch.device,
) -> ResampleTables:
    """Build and upload PIL-compatible resample tables for one resize.

    Args:
        src_h/src_w: Effective source image dimensions after optional crop.
        target_h/target_w: Network input dimensions after resize.
        device: CUDA device where the Triton kernels will run.

    Returns:
        ``ResampleTables`` with all starts/weights already copied to ``device``.
        The hot TRT path keeps this object in a shape-keyed cache so table
        construction is not repeated per frame.
    """
    ymin, wy, ksize_y = _bilinear_antialias_weights_1d_int(src_h, target_h)
    xmin, wx, ksize_x = _bilinear_antialias_weights_1d_int(src_w, target_w)
    return ResampleTables(
        ymin_gpu=torch.from_numpy(ymin).to(device=device, non_blocking=True),
        xmin_gpu=torch.from_numpy(xmin).to(device=device, non_blocking=True),
        wy_gpu=torch.from_numpy(wy.ravel()).to(device=device, non_blocking=True),
        wx_gpu=torch.from_numpy(wx.ravel()).to(device=device, non_blocking=True),
        ksize_y=ksize_y,
        ksize_x=ksize_x,
    )


def triton_preprocess_rfdetr_stretch_two_pass_preallocated(
    src: torch.Tensor,
    out: torch.Tensor,
    tmp: torch.Tensor,
    tables: ResampleTables,
    target_h: int,
    target_w: int,
    means: Tuple[float, float, float],
    stds: Tuple[float, float, float],
    swap_rb: bool,
    launch_config: Tuple[int, int, int, int],
    crop_offset_y: int = 0,
    crop_offset_x: int = 0,
    crop_h: Optional[int] = None,
    crop_w: Optional[int] = None,
) -> torch.Tensor:
    """Launch the fast two-pass preprocessor using caller-owned buffers.

    This is the hot path used by the TensorRT adapter. It intentionally assumes
    the caller already validated shapes, dtypes, device placement, and table
    compatibility so each frame only pays for the HtoD copy and two Triton
    kernel launches.

    Args:
        src: CUDA uint8 HWC source tensor, shape ``(raw_h, raw_w, 3)``.
        out: CUDA fp32 NCHW output tensor, shape
            ``(1, 3, target_h, target_w)``.
        tmp: CUDA uint8 CHW scratch tensor, shape ``(3, src_h, target_w)``.
        tables: ``ResampleTables`` built for ``(src_h, src_w)`` to
            ``(target_h, target_w)``.
        target_h/target_w: Network input dimensions.
        means/stds: Per-channel normalization constants in output channel
            order.
        swap_rb: Whether to swap source red/blue channels while writing
            network-order output channels.
        launch_config: Block sizes returned by ``resolve_two_pass_launch_config``.
        crop_offset_y/crop_offset_x: Optional top-left crop offset in ``src``.
        crop_h/crop_w: Optional logical source shape after crop. When omitted,
            the full source tensor shape is used.

    Returns:
        The same ``out`` tensor after scheduling both kernels on the current
        CUDA stream.
    """
    raw_src_h, raw_src_w = int(src.shape[0]), int(src.shape[1])
    src_h = crop_h if crop_h is not None else raw_src_h
    src_w = crop_w if crop_w is not None else raw_src_w
    src_stride_h = int(src.stride(0))
    src_stride_w = int(src.stride(1))
    src_stride_c = int(src.stride(2))
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
    block_h, block_w, horizontal_block_h, horizontal_block_w = launch_config

    # First reproduce PIL's horizontal resize into uint8 scratch. This is the
    # only pass that reads the raw HWC frame.
    horizontal_grid = (
        (src_h + horizontal_block_h - 1) // horizontal_block_h,
        (target_w + horizontal_block_w - 1) // horizontal_block_w,
    )
    horizontal_resize_uint8_all_channels_kernel[horizontal_grid](
        src,
        tmp,
        tables.xmin_gpu,
        tables.wx_gpu,
        src_h,
        src_w,
        src_stride_h,
        src_stride_w,
        src_stride_c,
        int(crop_offset_y),
        int(crop_offset_x),
        target_w,
        CH_R=ch_r,
        CH_G=ch_g,
        CH_B=ch_b,
        KSIZE_X=tables.ksize_x,
        PRECISION_BITS_C=PRECISION_BITS,
        HALF_C=_HALF,
        BLOCK_H=horizontal_block_h,
        BLOCK_W=horizontal_block_w,
    )
    # Then reproduce PIL's vertical resize and fuse the torchvision tensor
    # conversion and normalization into the final fp32 TensorRT input.
    grid = (
        (target_h + block_h - 1) // block_h,
        (target_w + block_w - 1) // block_w,
    )
    vertical_normalize_from_horizontal_kernel[(grid[0], grid[1], 3)](
        tmp,
        out,
        tables.ymin_gpu,
        tables.wy_gpu,
        src_h,
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
        KSIZE_Y=tables.ksize_y,
        PRECISION_BITS_C=PRECISION_BITS,
        HALF_C=_HALF,
        BLOCK_H=block_h,
        BLOCK_W=block_w,
    )
    return out


def triton_preprocess_rfdetr_stretch(
    src: torch.Tensor,
    tables: ResampleTables,
    target_h: int,
    target_w: int,
    means: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    stds: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    swap_rb: bool = True,
    crop_offset_y: int = 0,
    crop_offset_x: int = 0,
    crop_h: Optional[int] = None,
    crop_w: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
    tmp: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """PIL-exact resize + color swap + normalize using the two-pass kernels.

    Args:
        src: uint8 CUDA tensor, shape ``(raw_h, raw_w, 3)``, HWC layout.
        tables: precomputed int32 resample tables sized against the *cropped*
            source ``(crop_h, crop_w)`` to ``(target_h, target_w)``.
        target_h, target_w: output spatial dims.
        means, stds: normalization in output channel order (R, G, B for
            network_input.color_mode == 'rgb').
        swap_rb: if True, source channel 0 → output B (BGR input, RGB network).
        crop_offset_y/_x: load-time offset into `src` for a static crop. 0
            means no crop.
        crop_h/_w: effective source dims after crop. Defaults to src dims
            when no crop is configured.
        out: optional preallocated fp32 ``(1, 3, target_h, target_w)`` CUDA
            tensor.
        tmp: optional preallocated uint8 ``(3, crop_h/raw_h, target_w)`` CUDA
            tensor used by the horizontal pass.

    Returns:
        fp32 ``(1, 3, target_h, target_w)`` on the same device as ``src``.
    """
    if not TRITON_AVAILABLE:
        raise MissingDependencyError(
            message="triton is not installed",
            help_url="https://inference-models.roboflow.com/errors/runtime-environment/#missingdependencyerror",
        )
    if not src.is_cuda:
        raise ModelInputError(
            message=f"expected CUDA src tensor, got device={src.device}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if src.dtype != torch.uint8:
        raise ModelInputError(
            message=f"expected uint8 src, got {src.dtype}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if src.ndim != 3 or src.shape[2] != 3:
        raise ModelInputError(
            message=f"expected HWC 3-channel, got shape={tuple(src.shape)}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )

    src = src.contiguous()
    raw_src_h, raw_src_w = int(src.shape[0]), int(src.shape[1])
    src_h = crop_h if crop_h is not None else raw_src_h
    src_w = crop_w if crop_w is not None else raw_src_w
    src_stride_h = int(src.stride(0))
    src_stride_w = int(src.stride(1))

    if out is None:
        out = torch.empty(
            (1, 3, target_h, target_w), dtype=torch.float32, device=src.device
        )
    else:
        if tuple(out.shape) != (1, 3, target_h, target_w):
            raise ModelRuntimeError(
                message=(
                    f"out has shape {tuple(out.shape)}, expected "
                    f"(1, 3, {target_h}, {target_w})"
                ),
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        if out.dtype != torch.float32 or not out.is_cuda:
            raise ModelRuntimeError(
                message="out must be fp32 CUDA tensor",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )

    if tmp is None:
        tmp = torch.empty((3, src_h, target_w), dtype=torch.uint8, device=src.device)
    else:
        if tuple(tmp.shape) != (3, src_h, target_w):
            raise ModelRuntimeError(
                message=(
                    f"tmp has shape {tuple(tmp.shape)}, expected "
                    f"(3, {src_h}, {target_w})"
                ),
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )
        if tmp.dtype != torch.uint8 or not tmp.is_cuda:
            raise ModelRuntimeError(
                message="tmp must be uint8 CUDA tensor",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            )

    return triton_preprocess_rfdetr_stretch_two_pass_preallocated(
        src=src,
        out=out,
        tmp=tmp,
        tables=tables,
        target_h=target_h,
        target_w=target_w,
        means=means,
        stds=stds,
        swap_rb=swap_rb,
        launch_config=resolve_two_pass_launch_config(),
        crop_offset_y=crop_offset_y,
        crop_offset_x=crop_offset_x,
        crop_h=crop_h,
        crop_w=crop_w,
    )
