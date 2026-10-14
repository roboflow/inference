import torch
import triton
import triton.language as tl

from inference_models.models.common.roboflow.opencv_resize import (
    TRITON_RESIZE_BLOCK,
    stream_safe_linear_coefficients,
)


@triton.jit(
    do_not_specialize=[
        "channels",
        "target_height",
        "target_width",
        "stride_n",
        "stride_c",
        "stride_h",
        "stride_w",
    ]
)
def _resize_kernel(
    source,
    output,
    x0_table,
    x1_table,
    alpha0_table,
    alpha1_table,
    y0_table,
    y1_table,
    beta0_table,
    beta1_table,
    channels,
    target_height,
    target_width,
    stride_n,
    stride_c,
    stride_h,
    stride_w,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    plane = tl.program_id(2).to(tl.int64)
    mask = x < target_width
    x0 = tl.load(x0_table + x, mask, 0)
    x1 = tl.load(x1_table + x, mask, 0)
    alpha0 = tl.load(alpha0_table + x, mask, 0)
    alpha1 = tl.load(alpha1_table + x, mask, 0)
    y0 = tl.load(y0_table + row)
    y1 = tl.load(y1_table + row)
    beta0 = tl.load(beta0_table + row)
    beta1 = tl.load(beta1_table + row)
    base = source + (plane // channels) * stride_n + (plane % channels) * stride_c
    row0 = base + y0 * stride_h
    row1 = base + y1 * stride_h
    horizontal0 = (
        tl.load(row0 + x0 * stride_w, mask, 0).to(tl.int32) * alpha0
        + tl.load(row0 + x1 * stride_w, mask, 0).to(tl.int32) * alpha1
    )
    horizontal1 = (
        tl.load(row1 + x0 * stride_w, mask, 0).to(tl.int32) * alpha0
        + tl.load(row1 + x1 * stride_w, mask, 0).to(tl.int32) * alpha1
    )
    vertical = (
        (((horizontal0 >> 4) * beta0) >> 16) + (((horizontal1 >> 4) * beta1) >> 16) + 2
    ) >> 2
    vertical = tl.minimum(tl.maximum(vertical, 0), 255)
    offset = (plane * target_height + row) * target_width + x
    tl.store(output + offset, vertical.to(tl.uint8), mask)


def resize_uint8_tensor_like_opencv_triton(
    image: torch.Tensor, height: int, width: int
) -> torch.Tensor:
    """Triton twin of ``resize_uint8_tensor_like_opencv_eager`` for CUDA tensors.

    Only for resizes that change the size, are not an exact 2x reduction of an
    image with other than 1, 3 or 4 channels, and fit the CUDA grid limits.

    Args:
        image: ``N x C x H x W`` uint8 CUDA tensor, any strides.
        height: Output height.
        width: Output width.

    Returns:
        ``N x C x height x width`` uint8 tensor equal to the eager result.
    """
    batch, channels, source_height, source_width = image.shape
    x0, x1, alpha0, alpha1 = stream_safe_linear_coefficients(
        source_width, width, clamp_fraction=True, device=image.device
    )
    y0, y1, beta0, beta1 = stream_safe_linear_coefficients(
        source_height, height, clamp_fraction=False, device=image.device
    )
    output = torch.empty(
        (batch, channels, height, width), dtype=torch.uint8, device=image.device
    )
    grid = (height, triton.cdiv(width, TRITON_RESIZE_BLOCK), batch * channels)
    _resize_kernel[grid](
        image,
        output,
        x0,
        x1,
        alpha0,
        alpha1,
        y0,
        y1,
        beta0,
        beta1,
        channels,
        height,
        width,
        *image.stride(),
        BLOCK=TRITON_RESIZE_BLOCK,
    )
    return output
