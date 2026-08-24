"""Fused CUDA conversion for YOLO26 depth-estimation preprocessing.

The preserved CPU OpenCV resize produces a contiguous uint8 HWC image. This
module converts that staging image directly to the float32 NCHW TensorRT input,
optionally reversing BGR/RGB channels and applying the model scaling factor in
one Triton launch.
"""

from __future__ import annotations

from typing import Tuple

import torch

from inference_models.errors import MissingDependencyError, ModelRuntimeError

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


_BLOCK_SIZE = 256


if TRITON_AVAILABLE:

    @triton.jit
    def _uint8_hwc_to_float32_nchw_kernel(
        source,
        destination,
        spatial_elements,
        scaling_factor,
        REVERSE_CHANNELS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        output_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        output_elements = spatial_elements * 3
        output_mask = output_offsets < output_elements
        output_channels = output_offsets // spatial_elements
        spatial_offsets = output_offsets % spatial_elements
        source_channels = 2 - output_channels if REVERSE_CHANNELS else output_channels
        source_offsets = spatial_offsets * 3 + source_channels
        values = tl.load(source + source_offsets, mask=output_mask, other=0)
        scaled_values = tl.div_rn(values.to(tl.float32), scaling_factor)
        tl.store(destination + output_offsets, scaled_values, mask=output_mask)


class ExactTritonImageTensorConverter:
    """Convert one CUDA uint8 HWC image into an owned float32 NCHW batch."""

    def __init__(self, *, device: torch.device) -> None:
        if not TRITON_AVAILABLE:
            raise MissingDependencyError(
                message=(
                    "The explicit YOLO26 Triton preprocessor requires the "
                    "`triton` package."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            )
        if device.type != "cuda":
            raise ModelRuntimeError(
                message=(
                    "The explicit YOLO26 Triton preprocessor requires a CUDA "
                    f"device, received {device}."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "models-runtime/#modelruntimeerror"
                ),
            )
        self._device = device

    def convert(
        self,
        *,
        image: torch.Tensor,
        reverse_channels: bool,
        scaling_factor: float,
    ) -> torch.Tensor:
        """Launch the exact fused layout, channel, and scaling conversion."""
        self._validate_input(image=image, scaling_factor=scaling_factor)
        height, width, _ = image.shape
        spatial_elements = height * width
        output = torch.empty(
            (1, 3, height, width),
            dtype=torch.float32,
            device=self._device,
        )
        grid: Tuple[int, ...] = (triton.cdiv(spatial_elements * 3, _BLOCK_SIZE),)
        _uint8_hwc_to_float32_nchw_kernel[grid](
            image,
            output,
            spatial_elements,
            float(scaling_factor),
            REVERSE_CHANNELS=reverse_channels,
            BLOCK_SIZE=_BLOCK_SIZE,
            num_warps=4,
        )

        return output

    def _validate_input(
        self,
        *,
        image: torch.Tensor,
        scaling_factor: float,
    ) -> None:
        reasons = []
        if image.device.type != self._device.type or (
            self._device.index is not None and image.device.index != self._device.index
        ):
            reasons.append(
                f"image device must be {self._device}, received {image.device}"
            )
        if image.dtype != torch.uint8:
            reasons.append(f"image dtype must be uint8, received {image.dtype}")
        if image.ndim != 3 or image.shape[-1] != 3:
            reasons.append(
                "image must have HWC shape with three channels, received "
                f"{tuple(image.shape)}"
            )
        if image.ndim == 3 and image.shape[-1] == 3 and not image.is_contiguous():
            reasons.append(
                "image must be contiguous HWC; implicit contiguous copies are disabled"
            )
        if scaling_factor != 255.0:
            reasons.append(
                "scaling_factor must be exactly 255.0 to preserve the validated "
                f"numerical contract, received {scaling_factor!r}"
            )
        if reasons:
            raise ModelRuntimeError(
                message=(
                    "Input is incompatible with the explicit YOLO26 Triton "
                    "preprocessor: " + "; ".join(reasons)
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "models-runtime/#modelruntimeerror"
                ),
            )
