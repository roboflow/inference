"""Triton antialiased resize for YOLO26 depth-map postprocessing.

The weight construction and accumulation order mirror PyTorch 2.6's CUDA
``upsample_gen2d_aa_out_frame`` implementation. Axis tables are cached because
the frozen profiling workloads repeatedly use the same source and destination
shapes. The kernel consumes strided single-channel views and writes one owned,
contiguous float32 depth map.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional

from inference_models.errors import MissingDependencyError, ModelRuntimeError

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover
    triton = None
    tl = None
    TRITON_AVAILABLE = False


_MAX_TABLE_CACHE_ENTRIES = 8
_OUTPUT_BLOCK_SIZE = 256


@dataclass(frozen=True)
class _AxisTable:
    starts: torch.Tensor
    sizes: torch.Tensor
    weights: torch.Tensor
    maximum_size: int
    ready_event: torch.cuda.Event


def _build_axis_table(
    *,
    input_size: int,
    output_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build PyTorch CUDA bilinear-antialias indices and float32 weights."""
    scale = np.float32(input_size / output_size)
    support = scale if scale >= np.float32(1.0) else np.float32(1.0)
    maximum_size = int(math.ceil(float(support))) * 2 + 1
    inverse_scale = (
        np.float32(1.0) / scale if scale >= np.float32(1.0) else np.float32(1.0)
    )

    starts = np.zeros(output_size, dtype=np.int32)
    sizes = np.zeros(output_size, dtype=np.int32)
    weights = np.zeros((output_size, maximum_size), dtype=np.float32)
    half = np.float32(0.5)

    for output_index in range(output_size):
        center = np.float32(scale * np.float32(output_index + 0.5))
        start = max(int(np.float32(center - support + half)), 0)
        stop = min(int(np.float32(center + support + half)), input_size)
        size = stop - start
        starts[output_index] = start
        sizes[output_index] = size

        total_weight = np.float32(0.0)
        for kernel_index in range(size):
            distance = np.float32(np.float32(kernel_index + start) - center + half)
            distance = np.float32(distance * inverse_scale)
            if distance < np.float32(0.0):
                distance = np.float32(-distance)
            weight = (
                np.float32(1.0) - distance
                if distance < np.float32(1.0)
                else np.float32(0.0)
            )
            weights[output_index, kernel_index] = weight
            total_weight = np.float32(total_weight + weight)

        if total_weight != np.float32(0.0):
            for kernel_index in range(size):
                weights[output_index, kernel_index] = np.float32(
                    weights[output_index, kernel_index] / total_weight
                )

    return starts, sizes, weights, maximum_size


if TRITON_AVAILABLE:

    @triton.jit
    def _resize_bilinear_antialias_kernel(
        source,
        destination,
        y_starts,
        y_sizes,
        y_weights,
        x_starts,
        x_sizes,
        x_weights,
        output_height,
        output_width,
        source_stride_h,
        source_stride_w,
        MAXIMUM_Y_SIZE: tl.constexpr,
        MAXIMUM_X_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        output_elements = output_height * output_width
        output_mask = offsets < output_elements
        output_y = offsets // output_width
        output_x = offsets % output_width

        y_start = tl.load(y_starts + output_y, mask=output_mask, other=0)
        y_size = tl.load(y_sizes + output_y, mask=output_mask, other=0)
        x_start = tl.load(x_starts + output_x, mask=output_mask, other=0)
        x_size = tl.load(x_sizes + output_x, mask=output_mask, other=0)

        output = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for kernel_y in tl.static_range(MAXIMUM_Y_SIZE):
            horizontal = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
            source_y = y_start + kernel_y
            y_mask = output_mask & (kernel_y < y_size)
            for kernel_x in tl.static_range(MAXIMUM_X_SIZE):
                source_x = x_start + kernel_x
                sample = tl.load(
                    source + source_y * source_stride_h + source_x * source_stride_w,
                    mask=y_mask & (kernel_x < x_size),
                    other=0.0,
                )
                weight_x = tl.load(
                    x_weights + output_x * MAXIMUM_X_SIZE + kernel_x,
                    mask=output_mask & (kernel_x < x_size),
                    other=0.0,
                )
                horizontal += sample * weight_x

            weight_y = tl.load(
                y_weights + output_y * MAXIMUM_Y_SIZE + kernel_y,
                mask=y_mask,
                other=0.0,
            )
            output += horizontal * weight_y

        tl.store(destination + offsets, output, mask=output_mask)

    @triton.jit
    def _resize_horizontal_exact_kernel(
        source,
        workspace,
        x_starts,
        x_sizes,
        x_weights,
        input_height,
        output_width,
        source_stride_h,
        source_stride_w,
        MAXIMUM_X_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        workspace_elements = input_height * output_width
        output_mask = offsets < workspace_elements
        source_y = offsets // output_width
        output_x = offsets % output_width
        x_start = tl.load(x_starts + output_x, mask=output_mask, other=0)
        x_size = tl.load(x_sizes + output_x, mask=output_mask, other=0)

        first_sample = tl.load(
            source + source_y * source_stride_h + x_start * source_stride_w,
            mask=output_mask,
            other=0.0,
        )
        first_weight = tl.load(
            x_weights + output_x * MAXIMUM_X_SIZE,
            mask=output_mask,
            other=0.0,
        )
        output = first_sample * first_weight
        for kernel_x in tl.static_range(1, MAXIMUM_X_SIZE):
            sample = tl.load(
                source
                + source_y * source_stride_h
                + (x_start + kernel_x) * source_stride_w,
                mask=output_mask & (kernel_x < x_size),
                other=0.0,
            )
            weight = tl.load(
                x_weights + output_x * MAXIMUM_X_SIZE + kernel_x,
                mask=output_mask & (kernel_x < x_size),
                other=0.0,
            )
            accumulated = output + sample * weight
            output = tl.where(
                output_mask & (kernel_x < x_size),
                accumulated,
                output,
            )

        tl.store(workspace + offsets, output, mask=output_mask)

    @triton.jit
    def _resize_vertical_exact_kernel(
        workspace,
        destination,
        y_starts,
        y_sizes,
        y_weights,
        output_height,
        output_width,
        workspace_stride_h,
        MAXIMUM_Y_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        output_elements = output_height * output_width
        output_mask = offsets < output_elements
        output_y = offsets // output_width
        output_x = offsets % output_width
        y_start = tl.load(y_starts + output_y, mask=output_mask, other=0)
        y_size = tl.load(y_sizes + output_y, mask=output_mask, other=0)

        first_sample = tl.load(
            workspace + y_start * workspace_stride_h + output_x,
            mask=output_mask,
            other=0.0,
        )
        first_weight = tl.load(
            y_weights + output_y * MAXIMUM_Y_SIZE,
            mask=output_mask,
            other=0.0,
        )
        output = first_sample * first_weight
        for kernel_y in tl.static_range(1, MAXIMUM_Y_SIZE):
            sample = tl.load(
                workspace + (y_start + kernel_y) * workspace_stride_h + output_x,
                mask=output_mask & (kernel_y < y_size),
                other=0.0,
            )
            weight = tl.load(
                y_weights + output_y * MAXIMUM_Y_SIZE + kernel_y,
                mask=output_mask & (kernel_y < y_size),
                other=0.0,
            )
            accumulated = output + sample * weight
            output = tl.where(
                output_mask & (kernel_y < y_size),
                accumulated,
                output,
            )

        tl.store(destination + offsets, output, mask=output_mask)


class TritonDepthMapResizer:
    """Resize CUDA float32 depth maps with cached antialias tables."""

    _IMPLEMENTATION_ID = "triton-aa-resize-v1"

    def __init__(self, *, device: torch.device) -> None:
        if not TRITON_AVAILABLE:
            raise MissingDependencyError(
                message=("triton-aa-resize-v1 requires the optional triton runtime."),
                help_url=(
                    "https://inference-models.roboflow.com/errors/"
                    "runtime-environment/#missingdependencyerror"
                ),
            )

        self._device = device
        self._cache: "OrderedDict[Tuple[int, int], _AxisTable]" = OrderedDict()
        self._cache_lock = threading.Lock()

    def resize(
        self,
        image: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize a single-channel CUDA float32 image.

        Args:
            image: Strided ``(1, H, W)`` CUDA float32 source view.
            size: Requested ``(height, width)`` output dimensions.

        Returns:
            Owned contiguous ``(1, output_height, output_width)`` tensor.

        Raises:
            ModelRuntimeError: If the request violates the candidate contract.
        """
        self._validate_request(image=image, size=size)
        output_height, output_width = size
        _, input_height, input_width = image.shape
        y_table = self._axis_table(
            input_size=input_height,
            output_size=output_height,
        )
        x_table = self._axis_table(
            input_size=input_width,
            output_size=output_width,
        )

        output = torch.empty(
            (1, output_height, output_width),
            dtype=torch.float32,
            device=self._device,
        )
        output_elements = output_height * output_width
        grid = (triton.cdiv(output_elements, _OUTPUT_BLOCK_SIZE),)
        _resize_bilinear_antialias_kernel[grid](
            image,
            output,
            y_table.starts,
            y_table.sizes,
            y_table.weights,
            x_table.starts,
            x_table.sizes,
            x_table.weights,
            output_height,
            output_width,
            image.stride(1),
            image.stride(2),
            MAXIMUM_Y_SIZE=y_table.maximum_size,
            MAXIMUM_X_SIZE=x_table.maximum_size,
            BLOCK_SIZE=_OUTPUT_BLOCK_SIZE,
        )

        return output

    def _axis_table(self, *, input_size: int, output_size: int) -> _AxisTable:
        key = (input_size, output_size)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._prepare_table_for_stream(table=cached)

                return cached

            starts, sizes, weights, maximum_size = _build_axis_table(
                input_size=input_size,
                output_size=output_size,
            )
            starts_tensor = torch.from_numpy(starts).to(device=self._device)
            sizes_tensor = torch.from_numpy(sizes).to(device=self._device)
            weights_tensor = torch.from_numpy(weights).to(device=self._device)
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(device=self._device))
            table = _AxisTable(
                starts=starts_tensor,
                sizes=sizes_tensor,
                weights=weights_tensor,
                maximum_size=maximum_size,
                ready_event=ready_event,
            )
            self._cache[key] = table
            if len(self._cache) > _MAX_TABLE_CACHE_ENTRIES:
                self._cache.popitem(last=False)
            self._prepare_table_for_stream(table=table)

        return table

    def _prepare_table_for_stream(self, *, table: _AxisTable) -> None:
        stream = torch.cuda.current_stream(device=self._device)
        stream.wait_event(table.ready_event)
        table.starts.record_stream(stream)
        table.sizes.record_stream(stream)
        table.weights.record_stream(stream)

    def _validate_request(
        self,
        *,
        image: torch.Tensor,
        size: Tuple[int, int],
    ) -> None:
        reasons = []
        if image.device != self._device:
            reasons.append(
                f"source device {image.device} does not match {self._device}"
            )
        if image.dtype != torch.float32:
            reasons.append(f"source dtype must be float32, received {image.dtype}")
        if image.ndim != 3 or image.shape[0] != 1:
            reasons.append(
                f"source shape must be (1, H, W), received {tuple(image.shape)}"
            )
        if image.ndim == 3 and image.stride(2) != 1:
            reasons.append(f"source width stride must be 1, received {image.stride(2)}")
        if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
            reasons.append(f"output size must contain two positive values: {size}")
        if image.ndim == 3 and len(size) == 2 and size[0] > 0 and size[1] > 0:
            maximum_filter_size = max(
                int(math.ceil(max(image.shape[1] / size[0], 1.0))) * 2 + 1,
                int(math.ceil(max(image.shape[2] / size[1], 1.0))) * 2 + 1,
            )
            if maximum_filter_size > 5:
                reasons.append(
                    "antialias filter size must be <= 5, received "
                    f"{maximum_filter_size}"
                )

        if reasons:
            raise ModelRuntimeError(
                message=(
                    f"{self._IMPLEMENTATION_ID} cannot execute this request: "
                    + "; ".join(reasons)
                    + "."
                ),
                help_url=(
                    "https://inference-models.roboflow.com/errors/models-runtime/"
                    "#modelruntimeerror"
                ),
            )


class ExactSeparableTritonDepthMapResizer(TritonDepthMapResizer):
    """Two-pass resize using interpolation weights produced by torchvision CUDA.

    Weight tables are derived during first use by applying the preserved base
    operation to basis vectors on the target device. The steady-state kernels
    then preserve PyTorch's horizontal-then-vertical float32 accumulation order
    while reusing each horizontal interpolation across output rows.
    """

    _IMPLEMENTATION_ID = "triton-aa-resize-exact-v2"

    def resize(
        self,
        image: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """Resize one CUDA depth map with exact cached target-side weights."""
        self._validate_request(image=image, size=size)
        output_height, output_width = size
        _, input_height, input_width = image.shape
        y_table = self._axis_table(
            input_size=input_height,
            output_size=output_height,
        )
        x_table = self._axis_table(
            input_size=input_width,
            output_size=output_width,
        )
        workspace = torch.empty(
            (input_height, output_width),
            dtype=torch.float32,
            device=self._device,
        )
        output = torch.empty(
            (1, output_height, output_width),
            dtype=torch.float32,
            device=self._device,
        )

        workspace_elements = input_height * output_width
        horizontal_grid = (triton.cdiv(workspace_elements, _OUTPUT_BLOCK_SIZE),)
        _resize_horizontal_exact_kernel[horizontal_grid](
            image,
            workspace,
            x_table.starts,
            x_table.sizes,
            x_table.weights,
            input_height,
            output_width,
            image.stride(1),
            image.stride(2),
            MAXIMUM_X_SIZE=x_table.maximum_size,
            BLOCK_SIZE=_OUTPUT_BLOCK_SIZE,
        )
        output_elements = output_height * output_width
        vertical_grid = (triton.cdiv(output_elements, _OUTPUT_BLOCK_SIZE),)
        _resize_vertical_exact_kernel[vertical_grid](
            workspace,
            output,
            y_table.starts,
            y_table.sizes,
            y_table.weights,
            output_height,
            output_width,
            workspace.stride(0),
            MAXIMUM_Y_SIZE=y_table.maximum_size,
            BLOCK_SIZE=_OUTPUT_BLOCK_SIZE,
        )

        return output

    def _axis_table(self, *, input_size: int, output_size: int) -> _AxisTable:
        key = (input_size, output_size)
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                self._prepare_table_for_stream(table=cached)

                return cached

            starts, sizes, _, maximum_size = _build_axis_table(
                input_size=input_size,
                output_size=output_size,
            )
            starts_tensor = torch.from_numpy(starts).to(device=self._device)
            sizes_tensor = torch.from_numpy(sizes).to(device=self._device)
            basis = torch.eye(
                input_size,
                dtype=torch.float32,
                device=self._device,
            ).reshape(input_size, 1, 1, input_size)
            resized_basis = functional.resize(
                basis,
                [1, output_size],
                interpolation=functional.InterpolationMode.BILINEAR,
            )
            dense_weights = resized_basis[:, 0, 0, :].transpose(0, 1)
            offsets = torch.arange(
                maximum_size,
                dtype=torch.int64,
                device=self._device,
            )
            indices = starts_tensor.to(torch.int64).unsqueeze(1) + offsets
            valid = offsets.unsqueeze(0) < sizes_tensor.to(torch.int64).unsqueeze(1)
            weights_tensor = dense_weights.gather(
                1,
                indices.clamp(max=input_size - 1),
            )
            weights_tensor.masked_fill_(~valid, 0.0)
            weights_tensor = weights_tensor.contiguous()
            ready_event = torch.cuda.Event()
            ready_event.record(torch.cuda.current_stream(device=self._device))
            table = _AxisTable(
                starts=starts_tensor,
                sizes=sizes_tensor,
                weights=weights_tensor,
                maximum_size=maximum_size,
                ready_event=ready_event,
            )
            self._cache[key] = table
            if len(self._cache) > _MAX_TABLE_CACHE_ENTRIES:
                self._cache.popitem(last=False)
            self._prepare_table_for_stream(table=table)

        return table
