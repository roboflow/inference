import atexit
import threading
from functools import lru_cache
from typing import Callable, Dict, Optional, Set, Tuple

import cv2
import numpy as np
import torch

from inference_models.logger import LOGGER
from inference_models.models.optimization.triton_jit import classify_triton_jit_failure

TritonResize = Callable[[torch.Tensor, int, int], torch.Tensor]

_RESIZE_COEF_SCALE = 2048
TRITON_RESIZE_BLOCK = 256
_CUDA_MAX_GRID_YZ = 65535
_TRITON_RESIZES: Dict[int, TritonResize] = {}
_TRITON_STARTED: Set[int] = set()
_TRITON_LOCK = threading.Lock()
_TRITON_VERIFICATION_SHAPES = [
    (1, 3, 640, 640, 448, 448),
    (2, 3, 1280, 1280, 640, 640),
    (1, 3, 486, 480, 640, 640),
    (1, 3, 37, 53, 101, 29),
    (3, 3, 255, 129, 17, 255),
    (1, 1, 99, 301, 200, 150),
    (1, 3, 1, 7, 5, 1),
]


def resize_uint8_tensor_like_opencv(
    image: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    """Resize a uint8 image batch with OpenCV-compatible ``INTER_LINEAR`` values.

    CPU tensors use ``cv2.resize``. Other devices use the eager port of OpenCV's
    generic (non-HAL) arithmetic; platform-specific OpenCV implementations may
    differ. On CUDA, Triton is used after background verification against eager,
    not the installed OpenCV build. A recognized Triton JIT failure disables it
    for that device and retries eagerly; unexpected errors propagate.

    Args:
        image: ``N x C x H x W`` uint8 tensor.
        height: Output height.
        width: Output width.

    Returns:
        ``N x C x height x width`` uint8 tensor on the input device. Same-size
        requests return the original tensor without starting Triton preparation.
    """
    if image.shape[2:] == (height, width):
        return image

    if image.device.type == "cpu":
        resized = _resize_with_opencv(image, height=height, width=width)
        return resized

    if image.device.type == "cuda":
        triton_resize = _get_triton_resize(image.device.index)
        if triton_resize is not None and _fits_triton_kernel(
            image, height=height, width=width
        ):
            try:
                with torch.cuda.device(image.device):
                    resized = triton_resize(image, height, width)
            except Exception as error:
                diagnostic = classify_triton_jit_failure(error)
                if diagnostic is None:
                    raise

                with _TRITON_LOCK:
                    failed_resize = _TRITON_RESIZES.pop(image.device.index, None)
                if failed_resize is not None:
                    LOGGER.warning(
                        "OpenCV-compatible tensor resize disabled Triton on %s "
                        "after %s; using eager torch. %s",
                        image.device,
                        diagnostic.category,
                        diagnostic.guidance,
                        exc_info=True,
                    )
            else:
                return resized

    resized = resize_uint8_tensor_like_opencv_eager(image, height=height, width=width)
    return resized


def resize_uint8_tensor_like_opencv_eager(
    image: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    """Torch port of OpenCV's uint8 ``INTER_LINEAR`` resize, on any device.

    Reproduces OpenCV's generic (non-HAL) implementation: fixed-point weights with
    11 fractional bits, a horizontal pass into ``int32`` and the vertical pass
    rounding back to uint8, plus the exact 2x down-scaling special case.

    Args:
        image: ``N x C x H x W`` uint8 tensor.
        height: Output height.
        width: Output width.

    Returns:
        ``N x C x height x width`` uint8 tensor on the input device.
    """
    source_height, source_width = image.shape[2], image.shape[3]
    if (source_height, source_width) == (height, width):
        return image

    if _is_half_without_fast_path(image, height=height, width=width):
        sums = (
            image[:, :, 0::2, 0::2].to(torch.int32)
            + image[:, :, 0::2, 1::2]
            + image[:, :, 1::2, 0::2]
            + image[:, :, 1::2, 1::2]
        )
        resized = torch.round(sums.to(torch.float32) / 4).to(torch.uint8)
        return resized

    x0, x1, alpha0, alpha1 = stream_safe_linear_coefficients(
        source_width, width, clamp_fraction=True, device=image.device
    )
    y0, y1, beta0, beta1 = stream_safe_linear_coefficients(
        source_height, height, clamp_fraction=False, device=image.device
    )
    vertical = None
    for rows, beta in ((y0[:, None], beta0), (y1[:, None], beta1)):
        horizontal = image[:, :, rows, x0].to(torch.int32) * alpha0
        horizontal += image[:, :, rows, x1].to(torch.int32) * alpha1
        horizontal >>= 4
        horizontal *= beta[:, None]
        horizontal >>= 16
        if vertical is None:
            vertical = horizontal
        else:
            vertical += horizontal
    vertical += 2
    vertical >>= 2
    resized = vertical.clamp_(0, 255).to(torch.uint8)
    return resized


def stream_safe_linear_coefficients(
    source_size: int, target_size: int, *, clamp_fraction: bool, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cached OpenCV ``INTER_LINEAR`` indices and fixed-point weights for one axis.

    The cached tensors are shared by every caller, so on CUDA they are recorded on
    the current stream to keep their memory alive after a cache eviction until the
    work queued on that stream has finished.

    Args:
        source_size: Source length along the axis.
        target_size: Output length along the axis.
        clamp_fraction: Zero the weight of the second sample at the borders, as
            OpenCV does horizontally.
        device: Device of the returned tensors.

    Returns:
        First and second source indices (int64) and their weights (int32).
    """
    coefficients = _opencv_linear_coefficients(
        source_size, target_size, clamp_fraction, device
    )
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        for tensor in coefficients:
            tensor.record_stream(stream)
    return coefficients


def _is_half_without_fast_path(image: torch.Tensor, *, height: int, width: int) -> bool:
    source_height, source_width = image.shape[2], image.shape[3]
    is_half = source_width == 2 * width and source_height == 2 * height
    return is_half and image.shape[1] not in (1, 3, 4)


def _fits_triton_kernel(image: torch.Tensor, *, height: int, width: int) -> bool:
    batch, channels, source_height, source_width = image.shape
    if (source_height, source_width) == (height, width) or image.numel() == 0:
        return False

    if _is_half_without_fast_path(image, height=height, width=width):
        return False

    column_blocks = (width + TRITON_RESIZE_BLOCK - 1) // TRITON_RESIZE_BLOCK
    return batch * channels <= _CUDA_MAX_GRID_YZ and column_blocks <= _CUDA_MAX_GRID_YZ


@lru_cache(maxsize=128)
def _opencv_linear_coefficients(
    source_size: int, target_size: int, clamp_fraction: bool, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1.0 / (target_size / source_size)
    positions = ((np.arange(target_size, dtype=np.float64) + 0.5) * scale - 0.5).astype(
        np.float32
    )
    starts = np.floor(positions).astype(np.int64)
    fractions = positions - starts.astype(np.float32)
    if clamp_fraction:
        outside = (starts < 0) | (starts >= source_size - 1)
        fractions = np.where(outside, np.float32(0), fractions)
    index0 = np.clip(starts, 0, source_size - 1)
    index1 = np.clip(starts + 1, 0, source_size - 1)
    weight0 = np.rint((np.float32(1) - fractions) * np.float32(_RESIZE_COEF_SCALE))
    weight1 = np.rint(fractions * np.float32(_RESIZE_COEF_SCALE))

    coefficients = tuple(
        torch.from_numpy(np.ascontiguousarray(values)).to(device)
        for values in (
            index0,
            index1,
            weight0.astype(np.int32),
            weight1.astype(np.int32),
        )
    )
    return coefficients


def _resize_with_opencv(
    image: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    resized_images = []
    for single_image in image:
        resized = cv2.resize(
            np.ascontiguousarray(single_image.permute(1, 2, 0).numpy()),
            (width, height),
        )
        resized_images.append(resized.reshape(height, width, -1))
    batch = torch.from_numpy(np.stack(resized_images)).permute(0, 3, 1, 2)
    return batch


def _get_triton_resize(device_index: int) -> Optional[TritonResize]:
    _start_triton_compilation(device_index)
    triton_resize = _TRITON_RESIZES.get(device_index)
    return triton_resize


def _start_triton_compilation(device_index: int) -> None:
    with _TRITON_LOCK:
        if device_index in _TRITON_STARTED:
            return
        _TRITON_STARTED.add(device_index)

    thread = threading.Thread(
        target=_switch_to_triton,
        args=(device_index,),
        name=f"opencv-resize-triton-{device_index}",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        LOGGER.warning(
            "OpenCV-exact tensor resize stays on eager torch: no compilation thread",
            exc_info=True,
        )
        return
    # Interpreter exit while Triton compiles aborts the process; wait for it.
    atexit.register(thread.join)


def _switch_to_triton(device_index: int) -> None:
    try:
        triton_resize = _load_triton_resize()
        with torch.cuda.device(device_index):
            _verify_triton_resize(
                triton_resize, device=torch.device("cuda", device_index)
            )
    except Exception:
        LOGGER.warning(
            "OpenCV-exact tensor resize stays on eager torch on cuda:%s",
            device_index,
            exc_info=True,
        )
        return

    _TRITON_RESIZES[device_index] = triton_resize
    LOGGER.info(
        "OpenCV-exact tensor resize switched to Triton on cuda:%s", device_index
    )


def _load_triton_resize() -> TritonResize:
    from inference_models.models.common.roboflow.opencv_resize_triton import (
        resize_uint8_tensor_like_opencv_triton,
    )

    return resize_uint8_tensor_like_opencv_triton


def _verify_triton_resize(triton_resize: TritonResize, *, device: torch.device) -> None:
    generator = torch.Generator(device="cpu").manual_seed(0)
    for index, (
        batch,
        channels,
        height,
        width,
        target_height,
        target_width,
    ) in enumerate(_TRITON_VERIFICATION_SHAPES):
        image = torch.randint(
            0,
            256,
            (batch, height, width, channels),
            dtype=torch.uint8,
            generator=generator,
        )
        image = image.to(device).permute(0, 3, 1, 2)
        if index % 2:
            image = image.contiguous()
        expected = resize_uint8_tensor_like_opencv_eager(
            image, height=target_height, width=target_width
        )
        actual = triton_resize(image, target_height, target_width)
        if not torch.equal(actual, expected):
            raise RuntimeError(
                f"Triton resize differs from eager for {tuple(image.shape)} -> "
                f"{(target_height, target_width)}"
            )
