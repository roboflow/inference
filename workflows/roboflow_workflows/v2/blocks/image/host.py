"""The CPU boundary of the image blocks: call V1's NumPy reference on a tensor.

V1 passed OpenCV and scikit-image an ``(height, width)`` array for one
channel and an ``(height, width, 3)`` array in BGR order for three. V2 holds
``(channels, height, width)`` RGB tensors on any device::

    tensor (C, H, W) RGB, any device
        -> host copy, BGR, (H, W, 3) or (H, W)  -> function(array)
        -> (C, H, W) RGB on the input's exact device (e.g. ``cuda:1``)

On CUDA, the copy to the host waits for queued work.
"""

from typing import Callable

import numpy as np
import torch


def call_on_host(
    pixels: torch.Tensor, *, function: Callable[[np.ndarray], np.ndarray]
) -> torch.Tensor:
    """Run a NumPy reference function on ``pixels`` in V1's layout.

    Args:
        pixels: ``(channels, height, width)`` tensor with 1 or 3 channels, on
            any device; not modified.
        function: Maps the host array to a new array of the same height and
            width. For a CPU tensor the array may share the tensor's memory,
            so the function must not write into it.

    Returns:
        A new ``(channels, height, width)`` tensor on ``pixels.device``.
    """
    host = pixels.detach().cpu()
    if host.shape[0] == 3:
        host = host.flip(0)
    array = np.ascontiguousarray(host.permute(1, 2, 0).numpy())
    if array.shape[2] == 1:
        array = array[:, :, 0]

    processed = function(array).reshape(pixels.shape[1], pixels.shape[2], -1)
    restored = torch.from_numpy(np.ascontiguousarray(processed)).permute(2, 0, 1)
    if restored.shape[0] == 3:
        restored = restored.flip(0)
    on_device = restored.contiguous().to(pixels.device)

    return on_device
