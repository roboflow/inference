import threading
from contextlib import contextmanager
from typing import Generator, Tuple

import torch

# TorchScript deserialization/compilation mutates a process-global type registry and
# compilation unit that is NOT thread-safe. Loading two TorchScript-backed models
# concurrently (e.g. preloading several PINNED_MODELS on a ThreadPoolExecutor) can
# corrupt it, surfacing non-deterministically as `KeyError: '__torch__...'` or
# `Enum<...___torch_mangle_N.InterpolationMode>` type mismatches. Serialize every
# TorchScript load behind this single process-wide lock.
_TORCHSCRIPT_LOAD_LOCK = threading.Lock()


@contextmanager
def torchscript_load_lock() -> Generator[None, None, None]:
    """Serialize TorchScript model loading across threads.

    Wrap any call that triggers TorchScript deserialization/compilation
    (`torch.jit.load`, or model builders that script submodules) so concurrent loads
    cannot corrupt TorchScript's global type registry. Guards one-time model
    construction only; it does not affect inference concurrency.
    """
    with _TORCHSCRIPT_LOAD_LOCK:
        yield


def generate_batch_chunks(
    input_batch: torch.Tensor,
    chunk_size: int,
) -> Generator[Tuple[torch.Tensor, int], None, None]:
    """Generate fixed-size chunks from a batch tensor with automatic padding.

    Splits a batch tensor into fixed-size chunks along the batch dimension (dim 0).
    If the last chunk is smaller than chunk_size, it is automatically padded with
    zeros to maintain consistent chunk sizes.

    This is useful for processing large batches through models with fixed batch
    size requirements or to avoid GPU memory issues.

    Args:
        input_batch: Input tensor with batch dimension as the first dimension.
            Shape: (batch_size, ...).

        chunk_size: Size of each chunk. All chunks will have this size in the
            batch dimension, with the last chunk padded if necessary.

    Yields:
        Tuples of (chunk, padding_size) where:
            - chunk: Tensor of shape (chunk_size, ...) containing the batch chunk
            - padding_size: Number of padding elements added (0 for full chunks)

    Examples:
        Process large batch in chunks:

        >>> from inference_models.developer_tools import generate_batch_chunks
        >>> import torch
        >>>
        >>> # Large batch of images
        >>> batch = torch.randn(100, 3, 640, 640)
        >>>
        >>> results = []
        >>> for chunk, padding in generate_batch_chunks(batch, chunk_size=16):
        ...     # Process chunk through model
        ...     output = model(chunk)
        ...
        ...     # Remove padding from results
        ...     if padding > 0:
        ...         output = output[:-padding]
        ...
        ...     results.append(output)
        >>>
        >>> # Concatenate all results
        >>> final_output = torch.cat(results, dim=0)

        Handle models with static batch size:

        >>> # Model requires exactly batch size of 8
        >>> batch = torch.randn(20, 3, 224, 224, device="cuda")
        >>>
        >>> for chunk, padding in generate_batch_chunks(batch, chunk_size=8):
        ...     # chunk.shape[0] is always 8 (padded if needed)
        ...     output = model(chunk)
        ...
        ...     # Last chunk has padding=4, so remove it
        ...     if padding > 0:
        ...         output = output[:-padding]

    Note:
        - Chunks are created as views when possible (no padding needed)
        - Padding is added with zeros matching the input dtype and device
        - The last chunk is always padded to chunk_size if needed
        - Padding size is 0 for all chunks except potentially the last one

    See Also:
        - `run_onnx_session_with_batch_size_limit()`: Uses this internally for ONNX models
    """
    n = input_batch.shape[0]
    for i in range(0, n, chunk_size):
        chunk = input_batch[i : i + chunk_size]
        padding_size = chunk_size - chunk.shape[0]
        if padding_size > 0:
            padding_shape = (padding_size,) + chunk.shape[1:]
            padding = torch.zeros(
                padding_shape, device=input_batch.device, dtype=input_batch.dtype
            )
            chunk = torch.cat([chunk, padding], dim=0)
        yield chunk, padding_size
