import threading
from contextlib import nullcontext
from typing import ContextManager, Dict, Optional, Tuple

import torch

_THREAD_LOCAL_STREAMS = threading.local()


def use_cuda_stream(
    stream: Optional[torch.cuda.Stream],
) -> ContextManager[None]:
    """Activate a CUDA stream, or do nothing when no stream is available.

    Args:
        stream: CUDA stream to activate, or None for non-CUDA devices.

    Returns:
        Context manager that activates the CUDA stream when available.
    """
    return torch.cuda.stream(stream) if stream is not None else nullcontext()


def get_cuda_stream(device: torch.device, purpose: str) -> Optional[torch.cuda.Stream]:
    """Get a CUDA stream shared by all models within the calling thread.

    Streams are cached per (thread, device, purpose) triple and shared across
    model instances. This keeps the number of distinct CUDA streams bounded by
    the number of live threads instead of the number of models created in the
    process - which matters because the torch caching allocator segregates its
    free blocks per stream and never returns cached memory to the driver, so
    every additional stream carrying torch operations multiplies the GPU memory
    footprint of the process.

    Sharing a stream between models running in the same thread does not weaken
    synchronization scoping: a thread executes sequentially and each processing
    stage synchronizes its stream before handing tensors over, so waiting on the
    stream only ever waits for work the calling thread enqueued itself.

    Args:
        device: Torch device the stream should belong to. For non-CUDA devices
            no stream is created.

        purpose: Label separating independent uses within a thread (e.g.
            "pre-processing", "post-processing", "inference"). Streams with
            different purposes are distinct objects, so a stage can synchronize
            its own stream without waiting for another stage's pending work.

    Returns:
        CUDA stream dedicated to the calling thread for the given device and
        purpose, or None when the device is not a CUDA device.

    Examples:
        >>> import torch
        >>> from inference_models.models.common.streams import (
        ...     get_cuda_stream,
        ...     use_cuda_stream,
        ... )
        >>>
        >>> stream = get_cuda_stream(
        ...     device=torch.device("cuda:0"), purpose="pre-processing"
        ... )
        >>> with use_cuda_stream(stream):
        ...     pass  # enqueue torch work
        >>> if stream is not None:
        ...     stream.synchronize()
    """
    if device.type != "cuda":
        return None
    registry: Optional[Dict[Tuple[int, str], torch.cuda.Stream]] = getattr(
        _THREAD_LOCAL_STREAMS, "registry", None
    )
    if registry is None:
        registry = {}
        _THREAD_LOCAL_STREAMS.registry = registry
    key = (device.index or 0, purpose)
    if key not in registry:
        registry[key] = torch.cuda.Stream(device=device)
    return registry[key]
