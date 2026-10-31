from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Iterator, Optional, TypeVar

import torch


F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def profiling_range(name: str, *, enabled: bool = True) -> Iterator[None]:
    """Create an NVTX range when CUDA profiling is available.

    Args:
        name (str): NVTX range name.
        enabled (bool): Whether the range is enabled.

    Returns:
        Context manager that is a no-op when NVTX is unavailable.
    """
    if not enabled or not torch.cuda.is_available():
        with nullcontext():
            yield
        return

    try:
        import torch.cuda.nvtx as nvtx
    except (ImportError, RuntimeError):
        with nullcontext():
            yield
        return

    with nvtx.range(name):
        yield


def profiling_range_if_cuda(
    name: str,
    *,
    device: Optional[torch.device] = None,
    tensor: Optional[torch.Tensor] = None,
    enabled: bool = True,
):
    """Create an NVTX range only for CUDA devices or tensors.

    Args:
        name (str): NVTX range name.
        device (Optional[torch.device]): Optional explicit device.
        tensor (Optional[torch.Tensor]): Optional tensor used to infer device.
        enabled (bool): Whether the range is enabled.

    Returns:
        NVTX context manager or no-op context manager.
    """
    if not enabled:
        context_manager = nullcontext()

        return context_manager

    resolved_device = device
    if resolved_device is None and tensor is not None:
        resolved_device = tensor.device

    if (
        resolved_device is None
        or resolved_device.type != "cuda"
        or not torch.cuda.is_available()
    ):
        context_manager = nullcontext()

        return context_manager

    try:
        import torch.cuda.nvtx as nvtx
    except (ImportError, RuntimeError):
        context_manager = nullcontext()

        return context_manager

    context_manager = nvtx.range(name)

    return context_manager


def nvtx_range(name: str, *, enabled: bool = True) -> Callable[[F], F]:
    """Decorate a function with an NVTX range.

    Args:
        name (str): NVTX range name.
        enabled (bool): Whether the range is enabled.

    Returns:
        Function decorator.
    """
    def decorator(func: F) -> F:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with profiling_range(name, enabled=enabled):
                result = func(*args, **kwargs)

                return result

        return wrapper  # type: ignore[return-value]

    range_decorator = decorator

    return range_decorator
