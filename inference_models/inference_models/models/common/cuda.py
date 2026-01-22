import contextlib
from typing import Generator

from inference_models.errors import MissingDependencyError

try:
    import pycuda.driver as cuda
except ImportError as import_error:
    raise MissingDependencyError(
        message="TODO",
        help_url="https://todo",
    ) from import_error


@contextlib.contextmanager
def use_primary_cuda_context(
    cuda_device: cuda.Device,
) -> Generator[cuda.Context, None, None]:
    """Context manager for using a CUDA device's primary context.

    Retains and activates the primary CUDA context for a device, ensuring proper
    cleanup when exiting the context. This is useful when working with TensorRT
    engines or other CUDA operations that require an active context.

    Args:
        cuda_device: PyCUDA Device object representing the CUDA device.

    Yields:
        cuda.Context: The active CUDA context for the device.

    Examples:
        Use primary CUDA context for TensorRT inference:

        >>> from inference_models.developer_tools import use_primary_cuda_context
        >>> import pycuda.driver as cuda
        >>>
        >>> cuda.init()
        >>> device = cuda.Device(0)  # GPU 0
        >>>
        >>> with use_primary_cuda_context(device) as ctx:
        ...     # Perform CUDA operations
        ...     # TensorRT engine execution, etc.
        ...     pass

    Note:
        - Requires PyCUDA to be installed
        - Automatically pushes and pops the context
        - Use this for TensorRT models or custom CUDA operations

    See Also:
        - `use_cuda_context()`: Use an existing CUDA context
    """
    context = cuda_device.retain_primary_context()
    with use_cuda_context(context) as ctx:
        yield ctx


@contextlib.contextmanager
def use_cuda_context(context: cuda.Context) -> Generator[cuda.Context, None, None]:
    """Context manager for using an existing CUDA context.

    Pushes a CUDA context onto the context stack, making it active for the duration
    of the context manager, then pops it when exiting. This ensures proper context
    management for CUDA operations.

    Args:
        context: PyCUDA Context object to activate.

    Yields:
        cuda.Context: The active CUDA context.

    Examples:
        Use an existing CUDA context:

        >>> from inference_models.developer_tools import use_cuda_context
        >>> import pycuda.driver as cuda
        >>>
        >>> cuda.init()
        >>> device = cuda.Device(0)
        >>> context = device.retain_primary_context()
        >>>
        >>> with use_cuda_context(context) as ctx:
        ...     # Perform CUDA operations
        ...     pass

    Note:
        - Requires PyCUDA to be installed
        - Automatically pushes context on entry and pops on exit
        - Context is popped even if an exception occurs

    See Also:
        - `use_primary_cuda_context()`: Convenience wrapper for primary context
    """
    context.push()
    try:
        yield context
    finally:
        context.pop()
