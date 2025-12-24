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
    context = cuda_device.retain_primary_context()
    with use_cuda_context(context) as ctx:
        yield ctx


@contextlib.contextmanager
def use_cuda_context(context: cuda.Context) -> Generator[cuda.Context, None, None]:
    context.push()
    try:
        yield context
    finally:
        context.pop()
