from multiprocessing import shared_memory

import numpy as np
import pytest
import torch

from inference.enterprise.parallel.utils import (
    SharedMemoryMetadata,
    _image_to_numpy,
    shm_manager,
)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.uint8])
def test_numpy_preprocessing_keeps_array(dtype) -> None:
    image = np.arange(24, dtype=dtype).reshape(3, 2, 4)

    assert _image_to_numpy(image) is image


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA device required"
            ),
        ),
    ],
)
def test_tensor_preprocessing_shared_memory_round_trip(
    dtype, noncontiguous: bool, device: str
) -> None:
    expected = np.arange(24).reshape(3, 2, 4)
    image = torch.tensor(expected, dtype=dtype, device=device, requires_grad=True)
    if noncontiguous:
        image = image.transpose(1, 2)
        expected = expected.transpose(0, 2, 1)

    array = _image_to_numpy(image)
    shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
    with shm_manager(shm, unlink_on_success=True):
        shared = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        shared[:] = array
        metadata = SharedMemoryMetadata(shm.name, array.shape, array.dtype.name)

        # Reopen by the serialized metadata, as the inference process does.
        with shm_manager(metadata.shm_name) as (reader,):
            received = np.ndarray(
                metadata.array_shape, dtype=metadata.array_dtype, buffer=reader.buf
            )
            np.testing.assert_array_equal(received, expected)
            assert received.dtype == (
                np.dtype("float16") if dtype == torch.float16 else np.dtype("float32")
            )
