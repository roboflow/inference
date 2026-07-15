import numpy as np
import pytest
import torch

from inference_models.errors import ModelRuntimeError
from inference_models.models.rfdetr.pre_processing import (
    RFDETR_PREPROCESSOR_IMPLEMENTATIONS,
    RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1,
    resolve_rfdetr_preprocessor,
)
from inference_models.models.rfdetr.triton_universal_preprocess_runtime import (
    UniversalFastPreprocessRuntime,
    _build_metadata_batch,
    _canonicalize_batch,
)


@pytest.mark.parametrize(
    "images",
    [
        np.zeros((2, 8, 9, 3), dtype=np.uint8),
        [np.zeros((8, 9, 3), dtype=np.uint8) for _ in range(2)],
        torch.zeros((2, 3, 8, 9), dtype=torch.uint8),
        torch.zeros((2, 8, 9, 3), dtype=torch.uint8),
    ],
)
def test_canonicalize_uint8_cpu_batches(images) -> None:
    batch = _canonicalize_batch(images)

    assert batch.kind == "uint8"
    assert (batch.height, batch.width) == (8, 9)
    assert len(batch.items) == 2
    assert all(tuple(item.shape) == (8, 9, 3) for item in batch.items)


@pytest.mark.parametrize(
    "images",
    [
        torch.zeros((2, 3, 8, 9), dtype=torch.float32),
        torch.zeros((2, 8, 9, 3), dtype=torch.float16),
    ],
)
def test_canonicalize_float_cpu_batches(images) -> None:
    batch = _canonicalize_batch(images)

    assert batch.kind == "float"
    assert (batch.height, batch.width) == (8, 9)
    assert all(tuple(item.shape) == (3, 8, 9) for item in batch.items)


def test_canonicalize_float_numpy_matches_reference_uint8_conversion() -> None:
    image = np.array([[[0.0, 0.5, 1.5]]], dtype=np.float32)

    batch = _canonicalize_batch(image)

    np.testing.assert_array_equal(
        batch.items[0],
        np.array([[[0, 127, 255]]], dtype=np.uint8),
    )


def test_canonicalize_rejects_mixed_semantics() -> None:
    with pytest.raises(ModelRuntimeError, match="homogeneous batch"):
        _canonicalize_batch(
            [
                torch.zeros((3, 8, 9), dtype=torch.uint8),
                torch.zeros((3, 8, 9), dtype=torch.float32),
            ]
        )


def test_canonicalize_rejects_mixed_source_dimensions() -> None:
    with pytest.raises(ModelRuntimeError, match="equal source dimensions"):
        _canonicalize_batch(
            [
                np.zeros((8, 9, 3), dtype=np.uint8),
                np.zeros((10, 9, 3), dtype=np.uint8),
            ]
        )


def test_metadata_batch_describes_stretch() -> None:
    metadata = _build_metadata_batch(
        batch_size=2,
        source_h=8,
        source_w=10,
        target_h=16,
        target_w=40,
    )

    assert len(metadata) == 2
    assert metadata[0].scale_height == 2
    assert metadata[0].scale_width == 4
    assert metadata[0].static_crop_offset.crop_width == 10
    assert metadata[0].static_crop_offset.crop_height == 8


def test_universal_candidate_is_explicitly_selectable() -> None:
    assert (
        resolve_rfdetr_preprocessor(RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1)
        == RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    )
    metadata = RFDETR_PREPROCESSOR_IMPLEMENTATIONS[
        RFDETR_PREPROCESSOR_TRITON_UNIVERSAL_V1
    ]
    assert metadata["validated_environments"] == ()
    assert metadata["fallback_id"] == "base"


def test_universal_runtime_requires_cuda_device() -> None:
    with pytest.raises(ModelRuntimeError, match="requires a CUDA target"):
        UniversalFastPreprocessRuntime(device=torch.device("cpu"))
