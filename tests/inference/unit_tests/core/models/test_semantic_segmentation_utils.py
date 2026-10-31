import numpy as np
import pytest
import torch

from inference.core.models.semantic_segmentation_utils import (
    present_class_ids_from_label_map,
)


@pytest.mark.parametrize(
    "label_map",
    [
        np.zeros((13, 17), dtype=np.uint8),
        np.full((13, 17), 255, dtype=np.uint8),
        np.full((13, 17), 7, dtype=np.uint8),
        np.array([[0, 1], [254, 255]], dtype=np.uint8),
    ],
)
def test_present_class_ids_matches_np_unique_on_edge_cases(
    label_map: np.ndarray,
) -> None:
    result = present_class_ids_from_label_map(torch.from_numpy(label_map))

    assert result == np.unique(label_map).tolist()


def test_present_class_ids_matches_np_unique_on_random_map() -> None:
    rng = np.random.default_rng(3)
    label_map = rng.choice(np.array([0, 3, 7, 42, 255], dtype=np.uint8), size=(101, 57))

    result = present_class_ids_from_label_map(torch.from_numpy(label_map))

    assert result == np.unique(label_map).tolist()
    assert result == sorted(result)


def test_present_class_ids_handles_non_contiguous_tensor() -> None:
    rng = np.random.default_rng(5)
    label_map = rng.choice(np.array([0, 9], dtype=np.uint8), size=(64, 64))
    view = torch.from_numpy(label_map)[::2, ::2]

    result = present_class_ids_from_label_map(view)

    assert result == np.unique(label_map[::2, ::2]).tolist()


def test_present_class_ids_falls_back_to_int64_when_native_kernel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # simulate builds whose bincount only ships an int64 kernel
    original_bincount = torch.bincount

    def _int64_only_bincount(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if tensor.dtype != torch.int64:
            raise RuntimeError('"bincount" not implemented for the given dtype')
        return original_bincount(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "bincount", _int64_only_bincount)
    rng = np.random.default_rng(9)
    label_map = rng.choice(np.array([0, 5, 200, 255], dtype=np.uint8), size=(37, 53))

    result = present_class_ids_from_label_map(torch.from_numpy(label_map))

    assert result == np.unique(label_map).tolist()


def test_present_class_ids_reraises_cuda_oom_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def _oom_bincount(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        calls.append(tensor.dtype)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    monkeypatch.setattr(torch, "bincount", _oom_bincount)

    with pytest.raises(torch.cuda.OutOfMemoryError):
        present_class_ids_from_label_map(torch.zeros((4, 4), dtype=torch.uint8))

    # the fallback must NOT have retried with a widened copy
    assert calls == [torch.uint8]
