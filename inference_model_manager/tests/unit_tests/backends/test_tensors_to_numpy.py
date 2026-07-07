"""_tensors_to_numpy must handle frozen dataclasses (SAM predictions)."""

from dataclasses import dataclass

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference_model_manager.backends.subproc import _tensors_to_numpy


@dataclass(frozen=True)
class _FrozenPrediction:
    masks: torch.Tensor
    scores: torch.Tensor


@dataclass
class _MutablePrediction:
    boxes: torch.Tensor


def test_frozen_dataclass_tensors_converted():
    prediction = _FrozenPrediction(masks=torch.ones(2, 4, 4), scores=torch.zeros(2))
    result = _tensors_to_numpy(prediction)
    assert isinstance(result.masks, np.ndarray)
    assert isinstance(result.scores, np.ndarray)


def test_mutable_dataclass_and_containers():
    result = _tensors_to_numpy(
        {"a": [_MutablePrediction(boxes=torch.ones(1, 4))], "b": (torch.zeros(3),)}
    )
    assert isinstance(result["a"][0].boxes, np.ndarray)
    assert isinstance(result["b"][0], np.ndarray)
