"""Regression tests for Grounding DINO adapter threshold forwarding."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import inference
from inference.core.entities.requests.groundingdino import GroundingDINOInferenceRequest


@pytest.fixture
def adapter_class(monkeypatch):
    # Load the real adapter without importing the optional Grounding DINO models
    # through the legacy package __init__ or the backend's type annotation.
    backend_name = "inference_models.models.grounding_dino.grounding_dino_torch"
    backend_module = ModuleType(backend_name)
    backend_module.GroundingDinoForObjectDetectionTorch = object
    monkeypatch.setitem(sys.modules, backend_name, backend_module)
    adapter_path = (
        Path(inference.__file__).parent
        / "models/grounding_dino/grounding_dino_inference_models.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_grounding_dino_adapter_under_test", adapter_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.InferenceModelsGroundingDINOAdapter


@pytest.mark.parametrize("box_threshold,text_threshold", [(0.91, 0.73), (0.23, 0.87)])
def test_infer_from_request_preserves_custom_thresholds(
    adapter_class, box_threshold, text_threshold
):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    request = GroundingDINOInferenceRequest(
        image={"type": "numpy", "value": image},
        text=["car"],
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        class_agnostic_nms=True,
    )
    detections = SimpleNamespace(
        xyxy=np.array([[10.0, 12.0, 30.0, 36.0]]),
        confidence=np.array([0.97]),
        class_id=np.array([0]),
    )
    backend_infer = Mock(return_value=[detections])
    adapter = adapter_class.__new__(adapter_class)
    adapter._model = SimpleNamespace(infer=backend_infer, resolved_model=None)
    adapter.preproc_image = Mock(return_value=image)

    response = adapter.infer_from_request(request)

    backend_infer.assert_called_once_with(
        images=image,
        classes=["car"],
        box_confidence=box_threshold,
        text_confidence=text_threshold,
        class_agnostic_nms=True,
    )
    prediction = response.predictions[0]
    assert (prediction.x, prediction.y, prediction.width, prediction.height) == (
        20.0,
        24.0,
        20.0,
        24.0,
    )
    assert prediction.class_name == "car"
    assert prediction.confidence == 0.97
