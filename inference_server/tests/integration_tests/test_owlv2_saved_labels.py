"""Saved-label metadata across the in-process serving boundaries."""

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.models.roboflow.object_detection.v3_tensor import (
    _class_names_map,
    _filter_classes_native,
)

from inference_model_manager.model_manager import ModelManager
from inference_model_manager.serializers_typed import serialize_detections_compact
from inference_models import Detections
from inference_models.models.owlv2.owlv2_hf import OWLv2HF
from inference_server.legacy.bridge import Route, _apply_metadata
from inference_server.legacy.entities import ObjectDetectionInferenceRequest
from inference_server.legacy.translation import repack_prediction
from inference_server.workflows.models_provider import GatewayModelsProvider


@pytest.fixture
def saved_label_manager(monkeypatch):
    # Exercise the real metadata property without downloading model weights.
    # The existing OWLv2 model tests cover inference and prediction metadata.
    model = OWLv2HF.__new__(OWLv2HF)
    model._class_names = ["label-a", "label-b"]
    monkeypatch.setattr(
        "inference_model_manager.pipelines.load_model", lambda *args, **kwargs: model
    )
    manager = ModelManager()
    try:
        manager.load("uploaded/1", api_key="", device="cpu", warmup_iters=0)
        yield manager, model
    finally:
        manager.shutdown()


def _route_from_stats(manager):
    entry = next(m for m in manager.stats()["models"] if m["model_id"] == "uploaded/1")
    route = Route("uploaded/1", "uploaded/1", "object-detection", "infer")
    _apply_metadata(route, entry)

    return route


def _detections(*, names, empty=False):
    detections = Detections(
        xyxy=torch.tensor([[4, 4, 12, 12], [20, 20, 28, 28]], dtype=torch.int32),
        class_id=torch.tensor([0, 1], dtype=torch.int32),
        confidence=torch.tensor([0.9, 0.8]),
        image_metadata={"class_names": names},
    )
    if empty:
        detections.xyxy = detections.xyxy[:0]
        detections.class_id = detections.class_id[:0]
        detections.confidence = detections.confidence[:0]

    return detections


@pytest.mark.parametrize("empty", [False, True])
def test_saved_labels_reach_legacy_response(saved_label_manager, empty):
    manager, model = saved_label_manager
    route = _route_from_stats(manager)
    detections = _detections(names=model.saved_class_names, empty=empty)
    request = ObjectDetectionInferenceRequest(
        model_id="uploaded/1",
        image={"type": "numpy", "value": np.zeros((16, 32, 3), dtype=np.uint8)},
    )

    response = repack_prediction(
        task_type=route.task_type,
        action=route.action,
        prediction=detections,
        dims=(32, 16),
        route=route,
        request=request,
    )

    assert route.class_names == ["label-a", "label-b"]
    assert [p.class_name for p in response.predictions] == (
        [] if empty else ["label-a", "label-b"]
    )


@pytest.mark.parametrize("empty", [False, True])
def test_saved_labels_reach_workflow_class_filter(saved_label_manager, empty):
    manager, model = saved_label_manager
    route = _route_from_stats(manager)
    provider = GatewayModelsProvider(bridge=None, api_key=None)
    provider._routes["uploaded/1"] = route
    detections = _detections(names=model.saved_class_names, empty=empty)

    names = provider.get_class_names("uploaded/1")
    filtered = _filter_classes_native(
        detections, class_filter=["label-b"], class_names=_class_names_map(names)
    )

    assert names == ["label-a", "label-b"]
    assert filtered.class_id.tolist() == ([] if empty else [1])
    assert filtered.xyxy.tolist() == ([] if empty else [[20, 20, 28, 28]])


@pytest.mark.parametrize("names", [["a", "b"], ["reference-a", "reference-b"]])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_request_labels_remain_independent_of_saved_metadata(
    saved_label_manager, names, batch_size, empty
):
    manager, model = saved_label_manager
    route = _route_from_stats(manager)
    predictions = [_detections(names=names, empty=empty) for _ in range(batch_size)]

    result = serialize_detections_compact(predictions, model)

    assert result["class_names"] == names
    assert route.class_names == ["label-a", "label-b"]
    assert _route_from_stats(manager).class_names == ["label-a", "label-b"]
