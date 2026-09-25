"""Unit tests for the Detections Class Router block."""

import numpy as np
import pytest
import supervision as sv
from roboflow_workflows.core_steps.formatters.detections_class_router.v1 import (
    BlockManifest as RouterManifest,
)
from roboflow_workflows.core_steps.formatters.detections_class_router.v1 import (
    DetectionsClassRouterBlockV1,
)


def _classed(*items):
    """items: (class_name, confidence)"""
    det = sv.Detections(
        xyxy=(
            np.array([[0.0, 0.0, 10.0, 10.0]] * len(items))
            if items
            else np.empty((0, 4))
        ),
        confidence=np.array([c for _, c in items], dtype=float),
        class_id=np.arange(len(items)),
    )
    det.data["class_name"] = np.array([n for n, _ in items], dtype=object)
    return det


ROUTES = {"dog": "Dog", "cat": "Cat", "apple": "Apple"}


def _route(predictions, **overrides):
    kwargs = dict(
        predictions=predictions,
        routes=ROUTES,
        default_value=None,
        confidence_threshold=0.0,
        case_insensitive=True,
    )
    kwargs.update(overrides)
    return DetectionsClassRouterBlockV1().run(**kwargs)


def test_router_maps_most_confident_routed_class():
    result = _route(_classed(("cat", 0.6), ("dog", 0.9)))

    assert result == {"value": "Dog", "matched_class": "dog", "matched": True}


def test_router_ignores_unrouted_classes_even_when_more_confident():
    # the bug this block exists to kill: a background 'person' outranking the cat
    result = _route(_classed(("person", 0.98), ("book", 0.9), ("cat", 0.55)))

    assert result["value"] == "Cat"


def test_router_emits_none_when_nothing_routed_and_no_default():
    assert _route(_classed(("person", 0.9))) == {
        "value": None,
        "matched_class": None,
        "matched": False,
    }
    assert _route(sv.Detections.empty())["matched"] is False


def test_router_uses_default_value_when_nothing_routed():
    result = _route(_classed(("car", 0.9)), default_value="Fireworks")

    assert result["value"] == "Fireworks"
    assert result["matched"] is False


def test_router_respects_confidence_threshold_and_case():
    assert _route(_classed(("Dog", 0.3)), confidence_threshold=0.5)["matched"] is False
    assert _route(_classed(("DOG", 0.8)))["value"] == "Dog"
    assert _route(_classed(("DOG", 0.8)), case_insensitive=False)["matched"] is False


def test_router_manifest_requires_at_least_one_route():
    with pytest.raises(ValueError):
        RouterManifest.model_validate(
            {
                "type": "roboflow_core/detections_class_router@v1",
                "name": "r",
                "predictions": "$steps.model.predictions",
                "routes": {},
            }
        )


def _tensor_detections(class_names, confidences, boxes=None):
    """inference_models.Detections as produced with ENABLE_TENSOR_DATA_REPRESENTATION."""
    torch = pytest.importorskip("torch")
    from inference_models.models.base.object_detection import Detections

    lookup = {i: n for i, n in enumerate(dict.fromkeys(class_names))}
    inverse = {n: i for i, n in lookup.items()}
    boxes = boxes or [(0, 0, 10, 10)] * len(class_names)
    return Detections(
        xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor([inverse[n] for n in class_names], dtype=torch.long),
        confidence=torch.tensor(confidences, dtype=torch.float32),
        image_metadata={"class_names": lookup},
    )


def test_router_reads_tensor_native_detections():
    predictions = _tensor_detections(["person", "cup", "cat"], [0.9, 0.7, 0.8])

    result = _route(predictions=predictions, routes={"cup": "Coffee", "cat": "Cat"})

    assert result == {"value": "Cat", "matched_class": "cat", "matched": True}


def test_router_treats_empty_tensor_native_detections_as_nothing_routed():
    predictions = _tensor_detections([], [])

    result = _route(predictions=predictions, routes={"cup": "Coffee"})

    assert result == {"value": None, "matched_class": None, "matched": False}
