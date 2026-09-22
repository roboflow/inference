import inspect
from types import SimpleNamespace

import numpy as np
import pytest
from roboflow_workflows.prototypes.models_provider import (
    InferenceResultsDC,
    ModelsProvider,
)

from inference_server.legacy.bridge import Route
from inference_server.workflows.models_provider import GatewayModelsProvider


class FakeSyncBridge:
    accepts_ndarray = True

    def __init__(self):
        self.routes = {}
        self.calls = []
        self.predictions = {}

    def resolve(self, model_id, api_key):
        return self.routes[model_id]

    def ensure_loaded(self, route, api_key): ...

    def infer(self, route, api_key, action, images, params):
        self.calls.append((route.model_id, action, params, [i.data for i in images]))
        return [self.predictions[(route.model_id, action)] for _ in images]

    def infer_params_only(self, route, api_key, action, params):
        self.calls.append((route.model_id, action, params, None))
        return self.predictions[(route.model_id, action)]

    def fetch_image(self, url):
        raise AssertionError("not used")

    def __contains__(self, model_id):
        return model_id in self.routes


def _od_bridge():
    b = FakeSyncBridge()
    b.routes["ds/1"] = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="object-detection",
        action="infer",
        tasks={"infer"},
        class_names=["cat"],
    )
    b.predictions[("ds/1", "infer")] = SimpleNamespace(
        xyxy=np.array([[0, 0, 2, 2]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    return b


def test_run_object_detection_returns_legacy_dicts():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, api_key="req-key")
    provider.add_model("ds/1", "k")
    img = np.zeros((4, 6, 3), dtype=np.uint8)
    out = provider.run_object_detection(
        "ds/1", [{"type": "numpy_object", "value": img}], api_key="k", confidence=0.5
    )
    assert (
        out[0]["image"] == {"width": 6, "height": 4}
        and out[0]["predictions"][0]["class"] == "cat"
    )
    assert bridge.calls[0][2]["confidence"] == 0.5 and bridge.calls[0][3][0] is img


def test_instance_segmentation_raw_responses():
    b = FakeSyncBridge()
    b.routes["ds/1"] = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="instance-segmentation",
        action="infer",
        tasks={"infer"},
        class_names=["cat"],
    )
    mask = np.zeros((1, 4, 6), dtype=bool)
    mask[0, 1:3, 1:3] = True
    b.predictions[("ds/1", "infer")] = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 3]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        mask=mask,
    )
    out = GatewayModelsProvider(b, api_key=None).run_instance_segmentation(
        "ds/1",
        [{"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)}],
        confidence=0.5,
        return_raw_responses=True,
    )
    assert (
        isinstance(out, InferenceResultsDC)
        and out.raw_responses[0].predictions[0].class_name == "cat"
    )


def test_none_confidence_is_forwarded_and_rejected_by_the_request_entity():
    import pydantic

    from inference_server.legacy.entities import ObjectDetectionInferenceRequest

    provider = GatewayModelsProvider(_od_bridge(), api_key="req-key")
    with pytest.raises(pydantic.ValidationError):
        ObjectDetectionInferenceRequest(
            model_id="ds/1", image=[], confidence=None, source="workflow-execution"
        )
    with pytest.raises(pydantic.ValidationError):
        provider.run_object_detection(
            "ds/1",
            [{"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)}],
            api_key="k",
            confidence=None,
        )


def test_classification_inference_kwargs_override_task_params():
    b = FakeSyncBridge()
    b.routes["cls/1"] = Route(
        model_id="cls/1",
        registry_id="cls/1",
        task_type="classification",
        action="infer",
        tasks={"infer"},
        class_names=["cat"],
    )
    b.predictions[("cls/1", "infer")] = SimpleNamespace(
        confidence=np.array([0.9]), class_id=np.array([0])
    )
    GatewayModelsProvider(b, api_key=None).run_classification(
        "cls/1",
        [{"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)}],
        confidence=0.5,
        inference_kwargs={"confidence": 0.9},
    )
    assert b.calls[0][2]["confidence"] == 0.9


def test_get_class_names_and_contains():
    p = GatewayModelsProvider(_od_bridge(), api_key=None)
    assert p.get_class_names("ds/1") == ["cat"] and "ds/1" in p and "x/1" not in p


def test_stream_pipeline_members_report_no_pipeline():
    p = GatewayModelsProvider(_od_bridge(), api_key=None)
    assert (
        p.model_supports_stream_pipeline("ds/1") is False
        and p.get_model_pipeline_depth("ds/1") == 0
    )


def test_tensor_native_is_501():
    from inference_server.legacy.errors import LegacyHTTPError

    with pytest.raises(LegacyHTTPError):
        GatewayModelsProvider(_od_bridge(), api_key=None).run_tensor_native_inference(
            "ds/1"
        )


def test_artifact_cache_property_uses_shared_blob_cache(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "inference_models.utils.model_blob_cache.get_shared_model_blob_cache",
        lambda: sentinel,
    )
    provider = GatewayModelsProvider(_od_bridge(), api_key=None)
    assert provider.content_addressed_artifact_cache is sentinel
    assert provider.content_addressed_artifact_cache is sentinel


def test_key_remembered_by_add_model_is_reused_by_keyless_calls():
    class RecordingBridge(FakeSyncBridge):
        def resolve(self, model_id, api_key):
            self.calls.append(("resolve", model_id, api_key))
            return super().resolve(model_id, api_key)

    b = RecordingBridge()
    b.routes["depth-anything-v2/small"] = Route(
        model_id="depth-anything-v2/small",
        registry_id="depth-anything-v2/small",
        task_type="depth-estimation",
        action="infer",
        tasks={"infer"},
    )
    b.predictions[("depth-anything-v2/small", "infer")] = np.array(
        [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
    )
    provider = GatewayModelsProvider(b, api_key="req-key")
    provider.add_model("depth-anything-v2/small", "model-key")
    provider.run_depth_estimation(
        "depth-anything-v2/small",
        {"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)},
    )
    assert ("resolve", "depth-anything-v2/small", "model-key") in b.calls[1:]
    provider2 = GatewayModelsProvider(b, api_key="req-key")
    provider2.run_depth_estimation(
        "depth-anything-v2/small",
        {"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)},
    )
    assert ("resolve", "depth-anything-v2/small", "req-key") in b.calls


def test_depth_estimation_returns_normalized_depth_and_image():
    b = FakeSyncBridge()
    b.routes["depth-anything-v2/small"] = Route(
        model_id="depth-anything-v2/small",
        registry_id="depth-anything-v2/small",
        task_type="depth-estimation",
        action="infer",
        tasks={"infer"},
    )
    b.predictions[("depth-anything-v2/small", "infer")] = np.array(
        [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
    )
    out = GatewayModelsProvider(b, api_key=None).run_depth_estimation(
        "depth-anything-v2/small",
        {"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)},
    )
    assert out["normalized_depth"].shape == (2, 2)
    assert isinstance(out["image"].base64_image, str)


def test_clip_text_embedding():
    b = FakeSyncBridge()
    b.routes["clip/ViT-B-16"] = Route(
        model_id="clip/ViT-B-16",
        registry_id="clip/ViT-B-16",
        task_type="embedding",
        action="embed_images",
        tasks={"embed_images", "embed_text"},
    )
    b.predictions[("clip/ViT-B-16", "embed_text")] = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = GatewayModelsProvider(b, api_key=None).run_clip_text_embedding(
        "clip/ViT-B-16", "ViT-B-16", ["a", "b"], api_key="k"
    )
    assert out == [[1.0, 2.0], [3.0, 4.0]] and b.calls[0][2] == {"texts": ["a", "b"]}


def test_clip_comparison_returns_similarity_payload():
    b = FakeSyncBridge()
    b.routes["clip/ViT-B-16"] = Route(
        model_id="clip/ViT-B-16",
        registry_id="clip/ViT-B-16",
        task_type="embedding",
        action="embed_images",
        tasks={"embed_images", "embed_text", "compare"},
    )
    b.predictions[("clip/ViT-B-16", "embed_images")] = np.array([[1.0, 0.0]])
    b.predictions[("clip/ViT-B-16", "embed_text")] = np.array([[1.0, 0.0], [0.0, 1.0]])
    out = GatewayModelsProvider(b, api_key=None).run_clip_comparison(
        subject={"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)},
        subject_type="image",
        prompt=["a", "b"],
        prompt_type="text",
        version_id="ViT-B-16",
    )
    assert out["similarity"] == [1.0, 0.0]


def test_sam2_segmentation_returns_responses():
    b = FakeSyncBridge()
    b.routes["sam2/hiera_large"] = Route(
        model_id="sam2/hiera_large",
        registry_id="sam2/hiera_large",
        task_type="interactive-instance-segmentation",
        action="embed",
        tasks={"embed", "segment_with_visual_prompts"},
    )
    mask = np.zeros((1, 4, 6), dtype=bool)
    mask[0, 1:3, 1:3] = True
    b.predictions[("sam2/hiera_large", "segment_with_visual_prompts")] = (
        SimpleNamespace(masks=mask, scores=np.array([0.7]))
    )
    out = GatewayModelsProvider(b, api_key=None).run_sam2_segmentation(
        "sam2/hiera_large",
        {"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)},
        [{"points": [{"x": 2, "y": 2, "positive": True}]}],
        version_id="hiera_large",
    )
    assert len(out) == 1 and out[0].predictions[0].confidence == pytest.approx(0.7)


def test_sam3_3d_objects_and_action_recognition_are_501():
    from inference_server.legacy.errors import LegacyHTTPError

    provider = GatewayModelsProvider(_od_bridge(), api_key=None)
    with pytest.raises(LegacyHTTPError):
        provider.run_sam3_3d_objects("sam3/sam3_final", None, None)
    with pytest.raises(LegacyHTTPError):
        provider.load_action_recognition_model("ds/1")


def test_provider_covers_protocol():
    for name, member in inspect.getmembers(ModelsProvider, inspect.isfunction):
        if name.startswith("_") and name != "__contains__":
            continue
        impl = getattr(GatewayModelsProvider, name, None)
        assert impl is not None, name
        assert list(inspect.signature(impl).parameters) == list(
            inspect.signature(member).parameters
        ), name
