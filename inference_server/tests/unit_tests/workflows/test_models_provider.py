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
        self.records = []
        self.recorded_requests = []

    def resolve(self, model_id, api_key):
        return self.routes[model_id]

    def record_request(self, route, model_id_as_requested, path, *, alias=None):
        self.recorded_requests.append(
            (route.registry_id, model_id_as_requested, path, alias)
        )

    def ensure_loaded(self, route, api_key): ...

    def infer(self, route, api_key, action, images, params, record=True):
        self.calls.append((route.model_id, action, params, [i.data for i in images]))
        self.records.append(record)
        return [self.predictions[(route.model_id, action)] for _ in images]

    def infer_params_only(self, route, api_key, action, params, record=True):
        self.calls.append((route.model_id, action, params, None))
        self.records.append(record)
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
        actions={"infer"},
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


def test_add_model_records_the_request_under_the_model_id():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.add_model("ds/1", "k")

    assert bridge.recorded_requests == [("ds/1", "ds/1", "/workflows/run", None)]


def test_add_model_with_alias_records_the_alias_row_and_the_model_id_as_alias():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.add_model("ds/1", "k", model_id_alias="alias-1")

    assert bridge.recorded_requests == [("ds/1", "alias-1", "/workflows/run", "ds/1")]


def test_add_model_with_alias_equal_to_the_model_id_records_no_alias():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.add_model("ds/1", "k", model_id_alias="ds/1")

    assert bridge.recorded_requests == [("ds/1", "ds/1", "/workflows/run", None)]


def test_add_model_without_request_path_records_the_row_with_no_path():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, api_key="req-key")

    provider.add_model("ds/1", "k")

    assert bridge.recorded_requests == [("ds/1", "ds/1", "", None)]


def test_add_model_records_an_sdk_alias_as_passed():
    bridge = FakeSyncBridge()
    bridge.routes["yolov8n-640"] = Route(
        model_id="yolov8n-640",
        registry_id="coco/3",
        task_type="object-detection",
        action="infer",
    )
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.add_model("yolov8n-640", "k")

    assert bridge.recorded_requests == [
        ("coco/3", "yolov8n-640", "/workflows/run", None)
    ]


def test_add_model_records_nothing_when_the_model_does_not_resolve():
    bridge = FakeSyncBridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    with pytest.raises(KeyError):
        provider.add_model("ds/1", "k")

    assert bridge.recorded_requests == []


def test_inference_without_add_model_records_no_request():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.run_object_detection(
        "ds/1",
        [{"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)}],
        api_key="k",
        confidence=0.5,
    )
    provider.get_class_names("ds/1")

    assert bridge.calls and bridge.recorded_requests == []


def test_clip_comparison_records_the_core_model_it_registers():
    bridge = FakeSyncBridge()
    bridge.routes["clip/ViT-B-16"] = Route(
        model_id="clip/ViT-B-16",
        registry_id="clip/ViT-B-16",
        task_type="embedding",
        action="embed_images",
        actions={"embed_images", "embed_text", "compare"},
    )
    bridge.predictions[("clip/ViT-B-16", "embed_text")] = np.array([[1.0, 0.0]])
    provider = GatewayModelsProvider(bridge, None, "/workflows/run")

    provider.run_clip_comparison(
        subject="a",
        subject_type="text",
        prompt=["b"],
        prompt_type="text",
        version_id="ViT-B-16",
    )

    assert bridge.recorded_requests == [
        ("clip/ViT-B-16", "clip/ViT-B-16", "/workflows/run", None)
    ]


def test_pp_ocr_records_the_core_model_it_registers():
    bridge = FakeSyncBridge()
    bridge.routes["pp_ocr/small-small"] = Route(
        model_id="pp_ocr/small-small",
        registry_id="pp_ocr/small-small",
        task_type="structured-ocr",
        action="infer",
        actions={"infer"},
    )
    bridge.predictions[("pp_ocr/small-small", "infer")] = (["hello"], None)
    provider = GatewayModelsProvider(bridge, None, "/ws/workflows/wf")

    out = provider.run_pp_ocr(
        {"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)},
        api_key="k",
        text_detection="small",
        text_recognition="small",
    )

    assert out["result"] == "hello"
    assert bridge.recorded_requests == [
        ("pp_ocr/small-small", "pp_ocr/small-small", "/ws/workflows/wf", None)
    ]


def test_instance_segmentation_raw_responses():
    b = FakeSyncBridge()
    b.routes["ds/1"] = Route(
        model_id="ds/1",
        registry_id="ds/1",
        task_type="instance-segmentation",
        action="infer",
        actions={"infer"},
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


def test_classification_inference_kwargs_override_action_params():
    b = FakeSyncBridge()
    b.routes["cls/1"] = Route(
        model_id="cls/1",
        registry_id="cls/1",
        task_type="classification",
        action="infer",
        actions={"infer"},
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
        actions={"infer"},
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
        actions={"infer"},
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
        actions={"embed_images", "embed_text"},
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
        actions={"embed_images", "embed_text", "compare"},
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


class _TimedEmbeddingBridge(FakeSyncBridge):
    def __init__(self, clock, script):
        super().__init__()
        self.clock = clock
        self.script = script
        self.routes["clip/ViT-B-16"] = Route(
            model_id="clip/ViT-B-16",
            registry_id="clip/ViT-B-16",
            task_type="embedding",
            action="embed_images",
            actions={"embed_images", "embed_text", "compare"},
        )

    def _embed(self, params):
        duration = self.script.pop(0)
        if duration is None:
            raise RuntimeError("model failed")
        self.clock[0] += duration
        return np.array([[1.0, 0.0]] * len(params["texts"]))

    def infer(self, route, api_key, action, images, params, record=True):
        self.records.append(record)
        return [self._embed(params) for _ in images]

    def infer_params_only(self, route, api_key, action, params, record=True):
        self.records.append(record)
        return self._embed(params)


@pytest.fixture
def metrics_clock(monkeypatch):
    import inference_server.prometheus as prometheus_mod

    now = [1000.0]
    monkeypatch.setattr(
        prometheus_mod, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    prometheus_mod.MODEL_METRICS.clear()
    yield now
    prometheus_mod.MODEL_METRICS.clear()


def _clip_metrics():
    from inference_server.prometheus import MODEL_METRICS

    MODEL_METRICS.observe_loaded(["clip/ViT-B-16"])
    return MODEL_METRICS.metrics()["clip/ViT-B-16"]


def test_clip_comparison_records_one_event_spanning_every_call(metrics_clock):
    bridge = _TimedEmbeddingBridge(metrics_clock, [0.01, 0.1])

    out = GatewayModelsProvider(bridge, api_key=None).run_clip_comparison(
        subject="a",
        subject_type="text",
        prompt=["b"],
        prompt_type="text",
        version_id="ViT-B-16",
    )

    assert out["similarity"] == [1.0]
    assert bridge.records == [False, False]
    metrics = _clip_metrics()
    assert metrics["num_inferences"] == 1
    assert metrics["num_errors"] == 0
    assert metrics["avg_inference_time"] == pytest.approx(0.11)


def test_clip_comparison_with_failing_call_records_one_error(metrics_clock):
    bridge = _TimedEmbeddingBridge(metrics_clock, [0.01, None])

    with pytest.raises(RuntimeError):
        GatewayModelsProvider(bridge, api_key=None).run_clip_comparison(
            subject="a",
            subject_type="text",
            prompt=["b"],
            prompt_type="text",
            version_id="ViT-B-16",
        )

    metrics = _clip_metrics()
    assert metrics["num_inferences"] == 0
    assert metrics["num_errors"] == 1


def test_sam2_segmentation_returns_responses():
    b = FakeSyncBridge()
    b.routes["sam2/hiera_large"] = Route(
        model_id="sam2/hiera_large",
        registry_id="sam2/hiera_large",
        task_type="interactive-instance-segmentation",
        action="embed",
        actions={"embed", "segment_with_visual_prompts"},
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
