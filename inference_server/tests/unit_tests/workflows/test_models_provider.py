import asyncio
import inspect
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from roboflow_workflows.prototypes.models_provider import (
    InferenceResultsDC,
    ModelsProvider,
)

from inference_models.errors import ModelInputError, ModelNotFoundError

from inference_server.gateway import ModelManagerGateway
from inference_server.legacy.bridge import (
    LegacyModelBridge,
    LoopBridge,
    Route,
    SyncLegacyBridge,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.routing import routing_key
from inference_server.workflows.models_provider import GatewayModelsProvider
from tests.unit_tests.legacy import conftest as legacy_conftest


class FakeSyncBridge:
    accepts_ndarray = True

    def __init__(self):
        self.routes = {}
        self.calls = []
        self.predictions = {}
        self.records = []
        self.recorded_requests = []
        self.resolved_rows = []

    def resolve(
        self, model_id, api_key, *, row_key=None, path="", alias=None, instance=""
    ):
        self.resolved_rows.append((model_id, row_key, path, alias))
        return self.routes[routing_key(model_id, instance)]

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


def test_add_model_hands_the_row_key_path_and_alias_to_the_load():
    bridge = FakeSyncBridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    with pytest.raises(KeyError):
        provider.add_model("ds/1", "k", model_id_alias="alias-1")
    with pytest.raises(KeyError):
        provider.add_model("ds/2", "k")

    assert bridge.resolved_rows == [
        ("ds/1", "alias-1", "/workflows/run", "ds/1"),
        ("ds/2", "ds/2", "/workflows/run", None),
    ]


def test_inference_hands_no_row_to_the_load():
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.run_object_detection(
        "ds/1",
        [{"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)}],
        api_key="k",
        confidence=0.5,
    )

    assert bridge.resolved_rows == [("ds/1", None, "", None)]


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
    p.add_model("ds/1", "k")
    assert (
        p.model_supports_stream_pipeline("ds/1") is False
        and p.get_model_pipeline_depth("ds/1") == 1
        and p.flush_model_stream_pipeline("ds/1") is None
        and p.shutdown_model_stream_pipeline("ds/1") is None
    )


def test_stream_pipeline_members_report_legacy_defaults_for_an_unknown_model():
    p = GatewayModelsProvider(_od_bridge(), api_key=None)
    assert (
        p.model_supports_stream_pipeline("x/1") is False
        and p.get_model_pipeline_depth("x/1") == 1
        and p.flush_model_stream_pipeline("x/1") is None
        and p.shutdown_model_stream_pipeline("x/1") is None
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
        def resolve(self, model_id, api_key, **row):
            self.calls.append(("resolve", model_id, api_key))
            return super().resolve(model_id, api_key, **row)

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


def _florence_bridge():
    b = FakeSyncBridge()
    b.routes["florence-2-base"] = Route(
        model_id="florence-2-base",
        registry_id="florence-2-base",
        task_type="vlm",
        action="prompt",
        actions={"prompt"},
        model_class_name="Florence2HF",
    )
    b.predictions[("florence-2-base", "prompt")] = ["done"]
    return b


def test_lmm_on_a_florence2_route_derives_the_task_from_the_prompt():
    b = _florence_bridge()
    GatewayModelsProvider(b, api_key=None).run_lmm(
        "florence-2-base",
        {"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)},
        "<OD>",
    )
    assert b.calls[0][2]["task"] == "<OD>"


def test_lmm_on_a_florence2_route_rejects_a_missing_prompt():
    b = _florence_bridge()
    with pytest.raises(LegacyHTTPError):
        GatewayModelsProvider(b, api_key=None).run_lmm(
            "florence-2-base",
            {"type": "numpy_object", "value": np.zeros((2, 2, 3), np.uint8)},
            "",
        )


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


def test_sam3_3d_objects_is_501():
    from inference_server.legacy.errors import LegacyHTTPError

    provider = GatewayModelsProvider(_od_bridge(), api_key=None)
    with pytest.raises(LegacyHTTPError):
        provider.run_sam3_3d_objects("sam3/sam3_final", None, None)


SAMPLING = {
    "window_seconds": 8.0,
    "sample_fps": 2.0,
    "min_frames": 4,
    "max_frame_side": 720,
    "mode": "sliding_window",
    "max_frames": 16,
}


def _action_bridge(video_sampling=SAMPLING, class_names=("wave", "jump")):
    bridge = FakeSyncBridge()
    bridge.routes["clips/1"] = Route(
        model_id="clips/1",
        registry_id="clips/1",
        task_type="action-recognition",
        action="infer",
        actions={"infer"},
        class_names=list(class_names) if class_names is not None else None,
        video_sampling=video_sampling,
    )
    bridge.predictions[("clips/1", "infer")] = ["segment"]

    return bridge


def test_action_recognition_model_classifies_one_window_as_params_only():
    bridge = _action_bridge()
    provider = GatewayModelsProvider(bridge, api_key="provider-key")
    frames = [np.zeros((4, 6, 3), np.uint8) for _ in range(3)]

    model = provider.load_action_recognition_model("clips/1")
    segments = model.infer(frames=frames, class_names=["wave"], fps=2.0)

    assert segments == ["segment"]
    assert bridge.calls == [
        (
            "clips/1",
            "infer",
            {"frames": frames, "class_names": ["wave"], "fps": 2.0},
            None,
        )
    ]
    assert bridge.calls[0][2]["frames"] is frames
    assert bridge.resolved_rows == [("clips/1", None, "", None)]


def test_action_recognition_model_exposes_class_names_and_video_sampling():
    from inference_models.models.base.action_recognition import VideoSampling

    model = GatewayModelsProvider(
        _action_bridge(), api_key=None
    ).load_action_recognition_model("clips/1", api_key="k")

    assert model.class_names == ["wave", "jump"]
    assert model.video_sampling == VideoSampling(**SAMPLING)


def test_action_recognition_model_defaults_sampling_and_class_names():
    from inference_models.models.base.action_recognition import VideoSampling

    bridge = _action_bridge(video_sampling=None, class_names=None)
    model = GatewayModelsProvider(bridge, api_key=None).load_action_recognition_model(
        "clips/1"
    )
    model.infer(frames=[np.zeros((2, 2, 3), np.uint8)], fps=4.0)

    assert model.class_names is None
    assert model.video_sampling == VideoSampling()
    assert bridge.calls[0][2]["class_names"] is None


def test_action_recognition_model_raises_bridge_errors():
    bridge = _action_bridge()

    def _boom(route, api_key, action, params, record=True):
        raise ModelInputError("fps is required")

    bridge.infer_params_only = _boom
    model = GatewayModelsProvider(bridge, api_key=None).load_action_recognition_model(
        "clips/1"
    )

    with pytest.raises(ModelInputError):
        model.infer(frames=[np.zeros((2, 2, 3), np.uint8)], fps=1.0)


def test_action_recognition_model_of_another_task_is_refused():
    from inference_server.legacy.errors import LegacyHTTPError

    with pytest.raises(LegacyHTTPError) as error:
        GatewayModelsProvider(_od_bridge(), api_key=None).load_action_recognition_model(
            "ds/1"
        )

    assert error.value.status_code == 400
    assert error.value.message == "Model 'ds/1' is a object-detection model."


def test_provider_covers_protocol():
    for name, member in inspect.getmembers(ModelsProvider, inspect.isfunction):
        if name.startswith("_") and name != "__contains__":
            continue
        impl = getattr(GatewayModelsProvider, name, None)
        assert impl is not None, name
        assert list(inspect.signature(impl).parameters) == list(
            inspect.signature(member).parameters
        ), name


_EMBEDDING_INFO = {
    "feature_definition": "classifier-linear-input@v1",
    "output_type": "feature_vector",
    "normalization": "none",
    "dimension": 2,
    "preprocessing": {
        "image_pre_processing": {"auto-orient": {"enabled": True}},
        "network_input": {"training_input_size": {"width": 224}},
    },
    "backend": "ResNetForClassificationOnnx",
    "precision": "torch.float32",
}
_NO_OVERRIDES = {
    "disable_preproc_auto_orient": False,
    "disable_preproc_contrast": False,
    "disable_preproc_grayscale": False,
    "disable_preproc_static_crop": False,
}


def _embedding_instance(output_type):
    instance = "capabilities=image_embeddings"
    if output_type == "logits":
        instance += ";output_type=logits"
    return instance


class _PerImageBridge(FakeSyncBridge):
    def infer(self, route, api_key, action, images, params, record=True):
        self.calls.append((route.model_id, action, params, [i.data for i in images]))
        self.records.append(record)
        return [self.predictions[(route.model_id, action)]() for _ in images]


def _embedding_bridge(output_type, vectors):
    bridge = _PerImageBridge()
    key = routing_key("cls/1", _embedding_instance(output_type))
    bridge.routes[key] = Route(
        model_id="cls/1",
        registry_id=key,
        task_type="classification",
        action="infer",
        actions={"infer", "embed_images"},
        class_names=["cat", "dog"],
    )
    remaining = iter(vectors)
    bridge.predictions[("cls/1", "embed_images")] = lambda: {
        "embeddings": np.array([next(remaining)], dtype=np.float32),
        "embedding_info": {**_EMBEDDING_INFO, "output_type": output_type},
    }
    return bridge


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_run_image_embeddings_returns_the_legacy_payload(output_type):
    bridge = _embedding_bridge(output_type, [[2.0, 3.0], [4.0, 5.0]])
    provider = GatewayModelsProvider(bridge, api_key="req-key")
    image = np.zeros((4, 6, 3), dtype=np.uint8)

    result = provider.run_image_embeddings(
        "cls/1",
        [{"type": "numpy_object", "value": image}] * 2,
        api_key="key",
        output_type=output_type,
    )

    assert result["embeddings"] == [[2.0, 3.0], [4.0, 5.0]]
    assert result["embedding_info"]["model_id"] == "cls/1"
    assert result["embedding_info"]["output_type"] == output_type
    assert result["embedding_info"]["preprocessing"] == {
        **_EMBEDDING_INFO["preprocessing"],
        "overrides": _NO_OVERRIDES,
    }
    assert len(result["embedding_info"]["space_id"]) == 64
    assert set(result) == {"inference_id", "time", "embeddings", "embedding_info"}
    assert bridge.resolved_rows == [("cls/1", None, "", None)]
    assert bridge.calls[0][1] == "embed_images"
    assert bridge.calls[0][2] == {"output_type": output_type}
    assert bridge.calls[0][3][0] is image


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_run_tensor_image_embeddings_returns_a_batched_tensor(output_type):
    bridge = _embedding_bridge(output_type, [[2.0, -3.0], [4.0, 5.0]])
    provider = GatewayModelsProvider(bridge, api_key="req-key")
    images = [torch.zeros((3, 4, 6), dtype=torch.uint8)] * 2

    result = provider.run_tensor_image_embeddings(
        "cls/1",
        images,
        input_color_format="rgb",
        api_key="key",
        output_type=output_type,
    )

    assert isinstance(result["embeddings"], torch.Tensor)
    assert result["embeddings"].tolist() == [[2.0, -3.0], [4.0, 5.0]]
    assert result["embedding_info"]["model_id"] == "cls/1"
    assert result["embedding_info"]["output_type"] == output_type
    assert result["embedding_info"]["preprocessing"]["overrides"] == _NO_OVERRIDES
    assert "space_id" in result["embedding_info"]
    assert bridge.calls[0][1] == "embed_images"
    assert bridge.calls[0][2] == {
        "output_type": output_type,
        "input_color_format": "rgb",
    }
    assert bridge.calls[0][3][0].shape == (4, 6, 3)


def test_run_tensor_image_embeddings_lands_on_the_workflow_tensor_device(monkeypatch):
    from inference_server.workflows import models_provider as provider_module

    monkeypatch.setattr(
        provider_module.environment, "WORKFLOWS_IMAGE_TENSOR_DEVICE", "meta"
    )
    bridge = _embedding_bridge("feature_vector", [[2.0, -3.0]])
    provider = GatewayModelsProvider(bridge, api_key="req-key")

    result = provider.run_tensor_image_embeddings(
        "cls/1",
        [torch.zeros((3, 4, 6), dtype=torch.uint8)],
        input_color_format="rgb",
        api_key="key",
    )

    assert result["embeddings"].device.type == "meta"
    assert result["embeddings"].shape == (1, 2)


def test_run_image_embeddings_time_includes_image_loading(monkeypatch):
    import time as time_module

    from inference_server.workflows import models_provider as provider_module

    original = provider_module.decode_inline_image

    def slow(image, *, ndarray_ok):
        time_module.sleep(0.05)
        return original(image, ndarray_ok=ndarray_ok)

    monkeypatch.setattr(provider_module, "decode_inline_image", slow)
    bridge = _embedding_bridge("feature_vector", [[2.0, 3.0]])
    provider = GatewayModelsProvider(bridge, api_key="req-key")

    result = provider.run_image_embeddings(
        "cls/1",
        [{"type": "numpy_object", "value": np.zeros((4, 6, 3), dtype=np.uint8)}],
        api_key="key",
    )

    assert result["time"] >= 0.05


def test_add_model_refuses_a_model_id_carrying_a_capability_marker(fake_stat):
    spoofed = "ds/1:capabilities=image_embeddings;output_type=logits"
    fake_stat[spoofed] = ("classification", "infer")
    gateway = legacy_conftest.FakeGateway()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        sync = SyncLegacyBridge(LegacyModelBridge(gateway), LoopBridge(loop))
        provider = GatewayModelsProvider(sync, api_key="k")

        with pytest.raises(LegacyHTTPError) as error:
            provider.add_model(spoofed, "k")
        with pytest.raises(LegacyHTTPError):
            provider.run_image_embeddings(
                spoofed,
                [{"type": "numpy_object", "value": np.zeros((4, 6, 3), np.uint8)}],
                api_key="k",
            )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert error.value.status_code == 404
    assert gateway.calls == []
    assert spoofed not in provider


def test_add_model_with_capabilities_registers_the_instance_under_its_key():
    bridge = _embedding_bridge("logits", [])
    provider = GatewayModelsProvider(bridge, "req-key", "/workflows/run")

    provider.add_model(
        "cls/1", "k", required_capabilities=["image_embeddings"], output_type="logits"
    )

    assert bridge.recorded_requests == [
        (
            "cls/1:capabilities=image_embeddings;output_type=logits",
            "cls/1:capabilities=image_embeddings:output_type=logits",
            "/workflows/run",
            None,
        )
    ]
    assert "cls/1:capabilities=image_embeddings:output_type=logits" in provider
    assert "cls/1" not in provider
    assert provider.get_model_pipeline_depth("cls/1") == 1


fake_stat = legacy_conftest.fake_stat


def test_provider_inference_raises_model_input_error_for_a_gateway_value_error(
    fake_stat,
):
    def _reject(image, params):
        raise ValueError("bad shape")

    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = legacy_conftest.FakeGateway(
        predictions={("ds/1", "infer"): _reject},
        model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
    )
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        sync = SyncLegacyBridge(LegacyModelBridge(gateway), LoopBridge(loop))
        provider = GatewayModelsProvider(sync, api_key="k")
        provider.add_model("ds/1", "k")
        image = np.zeros((4, 6, 3), dtype=np.uint8)

        with pytest.raises(ModelInputError) as exc:
            provider.run_object_detection(
                "ds/1",
                [{"type": "numpy_object", "value": image}],
                api_key="k",
                confidence=0.5,
            )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()

    assert not isinstance(exc.value, LegacyHTTPError)
    assert str(exc.value) == "bad shape"


class _FailingManager:
    executor = None

    def __init__(self, error):
        self.error = error

    def __contains__(self, key):
        return False

    def load(self, key, api_key, **kwargs):
        raise self.error


def _provider_over(gateway, loop):
    sync = SyncLegacyBridge(LegacyModelBridge(gateway), LoopBridge(loop))
    provider = GatewayModelsProvider(sync, api_key="k")
    return provider


@pytest.fixture
def server_loop():
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


def test_provider_load_failure_raises_the_class_the_load_failed_with(
    fake_stat, server_loop
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = ModelManagerGateway(_FailingManager(ModelNotFoundError("missing")))
    provider = _provider_over(gateway, server_loop)

    with pytest.raises(ModelNotFoundError) as exc:
        provider.add_model("ds/1", "k")

    assert str(exc.value) == "missing"


def test_provider_load_failure_without_a_description_is_a_broken_package(
    fake_stat, server_loop
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gateway = legacy_conftest.FakeGateway()
    gateway.ensure_results = [("error", 5)]
    provider = _provider_over(gateway, server_loop)

    with pytest.raises(LegacyHTTPError) as exc:
        provider.add_model("ds/1", "k")

    assert exc.value.status_code == 500
    assert exc.value.message == "Model package is broken."


def _png_bytes():
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (6, 4)).save(buffer, format="PNG")

    return buffer.getvalue()


@pytest.mark.parametrize("declared_type", ["file", "FILE"])
def test_run_object_detection_loads_local_file_image(
    tmp_path, monkeypatch, declared_type
):
    monkeypatch.setattr(
        "inference_server.legacy.common.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM",
        True,
    )
    bridge = _od_bridge()
    provider = GatewayModelsProvider(bridge, api_key="req-key")
    provider.add_model("ds/1", "k")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(_png_bytes())

    out = provider.run_object_detection(
        "ds/1",
        [{"type": declared_type, "value": str(image_path)}],
        api_key="k",
        confidence=0.5,
    )

    assert out[0]["image"] == {"width": 6, "height": 4}
    assert bridge.calls[0][3] == [_png_bytes()]


def test_run_object_detection_fetches_url_image_declared_in_upper_case():
    image = {"type": "URL", "value": "https://example.com/a.png"}
    bridge = _od_bridge()
    fetched = []

    def _fetch_image(url):
        fetched.append(url)
        return _png_bytes()

    bridge.fetch_image = _fetch_image
    provider = GatewayModelsProvider(bridge, api_key="req-key")
    provider.add_model("ds/1", "k")

    out = provider.run_object_detection("ds/1", [image], api_key="k", confidence=0.5)

    assert fetched == ["https://example.com/a.png"]
    assert out[0]["image"] == {"width": 6, "height": 4}
    assert bridge.calls[0][3] == [_png_bytes()]


class _StreamSyncBridge(FakeSyncBridge):
    def __init__(self):
        super().__init__()
        self.flushes = {}
        self.stream_calls = []

    def flush_model_stream_pipeline(self, model_id):
        self.stream_calls.append(("flush", model_id))
        return self.flushes.get(model_id)

    def shutdown_model_stream_pipeline(self, model_id):
        self.stream_calls.append(("shutdown", model_id))


def _segmentation_route(depth):
    return Route(
        model_id="seg/1",
        registry_id="seg/1",
        task_type="instance-segmentation",
        action="infer",
        actions={"infer"},
        class_names=["cat"],
        stream_pipeline_depth=depth,
    )


def test_stream_pipeline_methods_forward_to_gateway():
    bridge = _StreamSyncBridge()
    bridge.routes["seg/1"] = _segmentation_route(depth=2)
    bridge.flushes["seg/1"] = []
    provider = GatewayModelsProvider(bridge, api_key=None)
    provider.add_model("seg/1", "k")

    assert provider.model_supports_stream_pipeline("seg/1") is True
    assert provider.get_model_pipeline_depth("seg/1") == 2
    assert provider.flush_model_stream_pipeline("seg/1") == []
    assert provider.shutdown_model_stream_pipeline("seg/1") is None
    assert bridge.stream_calls == [("flush", "seg/1"), ("shutdown", "seg/1")]


def test_stream_pipeline_flush_is_not_forwarded_for_a_depth_one_model():
    bridge = _StreamSyncBridge()
    bridge.routes["seg/1"] = _segmentation_route(depth=1)
    provider = GatewayModelsProvider(bridge, api_key=None)
    provider.add_model("seg/1", "k")

    assert provider.flush_model_stream_pipeline("seg/1") is None
    assert provider.shutdown_model_stream_pipeline("seg/1") is None
    assert bridge.stream_calls == []


class _FrameFuture:
    def __init__(self, frame, model):
        self.frame = frame
        self.model = model
        self.preprocess_metadata = [{"frame": frame}]

    def submit_gpu_work(self, meta=None):
        return None

    def done(self):
        return True

    def result(self):
        return [self.model.detections(self.frame)]


def _segmentation_model_class():
    import torch
    from pycocotools import mask as mask_utils

    from inference_models.models.base.instance_segmentation import (
        InstanceDetections,
        InstanceSegmentationModel,
    )
    from inference_models.models.base.types import InstancesRLEMasks

    class _Model(InstanceSegmentationModel):
        def __init__(self):
            self.async_calls = []
            self.sync_calls = []

        @classmethod
        def from_pretrained(cls, model_name_or_path, **kwargs):
            return cls()

        @property
        def class_names(self):
            return ["cat", "dog"]

        @property
        def supported_mask_formats(self):
            return {"dense", "rle"}

        @property
        def supports_stream_pipeline(self):
            return True

        def detections(self, frame):
            counts = []
            for row in (frame, frame + 5):
                mask = np.zeros((20, 10), dtype=np.uint8, order="F")
                mask[row : row + 3, frame : frame + 3] = 1
                counts.append(mask_utils.encode(np.asfortranarray(mask))["counts"])
            return InstanceDetections(
                xyxy=torch.tensor(
                    [
                        [frame, frame, frame + 3, frame + 3],
                        [frame, frame + 5, frame + 3, frame + 8],
                    ],
                    dtype=torch.int32,
                ),
                class_id=torch.tensor([0, 1], dtype=torch.int32),
                confidence=torch.tensor([0.9, 0.8], dtype=torch.float32),
                mask=InstancesRLEMasks(image_size=(20, 10), masks=counts),
            )

        def infer(self, images, **kwargs):
            self.sync_calls.append(kwargs)
            return [self.detections(0)]

        def infer_async(self, images, **kwargs):
            self.async_calls.append(kwargs)
            return _FrameFuture(len(self.async_calls) - 1, self)

        def pre_process(self, images, **kwargs):
            raise AssertionError("not used")

        def forward(self, pre_processed_images, **kwargs):
            raise AssertionError("not used")

        def post_process(self, model_results, pre_processing_meta, **kwargs):
            raise AssertionError("not used")

    return _Model


class _PipelineBridge(_StreamSyncBridge):
    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline
        self.returned_context_ids = []
        self.unloaded = False

    def infer(self, route, api_key, action, images, params, record=True):
        from inference_models.models.base.async_handoff import (
            get_async_response_context_id,
        )

        from inference_model_manager.model_manager import ModelManager

        self.calls.append(
            (route.model_id, action, dict(params), [i.data for i in images])
        )
        predictions = [
            ModelManager._wire_marshal_result(
                self.pipeline.infer(images=image.data, **params), 1
            )
            for image in images
        ]
        self.returned_context_ids.extend(
            get_async_response_context_id(prediction) for prediction in predictions
        )
        return predictions

    def flush_model_stream_pipeline(self, model_id):
        self.stream_calls.append(("flush", model_id))
        if self.unloaded:
            return None
        return self.pipeline.flush()

    def shutdown_model_stream_pipeline(self, model_id):
        self.stream_calls.append(("shutdown", model_id))
        self.pipeline.shutdown_pipeline()


class _FailingBridge(_StreamSyncBridge):
    def infer(self, route, api_key, action, images, params, record=True):
        self.calls.append((route.model_id, action, dict(params), None))
        raise RuntimeError("submit failed")


_FRAME_SHAPES = [(20, 10), (30, 16), (40, 24)]
_FRAME_CLASS_FILTERS = [None, ["cat"], ["dog"]]


def _frame(index):
    from datetime import datetime

    from roboflow_workflows.execution_engine.entities.base import (
        Batch,
        ImageParentMetadata,
        VideoMetadata,
        WorkflowImageData,
    )

    height, width = _FRAME_SHAPES[index]

    return Batch(
        content=[
            WorkflowImageData(
                parent_metadata=ImageParentMetadata(parent_id=f"p{index}"),
                numpy_image=np.zeros((height, width, 3), dtype=np.uint8),
                video_metadata=VideoMetadata(
                    video_identifier="cam",
                    frame_number=index,
                    frame_timestamp=datetime.now(),
                ),
            )
        ],
        indices=[(0,)],
    )


def _run_block(block, images, class_filter):
    return block.run_locally(
        images=images,
        model_id="seg/1",
        class_agnostic_nms=False,
        class_filter=class_filter,
        confidence=0.4,
        iou_threshold=0.3,
        max_detections=300,
        max_candidates=3000,
        mask_decode_mode="accurate",
        tradeoff_factor=0.0,
        disable_active_learning=False,
        active_learning_target_dataset=None,
        enforce_dense_masks_in_inference_models=False,
    )


def _pipelined_provider(monkeypatch, bridge=None):
    from inference_model_manager.stream_pipeline import stream_pipeline_for

    monkeypatch.setenv("RFDETR_PIPELINE_DEPTH", "2")
    model = _segmentation_model_class()()
    pipeline = stream_pipeline_for(model)
    if bridge is None:
        bridge = _PipelineBridge(pipeline)
    bridge.routes["seg/1"] = _segmentation_route(depth=2)
    bridge.routes["seg/1"].class_names = ["cat", "dog"]
    provider = GatewayModelsProvider(bridge, api_key="k")
    provider.add_model("seg/1", "k")

    return provider, bridge, model, pipeline


def _numpy_image(index=0):
    height, width = _FRAME_SHAPES[index]
    return {"type": "numpy_object", "value": np.zeros((height, width, 3), np.uint8)}


def _recorded_frames(provider, registry_id="seg/1"):
    recorded = provider._stream_frames.get(registry_id)
    if recorded is None:
        return {}
    return recorded.frames


def _segmentation_block(provider):
    from roboflow_workflows.core_steps.common.entities import StepExecutionMode
    from roboflow_workflows.core_steps.models.roboflow.instance_segmentation.v3 import (
        RoboflowInstanceSegmentationModelBlockV3,
    )

    return RoboflowInstanceSegmentationModelBlockV3(
        model_manager=provider, api_key="k", step_execution_mode=StepExecutionMode.LOCAL
    )


def _execute_workflow(block):
    try:
        outputs = [
            _run_block(block, _frame(index), _FRAME_CLASS_FILTERS[index])
            for index in range(3)
        ]
        flushed = block.flush_stream_pipeline_outputs()
    finally:
        block.close_stream_pipeline()

    return outputs, flushed


def test_instance_segmentation_block_pairs_frames_through_the_pipeline(monkeypatch):
    from concurrent.futures import Future

    import supervision as sv

    provider, bridge, model, _ = _pipelined_provider(monkeypatch)
    block = _segmentation_block(provider)

    outputs, flushed = _execute_workflow(block)

    context_ids = [call[2]["stream_pipeline_context_id"] for call in bridge.calls]
    assert len(set(context_ids)) == 3 and all(isinstance(c, str) for c in context_ids)
    assert bridge.returned_context_ids == [
        context_ids[0],
        context_ids[0],
        context_ids[1],
    ]
    assert len(model.async_calls) == 3
    assert model.sync_calls == []
    first = outputs[0][0]["predictions"]
    assert isinstance(first, sv.Detections) and len(first) == 0
    delayed = []
    for output in outputs[1:]:
        assert isinstance(output[0]["predictions"], Future)
        delayed.append(output[0]["predictions"].result(timeout=5))
    assert [d["parent_id"].tolist() for d in delayed] == [["p0", "p0"], ["p1"]]
    assert [d["class_name"].tolist() for d in delayed] == [["cat", "dog"], ["cat"]]
    assert [int(d.xyxy[0][0]) for d in delayed] == [0, 1]
    assert [d["image_dimensions"].tolist() for d in delayed] == [
        [[20, 10], [20, 10]],
        [[30, 16]],
    ]
    assert [d.mask.shape for d in delayed] == [(2, 20, 10), (1, 30, 16)]
    assert len(flushed) == 1
    indices, flushed_outputs = flushed[0]
    assert indices == [(0,)]
    tail = flushed_outputs[0]["predictions"]
    assert tail["parent_id"].tolist() == ["p2"] and int(tail.xyxy[0][0]) == 2
    assert tail["class_name"].tolist() == ["dog"]
    assert tail["image_dimensions"].tolist() == [[40, 24]]
    assert tail.mask.shape == (1, 40, 24)
    assert bridge.stream_calls == [("flush", "seg/1"), ("shutdown", "seg/1")]
    assert len(block._pending_stream_prediction_contexts) == 0
    assert _recorded_frames(provider) == {}


def test_two_workflow_executions_share_one_pipelined_model(monkeypatch):
    from concurrent.futures import Future

    provider, bridge, model, pipeline = _pipelined_provider(monkeypatch)
    block = _segmentation_block(provider)

    _execute_workflow(block)
    outputs, flushed = _execute_workflow(block)

    assert len(model.async_calls) == 6
    assert model.sync_calls == []
    assert len(outputs[0][0]["predictions"]) == 0
    delayed = [output[0]["predictions"].result(timeout=5) for output in outputs[1:]]
    assert all(isinstance(output[0]["predictions"], Future) for output in outputs[1:])
    assert [d["parent_id"].tolist() for d in delayed] == [["p0", "p0"], ["p1"]]
    assert [int(d.xyxy[0][0]) for d in delayed] == [3, 4]
    tail = flushed[0][1][0]["predictions"]
    assert tail["parent_id"].tolist() == ["p2"] and int(tail.xyxy[0][0]) == 5
    assert bridge.stream_calls == [
        ("flush", "seg/1"),
        ("shutdown", "seg/1"),
        ("flush", "seg/1"),
        ("shutdown", "seg/1"),
    ]
    assert len(block._pending_stream_prediction_contexts) == 0
    assert _recorded_frames(provider) == {}
    assert pipeline._response_executor is None


def _submit_frame(provider, index, image_index=0, raw=False):
    result = provider.run_instance_segmentation(
        "seg/1",
        [_numpy_image(image_index)],
        api_key="k",
        confidence=0.4,
        stream_pipeline_context_id=f"c{index}",
        return_raw_responses=raw,
    )
    if raw:
        return result.raw_responses
    return result


def _second_provider(bridge):
    provider = GatewayModelsProvider(bridge, api_key="k")
    provider.add_model("seg/1", "k")

    return provider


def _resolved(predictions):
    from concurrent.futures import Future

    if isinstance(predictions, Future):
        return predictions.result(timeout=5)
    return predictions


def test_a_second_producer_gets_its_synchronous_result_as_a_done_future(
    monkeypatch,
):
    from concurrent.futures import Future

    from inference_models.models.base.async_handoff import (
        get_async_response_context_id,
        get_async_response_future,
    )

    provider_a, bridge, model, _ = _pipelined_provider(monkeypatch)
    provider_b = _second_provider(bridge)
    _submit_frame(provider_a, 0)

    responses = _submit_frame(provider_b, 1, image_index=1, raw=True)

    assert len(model.sync_calls) == 1 and len(model.async_calls) == 1
    future = get_async_response_future(responses[0])
    assert isinstance(future, Future) and future.done()
    assert future.result(timeout=0) == responses
    assert get_async_response_context_id(responses[0]) == "c1"
    assert responses[0].image.width == 16
    assert [p.class_name for p in responses[0].predictions] == ["cat", "dog"]
    assert _recorded_frames(provider_b) == {}
    assert list(_recorded_frames(provider_a)) == ["c0"]


def test_two_blocks_with_their_own_providers_share_one_wrapper(monkeypatch):
    provider_a, bridge, model, _ = _pipelined_provider(monkeypatch)
    provider_b = _second_provider(bridge)
    block_a = _segmentation_block(provider_a)
    block_b = _segmentation_block(provider_b)
    outputs_a, outputs_b = [], []

    try:
        for index in range(3):
            frame, class_filter = _frame(index), _FRAME_CLASS_FILTERS[index]
            outputs_a.append(_run_block(block_a, frame, class_filter))
            outputs_b.append(_run_block(block_b, frame, class_filter))
        flushed_a = block_a.flush_stream_pipeline_outputs()
        flushed_b = block_b.flush_stream_pipeline_outputs()
    finally:
        block_a.close_stream_pipeline()
        block_b.close_stream_pipeline()

    assert len(model.async_calls) == 3 and len(model.sync_calls) == 3
    resolved_a = [_resolved(output[0]["predictions"]) for output in outputs_a]
    resolved_b = [_resolved(output[0]["predictions"]) for output in outputs_b]
    assert [d["parent_id"].tolist() for d in resolved_a] == [[], ["p0", "p0"], ["p1"]]
    assert [int(d.xyxy[0][0]) for d in resolved_a[1:]] == [0, 1]
    assert [d["parent_id"].tolist() for d in resolved_b] == [
        ["p0", "p0"],
        ["p1"],
        ["p2"],
    ]
    assert [int(d.xyxy[0][0]) for d in resolved_b] == [0, 0, 0]
    assert [d["class_name"].tolist() for d in resolved_b] == [
        ["cat", "dog"],
        ["cat"],
        ["dog"],
    ]
    tail = flushed_a[0][1][0]["predictions"]
    assert tail["parent_id"].tolist() == ["p2"] and int(tail.xyxy[0][0]) == 2
    assert flushed_b == []
    assert len(block_a._pending_stream_prediction_contexts) == 0
    assert len(block_b._pending_stream_prediction_contexts) == 0
    assert _recorded_frames(provider_a) == {} and _recorded_frames(provider_b) == {}


def test_pipelined_path_sends_the_provider_as_the_producer(monkeypatch):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)

    _submit_frame(provider, 0)

    assert bridge.calls[0][2]["stream_pipeline_producer_id"] == str(id(provider))


def test_non_pipelined_path_sends_no_producer_id():
    bridge = FakeSyncBridge()
    bridge.routes["seg/1"] = _segmentation_route(depth=1)
    bridge.predictions[("seg/1", "infer")] = SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 3]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        mask=np.zeros((1, 20, 10), dtype=bool),
    )
    provider = GatewayModelsProvider(bridge, api_key="k")
    provider.add_model("seg/1", "k")

    _submit_frame(provider, 0)

    assert "stream_pipeline_producer_id" not in bridge.calls[0][2]
    assert "stream_pipeline_context_id" not in bridge.calls[0][2]


def test_failed_pipelined_submits_leave_no_stream_frames(monkeypatch):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch, bridge=_FailingBridge())

    for index in range(3):
        with pytest.raises(RuntimeError, match="submit failed"):
            _submit_frame(provider, index)

    assert len(bridge.calls) == 3
    assert _recorded_frames(provider) == {}


def test_pipelined_frames_map_is_bounded_by_the_depth(monkeypatch):
    provider, _, _, _ = _pipelined_provider(monkeypatch)

    for index in range(5):
        _submit_frame(provider, index)

    assert list(_recorded_frames(provider)) == ["c4"]


def test_flush_drops_unmatched_stream_frames(monkeypatch, caplog):
    provider, _, _, _ = _pipelined_provider(monkeypatch)
    for index in range(2):
        _submit_frame(provider, index, image_index=index)
    _recorded_frames(provider)["stale"] = _recorded_frames(provider)["c1"]

    with caplog.at_level("WARNING"):
        flushed = provider.flush_model_stream_pipeline("seg/1")

    assert [response.image.width for response in flushed] == [16]
    assert _recorded_frames(provider) == {}
    assert any("stale" in record.getMessage() for record in caplog.records)
    assert provider.flush_model_stream_pipeline("seg/1") == []


def test_shutdown_clears_the_frames_of_an_aborted_pipeline(monkeypatch):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)
    for index in range(2):
        _submit_frame(provider, index)
    assert _recorded_frames(provider) != {}

    provider.shutdown_model_stream_pipeline("seg/1")

    assert _recorded_frames(provider) == {}
    assert bridge.stream_calls == [("shutdown", "seg/1")]


def _reload_route(bridge, loaded_monotonic):
    bridge.routes["seg/1"] = _segmentation_route(depth=2)
    bridge.routes["seg/1"].class_names = ["cat", "dog"]
    bridge.routes["seg/1"].loaded_monotonic = loaded_monotonic


def test_flush_returning_none_clears_the_frames_and_a_reload_starts_afresh(
    monkeypatch,
):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)
    for index in range(2):
        _submit_frame(provider, index)
    assert _recorded_frames(provider) != {}
    bridge.unloaded = True

    assert provider.flush_model_stream_pipeline("seg/1") is None

    assert _recorded_frames(provider) == {}
    bridge.unloaded = False
    bridge.pipeline.shutdown_pipeline()
    _reload_route(bridge, loaded_monotonic=2.0)
    _submit_frame(provider, 2)
    assert list(_recorded_frames(provider)) == ["c2"]


def test_flush_raising_clears_the_frames(monkeypatch):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)
    _submit_frame(provider, 0)

    def failing_flush(model_id):
        raise RuntimeError("flush failed")

    bridge.flush_model_stream_pipeline = failing_flush
    with pytest.raises(RuntimeError, match="flush failed"):
        provider.flush_model_stream_pipeline("seg/1")

    assert _recorded_frames(provider) == {}


def test_frames_recorded_under_a_previous_route_generation_are_discarded(
    monkeypatch,
):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)
    _submit_frame(provider, 0)
    bridge.routes["seg/1"].loaded_monotonic = 5.0

    _submit_frame(provider, 1)

    assert list(_recorded_frames(provider)) == ["c1"]


def test_a_bare_placeholder_drops_the_frames_of_a_dead_pipeline(monkeypatch, caplog):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)
    _submit_frame(provider, 0)
    bridge.pipeline.shutdown_pipeline()

    with caplog.at_level("DEBUG", logger="inference_server.workflows.models_provider"):
        _submit_frame(provider, 1)

    assert list(_recorded_frames(provider)) == ["c1"]
    assert any("c0" in record.getMessage() for record in caplog.records)


def test_submit_unload_reload_cycles_leave_a_bounded_frame_map(monkeypatch):
    provider, bridge, _, _ = _pipelined_provider(monkeypatch)

    for cycle in range(4):
        _submit_frame(provider, cycle)
        bridge.pipeline.shutdown_pipeline()
        _reload_route(bridge, loaded_monotonic=float(cycle))
        _submit_frame(provider, 10 + cycle)
        assert list(_recorded_frames(provider)) == [f"c{10 + cycle}"]

    assert len(provider._stream_frames) == 1
