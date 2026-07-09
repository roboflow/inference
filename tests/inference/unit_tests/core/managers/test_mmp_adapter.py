import asyncio
import base64
import io
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.entities.responses.inference import (
    ObjectDetectionInferenceResponse,
)
from inference.core.exceptions import (
    InferenceModelNotFound,
    InferencePayloadTooLargeError,
    InputImageLoadError,
    InvalidImageTypeDeclared,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
    ModelManagerLockAcquisitionError,
    RoboflowAPIConnectionError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPITimeoutError,
)
from inference.core.managers import mmp_translation as translation
from inference.core.managers.mmp_adapter import ModelManagerAdapter


class FakeDetections:
    def __init__(self, xyxy, confidence, class_id):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.confidence = np.asarray(confidence, dtype=float)
        self.class_id = np.asarray(class_id)


class FakeClient:
    load_wait_s = 1.0
    infer_timeout_s = 1.0
    n_slots = 4

    def __init__(self):
        self.started = False
        self.loaded = []
        self.unloaded = []
        self.infer_calls = []
        self.load_result = ("ok",)
        self.ensure_result = ("model_ready",)
        self.tasks = {"infer": {}}
        self.class_names = ["cat", "dog"]
        self.key_points_classes = None
        self.infer_result = [
            FakeDetections(
                xyxy=[[10.0, 20.0, 30.0, 60.0]], confidence=[0.9], class_id=[1]
            )
        ]
        self.infer_error = None

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.started = False

    async def load(self, model_id, api_key=""):
        self.loaded.append(model_id)
        return self.load_result

    async def unload(self, model_id):
        self.unloaded.append(model_id)
        return ("ok",)

    async def ensure_loaded(self, model_id, instance="", api_key="", device=""):
        return self.ensure_result

    async def interface(self, model_id):
        return {"model_id": model_id, "tasks": self.tasks}

    model_class_name = "YOLOv8ObjectDetectionOnnx"

    async def stats(self):
        return {
            "mmp_models": {
                m: {
                    "class_names": self.class_names,
                    "key_points_classes": self.key_points_classes,
                    "model_class_name": self.model_class_name,
                }
                for m in self.loaded
            }
        }

    async def infer(self, *, model_id, image, task=None, instance="", params=None, **kw):
        self.infer_calls.append(
            {"model_id": model_id, "image": image, "task": task, "params": params}
        )
        if self.infer_error is not None:
            raise self.infer_error
        return self.infer_result


class FakeLegacy:
    def __init__(self):
        self.pingback = None
        self.metadata_calls = []

    def init_pingback(self):
        self.num_errors = 0

    def record_request_metadata(self, **kwargs):
        self.metadata_calls.append(kwargs)


def make_adapter():
    return ModelManagerAdapter(legacy_stack=FakeLegacy(), mmp_client=FakeClient())


def _seed_route(adapter, model_id, **overrides):
    route = {
        "supported": True,
        "mmp_model_id": model_id,
        "task_type": "object-detection",
        "action": "infer",
        "class_names": ["cat", "dog"],
        "key_points_classes": None,
    }
    route.update(overrides)
    adapter._routes[model_id] = route
    return route


def make_stat(task_type, action="infer"):
    async def fake_stat(model_id, api_key):
        return (task_type, action)

    return fake_stat


@pytest.fixture
def running_adapter():
    adapter = make_adapter()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(adapter.start(), loop).result(timeout=5)
    yield adapter
    asyncio.run_coroutine_threadsafe(adapter.shutdown(), loop).result(timeout=5)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


@pytest.fixture
def od_stat(monkeypatch):
    calls = []

    async def fake_stat(model_id, api_key):
        calls.append((model_id, api_key))
        return ("object-detection", "infer")

    monkeypatch.setattr(translation, "stat_model", fake_stat)
    return calls


@pytest.fixture
def png_image(monkeypatch):
    monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
    payload = b"\x89PNG fake payload"
    return SimpleNamespace(
        type="base64", value=base64.b64encode(payload).decode()
    ), payload


def od_request(image=None, **overrides):
    fields = {
        "id": "req-1",
        "api_key": "key",
        "confidence": 0.4,
        "iou_threshold": 0.3,
        "max_detections": 300,
        "max_candidates": 3000,
        "class_agnostic_nms": False,
        "class_filter": None,
        "visualize_predictions": False,
        "disable_preproc_auto_orient": False,
        "disable_preproc_contrast": False,
        "disable_preproc_grayscale": False,
        "disable_preproc_static_crop": False,
        "image": image,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestRouting:
    def test_add_model_loads_and_caches_route(self, running_adapter, od_stat):
        running_adapter.add_model("ws/1", api_key="key")
        assert running_adapter._client.loaded == ["ws/1"]
        assert "ws/1" in running_adapter
        assert running_adapter.get_task_type("ws/1") == "object-detection"
        assert running_adapter.get_class_names("ws/1") == ["cat", "dog"]
        assert od_stat == [("ws/1", "key")]

    def test_add_model_alias_shares_route(self, running_adapter, od_stat):
        running_adapter.add_model("ws/1", api_key="key", model_id_alias="alias/1")
        assert "alias/1" in running_adapter
        assert running_adapter.get_task_type("alias/1") == "object-detection"

    def test_unimplemented_task_type_is_terminal_unsupported(
        self, running_adapter, monkeypatch
    ):
        calls = []

        async def fake_stat(model_id, api_key):
            calls.append(model_id)
            return ("embedding", "embed_images")

        monkeypatch.setattr(translation, "stat_model", fake_stat)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("clip/1", api_key="key")
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("clip/1", api_key="key")
        assert calls == ["clip/1"]
        assert running_adapter._client.loaded == []
        assert "clip/1" not in running_adapter

    def test_passthrough_is_unsupported_without_stat(
        self, running_adapter, od_stat
    ):
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("passthrough", api_key="key")
        assert od_stat == []

    def test_runtime_tier_mismatch_unloads_and_errors(
        self, running_adapter, od_stat
    ):
        running_adapter._client.tasks = {"embed": {}}
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("ws/1", api_key="key")
        assert running_adapter._client.unloaded == ["ws/1"]
        assert "ws/1" not in running_adapter

    def test_load_error_code_translates(self, running_adapter, od_stat):
        running_adapter._client.load_result = ("error", 5)
        with pytest.raises(ModelArtefactError):
            running_adapter.add_model("ws/1", api_key="key")

    def test_load_timeout_translates_to_retryable(self, running_adapter, od_stat):
        running_adapter._client.load_result = ("load_timeout", 5)
        with pytest.raises(InferenceModelNotFound):
            running_adapter.add_model("ws/1", api_key="key")

    def test_busy_code_translates_to_retryable(self, running_adapter, od_stat):
        running_adapter._client.load_result = ("error", 1)
        with pytest.raises(ModelManagerLockAcquisitionError):
            running_adapter.add_model("ws/1", api_key="key")


class TestStatErrorTranslation:
    @pytest.fixture
    def stub_inference_server(self, monkeypatch):
        import sys
        import types

        root = types.ModuleType("inference_server")
        framework = types.ModuleType("inference_server.framework")
        entities = types.ModuleType("inference_server.framework.entities")
        model_stat = types.ModuleType("inference_server.framework.model_stat")

        class CommonRequestParams:
            def __init__(self, model_id, api_key):
                self.model_id = model_id
                self.api_key = api_key

        entities.CommonRequestParams = CommonRequestParams
        for name, module in (
            ("inference_server", root),
            ("inference_server.framework", framework),
            ("inference_server.framework.entities", entities),
            ("inference_server.framework.model_stat", model_stat),
        ):
            monkeypatch.setitem(sys.modules, name, module)
        return model_stat

    @pytest.mark.parametrize(
        "raised,expected",
        [
            (PermissionError("no"), RoboflowAPINotAuthorizedError),
            (LookupError("missing"), RoboflowAPINotNotFoundError),
            (RuntimeError("down"), RoboflowAPIConnectionError),
        ],
    )
    def test_stat_exceptions_map_to_legacy(
        self, stub_inference_server, raised, expected
    ):
        async def failing_stat(common_params):
            raise raised

        stub_inference_server.stat_model_while_checking_auth = failing_stat
        with pytest.raises(expected):
            asyncio.run(translation.stat_model(model_id="ws/1", api_key="key"))

    def test_stat_result_passes_through(self, stub_inference_server):
        async def ok_stat(common_params):
            assert common_params.model_id == "ws/1"
            assert common_params.api_key == "key"
            return ("object-detection", "infer")

        stub_inference_server.stat_model_while_checking_auth = ok_stat
        result = asyncio.run(translation.stat_model(model_id="ws/1", api_key="key"))
        assert result == ("object-detection", "infer")


class TestInfer:
    def test_sync_infer_happy_path(self, running_adapter, od_stat, png_image):
        image, payload = png_image
        request = od_request(image=image)
        response = running_adapter.infer_from_request_sync("ws/1", request)
        assert isinstance(response, ObjectDetectionInferenceResponse)
        assert running_adapter._client.infer_calls[0]["image"] == payload
        assert running_adapter._client.infer_calls[0]["task"] == "infer"
        assert running_adapter._client.infer_calls[0]["params"] == {
            "confidence": 0.4,
            "iou_threshold": 0.3,
            "max_detections": 300,
            "class_agnostic_nms": False,
        }
        prediction = response.predictions[0]
        assert prediction.x == 20.0 and prediction.y == 40.0
        assert prediction.width == 20.0 and prediction.height == 40.0
        assert prediction.class_name == "dog" and prediction.class_id == 1
        assert response.image.width == 64 and response.image.height == 48
        assert response.inference_id == "req-1"
        assert response.time is not None

    def test_async_infer_happy_path(self, running_adapter, od_stat, png_image):
        image, _ = png_image
        request = od_request(image=image)
        response = asyncio.run_coroutine_threadsafe(
            running_adapter.infer_from_request("ws/1", request),
            running_adapter._loop,
        ).result(timeout=5)
        assert isinstance(response, ObjectDetectionInferenceResponse)

    def test_class_filter_applied_post_repack(
        self, running_adapter, od_stat, png_image
    ):
        image, _ = png_image
        request = od_request(image=image, class_filter=["cat"])
        response = running_adapter.infer_from_request_sync("ws/1", request)
        assert response.predictions == []

    def test_numpy_image_forwarded_as_npy(self, running_adapter, od_stat):
        array = np.zeros((48, 64, 3), dtype=np.uint8)
        request = od_request(image=SimpleNamespace(type="numpy", value=array))
        response = running_adapter.infer_from_request_sync("ws/1", request)
        sent = running_adapter._client.infer_calls[0]["image"]
        assert sent.startswith(b"\x93NUMPY")
        restored = np.load(io.BytesIO(sent))
        assert restored.shape == (48, 64, 3)
        assert response.image.width == 64 and response.image.height == 48

    def test_visualize_predictions_errors(self, running_adapter, od_stat, png_image):
        image, _ = png_image
        request = od_request(image=image, visualize_predictions=True)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("ws/1", request)

    def test_disable_preproc_errors(self, running_adapter, od_stat, png_image):
        image, _ = png_image
        request = od_request(image=image, disable_preproc_contrast=True)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("ws/1", request)

    def test_non_default_max_candidates_errors(
        self, running_adapter, od_stat, png_image
    ):
        image, _ = png_image
        request = od_request(image=image, max_candidates=50)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("ws/1", request)

    def test_batch_returns_ordered_response_list(
        self, running_adapter, od_stat, png_image
    ):
        image, payload = png_image
        request = od_request(image=[image, image, image])
        responses = running_adapter.infer_from_request_sync("ws/1", request)
        assert isinstance(responses, list) and len(responses) == 3
        assert all(
            isinstance(r, ObjectDetectionInferenceResponse) for r in responses
        )
        assert len(running_adapter._client.infer_calls) == 3
        assert all(
            call["image"] == payload for call in running_adapter._client.infer_calls
        )

    def test_batch_first_error_fails_whole_request(
        self, running_adapter, od_stat, png_image
    ):
        image, _ = png_image
        running_adapter._client.infer_error = asyncio.TimeoutError()
        request = od_request(image=[image, image])
        with pytest.raises(RoboflowAPITimeoutError):
            running_adapter.infer_from_request_sync("ws/1", request)

    def test_infer_timeout_maps_to_legacy_timeout(
        self, running_adapter, od_stat, png_image
    ):
        image, _ = png_image
        running_adapter._client.infer_error = asyncio.TimeoutError()
        with pytest.raises(RoboflowAPITimeoutError):
            running_adapter.infer_from_request_sync("ws/1", od_request(image=image))

    def test_payload_too_large_maps_to_413_error(
        self, running_adapter, od_stat, png_image
    ):
        class PayloadTooLargeError(ValueError):
            pass

        image, _ = png_image
        running_adapter._client.infer_error = PayloadTooLargeError("too big")
        with pytest.raises(InferencePayloadTooLargeError):
            running_adapter.infer_from_request_sync("ws/1", od_request(image=image))

    def test_server_busy_maps_to_retryable(self, running_adapter, od_stat, png_image):
        class ServerBusyError(RuntimeError):
            pass

        image, _ = png_image
        running_adapter._client.infer_error = ServerBusyError("busy")
        with pytest.raises(ModelManagerLockAcquisitionError):
            running_adapter.infer_from_request_sync("ws/1", od_request(image=image))

    def test_unparseable_header_errors(self, running_adapter, od_stat, monkeypatch):
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: None)
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"junk").decode())
        with pytest.raises(InputImageLoadError):
            running_adapter.infer_from_request_sync("ws/1", od_request(image=image))

    def test_unknown_image_type_errors(self, running_adapter, od_stat):
        image = SimpleNamespace(type="multipart", value=b"x")
        with pytest.raises(InvalidImageTypeDeclared):
            running_adapter.infer_from_request_sync("ws/1", od_request(image=image))


class FakeInstanceDetections:
    def __init__(self, xyxy, confidence, class_id, mask):
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.confidence = np.asarray(confidence, dtype=float)
        self.class_id = np.asarray(class_id)
        self.mask = mask


class FakeKeyPoints:
    def __init__(self, xy, class_id, confidence):
        self.xy = np.asarray(xy, dtype=float)
        self.class_id = np.asarray(class_id)
        self.confidence = np.asarray(confidence, dtype=float)


class FakeClassification:
    def __init__(self, confidence, class_ids=None):
        self.confidence = np.asarray(confidence, dtype=float)
        if class_ids is not None:
            self.class_ids = np.asarray(class_ids)


class FakeSemanticSegmentation:
    def __init__(self, segmentation_map, confidence):
        self.segmentation_map = np.asarray(segmentation_map)
        self.confidence = np.asarray(confidence, dtype=float)


class TestPerTypeRepack:
    def _run(self, running_adapter, monkeypatch, task_type, result, request=None):
        monkeypatch.setattr(translation, "stat_model", make_stat(task_type))
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.infer_result = result
        image = SimpleNamespace(
            type="base64", value=base64.b64encode(b"fake").decode()
        )
        return running_adapter.infer_from_request_sync(
            "ws/1", request or od_request(image=image)
        )

    def test_instance_segmentation_polygon(self, running_adapter, monkeypatch):
        mask = np.zeros((1, 48, 64), dtype=np.uint8)
        mask[0, 10:20, 10:20] = 1
        result = [
            FakeInstanceDetections(
                xyxy=[[10.0, 10.0, 20.0, 20.0]],
                confidence=[0.8],
                class_id=[0],
                mask=mask,
            )
        ]
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(
            image=image,
            mask_decode_mode="accurate",
            tradeoff_factor=0.0,
            response_mask_format="polygon",
        )
        response = self._run(
            running_adapter, monkeypatch, "instance-segmentation", result, request
        )
        prediction = response.predictions[0]
        assert prediction.class_name == "cat"
        assert prediction.mask_format == "polygon"
        assert len(prediction.points) > 0

    def test_instance_segmentation_rle(self, running_adapter, monkeypatch):
        pytest.importorskip("pycocotools")
        mask = np.zeros((1, 48, 64), dtype=np.uint8)
        mask[0, 10:20, 10:20] = 1
        result = [
            FakeInstanceDetections(
                xyxy=[[10.0, 10.0, 20.0, 20.0]],
                confidence=[0.8],
                class_id=[0],
                mask=mask,
            )
        ]
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, response_mask_format="rle")
        response = self._run(
            running_adapter, monkeypatch, "instance-segmentation", result, request
        )
        prediction = response.predictions[0]
        assert prediction.mask_format == "rle"
        assert isinstance(prediction.rle["counts"], str)

    def test_instance_segmentation_mask_decode_mode_errors(
        self, running_adapter, monkeypatch
    ):
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, mask_decode_mode="fast")
        with pytest.raises(ModelDeploymentNotSupportedError):
            self._run(
                running_adapter, monkeypatch, "instance-segmentation", [], request
            )

    def test_instance_segmentation_tradeoff_factor_errors(
        self, running_adapter, monkeypatch
    ):
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, tradeoff_factor=0.5)
        with pytest.raises(ModelDeploymentNotSupportedError):
            self._run(
                running_adapter, monkeypatch, "instance-segmentation", [], request
            )

    def test_keypoints(self, running_adapter, monkeypatch):
        running_adapter._client.key_points_classes = [["nose", "tail"]]
        result = (
            [FakeKeyPoints(xy=[[[5.0, 6.0], [7.0, 8.0]]], class_id=[0], confidence=[[0.9, 0.0]])],
            [FakeDetections(xyxy=[[0.0, 0.0, 10.0, 10.0]], confidence=[0.7], class_id=[1])],
        )
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, keypoint_confidence=0.0)
        response = self._run(
            running_adapter, monkeypatch, "keypoint-detection", result, request
        )
        prediction = response.predictions[0]
        assert prediction.class_name == "dog"
        assert len(prediction.keypoints) == 1
        keypoint = prediction.keypoints[0]
        assert keypoint.class_name == "nose" and keypoint.class_id == 0
        assert running_adapter._client.infer_calls[0]["params"][
            "key_points_threshold"
        ] == 0.0

    def test_keypoints_without_skeleton_names_unsupported(
        self, running_adapter, monkeypatch
    ):
        running_adapter._client.key_points_classes = None
        monkeypatch.setattr(translation, "stat_model", make_stat("keypoint-detection"))
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("ws/kp", api_key="key")
        assert running_adapter._client.unloaded == ["ws/kp"]

    def test_classification(self, running_adapter, monkeypatch):
        result = FakeClassification(confidence=[0.7, 0.2])
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, confidence=0.4)
        response = self._run(
            running_adapter, monkeypatch, "classification", result, request
        )
        assert response.top == "cat"
        assert response.confidence == 0.7
        assert [p.class_name for p in response.predictions] == ["cat"]
        assert running_adapter._client.infer_calls[0]["params"] == {"confidence": 0.4}

    def test_multi_label_classification(self, running_adapter, monkeypatch):
        result = [FakeClassification(confidence=[0.7, 0.9], class_ids=[1])]
        response = self._run(
            running_adapter, monkeypatch, "multi-label-classification", result
        )
        assert response.predicted_classes == ["dog"]
        assert response.predictions["cat"].confidence == 0.7
        assert response.predictions["dog"].class_id == 1

    def test_semantic_segmentation(self, running_adapter, monkeypatch):
        seg_map = np.zeros((48, 64), dtype=np.uint8)
        seg_map[:10] = 1
        result = [
            FakeSemanticSegmentation(
                segmentation_map=seg_map, confidence=np.ones((48, 64)) * 0.5
            )
        ]
        response = self._run(
            running_adapter, monkeypatch, "semantic-segmentation", result
        )
        assert response.predictions.class_map == {"0": "cat", "1": "dog"}
        from PIL import Image as PILImage

        decoded = np.asarray(
            PILImage.open(
                io.BytesIO(base64.b64decode(response.predictions.segmentation_mask))
            )
        )
        assert decoded.shape == (48, 64)
        assert set(np.unique(decoded)) == {0, 1}

    def test_depth_estimation(self, running_adapter, monkeypatch):
        pytest.importorskip("matplotlib")
        depth = np.linspace(0.0, 10.0, 48 * 64, dtype=np.float32).reshape(48, 64)
        response = self._run(
            running_adapter, monkeypatch, "depth-estimation", [depth]
        )
        normalized = response.response["normalized_depth"]
        assert normalized.min() == 0.0 and normalized.max() == 1.0
        assert base64.b64decode(response.response["image"].base64_image)[:2] == b"\xff\xd8"
        assert response.inference_id == "req-1"

    def test_depth_no_variation_errors(self, running_adapter, monkeypatch):
        depth = np.ones((48, 64), dtype=np.float32)
        with pytest.raises(ModelArtefactError):
            self._run(running_adapter, monkeypatch, "depth-estimation", [depth])


class TestPhase3aRouting:
    def test_structured_ocr_with_bounding_boxes(self, running_adapter, monkeypatch):
        running_adapter._client.class_names = ["text"]
        result = (
            ["hello world"],
            [FakeDetections(xyxy=[[1.0, 2.0, 3.0, 4.0]], confidence=[0.9], class_id=[0])],
        )
        monkeypatch.setattr(translation, "stat_model", make_stat("structured-ocr"))
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.infer_result = result
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, generate_bounding_boxes=True)
        response = running_adapter.infer_from_request_sync("doctr/default", request)
        assert response.result == "hello world"
        assert response.predictions[0].class_name == "text"
        assert response.image.width == 64
        assert response.time > 0

    def test_structured_ocr_without_boxes(self, running_adapter, monkeypatch):
        result = (["abc"], [None])
        monkeypatch.setattr(translation, "stat_model", make_stat("structured-ocr"))
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.infer_result = result
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        response = running_adapter.infer_from_request_sync(
            "doctr/default", od_request(image=image)
        )
        assert response.result == "abc"
        assert response.predictions is None

    def test_easy_ocr_non_default_language_errors(self, running_adapter, monkeypatch):
        monkeypatch.setattr(translation, "stat_model", make_stat("structured-ocr"))
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, language_codes=["fr"])
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("easy_ocr/english_g2", request)

    def test_text_only_ocr(self, running_adapter, monkeypatch):
        monkeypatch.setattr(translation, "stat_model", make_stat("text-only-ocr"))
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.infer_result = "printed text"
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        response = running_adapter.infer_from_request_sync(
            "trocr/trocr-base-printed", od_request(image=image)
        )
        assert response.result == "printed text"

    def test_open_vocabulary_uses_requested_classes(
        self, running_adapter, monkeypatch
    ):
        monkeypatch.setattr(
            translation, "stat_model", make_stat("open-vocabulary-object-detection")
        )
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.infer_result = [
            FakeDetections(
                xyxy=[[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 5.0, 5.0]],
                confidence=[0.9, 0.8],
                class_id=[1, 2],
            )
        ]
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, text=["person", "dog"])
        response = running_adapter.infer_from_request_sync("grounding_dino/default", request)
        assert running_adapter._client.infer_calls[0]["params"]["classes"] == [
            "person",
            "dog",
        ]
        names = [p.class_name for p in response.predictions]
        assert names == ["dog", "2"]

    def test_open_vocabulary_without_classes_errors(
        self, running_adapter, monkeypatch
    ):
        monkeypatch.setattr(
            translation, "stat_model", make_stat("open-vocabulary-object-detection")
        )
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync(
                "owlv2/base", od_request(image=image)
            )

    def test_few_shot_training_data_errors(self, running_adapter, monkeypatch):
        monkeypatch.setattr(
            translation, "stat_model", make_stat("open-vocabulary-object-detection")
        )
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, text=["a"], training_data=[{"boxes": []}])
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("owlv2/base", request)

    def test_box_threshold_non_default_errors(self, running_adapter, monkeypatch):
        monkeypatch.setattr(
            translation, "stat_model", make_stat("open-vocabulary-object-detection")
        )
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, text=["a"], box_threshold=0.7)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("grounding_dino/default", request)

    def test_embedding_task_unsupported(self, running_adapter, monkeypatch):
        monkeypatch.setattr(translation, "stat_model", make_stat("embedding", "embed_images"))
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("clip/ViT-B-16", api_key="key")
        assert running_adapter._client.loaded == []


class TestVlmRouting:
    def _setup(self, running_adapter, monkeypatch, tasks=("prompt",)):
        monkeypatch.setattr(translation, "stat_model", make_stat("vlm", "prompt"))
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.tasks = {t: {} for t in tasks}

    def test_prompt_happy_path(self, running_adapter, monkeypatch):
        self._setup(running_adapter, monkeypatch)
        running_adapter._client.model_class_name = "Qwen25VLHF"
        running_adapter._client.infer_result = "a red truck"
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(
            image=image, prompt="describe", max_new_tokens=128, enable_thinking=False
        )
        response = running_adapter.infer_from_request_sync("qwen/1", request)
        assert response.response == "a red truck"
        assert response.image.width == 64
        params = running_adapter._client.infer_calls[0]["params"]
        assert params == {"prompt": "describe", "max_new_tokens": 128}
        assert running_adapter._client.infer_calls[0]["task"] == "prompt"

    def test_enable_thinking_forwarded_and_dict_response(
        self, running_adapter, monkeypatch
    ):
        self._setup(running_adapter, monkeypatch)
        running_adapter._client.model_class_name = "Qwen35HF"
        running_adapter._client.infer_result = [
            {"reasoning": "...", "answer": "cat"}
        ]
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, prompt="what?", enable_thinking=True)
        response = running_adapter.infer_from_request_sync("qwen35/1", request)
        assert response.response == {"reasoning": "...", "answer": "cat"}
        assert (
            running_adapter._client.infer_calls[0]["params"]["enable_thinking"] is True
        )

    def test_missing_prompt_errors(self, running_adapter, monkeypatch):
        self._setup(running_adapter, monkeypatch)
        image = SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode())
        request = od_request(image=image, prompt=None)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("qwen/1", request)

    def test_florence2_terminal_unsupported(self, running_adapter, monkeypatch):
        self._setup(running_adapter, monkeypatch)
        running_adapter._client.model_class_name = "Florence2HF"
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("florence-2-base/1", api_key="key")
        assert running_adapter._client.unloaded == ["florence-2-base/1"]

    def test_moondream_without_prompt_task_unsupported(
        self, running_adapter, monkeypatch
    ):
        self._setup(
            running_adapter,
            monkeypatch,
            tasks=("caption", "detect", "query", "point", "encode"),
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("moondream2/2b", api_key="key")


class SamEmbeddingRequest(SimpleNamespace):
    pass


class Sam2EmbeddingRequest(SimpleNamespace):
    pass


class Sam2SegmentationRequest(SimpleNamespace):
    pass


class Sam3SegmentationRequest(SimpleNamespace):
    pass


def sam_request(cls, **overrides):
    fields = {
        "id": "req-1",
        "api_key": "key",
        "image": SimpleNamespace(
            type="base64", value=base64.b64encode(b"f").decode()
        ),
        "image_id": None,
        "format": "polygon",
    }
    fields.update(overrides)
    return cls(**fields)


class TestSamRouting:
    def _setup(self, running_adapter, monkeypatch, tasks):
        monkeypatch.setattr(
            translation, "stat_model", make_stat("interactive-instance-segmentation", "embed")
        )
        monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))
        running_adapter._client.tasks = {t: {} for t in tasks}

    def test_sam2_embed_echoes_image_id(self, running_adapter, monkeypatch):
        self._setup(running_adapter, monkeypatch, ["embed", "segment"])
        running_adapter._client.infer_result = [
            SimpleNamespace(image_hash="deadbeef")
        ]
        request = sam_request(Sam2EmbeddingRequest, image_id="abc")
        response = running_adapter.infer_from_request_sync("sam2/hiera_large", request)
        assert response.image_id == "abc"
        assert running_adapter._client.infer_calls[0]["task"] == "embed"

    def test_sam3_embed_images_uses_hash_fallback(self, running_adapter, monkeypatch):
        self._setup(
            running_adapter,
            monkeypatch,
            ["embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"],
        )
        running_adapter._client.infer_result = [
            SimpleNamespace(image_hash="deadbeef")
        ]
        request = sam_request(Sam2EmbeddingRequest)
        response = running_adapter.infer_from_request_sync(
            "sam3/sam3_interactive", request
        )
        assert response.image_id == "deadbeef"
        assert running_adapter._client.infer_calls[0]["task"] == "embed_images"
        assert "image_hashes" not in running_adapter._client.infer_calls[0]["params"]

    def test_sam1_embed_returns_embeddings(self, running_adapter, monkeypatch):
        self._setup(running_adapter, monkeypatch, ["embed", "segment"])
        running_adapter._client.infer_result = [
            SimpleNamespace(embeddings=np.ones((1, 2, 2, 2)))
        ]
        request = sam_request(SamEmbeddingRequest, format="json")
        response = running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert response.embeddings == np.ones((1, 2, 2, 2)).tolist()

    def test_sam2_segment_is_unsupported(self, running_adapter, monkeypatch):
        self._setup(running_adapter, monkeypatch, ["embed", "segment"])
        request = sam_request(Sam2SegmentationRequest, prompts=None)
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam2/hiera_large", request)

    def test_sam3_visual_segment(self, running_adapter, monkeypatch):
        from inference.core.entities.requests.sam2 import (
            Box,
            Point,
            Sam2Prompt,
            Sam2PromptSet,
        )

        self._setup(
            running_adapter,
            monkeypatch,
            ["embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"],
        )
        masks = np.zeros((1, 3, 16, 16), dtype=np.float32)
        masks[0, 1, 4:8, 4:8] = 1.0
        running_adapter._client.infer_result = [
            SimpleNamespace(masks=masks, scores=np.asarray([[0.2, 0.9, 0.1]]))
        ]
        prompts = Sam2PromptSet(
            prompts=[
                Sam2Prompt(
                    box=Box(x=10, y=10, width=4, height=4),
                    points=[Point(x=1, y=2, positive=True)],
                )
            ]
        )
        request = sam_request(
            Sam2SegmentationRequest,
            prompts=prompts,
            multimask_output=True,
            image_id="img1",
        )
        response = running_adapter.infer_from_request_sync(
            "sam3/sam3_interactive", request
        )
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["boxes"] == [[[8.0, 8.0, 12.0, 12.0]]]
        assert params["point_coordinates"] == [[[[1.0, 2.0]]]]
        assert params["point_labels"] == [[[1]]]
        assert params["image_hashes"] == ["img1"]
        assert params["multi_mask_output"] is True
        assert len(response.predictions) == 1
        prediction = response.predictions[0]
        assert prediction.confidence == 0.9
        assert prediction.format == "polygon"
        assert len(prediction.masks) > 0

    def test_sam3_text_segment_with_prompt_threshold(
        self, running_adapter, monkeypatch
    ):
        from inference.core.entities.requests.sam3 import Sam3Prompt

        self._setup(
            running_adapter,
            monkeypatch,
            ["embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"],
        )
        masks = np.zeros((2, 16, 16), dtype=np.uint8)
        masks[0, 2:6, 2:6] = 1
        masks[1, 8:12, 8:12] = 1
        running_adapter._client.infer_result = [
            [{"prompt_index": 0, "masks": masks, "scores": [0.9, 0.3]}]
        ]
        prompt = Sam3Prompt(type="text", text="cat", output_prob_thresh=0.5)
        request = sam_request(
            Sam3SegmentationRequest,
            prompts=[prompt],
            output_prob_thresh=0.5,
            nms_iou_threshold=None,
        )
        response = running_adapter.infer_from_request_sync("sam3/sam3_final", request)
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["output_prob_thresh"] == 0.5
        assert params["prompts"][0]["text"] == "cat"
        result = response.prompt_results[0]
        assert result.echo.type == "text" and result.echo.text == "cat"
        assert len(result.predictions) == 1
        assert result.predictions[0].confidence == 0.9

    def test_mask_input_errors(self, running_adapter, monkeypatch):
        self._setup(
            running_adapter,
            monkeypatch,
            ["embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"],
        )
        request = sam_request(
            Sam2SegmentationRequest, prompts=None, mask_input=[[0.0]]
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync(
                "sam3/sam3_interactive", request
            )

    def test_nms_iou_threshold_errors(self, running_adapter, monkeypatch):
        from inference.core.entities.requests.sam3 import Sam3Prompt

        self._setup(
            running_adapter,
            monkeypatch,
            ["embed_images", "segment_with_visual_prompts", "segment_with_text_prompts"],
        )
        request = sam_request(
            Sam3SegmentationRequest,
            prompts=[Sam3Prompt(type="text", text="cat")],
            nms_iou_threshold=0.5,
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam3/sam3_final", request)


class TestUnsupportedModelOps:
    def test_lower_level_ops_raise(self):
        adapter = make_adapter()
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.predict("some/1")
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.model_infer_sync("some/1", request=None)
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.preprocess("some/1", request=None)
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.postprocess("some/1")
        with pytest.raises(ModelDeploymentNotSupportedError):
            adapter.make_response("some/1")


class TestContainerProtocol:
    def test_empty_state(self):
        adapter = make_adapter()
        assert len(adapter) == 0
        assert "some/1" not in adapter
        assert list(adapter.keys()) == []
        assert adapter.models() == {}
        assert adapter.describe_models() == []

    def test_route_state_visible(self):
        adapter = make_adapter()
        _seed_route(adapter, "some/1")
        assert len(adapter) == 1
        assert "some/1" in adapter
        assert list(adapter.keys()) == ["some/1"]
        assert list(adapter.models()) == ["some/1"]
        description = adapter.describe_models()[0]
        assert description.model_id == "some/1"
        assert description.task_type == "object-detection"

    def test_unsupported_routes_hidden(self):
        adapter = make_adapter()
        adapter._routes["some/1"] = {"supported": False, "task_type": "vlm"}
        assert len(adapter) == 0
        assert "some/1" not in adapter
        assert adapter.describe_models() == []

    def test_getitem_returns_inert_stub(self):
        adapter = make_adapter()
        _seed_route(adapter, "some/1")
        stub = adapter["some/1"]
        assert stub.model_id == "some/1"
        assert not hasattr(stub, "flush")
        assert getattr(stub, "_pipeline_depth", None) is None

    def test_getitem_unknown_raises_model_not_found(self):
        with pytest.raises(InferenceModelNotFound):
            make_adapter()["missing/1"]


class TestLegacyDelegation:
    def test_init_pingback_and_attributes_delegate(self):
        adapter = make_adapter()
        adapter.init_pingback()
        assert adapter.num_errors == 0
        assert adapter.pingback is None

    def test_record_request_metadata_delegates(self):
        adapter = make_adapter()
        adapter.record_request_metadata(model_id="some/1")
        assert adapter._legacy.metadata_calls == [{"model_id": "some/1"}]

    def test_pin_model_is_noop(self):
        make_adapter().pin_model("some/1")


class TestMmpReady:
    def test_ready_when_stats_answer(self, running_adapter):
        assert running_adapter.mmp_ready() is True

    def test_not_ready_when_stats_fail(self, running_adapter):
        async def failing_stats():
            raise RuntimeError("stats request failed")

        running_adapter._client.stats = failing_stats
        assert running_adapter.mmp_ready() is False

    def test_not_ready_before_start(self):
        assert make_adapter().mmp_ready() is False


class TestSyncBridge:
    def test_round_trips_result_from_worker_thread(self, running_adapter):
        async def coro():
            return 42

        assert running_adapter._run_sync(coro()) == 42

    def test_raises_when_called_on_adapter_loop(self, running_adapter):
        async def on_loop():
            running_adapter._run_sync(asyncio.sleep(0))

        future = asyncio.run_coroutine_threadsafe(on_loop(), running_adapter._loop)
        with pytest.raises(RuntimeError, match="deadlock"):
            future.result(timeout=5)

    def test_raises_before_start(self):
        with pytest.raises(RuntimeError, match="before start"):
            make_adapter()._run_sync(asyncio.sleep(0))


class TestRemoveAndClear:
    def test_remove_unloads_and_drops_route(self, running_adapter):
        _seed_route(running_adapter, "some/1")
        running_adapter.remove("some/1")
        assert running_adapter._client.unloaded == ["some/1"]
        assert "some/1" not in running_adapter

    def test_remove_drops_alias_keys_too(self, running_adapter):
        route = _seed_route(running_adapter, "some/1")
        running_adapter._routes["alias/1"] = route
        running_adapter.remove("alias/1")
        assert running_adapter._client.unloaded == ["some/1"]
        assert len(running_adapter) == 0

    def test_remove_unknown_is_noop(self, running_adapter):
        running_adapter.remove("missing/1")
        assert running_adapter._client.unloaded == []

    def test_clear_unloads_all_and_drops_terminal_state(self, running_adapter):
        _seed_route(running_adapter, "a/1")
        _seed_route(running_adapter, "b/2")
        running_adapter._routes["vlm/1"] = {"supported": False}
        running_adapter.clear()
        assert sorted(running_adapter._client.unloaded) == ["a/1", "b/2"]
        assert running_adapter._routes == {}
