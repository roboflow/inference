import asyncio
import base64
import threading

import numpy as np
import pytest

from inference.core.entities.requests.clip import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
)
from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.perception_encoder import (
    PerceptionEncoderCompareRequest,
    PerceptionEncoderImageEmbeddingRequest,
    PerceptionEncoderTextEmbeddingRequest,
)
from inference.core.entities.responses.clip import (
    ClipCompareResponse,
    ClipEmbeddingResponse,
)
from inference.core.entities.responses.perception_encoder import (
    PerceptionEncoderCompareResponse,
    PerceptionEncoderEmbeddingResponse,
)
from inference.core.env import CLIP_MAX_BATCH_SIZE
from inference.core.exceptions import (
    ModelDeploymentNotSupportedError,
    RoboflowAPITimeoutError,
)
from inference.core.managers import mmp_translation as translation
from inference.core.managers.mmp_adapter import ModelManagerAdapter


class FakeEmbeddingClient:
    load_wait_s = 1.0
    infer_timeout_s = 1.0
    n_slots = 4

    def __init__(self):
        self.loaded = []
        self.unloaded = []
        self.infer_calls = []
        self.load_result = ("ok",)
        self.ensure_result = ("model_ready",)
        self.tasks = {"embed_images": {}, "embed_text": {}}
        self.image_results = {}
        self.text_results = {}
        self.infer_error = None

    async def start(self):
        pass

    async def shutdown(self):
        pass

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

    async def stats(self):
        entry = {
            "class_names": None,
            "key_points_classes": None,
            "model_class_name": "ClipOnnx",
        }
        return {"mmp_models": {m: dict(entry) for m in self.loaded}}

    async def infer(self, *, model_id, image, task=None, instance="", params=None, **kw):
        self.infer_calls.append(
            {"model_id": model_id, "image": image, "task": task, "params": params}
        )
        if self.infer_error is not None:
            raise self.infer_error
        if task == "embed_text":
            return self.text_results[tuple(params["texts"])]
        return self.image_results[image]


class FakeLegacy:
    def init_pingback(self):
        pass

    def record_request_metadata(self, **kwargs):
        pass


@pytest.fixture
def running_adapter():
    adapter = ModelManagerAdapter(
        legacy_stack=FakeLegacy(), mmp_client=FakeEmbeddingClient()
    )
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
def embedding_stat(monkeypatch):
    async def fake_stat(model_id, api_key):
        return ("embedding", "embed_images")

    monkeypatch.setattr(translation, "stat_model", fake_stat)
    monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))


def make_image(tag):
    payload = f"img-{tag}".encode()
    image = InferenceRequestImage(
        type="base64", value=base64.b64encode(payload).decode()
    )
    return image, payload


class TestEmbeddingRouting:
    def test_clip_model_resolves_and_loads(self, running_adapter, embedding_stat):
        running_adapter.add_model("clip/ViT-B-16", api_key="key")
        assert running_adapter._client.loaded == ["clip/ViT-B-16"]
        assert "clip/ViT-B-16" in running_adapter
        assert running_adapter.get_task_type("clip/ViT-B-16") == "embedding"

    def test_model_without_embedding_tasks_is_unsupported(
        self, running_adapter, embedding_stat
    ):
        running_adapter._client.tasks = {"infer": {}}
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("clip/ViT-B-16", api_key="key")
        assert running_adapter._client.unloaded == ["clip/ViT-B-16"]


class TestEmbedImage:
    def test_single_image(self, running_adapter, embedding_stat):
        image, payload = make_image("a")
        running_adapter._client.image_results[payload] = np.asarray(
            [[1.0, 2.0, 3.0, 4.0]]
        )
        request = ClipImageEmbeddingRequest(api_key="key", image=image)
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response, ClipEmbeddingResponse)
        assert response.embeddings == [[1.0, 2.0, 3.0, 4.0]]
        call = running_adapter._client.infer_calls[0]
        assert call["task"] == "embed_images"
        assert call["image"] == payload
        assert call["params"] == {}
        assert response.time is not None
        assert response.inference_id is None

    def test_image_batch_returns_single_response_with_ordered_rows(
        self, running_adapter, embedding_stat
    ):
        images = []
        for index, tag in enumerate(["a", "b", "c"]):
            image, payload = make_image(tag)
            row = [0.0, 0.0, 0.0]
            row[index] = 1.0
            running_adapter._client.image_results[payload] = np.asarray([row])
            images.append(image)
        request = ClipImageEmbeddingRequest(api_key="key", image=images)
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response, ClipEmbeddingResponse)
        assert response.embeddings == [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        assert len(running_adapter._client.infer_calls) == 3
        assert all(
            call["task"] == "embed_images"
            for call in running_adapter._client.infer_calls
        )

    def test_image_batch_over_limit_errors_before_fetch(
        self, running_adapter, embedding_stat, monkeypatch
    ):
        fetches = []
        monkeypatch.setattr(
            translation, "forward_image", lambda image: fetches.append(image)
        )
        images = [make_image(str(i))[0] for i in range(CLIP_MAX_BATCH_SIZE + 1)]
        request = ClipImageEmbeddingRequest(api_key="key", image=images)
        with pytest.raises(ValueError, match="maximum number of images"):
            running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert fetches == []
        assert running_adapter._client.infer_calls == []


class TestEmbedText:
    def test_single_text(self, running_adapter, embedding_stat):
        running_adapter._client.text_results[("a dog",)] = np.asarray([[1.0, 0.5]])
        request = ClipTextEmbeddingRequest(api_key="key", text="a dog")
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response, ClipEmbeddingResponse)
        assert response.embeddings == [[1.0, 0.5]]
        call = running_adapter._client.infer_calls[0]
        assert call["task"] == "embed_text"
        assert call["image"] is None
        assert call["params"] == {"texts": ["a dog"]}
        assert response.time is not None
        assert response.inference_id is None

    def test_text_list_uses_single_call(self, running_adapter, embedding_stat):
        texts = ["a", "b", "c"]
        running_adapter._client.text_results[tuple(texts)] = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        )
        request = ClipTextEmbeddingRequest(api_key="key", text=texts)
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert response.embeddings == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        assert len(running_adapter._client.infer_calls) == 1

    def test_infer_timeout_translates(self, running_adapter, embedding_stat):
        running_adapter._client.infer_error = asyncio.TimeoutError()
        request = ClipTextEmbeddingRequest(api_key="key", text="x")
        with pytest.raises(RoboflowAPITimeoutError):
            running_adapter.infer_from_request_sync("clip/ViT-B-16", request)


class TestCompare:
    def test_image_subject_single_text_prompt(self, running_adapter, embedding_stat):
        image, payload = make_image("subject")
        client = running_adapter._client
        client.image_results[payload] = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        client.text_results[("dog",)] = np.asarray([[1.0, 1.0, 0.0, 0.0]])
        request = ClipCompareRequest(api_key="key", subject=image, prompt="dog")
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response, ClipCompareResponse)
        assert response.similarity == [pytest.approx(1.0 / np.sqrt(2.0))]
        assert [call["task"] for call in client.infer_calls] == [
            "embed_images",
            "embed_text",
        ]
        assert response.time is not None

    def test_image_subject_text_prompt_list_order(
        self, running_adapter, embedding_stat
    ):
        image, payload = make_image("subject")
        client = running_adapter._client
        client.image_results[payload] = np.asarray([[1.0, 0.0]])
        client.text_results[("a", "b")] = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        request = ClipCompareRequest(api_key="key", subject=image, prompt=["a", "b"])
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert response.similarity == [pytest.approx(1.0), pytest.approx(0.0)]
        assert len(client.infer_calls) == 2

    def test_text_subject_single_image_prompt(self, running_adapter, embedding_stat):
        image, payload = make_image("prompt")
        client = running_adapter._client
        client.text_results[("dog",)] = np.asarray([[1.0, 0.0]])
        client.image_results[payload] = np.asarray([[1.0, 0.0]])
        request = ClipCompareRequest(
            api_key="key",
            subject="dog",
            subject_type="text",
            prompt=image,
            prompt_type="image",
        )
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert response.similarity == [pytest.approx(1.0)]
        assert [call["task"] for call in client.infer_calls] == [
            "embed_text",
            "embed_images",
        ]

    def test_text_subject_single_image_prompt_dict_payload(
        self, running_adapter, embedding_stat
    ):
        image, payload = make_image("prompt")
        client = running_adapter._client
        client.text_results[("dog",)] = np.asarray([[1.0, 0.0]])
        client.image_results[payload] = np.asarray([[1.0, 0.0]])
        request = ClipCompareRequest(
            api_key="key",
            subject="dog",
            subject_type="text",
            prompt={"type": image.type, "value": image.value},
            prompt_type="image",
        )
        assert isinstance(request.prompt, dict)
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response, ClipCompareResponse)
        assert response.similarity == [pytest.approx(1.0)]
        assert [call["task"] for call in client.infer_calls] == [
            "embed_text",
            "embed_images",
        ]

    def test_text_subject_url_dict_prompt_forwards_fetched_bytes(
        self, running_adapter, embedding_stat, monkeypatch
    ):
        payload = b"fetched-image-bytes"
        fetched = []

        def fake_fetch(value):
            fetched.append(value)
            return payload

        monkeypatch.setattr(translation, "fetch_image_bytes_from_url", fake_fetch)
        client = running_adapter._client
        client.text_results[("The quick brown fox jumps over the lazy dog.",)] = (
            np.asarray([[1.0, 0.0]])
        )
        client.image_results[payload] = np.asarray([[1.0, 0.0]])
        request = ClipCompareRequest(
            api_key="key",
            subject="The quick brown fox jumps over the lazy dog.",
            subject_type="text",
            prompt={"type": "url", "value": "https://example.com/original.jpg"},
            prompt_type="image",
        )
        assert isinstance(request.prompt, dict)
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response, ClipCompareResponse)
        assert response.similarity == [pytest.approx(1.0)]
        assert fetched == ["https://example.com/original.jpg"]
        assert [call["task"] for call in client.infer_calls] == [
            "embed_text",
            "embed_images",
        ]
        assert client.infer_calls[1]["image"] == payload

    def test_text_subject_image_prompt_list(self, running_adapter, embedding_stat):
        client = running_adapter._client
        client.text_results[("dog",)] = np.asarray([[1.0, 0.0]])
        prompts = []
        first, first_payload = make_image("p1")
        second, second_payload = make_image("p2")
        client.image_results[first_payload] = np.asarray([[1.0, 0.0]])
        client.image_results[second_payload] = np.asarray([[0.0, 1.0]])
        prompts = [first, second]
        request = ClipCompareRequest(
            api_key="key",
            subject="dog",
            subject_type="text",
            prompt=prompts,
            prompt_type="image",
        )
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert response.similarity == [pytest.approx(1.0), pytest.approx(0.0)]
        assert [call["task"] for call in client.infer_calls] == [
            "embed_text",
            "embed_images",
            "embed_images",
        ]

    def test_named_text_prompts_return_dict(self, running_adapter, embedding_stat):
        image, payload = make_image("subject")
        client = running_adapter._client
        client.image_results[payload] = np.asarray([[1.0, 0.0]])
        client.text_results[("a", "b")] = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        request = ClipCompareRequest(
            api_key="key", subject=image, prompt={"Key1": "a", "Key2": "b"}
        )
        response = running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert isinstance(response.similarity, dict)
        assert response.similarity == {
            "Key1": pytest.approx(1.0),
            "Key2": pytest.approx(0.0),
        }

    def test_prompt_list_over_limit_errors_before_fetch(
        self, running_adapter, embedding_stat, monkeypatch
    ):
        fetches = []
        monkeypatch.setattr(
            translation, "forward_image", lambda image: fetches.append(image)
        )
        image, _ = make_image("subject")
        prompts = [f"prompt-{i}" for i in range(CLIP_MAX_BATCH_SIZE + 1)]
        request = ClipCompareRequest(api_key="key", subject=image, prompt=prompts)
        with pytest.raises(ValueError, match="maximum number of prompts"):
            running_adapter.infer_from_request_sync("clip/ViT-B-16", request)
        assert fetches == []
        assert running_adapter._client.infer_calls == []

    def test_invalid_subject_type_errors(self, running_adapter, embedding_stat):
        request = ClipCompareRequest(
            api_key="key", subject="dog", subject_type="audio", prompt="cat"
        )
        with pytest.raises(ValueError):
            running_adapter.infer_from_request_sync("clip/ViT-B-16", request)

    def test_invalid_prompt_type_errors(self, running_adapter, embedding_stat):
        request = ClipCompareRequest(
            api_key="key",
            subject="dog",
            subject_type="text",
            prompt="cat",
            prompt_type="audio",
        )
        with pytest.raises(ValueError):
            running_adapter.infer_from_request_sync("clip/ViT-B-16", request)

    def test_compare_without_needed_task_unsupported(
        self, running_adapter, embedding_stat
    ):
        running_adapter._client.tasks = {"embed_images": {}}
        image, payload = make_image("subject")
        running_adapter._client.image_results[payload] = np.asarray([[1.0, 0.0]])
        request = ClipCompareRequest(api_key="key", subject=image, prompt="dog")
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("clip/ViT-B-16", request)


class TestPerceptionEncoder:
    def test_embed_image_uses_pe_response(self, running_adapter, embedding_stat):
        image, payload = make_image("a")
        running_adapter._client.image_results[payload] = np.asarray([[1.0, 2.0]])
        request = PerceptionEncoderImageEmbeddingRequest(api_key="key", image=image)
        response = running_adapter.infer_from_request_sync(
            "perception_encoder/PE-Core-L14-336", request
        )
        assert isinstance(response, PerceptionEncoderEmbeddingResponse)
        assert response.embeddings == [[1.0, 2.0]]

    def test_embed_text_uses_pe_response(self, running_adapter, embedding_stat):
        running_adapter._client.text_results[("a dog",)] = np.asarray([[1.0, 0.5]])
        request = PerceptionEncoderTextEmbeddingRequest(api_key="key", text="a dog")
        response = running_adapter.infer_from_request_sync(
            "perception_encoder/PE-Core-L14-336", request
        )
        assert isinstance(response, PerceptionEncoderEmbeddingResponse)
        assert response.embeddings == [[1.0, 0.5]]

    def test_mmp_calls_use_platform_model_id(self, running_adapter, embedding_stat):
        image, payload = make_image("a")
        client = running_adapter._client
        client.image_results[payload] = np.asarray([[1.0, 2.0]])
        request = PerceptionEncoderImageEmbeddingRequest(api_key="key", image=image)
        response = running_adapter.infer_from_request_sync(
            "perception_encoder/PE-Core-L14-336", request
        )
        assert isinstance(response, PerceptionEncoderEmbeddingResponse)
        assert client.loaded == ["perception-encoder/PE-Core-L14-336"]
        assert client.infer_calls[0]["model_id"] == "perception-encoder/PE-Core-L14-336"
        assert "perception_encoder/PE-Core-L14-336" in running_adapter

    def test_compare_uses_pe_response(self, running_adapter, embedding_stat):
        image, payload = make_image("subject")
        client = running_adapter._client
        client.image_results[payload] = np.asarray([[1.0, 0.0]])
        client.text_results[("dog",)] = np.asarray([[1.0, 0.0]])
        request = PerceptionEncoderCompareRequest(
            api_key="key", subject=image, prompt="dog"
        )
        response = running_adapter.infer_from_request_sync(
            "perception_encoder/PE-Core-L14-336", request
        )
        assert isinstance(response, PerceptionEncoderCompareResponse)
        assert response.similarity == [pytest.approx(1.0)]
