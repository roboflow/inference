import asyncio
import base64
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam import (
    SamEmbeddingRequest,
    SamSegmentationRequest,
)
from inference.core.entities.requests.sam2 import (
    Box,
    Point,
    Sam2EmbeddingRequest,
    Sam2Prompt,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.sam import SamSegmentationResponse
from inference.core.entities.responses.sam2 import Sam2SegmentationResponse
from inference.core.exceptions import ModelDeploymentNotSupportedError
from inference.core.managers import mmp_translation as translation
from inference.core.managers.mmp_adapter import ModelManagerAdapter
from inference.core.utils.postprocess import masks2multipoly, masks2poly


class FakeSamClient:
    load_wait_s = 1.0
    infer_timeout_s = 1.0
    n_slots = 4

    def __init__(self):
        self.loaded = []
        self.unloaded = []
        self.infer_calls = []
        self.load_result = ("ok",)
        self.ensure_result = ("model_ready",)
        self.tasks = {"embed": {}, "segment": {}}
        self.infer_result = None
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
            "model_class_name": "SAMTorch",
        }
        return {"mmp_models": {m: dict(entry) for m in self.loaded}}

    async def infer(self, *, model_id, image, task=None, instance="", params=None, **kw):
        self.infer_calls.append(
            {"model_id": model_id, "image": image, "task": task, "params": params}
        )
        if self.infer_error is not None:
            raise self.infer_error
        return self.infer_result


class FakeLegacy:
    def init_pingback(self):
        pass

    def record_request_metadata(self, **kwargs):
        pass


@pytest.fixture
def running_adapter():
    adapter = ModelManagerAdapter(legacy_stack=FakeLegacy(), mmp_client=FakeSamClient())
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
def interactive_stat(monkeypatch):
    async def fake_stat(model_id, api_key):
        return ("interactive-instance-segmentation", "embed")

    monkeypatch.setattr(translation, "stat_model", fake_stat)
    monkeypatch.setattr(translation, "_read_image_dims", lambda data: (64, 48))


def make_image(tag="a"):
    payload = f"img-{tag}".encode()
    image = InferenceRequestImage(
        type="base64", value=base64.b64encode(payload).decode()
    )
    return image, payload


def sam_segmentation_result():
    masks = np.zeros((1, 8, 8), dtype=bool)
    masks[0, 2:6, 2:6] = True
    logits = np.full((1, 8, 8), -4.0, dtype=np.float32)
    logits[0, 3:5, 3:5] = 4.0
    return [SimpleNamespace(masks=masks, scores=np.asarray([0.9]), logits=logits)]


def sam2_segmentation_result():
    masks = np.full((1, 3, 16, 16), -5.0, dtype=np.float32)
    masks[0, 1, 4:8, 4:8] = 5.0
    return [SimpleNamespace(masks=masks, scores=np.asarray([[0.2, 0.9, 0.1]]))]


class TestSamSegmentParams:
    def test_cached_image_id_flow_goes_imageless(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam_segmentation_result()
        request = SamSegmentationRequest(
            api_key="key",
            image=None,
            image_id="test",
            point_coords=[[240, 195]],
            point_labels=[1],
        )
        response = running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert isinstance(response, SamSegmentationResponse)
        call = running_adapter._client.infer_calls[0]
        assert call["task"] == "segment"
        assert call["image"] is None
        assert call["params"] == {
            "multi_mask_output": False,
            "point_coordinates": [[[240.0, 195.0]]],
            "point_labels": [[1.0]],
            "image_hashes": ["test"],
        }

    def test_image_flow_forwards_bytes_and_id(self, running_adapter, interactive_stat):
        running_adapter._client.infer_result = sam_segmentation_result()
        image, payload = make_image()
        request = SamSegmentationRequest(
            api_key="key",
            image=image,
            image_id="test",
            point_coords=[[10, 20], [30, 40]],
            point_labels=[1, 0],
        )
        running_adapter.infer_from_request_sync("sam/vit_h", request)
        call = running_adapter._client.infer_calls[0]
        assert call["image"] == payload
        assert call["params"]["image_hashes"] == ["test"]
        assert call["params"]["point_coordinates"] == [
            [[10.0, 20.0], [30.0, 40.0]]
        ]
        assert call["params"]["point_labels"] == [[1.0, 0.0]]

    def test_default_sentinel_points_forwarded(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam_segmentation_result()
        image, _ = make_image()
        request = SamSegmentationRequest(api_key="key", image=image)
        running_adapter.infer_from_request_sync("sam/vit_h", request)
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["point_coordinates"] == [[[0.0, 0.0]]]
        assert params["point_labels"] == [[-1.0]]
        assert "image_hashes" not in params

    def test_embeddings_input_refused(self, running_adapter, interactive_stat):
        request = SamSegmentationRequest(
            api_key="key",
            embeddings=[[[[0.1, 0.2]]]],
            orig_im_size=[64, 48],
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert running_adapter._client.infer_calls == []

    def test_missing_image_and_id_raises_legacy_value_error(
        self, running_adapter, interactive_stat
    ):
        request = SamSegmentationRequest(api_key="key")
        with pytest.raises(ValueError) as error:
            running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert (
            str(error.value)
            == "Must provide either image, cached image_id, or embeddings"
        )

    def test_mask_cache_flow_enforces_mask_input(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam_segmentation_result()
        request = SamSegmentationRequest(
            api_key="key",
            image_id="test",
            has_mask_input=True,
        )
        running_adapter.infer_from_request_sync("sam/vit_h", request)
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["enforce_mask_input"] is True

    def test_mask_cache_flow_without_image_id_raises_legacy_value_error(
        self, running_adapter, interactive_stat
    ):
        image, _ = make_image()
        request = SamSegmentationRequest(
            api_key="key",
            image=image,
            has_mask_input=True,
        )
        with pytest.raises(ValueError) as error:
            running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert str(error.value) == "Must provide either mask_input or cached image_id"

    def test_explicit_mask_input_refused(self, running_adapter, interactive_stat):
        request = SamSegmentationRequest(
            api_key="key",
            image_id="test",
            has_mask_input=True,
            mask_input=[[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]]],
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam/vit_h", request)

    def test_mask_cache_disabled_refused(self, running_adapter, interactive_stat):
        request = SamSegmentationRequest(
            api_key="key",
            image_id="test",
            has_mask_input=True,
            use_mask_input_cache=False,
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam/vit_h", request)

    def test_binary_format_refused(self, running_adapter, interactive_stat):
        request = SamSegmentationRequest(
            api_key="key", image_id="test", format="binary"
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam/vit_h", request)

    @pytest.mark.parametrize("bad_format", [None, "foo"])
    def test_invalid_format_raises_legacy_value_error(
        self, running_adapter, interactive_stat, bad_format
    ):
        request = SamSegmentationRequest(
            api_key="key", image_id="test", format=bad_format
        )
        with pytest.raises(ValueError) as error:
            running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert str(error.value) == f"Invalid format {bad_format}"

    def test_segment_refused_when_model_lacks_task(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.tasks = {"embed": {}}
        request = SamSegmentationRequest(api_key="key", image_id="test")
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam/vit_h", request)


class TestSamSegmentRepack:
    def test_json_repack_polygonizes_masks_and_low_res(
        self, running_adapter, interactive_stat
    ):
        result = sam_segmentation_result()
        running_adapter._client.infer_result = result
        request = SamSegmentationRequest(api_key="key", image_id="test")
        response = running_adapter.infer_from_request_sync("sam/vit_h", request)
        assert isinstance(response, SamSegmentationResponse)
        expected_masks = [
            polygon.tolist() for polygon in masks2poly(result[0].masks)
        ]
        expected_low_res = [
            polygon.tolist() for polygon in masks2poly(result[0].logits > 0.0)
        ]
        assert response.masks == expected_masks
        assert response.low_res_masks == expected_low_res
        assert response.time >= 0.0

    def test_json_repack_thresholds_float_masks(
        self, running_adapter, interactive_stat
    ):
        masks = np.full((1, 8, 8), -2.0, dtype=np.float32)
        masks[0, 2:6, 2:6] = 3.0
        logits = np.full((1, 8, 8), -2.0, dtype=np.float32)
        running_adapter._client.infer_result = [
            SimpleNamespace(masks=masks, scores=np.asarray([0.5]), logits=logits)
        ]
        request = SamSegmentationRequest(api_key="key", image_id="test")
        response = running_adapter.infer_from_request_sync("sam/vit_h", request)
        expected_masks = [polygon.tolist() for polygon in masks2poly(masks > 0.0)]
        assert response.masks == expected_masks
        assert response.low_res_masks == [[]]


class TestSam2Segment:
    def test_points_and_boxes_translation(self, running_adapter, interactive_stat):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, payload = make_image()
        request = Sam2SegmentationRequest(
            api_key="key",
            image=image,
            image_id="img1",
            prompts=Sam2PromptSet(
                prompts=[
                    Sam2Prompt(
                        box=Box(x=10, y=10, width=4, height=4),
                        points=[Point(x=1, y=2, positive=True)],
                    )
                ]
            ),
        )
        response = running_adapter.infer_from_request_sync(
            "sam2/hiera_large", request
        )
        call = running_adapter._client.infer_calls[0]
        assert call["task"] == "segment"
        assert call["image"] == payload
        params = call["params"]
        assert params["boxes"] == [[[8.0, 8.0, 12.0, 12.0]]]
        assert params["point_coordinates"] == [[[[1.0, 2.0]]]]
        assert params["point_labels"] == [[[1]]]
        assert params["image_hashes"] == ["img1"]
        assert params["multi_mask_output"] is True
        assert params["return_logits"] is True
        assert isinstance(response, Sam2SegmentationResponse)
        assert len(response.predictions) == 1
        assert response.predictions[0].confidence == pytest.approx(0.9)
        assert response.predictions[0].format == "polygon"
        assert len(response.predictions[0].masks) > 0

    def test_json_format_maps_to_polygon_predictions(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, _ = make_image()
        request = Sam2SegmentationRequest(api_key="key", image=image, format="json")
        response = running_adapter.infer_from_request_sync(
            "sam2/hiera_large", request
        )
        assert response.predictions[0].format == "polygon"

    def test_empty_prompts_inject_legacy_sentinel_point(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, _ = make_image()
        request = Sam2SegmentationRequest(api_key="key", image=image)
        running_adapter.infer_from_request_sync("sam2/hiera_large", request)
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["point_coordinates"] == [[[0, 0]]]
        assert params["point_labels"] == [[-1]]
        assert "boxes" not in params

    def test_negative_point_translates_to_label_zero(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, _ = make_image()
        request = Sam2SegmentationRequest(
            api_key="key",
            image=image,
            prompts=Sam2PromptSet(
                prompts=[
                    Sam2Prompt(
                        points=[
                            Point(x=5, y=6, positive=True),
                            Point(x=7, y=8, positive=False),
                        ]
                    )
                ]
            ),
        )
        running_adapter.infer_from_request_sync("sam2/hiera_large", request)
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["point_coordinates"] == [[[[5.0, 6.0], [7.0, 8.0]]]]
        assert params["point_labels"] == [[[1, 0]]]

    def test_zero_logits_included_by_legacy_threshold(
        self, running_adapter, interactive_stat
    ):
        masks = np.full((1, 1, 8, 8), -5.0, dtype=np.float32)
        masks[0, 0, 2:6, 2:6] = 0.0
        running_adapter._client.infer_result = [
            SimpleNamespace(masks=masks, scores=np.asarray([[0.7]]))
        ]
        image, _ = make_image()
        request = Sam2SegmentationRequest(api_key="key", image=image)
        response = running_adapter.infer_from_request_sync(
            "sam2/hiera_large", request
        )
        expected = [
            polygon.tolist() for polygon in masks2multipoly(masks[0, 0:1] >= 0.0)[0]
        ]
        assert response.predictions[0].masks == expected
        assert len(response.predictions[0].masks) > 0

    @pytest.mark.parametrize("bad_format", [None, "polygon", "bogus"])
    def test_invalid_format_raises_legacy_value_error(
        self, running_adapter, interactive_stat, bad_format
    ):
        image, _ = make_image()
        request = Sam2SegmentationRequest(
            api_key="key", image=image, format=bad_format
        )
        with pytest.raises(ValueError) as error:
            running_adapter.infer_from_request_sync("sam2/hiera_large", request)
        assert str(error.value) == f"Invalid format {bad_format}"
        assert running_adapter._client.infer_calls == []

    def test_rle_format(self, running_adapter, interactive_stat):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, _ = make_image()
        request = Sam2SegmentationRequest(api_key="key", image=image, format="rle")
        response = running_adapter.infer_from_request_sync(
            "sam2/hiera_large", request
        )
        prediction = response.predictions[0]
        assert prediction.format == "rle"
        assert isinstance(prediction.masks, dict)
        assert prediction.masks["size"] == [16, 16]
        assert isinstance(prediction.masks["counts"], str)

    def test_binary_format_refused(self, running_adapter, interactive_stat):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, _ = make_image()
        request = Sam2SegmentationRequest(api_key="key", image=image, format="binary")
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam2/hiera_large", request)

    def test_multimask_output_false_forwarded(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = sam2_segmentation_result()
        image, _ = make_image()
        request = Sam2SegmentationRequest(
            api_key="key", image=image, multimask_output=False
        )
        running_adapter.infer_from_request_sync("sam2/hiera_large", request)
        params = running_adapter._client.infer_calls[0]["params"]
        assert params["multi_mask_output"] is False

    def test_logits_cache_flags_refused(self, running_adapter, interactive_stat):
        image, _ = make_image()
        request = Sam2SegmentationRequest(
            api_key="key", image=image, save_logits_to_cache=True
        )
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.infer_from_request_sync("sam2/hiera_large", request)


class TestEmbedImageIdLifecycle:
    def test_sam1_embed_forwards_image_id(self, running_adapter, interactive_stat):
        running_adapter._client.infer_result = [
            SimpleNamespace(embeddings=np.ones((1, 2, 2, 2)))
        ]
        image, _ = make_image()
        request = SamEmbeddingRequest(
            api_key="key", image=image, image_id="test", format="json"
        )
        response = running_adapter.infer_from_request_sync("sam/vit_h", request)
        call = running_adapter._client.infer_calls[0]
        assert call["task"] == "embed"
        assert call["params"] == {"image_hashes": ["test"]}
        assert response.embeddings == np.ones((1, 2, 2, 2)).tolist()

    def test_sam2_embed_forwards_image_id_and_echoes_it(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = [
            SimpleNamespace(image_hash="deadbeef")
        ]
        image, _ = make_image()
        request = Sam2EmbeddingRequest(api_key="key", image=image, image_id="abc")
        response = running_adapter.infer_from_request_sync(
            "sam2/hiera_large", request
        )
        params = running_adapter._client.infer_calls[0]["params"]
        assert params == {"image_hashes": ["abc"]}
        assert response.image_id == "abc"

    def test_sam2_embed_without_id_returns_worker_hash(
        self, running_adapter, interactive_stat
    ):
        running_adapter._client.infer_result = [
            SimpleNamespace(image_hash="deadbeef")
        ]
        image, _ = make_image()
        request = Sam2EmbeddingRequest(api_key="key", image=image)
        response = running_adapter.infer_from_request_sync(
            "sam2/hiera_large", request
        )
        params = running_adapter._client.infer_calls[0]["params"]
        assert "image_hashes" not in params
        assert response.image_id == "deadbeef"
