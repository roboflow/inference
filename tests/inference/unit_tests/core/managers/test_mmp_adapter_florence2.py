import asyncio
import base64
import threading
from types import SimpleNamespace

import pytest

from inference.core.exceptions import (
    InvalidImageTypeDeclared,
    ModelArtefactError,
    ModelDeploymentNotSupportedError,
)
from inference.core.managers import mmp_florence2
from inference.core.managers import mmp_translation as translation
from inference.core.managers.mmp_adapter import ModelManagerAdapter

FLORENCE_TASKS = (
    "caption",
    "detect",
    "ocr",
    "parse_document",
    "prompt",
    "segment_phrase",
    "ground_phrase",
    "classify_region",
    "caption_region",
    "ocr_region",
    "segment_region",
)

FLORENCE_ROUTE = {
    "model_class_name": "Florence2HF",
    "model_mro_names": None,
}


class FakeFlorenceClient:
    load_wait_s = 1.0
    infer_timeout_s = 1.0
    n_slots = 4

    def __init__(self):
        self.loaded = []
        self.unloaded = []
        self.infer_calls = []
        self.load_result = ("ok",)
        self.ensure_result = ("model_ready",)
        self.tasks = {task: {} for task in FLORENCE_TASKS}
        self.model_class_name = "Florence2HF"
        self.model_mro_names = ["Florence2HF", "object"]
        self.infer_result = (
            "</s><s>A green car parked in front of a yellow building.</s>"
        )
        self.infer_error = None

    async def start(self):
        pass

    async def shutdown(self):
        pass

    async def load(self, model_id, api_key="", timeout_s=None):
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
            "model_class_name": self.model_class_name,
            "model_mro_names": self.model_mro_names,
        }
        return {"mmp_models": {m: dict(entry) for m in self.loaded}}

    async def infer(
        self, *, model_id, image, task=None, instance="", params=None, **kw
    ):
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
    adapter = ModelManagerAdapter(
        legacy_stack=FakeLegacy(), mmp_client=FakeFlorenceClient()
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
def florence_stat(monkeypatch):
    async def fake_stat(model_id, api_key):
        return ("vlm", "prompt")

    monkeypatch.setattr(translation, "stat_model", fake_stat)
    monkeypatch.setattr(translation, "_read_image_dims", lambda data: (640, 480))


def lmm_request(prompt, **overrides):
    fields = {
        "id": "req-1",
        "api_key": "key",
        "prompt": prompt,
        "max_new_tokens": None,
        "enable_thinking": False,
        "visualize_predictions": False,
        "disable_preproc_auto_orient": False,
        "disable_preproc_contrast": False,
        "disable_preproc_grayscale": False,
        "disable_preproc_static_crop": False,
        "image": SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode()),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestFlorence2RouteSelection:
    def test_route_matched_by_model_class_name(self):
        route = {"model_class_name": "Florence2HF", "model_mro_names": None}
        assert mmp_florence2.is_florence2_route(route) is True

    def test_route_matched_by_mro(self):
        route = {
            "model_class_name": "SomeFineTunedFlorence",
            "model_mro_names": ["SomeFineTunedFlorence", "Florence2HF", "object"],
        }
        assert mmp_florence2.is_florence2_route(route) is True

    def test_other_vlm_not_matched(self):
        route = {
            "model_class_name": "Qwen25VLHF",
            "model_mro_names": ["Qwen25VLHF", "object"],
        }
        assert mmp_florence2.is_florence2_route(route) is False


class TestFlorence2PromptParams:
    @pytest.mark.parametrize(
        "prompt",
        [
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<OCR>",
            "<OCR_WITH_REGION>",
            "<OD>",
            "<DENSE_REGION_CAPTION>",
            "<REGION_PROPOSAL>",
            "<CAPTION_TO_PHRASE_GROUNDING>A green car",
            "<REFERRING_EXPRESSION_SEGMENTATION>a green car",
            "<REGION_TO_SEGMENTATION><loc_50><loc_60><loc_150><loc_160>",
            "<OPEN_VOCABULARY_DETECTION>a green car",
            "<REGION_TO_CATEGORY><loc_154><loc_258><loc_903><loc_621>",
            "<REGION_TO_DESCRIPTION><loc_154><loc_258><loc_903><loc_621>",
            "<REGION_TO_OCR><loc_154><loc_258><loc_903><loc_621>",
        ],
    )
    def test_prompt_forwarded_verbatim_with_legacy_generation_params(self, prompt):
        request = lmm_request(prompt)
        params = mmp_florence2.build_prompt_params(request)
        assert params == {
            "prompt": prompt,
            "max_new_tokens": 1000,
            "do_sample": False,
            "skip_special_tokens": False,
        }

    def test_request_max_new_tokens_is_ignored_like_legacy(self):
        request = lmm_request("<CAPTION>", max_new_tokens=512)
        params = mmp_florence2.build_prompt_params(request)
        assert params["max_new_tokens"] == 1000

    def test_enable_thinking_is_ignored_like_legacy(self):
        request = lmm_request("<CAPTION>", enable_thinking=True)
        params = mmp_florence2.build_prompt_params(request)
        assert "enable_thinking" not in params

    def test_empty_prompt_is_allowed(self):
        request = lmm_request("")
        params = mmp_florence2.build_prompt_params(request)
        assert params["prompt"] == ""

    def test_missing_prompt_fails_like_legacy(self):
        request = lmm_request(None)
        with pytest.raises(
            AttributeError, match="'NoneType' object has no attribute 'split'"
        ):
            mmp_florence2.build_prompt_params(request)

    def test_build_task_params_dispatches_florence_route(self):
        request = lmm_request("<CAPTION>", max_new_tokens=64)
        params = translation.build_task_params("vlm", "prompt", request, FLORENCE_ROUTE)
        assert params == {
            "prompt": "<CAPTION>",
            "max_new_tokens": 1000,
            "do_sample": False,
            "skip_special_tokens": False,
        }

    def test_build_task_params_keeps_other_vlms_unchanged(self):
        request = lmm_request("describe", max_new_tokens=64)
        route = {"model_class_name": "Qwen25VLHF", "model_mro_names": None}
        params = translation.build_task_params("vlm", "prompt", request, route)
        assert params == {"prompt": "describe", "max_new_tokens": 64}


class TestFlorence2RequestValidation:
    def test_disable_preproc_flags_ignored_like_legacy(self):
        request = lmm_request(
            "<CAPTION>",
            disable_preproc_auto_orient=True,
            disable_preproc_contrast=True,
            disable_preproc_grayscale=True,
            disable_preproc_static_crop=True,
        )
        translation.ensure_request_supported(
            "florence-2-base/1", request, route=FLORENCE_ROUTE
        )

    def test_disable_preproc_flags_still_refused_without_route(self):
        request = lmm_request("<CAPTION>", disable_preproc_contrast=True)
        with pytest.raises(ModelDeploymentNotSupportedError):
            translation.ensure_request_supported("florence-2-base/1", request)

    def test_disable_preproc_flags_still_refused_for_other_vlms(self):
        request = lmm_request("describe", disable_preproc_contrast=True)
        route = {"model_class_name": "Qwen25VLHF", "model_mro_names": None}
        with pytest.raises(ModelDeploymentNotSupportedError):
            translation.ensure_request_supported("qwen/1", request, route=route)

    def test_image_list_rejected_like_legacy(self):
        request = lmm_request(
            "<CAPTION>",
            image=[
                SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode()),
                SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode()),
            ],
        )
        with pytest.raises(InvalidImageTypeDeclared):
            translation.ensure_request_supported(
                "florence-2-base/1", request, route=FLORENCE_ROUTE
            )


class TestFlorence2TaskDerivation:
    @pytest.mark.parametrize(
        "prompt, task",
        [
            ("<CAPTION>", "<CAPTION>"),
            (
                "<CAPTION_TO_PHRASE_GROUNDING>A green car",
                "<CAPTION_TO_PHRASE_GROUNDING>",
            ),
            (
                "<REGION_TO_SEGMENTATION><loc_50><loc_60><loc_150><loc_160>",
                "<REGION_TO_SEGMENTATION>",
            ),
            ("<CUSTOM_TASK>", "<CUSTOM_TASK>"),
            ("hello", "hello>"),
            ("", ">"),
        ],
    )
    def test_task_token_matches_legacy_prompt_split(self, prompt, task):
        assert mmp_florence2.derive_task_token(prompt) == task


FLORENCE_RESPONSE_MATRIX = [
    (
        "<CAPTION>",
        "</s><s>A green car parked in front of a yellow building.</s>",
        {"<CAPTION>": "A green car parked in front of a yellow building."},
    ),
    (
        "<DETAILED_CAPTION>",
        "</s><s>The image shows a green car.</s>",
        {"<DETAILED_CAPTION>": "The image shows a green car."},
    ),
    (
        "<MORE_DETAILED_CAPTION>",
        "</s><s>A very long caption.  </s>",
        {"<MORE_DETAILED_CAPTION>": "A very long caption."},
    ),
    (
        "<OCR>",
        "</s><s>STOP</s>",
        {"<OCR>": "STOP"},
    ),
    (
        "<OD>",
        "</s><s>car<loc_54><loc_375><loc_906><loc_707>door<loc_710><loc_276>"
        "<loc_908><loc_537>wheel<loc_708><loc_531><loc_906><loc_704></s>",
        {
            "<OD>": {
                "bboxes": [
                    [34, 180, 580, 339],
                    [454, 132, 581, 258],
                    [453, 255, 580, 338],
                ],
                "labels": ["car", "door", "wheel"],
            }
        },
    ),
    (
        "<DENSE_REGION_CAPTION>",
        "</s><s>a green car<loc_54><loc_375><loc_906><loc_707>a dark door"
        "<loc_710><loc_276><loc_908><loc_537></s>",
        {
            "<DENSE_REGION_CAPTION>": {
                "bboxes": [[34, 180, 580, 339], [454, 132, 581, 258]],
                "labels": ["a green car", "a dark door"],
            }
        },
    ),
    (
        "<REGION_PROPOSAL>",
        "</s><s><loc_54><loc_375><loc_906><loc_707><loc_710><loc_276><loc_908>"
        "<loc_537></s>",
        {
            "<REGION_PROPOSAL>": {
                "bboxes": [[34, 180, 580, 339], [454, 132, 581, 258]],
                "labels": ["", ""],
            }
        },
    ),
    (
        "<CAPTION_TO_PHRASE_GROUNDING>A green car parked next to trees",
        "</s><s>A green car<loc_54><loc_375><loc_906><loc_707> parked next to trees"
        "<loc_10><loc_20><loc_200><loc_300><loc_400><loc_100><loc_600><loc_500></s>",
        {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [
                    [34, 180, 580, 339],
                    [6, 9, 128, 144],
                    [256, 48, 384, 240],
                ],
                "labels": [
                    "A green car",
                    "parked next to trees",
                    "parked next to trees",
                ],
            }
        },
    ),
    (
        "<CAPTION_TO_PHRASE_GROUNDING>image with a dog",
        "</s><s>image<loc_0><loc_0><loc_999><loc_999>a dog<loc_100><loc_200>"
        "<loc_300><loc_400></s>",
        {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[64, 96, 192, 192]],
                "labels": ["a dog"],
            }
        },
    ),
    (
        "<OCR_WITH_REGION>",
        "</s><s>STOP<loc_100><loc_100><loc_300><loc_100><loc_300><loc_200><loc_100>"
        "<loc_200>AHEAD<loc_100><loc_300><loc_300><loc_300><loc_300><loc_400>"
        "<loc_100><loc_400></s>",
        {
            "<OCR_WITH_REGION>": {
                "quad_boxes": [
                    [64, 48, 192, 48, 192, 96, 64, 96],
                    [64, 144, 192, 144, 192, 192, 64, 192],
                ],
                "labels": ["STOP", "AHEAD"],
            }
        },
    ),
    (
        "<REFERRING_EXPRESSION_SEGMENTATION>a green car",
        "</s><s><loc_100><loc_100><loc_200><loc_100><loc_200><loc_200><loc_100>"
        "<loc_200></s>",
        {
            "<REFERRING_EXPRESSION_SEGMENTATION>": {
                "polygons": [[[64, 48, 128, 48, 128, 96, 64, 96]]],
                "labels": [""],
            }
        },
    ),
    (
        "<REGION_TO_SEGMENTATION><loc_50><loc_60><loc_150><loc_160>",
        "</s><s><loc_50><loc_60><loc_150><loc_60><loc_150><loc_160><loc_50>"
        "<loc_160><loc_50><loc_61></s>",
        {
            "<REGION_TO_SEGMENTATION>": {
                "polygons": [[[32, 29, 96, 29, 96, 77, 32, 77, 32, 29]]],
                "labels": [""],
            }
        },
    ),
    (
        "<OPEN_VOCABULARY_DETECTION>a green car",
        "</s><s>a green car<loc_54><loc_375><loc_906><loc_707></s>",
        {
            "<OPEN_VOCABULARY_DETECTION>": {
                "bboxes": [[34, 180, 580, 339]],
                "bboxes_labels": ["a green car"],
                "polygons": [],
                "polygons_labels": [],
            }
        },
    ),
    (
        "<OPEN_VOCABULARY_DETECTION>a green car",
        "</s><s>a green car<poly><loc_100><loc_100><loc_200><loc_100><loc_200>"
        "<loc_200></poly></s>",
        {
            "<OPEN_VOCABULARY_DETECTION>": {
                "bboxes": [],
                "bboxes_labels": [],
                "polygons": [[[64, 48, 128, 48, 128, 96]]],
                "polygons_labels": ["a green car"],
            }
        },
    ),
    (
        "<REGION_TO_CATEGORY><loc_154><loc_258><loc_903><loc_621>",
        "</s><s>car<loc_154><loc_258><loc_903><loc_621></s>",
        {"<REGION_TO_CATEGORY>": "car<loc_154><loc_258><loc_903><loc_621>"},
    ),
    (
        "<REGION_TO_DESCRIPTION><loc_154><loc_258><loc_903><loc_621>",
        "</s><s>a shiny green car<loc_154><loc_258><loc_903><loc_621></s>",
        {
            "<REGION_TO_DESCRIPTION>": (
                "a shiny green car<loc_154><loc_258><loc_903><loc_621>"
            )
        },
    ),
    (
        "<REGION_TO_OCR><loc_154><loc_258><loc_903><loc_621>",
        "</s><s>STOP<loc_154><loc_258><loc_903><loc_621></s>",
        {"<REGION_TO_OCR>": "STOP<loc_154><loc_258><loc_903><loc_621>"},
    ),
    (
        "<CUSTOM_TASK>",
        "</s><s>whatever the model says</s>",
        {"<CUSTOM_TASK>": "whatever the model says"},
    ),
    (
        "hello",
        "</s><s>free text answer</s>",
        {"hello>": "free text answer"},
    ),
    (
        "",
        "</s><s>empty prompt answer</s>",
        {">": "empty prompt answer"},
    ),
]


class TestFlorence2ResponseReassembly:
    @pytest.mark.parametrize("prompt, decoded, expected", FLORENCE_RESPONSE_MATRIX)
    def test_wire_text_repacked_into_legacy_task_dict(self, prompt, decoded, expected):
        request = lmm_request(prompt)
        response = mmp_florence2.repack_response(decoded, request, (640, 480))
        assert response.response == expected
        assert response.image.width == 640
        assert response.image.height == 480

    def test_single_element_list_prediction_is_unwrapped(self):
        request = lmm_request("<CAPTION>")
        response = mmp_florence2.repack_response(
            ["</s><s>A dog.</s>"], request, (640, 480)
        )
        assert response.response == {"<CAPTION>": "A dog."}

    def test_non_text_prediction_is_rejected(self):
        request = lmm_request("<CAPTION>")
        with pytest.raises(ModelArtefactError):
            mmp_florence2.repack_response({"unexpected": True}, request, (640, 480))

    def test_repack_prediction_routes_florence_vlm(self):
        request = lmm_request("<CAPTION>")
        route = {"model_class_name": "Florence2HF", "model_mro_names": None}
        response = translation.repack_prediction(
            "vlm",
            "prompt",
            "</s><s>A dog.</s>",
            (640, 480),
            route,
            request,
        )
        assert response.response == {"<CAPTION>": "A dog."}

    def test_repack_prediction_keeps_other_vlms_raw(self):
        request = lmm_request("<CAPTION>")
        route = {"model_class_name": "Qwen25VLHF", "model_mro_names": None}
        response = translation.repack_prediction(
            "vlm",
            "prompt",
            "</s><s>A dog.</s>",
            (640, 480),
            route,
            request,
        )
        assert response.response == "</s><s>A dog.</s>"


class TestFlorence2AdapterFlow:
    def test_caption_request_end_to_end(self, running_adapter, florence_stat):
        request = lmm_request("<CAPTION>")
        response = running_adapter.infer_from_request_sync("florence-2-base/1", request)
        assert response.response == {
            "<CAPTION>": "A green car parked in front of a yellow building."
        }
        assert response.image.width == 640
        assert response.image.height == 480
        call = running_adapter._client.infer_calls[0]
        assert call["task"] == "prompt"
        assert call["params"] == {
            "prompt": "<CAPTION>",
            "max_new_tokens": 1000,
            "do_sample": False,
            "skip_special_tokens": False,
        }

    def test_grounding_request_end_to_end(self, running_adapter, florence_stat):
        running_adapter._client.infer_result = (
            "</s><s>A green car<loc_54><loc_375><loc_906><loc_707></s>"
        )
        request = lmm_request("<CAPTION_TO_PHRASE_GROUNDING>A green car")
        response = running_adapter.infer_from_request_sync("florence-2-base/1", request)
        assert response.response == {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[34, 180, 580, 339]],
                "labels": ["A green car"],
            }
        }
        call = running_adapter._client.infer_calls[0]
        assert call["params"]["prompt"] == "<CAPTION_TO_PHRASE_GROUNDING>A green car"

    def test_disable_preproc_request_end_to_end(self, running_adapter, florence_stat):
        request = lmm_request("<CAPTION>", disable_preproc_auto_orient=True)
        response = running_adapter.infer_from_request_sync("florence-2-base/1", request)
        assert response.response == {
            "<CAPTION>": "A green car parked in front of a yellow building."
        }

    def test_image_list_rejected_before_wire(self, running_adapter, florence_stat):
        request = lmm_request(
            "<CAPTION>",
            image=[
                SimpleNamespace(type="base64", value=base64.b64encode(b"f").decode()),
            ],
        )
        with pytest.raises(InvalidImageTypeDeclared):
            running_adapter.infer_from_request_sync("florence-2-base/1", request)
        assert running_adapter._client.infer_calls == []

    def test_florence_route_resolves_without_refusal(
        self, running_adapter, florence_stat
    ):
        running_adapter.add_model("florence-2-base/1", api_key="key")
        assert "florence-2-base/1" in running_adapter
        assert running_adapter._client.unloaded == []

    def test_unsupported_vlm_class_still_refused(
        self, running_adapter, florence_stat, monkeypatch
    ):
        monkeypatch.setattr(
            translation,
            "VLM_UNSUPPORTED_MODEL_CLASSES",
            frozenset(["UnsupportedVLMHF"]),
        )
        running_adapter._client.model_class_name = "UnsupportedVLMHF"
        running_adapter._client.model_mro_names = ["UnsupportedVLMHF", "object"]
        with pytest.raises(ModelDeploymentNotSupportedError):
            running_adapter.add_model("unsupported-vlm/1", api_key="key")
        assert running_adapter._client.unloaded == ["unsupported-vlm/1"]


class TestFlorence2FineTuneAdapterFlow:
    def test_caption_request_matches_pretrain_behavior(
        self, running_adapter, florence_stat
    ):
        request = lmm_request("<CAPTION>")
        response = running_adapter.infer_from_request_sync(
            "qwen_playground/80", request
        )
        assert response.response == {
            "<CAPTION>": "A green car parked in front of a yellow building."
        }
        assert running_adapter._routes["qwen_playground/80"] == {
            "supported": True,
            "mmp_model_id": "qwen_playground/80",
            "task_type": "vlm",
            "action": "prompt",
            "tasks": set(FLORENCE_TASKS),
            "class_names": None,
            "key_points_classes": None,
            "model_class_name": "Florence2HF",
            "model_mro_names": ["Florence2HF", "object"],
        }
        call = running_adapter._client.infer_calls[0]
        assert call["params"] == {
            "prompt": "<CAPTION>",
            "max_new_tokens": 1000,
            "do_sample": False,
            "skip_special_tokens": False,
        }

    def test_od_request_returns_populated_detections(
        self, running_adapter, florence_stat
    ):
        running_adapter._client.infer_result = (
            "</s><s>car<loc_54><loc_375><loc_906><loc_707>door<loc_710><loc_276>"
            "<loc_908><loc_537>wheel<loc_708><loc_531><loc_906><loc_704></s>"
        )
        request = lmm_request("<OD>")
        response = running_adapter.infer_from_request_sync(
            "qwen_playground/80", request
        )
        assert response.response == {
            "<OD>": {
                "bboxes": [
                    [34, 180, 580, 339],
                    [454, 132, 581, 258],
                    [453, 255, 580, 338],
                ],
                "labels": ["car", "door", "wheel"],
            }
        }

    def test_grounding_request_returns_populated_detections(
        self, running_adapter, florence_stat
    ):
        running_adapter._client.infer_result = (
            "</s><s>A green car<loc_54><loc_375><loc_906><loc_707></s>"
        )
        request = lmm_request("<CAPTION_TO_PHRASE_GROUNDING>A green car")
        response = running_adapter.infer_from_request_sync(
            "qwen_playground/80", request
        )
        assert response.response == {
            "<CAPTION_TO_PHRASE_GROUNDING>": {
                "bboxes": [[34, 180, 580, 339]],
                "labels": ["A green car"],
            }
        }
