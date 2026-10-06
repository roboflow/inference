from types import SimpleNamespace

import numpy as np
import pytest

from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    ClipCompareRequest,
    ClipImageEmbeddingRequest,
    ClipTextEmbeddingRequest,
    EasyOCRInferenceRequest,
    GroundingDINOInferenceRequest,
    LMMInferenceRequest,
    OwlV2InferenceRequest,
    YOLOWorldInferenceRequest,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.translation import (
    build_embedding_calls,
    build_few_shot_params,
    build_open_vocabulary_params,
    build_vlm_params,
    encode_normalized_depth_to_png8,
    encode_normalized_depth_to_png16,
    ensure_ocr_request_supported,
    few_shot_class_names,
    repack_depth_estimation,
    repack_embedding_response,
    repack_moondream_detection,
    repack_structured_ocr_response,
    repack_text_ocr_response,
    repack_vlm_response,
    resolve_request_action,
)

IMG = {"type": "base64", "value": "x"}


def _route(**kwargs) -> Route:
    defaults = dict(model_id="m/1", registry_id="m/1", task_type="vlm", action="prompt")
    defaults.update(kwargs)
    return Route(**defaults)


def test_resolve_request_action_prefers_moondream_detect():
    route = _route(actions={"detect"}, model_class_name="MoonDream2HF")
    assert (
        resolve_request_action(route, LMMInferenceRequest(model_id="m/1", image=IMG))
        == "detect"
    )


def test_resolve_request_action_uses_first_candidate_present_in_actions():
    route = _route(task_type="embedding", action="embed_images", actions={"embed_text"})
    request = ClipTextEmbeddingRequest(text="hello")
    assert resolve_request_action(route, request) == "embed_text"


def test_resolve_request_action_falls_back_to_route_action():
    route = _route(task_type="structured-ocr", action="infer")
    assert resolve_request_action(route, SimpleNamespace()) == "infer"


def test_resolve_request_action_falls_back_when_no_candidate_is_registered():
    route = _route(task_type="embedding", action="embed_images", actions={"compare"})
    request = ClipTextEmbeddingRequest(text="hello")
    assert resolve_request_action(route, request) == "embed_images"


def test_resolve_request_action_keeps_compare_when_model_only_embeds():
    route = _route(
        task_type="embedding",
        action="embed_images",
        actions={"embed_images", "embed_text"},
    )
    request = ClipCompareRequest(
        subject="a dog",
        subject_type="text",
        prompt=["cat"],
        prompt_type="text",
    )
    assert resolve_request_action(route, request) == "compare"


def test_resolve_request_action_ignores_moondream_mro_only_routes():
    route = _route(actions={"detect"}, model_mro_names=["MoonDream2HF"])
    request = LMMInferenceRequest(model_id="m/1", image=IMG)
    assert resolve_request_action(route, request) == "prompt"


def test_embedding_calls_for_compare_put_subject_first():
    req = ClipCompareRequest(
        subject="a dog",
        subject_type="text",
        prompt={"x": "cat", "y": "dog"},
        prompt_type="text",
    )
    calls, keys = build_embedding_calls("compare", req)
    assert [c["action"] for c in calls] == ["embed_text", "embed_text"]
    assert keys == ["x", "y"]
    assert calls[0]["params"] == {"texts": ["a dog"]}
    assert calls[1]["params"] == {"texts": ["cat", "dog"]}


def test_embedding_calls_for_images_are_one_per_image():
    req = ClipImageEmbeddingRequest(image=[IMG, IMG])
    calls, keys = build_embedding_calls("embed_images", req)
    assert keys is None
    assert [c["action"] for c in calls] == ["embed_images", "embed_images"]
    assert calls[0]["image"] is not None and calls[0]["params"] == {}


def test_embedding_calls_enforce_max_batch_size(monkeypatch):
    monkeypatch.setattr("inference_server.configuration.CLIP_MAX_BATCH_SIZE", 1)
    req = ClipImageEmbeddingRequest(image=[IMG, IMG])
    with pytest.raises(LegacyHTTPError) as caught:
        build_embedding_calls("embed_images", req)
    assert caught.value.status_code == 400


def test_repack_embeddings_stacks_results():
    req = ClipTextEmbeddingRequest(text=["a", "b"])
    resp = repack_embedding_response(
        "embed_text", req, [np.array([[1.0, 2.0], [3.0, 4.0]])], None
    )
    assert resp.embeddings == [[1.0, 2.0], [3.0, 4.0]]


def test_repack_compare_returns_named_similarities():
    req = ClipCompareRequest(
        subject="a",
        subject_type="text",
        prompt={"x": "b", "y": "c"},
        prompt_type="text",
    )
    resp = repack_embedding_response(
        "compare",
        req,
        [np.array([[1.0, 0.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])],
        ["x", "y"],
    )
    assert resp.similarity["x"] == 1.0
    assert abs(resp.similarity["y"]) < 1e-9


def test_open_vocabulary_params():
    req = YOLOWorldInferenceRequest(image=IMG, text=["cat", "dog"], confidence=0.2)
    assert build_open_vocabulary_params(req) == {
        "classes": ["cat", "dog"],
        "confidence": 0.2,
    }
    dino = GroundingDINOInferenceRequest(image=IMG, text=["cat"])
    assert build_open_vocabulary_params(dino) == {
        "classes": ["cat"],
        "box_confidence": 0.5,
        "text_confidence": 0.5,
        "class_agnostic_nms": False,
    }


def test_grounding_dino_thresholds_map_to_model_confidences():
    req = GroundingDINOInferenceRequest(
        image=IMG, text=["cat"], box_threshold=0.1, text_threshold=0.2
    )
    params = build_open_vocabulary_params(req)
    assert params["box_confidence"] == 0.1
    assert params["text_confidence"] == 0.2
    assert "box_threshold" not in params and "text_threshold" not in params


def test_vlm_params_carry_prompt_and_generation_options():
    assert build_vlm_params(
        LMMInferenceRequest(model_id="m/1", image=IMG, prompt="hi", max_new_tokens=5)
    ) == {"prompt": "hi", "max_new_tokens": 5}


@pytest.mark.parametrize("model_class_name", ["Florence2HF", "PaliGemmaHF"])
def test_vlm_params_require_prompt_for_families_without_a_default(model_class_name):
    with pytest.raises(LegacyHTTPError) as error:
        build_vlm_params(
            LMMInferenceRequest(model_id="m/1", image=IMG),
            model_class_name=model_class_name,
        )
    assert error.value.status_code == 400


@pytest.mark.parametrize(
    "model_class_name",
    [
        "Qwen25VLHF",
        "Qwen3VLHF",
        "Qwen35HF",
        "Cosmos3EdgeReasoner",
        "Gemma4HF",
        "SmolVLMHF",
        None,
    ],
)
def test_vlm_params_let_families_with_a_default_prompt_omit_it(model_class_name):
    params = build_vlm_params(
        LMMInferenceRequest(model_id="m/1", image=IMG),
        model_class_name=model_class_name,
    )
    assert params == {"prompt": None}


@pytest.mark.parametrize(
    "prompt,task",
    [
        ("<CAPTION>", "<CAPTION>"),
        ("<CAPTION_TO_PHRASE_GROUNDING>a cat", "<CAPTION_TO_PHRASE_GROUNDING>"),
        ("<OD>", "<OD>"),
        ("describe", "describe>"),
    ],
)
def test_florence2_task_is_derived_from_the_prompt(prompt, task):
    params = build_vlm_params(
        LMMInferenceRequest(model_id="m/1", image=IMG, prompt=prompt),
        model_class_name="Florence2HF",
    )
    assert params == {"prompt": prompt, "task": task}


def test_task_is_not_derived_for_other_vlm_families():
    params = build_vlm_params(
        LMMInferenceRequest(model_id="m/1", image=IMG, prompt="<OD>"),
        model_class_name="Qwen25VLHF",
    )
    assert params == {"prompt": "<OD>"}


def test_vlm_repack_string_and_dict():
    assert repack_vlm_response(["hello"], (2, 2)).response == "hello"
    assert repack_vlm_response({"a": 1}, (2, 2)).response == {"a": 1}


def test_moondream_detection_repack_uses_prompt_as_class():
    prediction = SimpleNamespace(xyxy=np.array([[0.0, 0.0, 2.0, 4.0]]))
    request = LMMInferenceRequest(model_id="m/1", image=IMG, prompt="dog")
    resp = repack_moondream_detection(prediction, request, (8, 6))
    assert resp.predictions[0].class_name == "dog"
    assert resp.predictions[0].width == 2.0
    assert resp.image.width == 8


def test_depth_repack_normalises():
    out = repack_depth_estimation(np.array([[0.0, 2.0], [4.0, 8.0]], dtype=np.float32))
    assert out["normalized_depth"].max() == 1.0
    assert out["image"]["base64_image"]


def test_depth_repack_rejects_flat_map():
    with pytest.raises(LegacyHTTPError) as error:
        repack_depth_estimation(np.ones((2, 2), dtype=np.float32))
    assert error.value.status_code == 500


def test_depth_png_encoders_return_base64_strings():
    normalized = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    png8 = encode_normalized_depth_to_png8(normalized)
    png16 = encode_normalized_depth_to_png16(normalized)
    assert isinstance(png8, str) and isinstance(png16, str)
    assert png8 != png16


def test_structured_ocr_with_boxes():
    det = SimpleNamespace(
        xyxy=np.array([[0, 0, 2, 2]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )
    req = SimpleNamespace(generate_bounding_boxes=True, class_filter=None)
    resp = repack_structured_ocr_response((["hi"], [det]), (4, 4), ["text"], req)
    assert resp.result == "hi"
    assert resp.predictions[0].class_name == "text"
    assert resp.image.width == 4


def test_structured_ocr_without_boxes_has_no_image():
    det = SimpleNamespace(
        xyxy=np.zeros((0, 4)), confidence=np.zeros(0), class_id=np.zeros(0)
    )
    req = SimpleNamespace(generate_bounding_boxes=False, class_filter=None)
    resp = repack_structured_ocr_response((["hi"], [det]), (4, 4), None, req)
    assert resp.result == "hi"
    assert resp.image is None and resp.predictions is None


def test_structured_ocr_boxes_can_be_forced_without_request_field():
    det = SimpleNamespace(
        xyxy=np.array([[0, 0, 2, 2]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
        bboxes_metadata=[{"polygon": [], "text": "hi"}],
    )
    req = SimpleNamespace(class_filter=None)
    resp = repack_structured_ocr_response(
        (["hi"], [det]),
        (4, 4),
        None,
        req,
        generate_bounding_boxes=True,
        class_from_text=True,
    )
    assert resp.predictions[0].class_name == "hi"
    assert resp.predictions[0].class_id == 0
    assert resp.image.width == 4 and resp.image.height == 4


def test_structured_ocr_box_text_replaces_class_only_when_requested():
    det = SimpleNamespace(
        xyxy=np.array([[0, 0, 2, 2], [0, 3, 2, 5]], dtype=float),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
        bboxes_metadata=[{"text": "first"}, {"text": ""}],
    )
    req = SimpleNamespace(generate_bounding_boxes=True, class_filter=None)
    resp = repack_structured_ocr_response(
        (["first"], [det]), (4, 8), ["text"], req, class_from_text=True
    )
    assert [p.class_name for p in resp.predictions] == ["first", ""]
    resp = repack_structured_ocr_response((["first"], [det]), (4, 8), ["text"], req)
    assert [p.class_name for p in resp.predictions] == ["text", "text"]


def test_doctr_boxes_keep_block_line_word_classes_despite_text_metadata():
    det = SimpleNamespace(
        xyxy=np.array([[0, 0, 4, 4], [0, 0, 4, 2], [0, 0, 2, 2]], dtype=float),
        confidence=np.array([0.9, 0.9, 0.9]),
        class_id=np.array([0, 1, 2]),
        bboxes_metadata=[{"text": "hi there"}, {"text": "hi there"}, {"text": "hi"}],
    )
    req = SimpleNamespace(generate_bounding_boxes=True, class_filter=None)
    resp = repack_structured_ocr_response(
        (["hi there"], [det]), (4, 4), ["block", "line", "word"], req
    )
    assert [p.class_name for p in resp.predictions] == ["block", "line", "word"]


def test_text_ocr():
    assert repack_text_ocr_response(["abc"], (1, 1)).result == "abc"


def test_pp_ocr_empty_text_box_keeps_empty_class_and_zero_class_id():
    det = SimpleNamespace(
        xyxy=np.array([[0, 0, 2, 2], [0, 3, 2, 5]], dtype=float),
        confidence=np.array([0.9, 0.8]),
        class_id=np.array([0, 0]),
        bboxes_metadata=[{"text": "word"}, {"text": ""}],
    )
    req = SimpleNamespace(class_filter=None)
    resp = repack_structured_ocr_response(
        ([""], [det]),
        (4, 8),
        None,
        req,
        generate_bounding_boxes=True,
        class_from_text=True,
    )
    assert [(p.class_name, p.class_id) for p in resp.predictions] == [
        ("word", 0),
        ("", 0),
    ]
    assert resp.image.width == 4 and resp.image.height == 8


def test_easy_ocr_options_are_accepted_and_ignored(caplog):
    request = EasyOCRInferenceRequest(image=IMG, language_codes=["pl"], quantize=True)
    with caplog.at_level("DEBUG", logger="inference_server.legacy.translation"):
        ensure_ocr_request_supported(request)
    assert len(caplog.records) == 1
    assert "language_codes" in caplog.text and "quantize" in caplog.text


def test_easy_ocr_default_options_log_nothing(caplog):
    with caplog.at_level("DEBUG", logger="inference_server.legacy.translation"):
        ensure_ocr_request_supported(EasyOCRInferenceRequest(image=IMG))
    assert caplog.records == []


def _owlv2_request(**kwargs):
    return OwlV2InferenceRequest(
        image=IMG,
        training_data=[
            {
                "image": {"type": "base64", "value": "ref"},
                "boxes": [
                    {"x": 10, "y": 20, "w": 30, "h": 40, "cls": "dog"},
                    {"x": 1, "y": 2, "w": 3, "h": 4, "cls": "cat", "negative": True},
                ],
            }
        ],
        **kwargs,
    )


def test_few_shot_params_map_training_data_to_reference_examples():
    assert build_few_shot_params(_owlv2_request(confidence=0.9), [b"ref"]) == {
        "reference_examples": [
            {
                "image": b"ref",
                "boxes": [
                    {
                        "x": 10,
                        "y": 20,
                        "w": 30,
                        "h": 40,
                        "cls": "dog",
                        "negative": False,
                    },
                    {"x": 1, "y": 2, "w": 3, "h": 4, "cls": "cat", "negative": True},
                ],
            }
        ],
        "confidence": 0.9,
    }


def test_few_shot_class_names_prefer_the_model_mapping():
    detections = SimpleNamespace(image_metadata={"class_names": ["cat", "dog"]})
    assert few_shot_class_names(detections, _owlv2_request()) == ["cat", "dog"]
    assert few_shot_class_names(SimpleNamespace(), _owlv2_request()) == ["cat", "dog"]
