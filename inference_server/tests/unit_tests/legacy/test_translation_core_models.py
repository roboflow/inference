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
    YOLOWorldInferenceRequest,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.translation import (
    build_embedding_calls,
    build_open_vocabulary_params,
    build_vlm_params,
    encode_normalized_depth_to_png8,
    encode_normalized_depth_to_png16,
    ensure_ocr_request_supported,
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
    route = _route(tasks={"detect"}, model_class_name="MoonDream2HF")
    assert (
        resolve_request_action(route, LMMInferenceRequest(model_id="m/1", image=IMG))
        == "detect"
    )


def test_resolve_request_action_uses_first_candidate_present_in_tasks():
    route = _route(task_type="embedding", action="embed_images", tasks={"embed_text"})
    request = ClipTextEmbeddingRequest(text="hello")
    assert resolve_request_action(route, request) == "embed_text"


def test_resolve_request_action_falls_back_to_route_action():
    route = _route(task_type="structured-ocr", action="infer")
    assert resolve_request_action(route, SimpleNamespace()) == "infer"


def test_resolve_request_action_falls_back_when_no_candidate_is_registered():
    route = _route(task_type="embedding", action="embed_images", tasks={"compare"})
    request = ClipTextEmbeddingRequest(text="hello")
    assert resolve_request_action(route, request) == "embed_images"


def test_resolve_request_action_keeps_compare_when_model_only_embeds():
    route = _route(
        task_type="embedding",
        action="embed_images",
        tasks={"embed_images", "embed_text"},
    )
    request = ClipCompareRequest(
        subject="a dog",
        subject_type="text",
        prompt=["cat"],
        prompt_type="text",
    )
    assert resolve_request_action(route, request) == "compare"


def test_resolve_request_action_ignores_moondream_mro_only_routes():
    route = _route(tasks={"detect"}, model_mro_names=["MoonDream2HF"])
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
    assert [c["task"] for c in calls] == ["embed_text", "embed_text"]
    assert keys == ["x", "y"]
    assert calls[0]["params"] == {"texts": ["a dog"]}
    assert calls[1]["params"] == {"texts": ["cat", "dog"]}


def test_embedding_calls_for_images_are_one_per_image():
    req = ClipImageEmbeddingRequest(image=[IMG, IMG])
    calls, keys = build_embedding_calls("embed_images", req)
    assert keys is None
    assert [c["task"] for c in calls] == ["embed_images", "embed_images"]
    assert calls[0]["image"] is not None and calls[0]["params"] == {}


def test_embedding_calls_enforce_max_batch_size(monkeypatch):
    monkeypatch.setattr("inference_server.configuration.CLIP_MAX_BATCH_SIZE", 1)
    req = ClipImageEmbeddingRequest(image=[IMG, IMG])
    with pytest.raises(ValueError):
        build_embedding_calls("embed_images", req)


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
        "class_agnostic_nms": False,
    }


def test_open_vocabulary_params_reject_custom_thresholds():
    req = GroundingDINOInferenceRequest(image=IMG, text=["cat"], box_threshold=0.1)
    with pytest.raises(LegacyHTTPError) as error:
        build_open_vocabulary_params(req)
    assert error.value.status_code == 501


def test_vlm_params_require_prompt():
    assert build_vlm_params(
        LMMInferenceRequest(model_id="m/1", image=IMG, prompt="hi", max_new_tokens=5)
    ) == {"prompt": "hi", "max_new_tokens": 5}
    with pytest.raises(LegacyHTTPError) as error:
        build_vlm_params(LMMInferenceRequest(model_id="m/1", image=IMG))
    assert error.value.status_code == 501


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


def test_ensure_ocr_request_supported():
    ensure_ocr_request_supported(EasyOCRInferenceRequest(image=IMG))
    with pytest.raises(LegacyHTTPError) as error:
        ensure_ocr_request_supported(
            EasyOCRInferenceRequest(image=IMG, language_codes=["pl"])
        )
    assert error.value.status_code == 501
    with pytest.raises(LegacyHTTPError) as quantize_error:
        ensure_ocr_request_supported(EasyOCRInferenceRequest(image=IMG, quantize=True))
    assert quantize_error.value.status_code == 501
