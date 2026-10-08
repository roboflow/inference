import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from PIL import Image

from tests.unit_tests.legacy.conftest import FakeGateway, route_paths


def _jpeg_b64(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _image():
    return {"type": "base64", "value": _jpeg_b64()}


def _det(class_id=0):
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([class_id]),
    )


def test_clip_embed_text_is_params_only_call(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={("clip/ViT-B-16", "embed_text"): np.array([[1.0, 2.0]])},
        model_info={
            "clip/ViT-B-16": {"actions": {"embed_text": {}, "embed_images": {}}}
        },
    )
    r = legacy_client(gw).post(
        "/clip/embed_text", json={"text": "hello", "api_key": "k"}
    )
    assert r.status_code == 200, r.text
    assert r.json()["embeddings"] == [[1.0, 2.0]]
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[2] == "embed_text"
    assert call[3] == {"texts": ["hello"]}
    assert call[4] is None


def test_clip_embed_image_sends_decoded_image(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={("clip/ViT-B-16", "embed_images"): np.array([[0.5, 0.5]])},
        model_info={"clip/ViT-B-16": {"actions": {"embed_images": {}}}},
    )
    r = legacy_client(gw).post("/clip/embed_image", json={"image": _image()})
    assert r.status_code == 200, r.text
    assert r.json()["embeddings"] == [[0.5, 0.5]]
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[2] == "embed_images" and isinstance(call[4], bytes)


def test_clip_compare_returns_named_similarities(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("clip/ViT-B-16", "embed_text"): lambda image, params: np.array(
                [[1.0, 0.0]] * len(params["texts"])
            )
        },
        model_info={"clip/ViT-B-16": {"actions": {"embed_text": {}, "compare": {}}}},
    )
    r = legacy_client(gw).post(
        "/clip/compare",
        json={
            "subject": "a",
            "subject_type": "text",
            "prompt": {"x": "b", "y": "c"},
            "prompt_type": "text",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["similarity"] == {"x": 1.0, "y": 1.0}
    assert body["parent_id"] is None
    assert body["inference_id"] is None and body["frame_id"] is None


def test_perception_encoder_uses_registry_alias(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("perception-encoder/PE-Core-L14-336", "embed_text"): np.array([[1.0]])
        },
        model_info={
            "perception-encoder/PE-Core-L14-336": {"actions": {"embed_text": {}}}
        },
    )
    r = legacy_client(gw).post("/perception_encoder/embed_text", json={"text": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["embeddings"] == [[1.0]]


def test_doctr_keeps_parent_id_null(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("doctr/default", "infer"): (
                ["hi"],
                [
                    SimpleNamespace(
                        xyxy=np.zeros((0, 4)),
                        confidence=np.zeros(0),
                        class_id=np.zeros(0),
                    )
                ],
            )
        },
        model_info={"doctr/default": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/doctr/ocr", json={"image": _image(), "api_key": "k"})
    assert r.status_code == 200, r.text
    assert r.json()["result"] == "hi"
    assert r.json()["parent_id"] is None


def test_easy_ocr_accepts_language_codes_and_quantize(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={("easy_ocr/english_g2", "infer"): (["a"], [_ocr_det(["a"])])},
        model_info={"easy_ocr/english_g2": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/easy_ocr/ocr",
        json={"image": _image(), "language_codes": ["pl"], "quantize": True},
    )
    assert r.status_code == 200, r.text
    assert r.json()["result"] == "a"
    assert _infer_params(gw) == {"confidence": 0.0}


def test_easy_ocr_returns_image_size_and_boxes_with_text_as_class(
    legacy_client, fake_stat
):
    gw = FakeGateway(
        predictions={
            ("easy_ocr/english_g2", "infer"): (
                ["hello world"],
                [_ocr_det(["hello", "world"])],
            )
        },
        model_info={"easy_ocr/english_g2": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/easy_ocr/ocr", json={"image": _image()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "hello world"
    assert body["image"] == {"width": 8, "height": 6}
    assert [p["class"] for p in body["predictions"]] == ["hello", "world"]
    first = body["predictions"][0]
    assert first["class_id"] == 0
    assert (first["x"], first["y"], first["width"], first["height"]) == (1, 1, 2, 2)
    assert first["confidence"] == 0.9


def _infer_params(gateway):
    return next(call for call in gateway.calls if call[0] == "infer")[3]


def test_easy_ocr_passes_zero_confidence_to_keep_low_confidence_text(
    legacy_client, fake_stat
):
    gw = FakeGateway(
        predictions={("easy_ocr/english_g2", "infer"): (["a"], [_ocr_det(["a"])])},
        model_info={"easy_ocr/english_g2": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/easy_ocr/ocr", json={"image": _image()})
    assert r.status_code == 200, r.text
    assert _infer_params(gw) == {"confidence": 0.0}


def test_other_ocr_routes_pass_no_params(legacy_client, fake_stat):
    fake_stat["pp-ocrv6-det/small"] = ("object-detection", "infer")
    fake_stat["pp-ocrv6-rec/small"] = ("text-only-ocr", "infer")
    gw = FakeGateway(
        predictions={
            ("doctr/default", "infer"): (["a"], [_ocr_det(["a"])]),
            ("trocr/trocr-base-printed", "infer"): ["a"],
            ("pp_ocr/small-small", "infer"): (["a"], [_ocr_det(["a"])]),
        },
        model_info={
            "doctr/default": {"actions": {"infer": {}}},
            "trocr/trocr-base-printed": {"actions": {"infer": {}}},
            "pp_ocr/small-small": {"actions": {"infer": {}}},
        },
    )
    client = legacy_client(gw)
    for path in ("/doctr/ocr", "/ocr/trocr", "/ocr/pp-ocr"):
        gw.calls.clear()
        r = client.post(path, json={"image": _image()})
        assert r.status_code == 200, r.text
        assert _infer_params(gw) == {}, path


def test_trocr_returns_text_only_response(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={("trocr/trocr-base-printed", "infer"): ["abc"]},
        model_info={"trocr/trocr-base-printed": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/ocr/trocr", json={"image": _image()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "abc" and body["parent_id"] is None
    assert "predictions" not in body


def test_pp_ocr_resolves_versioned_model_id(legacy_client, fake_stat):
    fake_stat["pp-ocrv6-det/small"] = ("object-detection", "infer")
    fake_stat["pp-ocrv6-rec/small"] = ("text-only-ocr", "infer")
    gw = FakeGateway(
        predictions={
            ("pp_ocr/small-small", "infer"): (
                ["ocr"],
                [
                    SimpleNamespace(
                        xyxy=np.zeros((0, 4)),
                        confidence=np.zeros(0),
                        class_id=np.zeros(0),
                    )
                ],
            )
        },
        model_info={"pp_ocr/small-small": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/ocr/pp-ocr", json={"image": _image()})
    assert r.status_code == 200, r.text
    assert r.json()["result"] == "ocr"


def _ocr_det(texts):
    return SimpleNamespace(
        xyxy=np.array([[0, 0, 2, 2]] * len(texts), dtype=float),
        confidence=np.array([0.9] * len(texts)),
        class_id=np.array([0] * len(texts)),
        bboxes_metadata=[{"text": t} for t in texts],
    )


def test_pp_ocr_always_returns_boxes_with_recognized_text(legacy_client, fake_stat):
    fake_stat["pp-ocrv6-det/small"] = ("object-detection", "infer")
    fake_stat["pp-ocrv6-rec/small"] = ("text-only-ocr", "infer")
    gw = FakeGateway(
        predictions={("pp_ocr/small-small", "infer"): (["ocr"], [_ocr_det(["ocr"])])},
        model_info={"pp_ocr/small-small": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/ocr/pp-ocr", json={"image": _image()})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "ocr"
    assert (
        body["predictions"][0]["class"] == "ocr"
        and body["predictions"][0]["class_id"] == 0
    )
    assert body["image"] == {"width": 8, "height": 6}


def test_pp_ocr_detect_only_returns_boxes_and_empty_result(legacy_client, fake_stat):
    fake_stat["pp-ocrv6-det/small"] = ("object-detection", "infer")
    gw = FakeGateway(
        predictions={("pp_ocr/small-none", "infer"): ([""], [_ocr_det(["", ""])])},
        model_info={"pp_ocr/small-none": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/ocr/pp-ocr", json={"image": _image(), "text_recognition": "none"}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "" and len(body["predictions"]) == 2


def test_yolo_world_is_404_without_loading_a_model(legacy_client, fake_stat):
    gw = FakeGateway()
    r = legacy_client(gw).post(
        "/yolo_world/infer",
        json={"image": _image(), "text": ["cat", "dog"], "confidence": 0.2},
    )
    assert r.status_code == 404, r.text
    assert "not supported" in r.json()["message"]
    assert gw.calls == []


def test_grounding_dino_thresholds_reach_the_model_as_confidences(
    legacy_client, fake_stat
):
    gw = FakeGateway(
        predictions={("grounding_dino/default", "infer"): _det()},
        model_info={"grounding_dino/default": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/grounding_dino/infer",
        json={
            "image": _image(),
            "text": ["cat"],
            "box_threshold": 0.1,
            "text_threshold": 0.2,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["predictions"][0]["class"] == "cat"
    params = _infer_params(gw)
    assert params["box_confidence"] == 0.1 and params["text_confidence"] == 0.2


_OWLV2_MODEL_ID = "owlv2/owlv2-large-patch14-ensemble"


def _owlv2_client(monkeypatch, fake_stat, gateway, task_type="object-detection"):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from inference_server.legacy.bridge import LegacyModelBridge
    from inference_server.legacy.router import include_legacy_routers

    fake_stat[_OWLV2_MODEL_ID] = (task_type, "infer_with_reference_examples")
    monkeypatch.setattr("inference_server.configuration.CORE_MODEL_OWLV2_ENABLED", True)
    app = FastAPI()
    app.state.legacy_bridge = LegacyModelBridge(gateway)
    include_legacy_routers(app)
    return TestClient(app, raise_server_exceptions=False)


def _owlv2_body(**extra):
    return {
        "image": _image(),
        "training_data": [
            {
                "image": {"type": "base64", "value": _jpeg_b64(4, 4)},
                "boxes": [{"x": 1, "y": 1, "w": 2, "h": 2, "cls": "cat"}],
            }
        ],
        **extra,
    }


@pytest.mark.parametrize(
    "task_type", ["object-detection", "open-vocabulary-object-detection"]
)
def test_owlv2_few_shot_calls_infer_with_reference_examples(
    monkeypatch, fake_stat, task_type
):
    detections = _det()
    detections.image_metadata = {"class_names": ["cat"]}
    gw = FakeGateway(
        predictions={(_OWLV2_MODEL_ID, "infer_with_reference_examples"): detections},
        model_info={
            _OWLV2_MODEL_ID: {"actions": {"infer_with_reference_examples": {}}}
        },
    )
    r = _owlv2_client(monkeypatch, fake_stat, gw, task_type).post(
        "/owlv2/infer", json=_owlv2_body(confidence=0.95)
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predictions"][0]["class"] == "cat"
    assert body["predictions"][0]["class_id"] == 0
    assert body["image"] == {"width": 8, "height": 6}
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[1] == _OWLV2_MODEL_ID
    assert call[2] == "infer_with_reference_examples"
    assert call[3] == {
        "reference_examples": [
            {
                "image": base64.b64decode(_jpeg_b64(4, 4)),
                "boxes": [
                    {"x": 1, "y": 1, "w": 2, "h": 2, "cls": "cat", "negative": False}
                ],
            }
        ],
        "confidence": 0.95,
    }
    assert isinstance(call[4], bytes)


def test_owlv2_reference_url_is_fetched_through_the_server_fetcher(
    monkeypatch, fake_stat
):
    seen = []
    fetched = base64.b64decode(_jpeg_b64(4, 4))

    async def _fetch(urls, destination_policy=None):
        seen.append(list(urls))
        return [fetched for _ in urls], None

    monkeypatch.setattr("inference_server.legacy.common.fetch_images_from_urls", _fetch)
    gw = FakeGateway(
        predictions={(_OWLV2_MODEL_ID, "infer_with_reference_examples"): _det()},
        model_info={
            _OWLV2_MODEL_ID: {"actions": {"infer_with_reference_examples": {}}}
        },
    )
    body = _owlv2_body()
    body["training_data"][0]["image"] = {
        "type": "url",
        "value": "https://example.com/ref.jpg",
    }
    r = _owlv2_client(monkeypatch, fake_stat, gw).post("/owlv2/infer", json=body)
    assert r.status_code == 200, r.text
    assert seen == [["https://example.com/ref.jpg"]]
    call = next(c for c in gw.calls if c[0] == "infer")
    images = [example["image"] for example in call[3]["reference_examples"]]
    assert images == [fetched]
    assert not any(isinstance(image, str) for image in images)


def test_owlv2_refused_reference_url_answers_like_a_refused_image_url(
    monkeypatch, fake_stat
):
    monkeypatch.setattr("inference_server.legacy.common.ALLOW_URL_INPUT", False)
    gw = FakeGateway()
    client = _owlv2_client(monkeypatch, fake_stat, gw)
    body = _owlv2_body()
    body["training_data"][0]["image"] = {
        "type": "url",
        "value": "https://example.com/ref.jpg",
    }
    refused_reference = client.post("/owlv2/infer", json=body)
    refused_image = client.post(
        "/owlv2/infer",
        json={
            **_owlv2_body(),
            "image": {"type": "url", "value": "https://example.com/a.jpg"},
        },
    )
    assert refused_reference.status_code == refused_image.status_code == 400
    assert refused_reference.json() == refused_image.json()
    assert [c for c in gw.calls if c[0] == "infer"] == []


def test_owlv2_batch_returns_one_response_per_image(monkeypatch, fake_stat):
    gw = FakeGateway(
        predictions={(_OWLV2_MODEL_ID, "infer_with_reference_examples"): _det()},
        model_info={
            _OWLV2_MODEL_ID: {"actions": {"infer_with_reference_examples": {}}}
        },
    )
    body = _owlv2_body()
    body["image"] = [_image(), _image()]
    r = _owlv2_client(monkeypatch, fake_stat, gw).post("/owlv2/infer", json=body)
    assert r.status_code == 200, r.text
    assert [item["predictions"][0]["class"] for item in r.json()] == ["cat", "cat"]


def test_owlv2_requires_training_data(monkeypatch, fake_stat):
    gw = FakeGateway()
    r = _owlv2_client(monkeypatch, fake_stat, gw).post(
        "/owlv2/infer", json={"image": _image()}
    )
    assert r.status_code == 422
    assert gw.calls == []


def test_lmm_path_body_mismatch_is_400(legacy_client, fake_stat):
    fake_stat["smolvlm2/x"] = ("vlm", "prompt")
    r = legacy_client(FakeGateway()).post(
        "/infer/lmm/smolvlm2/x",
        json={"model_id": "other/1", "image": _image(), "prompt": "hi"},
    )
    assert r.status_code == 400
    assert r.json()["detail"].startswith("Model ID mismatch:")


def test_lmm_returns_text_response(legacy_client, fake_stat):
    fake_stat["smolvlm2/x"] = ("vlm", "prompt")
    gw = FakeGateway(
        predictions={("smolvlm2/x", "prompt"): ["a cat"]},
        model_info={"smolvlm2/x": {"actions": {"prompt": {}}}},
    )
    r = legacy_client(gw).post(
        "/infer/lmm/smolvlm2/x",
        json={"model_id": "smolvlm2/x", "image": _image(), "prompt": "what is it?"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["response"] == "a cat" and body["image"] == {"width": 8, "height": 6}
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[3] == {"prompt": "what is it?"}


def test_lmm_moondream_route_detects(legacy_client, fake_stat):
    fake_stat["moondream2/moondream2"] = ("vlm", "prompt")
    gw = FakeGateway(
        predictions={
            ("moondream2/moondream2", "detect"): SimpleNamespace(
                xyxy=np.array([[0.0, 0.0, 2.0, 4.0]])
            )
        },
        model_info={
            "moondream2/moondream2": {
                "actions": {"detect": {}},
                "model_class_name": "MoonDream2HF",
            }
        },
    )
    r = legacy_client(gw).post(
        "/infer/lmm",
        json={"model_id": "moondream2/moondream2", "image": _image(), "prompt": "dog"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["predictions"][0]["class"] == "dog"
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[2] == "detect" and call[3] == {"classes": ["dog"]}


def test_depth_png8_is_string(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("depth-anything-v2/small", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={"depth-anything-v2/small": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/infer/depth-estimation",
        json={"image": _image(), "depth_map_format": "png8"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert isinstance(body["normalized_depth"], str)
    assert body["depth_map_format"] == "png8"
    assert body["image"]


def test_depth_json_format_returns_matrix(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("depth-anything-v2/small", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={"depth-anything-v2/small": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post("/infer/depth-estimation", json={"image": _image()})
    assert r.status_code == 200, r.text
    normalized_depth = r.json()["normalized_depth"]
    assert np.allclose(normalized_depth, [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]])


@pytest.mark.parametrize(
    "model_class_name,expected",
    [
        ("YOLO26ForDepthEstimation", [[1.0, 2.0 / 3.0], [1.0 / 3.0, 0.0]]),
        ("DepthAnythingV2", [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]),
        (None, [[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]]),
    ],
)
def test_depth_map_is_inverted_only_for_yolo26_models(
    legacy_client, fake_stat, model_class_name, expected
):
    gw = FakeGateway(
        predictions={
            ("depth-anything-v2/small", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={
            "depth-anything-v2/small": {
                "actions": {"infer": {}},
                "model_class_name": model_class_name,
            }
        },
    )
    r = legacy_client(gw).post("/infer/depth-estimation", json={"image": _image()})
    assert r.status_code == 200, r.text
    assert np.allclose(r.json()["normalized_depth"], expected)


def test_depth_path_model_id_is_used(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("depth-anything-v3/large", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={"depth-anything-v3/large": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/infer/depth-estimation/depth-anything-v3/large", json={"image": _image()}
    )
    assert r.status_code == 200, r.text
    assert any(c[0] == "infer" and c[1] == "depth-anything-v3/large" for c in gw.calls)


def test_gaze_is_410(legacy_client):
    r = legacy_client(FakeGateway()).post("/gaze/gaze_detection")
    assert r.status_code == 410
    body = r.json()
    assert body["error_type"] == "FeatureDeprecatedError"
    assert body["feature"] == "/gaze/gaze_detection"
    assert body["replacement"] is None


def test_action_recognition_route_validates_its_body(legacy_client):
    response = legacy_client(FakeGateway()).post("/infer/action_recognition", json={})

    assert response.status_code == 422
    missing = {tuple(error["loc"]) for error in response.json()["detail"]}
    assert missing == {("body", "model_id"), ("body", "video")}


def test_optional_stubs_register_only_with_their_flags(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr("inference_server.configuration.SAM3_3D_OBJECTS_ENABLED", True)
    monkeypatch.setattr("inference_server.configuration.CORE_MODEL_OWLV2_ENABLED", True)
    app = FastAPI()
    include_legacy_routers(app)
    paths = route_paths(app)
    assert {"/sam3_3d/infer", "/owlv2/infer"} <= paths
    assert TestClient(app).post("/sam3_3d/infer", json={}).status_code == 501


def test_decided_out_routes_document_reason(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr("inference_server.configuration.SAM3_3D_OBJECTS_ENABLED", True)
    app = FastAPI()
    app.state.legacy_bridge = SimpleNamespace()
    include_legacy_routers(app)
    client = TestClient(app)

    yolo_world = client.post(
        "/yolo_world/infer",
        json={"image": _image(), "text": ["cat"]},
    )
    sam3_3d = client.post("/sam3_3d/infer", json={})
    gaze = client.post("/gaze/gaze_detection")

    assert yolo_world.status_code == 404
    assert yolo_world.json() == {
        "message": "YOLO-World is not supported by this inference server configuration."
    }
    assert sam3_3d.status_code == 501
    assert sam3_3d.json() == {
        "message": "/sam3_3d/infer is not available on inference_server: no model "
        "class for this task is registered with the model manager"
    }
    assert gaze.status_code == 410
    assert gaze.json() == {
        "message": (
            "Feature '/gaze/gaze_detection' has been removed from inference. "
            "Reason: MediaPipe dependency removed from inference; endpoint is a "
            "410 stub.. Removed in end of Q2 2026. No drop-in replacement is "
            "provided; contact Roboflow if you require this capability."
        ),
        "error_type": "FeatureDeprecatedError",
        "feature": "/gaze/gaze_detection",
        "removal_release": "end of Q2 2026",
        "replacement": None,
        "reason": "MediaPipe dependency removed from inference; endpoint is a 410 stub.",
    }


def test_lmm_router_registers_with_lmm_flag_alone(monkeypatch):
    from fastapi import FastAPI

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr("inference_server.configuration.LMM_ENABLED", True)
    monkeypatch.setattr("inference_server.configuration.MOONDREAM2_ENABLED", False)
    app = FastAPI()
    include_legacy_routers(app)
    assert "/infer/lmm" in route_paths(app)


@pytest.mark.parametrize(
    "core_models,lmm,moondream,lambda_,expected",
    [
        (True, True, False, False, True),
        (True, False, True, False, True),
        (False, True, False, False, True),
        (False, False, True, False, True),
        (True, False, False, False, False),
        (False, False, False, False, False),
        (False, True, True, True, False),
    ],
)
def test_lmm_router_is_independent_of_core_models_flag(
    monkeypatch, core_models, lmm, moondream, lambda_, expected
):
    from fastapi import FastAPI

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr(
        "inference_server.configuration.CORE_MODELS_ENABLED", core_models
    )
    monkeypatch.setattr("inference_server.configuration.LMM_ENABLED", lmm)
    monkeypatch.setattr("inference_server.configuration.MOONDREAM2_ENABLED", moondream)
    monkeypatch.setattr("inference_server.configuration.LAMBDA", lambda_)
    app = FastAPI()
    include_legacy_routers(app)
    assert ("/infer/lmm" in route_paths(app)) is expected


def test_disabled_group_flag_removes_its_routes(monkeypatch):
    from fastapi import FastAPI

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr("inference_server.configuration.CORE_MODEL_CLIP_ENABLED", False)
    app = FastAPI()
    include_legacy_routers(app)
    paths = route_paths(app)
    assert "/clip/embed_text" not in paths
    assert "/doctr/ocr" in paths


def test_sam2_embed_image_sends_namespaced_image_hash(legacy_client, fake_stat):
    from inference_model_manager.hash_namespacing import namespace_client_hash_id

    gw = FakeGateway(
        predictions={
            ("sam2/hiera_large", "embed"): [
                SimpleNamespace(image_hash=namespace_client_hash_id("img-1", "k"))
            ]
        },
        model_info={"sam2/hiera_large": {"actions": {"embed": {}}}},
    )
    r = legacy_client(gw).post(
        "/sam2/embed_image",
        json={"image": _image(), "image_id": "img-1", "api_key": "k"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_id"] == "img-1"
    assert "frame_id" in body and body["frame_id"] is None
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[3] == {"image_hashes": [namespace_client_hash_id("img-1", "k")]}


def test_sam2_segment_image_binary_returns_compressed_npz(legacy_client, fake_stat):
    masks = np.zeros((1, 2, 6, 8), dtype=np.float32)
    masks[0, 1, 1:4, 1:5] = 1.0
    logits = np.zeros((1, 2, 256, 256), dtype=np.float32)
    logits[0, 1, :8, :8] = 2.0
    gw = FakeGateway(
        predictions={
            ("sam2/hiera_large", "segment"): [
                SimpleNamespace(
                    masks=masks, scores=np.array([[0.1, 0.8]]), logits=logits
                )
            ]
        },
        model_info={"sam2/hiera_large": {"actions": {"segment": {}}}},
    )
    r = legacy_client(gw).post(
        "/sam2/segment_image", json={"image": _image(), "format": "binary"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    decoded = np.load(io.BytesIO(r.content))
    assert set(decoded.files) == {"masks", "low_res_masks"}
    np.testing.assert_array_equal(decoded["masks"], masks[:, 1])
    np.testing.assert_array_equal(decoded["low_res_masks"], logits[:, 1])


def test_sam_segment_image_binary_returns_compressed_npz(legacy_client, fake_stat):
    masks = np.zeros((1, 6, 8), dtype=bool)
    masks[0, 1:4, 1:5] = True
    logits = np.full((1, 256, 256), -1.5, dtype=np.float32)
    gw = FakeGateway(
        predictions={
            ("sam/vit_h", "segment"): [SimpleNamespace(masks=masks, logits=logits)]
        },
        model_info={"sam/vit_h": {"actions": {"segment": {}}}},
    )
    r = legacy_client(gw).post(
        "/sam/segment_image",
        json={"image": _image(), "format": "binary", "point_coords": [[1, 1]]},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    decoded = np.load(io.BytesIO(r.content))
    np.testing.assert_array_equal(decoded["masks"], masks)
    np.testing.assert_array_equal(decoded["low_res_masks"], logits)


def test_sam_segment_image_with_embeddings_and_no_image_is_params_only(
    legacy_client, fake_stat
):
    from inference_model_manager.hash_namespacing import namespace_client_hash_id

    masks = np.zeros((1, 6, 8), dtype=bool)
    masks[0, 1:4, 1:5] = True
    logits = np.full((1, 256, 256), -1.5, dtype=np.float32)
    embeddings = np.random.default_rng(3).random((1, 2, 2, 2), dtype=np.float32)
    gw = FakeGateway(
        predictions={
            ("sam/vit_h", "segment"): [SimpleNamespace(masks=masks, logits=logits)]
        },
        model_info={"sam/vit_h": {"actions": {"segment": {}}}},
    )
    r = legacy_client(gw).post(
        "/sam/segment_image",
        json={
            "embeddings": embeddings.tolist(),
            "image_id": "img-1",
            "orig_im_size": [6, 8],
            "point_coords": [[1, 1]],
            "point_labels": [1],
            "api_key": "k",
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["masks"]) == 1
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[4] is None
    params = call[3]
    np.testing.assert_array_equal(params["embeddings"]["embeddings"], embeddings)
    assert params["embeddings"]["image_hash"] == namespace_client_hash_id("img-1", "k")
    assert params["embeddings"]["image_size_hw"] == [6, 8]
    assert "image_hashes" not in params


def test_sam_segment_image_embeddings_without_orig_im_size_is_400(
    legacy_client, fake_stat
):
    gw = FakeGateway(model_info={"sam/vit_h": {"actions": {"segment": {}}}})
    r = legacy_client(gw).post(
        "/sam/segment_image",
        json={"embeddings": [[[[0.5]]]], "image_id": "img-1"},
    )
    assert r.status_code == 400, r.text
    assert "orig_im_size is required when image not provided" in r.json()["message"]


def test_sam3_visual_segment_binary_returns_compressed_npz(legacy_client, fake_stat):
    masks = np.zeros((1, 1, 6, 8), dtype=np.float32)
    logits = np.zeros((1, 1, 256, 256), dtype=np.float32)
    gw = FakeGateway(
        predictions={
            ("sam3/sam3_interactive", "segment_with_visual_prompts"): [
                SimpleNamespace(masks=masks, scores=np.array([0.9]), logits=logits)
            ]
        },
        model_info={
            "sam3/sam3_interactive": {"actions": {"segment_with_visual_prompts": {}}}
        },
    )
    r = legacy_client(gw).post(
        "/sam3/visual_segment", json={"image": _image(), "format": "binary"}
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "application/octet-stream"
    decoded = np.load(io.BytesIO(r.content))
    assert set(decoded.files) == {"masks", "low_res_masks"}
    params = _infer_params(gw)
    assert params["mask_format"] == "dense" and params["return_logits"] is True


def test_lmm_missing_prompt_is_sent_as_none_for_families_with_a_default(
    legacy_client, fake_stat
):
    fake_stat["qwen25vl/x"] = ("vlm", "prompt")
    gw = FakeGateway(
        predictions={("qwen25vl/x", "prompt"): ["a cat"]},
        model_info={
            "qwen25vl/x": {"actions": {"prompt": {}}, "model_class_name": "Qwen25VLHF"}
        },
    )
    r = legacy_client(gw).post(
        "/infer/lmm", json={"model_id": "qwen25vl/x", "image": _image()}
    )
    assert r.status_code == 200, r.text
    assert _infer_params(gw) == {"prompt": None}


def test_lmm_missing_prompt_is_400_for_florence2(legacy_client, fake_stat):
    fake_stat["florence-2/x"] = ("vlm", "prompt")
    gw = FakeGateway(
        model_info={
            "florence-2/x": {
                "actions": {"prompt": {}},
                "model_class_name": "Florence2HF",
            }
        },
    )
    r = legacy_client(gw).post(
        "/infer/lmm", json={"model_id": "florence-2/x", "image": _image()}
    )
    assert r.status_code == 400, r.text
    assert not [c for c in gw.calls if c[0] == "infer"]


def test_lmm_florence2_prompt_carries_the_task_token(legacy_client, fake_stat):
    fake_stat["florence-2/x"] = ("vlm", "prompt")
    gw = FakeGateway(
        predictions={("florence-2/x", "prompt"): [{"<OD>": {"bboxes": []}}]},
        model_info={
            "florence-2/x": {
                "actions": {"prompt": {}},
                "model_class_name": "Florence2HF",
            }
        },
    )
    r = legacy_client(gw).post(
        "/infer/lmm",
        json={"model_id": "florence-2/x", "image": _image(), "prompt": "<OD>"},
    )
    assert r.status_code == 200, r.text
    assert _infer_params(gw) == {"prompt": "<OD>", "task": "<OD>"}


def test_sam3_concept_segment_rejects_fine_tuned_model_id(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr(
        "inference_server.configuration.SAM3_FINE_TUNED_MODELS_ENABLED", False
    )
    r = legacy_client(FakeGateway()).post(
        "/sam3/concept_segment",
        json={
            "image": _image(),
            "model_id": "my-project/3",
            "prompts": [{"text": "cat"}],
        },
    )
    assert r.status_code == 501, r.text
    assert "Fine-tuned SAM 3 models are not supported" in r.json()["message"]


def test_sam3_embed_image_resolves_interactive_model(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("sam3/sam3_interactive", "embed_images"): [
                SimpleNamespace(image_hash="server-hash")
            ]
        },
        model_info={"sam3/sam3_interactive": {"actions": {"embed_images": {}}}},
    )
    r = legacy_client(gw).post("/sam3/embed_image", json={"image": _image()})
    assert r.status_code == 200, r.text
    assert r.json()["image_id"] == "server-hash"
    call = next(c for c in gw.calls if c[0] == "infer")
    assert call[1] == "sam3/sam3_interactive"
    assert call[3]["return_embeddings"] is False


def test_sam3_embed_image_remote_exec_mode_is_501(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.configuration.SAM3_EXEC_MODE", "remote")
    embed = legacy_client(FakeGateway()).post(
        "/sam3/embed_image", json={"image": _image()}
    )
    assert embed.status_code == 501, embed.text
    assert embed.json() == {
        "detail": "SAM3 embedding is not supported in remote execution mode."
    }


_REMOTE_API_KEY = "remote-key-0123456789"
_SAM3_CONCEPT_BODY = {
    "prompt_results": [
        {
            "prompt_index": 0,
            "echo": {"prompt_index": 0, "type": "text", "text": "cat"},
            "predictions": [{"masks": [[[1, 1], [3, 1], [3, 5]]], "confidence": 0.8}],
        }
    ],
    "time": 0.2,
}
_SAM3_VISUAL_BODY = {
    "predictions": [
        {"masks": [[[1, 1], [3, 1], [3, 5]]], "confidence": 0.9, "format": "json"}
    ],
    "time": 0.3,
}


class _FakeRemote:
    def __init__(self, status_code=200, body=None, error=None, json_error=None):
        self.status_code = status_code
        self.body = body
        self.error = error
        self.json_error = json_error
        self.calls = []

    def post(self, url, headers=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, **kwargs})
        if self.error is not None:
            raise self.error
        return SimpleNamespace(status_code=self.status_code, json=self._json)

    def _json(self):
        if self.json_error is not None:
            raise self.json_error
        if self.body is None:
            raise ValueError("Expecting value")
        return self.body


@pytest.fixture
def sam3_remote(monkeypatch):
    from inference_server import platform_http

    monkeypatch.setattr("inference_server.configuration.SAM3_EXEC_MODE", "remote")
    monkeypatch.setattr(
        "inference_server.configuration.API_BASE_URL", "https://api.example.test"
    )
    monkeypatch.setattr(
        "inference_server.configuration.ROBOFLOW_INTERNAL_SERVICE_NAME", "gpu-pool"
    )
    monkeypatch.setattr(
        "inference_server.configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET",
        "pool-secret-0123456789",
    )

    def _install(**kwargs):
        remote = _FakeRemote(**kwargs)
        monkeypatch.setattr(platform_http.requests, "post", remote.post)
        return remote

    return _install


def test_sam3_concept_segment_remote_posts_to_seg_preview_proxy(
    legacy_client, sam3_remote
):
    remote = sam3_remote(body=_SAM3_CONCEPT_BODY)
    gw = FakeGateway()
    image = _image()
    r = legacy_client(gw).post(
        "/sam3/concept_segment?source=ui&source_info=canvas",
        json={
            "image": image,
            "model_id": "sam3/sam3_final",
            "prompts": [
                {"text": "cat"},
                {
                    "type": "visual",
                    "boxes": [{"x": 1, "y": 2, "width": 3, "height": 4}],
                },
            ],
            "output_prob_thresh": 0.3,
            "api_key": _REMOTE_API_KEY,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["prompt_results"][0]["predictions"][0]["confidence"] == 0.8
    assert gw.calls == []
    (call,) = remote.calls
    assert call["url"] == (
        f"https://api.example.test/inferenceproxy/seg-preview?api_key={_REMOTE_API_KEY}"
    )
    assert call["json"] == {
        "image": image,
        "prompts": [
            {"type": "text", "text": "cat"},
            {
                "type": "visual",
                "boxes": [{"x": 1.0, "y": 2.0, "width": 3.0, "height": 4.0}],
            },
        ],
        "output_prob_thresh": 0.3,
        "source": "ui",
        "source_info": "canvas",
    }
    assert call["timeout"] == 60
    assert call["headers"]["Content-Type"] == "application/json"
    assert call["headers"]["X-Roboflow-Internal-Service-Name"] == "gpu-pool"
    assert (
        call["headers"]["X-Roboflow-Internal-Service-Secret"]
        == "pool-secret-0123456789"
    )
    assert "x-roboflow-inference-version" in call["headers"]


def test_sam3_visual_segment_remote_posts_to_sam3_pvs_proxy(legacy_client, sam3_remote):
    remote = sam3_remote(body=_SAM3_VISUAL_BODY)
    gw = FakeGateway()
    image = _image()
    r = legacy_client(gw).post(
        f"/sam3/visual_segment?api_key={_REMOTE_API_KEY}&source=ui",
        json={
            "image": image,
            "prompts": [{"points": [{"x": 5, "y": 6, "positive": True}]}],
            "multimask_output": False,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["predictions"][0]["confidence"] == 0.9
    assert gw.calls == []
    (call,) = remote.calls
    assert call["url"] == (
        f"https://api.example.test/inferenceproxy/sam3-pvs?api_key={_REMOTE_API_KEY}"
    )
    assert call["json"] == {
        "image": image,
        "prompts": {"prompts": [{"points": [{"x": 5.0, "y": 6.0, "positive": True}]}]},
        "multimask_output": False,
        "source": "ui",
        "source_info": None,
    }
    assert call["timeout"] == 60
    assert call["headers"]["X-Roboflow-Internal-Service-Name"] == "gpu-pool"


def test_sam3_remote_omits_internal_service_headers_when_unset(
    legacy_client, sam3_remote, monkeypatch
):
    monkeypatch.setattr(
        "inference_server.configuration.ROBOFLOW_INTERNAL_SERVICE_NAME", None
    )
    monkeypatch.setattr(
        "inference_server.configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET", None
    )
    remote = sam3_remote(body=_SAM3_VISUAL_BODY)
    r = legacy_client(FakeGateway()).post(
        "/sam3/visual_segment", json={"image": _image()}
    )
    assert r.status_code == 200, r.text
    (call,) = remote.calls
    assert "X-Roboflow-Internal-Service-Name" not in call["headers"]
    assert "X-Roboflow-Internal-Service-Secret" not in call["headers"]


@pytest.mark.parametrize(
    "path, payload, detail",
    [
        (
            "/sam3/concept_segment",
            {"image": _image(), "prompts": [{"text": "cat"}]},
            "SAM3 remote request failed.",
        ),
        (
            "/sam3/visual_segment",
            {"image": _image()},
            "SAM3 visual_segment remote request failed.",
        ),
    ],
)
@pytest.mark.parametrize(
    "remote_kwargs",
    [
        {"status_code": 403, "body": {"message": "forbidden"}},
        {"status_code": 200, "body": None},
        {"status_code": 200, "body": {"unexpected": True}},
        {
            "error": requests.exceptions.ConnectionError(
                f"https://api.example.test/x?api_key={_REMOTE_API_KEY}"
            )
        },
        {
            "error": requests.exceptions.ReadTimeout(
                f"https://api.example.test/x?api_key={_REMOTE_API_KEY}"
            )
        },
    ],
)
def test_sam3_remote_failure_is_generic_500_without_the_api_key(
    legacy_client, sam3_remote, caplog, path, payload, detail, remote_kwargs
):
    sam3_remote(**remote_kwargs)
    with caplog.at_level("ERROR"):
        r = legacy_client(FakeGateway()).post(
            f"{path}?api_key={_REMOTE_API_KEY}", json=payload
        )
    assert r.status_code == 500, r.text
    assert r.json() == {"detail": detail}
    assert _REMOTE_API_KEY not in r.text
    assert any(detail.rstrip(".") in record.getMessage() for record in caplog.records)
    assert _REMOTE_API_KEY not in caplog.text


@pytest.mark.parametrize(
    "path, payload, route_text",
    [
        (
            "/sam3/concept_segment",
            {"image": _image(), "prompts": [{"text": "cat"}]},
            "SAM3 remote request failed",
        ),
        (
            "/sam3/visual_segment",
            {"image": _image()},
            "SAM3 visual_segment remote request failed",
        ),
    ],
)
@pytest.mark.parametrize(
    "remote_kwargs",
    [
        {
            "json_error": ValueError(
                f"bad body from https://api.example.test/x?api_key={_REMOTE_API_KEY}"
            )
        },
        {"body": {"time": _REMOTE_API_KEY, "predictions": _REMOTE_API_KEY}},
    ],
)
def test_sam3_remote_post_request_failure_is_redacted_and_chain_free(
    legacy_client,
    sam3_remote,
    caplog,
    monkeypatch,
    path,
    payload,
    route_text,
    remote_kwargs,
):
    from fastapi import HTTPException

    from inference_server.legacy import router

    sam3_remote(**remote_kwargs)
    raised = []
    original = router._sam3_remote

    async def _capture(*args, **kwargs):
        try:
            return await original(*args, **kwargs)
        except HTTPException as error:
            raised.append(error)
            raise

    monkeypatch.setattr(router, "_sam3_remote", _capture)
    with caplog.at_level("ERROR"):
        r = legacy_client(FakeGateway()).post(
            f"{path}?api_key={_REMOTE_API_KEY}", json=payload
        )
    assert r.status_code == 500, r.text
    assert _REMOTE_API_KEY not in r.text
    assert any(route_text in record.getMessage() for record in caplog.records)
    assert _REMOTE_API_KEY not in caplog.text
    (error,) = raised
    assert error.__cause__ is None
    assert error.__suppress_context__ is True


def test_sam3_concept_segment_remote_rejects_fine_tuned_before_proxying(
    legacy_client, sam3_remote, monkeypatch
):
    monkeypatch.setattr(
        "inference_server.configuration.SAM3_FINE_TUNED_MODELS_ENABLED", False
    )
    remote = sam3_remote(body=_SAM3_CONCEPT_BODY)
    r = legacy_client(FakeGateway()).post(
        "/sam3/concept_segment",
        json={
            "image": _image(),
            "model_id": "my-project/3",
            "prompts": [{"text": "cat"}],
        },
    )
    assert r.status_code == 501, r.text
    assert remote.calls == []


def test_sam3_concept_segment_keeps_echo_and_null_fields(legacy_client, fake_stat):
    masks = np.zeros((1, 6, 8), dtype=bool)
    masks[0, 1:4, 1:5] = True
    gw = FakeGateway(
        predictions={
            ("sam3/sam3_final", "segment_with_text_prompts"): [
                {"prompt_index": 0, "masks": masks, "scores": [0.8]}
            ]
        },
        model_info={"sam3/sam3_final": {"actions": {"segment_with_text_prompts": {}}}},
    )
    r = legacy_client(gw).post(
        "/sam3/concept_segment",
        json={
            "image": _image(),
            "model_id": "sam3/sam3_final",
            "prompts": [{"text": "cat"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prompt_results"][0]["echo"] == {
        "prompt_index": 0,
        "type": "text",
        "text": "cat",
        "num_boxes": 0,
    }
    assert body["prompt_results"][0]["predictions"][0]["confidence"] == 0.8
    assert "frame_id" in body and body["frame_id"] is None
    assert _infer_params(gw)["mask_format"] == "rle"


def test_depth_rejects_list_image(legacy_client, fake_stat):
    gw = FakeGateway(
        predictions={
            ("depth-anything-v2/small", "infer"): np.array(
                [[0.0, 1.0], [2.0, 3.0]], dtype=np.float32
            )
        },
        model_info={"depth-anything-v2/small": {"actions": {"infer": {}}}},
    )
    r = legacy_client(gw).post(
        "/infer/depth-estimation", json={"image": [_image(), _image()]}
    )
    assert r.status_code == 400, r.text
    assert r.json()["message"] == "Depth estimation accepts a single image."
    assert not [c for c in gw.calls if c[0] == "infer"]


def test_clip_embed_image_loads_every_image_in_one_call(
    legacy_client, fake_stat, monkeypatch
):
    import inference_server.legacy.router as router_module
    from inference_server.legacy.common import image_dims

    original = router_module.load_request_images
    load_calls = []

    async def spy(images, *, ndarray_ok):
        load_calls.append(list(images))
        return await original(images, ndarray_ok=ndarray_ok)

    monkeypatch.setattr(router_module, "load_request_images", spy)
    gw = FakeGateway(
        predictions={("clip/ViT-B-16", "embed_images"): np.array([[1.0, 0.0]])},
        model_info={"clip/ViT-B-16": {"actions": {"embed_images": {}}}},
    )
    first = {"type": "base64", "value": _jpeg_b64(8, 6)}
    second = {"type": "base64", "value": _jpeg_b64(4, 4)}
    r = legacy_client(gw).post("/clip/embed_image", json={"image": [first, second]})
    assert r.status_code == 200, r.text
    assert len(load_calls) == 1
    assert [image.value for image in load_calls[0]] == [
        first["value"],
        second["value"],
    ]
    infer_calls = [c for c in gw.calls if c[0] == "infer"]
    assert [c[2] for c in infer_calls] == ["embed_images", "embed_images"]
    assert [image_dims(c[4]) for c in infer_calls] == [(8, 6), (4, 4)]


def _pp_ocr_gateway():
    return FakeGateway(
        predictions={
            ("pp_ocr/small-small", "infer"): (
                ["ocr"],
                [
                    SimpleNamespace(
                        xyxy=np.zeros((0, 4)),
                        confidence=np.zeros(0),
                        class_id=np.zeros(0),
                    )
                ],
            )
        },
        model_info={"pp_ocr/small-small": {"actions": {"infer": {}}}},
    )


def test_pp_ocr_missing_stage_is_404(legacy_client, fake_stat):
    fake_stat["pp-ocrv6-det/small"] = ("object-detection", "infer")
    r = legacy_client(_pp_ocr_gateway()).post("/ocr/pp-ocr", json={"image": _image()})
    assert r.status_code == 404 and "message" in r.json()


def test_pp_ocr_denied_stage_is_401(legacy_client, fake_stat):
    from inference_models.errors import UnauthorizedModelAccessError

    fake_stat["pp-ocrv6-det/small"] = ("object-detection", "infer")
    fake_stat["pp-ocrv6-rec/small"] = UnauthorizedModelAccessError(
        message="pp-ocrv6-rec/small", help_url=""
    )
    r = legacy_client(_pp_ocr_gateway()).post("/ocr/pp-ocr", json={"image": _image()})
    assert r.status_code == 401


def _lmm_load_failure_status(legacy_client, fake_stat, error_type):
    fake_stat["qwen/1"] = ("vlm", "prompt")
    gw = FakeGateway(model_info={"qwen/1": {"actions": {"prompt": {}}}})
    gw.ensure_results = [
        ("error", 5, {"error_type": error_type, "message": "adapter rejected"})
    ]

    return legacy_client(gw).post(
        "/infer/lmm",
        json={"model_id": "qwen/1", "image": _image(), "prompt": "hi"},
    )


def test_lmm_load_failure_of_a_vllm_adapter_rejection_is_501(legacy_client, fake_stat):
    r = _lmm_load_failure_status(legacy_client, fake_stat, "AdapterNotServableError")

    assert r.status_code == 501
    assert r.json() == {"message": "adapter rejected"}


def test_lmm_load_failure_of_a_generic_class_is_still_500(legacy_client, fake_stat):
    r = _lmm_load_failure_status(legacy_client, fake_stat, "SomethingElseError")

    assert r.status_code == 500
