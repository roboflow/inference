import base64
import io
from types import SimpleNamespace

import numpy as np
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


def test_easy_ocr_rejects_other_languages(legacy_client, fake_stat):
    r = legacy_client(FakeGateway()).post(
        "/easy_ocr/ocr", json={"image": _image(), "language_codes": ["pl"]}
    )
    assert r.status_code == 501


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


def test_grounding_dino_custom_box_threshold_is_501(legacy_client, fake_stat):
    r = legacy_client(FakeGateway()).post(
        "/grounding_dino/infer",
        json={"image": _image(), "text": ["cat"], "box_threshold": 0.1},
    )
    assert r.status_code == 501


def test_owlv2_stub_is_501(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr("inference_server.configuration.CORE_MODEL_OWLV2_ENABLED", True)
    app = FastAPI()
    include_legacy_routers(app)
    assert TestClient(app).post("/owlv2/infer", json={}).status_code == 501


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


def test_action_recognition_stub_is_501(legacy_client):
    assert (
        legacy_client(FakeGateway())
        .post("/infer/action_recognition", json={})
        .status_code
        == 501
    )


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


def test_lmm_router_registers_with_lmm_flag_alone(monkeypatch):
    from fastapi import FastAPI

    from inference_server.legacy.router import include_legacy_routers

    monkeypatch.setattr("inference_server.configuration.LMM_ENABLED", True)
    monkeypatch.setattr("inference_server.configuration.MOONDREAM2_ENABLED", False)
    app = FastAPI()
    include_legacy_routers(app)
    assert "/infer/lmm" in route_paths(app)


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


def test_sam2_segment_image_binary_format_is_501(legacy_client, fake_stat):
    gw = FakeGateway(
        model_info={
            "sam2/hiera_large": {"actions": {"segment_with_visual_prompts": {}}}
        },
    )
    r = legacy_client(gw).post(
        "/sam2/segment_image", json={"image": _image(), "format": "binary"}
    )
    assert r.status_code == 501, r.text
    assert not [c for c in gw.calls if c[0] == "infer"]


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


def test_sam3_remote_exec_mode_is_501(legacy_client, fake_stat, monkeypatch):
    monkeypatch.setattr("inference_server.configuration.SAM3_EXEC_MODE", "remote")
    client = legacy_client(FakeGateway())
    for path, payload in (
        ("/sam3/embed_image", {"image": _image()}),
        (
            "/sam3/concept_segment",
            {"image": _image(), "prompts": [{"text": "cat"}]},
        ),
        ("/sam3/visual_segment", {"image": _image()}),
    ):
        response = client.post(path, json=payload)
        assert response.status_code == 501, (path, response.text)
    embed = client.post("/sam3/embed_image", json={"image": _image()})
    assert embed.json() == {
        "detail": "SAM3 embedding is not supported in remote execution mode."
    }


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
