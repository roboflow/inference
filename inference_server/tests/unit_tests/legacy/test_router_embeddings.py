import asyncio
import base64
import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tests.unit_tests.legacy.conftest import FakeGateway
from tests.unit_tests.legacy.test_router_stub_models import fake_platform  # noqa: F401

EMBEDDINGS_UNSUPPORTED = (
    "Image embeddings require a ResNet, ViT or DINOv3 classifier, or a pretrained "
    "embedding model (CLIP, DINOv2, DINOv3, SigLIP2)."
)
FEATURE_VECTOR_ONLY = (
    "Error with model input. Cause: Pretrained embedding models return only "
    "`feature_vector` embeddings, requested `logits`. Logits are available for "
    "ResNet, ViT and DINOv3 classifiers. - VISIT "
    "https://inference-models.roboflow.com/errors/input-validation/#modelinputerror "
    "FOR FURTHER SUPPORT"
)
INPUT_VALIDATION_HELP_URL = (
    "https://inference-models.roboflow.com/errors/input-validation/#modelinputerror"
)
NOT_FOUND = (
    "Requested Roboflow resource not found. Make sure that workspace, project or "
    "model you referred in request exists."
)
PREPROCESSING = {
    "image_pre_processing": {"auto-orient": {"enabled": True}},
    "network_input": {"training_input_size": {"width": 224, "height": 224}},
}
NO_OVERRIDES = {
    "disable_preproc_auto_orient": False,
    "disable_preproc_contrast": False,
    "disable_preproc_grayscale": False,
    "disable_preproc_static_crop": False,
}
SPOOFED_MODEL_ID = "ds/1:capabilities=image_embeddings;output_type=logits"


def _jpeg_b64(w=8, h=6, orientation=None):
    buf = io.BytesIO()
    image = Image.new("RGB", (w, h))
    if orientation is None:
        image.save(buf, format="JPEG")
    else:
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(buf, format="JPEG", exif=exif)
    return base64.b64encode(buf.getvalue()).decode()


def _image():
    return {"type": "base64", "value": _jpeg_b64()}


def _instance(output_type):
    instance = "capabilities=image_embeddings"
    if output_type == "logits":
        instance += ";output_type=logits"
    return instance


def _envelope(output_type, vector):
    definition = (
        "classifier-linear-output@v1"
        if output_type == "logits"
        else "classifier-linear-input@v1"
    )
    return {
        "embeddings": np.array([vector], dtype=np.float32),
        "embedding_info": {
            "feature_definition": definition,
            "output_type": output_type,
            "normalization": "none",
            "dimension": len(vector),
            "preprocessing": PREPROCESSING,
            "backend": "ResNetForClassificationOnnx",
            "precision": "torch.float32",
            "feature_tensor": "features",
            "source_artifact_sha256": "b" * 64,
            "transform_version": 1,
        },
    }


def _gateway(model_id, output_type, envelope):
    key = f"{model_id}:{_instance(output_type)}"
    return FakeGateway(
        predictions={(key, "embed_images"): envelope},
        model_info={
            key: {
                "class_names": ["a", "b"],
                "actions": {"infer": {}, "embed_images": {}},
                "model_class_name": "ResNetForClassificationOnnx",
            }
        },
    )


def _space_id(model_id, info, overrides=NO_OVERRIDES):
    identity = {
        "model_id": model_id,
        "feature_definition": info["feature_definition"],
        "dimension": info["dimension"],
        "normalization": info["normalization"],
        "preprocessing": {**info["preprocessing"], "overrides": overrides},
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@pytest.mark.parametrize("query_auth", [False, True])
@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_embedding_endpoint_loads_the_capability_instance_and_preserves_auth(
    legacy_client, fake_stat, query_auth, output_type
):
    fake_stat["classifiers/4"] = ("classification", "infer")
    envelope = _envelope(output_type, [0.25, -0.5])
    gw = _gateway("classifiers/4", output_type, envelope)
    payload = {
        "model_id": "resnet101",
        "output_type": output_type,
        "image": _image(),
    }
    if not query_auth:
        payload["api_key"] = "key"

    r = legacy_client(gw).post(
        "/infer/embeddings",
        params={"api_key": "key"} if query_auth else None,
        json=payload,
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["embeddings"] == [[0.25, -0.5]]
    assert body["embedding_info"] == {
        **envelope["embedding_info"],
        "preprocessing": {**PREPROCESSING, "overrides": NO_OVERRIDES},
        "model_id": "classifiers/4",
        "space_id": _space_id("classifiers/4", envelope["embedding_info"]),
    }
    assert set(body) == {"inference_id", "time", "embeddings", "embedding_info"}
    assert ("ensure_loaded", f"classifiers/4:{_instance(output_type)}", "key") in (
        gw.calls
    )
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[1] == f"classifiers/4:{_instance(output_type)}"
    assert infer_call[2] == "embed_images"
    assert infer_call[3] == {"output_type": output_type}
    assert isinstance(infer_call[4], bytes)


def test_embedding_endpoint_embeds_every_image_in_input_order(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("classification", "infer")
    vectors = iter([[1.0, 2.0], [3.0, 4.0]])
    gw = _gateway(
        "ds/1",
        "feature_vector",
        lambda image, params: _envelope("feature_vector", next(vectors)),
    )

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": [_image(), _image()],
            "disable_preproc_contrast": True,
        },
    )

    assert r.status_code == 200, r.text
    assert r.json()["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
    infer_calls = [c for c in gw.calls if c[0] == "infer"]
    assert len(infer_calls) == 2
    assert infer_calls[0][3] == {
        "output_type": "feature_vector",
        "disable_preproc_contrast": True,
    }


def test_embedding_endpoint_keeps_null_overrides_in_the_space_identity(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("classification", "infer")
    envelope = _envelope("feature_vector", [0.5])
    gw = _gateway("ds/1", "feature_vector", envelope)

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": _image(),
            "disable_preproc_grayscale": None,
            "disable_preproc_static_crop": True,
        },
    )

    assert r.status_code == 200, r.text
    overrides = {
        **NO_OVERRIDES,
        "disable_preproc_grayscale": None,
        "disable_preproc_static_crop": True,
    }
    info = r.json()["embedding_info"]
    assert info["preprocessing"]["overrides"] == overrides
    assert info["space_id"] == _space_id("ds/1", envelope["embedding_info"], overrides)
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[3] == {
        "output_type": "feature_vector",
        "disable_preproc_static_crop": True,
    }


def test_embedding_endpoint_keeps_the_image_orientation_when_auto_orient_is_off(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("classification", "infer")
    envelope = _envelope("feature_vector", [0.5])
    gw = _gateway("ds/1", "feature_vector", envelope)
    rotated = {"type": "base64", "value": _jpeg_b64(8, 6, orientation=6)}

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": rotated,
            "disable_preproc_auto_orient": True,
        },
    )

    assert r.status_code == 200, r.text
    overrides = {**NO_OVERRIDES, "disable_preproc_auto_orient": True}
    info = r.json()["embedding_info"]
    assert info["preprocessing"]["overrides"] == overrides
    assert info["space_id"] == _space_id("ds/1", envelope["embedding_info"], overrides)
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[4][:6] == b"\x93NUMPY"
    assert np.load(io.BytesIO(infer_call[4])).shape == (6, 8, 3)
    assert infer_call[3] == {"output_type": "feature_vector"}


def test_embedding_endpoint_keeps_the_orientation_off_the_event_loop(
    legacy_client, fake_stat, monkeypatch
):
    from inference_server.legacy import router

    keep_image_orientation = router.keep_image_orientation
    on_loop = []

    def _spy(payload, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return keep_image_orientation(payload, **kwargs)

    monkeypatch.setattr(router, "keep_image_orientation", _spy)
    fake_stat["ds/1"] = ("classification", "infer")
    gw = _gateway("ds/1", "feature_vector", _envelope("feature_vector", [0.5]))

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": [_image(), _image()],
            "disable_preproc_auto_orient": True,
        },
    )

    assert r.status_code == 200, r.text
    assert on_loop == [False, False]


def test_embedding_endpoint_time_includes_image_loading(
    legacy_client, fake_stat, monkeypatch
):
    import asyncio

    import inference_server.legacy.router as router_module

    original = router_module.load_request_images

    async def slow(images, *, ndarray_ok):
        await asyncio.sleep(0.05)
        return await original(images, ndarray_ok=ndarray_ok)

    monkeypatch.setattr(router_module, "load_request_images", slow)
    fake_stat["ds/1"] = ("classification", "infer")
    gw = _gateway("ds/1", "feature_vector", _envelope("feature_vector", [0.5]))

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 200, r.text
    assert r.json()["time"] >= 0.05


@pytest.mark.parametrize("path", ["/infer/classification", "/infer/embeddings"])
def test_capability_markers_in_a_model_id_never_load_a_capability_instance(
    legacy_client, fake_stat, monkeypatch, path
):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    gw = FakeGateway()

    r = legacy_client(gw).post(
        path, json={"model_id": SPOOFED_MODEL_ID, "api_key": "k", "image": _image()}
    )

    assert r.status_code == 404, r.text
    assert r.json() == {"message": NOT_FOUND}
    assert gw.calls == []


def test_capability_markers_in_a_model_id_are_refused_before_the_registry(
    legacy_client, fake_stat
):
    fake_stat[SPOOFED_MODEL_ID] = ("classification", "infer")
    gw = FakeGateway()

    r = legacy_client(gw).post(
        "/infer/classification",
        json={"model_id": SPOOFED_MODEL_ID, "api_key": "k", "image": _image()},
    )

    assert r.status_code == 404, r.text
    assert gw.calls == []


def test_capability_markers_in_a_model_id_are_refused_by_model_add(
    legacy_client, fake_stat
):
    fake_stat[SPOOFED_MODEL_ID] = ("classification", "infer")
    gw = FakeGateway()

    r = legacy_client(gw).post(
        "/model/add", json={"model_id": SPOOFED_MODEL_ID, "api_key": "k"}
    )

    assert r.status_code == 404, r.text
    assert gw.calls == []


@pytest.mark.parametrize(
    "model_id",
    [
        "ds/1:b:capabilities=image_embeddings",
        "ds/1:capabilities=image_embeddings:output_type=logits",
        "ds/1:capabilities=image_embeddings;output_type=logits:b",
    ],
)
def test_capability_markers_anywhere_in_a_model_id_are_refused(
    legacy_client, fake_stat, monkeypatch, model_id
):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    gw = FakeGateway()

    r = legacy_client(gw).post(
        "/infer/classification",
        json={"model_id": model_id, "api_key": "k", "image": _image()},
    )

    assert r.status_code == 404, r.text
    assert gw.calls == []


def _named_instance_gateway():
    return FakeGateway(
        predictions={
            ("ds/1:blue", "infer"): SimpleNamespace(confidence=np.array([0.1, 0.9]))
        },
        model_info={
            "ds/1:blue": {
                "class_names": ["a", "b"],
                "actions": {"infer": {}},
                "model_mro_names": ["ClassificationModel"],
            }
        },
    )


def test_named_instances_stay_usable_through_legacy_routes(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    gw = _named_instance_gateway()
    client = legacy_client(gw)
    asyncio.run(gw.load("ds/1:blue"))

    r = client.post(
        "/infer/classification",
        json={"model_id": "ds/1:blue", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 200, r.text
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[1] == "ds/1:blue"
    assert r.headers["X-Model-Id"] == "ds/1:blue"
    rows = {
        entry["model_id"] for entry in client.get("/model/registry").json()["models"]
    }
    assert "ds/1:blue" in rows


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/v2/models/load"),
        ("post", "/v2/models/unload"),
        ("get", "/v2/models/interface"),
    ],
)
@pytest.mark.parametrize(
    "model_id", [SPOOFED_MODEL_ID, "ds/1:b:capabilities=image_embeddings"]
)
def test_v2_model_management_refuses_capability_markers(
    legacy_client, fake_stat, monkeypatch, method, path, model_id
):
    gw = FakeGateway()
    client = _v2_client(legacy_client, monkeypatch, gw)

    r = getattr(client, method)(
        path, params={"model_id": model_id}, headers={"Authorization": "Bearer k"}
    )

    assert r.status_code == 404, r.text
    assert r.json()["error_code"] == "MODEL_NOT_FOUND"
    assert gw.calls == []


def _v2_client(legacy_client, monkeypatch, gateway):
    from unittest.mock import AsyncMock

    import inference_server.app as app_module

    monkeypatch.setattr(app_module._cfg, "ENABLE_CONTROL_PLANE_ROUTES", True)
    monkeypatch.setattr(
        app_module, "validate_api_key", AsyncMock(return_value=(True, "ws-1"))
    )
    return legacy_client(gateway)


def test_v2_load_keeps_named_instances(legacy_client, fake_stat, monkeypatch):
    gw = FakeGateway()
    client = _v2_client(legacy_client, monkeypatch, gw)

    r = client.post(
        "/v2/models/load",
        params={"model_id": "ds/1:blue"},
        headers={"Authorization": "Bearer k"},
    )

    assert r.status_code == 200, r.text
    assert ("load", "ds/1:blue", "k") in gw.calls


@pytest.mark.parametrize("via_url", [False, True])
def test_auto_orient_off_keeps_the_decode_ceiling(
    legacy_client, fake_stat, monkeypatch, via_url
):
    import cv2

    monkeypatch.setattr(
        "inference_model_manager.configuration.INFERENCE_DECODE_MAX_MEGAPIXELS", 1.0
    )
    decodes = []
    monkeypatch.setattr(cv2, "imdecode", lambda *args: decodes.append(args))
    large = base64.b64decode(_jpeg_b64(1100, 1000))
    if via_url:

        async def fetched(urls):
            return [large for _ in urls], None

        monkeypatch.setattr("inference_server.legacy.common.fetch_url_images", fetched)
        image = {"type": "url", "value": "https://example.com/large.jpg"}
    else:
        image = {"type": "base64", "value": base64.b64encode(large).decode()}
    fake_stat["ds/1"] = ("classification", "infer")
    gw = _gateway("ds/1", "feature_vector", _envelope("feature_vector", [0.5]))

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": image,
            "disable_preproc_auto_orient": True,
        },
    )

    assert r.status_code == 400, r.text
    assert r.json()["message"] == (
        "Error with model input. Cause: image is 1.1 megapixels (header), over "
        "the 1 megapixel decode limit"
    )
    assert decodes == []
    assert [c for c in gw.calls if c[0] == "infer"] == []


def _registry_rows(client):
    return {
        entry["model_id"]: entry
        for entry in client.get("/model/registry").json()["models"]
    }


def test_embedding_registrations_are_listed_and_removed_by_their_legacy_key(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("classification", "infer")
    gw = _gateway("ds/1", "logits", _envelope("logits", [0.5]))
    client = legacy_client(gw)
    r = client.post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": _image(),
            "output_type": "logits",
        },
    )
    assert r.status_code == 200, r.text
    legacy_key = "ds/1:capabilities=image_embeddings:output_type=logits"
    routing_key = "ds/1:capabilities=image_embeddings;output_type=logits"

    rows = _registry_rows(client)
    assert set(rows) == {legacy_key}
    assert rows[legacy_key]["task_type"] == "classification"
    assert rows[legacy_key]["request_aliases"] == []
    assert rows[legacy_key]["request_paths"] == ["/infer/embeddings"]

    untouched = client.post("/model/remove", json={"model_id": "ds/1"}).json()
    assert {entry["model_id"] for entry in untouched["models"]} == {legacy_key}
    assert [c for c in gw.calls if c[0] == "unload"] == []

    removed = client.post("/model/remove", json={"model_id": legacy_key}).json()
    assert removed["models"] == []
    assert [c for c in gw.calls if c[0] == "unload"] == [("unload", routing_key)]


def test_embedding_requests_for_unknown_models_never_reach_the_gateway(
    legacy_client, fake_stat
):
    gw = FakeGateway()
    client = legacy_client(gw)

    for index in range(3):
        r = client.post(
            "/infer/embeddings",
            json={"model_id": f"missing/{index}", "api_key": "k", "image": _image()},
        )
        assert r.status_code in (401, 404), r.text

    assert gw.calls == []


def test_aliased_embedding_registrations_are_listed_under_the_requested_alias(
    legacy_client, fake_stat
):
    fake_stat["classifiers/4"] = ("classification", "infer")
    gw = _gateway("classifiers/4", "feature_vector", _envelope("feature_vector", [0.5]))
    client = legacy_client(gw)
    r = client.post(
        "/infer/embeddings",
        json={"model_id": "resnet101", "api_key": "k", "image": _image()},
    )
    assert r.status_code == 200, r.text
    legacy_key = "resnet101:capabilities=image_embeddings"

    rows = _registry_rows(client)
    assert set(rows) == {legacy_key}
    assert rows[legacy_key]["request_aliases"] == ["classifiers/4"]

    removed = client.post("/model/remove", json={"model_id": legacy_key}).json()
    assert removed["models"] == []
    assert [c for c in gw.calls if c[0] == "unload"] == [
        ("unload", "classifiers/4:capabilities=image_embeddings")
    ]


def test_embedding_endpoint_rejects_a_model_of_another_task_without_loading(
    legacy_client, fake_stat
):
    fake_stat["ds/1"] = ("object-detection", "infer")
    gw = FakeGateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={"model_id": "ds/1", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 501
    assert r.json() == {"message": EMBEDDINGS_UNSUPPORTED}
    assert gw.calls == []


def test_embedding_endpoint_rejects_a_stub_model(
    legacy_client, fake_stat, fake_platform  # noqa: F811
):
    fake_platform.project = (200, {"project": {"type": "classification"}})
    gw = FakeGateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 501
    assert r.json() == {"message": EMBEDDINGS_UNSUPPORTED}
    assert gw.calls == []


def test_embedding_endpoint_requires_at_least_one_image(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("classification", "infer")

    r = legacy_client(FakeGateway()).post(
        "/infer/embeddings", json={"model_id": "ds/1", "api_key": "k", "image": []}
    )

    assert r.status_code == 422


def test_embedding_endpoint_rejects_an_unknown_output_type(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("classification", "infer")

    r = legacy_client(FakeGateway()).post(
        "/infer/embeddings",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": _image(),
            "output_type": "softmax",
        },
    )

    assert r.status_code == 422


def test_embedding_response_reports_the_legacy_registration_key(
    legacy_client, fake_stat
):
    fake_stat["classifiers/4"] = ("classification", "infer")
    gw = _gateway("classifiers/4", "logits", _envelope("logits", [0.5]))

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "resnet101",
            "api_key": "k",
            "image": _image(),
            "output_type": "logits",
        },
    )

    assert r.status_code == 200, r.text
    assert (
        r.headers["X-Model-Id"]
        == "resnet101:capabilities=image_embeddings:output_type=logits"
    )


def test_embedding_request_for_an_unknown_model_reports_its_registration_key(
    legacy_client, fake_stat
):
    r = legacy_client(FakeGateway()).post(
        "/infer/embeddings",
        json={"model_id": "missing/1", "api_key": "k", "image": _image()},
    )

    assert r.status_code in (401, 404), r.text
    assert r.headers["X-Model-Id"] == "missing/1:capabilities=image_embeddings"


def test_model_add_reports_the_canonical_model_id(legacy_client, fake_stat):
    fake_stat["coco/3"] = ("object-detection", "infer")

    r = legacy_client(FakeGateway()).post(
        "/model/add", json={"model_id": "yolov8n-640", "api_key": "k"}
    )

    assert r.status_code == 200, r.text
    assert r.headers["X-Model-Id"] == "coco/3"


PRETRAINED_PREPROCESSING = {
    "image_processor_type": "BitImageProcessor",
    "do_convert_rgb": True,
    "do_resize": True,
    "size": {"shortest_edge": 256},
    "resample": 3,
    "do_center_crop": True,
    "crop_size": {"height": 224, "width": 224},
    "do_rescale": True,
    "rescale_factor": 0.00392156862745098,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}
CLIP_PREPROCESSING = {
    "image_size": 224,
    "resize": "shortest_edge",
    "resample": "bicubic",
    "center_crop": True,
    "rescale_factor": 1 / 255,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}
# Golden ids of test_translation_embeddings.py, equal to the legacy server's.
DINOV2_SPACE_ID = "18f23774f892940283119ce8640ee99fc1f94060e190c3edad304690824dd7d5"
CLIP_SPACE_ID = "1bc0eb86e1747bddf8f0b1c0b109365a8347a2b40bfbeaab302b122e7d726ac1"
PRETRAINED_INSTANCE = "capabilities=image_embeddings"
PE_MODEL_ID = "perception_encoder/PE-Core-L14-336"


def _described_envelope(feature_definition, dimension, preprocessing, backend):
    return {
        "embeddings": np.array([[0.5] * dimension], dtype=np.float32),
        "embedding_info": {
            "feature_definition": feature_definition,
            "output_type": "feature_vector",
            "normalization": "none",
            "dimension": dimension,
            "preprocessing": preprocessing,
            "backend": backend,
            "precision": "torch.float32",
        },
    }


def _pretrained_gateway(model_id="dinov2/small"):
    key = f"{model_id}:{PRETRAINED_INSTANCE}"
    envelope = _described_envelope(
        "dinov2-cls-token@v1",
        384,
        PRETRAINED_PREPROCESSING,
        "PretrainedImageEmbeddingOnnx",
    )
    return FakeGateway(
        predictions={(key, "embed_images"): envelope},
        model_info={
            key: {
                "actions": {"embed_images": {}},
                "model_class_name": "PretrainedImageEmbeddingOnnx",
                "model_mro_names": [
                    "PretrainedImageEmbeddingOnnx",
                    "ImageEmbeddingModel",
                    "object",
                ],
            }
        },
    )


def _clip_gateway(actions=("embed_images", "embed_text", "embed_images_with_info")):
    envelope = _described_envelope(
        "clip-image-embeds@v1", 512, CLIP_PREPROCESSING, "ClipOnnx"
    )
    return FakeGateway(
        predictions={
            ("clip/ViT-B-16", "embed_images_with_info"): envelope,
            ("clip/ViT-B-16", "embed_images"): np.array([[0.5, 0.5]]),
        },
        model_info={
            "clip/ViT-B-16": {
                "actions": {action: {} for action in actions},
                "model_class_name": "ClipOnnx",
                "model_mro_names": ["ClipOnnx", "TextImageEmbeddingModel", "object"],
            }
        },
    )


def test_pretrained_embedding_model_returns_vectors_and_its_space(
    legacy_client, fake_stat
):
    fake_stat["dinov2/small"] = ("embedding", "embed_images")
    gw = _pretrained_gateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "dinov2/small",
            "api_key": "k",
            "image": [_image(), _image()],
            "disable_preproc_grayscale": True,
        },
    )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["embeddings"] == [[0.5] * 384, [0.5] * 384]
    assert body["embedding_info"] == {
        "model_id": "dinov2/small",
        "feature_definition": "dinov2-cls-token@v1",
        "output_type": "feature_vector",
        "normalization": "none",
        "dimension": 384,
        "preprocessing": PRETRAINED_PREPROCESSING,
        "space_id": DINOV2_SPACE_ID,
        "backend": "PretrainedImageEmbeddingOnnx",
        "precision": "torch.float32",
    }
    assert r.headers["X-Model-Id"] == "dinov2/small:capabilities=image_embeddings"
    key = f"dinov2/small:{PRETRAINED_INSTANCE}"
    infer_calls = [c for c in gw.calls if c[0] == "infer"]
    assert [c[1:3] for c in infer_calls] == [(key, "embed_images")] * 2
    assert infer_calls[0][3] == {
        "output_type": "feature_vector",
        "disable_preproc_grayscale": True,
    }


@pytest.mark.parametrize("model_id", ["dinov2/small", "clip/ViT-B-16"])
def test_pretrained_embedding_models_have_no_logits(legacy_client, fake_stat, model_id):
    fake_stat["dinov2/small"] = ("embedding", "embed_images")
    gw = _pretrained_gateway() if model_id == "dinov2/small" else _clip_gateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": model_id,
            "api_key": "k",
            "image": _image(),
            "output_type": "logits",
        },
    )

    assert r.status_code == 400, r.text
    assert r.json() == {
        "message": FEATURE_VECTOR_ONLY,
        "help_url": INPUT_VALIDATION_HELP_URL,
    }
    assert gw.calls == []


def test_classifier_space_keeps_the_request_overrides(legacy_client, fake_stat):
    fake_stat["my-classifier/1"] = ("classification", "infer")
    preprocessing = {
        "image_pre_processing": {
            "auto-orient": {"enabled": True},
            "grayscale": {"enabled": True},
        },
        "network_input": {"training_input_size": {"width": 224, "height": 224}},
    }
    envelope = _described_envelope(
        "classifier-linear-input@v1",
        2048,
        preprocessing,
        "ResNetForClassificationOnnx",
    )
    gw = _gateway("my-classifier/1", "feature_vector", envelope)

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": "my-classifier/1",
            "api_key": "k",
            "image": _image(),
            "disable_preproc_grayscale": True,
        },
    )

    assert r.status_code == 200, r.text
    info = r.json()["embedding_info"]
    assert info["preprocessing"] == {
        **preprocessing,
        "overrides": {**NO_OVERRIDES, "disable_preproc_grayscale": True},
    }
    # Golden id of test_translation_embeddings.py, equal to the legacy server's.
    assert info["space_id"] == (
        "b4e7f1b4acc1340e8bca7fcb58a13b61f404df9da5c8f30ef11645d78b1c84d0"
    )


def test_clip_returns_described_image_embeddings_from_its_plain_registration(
    legacy_client, fake_stat
):
    gw = _clip_gateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={"model_id": "clip/ViT-B-16", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 200, r.text
    info = r.json()["embedding_info"]
    assert info["feature_definition"] == "clip-image-embeds@v1"
    assert info["model_id"] == "clip/ViT-B-16"
    assert info["preprocessing"] == CLIP_PREPROCESSING
    assert info["space_id"] == CLIP_SPACE_ID
    assert info["backend"] == "ClipOnnx"
    assert ("ensure_loaded", "clip/ViT-B-16", "k") in gw.calls
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[1:4] == (
        "clip/ViT-B-16",
        "embed_images_with_info",
        {"output_type": "feature_vector"},
    )


def test_clip_embed_image_keeps_its_compact_embeddings(legacy_client, fake_stat):
    gw = _clip_gateway()

    r = legacy_client(gw).post(
        "/clip/embed_image", json={"image": _image(), "api_key": "k"}
    )

    assert r.status_code == 200, r.text
    assert r.json()["embeddings"] == [[0.5, 0.5]]
    assert "embedding_info" not in r.json()
    infer_call = next(c for c in gw.calls if c[0] == "infer")
    assert infer_call[1:3] == ("clip/ViT-B-16", "embed_images")


def test_clip_without_described_embeddings_is_refused(legacy_client, fake_stat):
    gw = _clip_gateway(actions=("embed_images", "embed_text"))

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={"model_id": "clip/ViT-B-16", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 501
    assert r.json() == {"message": EMBEDDINGS_UNSUPPORTED}
    assert [c for c in gw.calls if c[0] == "infer"] == []


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_perception_encoder_is_refused_before_loading(
    legacy_client, fake_stat, output_type
):
    gw = FakeGateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={
            "model_id": PE_MODEL_ID,
            "api_key": "k",
            "image": _image(),
            "output_type": output_type,
        },
    )

    assert r.status_code == 501
    assert r.json() == {"message": EMBEDDINGS_UNSUPPORTED}
    assert r.headers["X-Model-Id"] == f"{PE_MODEL_ID}:{_registration(output_type)}"
    assert gw.calls == []


def _registration(output_type):
    return _instance(output_type).replace(";", ":")


@pytest.mark.parametrize(
    "model_id,flag",
    [
        ("clip/ViT-B-16", "CORE_MODELS_ENABLED"),
        ("clip/ViT-B-16", "CORE_MODEL_CLIP_ENABLED"),
        (PE_MODEL_ID, "CORE_MODEL_PE_ENABLED"),
    ],
)
def test_disabled_text_image_models_answer_as_unregistered_in_legacy(
    legacy_client, fake_stat, monkeypatch, model_id, flag
):
    from inference_server import configuration

    gw = _clip_gateway()
    client = legacy_client(gw)
    monkeypatch.setattr(configuration, flag, False)

    r = client.post(
        "/infer/embeddings",
        json={"model_id": model_id, "api_key": "k", "image": _image()},
    )

    assert r.status_code == 500
    assert r.json() == {"message": "Service misconfiguration."}
    assert gw.calls == []


def test_disabled_pretrained_embedding_models_answer_as_unregistered_in_legacy(
    legacy_client, fake_stat, monkeypatch
):
    monkeypatch.setattr(
        "inference_server.legacy.bridge.CORE_MODEL_PRETRAINED_IMAGE_EMBEDDINGS_ENABLED",
        False,
    )
    fake_stat["dinov2/small"] = ("embedding", "embed_images")
    gw = _pretrained_gateway()

    r = legacy_client(gw).post(
        "/infer/embeddings",
        json={"model_id": "dinov2/small", "api_key": "k", "image": _image()},
    )

    assert r.status_code == 500
    assert r.json() == {"message": "Service misconfiguration."}
    assert gw.calls == []
