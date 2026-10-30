import base64
import io

import pytest
from PIL import Image

from inference_server import configuration, platform_http
from inference_server.legacy import bridge as bridge_mod
from inference_server.legacy.bridge import LegacyModelBridge
from inference_server.legacy.errors import (
    LegacyHTTPError,
    NOT_FOUND_MESSAGE,
    REGISTRY_REQUEST_FAILED_MESSAGE,
    REGISTRY_UNREACHABLE_MESSAGE,
    SERVICE_MISCONFIGURATION_MESSAGE,
    UNAUTHORIZED_MESSAGE,
)
from tests.unit_tests.legacy.conftest import FakeGateway

INFER_ROUTES = (
    "/infer/object_detection",
    "/infer/instance_segmentation",
    "/infer/semantic_segmentation",
    "/infer/classification",
    "/infer/keypoints_detection",
)
WORKSPACE_URL = f"{configuration.API_BASE_URL}/?api_key=k&nocache=true"
PROJECT_URL = f"{configuration.API_BASE_URL}/ws/ds?api_key=k&nocache=true"


def _jpeg_b64(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _image():
    return {"type": "base64", "value": _jpeg_b64()}


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakePlatform:
    def __init__(self):
        self.calls = []
        self.workspace = (200, {"workspace": "ws"})
        self.project = (200, {"project": {"type": "object-detection"}})

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        if url.startswith(f"{configuration.API_BASE_URL}/?"):
            return _Response(*self.workspace)
        return _Response(*self.project)


@pytest.fixture(autouse=True)
def _reset_stub_cache():
    bridge_mod._reset_stub_cache_for_tests()
    yield
    bridge_mod._reset_stub_cache_for_tests()


@pytest.fixture
def fake_platform(monkeypatch):
    platform = FakePlatform()
    monkeypatch.setattr(platform_http, "_platform_request", platform.request)
    return platform


@pytest.mark.parametrize("path", INFER_ROUTES)
def test_stub_model_returns_stub_response_without_load(
    legacy_client, fake_stat, fake_platform, path
):
    gw = FakeGateway()
    r = legacy_client(gw).post(
        path, json={"model_id": "ds/0", "api_key": "k", "image": _image()}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"is_stub", "model_id", "task_type", "time"}
    assert body["is_stub"] is True
    assert body["model_id"] == "ds/0"
    assert body["task_type"] == "object-detection"
    assert gw.calls == []
    assert fake_platform.calls == [("get", WORKSPACE_URL), ("get", PROJECT_URL)]


def test_stub_model_visualization_is_a_black_jpeg(
    legacy_client, fake_stat, fake_platform
):
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={
            "model_id": "ds/0",
            "api_key": "k",
            "image": _image(),
            "visualize_predictions": True,
        },
    )
    assert r.status_code == 200, r.text
    image = Image.open(io.BytesIO(base64.b64decode(r.json()["visualization"])))
    assert image.size == (128, 128)


def test_stub_model_answers_the_catch_all(legacy_client, fake_stat, fake_platform):
    fake_platform.project = (200, {"project": {"type": "classification"}})
    gw = FakeGateway()
    r = legacy_client(gw).post(
        "/ds/0?api_key=k",
        content=base64.b64encode(b"not-decoded"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["is_stub"] is True and body["model_id"] == "ds/0"
    assert body["task_type"] == "classification"
    assert gw.calls == []


def test_stub_model_catch_all_image_format_returns_jpeg(
    legacy_client, fake_stat, fake_platform
):
    r = legacy_client(FakeGateway()).post(
        "/ds/0?api_key=k&format=image",
        content=base64.b64encode(b"not-decoded"),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(r.content)).size == (128, 128)


def test_stub_model_without_api_key_is_400(
    legacy_client, fake_stat, fake_platform, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.common.DEFAULT_API_KEY", None)
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection", json={"model_id": "ds/0", "image": _image()}
    )
    assert r.status_code == 400
    assert r.json()["message"] == bridge_mod.MISSING_API_KEY_MESSAGE
    assert fake_platform.calls == []


def test_stub_model_defaults_unknown_project_type_to_object_detection(
    legacy_client, fake_stat, fake_platform
):
    fake_platform.project = (200, {"project": {}})
    r = legacy_client(FakeGateway()).post(
        "/infer/classification",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )
    assert r.status_code == 200, r.text
    assert r.json()["task_type"] == "object-detection"


def test_stub_model_second_call_is_served_from_cache(
    legacy_client, fake_stat, fake_platform
):
    client = legacy_client(FakeGateway())
    payload = {"model_id": "ds/0", "api_key": "k", "image": _image()}
    assert client.post("/infer/object_detection", json=payload).status_code == 200
    assert client.post("/infer/object_detection", json=payload).status_code == 200
    assert len(fake_platform.calls) == 2


def test_stub_model_cache_is_keyed_by_api_key(legacy_client, fake_stat, fake_platform):
    client = legacy_client(FakeGateway())
    client.post(
        "/infer/object_detection",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )
    client.post(
        "/infer/object_detection",
        json={"model_id": "ds/0", "api_key": "other", "image": _image()},
    )
    assert len(fake_platform.calls) == 4


def test_stub_model_offline_is_503_without_platform_calls(
    legacy_client, fake_stat, fake_platform, monkeypatch
):
    monkeypatch.setattr("inference_server.legacy.bridge.LEGACY_OFFLINE_MODE", True)
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )
    assert r.status_code == 503
    assert r.json()["message"] == REGISTRY_UNREACHABLE_MESSAGE
    assert fake_platform.calls == []


def test_stub_model_project_type_without_legacy_stub_is_500(
    legacy_client, fake_stat, fake_platform
):
    fake_platform.project = (200, {"project": {"type": "semantic-segmentation"}})
    r = legacy_client(FakeGateway()).post(
        "/infer/semantic_segmentation",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )
    assert r.status_code == 500
    assert r.json()["message"] == SERVICE_MISCONFIGURATION_MESSAGE


@pytest.mark.parametrize(
    "status_code, expected_status, expected_message",
    [
        (401, 401, UNAUTHORIZED_MESSAGE),
        (404, 404, NOT_FOUND_MESSAGE),
        (500, 502, REGISTRY_REQUEST_FAILED_MESSAGE),
    ],
)
def test_stub_model_maps_platform_errors(
    legacy_client,
    fake_stat,
    fake_platform,
    status_code,
    expected_status,
    expected_message,
):
    fake_platform.project = (status_code, {})
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )
    assert r.status_code == expected_status
    assert r.json()["message"] == expected_message


@pytest.mark.parametrize("model_id", ["a?b/0", "../x/0", "a%2Fb/0", "a b/0"])
def test_stub_model_with_malformed_dataset_id_is_404_without_platform_call(
    legacy_client, fake_stat, fake_platform, model_id
):
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": model_id, "api_key": "k", "image": _image()},
    )
    assert r.status_code == 404
    assert r.json()["message"] == bridge_mod.NOT_FOUND_MESSAGE
    assert fake_platform.calls == []


def test_stub_model_with_valid_dataset_id_resolves(
    legacy_client, fake_stat, fake_platform
):
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds-1_x/0", "api_key": "k", "image": _image()},
    )
    assert r.status_code == 200, r.text
    assert fake_platform.calls[-1] == (
        "get",
        f"{configuration.API_BASE_URL}/ws/ds-1_x?api_key=k&nocache=true",
    )


def test_stub_model_without_workspace_is_502(legacy_client, fake_stat, fake_platform):
    fake_platform.workspace = (200, {})
    r = legacy_client(FakeGateway()).post(
        "/infer/object_detection",
        json={"model_id": "ds/0", "api_key": "k", "image": _image()},
    )
    assert r.status_code == 502
    assert r.json()["message"] == REGISTRY_REQUEST_FAILED_MESSAGE
    assert fake_platform.calls == [("get", WORKSPACE_URL)]


@pytest.mark.asyncio
async def test_resolve_stub_route_skips_registry_and_gateway(fake_stat, fake_platform):
    gw = FakeGateway()
    bridge = LegacyModelBridge(gw)
    route = await bridge.resolve("ds/0", "k")
    assert route.is_stub is True
    assert (route.task_type, route.action) == ("object-detection", "infer")
    assert route.model_architecture == "stub"
    assert gw.calls == [] and fake_stat == {}


@pytest.mark.asyncio
async def test_resolve_stub_route_with_empty_api_key_is_502(fake_stat, fake_platform):
    with pytest.raises(LegacyHTTPError) as exc:
        await LegacyModelBridge(FakeGateway()).resolve("ds/0", "")
    assert exc.value.status_code == 502
    assert fake_platform.calls == []
