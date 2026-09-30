import base64
import io
import json
import time
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from PIL import Image

from inference_server.gateway import ModelManagerGateway


def _jpeg_b64(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _det():
    return SimpleNamespace(
        xyxy=np.array([[1, 1, 3, 5]], dtype=float),
        confidence=np.array([0.9]),
        class_id=np.array([0]),
    )


class _LoadingManager:
    """In-process manager double: loads on demand, serves one detection."""

    executor = None

    def __init__(self) -> None:
        self.loaded: set[str] = set()

    def __contains__(self, key: str) -> bool:
        return key in self.loaded

    def is_healthy(self, key: str) -> bool:
        return True

    def load(self, key: str, api_key: str, **kwargs) -> None:
        time.sleep(0.01)
        self.loaded.add(key)

    def unload(self, key: str) -> None:
        self.loaded.discard(key)

    def shutdown(self) -> None: ...

    def stats(self) -> dict:
        return {
            "models": [
                {"model_id": key, "class_names": ["cat"], "actions": {"infer": {}}}
                for key in self.loaded
            ]
        }

    async def process_async(self, key, action=None, **kwargs):
        return _det()


def _infer(client, **extra_headers):
    return client.post(
        "/infer/object_detection",
        json={
            "model_id": "ds/1",
            "api_key": "k",
            "image": {"type": "base64", "value": _jpeg_b64()},
        },
        headers=extra_headers,
    )


def test_first_request_reports_cold_start_and_second_is_warm(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_LoadingManager()))

    cold = _infer(client)
    warm = _infer(client)

    assert cold.status_code == 200, cold.text
    assert cold.headers["x-model-id"] == "ds/1"
    assert cold.headers["x-inference-engine"] == "inference-models"
    assert cold.headers["x-model-cold-start"] == "true"
    assert cold.headers["x-model-cold-start-count"] == "1"
    load_time = float(cold.headers["x-model-load-time"])
    assert load_time > 0
    details = json.loads(cold.headers["x-model-load-details"])
    assert details == [{"m": "ds/1", "t": load_time}]
    assert UUID(cold.headers["x-request-id"]).version == 4

    assert warm.status_code == 200, warm.text
    assert warm.headers["x-model-id"] == "ds/1"
    assert warm.headers["x-inference-engine"] == "inference-models"
    assert warm.headers["x-model-cold-start"] == "false"
    assert warm.headers["x-model-cold-start-count"] == "0"
    assert "x-model-load-time" not in warm.headers
    assert "x-model-load-details" not in warm.headers
    assert warm.headers["x-request-id"] != cold.headers["x-request-id"]


def test_legacy_project_version_route_carries_model_headers(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_LoadingManager()))
    buf = io.BytesIO()
    Image.new("RGB", (8, 6)).save(buf, format="JPEG")

    response = client.post(
        "/ds/1?api_key=k",
        content=base64.b64encode(buf.getvalue()),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert response.status_code == 200, response.text
    assert response.headers["x-model-id"] == "ds/1"
    assert response.headers["x-model-cold-start"] == "true"
    assert response.headers["x-model-cold-start-count"] == "1"
    assert float(response.headers["x-model-load-time"]) > 0


@pytest.mark.parametrize(
    "incoming", ["3f2b8c1e-5d4a-4b6e-9c7d-1a2b3c4d5e6f", "client-trace-42"]
)
def test_incoming_request_id_is_echoed(legacy_client, fake_stat, incoming):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_LoadingManager()))

    response = _infer(client, **{"X-Request-ID": incoming})

    assert response.headers["x-request-id"] == incoming


def test_model_headers_are_exposed_to_cors_clients(legacy_client, fake_stat):
    fake_stat["ds/1"] = ("object-detection", "infer")
    client = legacy_client(ModelManagerGateway(_LoadingManager()))

    response = _infer(client, Origin="https://app.example")

    exposed = {
        name.strip().lower()
        for name in response.headers["access-control-expose-headers"].split(",")
    }
    assert {
        "x-model-cold-start",
        "x-model-cold-start-count",
        "x-model-load-time",
        "x-model-load-details",
        "x-model-id",
    } <= exposed


def test_route_without_models_reports_no_cold_start(legacy_client):
    client = legacy_client(ModelManagerGateway(_LoadingManager()))

    response = client.get("/v2/server/health")

    assert response.headers["x-inference-engine"] == "inference-models"
    assert response.headers["x-model-cold-start"] == "false"
    assert response.headers["x-model-cold-start-count"] == "0"
    assert "x-model-id" not in response.headers
    assert UUID(response.headers["x-request-id"]).version == 4
