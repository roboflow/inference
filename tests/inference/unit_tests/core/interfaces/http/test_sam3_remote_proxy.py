"""SAM 3 routes forward requests to the Roboflow inference proxy when
``SAM3_EXEC_MODE=remote``."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from inference.core.interfaces.http import http_api

IMAGE = {"type": "base64", "value": "image"}
RLE_MASK = {"size": [8, 8], "counts": "52000N`0"}
POLYGON_MASK = [[[1, 1], [3, 1], [3, 5]]]


class _RecordingPost:
    def __init__(self, body: dict) -> None:
        self.body = body
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return SimpleNamespace(status_code=200, url=url, json=lambda: self.body)


@pytest.fixture
def remote_client(monkeypatch):
    for name in ("CORE_MODELS_ENABLED", "CORE_MODEL_SAM3_ENABLED"):
        monkeypatch.setattr(http_api, name, True)
    monkeypatch.setattr(http_api, "SAM3_EXEC_MODE", "remote")
    monkeypatch.setattr(http_api, "API_BASE_URL", "https://api.example.test")
    monkeypatch.setattr(http_api, "ROBOFLOW_INTERNAL_SERVICE_NAME", None)
    monkeypatch.setattr(http_api, "ROBOFLOW_INTERNAL_SERVICE_SECRET", None)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    manager = MagicMock()
    manager.pingback = None
    client = TestClient(http_api.HttpInterface(model_manager=manager).app)

    def install_proxy(body: dict) -> _RecordingPost:
        post = _RecordingPost(body=body)
        monkeypatch.setattr(http_api.requests, "post", post)
        return post

    return client, install_proxy


def _visual_body(masks) -> dict:
    return {"predictions": [{"masks": masks, "confidence": 0.9}], "time": 0.1}


def _concept_body(masks) -> dict:
    return {
        "prompt_results": [
            {
                "prompt_index": 0,
                "echo": {"prompt_index": 0, "type": "text", "text": "cat"},
                "predictions": [{"masks": masks, "confidence": 0.8}],
            }
        ],
        "time": 0.1,
    }


def test_visual_segment_forwards_the_rle_format(remote_client) -> None:
    client, install_proxy = remote_client
    proxy = install_proxy(body=_visual_body(masks=RLE_MASK))

    response = client.post(
        "/sam3/visual_segment",
        json={
            "image": IMAGE,
            "prompts": [{"points": [{"x": 5, "y": 6, "positive": True}]}],
            "format": "rle",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["predictions"][0]["masks"] == RLE_MASK
    (call,) = proxy.calls
    assert call["url"].startswith("https://api.example.test/inferenceproxy/sam3-pvs")
    assert call["json"]["format"] == "rle"


@pytest.mark.parametrize("response_format", ["json", "binary"])
def test_visual_segment_does_not_forward_other_formats(
    remote_client, response_format
) -> None:
    client, install_proxy = remote_client
    proxy = install_proxy(body=_visual_body(masks=POLYGON_MASK))

    response = client.post(
        "/sam3/visual_segment",
        json={"image": IMAGE, "format": response_format},
    )

    assert response.status_code == 200, response.text
    (call,) = proxy.calls
    assert "format" not in call["json"]


def test_concept_segment_forwards_the_rle_format_and_nms_threshold(
    remote_client,
) -> None:
    client, install_proxy = remote_client
    proxy = install_proxy(body=_concept_body(masks=RLE_MASK))

    response = client.post(
        "/sam3/concept_segment",
        json={
            "image": IMAGE,
            "prompts": [{"type": "text", "text": "cat"}],
            "format": "rle",
            "nms_iou_threshold": 0.5,
        },
    )

    assert response.status_code == 200, response.text
    prediction = response.json()["prompt_results"][0]["predictions"][0]
    assert prediction["masks"] == RLE_MASK
    (call,) = proxy.calls
    assert call["url"].startswith("https://api.example.test/inferenceproxy/seg-preview")
    assert call["json"]["format"] == "rle"
    assert call["json"]["nms_iou_threshold"] == 0.5


def test_concept_segment_keeps_the_default_payload(remote_client) -> None:
    client, install_proxy = remote_client
    proxy = install_proxy(body=_concept_body(masks=POLYGON_MASK))

    response = client.post(
        "/sam3/concept_segment",
        json={"image": IMAGE, "prompts": [{"type": "text", "text": "cat"}]},
    )

    assert response.status_code == 200, response.text
    (call,) = proxy.calls
    assert set(call["json"]) == {
        "image",
        "prompts",
        "output_prob_thresh",
        "source",
        "source_info",
    }
