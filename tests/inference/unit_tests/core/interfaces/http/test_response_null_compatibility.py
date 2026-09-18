from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from starlette.testclient import TestClient

from inference.core.entities.responses.clip import ClipCompareResponse
from inference.core.entities.responses.perception_encoder import (
    PerceptionEncoderCompareResponse,
)
from inference.core.entities.responses.sam3 import (
    Sam3PromptEcho,
    Sam3PromptResult,
    Sam3SegmentationResponse,
)
from inference.core.interfaces.http import http_api


@pytest.fixture
def client_and_manager(monkeypatch):
    for name in (
        "CORE_MODELS_ENABLED",
        "CORE_MODEL_CLIP_ENABLED",
        "CORE_MODEL_PE_ENABLED",
        "CORE_MODEL_SAM3_ENABLED",
        "DEPTH_ESTIMATION_ENABLED",
    ):
        monkeypatch.setattr(http_api, name, True)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    manager = MagicMock()
    manager.pingback = None
    return TestClient(http_api.HttpInterface(model_manager=manager).app), manager


@pytest.mark.parametrize(
    "endpoint,response_type",
    [
        ("/clip/compare", ClipCompareResponse),
        ("/perception_encoder/compare", PerceptionEncoderCompareResponse),
    ],
)
def test_embedding_comparison_keeps_existing_null_fields(
    client_and_manager, endpoint, response_type
):
    client, manager = client_and_manager
    manager.infer_from_request_sync.return_value = response_type(similarity=[0.5])

    response = client.post(
        endpoint,
        json={"subject": "cat", "subject_type": "text", "prompt": "animal"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    for field in ("parent_id", "time", "frame_id", "inference_id"):
        assert field in payload
        assert payload[field] is None


def test_sam3_keeps_null_prompt_echo_fields(client_and_manager):
    client, manager = client_and_manager
    manager.infer_from_request_sync.return_value = Sam3SegmentationResponse(
        prompt_results=[
            Sam3PromptResult(
                prompt_index=0,
                echo=Sam3PromptEcho(prompt_index=0, type="text", text="cat"),
                predictions=[],
            )
        ],
        time=0.1,
    )

    response = client.post(
        "/sam3/concept_segment",
        json={
            "image": {"type": "base64", "value": "image"},
            "prompts": [{"type": "text", "text": "cat"}],
        },
    )

    assert response.status_code == 200, response.text
    echo = response.json()["prompt_results"][0]["echo"]
    assert "num_boxes" in echo
    assert echo["num_boxes"] is None


@pytest.mark.parametrize(
    "endpoint",
    ["/infer/depth-estimation", "/infer/depth-estimation/test/1"],
)
def test_depth_estimation_keeps_null_image(client_and_manager, endpoint):
    client, manager = client_and_manager
    manager.infer_from_request_sync.return_value = SimpleNamespace(
        response={
            "normalized_depth": np.zeros((1, 1)),
            "image": SimpleNamespace(base64_image=None),
        }
    )

    response = client.post(
        endpoint,
        json={
            "model_id": "test/1",
            "image": {"type": "base64", "value": "image"},
            "depth_map_format": "json",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "image" in payload
    assert payload["image"] is None
