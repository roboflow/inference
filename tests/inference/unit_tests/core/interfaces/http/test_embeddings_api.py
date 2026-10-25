from unittest.mock import MagicMock

import pytest
from starlette.testclient import TestClient

from inference.core.entities.responses.embeddings import ImageEmbeddingResponse
from inference.core.models.embeddings import make_embedding_info


@pytest.mark.parametrize("query_auth", [False, True])
@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_embedding_endpoint_negotiates_capability_and_preserves_auth(
    monkeypatch, query_auth, output_type
):
    from inference.core.interfaces.http import http_api

    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    manager = MagicMock()
    manager.pingback = None
    manager.num_errors = 0
    manager.infer_from_request_sync.return_value = ImageEmbeddingResponse(
        embeddings=[[2.0, 3.0]],
        embedding_info=make_embedding_info(
            "classifiers/4",
            {
                "feature_definition": "classifier-linear-input@v1",
                "normalization": "none",
            },
            {},
            "onnx",
            "float32",
            2,
        ),
    )
    interface = http_api.HttpInterface(model_manager=manager)
    with TestClient(interface.app) as client:
        payload = {
            "model_id": "resnet101",
            "output_type": output_type,
            "image": {"type": "url", "value": "https://example.com/image.jpg"},
        }
        if not query_auth:
            payload["api_key"] = "key"
        result = client.post(
            "/infer/embeddings",
            params={"api_key": "key"} if query_auth else None,
            json=payload,
        )
    assert result.status_code == 200, result.text
    assert result.json()["embeddings"] == [[2.0, 3.0]]
    assert result.json()["embedding_info"]["model_id"] == "classifiers/4"
    manager.add_model.assert_called_once_with(
        "classifiers/4",
        "key",
        model_id_alias="resnet101",
        countinference=None,
        service_secret=None,
        required_capabilities=["image_embeddings"],
        output_type=output_type,
    )
    assert manager.infer_from_request_sync.call_args.args[
        0
    ] == "resnet101:capabilities=image_embeddings" + (
        ":output_type=logits" if output_type == "logits" else ""
    )

    assert manager.infer_from_request_sync.call_args.args[1].output_type == output_type
