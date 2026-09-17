from unittest.mock import MagicMock

import pytest

from inference.core.constants import MODEL_SELECTION_HEADER
from inference.core.interfaces.http import http_api


@pytest.fixture
def openapi_schema(monkeypatch):
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "OFFLINE_MODE", True)
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    return http_api.HttpInterface(model_manager=MagicMock()).app.openapi()


def test_openapi_describes_selectors_in_body_and_legacy_query(openapi_schema):
    schemas = openapi_schema["components"]["schemas"]
    legacy = openapi_schema["paths"]["/{dataset_id}/{version_id}"]["post"]
    parameters = {parameter["name"]: parameter for parameter in legacy["parameters"]}

    for selector in ("model_package_id", "backend", "quantization"):
        description = schemas["ObjectDetectionInferenceRequest"]["properties"][
            selector
        ]["description"]
        assert "USE_INFERENCE_MODELS=true" in description
        assert parameters[selector]["description"] == description
        for request in ("AddModelRequest", "ClearModelRequest"):
            assert (
                schemas[request]["properties"][selector]["description"] == description
            )


@pytest.mark.parametrize(
    "path,method",
    [
        ("/model/add", "post"),
        ("/infer/object_detection", "post"),
        ("/infer/instance_segmentation", "post"),
        ("/infer/semantic_segmentation", "post"),
        ("/infer/classification", "post"),
        ("/infer/keypoints_detection", "post"),
        ("/{dataset_id}/{version_id}", "get"),
        ("/{dataset_id}/{version_id}", "post"),
    ],
)
def test_openapi_documents_selection_acknowledgment(openapi_schema, path, method):
    response = openapi_schema["paths"][path][method]["responses"]["200"]
    assert response["headers"][MODEL_SELECTION_HEADER]["schema"] == {
        "type": "string",
        "enum": ["applied"],
    }
    assert "application/json" in response["content"]
