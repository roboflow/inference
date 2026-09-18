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
    monkeypatch.setattr(http_api, "ACTION_RECOGNITION_ENABLED", True)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    return http_api.HttpInterface(model_manager=MagicMock()).app.openapi()


def test_openapi_exposes_optional_string_selectors_in_body_and_legacy_query(
    openapi_schema,
):
    schemas = openapi_schema["components"]["schemas"]
    legacy = openapi_schema["paths"]["/{dataset_id}/{version_id}"]["post"]
    parameters = {parameter["name"]: parameter for parameter in legacy["parameters"]}

    for selector in ("model_package_id", "backend", "quantization"):
        parameter = parameters[selector]
        assert parameter["in"] == "query"
        assert parameter["required"] is False
        assert {"type": "string", "minLength": 1} in parameter["schema"]["anyOf"]
        assert {"type": "null"} in parameter["schema"]["anyOf"]
        for request in (
            "ObjectDetectionInferenceRequest",
            "AddModelRequest",
            "ClearModelRequest",
            "ActionRecognitionInferenceRequest",
        ):
            schema = schemas[request]
            field = schema["properties"][selector]
            assert selector not in schema.get("required", [])
            assert {"type": "string", "minLength": 1} in field["anyOf"]
            assert {"type": "null"} in field["anyOf"]


@pytest.mark.parametrize(
    "schema_name,expected_fields",
    [
        (
            "AddModelRequest",
            {
                "model_id",
                "model_type",
                "api_key",
                "model_package_id",
                "backend",
                "quantization",
            },
        ),
        (
            "ClearModelRequest",
            {"model_id", "api_key", "model_package_id", "backend", "quantization"},
        ),
    ],
)
def test_model_management_schemas_only_include_model_management_fields(
    openapi_schema, schema_name, expected_fields
):
    assert (
        set(openapi_schema["components"]["schemas"][schema_name]["properties"])
        == expected_fields
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
        ("/infer/action_recognition", "post"),
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
