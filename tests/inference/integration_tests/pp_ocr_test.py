import os
from copy import deepcopy

import pytest
import requests

from tests.inference.integration_tests.conftest import (
    api_key_auth_headers,
    without_api_key_in_header_mode,
)

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

TESTS = [
    {
        "description": "PP-OCR default (small-small)",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/swift.png",
            }
        },
    },
    {
        "description": "PP-OCR explicit stages",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/swift.png",
            },
            "text_detection": "small",
            "text_recognition": "small",
        },
    },
    {
        "description": "PP-OCR batch input",
        "payload": {
            "image": [
                {
                    "type": "url",
                    "value": "https://media.roboflow.com/swift.png",
                },
                {
                    "type": "url",
                    "value": "https://media.roboflow.com/swift.png",
                },
            ]
        },
    },
    {
        "description": "PP-OCR detect-only",
        "payload": {
            "image": {
                "type": "url",
                "value": "https://media.roboflow.com/swift.png",
            },
            "text_recognition": "none",
        },
    },
]


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_PP_OCR_TEST", False))
    or os.getenv("USE_INFERENCE_MODELS", "false").lower() != "true",
    reason="Skipping PP-OCR test (requires USE_INFERENCE_MODELS=true)",
)
@pytest.mark.parametrize("test", TESTS, ids=lambda t: t["description"])
def test_pp_ocr(test, auth_mode, clean_loaded_models_fixture):
    payload = deepcopy(test["payload"])
    payload["api_key"] = api_key
    response = requests.post(
        f"{base_url}:{port}/ocr/pp-ocr",
        json=without_api_key_in_header_mode(auth_mode, payload),
        headers=api_key_auth_headers(auth_mode, api_key),
    )
    response.raise_for_status()
    data = response.json()
    batch_input = type(test["payload"]["image"]) is list
    results = data if batch_input else [data]
    if batch_input:
        assert len(results) == len(test["payload"]["image"])
    for result in results:
        assert "result" in result
        assert "predictions" in result
        assert isinstance(result["result"], str)
        if test["payload"].get("text_recognition") == "none":
            assert result["result"] == ""
            assert len(result["predictions"]) > 0
        else:
            assert len(result["result"]) > 0
