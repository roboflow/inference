import os

import pytest
import requests

from tests.inference.integration_tests.conftest import (
    api_key_auth_headers,
    without_api_key_in_header_mode,
)
from tests.inference.integration_tests.regression_test import bool_env

api_key = os.environ.get("API_KEY")


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_GLM_OCR_TEST", False))
    or os.getenv("USE_INFERENCE_MODELS", "false").lower() != "true",
    reason="Skipping GLM OCR test (requires USE_INFERENCE_MODELS=true)",
)
def test_glm_ocr_inference(
    server_url: str, auth_mode: str, clean_loaded_models_every_test_fixture
) -> None:
    # given
    payload = {
        "api_key": api_key,
        "image": {
            "type": "url",
            "value": "https://media.roboflow.com/serial_number.png",
        },
        "prompt": "Text Recognition:",
        "model_id": "glm-ocr",
    }

    # when
    response = requests.post(
        f"{server_url}/infer/lmm",
        json=without_api_key_in_header_mode(auth_mode, payload),
        headers=api_key_auth_headers(auth_mode, api_key),
    )

    # then
    response.raise_for_status()
    data = response.json()
    assert len(data["response"]) > 0, "Expected non empty generation"
