import os

import requests

from tests.inference.integration_tests.conftest import (
    api_key_auth_headers,
    without_api_key_in_header_mode,
)

API_KEY = os.environ.get("API_KEY")


def test_list_pipeline_endpoint_being_enabled(server_url: str, auth_mode: str) -> None:
    # when
    response = requests.get(
        f"{server_url}/inference_pipelines/list",
        json=without_api_key_in_header_mode(
            auth_mode,
            {
                "api_key": API_KEY,
            },
        ),
        headers=api_key_auth_headers(auth_mode, API_KEY),
    )

    # then
    response.raise_for_status()
