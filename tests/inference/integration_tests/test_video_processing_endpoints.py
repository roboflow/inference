import os

import pytest
import requests

from tests.inference.integration_tests.conftest import (
    api_key_auth_headers,
    without_api_key_in_header_mode,
)

API_KEY = os.environ.get("API_KEY")


@pytest.mark.skipif(
    not os.getenv("STREAM_API_KEY"),
    reason="Requires an explicitly enabled server and STREAM_API_KEY",
)
def test_list_pipeline_endpoint_being_enabled(server_url: str, auth_mode: str) -> None:
    stream_api_key = os.environ["STREAM_API_KEY"]
    # when
    response = requests.get(
        f"{server_url}/inference_pipelines/list",
        json=without_api_key_in_header_mode(
            auth_mode,
            {
                "api_key": API_KEY,
            },
        ),
        headers={
            **api_key_auth_headers(auth_mode, API_KEY),
            "X-Stream-API-Key": stream_api_key,
        },
        allow_redirects=False,
    )

    # then
    response.raise_for_status()
