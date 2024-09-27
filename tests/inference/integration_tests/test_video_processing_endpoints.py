import os

import requests

API_KEY = os.environ.get("API_KEY")


def test_list_pipeline_endpoint_being_enabled(server_url: str) -> None:
    # when
    response = requests.get(
        f"{server_url}/inference_pipelines/list",
        json={
            "api_key": API_KEY,
        }
    )

    # then
    response.raise_for_status()
