import json
from typing import List, Optional, Union

import backoff
import requests
from requests import Response, Timeout

from inference_cli.lib.env import API_BASE_URL
from inference_cli.lib.roboflow_cloud.config import HTTP_CODES_TO_RETRY, REQUEST_TIMEOUT
from inference_cli.lib.roboflow_cloud.errors import (
    RetryError,
    RFAPICallError,
    UnauthorizedRequestError,
)


def ensure_api_key_is_set(api_key: Optional[str]) -> None:
    if api_key is None:
        raise UnauthorizedRequestError(
            "Request unauthorised. Are you sure you use valid Roboflow API key? "
            "See details here: https://docs.roboflow.com/api-reference/authentication and "
            "export key to `ROBOFLOW_API_KEY` environment variable"
        )


@backoff.on_exception(
    backoff.constant,
    exception=RetryError,
    max_tries=3,
    interval=1,
)
def get_workspace(api_key: str) -> str:
    try:
        response = requests.get(
            f"{API_BASE_URL}",
            params={"api_key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
    except (ConnectionError, Timeout):
        raise RetryError(
            f"Connectivity error. Try reaching Roboflow API in browser: {API_BASE_URL}"
        )
    handle_response_errors(response=response, operation_name="list batches")
    try:
        return response.json()["workspace"]
    except (KeyError, ValueError) as error:
        raise RFAPICallError("Could not decode Roboflow API response.") from error


def handle_response_errors(response: Response, operation_name: str) -> None:
    if response.status_code in HTTP_CODES_TO_RETRY:
        raise RetryError(
            f"Roboflow API returned invalid response code for {operation_name} operation "
            f"{response.status_code}."
        )
    if response.status_code == 401:
        raise UnauthorizedRequestError(
            f"Could not {operation_name}. Request unauthorised. Are you sure you use valid Roboflow API key? "
            "See details here: https://docs.roboflow.com/api-reference/authentication and "
            "export key to `ROBOFLOW_API_KEY` environment variable"
        )
    if response.status_code >= 400:
        response_payload = _get_response_payload(response=response)
        raise RFAPICallError(
            f"Roboflow API returned invalid response code for {operation_name} operation "
            f"{response.status_code}.\n\nResponse:\n{response_payload}"
        )


def _get_response_payload(response: Response) -> str:
    try:
        return json.dumps(response.json(), indent=4)
    except ValueError:
        return response.text


def prepare_status_type_emoji(status_type: str) -> str:
    if "error" in status_type.lower():
        return "🚨"
    return "🟢"


def read_jsonl_file(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines() if line.strip()]
