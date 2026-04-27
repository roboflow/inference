import json
from typing import Dict

import backoff
import requests  # type: ignore
from requests import Response, Timeout

from inference_cli.lib.enterprise.inference_compiler.constants import (
    HTTP_CODES_TO_RETRY,
)
from inference_cli.lib.enterprise.inference_compiler.errors import (
    RequestError,
    RetryError,
)


@backoff.on_exception(
    backoff.fibo,
    exception=RetryError,
    max_tries=3,
    max_value=5,
)
def upload_file_to_cloud(
    file_path: str,
    url: str,
    headers: Dict[str, str],
) -> None:
    try:
        with open(file_path, "rb") as f:
            response = requests.put(url, headers=headers, data=f)
            response.raise_for_status()
    except (ConnectionError, Timeout, requests.exceptions.ConnectionError):
        raise RetryError("Connectivity error")
    handle_response_errors(response=response)


def handle_response_errors(response: Response) -> None:
    if response.status_code in HTTP_CODES_TO_RETRY:
        raise RetryError(f"Service returned {response.status_code}")
    if response.status_code >= 400:
        response_payload = get_error_response_payload(response=response)
        raise RequestError(
            message=f"RF API responded with status: {response.status_code} - error message: {response_payload}",
            status_code=response.status_code,
        )


def get_error_response_payload(response: Response) -> str:
    try:
        return json.dumps(response.json(), indent=4)
    except ValueError:
        return response.text  # type: ignore
