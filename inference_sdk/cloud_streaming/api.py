"""The Cloud Streaming v1 REST API, scoped to one workspace."""

import time
from typing import Any, Dict, List, Optional

import requests

from inference_sdk.config import RF_API_BASE_URL

REQUEST_TIMEOUT_S = 30.0
# Requests that carry an Idempotency-Key are safe to resend with the same key.
IDEMPOTENT_ATTEMPTS = 3


class CloudStreamingError(Exception):
    """A Cloud Streaming request failed, or a stream ended with an error."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class CloudStreamingAPI:
    def __init__(
        self,
        api_key: str,
        workspace: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self._api_url = (api_url or RF_API_BASE_URL).rstrip("/")
        self._workspace = workspace
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {api_key}"

    @property
    def workspace(self) -> str:
        if self._workspace is None:
            self._workspace = self._request("GET", "/")["workspace"]
        return self._workspace

    def list_sources(self) -> List[Dict[str, Any]]:
        return self._workspace_request("GET", "/video-sources/v1")["sources"]

    def create_session(self, name: str, idempotency_key: str) -> Dict[str, Any]:
        return self._workspace_request(
            "POST",
            "/video-sources/v1/temporary",
            json={"name": name},
            idempotency_key=idempotency_key,
        )

    def stop_session(self, source_id: str) -> None:
        self._workspace_request("POST", f"/video-sources/v1/{source_id}/stop-session")

    def start_job(
        self, source_id: str, job: Dict[str, Any], idempotency_key: str
    ) -> Dict[str, Any]:
        return self._workspace_request(
            "POST",
            f"/video-sources/v1/{source_id}/jobs",
            json=job,
            idempotency_key=idempotency_key,
        )["job"]

    def get_job(self, job_id: str) -> Dict[str, Any]:
        return self._workspace_request("GET", f"/video-jobs/v1/{job_id}")["job"]

    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        return self._workspace_request("POST", f"/video-jobs/v1/{job_id}/cancel")["job"]

    def hold_preview(self, job_id: str) -> bool:
        return self._workspace_request("POST", f"/video-jobs/v1/{job_id}/hold-preview")[
            "held"
        ]

    def watch(self, job_id: str, output: Optional[str] = None) -> Dict[str, Any]:
        return self._workspace_request(
            "POST", f"/video-jobs/v1/{job_id}/watch", json={"output": output}
        )

    def subscribe_results(self, job_id: str) -> Dict[str, Any]:
        return self._workspace_request(
            "POST", f"/video-jobs/v1/{job_id}/results/subscribe"
        )

    def _workspace_request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        return self._request(method, f"/{self.workspace}{path}", **kwargs)

    def _request(
        self,
        method: str,
        path: str,
        json: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
        attempts = IDEMPOTENT_ATTEMPTS if idempotency_key else 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.request(
                    method,
                    self._api_url + path,
                    json=json,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT_S,
                )
            except requests.RequestException as error:
                if attempt == attempts:
                    raise CloudStreamingError(
                        f"{method} {path} failed: {type(error).__name__}"
                    ) from error
                time.sleep(attempt)
                continue
            if response.status_code >= 500 and attempt < attempts:
                time.sleep(attempt)
                continue
            return _json_or_raise(response, f"{method} {path}")


def _json_or_raise(response: requests.Response, request: str) -> Dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.ok and isinstance(body, dict):
        return body
    raise CloudStreamingError(
        f"{request} failed ({response.status_code}): {_error_message(body, response)}",
        status=response.status_code,
    )


def _error_message(body: Any, response: requests.Response) -> str:
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        error = error.get("message")
    if isinstance(error, str) and error:
        return error
    return response.reason or "unexpected response"
