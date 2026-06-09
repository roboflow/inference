"""HTTP client for the vLLM sidecar (OpenAI-compatible API).

The vLLM container owns the GPU and exposes:
- `POST /v1/chat/completions` - generation (with `image_url` base64 data-URI
  content parts for multimodal input),
- `POST /v1/load_lora_adapter` / `POST /v1/unload_lora_adapter` - dynamic
  LoRA registration (requires `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` on the
  vLLM side),
- `GET /v1/models`, `GET /health`.

A single shared `requests.Session` with a large connection pool is used so
many concurrent uvicorn workers / threads can proxy requests without
exhausting sockets.
"""

from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter

from inference.core import logger
from inference.models.vllm_proxy.config import (
    get_vllm_base_url,
    get_vllm_request_timeout_s,
)
from inference.models.vllm_proxy.errors import VLLMConnectionError, VLLMHTTPError

CONNECTION_POOL_SIZE = 128

_CONNECTIVITY_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    ConnectionError,
)


def build_image_content_part(
    image_base64: str, mime_type: str = "image/png"
) -> Dict[str, Any]:
    """Builds an OpenAI-style `image_url` content part with a base64 data URI."""
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
    }


class VLLMClient:
    """Thin, typed client for the vLLM sidecar HTTP API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        request_timeout_s: Optional[float] = None,
        pool_size: int = CONNECTION_POOL_SIZE,
    ):
        self._base_url = (base_url or get_vllm_base_url()).rstrip("/")
        self._request_timeout_s = (
            request_timeout_s
            if request_timeout_s is not None
            else get_vllm_request_timeout_s()
        )
        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

    @property
    def base_url(self) -> str:
        return self._base_url

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        **extra_body: Any,
    ) -> Dict[str, Any]:
        """Runs `POST /v1/chat/completions` and returns the decoded response.

        Args:
            model: Served model name - either the base model name or a
                registered LoRA adapter name.
            messages: OpenAI-style chat messages. Image inputs must be
                provided as `image_url` content parts with base64 data URIs
                (see `build_image_content_part`).
            temperature: Sampling temperature (0 = greedy, matching the HF
                path's `do_sample=False` default).
            max_tokens: Maximum number of tokens to generate.
            chat_template_kwargs: Extra kwargs forwarded to the chat template
                (e.g. `{"enable_thinking": True}` for Qwen3.5).
            **extra_body: Additional fields merged into the request payload.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = chat_template_kwargs
        payload.update(extra_body)
        response = self._request("POST", "/v1/chat/completions", json=payload)
        return response.json()

    def load_lora_adapter(self, name: str, path: str) -> None:
        """Registers a LoRA adapter stored at `path` under `name`.

        Idempotent: vLLM responds with 400 and an 'already been loaded'
        message when the adapter is registered twice - that case is treated
        as success.
        """
        try:
            self._request(
                "POST",
                "/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": path},
            )
        except VLLMHTTPError as error:
            if error.status_code == 400 and "already been loaded" in (
                error.response_body or ""
            ):
                logger.debug("LoRA adapter %s already loaded in vLLM", name)
                return
            raise

    def unload_lora_adapter(self, name: str) -> None:
        """Unregisters the LoRA adapter served under `name`."""
        self._request("POST", "/v1/unload_lora_adapter", json={"lora_name": name})

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns the list of served models (base + registered adapters)."""
        response = self._request("GET", "/v1/models")
        return response.json().get("data", [])

    def health(self) -> bool:
        """Returns True if the vLLM sidecar reports healthy."""
        try:
            response = self._session.get(
                f"{self._base_url}/health", timeout=self._request_timeout_s
            )
        except _CONNECTIVITY_ERRORS:
            return False
        return response.status_code == 200

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.request(
                method, url, timeout=self._request_timeout_s, **kwargs
            )
        except _CONNECTIVITY_ERRORS as error:
            raise VLLMConnectionError(
                f"Could not reach vLLM sidecar at {url}: {error}"
            ) from error
        if response.status_code >= 400:
            raise VLLMHTTPError(
                message=f"vLLM sidecar returned HTTP {response.status_code} "
                f"for {method} {path}",
                status_code=response.status_code,
                response_body=response.text,
            )
        return response
