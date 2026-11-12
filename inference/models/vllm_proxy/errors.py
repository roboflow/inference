"""Typed exceptions for the vLLM proxy backend.

`NotServableOnVLLMError` / `AdapterNotServableError` subclass
`ModelDeploymentNotSupportedError`, which the existing HTTP error handlers
(`inference/core/interfaces/http/error_handlers.py`) surface as a 501 response
with `str(error)` as the message - the same convention used when a model type
is not supported by a deployment.
"""

from typing import Optional

from inference.core.exceptions import ModelDeploymentNotSupportedError

VLLM_UNAVAILABLE_STATUS_CODE = 503
VLLM_TIMEOUT_STATUS_CODE = 504


class NotServableOnVLLMError(ModelDeploymentNotSupportedError):
    """The requested model cannot be served by the configured vLLM sidecar.

    Raised e.g. when the model's base variant does not match the base model
    loaded in vLLM.
    """


class AdapterNotServableError(NotServableOnVLLMError):
    """The model's LoRA adapter cannot be transformed into a form vLLM accepts.

    Raised e.g. for adapters with `modules_to_save`, excessive rank, DoRA
    adapters under the `reject` policy, or adapters that meaningfully trained
    the vision tower.
    """


class VLLMProxyError(Exception):
    """Base class for errors talking to the vLLM sidecar."""


class VLLMConnectionError(VLLMProxyError):
    """Could not connect to (or timed out talking to) the vLLM sidecar.

    `status_code` is the HTTP status the failure maps to:
    `VLLM_TIMEOUT_STATUS_CODE` when the sidecar did not answer within the
    request timeout, `VLLM_UNAVAILABLE_STATUS_CODE` when the connection could
    not be established or was dropped. Usage tracking records it as
    `error_status_code` and the HTTP error handlers answer with it.
    """

    def __init__(self, message: str, status_code: int = VLLM_UNAVAILABLE_STATUS_CODE):
        super().__init__(message)
        self.status_code = status_code


class VLLMHTTPError(VLLMProxyError):
    """The vLLM sidecar returned an HTTP error response."""

    def __init__(self, message: str, status_code: int, response_body: Optional[str]):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
