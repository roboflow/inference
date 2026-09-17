"""The port through which Workflow blocks reach the Roboflow platform.

Implemented in the server by
`inference.core.interfaces.roboflow_platform_client.ServerRoboflowPlatformClient`,
which forwards to `inference.core.roboflow_api` and
`inference.core.utils.url_utils`. Declared here so that a model block proxying
a VLM call through Roboflow does not import the platform's HTTP client.

`wrap_url` is a member on purpose: it is the secure-gateway proxy wrapper, a
security control that must be injected rather than reimplemented, and every URL
it guards in Workflows is produced next to a `build_api_headers` call.
"""

from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError

HttpErrorHandlers = Dict[int, Callable[[Exception], None]]


class RoboflowPlatformClient(Protocol):
    """Deliberately NOT `runtime_checkable`: nothing isinstance-checks it."""

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict: ...

    def build_api_headers(
        self, explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> Dict[str, Union[str, List[str]]]: ...

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Dict[str, str]]: ...

    def wrap_url(self, url: str) -> str: ...


class OfflineRoboflowPlatformClient:
    """Standalone default: no platform, and it says so.

    Registered as `REGISTERED_INITIALIZERS["platform_client"]`; the server
    overrides it at every composition root. Header builders return empty/None
    rather than raising, so a block can still assemble a request for a
    non-Roboflow endpoint; only `post` - which can only target Roboflow -
    refuses. `wrap_url` is the identity, which is what the real `wrap_url` does
    when `SECURE_GATEWAY` is unset.
    """

    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict:
        raise WorkflowEnvironmentConfigurationError(
            public_message=(
                "This step routes its request through the Roboflow API, which is "
                "not available in this installation of `workflows`. Provide a "
                "`workflows_core.platform_client` init parameter, or configure the "
                "step to call its provider directly with your own API key."
            ),
            context="workflow_execution | step_execution | roboflow_platform_access",
        )

    def build_api_headers(
        self, explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> Dict[str, Union[str, List[str]]]:
        return dict(explicit_headers or {})

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        return None

    def wrap_url(self, url: str) -> str:
        return url


# One shared instance. Stateless; it is what
# `REGISTERED_INITIALIZERS["platform_client"]` binds and the `__init__` default
# of every block that takes the port, so a block constructed directly (as 69
# existing unit-test call sites do: 47 for the Task 9.4 classes, 22 for the
# Task 9.5 ones) still works while the engine always passes
# the resolved value explicitly.
OFFLINE_PLATFORM_CLIENT = OfflineRoboflowPlatformClient()
