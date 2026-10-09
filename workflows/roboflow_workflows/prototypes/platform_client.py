"""The port through which Workflow blocks reach the Roboflow platform.

Implemented in the server by
`inference.core.interfaces.roboflow_platform_client.ServerRoboflowPlatformClient`,
which forwards to `inference.core.roboflow_api` and
`inference.core.utils.url_utils`. Declared here so that a model block proxying
a VLM call through Roboflow does not import the platform's HTTP client.

`wrap_url` is a member on purpose: it is the secure-gateway proxy wrapper, a
security control that must be injected rather than reimplemented, and every URL
it guards in Workflows is produced next to a `build_api_headers` call.

The Roboflow-platform blocks (dataset upload, custom metadata, model
monitoring, vision events, asset-library attributes, visual search) reach the
Roboflow API through the named operations below. Their names, arguments and
errors are those of the `inference.core.roboflow_api` functions of the same
name: failures raise the `RoboflowAPI*` errors from
`roboflow_workflows.prototypes.platform_errors`.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union

from roboflow_workflows.errors import WorkflowEnvironmentConfigurationError

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

    def get_roboflow_workspace(self, api_key: str) -> str:
        """Workspace of `api_key`; raises instead of returning `None`."""
        ...

    def add_custom_metadata(
        self,
        api_key: str,
        workspace_id: str,
        inference_ids: List[str],
        field_name: str,
        field_value: str,
    ) -> None: ...

    def register_image_at_roboflow(
        self,
        api_key: str,
        dataset_id: str,
        local_image_id: str,
        image_bytes: bytes,
        batch_name: str,
        tags: Optional[List[str]] = None,
        inference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        annotation_follows: bool = False,
    ) -> dict: ...

    def annotate_image_at_roboflow(
        self,
        api_key: str,
        dataset_id: str,
        local_image_id: str,
        roboflow_image_id: str,
        annotation_content: str,
        annotation_file_type: str,
        is_prediction: bool = True,
    ) -> dict: ...

    def update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        image_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        add_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...

    def batch_update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]: ...

    def search_project_images_at_roboflow(
        self,
        api_key: str,
        workspace: str,
        project: str,
        image_base64: str,
        limit: int,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]: ...

    def send_inference_results_to_model_monitoring(
        self,
        api_key: str,
        workspace_id: str,
        inference_data: dict,
    ) -> None: ...

    # What the model-monitoring block reports about the host it runs on.
    def get_device_id(self) -> Optional[str]: ...

    def get_server_version(self) -> str: ...

    def get_system_info(self) -> dict: ...


class OfflineRoboflowPlatformClient:
    """Standalone default: no platform, and it says so.

    Registered as `REGISTERED_INITIALIZERS["platform_client"]`; the server
    overrides it at every composition root. Header builders return empty/None
    rather than raising, so a block can still assemble a request for a
    non-Roboflow endpoint; `post` and the named Roboflow API operations - which
    can only target Roboflow - refuse. `wrap_url` is the identity, which is
    what the real `wrap_url` does when `SECURE_GATEWAY` is unset. There is no
    host to describe, so the host-identity getters return empty values.
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

    def get_roboflow_workspace(self, api_key: str) -> str:
        raise _roboflow_api_unavailable()

    def add_custom_metadata(
        self,
        api_key: str,
        workspace_id: str,
        inference_ids: List[str],
        field_name: str,
        field_value: str,
    ) -> None:
        raise _roboflow_api_unavailable()

    def register_image_at_roboflow(
        self,
        api_key: str,
        dataset_id: str,
        local_image_id: str,
        image_bytes: bytes,
        batch_name: str,
        tags: Optional[List[str]] = None,
        inference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        annotation_follows: bool = False,
    ) -> dict:
        raise _roboflow_api_unavailable()

    def annotate_image_at_roboflow(
        self,
        api_key: str,
        dataset_id: str,
        local_image_id: str,
        roboflow_image_id: str,
        annotation_content: str,
        annotation_file_type: str,
        is_prediction: bool = True,
    ) -> dict:
        raise _roboflow_api_unavailable()

    def update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        image_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        add_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raise _roboflow_api_unavailable()

    def batch_update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        raise _roboflow_api_unavailable()

    def search_project_images_at_roboflow(
        self,
        api_key: str,
        workspace: str,
        project: str,
        image_base64: str,
        limit: int,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raise _roboflow_api_unavailable()

    def send_inference_results_to_model_monitoring(
        self,
        api_key: str,
        workspace_id: str,
        inference_data: dict,
    ) -> None:
        raise _roboflow_api_unavailable()

    def get_device_id(self) -> Optional[str]:
        return None

    def get_server_version(self) -> str:
        return "unknown"

    def get_system_info(self) -> dict:
        return {}


def _roboflow_api_unavailable() -> WorkflowEnvironmentConfigurationError:
    return WorkflowEnvironmentConfigurationError(
        public_message=(
            "This step calls the Roboflow API, which is not available in this "
            "installation of `workflows`. Provide a "
            "`workflows_core.platform_client` init parameter."
        ),
        context="workflow_execution | step_execution | roboflow_platform_access",
    )


# One shared instance. Stateless; it is what
# `REGISTERED_INITIALIZERS["platform_client"]` binds and the `__init__` default
# of every block that takes the port, so a block constructed directly (as 69
# existing unit-test call sites do: 47 for the Task 9.4 classes, 22 for the
# Task 9.5 ones) still works while the engine always passes
# the resolved value explicitly.
OFFLINE_PLATFORM_CLIENT = OfflineRoboflowPlatformClient()
