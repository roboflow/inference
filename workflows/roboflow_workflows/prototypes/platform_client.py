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
        """Return the workspace that owns an API key.

        Args:
            api_key (str): Roboflow API key.

        Returns:
            str: The workspace ID; never `None`.

        Raises:
            WorkspaceLoadError: If the key is empty or maps to no workspace.
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

    def add_custom_metadata(
        self,
        api_key: str,
        workspace_id: str,
        inference_ids: List[str],
        field_name: str,
        field_value: str,
    ) -> None:
        """Attach a custom metadata field to inference results.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace that owns the results.
            inference_ids (List[str]): IDs of the inference results to update.
            field_name (str): Name of the metadata field.
            field_value (str): Value of the metadata field.

        Raises:
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

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
        """Upload an image to a Roboflow dataset.

        Args:
            api_key (str): Roboflow API key.
            dataset_id (str): Target project.
            local_image_id (str): Name the image is uploaded under.
            image_bytes (bytes): Encoded image.
            batch_name (str): Labeling batch the image joins.
            tags (Optional[List[str]]): Tags for the image.
            inference_id (Optional[str]): Inference result the image belongs to.
            metadata (Optional[Dict[str, Any]]): User metadata for the image.
            annotation_follows (bool): Whether an annotation upload follows.

        Returns:
            dict: The Roboflow API response, including the image `id`.

        Raises:
            RoboflowAPIImageUploadRejectionError: If the API rejects the image.
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

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
        """Upload an annotation for an image already in a dataset.

        Args:
            api_key (str): Roboflow API key.
            dataset_id (str): Project that holds the image.
            local_image_id (str): Name the image was uploaded under.
            roboflow_image_id (str): Image ID returned by the upload.
            annotation_content (str): Serialized annotation.
            annotation_file_type (str): Annotation format, such as `json`.
            is_prediction (bool): Whether the annotation is a model prediction.

        Returns:
            dict: The Roboflow API response.

        Raises:
            RoboflowAPIIAnnotationRejectionError: If the annotation is rejected.
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

    def update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        image_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        add_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update the metadata and tags of an image in the asset library.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace that owns the image.
            image_id (str): Image to update.
            metadata (Optional[Dict[str, Any]]): Metadata fields to set.
            add_tags (Optional[List[str]]): Tags to add.

        Returns:
            Dict[str, Any]: The Roboflow API response.

        Raises:
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

    def batch_update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update the metadata and tags of several images in one request.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace that owns the images.
            updates (List[Dict[str, Any]]): One update per image.

        Returns:
            Dict[str, Any]: The Roboflow API response.

        Raises:
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

    def search_project_images_at_roboflow(
        self,
        api_key: str,
        workspace: str,
        project: str,
        image_base64: str,
        limit: int,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Find the project images most similar to a query image.

        Args:
            api_key (str): Roboflow API key.
            workspace (str): Workspace that owns the project.
            project (str): Project to search.
            image_base64 (str): Query image, base64-encoded.
            limit (int): Maximum number of results.
            fields (Optional[List[str]]): Image fields to return.

        Returns:
            Dict[str, Any]: The Roboflow API response, with `results`.

        Raises:
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

    def send_inference_results_to_model_monitoring(
        self,
        api_key: str,
        workspace_id: str,
        inference_data: dict,
    ) -> None:
        """Report aggregated inference results to model monitoring.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace the results are reported to.
            inference_data (dict): Report payload.

        Raises:
            RoboflowAPIRequestError: If the Roboflow API request fails.
        """

    # What the model-monitoring block reports about the host it runs on.
    def get_device_id(self) -> Optional[str]:
        """Return the ID of the device this host runs on.

        Returns:
            Optional[str]: The device ID, or `None` when it is not set.
        """

    def get_server_version(self) -> str:
        """Return the version of the host server.

        Returns:
            str: The version string.
        """

    def get_system_info(self) -> dict:
        """Return a description of the host system for model monitoring.

        Returns:
            dict: System information.
        """


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
        """Return the workspace that owns an API key.

        Args:
            api_key (str): Roboflow API key.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
        raise _roboflow_api_unavailable()

    def add_custom_metadata(
        self,
        api_key: str,
        workspace_id: str,
        inference_ids: List[str],
        field_name: str,
        field_value: str,
    ) -> None:
        """Attach a custom metadata field to inference results.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace that owns the results.
            inference_ids (List[str]): IDs of the inference results to update.
            field_name (str): Name of the metadata field.
            field_value (str): Value of the metadata field.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
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
        """Upload an image to a Roboflow dataset.

        Args:
            api_key (str): Roboflow API key.
            dataset_id (str): Target project.
            local_image_id (str): Name the image is uploaded under.
            image_bytes (bytes): Encoded image.
            batch_name (str): Labeling batch the image joins.
            tags (Optional[List[str]]): Tags for the image.
            inference_id (Optional[str]): Inference result the image belongs to.
            metadata (Optional[Dict[str, Any]]): User metadata for the image.
            annotation_follows (bool): Whether an annotation upload follows.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
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
        """Upload an annotation for an image already in a dataset.

        Args:
            api_key (str): Roboflow API key.
            dataset_id (str): Project that holds the image.
            local_image_id (str): Name the image was uploaded under.
            roboflow_image_id (str): Image ID returned by the upload.
            annotation_content (str): Serialized annotation.
            annotation_file_type (str): Annotation format, such as `json`.
            is_prediction (bool): Whether the annotation is a model prediction.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
        raise _roboflow_api_unavailable()

    def update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        image_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        add_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Update the metadata and tags of an image in the asset library.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace that owns the image.
            image_id (str): Image to update.
            metadata (Optional[Dict[str, Any]]): Metadata fields to set.
            add_tags (Optional[List[str]]): Tags to add.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
        raise _roboflow_api_unavailable()

    def batch_update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Update the metadata and tags of several images in one request.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace that owns the images.
            updates (List[Dict[str, Any]]): One update per image.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
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
        """Find the project images most similar to a query image.

        Args:
            api_key (str): Roboflow API key.
            workspace (str): Workspace that owns the project.
            project (str): Project to search.
            image_base64 (str): Query image, base64-encoded.
            limit (int): Maximum number of results.
            fields (Optional[List[str]]): Image fields to return.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
        raise _roboflow_api_unavailable()

    def send_inference_results_to_model_monitoring(
        self,
        api_key: str,
        workspace_id: str,
        inference_data: dict,
    ) -> None:
        """Report aggregated inference results to model monitoring.

        Args:
            api_key (str): Roboflow API key.
            workspace_id (str): Workspace the results are reported to.
            inference_data (dict): Report payload.

        Raises:
            WorkflowEnvironmentConfigurationError: Always; no Roboflow API.
        """
        raise _roboflow_api_unavailable()

    def get_device_id(self) -> Optional[str]:
        """Return the ID of the device this host runs on.

        Returns:
            Optional[str]: Always `None`.
        """
        return None

    def get_server_version(self) -> str:
        """Return the version of the host server.

        Returns:
            str: Always `"unknown"`.
        """
        return "unknown"

    def get_system_info(self) -> dict:
        """Return a description of the host system for model monitoring.

        Returns:
            dict: Always empty.
        """
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

_PLATFORM_BLOCKS_OPERATIONS = (
    "build_api_headers",
    "wrap_url",
    "get_roboflow_workspace",
    "add_custom_metadata",
    "register_image_at_roboflow",
    "annotate_image_at_roboflow",
    "update_image_metadata_at_roboflow",
    "batch_update_image_metadata_at_roboflow",
    "search_project_images_at_roboflow",
    "send_inference_results_to_model_monitoring",
    "get_device_id",
    "get_server_version",
    "get_system_info",
)
_PLATFORM_BLOCKS_CLIENT_FACTORY: Optional[Callable[[], RoboflowPlatformClient]] = None


def set_platform_blocks_client_factory(
    factory: Optional[Callable[[], RoboflowPlatformClient]],
) -> None:
    """Register the host client that the Roboflow-platform blocks fall back to.

    The fallback is used only by the Roboflow-platform blocks, and only when
    they get no client, the offline client, or a client that lacks one of the
    operations they call or only inherits its `RoboflowPlatformClient` stub.
    Other blocks keep the client they are given.

    Args:
        factory (Optional[Callable[[], RoboflowPlatformClient]]): Called on
            each resolution to return the host client; `None` removes the
            fallback.
    """
    global _PLATFORM_BLOCKS_CLIENT_FACTORY
    _PLATFORM_BLOCKS_CLIENT_FACTORY = factory


def resolve_platform_blocks_client(
    platform_client: Optional[RoboflowPlatformClient],
) -> RoboflowPlatformClient:
    """Pick the client a Roboflow-platform block uses to reach the Roboflow API.

    Args:
        platform_client (Optional[RoboflowPlatformClient]): The client passed
            to the block or helper, if any.

    Returns:
        RoboflowPlatformClient: `platform_client` when it implements every
        operation the Roboflow-platform blocks call; otherwise the client of
        the factory registered with `set_platform_blocks_client_factory`, and
        without a factory `platform_client` or the offline client.
    """
    if platform_client is not None and _implements_platform_blocks_operations(
        platform_client
    ):
        return platform_client

    factory = _PLATFORM_BLOCKS_CLIENT_FACTORY
    if factory is not None:
        host_client = factory()
        return host_client
    if platform_client is None:
        return OFFLINE_PLATFORM_CLIENT
    return platform_client


def _implements_platform_blocks_operations(platform_client: Any) -> bool:
    if isinstance(platform_client, OfflineRoboflowPlatformClient):
        return False
    return all(
        _implements_operation(platform_client=platform_client, operation=operation)
        for operation in _PLATFORM_BLOCKS_OPERATIONS
    )


def _implements_operation(platform_client: Any, operation: str) -> bool:
    implementation = getattr(platform_client, operation, None)
    if not callable(implementation):
        return False
    protocol_stub = RoboflowPlatformClient.__dict__[operation]
    return getattr(implementation, "__func__", implementation) is not protocol_stub


def ensure_roboflow_api_access(platform_client: Any) -> None:
    """Refuse a direct Roboflow API request that the client cannot make.

    For blocks that send their own requests with the client's URL and header
    helpers, this gives the same refusal as the named operations.

    Args:
        platform_client (Any): The resolved client of a Roboflow-platform block.

    Raises:
        WorkflowEnvironmentConfigurationError: If the client is the offline
            client or does not implement the Roboflow-platform blocks'
            operations.
    """
    if not _implements_platform_blocks_operations(platform_client):
        raise _roboflow_api_unavailable()
