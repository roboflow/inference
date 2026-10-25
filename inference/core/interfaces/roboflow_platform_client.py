"""Server implementations of the Workflows Roboflow-platform ports.

Everything forwards to `inference.core.roboflow_api` and
`inference.core.utils.url_utils`; no security control is reimplemented. Calls
go through the module objects rather than attributes captured at construction,
so a test that monkeypatches SECURE_GATEWAY, the transport, the header-policy
flags or a `roboflow_api` function observes what production does.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import inference.core.env as env
import inference.core.managers.metrics as metrics
import inference.core.roboflow_api as roboflow_api
import inference.core.utils.url_utils as url_utils
from inference.core.cache import cache as server_cache
from inference.core.exceptions import WorkspaceLoadError
from inference.core.version import __version__
from inference.core.workflows.errors import WorkflowDefinitionError
from inference.core.workflows.prototypes.platform_client import HttpErrorHandlers


class ServerRoboflowPlatformClient:
    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict:
        return roboflow_api.post_to_roboflow_api(
            endpoint=endpoint,
            api_key=api_key,
            payload=payload,
            params=params,
            http_errors_handlers=http_errors_handlers,
        )

    def build_api_headers(
        self, explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> Dict[str, Union[str, List[str]]]:
        return roboflow_api.build_roboflow_api_headers(
            explicit_headers=explicit_headers
        )

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        return roboflow_api.get_extra_weights_provider_headers(
            countinference=countinference, service_secret=service_secret
        )

    def wrap_url(self, url: str) -> str:
        return url_utils.wrap_url(url)

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
        return roboflow_api.get_roboflow_workspace(api_key=api_key)

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
        return roboflow_api.add_custom_metadata(
            api_key=api_key,
            workspace_id=workspace_id,
            inference_ids=inference_ids,
            field_name=field_name,
            field_value=field_value,
        )

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
        return roboflow_api.register_image_at_roboflow(
            api_key=api_key,
            dataset_id=dataset_id,
            local_image_id=local_image_id,
            image_bytes=image_bytes,
            batch_name=batch_name,
            tags=tags,
            inference_id=inference_id,
            metadata=metadata,
            annotation_follows=annotation_follows,
        )

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
        return roboflow_api.annotate_image_at_roboflow(
            api_key=api_key,
            dataset_id=dataset_id,
            local_image_id=local_image_id,
            roboflow_image_id=roboflow_image_id,
            annotation_content=annotation_content,
            annotation_file_type=annotation_file_type,
            is_prediction=is_prediction,
        )

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
        return roboflow_api.update_image_metadata_at_roboflow(
            api_key=api_key,
            workspace_id=workspace_id,
            image_id=image_id,
            metadata=metadata,
            add_tags=add_tags,
        )

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
        return roboflow_api.batch_update_image_metadata_at_roboflow(
            api_key=api_key,
            workspace_id=workspace_id,
            updates=updates,
        )

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
        return roboflow_api.search_project_images_at_roboflow(
            api_key=api_key,
            workspace=workspace,
            project=project,
            image_base64=image_base64,
            limit=limit,
            fields=fields,
        )

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
        return roboflow_api.send_inference_results_to_model_monitoring(
            api_key, workspace_id, inference_data
        )

    def get_device_id(self) -> Optional[str]:
        """Return the ID of the device this host runs on.

        Returns:
            Optional[str]: The device ID, or `None` when it is not set.
        """
        return env.DEVICE_ID

    def get_server_version(self) -> str:
        """Return the version of the host server.

        Returns:
            str: The version string.
        """
        return __version__

    def get_system_info(self) -> dict:
        """Return a description of the host system for model monitoring.

        Returns:
            dict: System information.
        """
        return metrics.get_system_info()


class ServerWorkspaceResolver:
    """Swallows `WorkspaceLoadError` into `None`, exactly what both former call
    sites in `block_scaffolding.py` did with their own `try/except`."""

    def resolve_workspace(self, api_key: Optional[str]) -> Optional[str]:
        try:
            return roboflow_api.get_roboflow_workspace(api_key)
        except WorkspaceLoadError:
            return None


def default_inner_workflow_spec_resolver(
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
    init_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Fetch an inner workflow's definition from the Roboflow API.

    Moved out of `execution_engine/v1/inner_workflow/reference_resolution.py`
    so that module stops importing `roboflow_api`. Installed at every server
    composition root, like `resolve_step_error_handler()`.
    """
    api_key = init_parameters.get("workflows_core.api_key")
    if workspace_id != "local" and not api_key:
        raise WorkflowDefinitionError(
            public_message=(
                "Resolving an `inner_workflow` step by workflow id requires a Roboflow API key. "
                "Set `workflows_core.api_key` in workflow init_parameters, inject "
                "`workflows_core.inner_workflow_spec_resolver`, or use "
                '`workflow_workspace_id` `"local"` with a matching on-disk workflow '
                "definition."
            ),
            context="workflow_compilation | inner_workflow_spec_resolution",
        )
    return roboflow_api.get_workflow_specification(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )


# Module-level singletons: one client and one resolver per process.
SERVER_PLATFORM_CLIENT = ServerRoboflowPlatformClient()
SERVER_WORKSPACE_RESOLVER = ServerWorkspaceResolver()


def workflows_platform_bindings() -> Dict[str, Any]:
    """The `workflows_core.*` init parameters every composition root installs."""
    return {
        "workflows_core.cache": server_cache,
        "workflows_core.platform_client": SERVER_PLATFORM_CLIENT,
        "workflows_core.workspace_resolver": SERVER_WORKSPACE_RESOLVER,
        "workflows_core.inner_workflow_spec_resolver": default_inner_workflow_spec_resolver,
    }


def install_workflows_platform_bindings(
    init_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    """Install the server defaults WITHOUT replacing explicit values.

    `setdefault`, not `update`: `InferencePipeline.init_with_workflow` forwards
    a caller's `workflow_init_parameters` dict, and
    `workflows_core.inner_workflow_spec_resolver` overriding the default is an
    existing contract (`reference_resolution.get_inner_workflow_spec_resolver`).

    ADDITIVE by construction: it only fills in keys that are absent, so a phase
    that adds its own `workflows_core.*` binding at the same roots (Phase 6's
    observer, Phase 11's models adapter) composes with this instead of
    replacing it.
    """
    for key, value in workflows_platform_bindings().items():
        init_parameters.setdefault(key, value)
    return init_parameters
