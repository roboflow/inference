"""Roboflow platform adapter of the Workflows V2 ``roboflow_platform`` blocks.

One object implements ``CustomMetadataPort`` and ``VisualSearchPort`` by
calling the server's existing ``inference.core.roboflow_api`` functions, so
``API_BASE_URL``, headers, timeouts, ``OFFLINE_MODE``, retries, error mapping
and the workspace cache behave as they do for the V1 blocks::

    block ─▶ adapter.<operation>(api_key=Secret, ...)
               reveal the key ─▶ roboflow_api.<function>(api_key=plaintext, ...)
               failure ─▶ PlatformRequestError(fixed operation/category message), no chain

Candidate images load through ``image_utils.load_image_from_url``, which keeps
the server's URL guards (``ALLOW_URL_INPUT``, scheme, FQDN and destination
rules). The module attribute is called each time, so patching it reaches every
call.

Nothing registers the adapter. A host passes it explicitly::

    adapter = RoboflowPlatformAdapter()
    session = plan.create_session(
        resources={
            "custom_metadata_port": adapter,
            "visual_search_port": adapter,
            "api_key": api_key,
        }
    )

V2 diagnostics use fixed operation/category text. Audited synchronous host and
urllib3 logs are protected only during these V2 calls; V1 calls keep their
existing diagnostic contents. Arbitrary callback threads are outside this scope.
"""

from typing import Any, Dict, Optional, Sequence

import inference.core.roboflow_api as roboflow_api
import numpy as np
from inference.core.utils import image_utils
from inference.roboflow_workflows_plugin.v2.credential_boundary import (
    HostOperation,
    call_host,
    register_diagnostic_filters,
)
from roboflow_workflows.v2.blocks.roboflow_platform.ports import PlatformRequestError
from roboflow_workflows.v2.data.secrets import Secret, reveal_secrets


class RoboflowPlatformAdapter:
    """Platform operations of the R1 blocks, through ``roboflow_api``.

    Stateless; one instance may serve several resources, steps and sessions.
    """

    def __init__(self) -> None:
        register_diagnostic_filters()

    def get_workspace(self, *, api_key: Secret) -> str:
        """Return the workspace of ``api_key`` (``get_roboflow_workspace``).

        Args:
            api_key: Roboflow API key.

        Returns:
            Workspace URL slug.

        Raises:
            SecretUnavailableError: When ``api_key`` has no value; nothing is sent.
            PlatformRequestError: When the lookup fails.
        """
        workspace = call_host(
            HostOperation.WORKSPACE,
            roboflow_api.get_roboflow_workspace,
            platform=True,
            api_key=reveal_secrets(api_key).value,
        )

        return workspace

    def add_custom_metadata(
        self,
        *,
        api_key: Secret,
        workspace_id: str,
        inference_ids: Sequence[str],
        field_name: str,
        field_value: str,
    ) -> None:
        """Tag inference results (``add_custom_metadata``).

        Under ``OFFLINE_MODE`` this returns without a request, as for V1.

        Args:
            api_key: Roboflow API key.
            workspace_id: Workspace that owns the results.
            inference_ids: Ids of the results to tag.
            field_name: Metadata field name.
            field_value: Metadata field value.

        Raises:
            SecretUnavailableError: When ``api_key`` has no value; nothing is sent.
            PlatformRequestError: When the request fails.
        """
        call_host(
            HostOperation.METADATA,
            roboflow_api.add_custom_metadata,
            platform=True,
            api_key=reveal_secrets(api_key).value,
            workspace_id=workspace_id,
            inference_ids=list(inference_ids),
            field_name=field_name,
            field_value=field_value,
        )

    def search_project_images(
        self,
        *,
        api_key: Secret,
        workspace: str,
        project: str,
        image_base64: str,
        limit: int,
        fields: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Search a project (``search_project_images_at_roboflow``).

        Args:
            api_key: Roboflow API key.
            workspace: Workspace URL slug.
            project: Project URL slug.
            image_base64: Base64 JPEG of the query image.
            limit: Maximum number of results.
            fields: Result fields; ``None`` keeps ``roboflow_api``'s default.

        Returns:
            The decoded JSON response.

        Raises:
            SecretUnavailableError: When ``api_key`` has no value; nothing is sent.
            PlatformRequestError: When the request fails.
        """
        response = call_host(
            HostOperation.SEARCH,
            roboflow_api.search_project_images_at_roboflow,
            platform=True,
            api_key=reveal_secrets(api_key).value,
            workspace=workspace,
            project=project,
            image_base64=image_base64,
            limit=limit,
            fields=None if fields is None else list(fields),
        )

        return response

    def load_candidate_image(self, url: str) -> np.ndarray:
        """Download a candidate image through the server's guarded loader.

        Args:
            url: Image URL from a search result.

        Returns:
            ``uint8`` RGB array of shape ``(height, width, 3)``.

        Raises:
            PlatformRequestError: When the URL is refused or the image cannot
                be loaded.
        """
        bgr = call_host(
            HostOperation.IMAGE,
            image_utils.load_image_from_url,
            platform=True,
            value=url,
        )
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])

        return rgb

    def __repr__(self) -> str:
        return "RoboflowPlatformAdapter()"
