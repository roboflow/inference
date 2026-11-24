"""Narrow Roboflow platform ports of the ``roboflow_platform`` blocks.

Each block family names the few platform operations it needs in a small
``Protocol`` and takes one constructor resource of that type, offline by
default. The host fills the resource with an adapter; only the adapter sees
the plaintext API key::

    standalone package                     host (e.g. the inference plugin)
    ──────────────────                     ────────────────────────────────
    block.run
      api_key = api_key_for_dispatch(...)  str ─▶ Secret("api_key"); None or an
      │                                    unavailable Secret fails here
      └─▶ port.<operation>(api_key=Secret, ...)
            ├─ default OfflinePlatformPort ─▶ PlatformUnavailableError
            └─ host adapter: reveal ─▶ HTTP ─▶ result
                             failure ─▶ PlatformRequestError, scrubbed, unchained

    resource               Protocol             operations
    custom_metadata_port   CustomMetadataPort   get_workspace, add_custom_metadata
    visual_search_port     VisualSearchPort     get_workspace, search_project_images,
                                                load_candidate_image

Contract every implementation keeps:

* Credentials arrive as ``Secret``. Reveal them only to send a request.
* A failed request raises ``PlatformRequestError``. Its message never holds a
  revealed value, and its ``__cause__`` and ``__context__`` are ``None``: the
  original exception, whose text may hold the key, is not reachable from it.
* Inputs are read, never modified; results are new objects.

A further family copies the pattern in its own block module: a ``Protocol``
that extends ``WorkspacePort`` when it needs the workspace, a resource name
``<family>_port``, an ``OfflinePlatformPort``-style default built with
``platform_unavailable`` and a host adapter in its own plugin module.
"""

from typing import Any, Dict, Optional, Protocol, Sequence, Union

import numpy as np
from roboflow_workflows.v2.blocks.platform import PlatformUnavailableError
from roboflow_workflows.v2.data.secrets import (
    Secret,
    SecretUnavailableError,
    to_secret,
)

CUSTOM_METADATA_PORT_RESOURCE = "custom_metadata_port"
"""Constructor resource name of ``CustomMetadataPort`` consumers."""

VISUAL_SEARCH_PORT_RESOURCE = "visual_search_port"
"""Constructor resource name of ``VisualSearchPort`` consumers."""

API_KEY_DOCS_URL = (
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key"
)


class PlatformRequestError(RuntimeError):
    """A platform request failed; the message is free of credentials.

    Args:
        message: Failure description with every revealed value scrubbed.
        error_type: Qualified name of the original exception class, e.g.
            ``"inference.core.exceptions.RoboflowAPIConnectionError"``.
    """

    def __init__(self, message: str, *, error_type: str) -> None:
        super().__init__(message)
        self.error_type = error_type


class WorkspacePort(Protocol):
    """Resolve the workspace an API key belongs to."""

    def get_workspace(self, *, api_key: Secret) -> str:
        """Return the workspace URL slug of ``api_key``.

        Args:
            api_key: Roboflow API key.

        Returns:
            Workspace URL slug, e.g. ``"my-workspace"``.

        Raises:
            PlatformRequestError: When the platform rejects or cannot answer.
        """


class CustomMetadataPort(WorkspacePort, Protocol):
    """Model Monitoring custom metadata, as V1 ``add_custom_metadata``."""

    def add_custom_metadata(
        self,
        *,
        api_key: Secret,
        workspace_id: str,
        inference_ids: Sequence[str],
        field_name: str,
        field_value: str,
    ) -> None:
        """Attach one ``field_name: field_value`` pair to inference results.

        Args:
            api_key: Roboflow API key.
            workspace_id: Workspace that owns the inference results.
            inference_ids: Ids of the results to tag.
            field_name: Metadata field name.
            field_value: Metadata field value.

        Raises:
            PlatformRequestError: When the platform rejects or cannot answer.
        """


class VisualSearchPort(WorkspacePort, Protocol):
    """Roboflow project image search and candidate image loading."""

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
        """Search a project for images similar to a query image.

        Args:
            api_key: Roboflow API key.
            workspace: Workspace URL slug that owns ``project``.
            project: Project URL slug.
            image_base64: Base64 JPEG of the query image.
            limit: Maximum number of results.
            fields: Result fields to request; ``None`` asks for the platform
                client's default fields.

        Returns:
            The search response; candidates are under ``"results"``.

        Raises:
            PlatformRequestError: When the platform rejects or cannot answer.
        """

    def load_candidate_image(self, url: str) -> np.ndarray:
        """Download a search result image.

        Args:
            url: ``image_url`` of a candidate.

        Returns:
            ``uint8`` RGB array of shape ``(height, width, 3)``.

        Raises:
            PlatformRequestError: When the image cannot be loaded.
        """


def platform_unavailable(*, resource: str, action: str) -> PlatformUnavailableError:
    """Build the error an offline port raises.

    Args:
        resource: Constructor resource name the host did not provide.
        action: What the step wanted to do, e.g. ``"project image search"``.

    Returns:
        An error that names the missing resource and how to provide it.
    """
    error = PlatformUnavailableError(
        f"this step needs the Roboflow API ({action}), but the host provided no "
        f"'{resource}' resource. Pass one when creating the session, e.g. "
        f"resources={{'{resource}': RoboflowPlatformAdapter()}} from "
        "inference.roboflow_workflows_plugin.v2.platform_adapter."
    )

    return error


class OfflinePlatformPort:
    """Default of every R1 port resource: no platform, and it says so.

    Every operation raises ``PlatformUnavailableError`` before touching its
    arguments, so no credential is revealed and nothing is sent.

    Args:
        resource: Resource name reported in the error.
    """

    def __init__(self, resource: str) -> None:
        self._resource = resource

    def get_workspace(self, *, api_key: Secret) -> str:
        """Refuse.

        Args:
            api_key: Ignored.

        Raises:
            PlatformUnavailableError: Always.
        """
        raise platform_unavailable(resource=self._resource, action="workspace lookup")

    def add_custom_metadata(
        self,
        *,
        api_key: Secret,
        workspace_id: str,
        inference_ids: Sequence[str],
        field_name: str,
        field_value: str,
    ) -> None:
        """Refuse.

        Args:
            api_key: Ignored.
            workspace_id: Ignored.
            inference_ids: Ignored.
            field_name: Ignored.
            field_value: Ignored.

        Raises:
            PlatformUnavailableError: Always.
        """
        raise platform_unavailable(resource=self._resource, action="custom metadata")

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
        """Refuse.

        Args:
            api_key: Ignored.
            workspace: Ignored.
            project: Ignored.
            image_base64: Ignored.
            limit: Ignored.
            fields: Ignored.

        Raises:
            PlatformUnavailableError: Always.
        """
        raise platform_unavailable(
            resource=self._resource, action="project image search"
        )

    def load_candidate_image(self, url: str) -> np.ndarray:
        """Refuse.

        Args:
            url: Ignored.

        Raises:
            PlatformUnavailableError: Always.
        """
        raise platform_unavailable(
            resource=self._resource, action="candidate image loading"
        )

    def __repr__(self) -> str:
        return f"OfflinePlatformPort({self._resource!r})"


OFFLINE_CUSTOM_METADATA_PORT = OfflinePlatformPort(CUSTOM_METADATA_PORT_RESOURCE)
"""Default ``custom_metadata_port`` resource."""

OFFLINE_VISUAL_SEARCH_PORT = OfflinePlatformPort(VISUAL_SEARCH_PORT_RESOURCE)
"""Default ``visual_search_port`` resource."""


def api_key_for_dispatch(
    api_key: Optional[Union[str, Secret]], *, block_name: str
) -> Secret:
    """Check the ``api_key`` resource before any platform request.

    Args:
        api_key: The block's ``api_key`` resource: a ``str``, a ``Secret`` or
            ``None``.
        block_name: Block name used in the error, e.g. ``"Roboflow Visual
            Search"``.

    Returns:
        The key as a ``Secret``; a ``str`` is wrapped under the name
        ``"api_key"``.

    Raises:
        ValueError: When there is no key, with V1's message.
        SecretUnavailableError: When the key is a secret without a value, e.g.
            one restored from a recording. Nothing is sent.
        TypeError: When the key is neither a ``str`` nor a ``Secret``.
    """
    if api_key is None:
        raise ValueError(
            f"{block_name} block cannot run without a Roboflow API key. Visit "
            f"{API_KEY_DOCS_URL} to learn how to retrieve one."
        )

    secret = to_secret(api_key, name="api_key")
    if not secret.available:
        raise SecretUnavailableError(
            f"secret {secret.name!r} has no value here: recordings and replays "
            "keep only its name. Supply the API key again from the host."
        )

    return secret
