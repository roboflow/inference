"""Saved-workflow resolver for the Workflows V2 compiler, backed by the Roboflow API.

The standalone V2 compiler fetches a saved child workflow through the caller's
``reference_resolver``. This host adapter answers it with the server's existing
``roboflow_api.get_workflow_specification``, so retrieval, caching, ``local``
workspace files behave as they do for V1. V2 failures use fixed useful
operation/category text and fresh errors without upstream exception chains. The compiler
still owns cycle, depth and count limits and validates the fetched definition
as V2; nothing here converts V1 definitions.

Nothing registers this resolver. A host wires it explicitly::

    plan = compile_workflow(
        definition,
        catalogue=catalogue,
        reference_resolver=RoboflowWorkflowResolver(api_key=api_key),
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import inference.core.roboflow_api as roboflow_api
from inference.core.cache.base import BaseCache
from inference.roboflow_workflows_plugin.v2.credential_boundary import (
    HostOperation,
    call_host,
    register_diagnostic_filters,
)
from roboflow_workflows.v2.engine.compilation import WorkflowReference

LOCAL_WORKSPACE = "local"
"""Workspace whose workflows the host reads from its model cache directory."""

HOST_ADDED_KEY = "id"
"""Key ``get_workflow_specification`` adds to every specification it returns."""


class RoboflowApiKeyRequiredError(ValueError):
    """A saved workflow outside the ``local`` workspace needs an API key."""


@dataclass(frozen=True)
class RoboflowWorkflowResolver:
    """Fetch saved V2 workflow definitions for ``compile_workflow``.

    Calls go through the ``roboflow_api`` module object, so a test that
    redirects ``API_BASE_URL`` or the transport observes what production does.

    Args:
        api_key: Roboflow API key. Required unless the reference names the
            ``local`` workspace, matching the V1 server resolver. Hidden from
            ``repr``.
        use_cache: Read and write the host's workflow-definition cache, as
            the V1 server resolver does by default.
        ephemeral_cache: Cache backend; ``None`` uses the process cache.
    """

    api_key: Optional[str] = field(default=None, repr=False)
    use_cache: bool = True
    ephemeral_cache: Optional[BaseCache] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        register_diagnostic_filters()

    def __call__(self, reference: WorkflowReference) -> Dict[str, Any]:
        """Return the definition of one saved workflow.

        Args:
            reference: Saved workflow identity from a nested step; a ``None``
                ``version_id`` asks the platform for its default version.

        Returns:
            The saved specification without the key the host adds to it.

        Raises:
            RoboflowApiKeyRequiredError: No API key for a non-local workspace.
            MalformedWorkflowResponseError: The response holds no specification.
            RoboflowAPIRequestError: The request failed; host subclasses name
                authorization, not-found and connection failures.
            FileNotFoundError: A ``local`` workflow file is missing.
        """
        if reference.workspace_id != LOCAL_WORKSPACE and not self.api_key:
            raise RoboflowApiKeyRequiredError(
                f"saved workflow {reference.describe()} needs a Roboflow API key; "
                "pass api_key to RoboflowWorkflowResolver or use the "
                f"{LOCAL_WORKSPACE!r} workspace"
            )

        specification = call_host(
            HostOperation.WORKFLOW,
            roboflow_api.get_workflow_specification,
            api_key=self.api_key,
            workspace_id=reference.workspace_id,
            workflow_id=reference.workflow_id,
            use_cache=self.use_cache,
            ephemeral_cache=self.ephemeral_cache,
            workflow_version_id=reference.version_id,
        )
        # The host overwrites "id" with the platform's record id; V2 rejects
        # unknown top-level keys, so the saved definition is everything else.
        definition = {
            key: value for key, value in specification.items() if key != HOST_ADDED_KEY
        }

        return definition
