"""Server implementations of the Workflows Roboflow-platform ports.

Everything forwards to `inference.core.roboflow_api` and
`inference.core.utils.url_utils`; no security control is reimplemented. Calls
go through the module objects rather than attributes captured at construction,
so a test that monkeypatches SECURE_GATEWAY, the transport or the header-policy
flags observes what production does.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import inference.core.roboflow_api as roboflow_api
import inference.core.utils.url_utils as url_utils
from inference.core.cache import cache as server_cache
from inference.core.exceptions import WorkspaceLoadError
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
