"""Resolve the Roboflow workspace id for the configured API key.

Used by the dynamic-block scaffolding to name the Modal sandbox that runs
custom Python. `None` means "unknown", and the caller falls back to the
anonymous workspace - which is also what happened when
`get_roboflow_workspace` raised `WorkspaceLoadError`.

A METHOD, not a bare callable: `steps_initialiser.call_if_callable` invokes any
callable registered in `REGISTERED_INITIALIZERS`.
"""

from typing import Optional, Protocol


class WorkspaceResolver(Protocol):
    def resolve_workspace(self, api_key: Optional[str]) -> Optional[str]: ...


class NullWorkspaceResolver:
    """Standalone default: no platform lookup, so no workspace."""

    def resolve_workspace(self, api_key: Optional[str]) -> Optional[str]:
        return None


NULL_WORKSPACE_RESOLVER = NullWorkspaceResolver()
