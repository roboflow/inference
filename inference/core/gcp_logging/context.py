"""
GCP Serverless Logging Request Context.

This module provides context tracking for GCP logging using ContextVars.
It tracks request IDs, workflow context, and invocation source.
"""

import hashlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass
class GCPRequestContext:
    """Context for a single GCP serverless request."""

    request_id: str
    api_key_hash: Optional[str] = None
    invocation_source: str = "direct"  # "direct" or "workflow"
    workflow_instance_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_name: Optional[str] = None
    last_model_cache_hit: Optional[bool] = None  # Track cache hit for current model load


# Context variable for request tracking (thread-safe)
_gcp_request_context: ContextVar[Optional[GCPRequestContext]] = ContextVar(
    "gcp_request_context", default=None
)


def hash_api_key(api_key: Optional[str]) -> Optional[str]:
    """
    Hash API key for logging (privacy protection).

    Returns first 16 characters of SHA256 hash.
    """
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def set_gcp_context(context: GCPRequestContext) -> None:
    """Set the GCP request context for the current execution."""
    _gcp_request_context.set(context)


def get_gcp_context() -> Optional[GCPRequestContext]:
    """Get the current GCP request context."""
    return _gcp_request_context.get()


def clear_gcp_context() -> None:
    """Clear the GCP request context."""
    _gcp_request_context.set(None)


def update_gcp_context(
    workflow_instance_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    step_name: Optional[str] = None,
    invocation_source: Optional[str] = None,
    last_model_cache_hit: Optional[bool] = None,
) -> None:
    """
    Update the current GCP context with workflow information.

    This is called when entering a workflow step to add step-specific context.
    """
    current = get_gcp_context()
    if current is None:
        return

    if workflow_instance_id is not None:
        current.workflow_instance_id = workflow_instance_id
    if workflow_id is not None:
        current.workflow_id = workflow_id
    if step_name is not None:
        current.step_name = step_name
    if invocation_source is not None:
        current.invocation_source = invocation_source
    if last_model_cache_hit is not None:
        current.last_model_cache_hit = last_model_cache_hit
