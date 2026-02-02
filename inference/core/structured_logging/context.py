"""
Structured Logging Request Context.

This module provides context tracking for structured logging using ContextVars.
It tracks request IDs, workflow context, and invocation source.
"""

import hashlib
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestContext:
    """Context for a single request."""

    request_id: str
    api_key_hash: Optional[str] = None
    invocation_source: str = "direct"  # "direct" or "workflow"
    workflow_instance_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_name: Optional[str] = None
    last_model_cache_hit: Optional[bool] = None  # Track cache hit for current model load


# Context variable for request tracking (thread-safe)
_request_context: ContextVar[Optional[RequestContext]] = ContextVar(
    "request_context", default=None
)


def hash_api_key(api_key: Optional[str]) -> Optional[str]:
    """
    Hash API key for logging (privacy protection).

    Returns first 16 characters of SHA256 hash.
    """
    if not api_key:
        return None
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def set_request_context(context: RequestContext) -> None:
    """Set the request context for the current execution."""
    _request_context.set(context)


def get_request_context() -> Optional[RequestContext]:
    """Get the current request context."""
    return _request_context.get()


def clear_request_context() -> None:
    """Clear the request context."""
    _request_context.set(None)


def update_request_context(
    workflow_instance_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    step_name: Optional[str] = None,
    invocation_source: Optional[str] = None,
    last_model_cache_hit: Optional[bool] = None,
) -> None:
    """
    Update the current context with workflow information.

    This is called when entering a workflow step to add step-specific context.
    """
    current = get_request_context()
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
