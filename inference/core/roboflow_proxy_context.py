"""Request-scoped Auto Label context for Roboflow-managed provider proxies.

Folder-scoped Roboflow API keys require a jobId and/or projectId on
`/apiproxy/openai*` and `/apiproxy/gemini` so the platform can authorize the
key against the project's billing folder without swapping to the workspace
key. Workflow blocks (OpenAI / Gemini) run deep inside the execution engine
and cannot take new required init parameters without breaking every existing
workflow, so the Auto Label worker and HTTP preview path stash the ids here
and the proxied request helpers read them when building the proxy payload.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

_autolabel_job_id: ContextVar[Optional[str]] = ContextVar(
    "roboflow_autolabel_job_id", default=None
)
_autolabel_project_id: ContextVar[Optional[str]] = ContextVar(
    "roboflow_autolabel_project_id", default=None
)


def get_autolabel_job_id() -> Optional[str]:
    return _autolabel_job_id.get()


def get_autolabel_project_id() -> Optional[str]:
    return _autolabel_project_id.get()


def proxy_context_fields() -> dict:
    """Fields to merge into an `/apiproxy/*` JSON body when set."""
    payload = {}
    job_id = get_autolabel_job_id()
    project_id = get_autolabel_project_id()
    if job_id:
        payload["jobId"] = job_id
    if project_id:
        payload["projectId"] = project_id
    return payload


@contextmanager
def autolabel_proxy_context(
    *,
    job_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Iterator[None]:
    job_token = _autolabel_job_id.set(job_id)
    project_token = _autolabel_project_id.set(project_id)
    try:
        yield
    finally:
        _autolabel_job_id.reset(job_token)
        _autolabel_project_id.reset(project_token)
